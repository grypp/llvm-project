# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Staging: Python values versus MLIR ops.

A traced function handles two kinds of values. A Python value is known while
tracing and folds into the program; an MLIR op (the result of one) is only
known when the compiled program runs. :func:`is_mlir_op` tells them apart:
inside a trace the active DSL answers (``BaseDSL.is_mlir_op``, which a sub-DSL
with its own value model overrides), the default judging a leaf by whether it holds an SSA value. At the host
boundary :func:`is_argument_meta` applies the same split to the arguments of
a ``@jit`` call: the annotation decides, a DSL type is an MLIR op and anything
else is a Python value the trace specialises on.

Every ``BaseDSL`` also owns one :class:`Executor`: the DSL's meaning of
Python's own keywords (``for``, ``if``, ``while``, ``x if c else y``, the
comparison operators, the built-in functions) when their operands are MLIR
ops. Python cannot run ``if n > 0:`` on a value that does not exist yet, so the
AST preprocessor rewrites each such statement into a call on the executor, and
the DSL decides what the keyword means: build an ``scf.if``, unroll a loop,
emit an ``arith.cmpi``. This is language virtualization of the keywords; the
rest of Python keeps its ordinary meaning. The helpers the rewritten code
calls (``plugins/ast_preprocessor/helpers.py``) reach the active DSL's
executor through the module-level ``executor`` proxy.
"""

import dataclasses
import inspect
from inspect import Parameter
from typing import Annotated, get_args, get_origin
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Optional

from ... import ir
from .common import DSLRuntimeError, DSLUserCodeError, get_current_dsl
from .diagnostics import DiagId
from ..util.logger import log
from ..types.typing import NumericMeta, TypedPointer
from ..util.tree_utils import (
    contains_leaf,
    is_frozen_dataclass,
    is_leaf,
    is_staged_leaf,
)

__all__ = [
    "Executor",
    "executor",
    "is_argument_meta",
    "is_mlir_op",
    "is_reserved_python_func_arg",
]

_EXECUTORS_NOT_INSTALLED = (
    "executors not installed: the DSL's AST preprocessor plugin fills them when "
    "the DSL is constructed (`ASTPreprocessorPlugin.executors`)"
)


class Executor:
    """What Python's keywords mean inside a traced function.

    A traced function is ordinary Python until a keyword meets an MLIR op:
    ``for i in range(n)`` with an ``n`` that is only known when the compiled
    program runs, ``if x > 0`` on such an ``x``, ``while``, ``a if c else b``,
    a comparison, a call of ``max``. Python itself cannot execute these, so
    the AST preprocessor rewrites each of them into a call of one method
    here, with the body of the statement as a Python function::

        for i in range(n):          ->  executor.for_execute(body, 0, n, 1, ...)
            acc = acc + i
        if x > 0:                   ->  executor.if_execute(x > 0, then, else_, ...)
            y = 1
        else:
            y = 2

    This object holds one slot per keyword; the DSL's AST preprocessor plugin
    fills them through :meth:`set_functions` with the DSL's own meaning of
    the keyword (the scf plugin builds ``scf.for``, ``scf.if``, ``scf.while``
    and ``arith.cmpi``; another dialect builds its own). A keyword whose
    operands are plain Python values never arrives here: the preprocessor
    leaves it to Python.

    Every ``BaseDSL`` instance owns one executor. Until its functions are set
    a rewritten statement raises ``DSLRuntimeError``.
    """

    def __init__(self) -> None:
        self._loop_execute_range_dynamic: Callable[..., Any] | None = None
        self._if_dynamic: Callable[..., Any] | None = None
        self._while_dynamic: Callable[..., Any] | None = None
        self._compare_executor: Callable[..., Any] | None = None
        self._builtin_redirector: Callable[..., Any] | None = None
        self._ifexp_dynamic: Callable[..., Any] | None = None

    @staticmethod
    def _default_builtin_redirector(fcn: Callable[..., Any]) -> Callable[..., Any]:
        """The default ``builtin_redirector``: every builtin is left as it is."""
        return fcn

    @staticmethod
    def _accepts_keyword(func: Callable[..., Any], keyword: str) -> bool:
        """Whether ``func`` declares ``keyword`` (or ``**kwargs``); True if unknown."""
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return True

        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
            for name, param in parameters.items()
        )

    @classmethod
    def _with_optional_keyword(
        cls, func: Callable[..., Any], keyword: str, default: Any
    ) -> Callable[..., Any]:
        """Let ``func`` ignore ``keyword`` when it does not declare it.

        The base passes every keyword of its protocol; an executor written
        against an older keyword set still works as long as the caller passes
        the default for the keywords it lacks.
        """
        if cls._accepts_keyword(func, keyword):
            return func

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            value = kwargs.pop(keyword, default)
            if value != default:
                raise DSLRuntimeError(f"{func.__name__} does not accept {keyword}")
            return func(*args, **kwargs)

        return wrapped

    @classmethod
    def _with_optional_branch_weights(
        cls, func: Callable[..., Any]
    ) -> Callable[..., Any]:
        return cls._with_optional_keyword(func, "branch_weights", None)

    def set_functions(
        self,
        *,
        loop_execute_range_dynamic: Callable[..., Any],
        if_dynamic: Callable[..., Any],
        while_dynamic: Callable[..., Any],
        compare_executor: Callable[..., Any],
        builtin_redirector: Callable[..., Any] = _default_builtin_redirector,
        ifexp_dynamic: Callable[..., Any] | None = None,
    ) -> None:
        """Give each Python keyword its meaning for this DSL.

        Called once per DSL instance by ``BaseDSL.__init__`` with what the
        DSL's ``ASTPreprocessorPlugin.executors`` returns.

        :param loop_execute_range_dynamic: builds the staged loop, called as
            ``(body, start, stop, step, write_args=, full_write_args_count=,
            write_args_names=, mutated_names=, unroll=, unroll_full=,
            **options)``
        :param if_dynamic: runs or stages an ``if``, called as ``(pred,
            then_block, else_block, write_args, full_write_args_count,
            write_args_names, mutated_names=, branch_weights=)``
        :param while_dynamic: runs or stages a ``while``, called as
            ``(before_block, after_block, write_args, full_write_args_count,
            write_args_names, mutated_names=)``
        :param compare_executor: evaluates a comparison chain, called as
            ``(left, comparators, ops)``
        :param builtin_redirector: maps a builtin function to the DSL's
            replacement; the default keeps every builtin
        :param ifexp_dynamic: stages a ternary expression, called as ``(pred,
            block_args, then_block, else_block, branch_weights=)``; ``None``
            leaves staged ternaries uninstalled

        An executor that does not declare ``mutated_names`` or
        ``branch_weights`` is wrapped so the base may still pass the default.
        """
        self._loop_execute_range_dynamic = self._with_optional_keyword(
            loop_execute_range_dynamic, "mutated_names", ()
        )
        self._if_dynamic = self._with_optional_keyword(
            self._with_optional_branch_weights(if_dynamic), "mutated_names", ()
        )
        self._while_dynamic = self._with_optional_keyword(
            while_dynamic, "mutated_names", ()
        )
        self._compare_executor = compare_executor
        self._builtin_redirector = builtin_redirector
        self._ifexp_dynamic = (
            self._with_optional_branch_weights(ifexp_dynamic)
            if ifexp_dynamic is not None
            else None
        )

    def _installed(self, slot: str) -> Callable[..., Any]:
        """Return the executor stored in ``slot`` or raise if none is set."""
        func = getattr(self, slot)
        if func is None:
            raise DSLRuntimeError(_EXECUTORS_NOT_INSTALLED)
        return func

    def for_execute(
        self,
        func: Callable[..., Any],
        start: Any,
        stop: Any,
        step: Any,
        write_args: Sequence[Any] = (),
        full_write_args_count: int = 0,
        write_args_names: Sequence[str] = (),
        mutated_names: tuple[str, ...] = (),
        unroll: int = -1,
        unroll_full: bool = False,
        **options: Any,
    ) -> Any:
        """What ``for i in range(start, stop, step):`` means when a bound is an MLIR op.

        :param func: the loop body, called as ``func(iv, *write_args)`` and
            returning the new values of ``write_args``
        :param write_args: the values the body may rebind, in ``write_args_names`` order
        :param full_write_args_count: how many leading write_args are stored
            in the body (the rest are method-call receivers)
        :param mutated_names: write_args that are only store bases (mutated
            in place, never rebound)
        :param unroll: unroll count for the loop annotation, ``-1`` for none
        :param unroll_full: request full unrolling
        :param options: sub-DSL loop options forwarded from the DSL ``range``
        :return: ``None``, one value, or a sequence of ``len(write_args)`` values
        """
        log().debug("start [%s] stop [%s] step [%s]", start, stop, step)

        return self._installed("_loop_execute_range_dynamic")(
            func,
            start,
            stop,
            step,
            write_args=write_args,
            full_write_args_count=full_write_args_count,
            write_args_names=write_args_names,
            mutated_names=mutated_names,
            unroll=unroll,
            unroll_full=unroll_full,
            **options,
        )

    def if_execute(
        self,
        pred: Any,
        then_block: Callable[..., Any],
        else_block: Callable[..., Any] | None = None,
        write_args: Sequence[Any] = (),
        full_write_args_count: int = 0,
        write_args_names: Sequence[str] = (),
        mutated_names: tuple[str, ...] = (),
        branch_weights: Optional[tuple[int, int]] = None,
    ) -> Any:
        """What ``if pred: ... else: ...`` means when ``pred`` is an MLIR op.

        Each arm is called as ``block(*write_args)`` and returns the new
        values; a ``None`` ``else_block`` passes the write_args through.
        :return: ``None``, one value, or a sequence of ``len(write_args)`` values
        """
        return self._installed("_if_dynamic")(
            pred,
            then_block,
            else_block,
            write_args,
            full_write_args_count,
            write_args_names,
            mutated_names=mutated_names,
            branch_weights=branch_weights,
        )

    def while_execute(
        self,
        while_before_block: Callable[..., Any],
        while_after_block: Callable[..., Any],
        write_args: Sequence[Any] = (),
        full_write_args_count: int = 0,
        write_args_names: Sequence[str] = (),
        mutated_names: tuple[str, ...] = (),
    ) -> Any:
        """What ``while cond: ...`` means when ``cond`` is an MLIR op.

        ``while_before_block(*write_args)`` returns ``[condition, write_args]``
        and ``while_after_block(*write_args)`` the write_args after one body.
        :return: ``None``, one value, or a sequence of ``len(write_args)`` values
        """
        return self._installed("_while_dynamic")(
            while_before_block,
            while_after_block,
            write_args,
            full_write_args_count,
            write_args_names,
            mutated_names=mutated_names,
        )

    def ifexp_execute(
        self,
        pred: Any,
        block_args: tuple[Any, ...],
        then_block: Callable[..., Any],
        else_block: Callable[..., Any],
        branch_weights: Optional[tuple[int, int]] = None,
    ) -> Any:
        """What ``a if pred else b`` means when ``pred`` is an MLIR op: each arm
        is a function of ``block_args`` and yields the value of the expression."""
        return self._installed("_ifexp_dynamic")(
            pred, block_args, then_block, else_block, branch_weights=branch_weights
        )


# =============================================================================
# The boundary: which arguments of a @jit call are Python values
# =============================================================================


def is_reserved_python_func_arg(
    arg_index: int, arg_name: str, func: Optional[Callable[..., Any]]
) -> bool:
    """True for the receiver of a method, never a runtime argument: ``self`` in
    the first position, or ``cls`` in the first position of a ``classmethod``."""

    if arg_index != 0:
        return False

    if arg_name == "self":
        return True

    if func:
        is_classmethod = isinstance(func, classmethod) or (
            hasattr(func, "__func__") and isinstance(func.__func__, classmethod)
        )
        return arg_name == "cls" and is_classmethod
    return False


def _is_type_argument(arg: Any, arg_annotation: Any) -> bool:
    """True if ``arg`` is a class passed where the annotation is absent or
    ``type[X]``: a type is always a Python value."""

    return isinstance(arg, type) and (
        arg_annotation is Parameter.empty or get_origin(arg_annotation) is type
    )


def _is_dsl_type_annotation(annotation: Any) -> bool:
    """True if the annotation declares a DSL type: a ``Numeric`` class, a
    ``Pointer[T]`` or a registered leaf class, possibly inside
    ``Annotated[...]``. ``int``, ``str``, a user class or none are Python
    annotations (their arguments are Python values)."""
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if isinstance(annotation, (NumericMeta, TypedPointer)):
        return True
    return isinstance(annotation, type) and is_leaf(annotation)


def is_argument_meta(
    arg: Any,
    arg_annotation: Any,
    arg_name: str,
    arg_index: int,
    owning_func: Callable[..., Any],
) -> bool:
    """Whether a ``@jit`` argument is a Python value the trace specialises on
    (the alternative being an MLIR op the compiled function takes at run time).

    Decided by the annotation and the value alone: a reserved receiver, a type
    argument and ``None`` are Python values; so is any argument whose
    annotation is not a DSL type and whose (adapted) value neither is nor
    contains a leaf instance. Everything else becomes an MLIR op.

    :param arg: The (adapted) argument value
    :param arg_annotation: The parameter's annotation
    """
    if (
        is_reserved_python_func_arg(arg_index, arg_name, owning_func)
        or _is_type_argument(arg, arg_annotation)
        or arg is None
    ):
        return True
    return not _is_dsl_type_annotation(arg_annotation) and not contains_leaf(arg)


# =============================================================================
# The active DSL's executor
# =============================================================================


def _active_executor() -> Executor:
    """The :class:`Executor` of the DSL whose function is being traced right now."""
    dsl = get_current_dsl()
    if dsl is None:
        # A rewritten `for`/`if`/`while` met an MLIR op outside any trace (a
        # value kept from an earlier trace, a `__wrapped__` call).
        raise DSLUserCodeError(
            DiagId.CALL_OUTSIDE_JIT,
            api="A `for`/`if`/`while` on an MLIR op",
            decorator="@jit",
        )
    active = getattr(dsl, "executor", None)
    if active is None:
        raise DSLRuntimeError(f"{type(dsl).__name__} has no `executor`")
    return active


class _ActiveExecutorProxy:
    """The module-level ``executor`` the rewritten code calls.

    The rewrite is produced once and does not know which DSL will run it, so
    it calls ``executor.for_execute(...)`` on this proxy; every attribute
    access resolves against the DSL that is tracing at that moment, so each
    DSL instance keeps its own :class:`Executor`.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_active_executor(), name)


executor = _ActiveExecutorProxy()


def _is_mlir_op_leaf(value: Any) -> bool:
    """One value, no container walk: a raw ``ir.Value`` or ``ir.BlockArgumentList``,
    or a registered leaf whose payload is an SSA value (a staged ``Int32``,
    never a host one or a host ``Pointer``). ``BaseDSL.is_mlir_op``'s default."""
    return isinstance(value, ir.BlockArgumentList) or is_staged_leaf(value)


def _any_leaf(value: Any, judge: Callable[[Any], bool]) -> bool:
    """Walk tuples, lists and frozen records; ``judge`` decides each leaf."""
    if isinstance(value, (tuple, list)):
        return any(_any_leaf(x, judge) for x in value)
    if is_frozen_dataclass(value) and not isinstance(value, type):
        return any(
            _any_leaf(getattr(value, f.name, None), judge)
            for f in dataclasses.fields(value)
        )
    return bool(judge(value))


def is_mlir_op(value: Any) -> bool:
    """Whether ``value`` is an MLIR op (its result) rather than a Python value.

    A tuple, list or frozen record (a ``@struct``, a frozen dataclass) is an
    MLIR op when any of its leaves is; dicts and other objects are Python
    values. Each leaf is judged by the DSL that is tracing
    (``BaseDSL.is_mlir_op``, which a sub-DSL with its own value model
    overrides); outside a trace a leaf is an MLIR op when it holds an SSA
    value. User code branches on it at trace time where Python values and
    MLIR ops need different handling::

        if is_mlir_op(n):
            ...  # an MLIR op: build IR
        else:
            ...  # a Python value: plain Python
    """
    dsl = get_current_dsl()
    judge = dsl.is_mlir_op if dsl is not None else _is_mlir_op_leaf
    return _any_leaf(value, judge)
