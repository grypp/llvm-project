# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Staging: the executor slots of a DSL instance and the staged-value tests.

Every ``BaseDSL`` owns one :class:`Executor`; the ``ASTPreprocessorPlugin`` of
the DSL fills its slots (``set_functions``) with the callables that stage a
``for``/``if``/``while`` on a runtime value, compare, redirect built-ins and
decide what is staged. The helpers the rewritten code calls
(``plugins/ast_preprocessor/helpers.py``) reach the active DSL's executor
through the module-level ``executor`` proxy. :func:`is_dynamic_expr` is the
user-facing question "is this value staged?", answered by the active DSL.
"""

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, Optional

from ... import ir
from .common import DSLRuntimeError, get_current_dsl
from ..util.logger import log
from ..util.tree_utils import is_frozen_dataclass, is_staged_leaf

__all__ = ["Executor", "executor", "is_dynamic_expr", "is_dynamic_expression"]

_EXECUTORS_NOT_INSTALLED = (
    "executors not installed: call `set_functions` in your DSL's `__init__`"
)


class Executor:
    """
    The Executor class handles staged and Meta (trace-time) execution
    of "for" loops and "if-else-elif" statements.

    Every ``BaseDSL`` instance owns one; the DSL's ``__init__`` fills the slots
    through :meth:`set_functions`. Until then every slot is ``None`` and a
    staged region raises ``DSLRuntimeError``.
    """

    def __init__(self) -> None:
        self._is_dynamic_expression: Callable[..., Any] | None = None
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
        is_dynamic_expression: Callable[..., Any],
        loop_execute_range_dynamic: Callable[..., Any],
        if_dynamic: Callable[..., Any],
        while_dynamic: Callable[..., Any],
        compare_executor: Callable[..., Any],
        builtin_redirector: Callable[..., Any] = _default_builtin_redirector,
        ifexp_dynamic: Callable[..., Any] | None = None,
    ) -> None:
        """Install the executors of a DSL (called from its ``__init__``).

        :param is_dynamic_expression: ``value -> bool``, the DSL's notion of a
            staged value
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
        self._is_dynamic_expression = is_dynamic_expression
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
        """Run ``func`` as the body of a staged loop over ``range(start, stop, step)``.

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
        """Run or stage an ``if`` whose arms are ``then_block``/``else_block``.

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
        """Run or stage a ``while``.

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
        """Stage ``then_block(*block_args) if pred else else_block(*block_args)``."""
        return self._installed("_ifexp_dynamic")(
            pred, block_args, then_block, else_block, branch_weights=branch_weights
        )


# =============================================================================
# Active executor
# =============================================================================


def _active_executor() -> Executor:
    """Return the :class:`Executor` of the DSL that is currently tracing."""
    dsl = get_current_dsl()
    if dsl is None:
        raise DSLRuntimeError("control-flow helper called outside a DSL trace")
    active = getattr(dsl, "executor", None)
    if active is None:
        raise DSLRuntimeError(f"{type(dsl).__name__} has no `executor`")
    return active


class _ActiveExecutorProxy:
    """Module-level ``executor`` that forwards to the active DSL's executor.

    The selectors below are spelled as if ``executor`` were one object; each
    attribute access resolves against ``get_current_dsl().executor`` so every
    DSL instance keeps its own :class:`Executor`.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_active_executor(), name)


executor = _ActiveExecutorProxy()


def is_dynamic_expression(value: object) -> bool:
    """True when ``value`` is, or a tuple/list of it holds, a staged value.

    A registered leaf is dynamic when its payload is an ``ir.Value`` (for a
    ``Numeric``, only the staged payload; a host ``Pointer`` is not); a raw
    ``ir.Value`` or ``ir.BlockArgumentList`` always is. Tuples and lists are
    walked recursively; every other object (dicts included) is a Meta value.

    :param value: The object to test
    :return: Whether a staged value was found
    """
    if isinstance(value, (tuple, list)):
        for x in value:
            if is_dynamic_expression(x):
                return True
    elif isinstance(value, ir.BlockArgumentList) or is_staged_leaf(value):
        return True
    return False


def _is_dynamic(value: Any) -> bool:
    """The active DSL's notion of a staged value; nothing is staged outside a trace."""
    if get_current_dsl() is None:
        return False
    return bool(_active_executor()._installed("_is_dynamic_expression")(value))


def is_dynamic_expr(value: Any) -> bool:
    """Whether ``value`` is a staged (runtime) value.

    Inside a trace this is the active DSL's notion (its ``is_dynamic_expression``
    executor function); outside a trace it is the base notion of
    ``util.tree_utils.is_dynamic_expression``; a ``@struct`` or frozen
    dataclass is staged when any field is. User code branches on it at
    trace time where it needs different handling for Meta and staged values::

        if is_dynamic_expr(n):
            ...  # staged: build IR
        else:
            ...  # Meta: plain Python
    """
    if is_frozen_dataclass(value) and not isinstance(value, type):
        # A @struct or a frozen dataclass: staged when any field is.
        return any(
            is_dynamic_expr(getattr(value, f.name)) for f in dataclasses.fields(value)
        )
    if get_current_dsl() is None:
        return is_dynamic_expression(value)
    return _is_dynamic(value)
