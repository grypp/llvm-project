# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Trace-time helpers referenced by the code the AST preprocessor generates.

``DSLPreprocessor`` rewrites native ``for``/``if``/``while`` statements into
region functions decorated with the selectors below (``loop_selector``,
``if_selector``, ``while_selector``) and routes expressions through the
executors (``if_executor``, ``ifExp_executor``, ``compare_executor``,
``assert_executor``, ...). The selectors forward to the :class:`Executor` of
the DSL that is tracing, which the DSL fills through
:meth:`Executor.set_functions` (Design 7.3). The module also holds the loop
iterables (``range``) and the trace-time dispatch helpers
(``materialize_for_iter``, ``is_dynamic_range``) that choose between a staged
loop and a native Python loop (Design 7.4).
"""

import builtins
import inspect
import itertools
from collections.abc import Callable, Iterator, Sequence
from functools import wraps
from types import BuiltinFunctionType
from typing import Any, Optional

from ...core.common import DSLRuntimeError, DSLUserCodeError, get_current_dsl
from ...core.diagnostics import DiagId, _is_dsl_module
from ...core.executor import (
    Executor,
    _active_executor,
    _is_dynamic,
    executor,
    is_dynamic_expr,
)
from ...util.logger import log
from ...core.executor import is_dynamic_expression

__all__ = [
    "Executor",
    "executor",
    "loop_selector",
    "if_selector",
    "while_selector",
    "if_executor",
    "while_executor",
    "ifExp_executor",
    "range",
    "is_dynamic_range",
    "materialize_for_iter",
    "register_deferred_for_error",
    "raise_deferred_for_error",
    "discard_deferred_for_error",
    "is_dynamic_expr",
    "assert_executor",
    "bool_short_circuits",
    "bool_cast",
    "compare_executor",
    "cf_symbol_check",
    "redirect_builtin_function",
    "get_locals_or_none",
    "closure_check",
    "early_exit_predicate",
]

_DSL_CONDITION_ATTR = "_dsl_condition"
_DSL_BRANCH_WEIGHTS_ATTR = "_dsl_branch_weights"


def _extract_condition_branch_weights(
    pred: Any,
) -> tuple[Any, Optional[tuple[int, int]]]:
    """Unwrap a sub-DSL's weighted condition into ``(condition, weights)``.

    A predicate object carrying ``_dsl_condition`` and a two-element
    ``_dsl_branch_weights`` is unwrapped; any other predicate is returned as is
    with ``None`` weights.
    """
    branch_weights = getattr(pred, _DSL_BRANCH_WEIGHTS_ATTR, None)
    if branch_weights is None:
        return pred, None

    weights = tuple(branch_weights)
    if len(weights) != 2:
        return pred, None

    return getattr(pred, _DSL_CONDITION_ATTR), (weights[0], weights[1])


# =============================================================================
# Decorator
# =============================================================================


def loop_selector(
    start: Any,
    stop: Any,
    step: Any,
    *,
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: tuple[str, ...] = (),
    unroll: int = -1,
    unroll_full: bool = False,
    **options: Any,
) -> Callable[..., Any]:
    """Decorator the preprocessor puts on a staged loop body (Design 7.2).

    The decorated body function is executed immediately through
    ``Executor.for_execute`` and the decoration evaluates to the loop's
    results, which the generated code writes back to the ``write_args`` names.
    """
    log().debug(
        "start [%s] stop [%s] step [%s] write_args [%s] full_write_args_count [%s] "
        "write_args_names [%s] mutated_names [%s] unroll [%s] unroll_full [%s] "
        "options [%s]",
        start,
        stop,
        step,
        write_args,
        full_write_args_count,
        write_args_names,
        mutated_names,
        unroll,
        unroll_full,
        options,
    )

    def ir_loop(func: Callable[..., Any]) -> Any:
        return executor.for_execute(
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

    return ir_loop


def if_selector(pred: Any, write_args: Sequence[Any] = ()) -> Callable[..., Any]:
    """Decorator on a generated ``if`` region: calls it as ``region(pred, *write_args)``.

    The region function defines the arm blocks and returns ``if_executor(...)``.
    """
    log().debug("pred [%s] write_args [%s]", pred, write_args)

    def ir_if(func: Callable[..., Any]) -> Any:
        return func(pred, *write_args)

    return ir_if


def while_selector(*, write_args: Sequence[Any] = ()) -> Callable[..., Any]:
    """Decorator on a generated ``while`` region: calls it as ``region(*write_args)``.

    The region function defines the before/after blocks and returns
    ``while_executor(...)``.
    """

    def ir_while_loop(func: Callable[..., Any]) -> Any:
        return func(*write_args)

    return ir_while_loop


def while_executor(
    while_before_block: Callable[..., Any],
    while_after_block: Callable[..., Any],
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: tuple[str, ...] = (),
) -> Any:
    """Forward a generated ``while`` region to the active ``Executor.while_execute``."""
    return executor.while_execute(
        while_before_block,
        while_after_block,
        write_args,
        full_write_args_count,
        write_args_names,
        mutated_names=mutated_names,
    )


def if_executor(
    pred: Any,
    then_block: Callable[..., Any],
    else_block: Callable[..., Any] | None = None,
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: tuple[str, ...] = (),
    branch_weights: Optional[tuple[int, int]] = None,
) -> Any:
    """Forward a generated ``if`` region to the active ``Executor.if_execute``.

    A weighted condition object (a sub-DSL wrapper) is unwrapped first; an
    explicit ``branch_weights`` wins over the wrapper's.
    """
    pred, wrapped_branch_weights = _extract_condition_branch_weights(pred)
    if branch_weights is None:
        branch_weights = wrapped_branch_weights

    return executor.if_execute(
        pred,
        then_block,
        else_block,
        write_args,
        full_write_args_count,
        write_args_names,
        mutated_names=mutated_names,
        branch_weights=branch_weights,
    )


def ifExp_executor(
    *,
    pred: Any,
    block_args: tuple[Any, ...],
    then_block: Callable[..., Any],
    else_block: Callable[..., Any],
    branch_weights: Optional[tuple[int, int]] = None,
) -> Any:
    """Evaluate a rewritten ternary ``then if pred else else``.

    A Python predicate runs one arm here; a staged predicate goes to the
    active ``Executor.ifexp_execute``. ``block_args`` are the comprehension
    and lambda variables the arms close over.
    """
    pred, wrapped_branch_weights = _extract_condition_branch_weights(pred)
    if branch_weights is None:
        branch_weights = wrapped_branch_weights

    if not _is_dynamic(pred):
        return then_block(*block_args) if pred else else_block(*block_args)
    return executor.ifexp_execute(
        pred, block_args, then_block, else_block, branch_weights=branch_weights
    )


# =============================================================================
# Range
# =============================================================================


class range:
    """Trace-time representation of a DSL range: the loop over it is always staged.

    Accepts ``range(stop)``, ``range(start, stop)`` and ``range(start, stop,
    step)``; ``unroll``/``unroll_full`` become the loop annotation of the
    ``scf.for``; any other keyword is forwarded untouched to the DSL's loop
    executor, which consumes or rejects it (``CALL_UNEXPECTED_KWARG`` in the
    base). Iterating it directly is an error: only the preprocessor's rewrite
    consumes it.
    """

    def __init__(
        self,
        *args: Any,
        unroll: int = -1,
        unroll_full: bool = False,
        **options: Any,
    ) -> None:
        if len(args) not in (1, 2, 3):
            raise DSLUserCodeError(DiagId.UNSUP_RANGE_ARGS)
        self.start = 0 if len(args) == 1 else args[0]
        self.stop = args[0] if len(args) == 1 else args[1]
        self.step = args[2] if len(args) == 3 else 1
        self.unroll = unroll
        self.unroll_full = unroll_full
        self.options: dict[str, Any] = dict(options)

    def __iter__(self) -> Iterator[int]:
        # Only reachable when the preprocessor did not rewrite the `for`.
        raise DSLUserCodeError(DiagId.PHASE_DYNAMIC_INDEX)


def is_dynamic_range(value: object) -> bool:
    """Return whether a trace-time iterable should lower to a staged loop.

    Only the DSL :class:`range` stages; a ``builtins.range`` runs as a native
    Python loop (``materialize_for_iter`` picks between the two).
    """
    return isinstance(value, range)


class _DeferredForErrors:
    """Errors from the speculative dynamic arm of a preprocessed ``for``.

    The preprocessor registers the error at rewrite time and bakes the id into
    the generated code, which raises or discards it once trace-time dispatch
    has selected an arm. One registry serves the whole process: a preprocessed
    function can be traced more than once and under any DSL instance, and every
    dynamic selection must report the same preprocessor diagnostic.
    """

    def __init__(self) -> None:
        self._errors: dict[int, BaseException] = {}
        self._ids = itertools.count()

    def register(self, error: BaseException) -> int:
        error_id = next(self._ids)
        self._errors[error_id] = error
        return error_id

    def raise_(self, error_id: int) -> None:
        raise self._errors[error_id]

    def discard(self, error_id: int) -> None:
        self._errors.pop(error_id, None)


_deferred_for_errors = _DeferredForErrors()


def register_deferred_for_error(error: BaseException) -> int:
    """Retain a speculative dynamic-loop error until trace-time dispatch."""
    return _deferred_for_errors.register(error)


def raise_deferred_for_error(error_id: int) -> None:
    """Raise an error selected by dynamic dispatch."""
    _deferred_for_errors.raise_(error_id)


def discard_deferred_for_error(error_id: int) -> None:
    """Release an error from an unselected dynamic branch."""
    _deferred_for_errors.discard(error_id)


def materialize_for_iter(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Build the iterable of a dispatched ``for`` (Design 7.4).

    A ``builtins.range`` stages (becomes the DSL :class:`range`) only when one
    of its bounds is a staged value of the active DSL; otherwise, and outside
    a trace, it is a native Python loop and the loop options (``unroll=`` and
    the like, which only describe a staged loop) are dropped: ``builtins.range``
    takes no keywords. The DSL :class:`range` stages unconditionally; any other
    factory is called as written.
    """
    if factory is builtins.range:
        if _is_dynamic(args):
            return range(*args, **kwargs)
        return builtins.range(*args)
    if factory is range:
        return range(*args, **kwargs)

    return factory(*args, **kwargs)


# =============================================================================
# If expressions
# =============================================================================


# =============================================================================
# Assertion & casting
# =============================================================================


def assert_executor(test: Any, msg: str | None = None) -> None:
    """The rewrite of ``assert test, msg``: a Python assertion on a Python value.

    A staged ``test`` cannot be decided at trace time and raises
    ``PHASE_REQUIRES_CONSTANT``; a Python ``test`` behaves like ``assert``.
    """
    # A staged value must not be converted to bool implicitly, hence the
    # explicit None check before asking the DSL.
    if test is not None and _is_dynamic(test):
        raise DSLUserCodeError(
            DiagId.PHASE_REQUIRES_CONSTANT,
            what="`assert`",
        )
    assert test, msg


def bool_short_circuits(value: Any, short_circuit_value: bool) -> bool:
    """True when the and/or LHS *value* short-circuits in Python: it is a
    plain ``bool`` whose truth equals *short_circuit_value*. A staged or
    non-bool value answers False, so the rewrite's other arm evaluates (the
    ``and_``/``or_`` helper)."""
    return type(value) is bool and value == short_circuit_value


def bool_cast(value: Any) -> bool:
    """The rewrite of ``bool(value)``: ``PHASE_REQUIRES_CONSTANT`` on a staged value."""
    if _is_dynamic(value):
        raise DSLUserCodeError(
            DiagId.PHASE_REQUIRES_CONSTANT,
            what="Explicit boolean conversion",
        )
    return bool(value)


def compare_executor(left: Any, comparators: list[Any], ops: list[Any]) -> Any:
    """
    Executes comparison operations with a left operand and a list of comparators.

    :param left: The leftmost value in the comparison chain
    :param comparators: A list of values to compare against
    :param ops: A list of comparison operators to apply
    :return: The result of the comparison chain
    """
    return _active_executor()._installed("_compare_executor")(left, comparators, ops)


# =============================================================================
# Control flow checks
# =============================================================================
def cf_symbol_check(symbol: Any) -> None:
    """
    Check if the symbol is control flow symbol from a DSL package.

    A symbol qualifies when it is a registered DSL package (``mlir.dsl`` or a
    sub-DSL that called ``register_dsl_package``) or is defined in one.
    """
    name = symbol.__name__
    module = symbol if inspect.ismodule(symbol) else inspect.getmodule(symbol)
    module_name = module.__name__ if module is not None else ""
    if not _is_dsl_module(module_name):
        raise DSLUserCodeError(
            DiagId.CALL_WRONG_IMPORT,
            name=name,
        )


def redirect_builtin_function(fcn: Any) -> Any:
    """Map a builtin the rewritten code calls to the DSL's replacement.

    ``bool`` becomes :func:`bool_cast`; ``exec``/``eval`` are rejected
    (``UNSUP_BUILTIN``); any other builtin function goes through the active
    DSL's ``builtin_redirector``; anything else is returned unchanged.
    """
    if fcn is builtins.bool:
        return bool_cast

    if isinstance(fcn, BuiltinFunctionType):
        if fcn in (builtins.exec, builtins.eval):
            raise DSLUserCodeError(
                DiagId.UNSUP_BUILTIN,
                name=fcn.__name__,
            )
        redirector = executor._builtin_redirector
        if redirector is not None:
            return redirector(fcn)
    return fcn


def get_locals_or_none(locals: dict[str, Any], symbols: list[str]) -> list[Any]:
    """The values of ``symbols`` in a ``locals()`` dict, ``None`` for an unbound one.

    This seeds the ``write_args`` of a generated region (Design 7.2): a name
    first bound inside the region enters as ``None``.
    """
    return [locals.get(symbol) for symbol in symbols]


def early_exit_predicate(
    predicate: Any,
    *,
    kind: str,
    where: str,
    filename: str | None = None,
    lineno: int | None = None,
    col_offset: int | None = None,
    end_col_offset: int | None = None,
) -> bool:
    """The test of a native ``if``/``while`` that owns an early exit.

    The preprocessor keeps such a statement in Python (its ``return``, ``raise``,
    ``break`` or ``continue`` cannot leave an outlined region) and routes the
    test through here: a Meta value is returned as Python's ``bool`` of it, a
    staged value raises ``UNSUP_EARLY_EXIT`` pointing at the exit statement.

    :param predicate: The evaluated test expression
    :param kind: ``"return"``, ``"raise"``, ``"break"`` or ``"continue"``
    :param where: The function the exit would leave, for the message
    """
    if _is_dynamic(predicate):
        raise DSLUserCodeError(
            DiagId.UNSUP_EARLY_EXIT,
            filename=filename,
            lineno=lineno,
            col_offset=col_offset,
            end_col_offset=end_col_offset,
            kind=kind,
            where=where,
        )
    return bool(predicate)


def closure_check(
    closures: list[Any], _visited: set[tuple[str, int]] | None = None
) -> None:
    """Reject a nested function called from a staged region that captures a variable.

    Captured modules are fine and captured functions are checked recursively;
    any other captured name raises ``SCOPE_CLOSURE_CAPTURE``. The preprocessor
    emits ``closure_check([...])`` before a region that calls nested
    definitions; a AST preprocessor plugin with ``closure_check=False`` skips the check.
    """
    plugin = getattr(get_current_dsl(), "ast_preprocessor", None)
    if plugin is not None and not plugin.closure_check:
        return
    if _visited is None:
        _visited = set()

    for closure in closures:
        # Use (function name, id) as identity to skip already-processed closures
        # and prevent infinite recursion with mutually-recursive captured functions
        closure_identity = (closure.__name__, id(closure))
        if closure_identity in _visited:
            continue
        _visited.add(closure_identity)

        closure_vars = inspect.getclosurevars(closure)
        for name, value in closure_vars.nonlocals.items():
            if inspect.ismodule(value):
                continue
            if inspect.isfunction(value) or inspect.ismethod(value):
                closure_check([value], _visited)
                continue
            raise DSLUserCodeError(
                DiagId.SCOPE_CLOSURE_CAPTURE,
                func_name=closure.__name__,
                var_name=name,
            )
