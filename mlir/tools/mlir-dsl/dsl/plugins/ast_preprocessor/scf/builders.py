# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``scf`` dialect: the explicit builders ``for_``, ``if_``, ``while_`` and ``yield_``.

The layer the ``scf`` AST preprocessor plugin targets, usable directly under
``@jit(preprocess=False)`` or from any DSL that wants to build an ``scf``
region by hand. Carries are pytrees of DSL values (``tree_utils``). An op
module, not a plugin; ``_promote_loop_bounds`` (the bound promotion of a
staged loop) lives here so the preprocessor's executors share it.
"""

import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional

from ..... import ir
from .....dialects import scf
from ....core.common import DSLUserCodeError
from ....core.user_op import dsl_user_op
from ....core.diagnostics import DiagId
from ....types.typing import (
    Boolean,
    DslType,
    Pointer,
    Struct,
    TypedPointer,
    _binary_op_type_promote,
    as_numeric,
)
from ....util.tree_utils import describe_tree_difference, tree_flatten, tree_unflatten

__all__ = [
    "WhileLoopContext",
    "_promote_loop_bounds",
    "for_",
    "if_",
    "while_",
    "yield_",
]


# =============================================================================
# Loop bounds
# =============================================================================


def _promote_loop_bounds(start: Any, stop: Any, step: Any) -> tuple:
    """``as_numeric`` the bounds, require integers, promote them to one dtype.

    :return: ``(start, stop, step, dtype)``, the bounds cast to the promoted
        integer dtype, which is also the induction variable's type
    """
    start_n = as_numeric(start)
    stop_n = as_numeric(stop)
    step_n = as_numeric(step)
    for name, n in (("start", start_n), ("stop", stop_n), ("step", step_n)):
        if not n.dtype.is_integer:
            raise DSLUserCodeError(
                DiagId.TYPE_LOOP_BOUND_NOT_INT, name=name, dtype=n.dtype.__name__
            )
    # Promote to a common integer type using pairwise type promotion
    _, _, tmp_dtype = _binary_op_type_promote(start_n, stop_n)
    _, _, dst_dtype = _binary_op_type_promote(start_n.to(tmp_dtype), step_n)
    step_ = step_n.to(dst_dtype)
    if isinstance(step_.value, int) and step_.value <= 0:
        raise DSLUserCodeError(
            f"The loop's `step` is `{step_.value}`, but a loop controlled by a "
            "runtime value needs a positive step.",
            suggestion=[
                "Loop over a positive count and compute the reversed or scaled index "
                "inside the body, e.g. `j = n - 1 - i`.",
                "If the bounds are Python values, a plain `range(...)` runs in Python, so the "
                "loop runs in Python.",
            ],
        )
    return start_n.to(dst_dtype), stop_n.to(dst_dtype), step_, dst_dtype


# =============================================================================
# Terminator
# =============================================================================


# The carries the enclosing ``for_``/``while_`` expects from ``yield_``:
# ``(treedef, SSA types, builder name)``, or None inside an ``if_`` arm, whose
# results the ``if_`` checks itself.
_yield_expectations = threading.local()


def _push_expected_yield(expected: Any) -> None:
    stack = getattr(_yield_expectations, "stack", None)
    if stack is None:
        stack = _yield_expectations.stack = []
    stack.append(expected)


def _pop_expected_yield() -> None:
    _yield_expectations.stack.pop()


def _expected_yield() -> Any:
    stack = getattr(_yield_expectations, "stack", None)
    return stack[-1] if stack else None


def _needs_terminator(block: ir.Block) -> bool:
    """True when ``block`` does not end with an ``scf.yield``."""
    ops = block.operations
    return len(ops) == 0 or ops[len(ops) - 1].operation.name != "scf.yield"


@dsl_user_op
def yield_(
    args: Any = (),
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Terminate the enclosing ``for_``/``if_``/``while_`` region with ``scf.yield``.

    :param args: The region's results: one value, or a list or tuple of them;
        containers, structs and the other leaves flatten to their SSA values
    """
    values, _, treedef = tree_flatten(
        list(args) if isinstance(args, (list, tuple)) else [args]
    )
    expected = _expected_yield()
    if expected is not None:
        entry_def, entry_types, op_name = expected
        difference = describe_tree_difference(entry_def, treedef, "the carries")
        if not difference and [v.type for v in values] != entry_types:
            difference = "the yielded types differ from the carries' types"
        if difference:
            raise DSLUserCodeError(
                DiagId.CONTAINER_STRUCTURE_CHANGED,
                var="the carried values",
                op_type=op_name,
                detail=f" ({difference})",
            )
    scf.yield_(values, loc=loc, ip=ip)


# =============================================================================
# For Loop
# =============================================================================


@dsl_user_op
def for_(
    start: Any,
    stop: Any = None,
    step: Any = None,
    iter_args: Optional[Sequence[Any]] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """Build an ``scf.for`` over integer bounds; a generator for ``for ... in``.

    ``for i in for_(n)`` yields the induction variable alone and the body
    needs no terminator. With ``iter_args`` it yields ``(iv, iter_args,
    results)`` (one carry: ``(iv, iter_arg, result)``) and the body must end
    with ``yield_(...)``. The bounds are promoted to one integer dtype, which
    is also ``iv``'s type.

    :param start: The lower bound, or the upper bound when ``stop`` is None
    :param stop: The upper bound (exclusive)
    :param step: The step (default 1); a Python step must be positive
    :param iter_args: The loop carries; flattened, so containers and structs
        work and are restored with their types
    :return: A generator yielding ``iv`` or ``(iv, iter_args, results)``
    """
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    start_, stop_, step_, dtype = _promote_loop_bounds(start, stop, step)

    prototypes = list(iter_args) if iter_args is not None else []
    ir_iter_args, _, treedef = tree_flatten(prototypes)
    for_op = scf.ForOp(
        start_.ir_value(),
        stop_.ir_value(),
        step_.ir_value(),
        ir_iter_args,
        loc=loc,
        ip=ip,
    )

    iv = dtype(for_op.induction_variable)
    new_results = tree_unflatten(treedef, list(for_op.results))
    new_iter_args = tuple(tree_unflatten(treedef, list(for_op.inner_iter_args)))

    with ir.InsertionPoint(for_op.body):
        _push_expected_yield((treedef, [v.type for v in ir_iter_args], "for_"))
        try:
            if len(new_iter_args) > 1:
                yield iv, new_iter_args, new_results
            elif len(new_iter_args) == 1:
                yield iv, new_iter_args[0], new_results[0]
            else:
                yield iv
        finally:
            _pop_expected_yield()
        # The body ran: carries need the user's `yield_`, a carry-free body
        # is terminated here unless the user did.
        if _needs_terminator(for_op.body):
            if new_iter_args:
                raise DSLUserCodeError(
                    "The body of this `for_` has `iter_args` but did not end with "
                    "`yield_(...)`.",
                    suggestion="End the loop body with `yield_([...])`, passing the "
                    "next value of every carry in order.",
                )
            scf.yield_([], loc=loc)
        else:
            ops = for_op.body.operations
            yielded = len(ops[len(ops) - 1].operands)
            if yielded != len(ir_iter_args):
                raise DSLUserCodeError(
                    f"The body of this `for_` yielded {yielded} value(s) for "
                    f"{len(ir_iter_args)} carried value(s).",
                    suggestion="Pass the next value of every carry to `yield_([...])` in "
                    "the order of `iter_args`; a struct or tuple carry counts one "
                    "value per field.",
                )


# =============================================================================
# If/Else
# =============================================================================


def _result_mlir_types(result_type: Any) -> list[ir.Type]:
    """The SSA types of one declared ``if_`` result: a DSL type is one value, a
    ``@struct`` class its fields (recursively)."""
    if isinstance(result_type, type) and issubclass(result_type, Struct):
        return [
            t
            for ann in result_type._field_annotations.values()
            for t in _result_mlir_types(ann)
        ]
    return [result_type.mlir_type]  # a numeric dtype or a ``Pointer[T]``


def _flatten_result(result_type: Any, value: Any) -> list[ir.Value]:
    """The SSA values an arm yields for one declared result."""
    if isinstance(result_type, type) and issubclass(result_type, Struct):
        if not isinstance(value, result_type):
            raise DSLUserCodeError(
                DiagId.ARG_ANNOTATION_MISMATCH,
                num=1,
                arg_name="result",
                expected=f"a `{result_type.__name__}`",
                got=type(value).__name__,
            )
        return [
            v
            for name, ann in result_type._field_annotations.items()
            for v in _flatten_result(ann, getattr(value, name))
        ]
    if isinstance(result_type, TypedPointer):
        if not isinstance(value, Pointer):
            raise DSLUserCodeError(
                DiagId.ARG_ANNOTATION_MISMATCH,
                num=1,
                arg_name="result",
                expected=f"a `{result_type!r}`",
                got=type(value).__name__,
            )
        return [value.ir_value()]
    return [result_type(value).ir_value()]


def _restore_result(result_type: Any, values: Any) -> Any:
    """Rebuild one declared result from an iterator of ``scf.if`` result values."""
    if isinstance(result_type, type) and issubclass(result_type, Struct):
        return result_type._build(
            {
                name: _restore_result(ann, values)
                for name, ann in result_type._field_annotations.items()
            }
        )
    if isinstance(result_type, TypedPointer):
        return Pointer(next(values), dtype=result_type.dtype, space=result_type.space)
    return result_type(next(values))


@dsl_user_op
def if_(
    cond: Any,
    then_body: Callable[..., Any],
    else_body: Optional[Callable[..., Any]] = None,
    input_args: Optional[Sequence[Any]] = None,
    return_types: Optional[Sequence[Any]] = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """Build an ``scf.if`` from Python callables for the arms.

    Each body is called with ``input_args`` under its block's insertion point
    and returns the arm's results, which are cast to ``return_types`` and
    yielded. A Python ``bool`` condition still builds the ``scf.if``.

    :param cond: The condition; anything ``Boolean`` accepts
    :param then_body: Called to build the then arm
    :param else_body: Called to build the else arm; with ``return_types`` and
        no ``else_body`` the else arm yields ``input_args`` unchanged
    :param input_args: Positional arguments of both bodies
    :param return_types: The DSL type of each result: a numeric type such as
        ``Int32`` or a ``@struct`` class (one result per field); ``None`` for an ``if`` without
        results
    :return: The typed result, the list of them, or ``[]`` without results
    """
    input_args = list(input_args or [])

    mlir_return_types = []
    if return_types is not None:
        for t in return_types:
            is_struct_class = isinstance(t, type) and issubclass(t, Struct)
            if not isinstance(t, DslType) and not is_struct_class:
                raise DSLUserCodeError(
                    f"`if_` expects DSL types in `return_types`, but got `{t!r}`.",
                    suggestion="Pass DSL types such as `Int32` or `Float32`, or a `@struct` class.",
                )
            mlir_return_types.extend(_result_mlir_types(t))

    # Without an else but with results, synthesize a passthrough else that
    # yields the input_args unchanged.
    if else_body is None and return_types is not None:
        else_body = lambda *args: args if len(args) > 1 else args[0]  # noqa: E731
    has_else = else_body is not None

    if_op = scf.IfOp(
        Boolean(cond).ir_value(), mlir_return_types, has_else=has_else, loc=loc, ip=ip
    )

    def _execute_and_yield_out(body: Callable[..., Any], input_args: list[Any]) -> None:
        yield_vals = body(*input_args)
        if return_types is not None:
            if yield_vals is None:
                yield_vals = []
            elif not isinstance(yield_vals, (list, tuple)):
                yield_vals = [yield_vals]  # the body returned one value
            if len(yield_vals) != len(return_types):
                raise DSLUserCodeError(
                    f"An `if_` body returned {len(yield_vals)} value(s), but "
                    f"`return_types` lists {len(return_types)}.",
                    suggestion="Return one value per entry of `return_types` from "
                    "both bodies, in the same order.",
                )
            yield_vals = [
                v
                for t, r in zip(return_types, yield_vals)
                for v in _flatten_result(t, r)
            ]
        _push_expected_yield(None)  # the if_ checks its arms itself
        try:
            yield_(yield_vals if yield_vals is not None else [])
        finally:
            _pop_expected_yield()

    # Generate the body for 'then'.
    with ir.InsertionPoint(if_op.then_block):
        _execute_and_yield_out(then_body, input_args)

    # Generate the body for 'else' if provided.
    if has_else:
        with ir.InsertionPoint(if_op.else_block):
            _execute_and_yield_out(else_body, input_args)  # type: ignore[arg-type]

    # Wrap the results with their DSL types.
    if return_types is None:
        return []
    results = iter(if_op.results)
    vals = [_restore_result(t, results) for t in return_types]
    if len(vals) == 1:
        return vals[0]
    return vals


# =============================================================================
# While Loop
# =============================================================================


class WhileLoopContext:
    """Context manager over an ``scf.while``.

    ``with while_(inputs, condition) as carries:`` traces ``condition`` into
    the before block (``scf.condition`` forwards the carries unchanged) and
    opens the after block for the body, which must end with
    ``yield_(new_carries)``; ``.results`` restores the loop's results in the
    shape of ``inputs``. ``inputs`` are flattened, so containers and structs
    work.
    """

    def __init__(
        self,
        inputs: Sequence[Any],
        condition: Callable[..., Any],
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        # The inputs' tree restores the carries and results with their types.
        self.inputs = list(inputs)
        self.input_ir_values, _, self.treedef = tree_flatten(self.inputs)
        self.condition = condition
        self.input_ir_types = [i.type for i in self.input_ir_values]
        self.while_op = scf.WhileOp(
            self.input_ir_types, self.input_ir_values, loc=loc, ip=ip
        )

        self.before_region = self.while_op.before
        self.after_region = self.while_op.after
        self.before_block = self.before_region.blocks.append(*self.input_ir_types)
        self.after_block = self.after_region.blocks.append(*self.input_ir_types)
        self.ipoint_op: Optional[ir.InsertionPoint] = None

    def __enter__(self) -> list[Any]:
        with ir.InsertionPoint(self.before_block):
            args = tree_unflatten(self.treedef, list(self.before_block.arguments))
            cond = self.condition(*args)
            scf.ConditionOp(Boolean(cond).ir_value(), list(self.before_block.arguments))
        self.ipoint_op = ir.InsertionPoint(self.after_block)
        self.ipoint_op.__enter__()
        _push_expected_yield((self.treedef, self.input_ir_types, "while_"))
        return tree_unflatten(self.treedef, list(self.after_block.arguments))

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: object,
    ) -> None:
        _pop_expected_yield()
        if self.ipoint_op is not None:
            self.ipoint_op.__exit__(exc_type, exc_value, traceback)
        if exc_type is None and _needs_terminator(self.after_block):
            raise DSLUserCodeError(
                "The body of this `while_` did not end with `yield_(...)`.",
                suggestion="End the `with while_(...)` block with `yield_([...])`, "
                "passing the next value of every carry in order.",
            )

    @property
    def results(self) -> list[Any]:
        """The loop's results, restored in the shape and types of ``inputs``."""
        return tree_unflatten(self.treedef, list(self.while_op.results))


@dsl_user_op
def while_(
    inputs: Sequence[Any],
    condition: Callable[..., Any],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> WhileLoopContext:
    """Build an ``scf.while``; use as ``with while_(inputs, condition) as carries:``.

    :param inputs: The initial carries; flattened, so containers and structs
        work
    :param condition: Called with the carries, returns the loop condition
    :return: The :class:`WhileLoopContext`; its ``results`` are the loop's
    """
    return WhileLoopContext(inputs, condition, loc=loc, ip=ip)
