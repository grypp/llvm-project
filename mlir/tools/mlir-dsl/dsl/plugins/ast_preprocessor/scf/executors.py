# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
The ``scf`` executors of the ``scf`` AST preprocessor plugin and the logical
helpers of its rewrites.

The preprocessor turns native ``for``/``if``/``while`` statements into region
functions that follow the write_args protocol; ``BaseDSL.__init__`` installs
the executors below (``ASTPreprocessor.executors``) through
``Executor.set_functions``. Every executor threads
the region's write_args through ``tree_flatten``/``tree_unflatten``: leaves
become ``scf`` iter_args and results, META slots pass through and must stay
equal, a reference leaf that is only a store base is not carried, and a slot
that is META at entry and a leaf at the yield is a mutation of a Python value.
Loop bounds keep their promoted integer dtype (no ``index``), ``if`` arms are
traced into detached holder blocks so the result types follow from the arms,
and a ``while`` probes its condition once to choose between a Python loop and
``scf.while``.
"""

import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from ..... import ir
from .....dialects import scf
from ....core.common import DSLRuntimeError, DSLUserCodeError
from ....core.diagnostics import DiagId
from ....core.staging import is_mlir_op
from .builders import _promote_loop_bounds
from ....types.typing import Boolean, as_numeric
from ....util.logger import log
from ....util.tree_utils import (
    check_tree_equal,
    describe_tree_difference,
    is_frozen_dataclass,
    is_reference_leaf,
    Leaf,
    leaf_type_name,
    PyTreeDef,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)

__all__ = ["LoopUnroll", "ScfGenerator", "and_", "or_", "not_", "any_", "all_", "in_"]

# =============================================================================
# AST Helpers
# =============================================================================

# The name the SCF lowering copies from ``scf.for`` onto the latch branch.
_LOOP_ANNOTATION_ATTR = "llvm.loop_annotation"


class LoopUnroll(ir.Attribute):
    """``#llvm.loop_annotation<unroll = <...>>`` built from ``count``/``full``.

    ``count=1`` also sets ``disable = true`` (an unroll count of one asks for
    no unrolling); other keywords are ignored.
    """

    def __init__(self, **kwargs: Union[int, bool]) -> None:
        valid_keys = ("count", "full")

        def to_mlir_attr(val: Union[int, bool]) -> str:
            if isinstance(val, bool):
                return "true" if val else "false"
            elif isinstance(val, int):
                return f"{val} : i32"
            else:
                raise DSLRuntimeError(f"{type(val)} is not supported")

        cfg = {key: to_mlir_attr(kwargs[key]) for key in valid_keys if key in kwargs}
        if kwargs.get("count", None) == 1:
            cfg["disable"] = "true"

        unroll = "<" + ", ".join(f"{key} = {value}" for key, value in cfg.items()) + ">"

        super().__init__(
            ir.Attribute.parse(f"#llvm.loop_annotation<unroll = {unroll}>")
        )


def _meta_equal(lhs: Any, rhs: Any) -> bool:
    """Compare two META values with ``==``, falling back to identity."""
    if lhs is rhs:
        return True
    try:
        return bool(lhs == rhs)
    except Exception:
        return False


def _leaf_type_name(leaf: Leaf) -> str:
    """Name the value a ``Leaf`` stands for, for a join diagnostic."""
    if leaf.is_none:
        return "None"
    if leaf.is_meta:
        return type(leaf.meta).__name__
    return leaf_type_name(leaf.prototype)


def _prototype_differs(lhs: Leaf, rhs: Leaf) -> bool:
    """The join type-stability rule on two non-None, non-META leaves."""
    if lhs.ir_type_str != rhs.ir_type_str:
        return True
    lp, rp = lhs.prototype, rhs.prototype
    if isinstance(lp, type) or isinstance(rp, type):
        return lp is not rp
    if isinstance(lp, tuple) and isinstance(rp, tuple):
        return lp != rp
    return type(lp) is not type(rp)


def _detached_block() -> tuple[ir.Operation, ir.Block]:
    """A holder op that is inserted nowhere, with one empty block to trace into."""
    holder = ir.Operation.create("scf.execute_region", regions=1, ip=False)
    return holder, holder.regions[0].blocks.append()


def _condition(pred: Any, what: str) -> Boolean:
    """The ``Boolean`` of a staged condition. A tuple, list or record holding
    MLIR ops is an MLIR op to ``is_mlir_op`` but not a condition: it is refused
    by name instead of as an argument of a ``Boolean(...)`` call the user never
    wrote."""
    if isinstance(pred, (tuple, list)) or is_frozen_dataclass(pred):
        raise DSLUserCodeError(
            DiagId.ARG_NOT_NUMERIC, arg_name=what, arg_type=type(pred).__name__
        )
    return Boolean(pred)


def _create_if_op(
    pred: Any, result_types: Sequence[ir.Type], blocks: Sequence[ir.Block]
) -> ir.Operation:
    """Create ``scf.if`` from the predicate and move the traced arm blocks into it."""
    pred_ = _condition(pred, "if condition")
    try:
        if_op = ir.Operation.create(
            "scf.if", results=list(result_types), operands=[pred_.ir_value()], regions=2
        )
    except Exception as e:
        raise DSLRuntimeError(
            f"Failed to create dynamic if \n\t\tpred={pred_}: type : {type(pred_)}"
        ) from e
    for block, region in zip(blocks, if_op.regions):
        block.append_to(region)
    return if_op


def _attr_const_check(attr: object, expected_type: type, attr_name: str) -> None:
    """Require the loop option ``attr_name`` to be a Python ``expected_type``.

    A staged value raises ``PHASE_REQUIRES_CONSTANT``; the exact type check
    keeps a ``bool`` out of an ``int`` option and vice versa.
    """
    if is_mlir_op(attr):
        raise DSLUserCodeError(DiagId.PHASE_REQUIRES_CONSTANT, what=f"`{attr_name}`")
    if type(attr) is not expected_type:
        raise DSLUserCodeError(
            f"The loop option `{attr_name}` must be a Python `{expected_type.__name__}`, "
            f"but got `{attr!r}` of type `{type(attr).__name__}`.",
            suggestion=f"Pass a compile-time `{expected_type.__name__}` for `{attr_name}`.",
        )


class ScfGenerator:
    """
    Encapsulates common scf dialect functionality: pack, unpack, and the join checks.

    One instance holds the write_args of a staged region split into the slots
    carried through SSA and the pass-through slots (reference leaves that are
    only store bases, ``mutated_names``). ``unpack`` flattens the carried slots
    at region entry, ``pack`` restores them from block arguments or results and
    ``yield_values`` flattens a region result and checks it against the entry.
    """

    def __init__(
        self,
        op_type_name: str,
        mix_iter_args: Sequence[Any],
        mix_iter_arg_names: Sequence[str],
        mutated_names: Sequence[str] = (),
    ) -> None:
        self.op_type_name = op_type_name
        self.mix_iter_args = list(mix_iter_args)
        self.names = [
            mix_iter_arg_names[i] if i < len(mix_iter_arg_names) else f"arg{i}"
            for i in range(len(self.mix_iter_args))
        ]
        # A store-base-only reference leaf is a memory side effect: no carry.
        self.carried = [
            i
            for i, (arg, name) in enumerate(zip(self.mix_iter_args, self.names))
            if not (name in mutated_names and is_reference_leaf(arg))
        ]
        self.ir_values: list[ir.Value] = []
        self.pytree_def: Optional[PyTreeDef] = None

    @staticmethod
    def _normalize_region_result_to_list(region_result: Any) -> list[Any]:
        """None -> [], a list as is, anything else -> a one-element list."""
        if region_result is None:
            return []
        if not isinstance(region_result, list):
            return [region_result]
        return region_result

    def _carried_args(self, args: Sequence[Any]) -> list[Any]:
        """Select the carried slots of a full write_args list."""
        if len(args) != len(self.mix_iter_args):
            raise DSLRuntimeError(
                f"the `{self.op_type_name}` region returned {len(args)} values for "
                f"{len(self.mix_iter_args)} write_args"
            )
        return [args[i] for i in self.carried]

    def _mutate_python(self, var: str) -> DSLUserCodeError:
        """The ``PHASE_MUTATE_PYTHON`` error for ``var`` in this region."""
        return DSLUserCodeError(
            DiagId.PHASE_MUTATE_PYTHON,
            var=var,
            detail=f" (this `{self.op_type_name}`)",
            context={"region": self.op_type_name},
        )

    def unpack(self, mutated_names: Sequence[str] = ()) -> list[ir.Value]:
        """Flatten the carried write_args at region entry; apply the entry mutation rule."""
        for i in self.carried:
            arg, name = self.mix_iter_args[i], self.names[i]
            # Flattened once by name, so a container error names the variable.
            _, _, treedef = tree_flatten(arg, return_ir_values=False, root=name)
            if name in mutated_names and arg is not None:
                if isinstance(treedef, Leaf) and treedef.is_meta:
                    raise self._mutate_python(name)
        self.ir_values, _, self.pytree_def = tree_flatten(
            self._carried_args(self.mix_iter_args)
        )
        log().debug("scf.%s carries %d values", self.op_type_name, len(self.ir_values))
        return self.ir_values

    @property
    def ir_types(self) -> list[ir.Type]:
        """The SSA types of the carried values, after ``unpack``."""
        return [v.type for v in self.ir_values]

    def _entry_def(self) -> PyTreeDef:
        """The tree of the carried slots at region entry (set by ``unpack``)."""
        if self.pytree_def is None:
            raise DSLRuntimeError("ScfGenerator: unpack must run before pack/yield")
        return self.pytree_def

    def pack(
        self, ir_values: Sequence[ir.Value], treedef: Optional[PyTreeDef] = None
    ) -> list[Any]:
        """Rebuild the full write_args list from block arguments or op results.

        ``treedef`` defaults to the entry tree; an ``if`` restores its results
        from the arms' tree, where a ``None``-seeded slot has become a leaf.
        """
        treedef = self._entry_def() if treedef is None else treedef
        restored = iter(tree_unflatten(treedef, list(ir_values)))
        return [
            next(restored) if i in self.carried else arg
            for i, arg in enumerate(self.mix_iter_args)
        ]

    def results(
        self, ir_values: Sequence[ir.Value], treedef: Optional[PyTreeDef] = None
    ) -> Any:
        """``pack`` the op results in the write-back shape: None, one value, or a list."""
        final_results = self.pack(ir_values, treedef)
        if not final_results:
            return None
        if len(final_results) == 1:
            return final_results[0]
        return final_results

    def flatten_result(self, region_result: Any) -> tuple[list[ir.Value], PyTreeDef]:
        """Flatten a region result under the current insertion point (constants land there)."""
        values, _, treedef = tree_flatten(
            self._carried_args(self._normalize_region_result_to_list(region_result))
        )
        return values, treedef  # type: ignore[return-value]

    def _check_leaf(
        self, var: str, old: Leaf, new: Leaf, *, mutation: bool, detail: str
    ) -> None:
        """Validate one slot of a join: ``old`` entered (or came from the other arm)."""
        if old.is_none and new.is_none:
            return
        if old.is_meta and new.is_meta and _meta_equal(old.meta, new.meta):
            return
        if old.is_meta and new.is_meta and not mutation:
            return  # two arms disagree on a Python value: reported by structure
        if mutation and old.is_meta and not new.is_none:
            raise self._mutate_python(var)
        if (
            old.is_none
            or new.is_none
            or old.is_meta
            or new.is_meta
            or _prototype_differs(old, new)
        ):
            raise DSLUserCodeError(
                DiagId.TYPE_UNSTABLE_JOIN,
                var=var,
                old_type=_leaf_type_name(old),
                new_type=_leaf_type_name(new),
                detail=detail,
            )

    def _check_region_result(
        self,
        old_def: PyTreeDef,
        new_def: PyTreeDef,
        *,
        mutation: bool,
        detail: str,
        skip_none: bool = False,
    ) -> None:
        """
        Validate that a region result maintains the type and structure of the original value.

        Leaf pairs first (``PHASE_MUTATE_PYTHON`` for a Python value that
        changed, ``TYPE_UNSTABLE_JOIN`` for a type change), then the container
        structure (``CONTAINER_STRUCTURE_CHANGED`` with the first difference).
        """
        for index, (old, new) in enumerate(
            zip(old_def.child_treedefs, new_def.child_treedefs)
        ):
            name = self.names[self.carried[index]]
            if skip_none and isinstance(old, Leaf) and old.is_none:
                continue  # a None-seeded slot: the arms decide its type
            old_leaves, new_leaves = tree_leaves(old), tree_leaves(new)
            if len(old_leaves) == len(new_leaves):
                for (path, old_leaf), (_, new_leaf) in zip(old_leaves, new_leaves):
                    self._check_leaf(
                        name + path,
                        old_leaf,
                        new_leaf,
                        mutation=mutation,
                        detail=detail,
                    )
            difference = describe_tree_difference(old, new, name)
            if difference:
                raise DSLUserCodeError(
                    DiagId.CONTAINER_STRUCTURE_CHANGED,
                    var=name,
                    op_type=self.op_type_name,
                    detail=f" ({difference})",
                )

    def yield_values(self, region_result: Any) -> list[ir.Value]:
        """Flatten a region result and check it against the entry state."""
        values, treedef = self.flatten_result(region_result)
        self._check_region_result(
            self._entry_def(),
            treedef,
            mutation=True,
            detail=f" (in this `{self.op_type_name}`)",
        )
        return values

    def check_arms(self, then_def: PyTreeDef, else_def: PyTreeDef) -> None:
        """Check the two arms of an ``if`` against the entry and against each other."""
        for arm, treedef in (("then", then_def), ("else", else_def)):
            self._check_region_result(
                self._entry_def(),
                treedef,
                mutation=True,
                detail=f" (in the `{arm}` arm of this `{self.op_type_name}`)",
                skip_none=True,
            )
        self._check_region_result(
            then_def,
            else_def,
            mutation=False,
            detail=f" (between the `then` and `else` arms of this `{self.op_type_name}`)",
        )


# =============================================================================
# Executors
# =============================================================================


def _loop_execute_range_dynamic(
    func: Callable[..., Any],
    start: Any,
    stop: Any,
    step: Any,
    *,
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: Sequence[str] = (),
    unroll: int = -1,
    unroll_full: bool = False,
    **options: Any,
) -> Any:
    """Build an ``scf.for`` over integer bounds, with an optional unroll annotation.

    :param func: The loop body, ``func(iv, *write_args) -> write_args``
    :param start: The bounds; ``as_numeric``-ed, integer, promoted to one dtype
        that is also the induction variable's (no ``index``)
    :param write_args: The region's write_args, seeded from the enclosing scope
    :param full_write_args_count: The count of leading plain-store write_args
        (the rest are method-call receivers); part of the protocol, every
        write_arg is threaded alike
    :param write_args_names: The write_args' names, for the diagnostics
    :param mutated_names: The store-base-only write_args (the mutation rule)
    :param unroll: The unroll count of the ``llvm.loop_annotation``; -1 for none
    :param unroll_full: Ask for full unrolling instead of a count
    :param options: Any other keyword is rejected with ``CALL_ARGUMENTS``
    :return: The write-back value: ``None``, one value, or a list of them
    """
    if options:
        raise DSLUserCodeError(
            DiagId.CALL_ARGUMENTS,
            function_name="range",
            detail=f"no loop option is named `{next(iter(options))}`",
        )
    start_, stop_, step_, dtype = _promote_loop_bounds(start, stop, step)

    # The loop options become attributes, so they must be Python values.
    _attr_const_check(unroll, int, "unroll")
    _attr_const_check(unroll_full, bool, "unroll_full")
    unroll_attr = None
    if unroll_full:
        unroll_attr = LoopUnroll(full=True)
    elif unroll != -1:
        unroll_attr = LoopUnroll(count=unroll)

    scf_gen = ScfGenerator("for", write_args, write_args_names, mutated_names)
    dyn_yield_ops = scf_gen.unpack(mutated_names)
    log().debug("Creating scf.ForOp start=%s stop=%s step=%s", start_, stop_, step_)
    try:
        for_op = scf.ForOp(
            start_.ir_value(), stop_.ir_value(), step_.ir_value(), dyn_yield_ops
        )
    except Exception as e:
        yield_ops = "\n".join(
            f"\t\t{i} => {d} : type : {type(d)}" for i, d in enumerate(dyn_yield_ops)
        )
        raise DSLRuntimeError(
            f"Failed to create dynamic for loop \n\t\tstart={start_}: type : {type(start_)}"
            f"\n\t\tstop={stop_}: type : {type(stop_)}\n\t\tstep={step_}: type : {type(step_)}"
            f", \n\tdyn_yield_ops:\n{yield_ops}"
        ) from e
    if unroll_attr is not None:
        for_op.attributes[_LOOP_ANNOTATION_ATTR] = unroll_attr

    with ir.InsertionPoint(for_op.body):
        # The induction variable takes the promoted dtype; no cast.
        iv = dtype(for_op.induction_variable)
        func_args = scf_gen.pack(for_op.inner_iter_args)
        log().debug("For body builder: %s func_args: %s", iv, func_args)
        scf.YieldOp(scf_gen.yield_values(func(iv, *func_args)))

    log().debug("Completed scf.for \n[%s]", for_op)
    return scf_gen.results(for_op.results)


def _if_execute_dynamic(
    pred: Any,
    then_block: Callable[..., Any],
    else_block: Optional[Callable[..., Any]] = None,
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: Sequence[str] = (),
    branch_weights: Optional[Sequence[int]] = None,
) -> Any:
    """Run one arm in Python for a Python predicate, else build an ``scf.if``.

    Both arms are traced into detached holder blocks first, so the result types
    follow from what the arms yield (a ``None``-seeded slot is legal when both
    arms yield a leaf); the blocks are then moved into the ``scf.if``. An
    ``if`` without results and with an empty else arm gets no else region.

    :param pred: The condition; a Python value selects the arm at trace time
    :param then_block: The then arm, ``then_block(*write_args) -> write_args``
    :param else_block: The else arm; ``None`` passes the carries through
    :param write_args: The region's write_args
    :param full_write_args_count: The count of leading plain-store write_args;
        part of the protocol, every write_arg is threaded alike
    :param write_args_names: The write_args' names, for the diagnostics
    :param mutated_names: The store-base-only write_args (the mutation rule)
    :param branch_weights: Accepted for the protocol; the base attaches none
    :return: The write-back value: ``None``, one value, or a list of them
    """
    del branch_weights  # the base carries no branch-weight attribute
    if not is_mlir_op(pred):
        if pred:
            region_result = then_block(*write_args)
        elif else_block is not None:
            region_result = else_block(*write_args)
        else:
            region_result = list(write_args)
        results = ScfGenerator._normalize_region_result_to_list(region_result)
        if not results:
            return None
        return results[0] if len(results) == 1 else results

    scf_gen = ScfGenerator("if", write_args, write_args_names, mutated_names)
    dyn_yield_ops = scf_gen.unpack(mutated_names)
    if else_block is None:
        # Pass-through else: the carries leave unchanged.
        else_block = lambda *args: list(args)  # noqa: E731

    holders: list[ir.Operation] = []
    blocks: list[ir.Block] = []
    outputs: list[tuple[list[ir.Value], PyTreeDef]] = []
    try:
        for builder in (then_block, else_block):
            holder, block = _detached_block()
            holders.append(holder)
            blocks.append(block)
            with ir.InsertionPoint(block):
                outputs.append(
                    scf_gen.flatten_result(builder(*scf_gen.pack(dyn_yield_ops)))
                )
        (then_values, then_def), (else_values, else_def) = outputs
        scf_gen.check_arms(then_def, else_def)
        for block, values in zip(blocks, (then_values, else_values)):
            with ir.InsertionPoint(block):
                scf.YieldOp(values)
        result_types = [v.type for v in then_values]
        if not result_types and len(blocks[1].operations) == 1:
            blocks.pop()  # an empty else: no else region
        if_op = _create_if_op(pred, result_types, blocks)
    finally:
        for holder in holders:
            holder.erase()

    log().debug("Completed scf.if \n[%s]", if_op)
    return scf_gen.results(if_op.results, then_def)


def _while_execute_dynamic(
    while_before_block: Callable[..., Any],
    while_after_block: Optional[Callable[..., Any]] = None,
    write_args: Sequence[Any] = (),
    full_write_args_count: int = 0,
    write_args_names: Sequence[str] = (),
    mutated_names: Sequence[str] = (),
) -> Any:
    """Run a Python-bool ``while`` in Python, else build an ``scf.while``.

    The condition is probed once in a detached holder block: a Python-bool
    condition runs the loop in Python (the holder is erased; if the probe
    built IR the condition is evaluated again under the live insertion
    point), a runtime one builds ``scf.while``. The condition's Python side
    effects may therefore run twice.

    :param while_before_block: ``before(*write_args) -> (condition, write_args)``
    :param while_after_block: The body, ``after(*write_args) -> write_args``
    :param write_args: The region's write_args
    :param full_write_args_count: The count of leading plain-store write_args;
        part of the protocol, every write_arg is threaded alike
    :param write_args_names: The write_args' names, for the diagnostics
    :param mutated_names: The store-base-only write_args (the mutation rule)
    :return: The write-back value: ``None``, one value, or a list of them
    """
    if while_after_block is None:
        raise DSLRuntimeError("_while_execute_dynamic needs a while_after_block")
    args = list(write_args)

    holder, block = _detached_block()
    try:
        with ir.InsertionPoint(block):
            cond, before_results = while_before_block(*args)
        staged = is_mlir_op(cond)
        probe_is_pure = len(block.operations) == 0
    finally:
        holder.erase()

    if not staged:
        if not probe_is_pure:
            # The probe built IR the Python loop cannot keep: evaluate it again.
            cond, before_results = while_before_block(*args)
        state = ScfGenerator._normalize_region_result_to_list(before_results)
        while cond:
            state = ScfGenerator._normalize_region_result_to_list(
                while_after_block(*state)
            )
            cond, before_results = while_before_block(*state)
            if is_mlir_op(cond):
                raise DSLUserCodeError(
                    DiagId.PHASE_DYNAMIC_TO_STATIC_BOOL,
                    context="the `while` condition was a Python value on the first "
                    "evaluation and a runtime value on a later one",
                )
            state = ScfGenerator._normalize_region_result_to_list(before_results)
        if not state:
            return None
        return state[0] if len(state) == 1 else state

    scf_gen = ScfGenerator("while", args, write_args_names, mutated_names)
    dyn_yield_ops = scf_gen.unpack(mutated_names)
    result_types = scf_gen.ir_types
    try:
        while_op = scf.WhileOp(result_types, dyn_yield_ops)
        before = while_op.before.blocks.append(*result_types)
        after = while_op.after.blocks.append(*result_types)
    except Exception as e:
        yield_ops = "\n".join(
            f"\t\t{i} => {d} : type : {type(d)}" for i, d in enumerate(dyn_yield_ops)
        )
        raise DSLRuntimeError(
            f"Failed to create dynamic while loop with yield_ops:\n{yield_ops}"
        ) from e

    with ir.InsertionPoint(before):
        # Build the before (condition) block
        flat_args = scf_gen.pack(before.arguments)
        cond, before_results = while_before_block(*flat_args)
        ir_results_list = scf_gen.yield_values(before_results)
        scf.ConditionOp(_condition(cond, "while condition").ir_value(), ir_results_list)

    with ir.InsertionPoint(after):
        # Build the after (body) block
        flat_args = scf_gen.pack(after.arguments)
        scf.YieldOp(scf_gen.yield_values(while_after_block(*flat_args)))

    log().debug("Completed scf.while \n[%s]", while_op)
    return scf_gen.results(while_op.results)


def _ifexp_execute_dynamic(
    pred: Any,
    block_args: tuple,
    then_block: Callable[..., Any],
    else_block: Callable[..., Any],
    branch_weights: Optional[Sequence[int]] = None,
) -> Any:
    """Build an ``scf.if`` for ``x if pred else y`` on a runtime ``pred``.

    Each arm is traced exactly once into its own detached block; the ``scf.if``
    is created from the inferred result types and the blocks are moved into
    it. Both arms must produce the same tree structure and types
    (``TYPE_CONDITIONAL_BRANCH_MISMATCH`` otherwise).

    :param pred: The runtime condition (a Python one is folded by the caller)
    :param block_args: Positional arguments of both arms
    :param then_block: The then arm, ``then_block(*block_args) -> value``
    :param else_block: The else arm, ``else_block(*block_args) -> value``
    :param branch_weights: Accepted for the protocol; the base attaches none
    :return: The expression's value, restored in the arms' shape
    """
    del branch_weights  # the base carries no branch-weight attribute
    holders: list[ir.Operation] = []
    blocks: list[ir.Block] = []
    outputs: list[tuple[list[ir.Value], Any]] = []
    try:
        for builder in (then_block, else_block):
            holder, block = _detached_block()
            holders.append(holder)
            blocks.append(block)
            with ir.InsertionPoint(block):
                # One result: the arm's value as is (a list is a value too).
                values, _, treedef = tree_flatten([builder(*block_args)])
                outputs.append((values, treedef))
        (then_values, then_def), (else_values, else_def) = outputs
        try:
            mismatch = check_tree_equal(then_def, else_def)
        except DSLRuntimeError:
            mismatch = 0
        if mismatch != -1:
            raise DSLUserCodeError(DiagId.TYPE_CONDITIONAL_BRANCH_MISMATCH)
        for block, values in zip(blocks, (then_values, else_values)):
            with ir.InsertionPoint(block):
                scf.YieldOp(values)
        if_op = _create_if_op(pred, [v.type for v in then_values], blocks)
    finally:
        for holder in holders:
            holder.erase()

    return tree_unflatten(then_def, list(if_op.results))[0]


# =============================================================================
# Logical Operators
# =============================================================================


def and_(
    *args: Any,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """``and_(a, b, ...)``: Python's ``and`` through ``__dsl_and__`` when an operand is a runtime value."""
    if len(args) == 0:
        raise DSLRuntimeError("and_() requires at least one argument")

    def and_op(lhs: Any, rhs: Any) -> Any:
        if hasattr(lhs, "__dsl_and__"):
            return lhs.__dsl_and__(rhs, loc=loc, ip=ip)
        if is_mlir_op(lhs) or is_mlir_op(rhs):
            return as_numeric(lhs).__dsl_and__(as_numeric(rhs), loc=loc, ip=ip)
        return lhs and rhs

    return functools.reduce(and_op, args[1:], args[0])


def or_(
    *args: Any,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """``or_(a, b, ...)``: Python's ``or`` through ``__dsl_or__`` when an operand is a runtime value."""
    if len(args) == 0:
        raise DSLRuntimeError("or_() requires at least one argument")

    def or_op(lhs: Any, rhs: Any) -> Any:
        if hasattr(lhs, "__dsl_or__"):
            return lhs.__dsl_or__(rhs, loc=loc, ip=ip)
        if is_mlir_op(lhs) or is_mlir_op(rhs):
            return as_numeric(lhs).__dsl_or__(as_numeric(rhs), loc=loc, ip=ip)
        return lhs or rhs

    return functools.reduce(or_op, args[1:], args[0])


def not_(
    lhs: Any,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """``not_(x)``: Python's ``not`` through ``__dsl_not__`` when ``x`` is a runtime value."""
    # A Python bool first: it has no `__dsl_not__` and must not recurse.
    if type(lhs) is bool:
        return lhs ^ True
    if hasattr(lhs, "__dsl_not__"):
        return lhs.__dsl_not__(loc=loc, ip=ip)
    if is_mlir_op(lhs):
        return as_numeric(lhs).__dsl_not__(loc=loc, ip=ip)
    return bool(lhs) ^ True


def all_(iterable: Any) -> Boolean:
    """Logical AND operation for all elements in an iterable (the DSL's ``all``)."""
    bool_iterable = [_condition(i, "all() element") for i in iterable]
    return functools.reduce(
        lambda lhs, rhs: lhs.__dsl_and__(rhs), bool_iterable, Boolean(True)
    )


def any_(iterable: Any) -> Boolean:
    """Logical OR operation for any element in an iterable (the DSL's ``any``)."""
    bool_iterable = [_condition(i, "any() element") for i in iterable]
    return functools.reduce(
        lambda lhs, rhs: lhs.__dsl_or__(rhs), bool_iterable, Boolean(False)
    )


def in_(lhs: Any, rhs: Any) -> Any:
    """``lhs in rhs``: Python when nothing is staged, else ``any_`` over ``==``."""
    if not is_mlir_op(lhs) and not is_mlir_op(rhs):
        return lhs in rhs
    if not isinstance(rhs, Sequence):
        raise DSLUserCodeError(
            DiagId.UNSUP_SYNTAX,
            what="The comparison operator `in`",
            detail=": use one of `==`, `!=`, `<`, `>`, `<=`, `>=`",
        )
    return any_(lhs == r for r in rhs)


# =============================================================================
# Comparison
# =============================================================================


def _compare_dispatch(lhs: Any, rhs: Any, op: str) -> Any:
    """Apply one comparison operator; ``is``/``is not`` are pure Python."""
    if op == "is":
        return lhs is rhs
    elif op == "is not":
        return lhs is not rhs
    elif op == "in":
        return in_(lhs, rhs)
    elif op == "not in":
        return not_(in_(lhs, rhs))
    elif op == "==":
        return lhs == rhs
    elif op == "!=":
        return lhs != rhs
    elif op == "<":
        return lhs < rhs
    elif op == ">":
        return lhs > rhs
    elif op == ">=":
        return lhs >= rhs
    elif op == "<=":
        return lhs <= rhs
    else:
        raise DSLUserCodeError(
            DiagId.UNSUP_SYNTAX,
            what=f"The comparison operator `{op}`",
            detail=": use one of `==`, `!=`, `<`, `>`, `<=`, `>=`",
        )


def _compare_executor(left: Any, comparators: Sequence[Any], ops: Sequence[str]) -> Any:
    """Evaluate ``left ops[0] comparators[0] ops[1] comparators[1] ...``.

    A single comparison is the operator's result; a chain is the ``and_`` of
    its pairwise comparisons, so a staged link makes the chain a ``Boolean``.
    """
    if len(comparators) == 1:
        return _compare_dispatch(left, comparators[0], ops[0])

    result: Any = True
    current = left
    for comparator, op in zip(comparators, ops):
        cmp_result = _compare_dispatch(current, comparator, op)
        result = and_(result, cmp_result)
        current = comparator

    return result
