# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Flattening of staged values: leaves, tuples, lists and frozen records.

``tree_flatten`` turns a value carried through a staged region or passed at
the ``@jit`` boundary into its SSA values plus a ``PyTreeDef``;
``tree_unflatten`` rebuilds it from new SSA values. The value model is small
on purpose: a leaf is an instance of a registered leaf class (``Numeric``,
``Pointer``, ``Vector``, whatever a sub-DSL adds with ``register_leaf``); a
container is a ``tuple``, a ``list`` or a frozen dataclass (a ``@struct`` is
one), nested as deep as needed; ``None`` is an empty slot; any other object
is a Python value written back verbatim. A dict, set, frozenset,
namedtuple, tuple subclass or non-frozen dataclass holding DSL values is
rejected, not flattened; so is a container that contains itself or that is
nested deeper than Python's recursion limit. Containers come back as fresh objects, leaves are restored from
their prototypes.
"""

import dataclasses
import itertools as it
from collections.abc import Callable, Iterable, Iterator
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional


from ... import ir
from ..core.common import DSLRuntimeError, DSLUserCodeError
from ..core.diagnostics import DiagId
from ..types.typing import Numeric

__all__ = [
    "Leaf",
    "LeafEntry",
    "PyTreeDef",
    "contains_leaf",
    "is_leaf",
    "is_staged_leaf",
    "leaf_entry",
    "leaf_ir_types",
    "tree_flatten",
    "tree_leaves",
]


class LeafEntry(NamedTuple):
    """One registered leaf class and the callables that flatten and restore it.

    The first six fields are the ``register_leaf`` arguments; ``prototype``
    picks what a ``Leaf`` stores for the restore, ``staged`` tells a staged
    payload and ``wrap`` claims a raw ``ir.Value`` of a matching type.
    """

    cls: type
    ir_types: Callable[[Any], list[ir.Type]]
    ir_values: Callable[[Any], list[ir.Value]]
    from_ir_values: Callable[[Any, list[ir.Value]], Any]
    reference: bool
    prototype: Callable[[Any], Any] = lambda v: v
    staged: Optional[Callable[[Any], bool]] = None
    wrap: Optional[Callable[[ir.Value], Any]] = None


_leaf_registry: dict[type, LeafEntry] = {}


def _add_entry(entry: LeafEntry) -> None:
    if not isinstance(entry.cls, type):
        raise DSLRuntimeError(f"register_leaf expects a class, got {entry.cls!r}")
    if entry.cls in _leaf_registry:
        raise DSLRuntimeError(
            f"leaf class `{entry.cls.__qualname__}` is already registered; a leaf "
            "registers once, at import of its module"
        )
    _leaf_registry[entry.cls] = entry


def leaf_entry(cls: type) -> Optional[LeafEntry]:
    """Return the entry serving instances of ``cls`` (MRO walk), or None."""
    if not isinstance(cls, type):
        return None
    for base in cls.__mro__:
        entry = _leaf_registry.get(base)
        if entry is not None:
            return entry
    return None


def is_leaf(obj_or_cls: Any) -> bool:
    """True if ``obj_or_cls`` is a registered leaf class or an instance of one."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return leaf_entry(cls) is not None


def is_staged_leaf(value: Any) -> bool:
    """True for a leaf whose payload is an ``ir.Value``, or a raw ``ir.Value``.

    A ``Numeric`` with a Python payload and a host ``Pointer`` are not staged.
    """
    entry = leaf_entry(type(value))
    if entry is None:
        return isinstance(value, ir.Value)
    if entry.staged is not None:
        return bool(entry.staged(value))
    # No hook: a staged payload is an ``ir.Value`` somewhere in the instance.
    payload = getattr(value, "__dict__", None) or {}
    return any(isinstance(v, ir.Value) for v in payload.values())


def contains_leaf(value: Any) -> bool:
    """True if ``value`` is, or holds inside containers, a leaf or a raw ``ir.Value``.
    The walk is wider than ``tree_flatten`` on purpose: tuples, lists, sets,
    frozensets, dict keys and values and the fields of any dataclass (frozen
    or not), so that a DSL value hidden in a container ``tree_flatten`` would
    treat as a Python value is found and refused. A leaf stops the walk.
    """
    active: set[int] = set()

    def visit(x: Any) -> bool:
        if leaf_entry(type(x)) is not None or isinstance(x, ir.Value):
            return True
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            children: Iterable[Any] = (
                getattr(x, f.name, None) for f in dataclasses.fields(x)
            )
        elif isinstance(x, dict):
            children = it.chain(x.keys(), x.values())
        elif isinstance(x, (tuple, list, set, frozenset)):
            children = x
        else:
            return False
        if id(x) in active:
            return False
        active.add(id(x))
        try:
            for c in children:
                if visit(c):
                    return True
            return False
        finally:
            active.discard(id(x))

    try:
        return visit(value)
    except RecursionError:
        raise DSLUserCodeError(
            DiagId.CONTAINER_TOO_DEEP, var="value", type=type(value).__name__
        ) from None


# =============================================================================
# The tree definition
# =============================================================================


class NodeType(NamedTuple):
    """A container node kind: its name and the to/from iterable functions."""

    name: str
    to_iterable: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    """The structure of a flattened container.

    ``paths`` holds the path of every ``Leaf`` below this node, relative to
    it and in flatten order (``.field``, ``[index]``, ``['key']``).
    """

    node_type: NodeType
    node_metadata: SimpleNamespace
    child_treedefs: tuple["PyTreeDef | Leaf", ...]
    paths: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Leaf:
    """A leaf of a flattened tree.

    ``prototype`` restores the leaf: the dtype class of a ``Numeric``,
    ``(dtype, addrspace)`` of a ``Pointer``, ``(Vector, dtype, lanes)`` of a
    ``Vector``, the entering instance of any other registered leaf; it is None
    for a None slot and for a Python-value slot, whose value ``meta`` is
    written back verbatim.
    """

    is_none: bool = False
    node_metadata: SimpleNamespace | None = None
    ir_type_str: str | None = None
    prototype: Any = None
    meta: Any = None

    @property
    def is_meta(self) -> bool:
        return not self.is_none and self.prototype is None


# =============================================================================
# tree_flatten and tree_unflatten
# =============================================================================


def _require_context(what: str) -> None:
    if ir.Context.current is None:
        raise DSLRuntimeError(f"{what} needs an active MLIR context")


def _require_tree(tree: Any, what: str) -> None:
    if not isinstance(tree, (PyTreeDef, Leaf)):
        raise DSLRuntimeError(f"{what} expects a PyTreeDef or Leaf, got {tree!r}")


def leaf_ir_types(leaf: Leaf) -> list[ir.Type]:
    """Return the SSA types of a flattened leaf from its prototype (needs a context)."""
    _require_tree(leaf, "leaf_ir_types")
    if leaf.is_none or leaf.is_meta:
        return []
    _require_context("leaf_ir_types")
    return list(leaf_entry(leaf.node_metadata.cls).ir_types(leaf.prototype))


def _flatten_leaf(
    x: Any, entry: LeafEntry, return_ir_values: bool
) -> tuple[list[Any], list[Any], Leaf]:
    metadata = SimpleNamespace(
        type_str=f"{type(x).__module__}.{type(x).__qualname__}",
        cls=type(x),
        ir_values=return_ir_values,
        num_values=1,
    )
    leaf = Leaf(
        node_metadata=metadata,
        prototype=entry.prototype(x),
    )
    if return_ir_values:
        _require_context("tree_flatten(return_ir_values=True)")
        values = list(entry.ir_values(x))
        metadata.num_values = len(values)
        ir_type_str = ", ".join(str(v.type) for v in values)
    else:
        values = [x]
        ir_type_str = None
        if ir.Context.current is not None:
            ir_type_str = ", ".join(str(t) for t in leaf_ir_types(leaf))
    leaf = dataclasses.replace(leaf, ir_type_str=ir_type_str)
    if ir.Context.current is None:
        return values, [None] * len(values), leaf
    return values, [ir.DictAttr.get({}) for _ in values], leaf


def tree_flatten(
    x: Any, return_ir_values: bool = True, *, root: str = ""
) -> tuple[list[Any], list[ir.Attribute], PyTreeDef | Leaf]:
    """Flatten ``x`` into its values and a tree definition.

    :param x: A leaf, a tuple, list or frozen record of leaves (nested), None,
        or a Python value
    :param return_ir_values: Return the leaves' ``ir.Value`` s (one per SSA
        value) instead of the leaf objects themselves
    :param root: The name of ``x`` in diagnostics (``acc``), prefixed to the
        paths of its parts (``acc.x``, ``acc[1]``)
    :return: ``(values, attributes, treedef)``; ``attributes`` holds one
        argument attribute dict per value (``None`` outside an MLIR context)
    """
    try:
        values, attrs, treedef = _tree_flatten(x, return_ir_values, root, set())
        return list(values), list(attrs), treedef
    except RecursionError:
        raise DSLUserCodeError(
            DiagId.CONTAINER_TOO_DEEP, var=path or "value", type=type(x).__name__
        ) from None


def _tree_flatten(
    x: Any, return_ir_values: bool, path: str, active: set[int]
) -> tuple[Iterable[Any], Iterable[Any], PyTreeDef | Leaf]:
    """``active`` holds the ids of the containers on the path to ``x``: a
    container that contains itself is a user error, not a stack overflow."""
    if x is None:
        return [], [], Leaf(is_none=True)
    entry = leaf_entry(type(x))
    if entry is not None:
        return _flatten_leaf(x, entry, return_ir_values)
    if isinstance(x, (tuple, list, dict)) or dataclasses.is_dataclass(x):
        raise DSLUserCodeError(
            DiagId.CONTAINER_UNSUPPORTED, var=path or "value", type=type(x).__name__
        )
    return [], [], Leaf(meta=x)


def tree_leaves(treedef: PyTreeDef | Leaf) -> list[tuple[str, Leaf]]:
    """Return ``(path, leaf)`` for every ``Leaf`` of ``treedef`` in flatten order."""
    _require_tree(treedef, "tree_leaves")

    def visit(tree: PyTreeDef | Leaf) -> Iterator[Leaf]:
        if isinstance(tree, Leaf):
            yield tree
        else:
            for child in tree.child_treedefs:
                yield from visit(child)

    if isinstance(treedef, Leaf):
        return [("", treedef)]
    return list(zip(treedef.paths, visit(treedef)))


# =============================================================================
# Built-in leaves
# =============================================================================


_add_entry(
    LeafEntry(
        Numeric,
        ir_types=lambda p: [p.mlir_type],
        ir_values=lambda v: [v.ir_value()],
        from_ir_values=lambda p, vs: p(vs[0]),
        reference=False,
        prototype=lambda v: type(v),
        staged=lambda v: isinstance(v.value, ir.Value),
    )
)
