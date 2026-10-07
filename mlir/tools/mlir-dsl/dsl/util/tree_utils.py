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

import ctypes

from ... import ir
from ..core.common import DSLRuntimeError, DSLUserCodeError
from ..core.diagnostics import DiagId
from ..core.mlir_op import current_emitter
from ..types.typing import Numeric, Pointer, _lookup_mlir_type
from ..types.vector import Vector

__all__ = [
    "Leaf",
    "LeafEntry",
    "PyTreeDef",
    "check_tree_equal",
    "contains_leaf",
    "describe_tree_difference",
    "is_frozen_dataclass",
    "is_leaf",
    "is_reference_leaf",
    "is_staged_leaf",
    "leaf_entry",
    "leaf_ir_types",
    "register_leaf",
    "tree_flatten",
    "tree_leaves",
    "tree_unflatten",
    "trees_equal",
    "leaf_type_name",
    "wrap_ir_value",
]


def is_frozen_dataclass(obj_or_cls: Any) -> bool:
    """Check if an object or class is a dataclass declared with ``frozen=True``."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    return (
        dataclasses.is_dataclass(cls)
        and getattr(cls, "__dataclass_params__", None) is not None
        and cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
    )


# =============================================================================
# Leaf registry
# =============================================================================


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
    marshal: Optional[Callable[[Any], list[ctypes.c_void_p]]]
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


def register_leaf(
    cls: type,
    *,
    ir_types: Callable[[Any], list[ir.Type]],
    ir_values: Callable[[Any], list[ir.Value]],
    from_ir_values: Callable[[Any, list[ir.Value]], Any],
    marshal: Optional[Callable[[Any], list[ctypes.c_void_p]]] = None,
    reference: bool = False,
    staged: Optional[Callable[[Any], bool]] = None,
    prototype: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Register ``cls`` as a leaf class of the DSL's value model.

    Instances of ``cls`` (and of its subclasses) flatten to the SSA values
    ``ir_values(value)`` and are restored by ``from_ir_values(prototype,
    values)``, where the prototype is the instance that entered the region.
    ``ir_types(prototype)`` gives the SSA types at the ``@jit`` and kernel
    boundaries, where ``marshal(value)`` returns one owning ``c_void_p`` per
    SSA value for a host argument (a leaf without ``marshal`` is staged-only).
    A leaf registers once, at import of its module. ``reference=True`` marks a
    reference leaf whose stores are memory side effects (``Pointer``): the
    executors carry it only when its name is rebound. ``staged(value)`` tells
    a staged payload from a host one (without it, the payload's attributes are
    scanned for an ``ir.Value``); ``prototype(value)`` picks what a ``Leaf``
    stores to restore the value and to compare it at a join (the instance by
    default; a tuple of classes and ints compares structurally).
    """
    entry = LeafEntry(cls, ir_types, ir_values, from_ir_values, marshal, reference)
    if prototype is not None:
        entry = entry._replace(prototype=prototype)
    if staged is not None:
        entry = entry._replace(staged=staged)
    _add_entry(entry)


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


def is_reference_leaf(value: Any) -> bool:
    """True if ``value`` is an instance of a reference leaf class."""
    entry = leaf_entry(type(value))
    return entry is not None and entry.reference


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


def wrap_ir_value(value: ir.Value) -> Any:
    """Wrap a raw ``ir.Value`` in the DSL leaf that claims its type.

    A scalar type of a registered dtype becomes that ``Numeric``; then
    the registered leaves are asked in registration order, through their
    ``wrap`` hook (the built-ins) or through ``ir_types`` when it needs no
    prototype. No claimant raises ``TYPE_UNSUPPORTED_MLIR_TYPE``.
    """
    dt = _lookup_mlir_type(value.type)
    if dt is not None:
        return dt(value)
    for entry in _leaf_registry.values():
        if entry.wrap is not None:
            wrapped = entry.wrap(value)
            if wrapped is not None:
                return wrapped
            continue
        try:
            types = list(entry.ir_types(None))
        except Exception:
            continue
        if types == [value.type]:
            return entry.from_ir_values(None, [value])
    raise DSLUserCodeError(DiagId.TYPE_UNSUPPORTED_MLIR_TYPE, mlir_type=str(value.type))


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


_UNSET = object()


def _record_to_iterable(x: Any, path: str = "") -> tuple[SimpleNamespace, list[Any]]:
    """A frozen record's fields in order. A field without a value is an error;
    an instance attribute that is not a field is kept verbatim on the rebuilt
    record when it holds no DSL value, refused when it does."""
    fields = [f.name for f in dataclasses.fields(x)]
    members = []
    for name in fields:
        value = getattr(x, name, _UNSET)
        if value is _UNSET:
            raise DSLUserCodeError(
                DiagId.CONTAINER_INVALID_RECORD,
                var=_display_path(path),
                type=type(x).__name__,
                detail=f"its field `{name}` has no value",
            )
        members.append(value)
    extras = (
        {k: v for k, v in vars(x).items() if k not in fields}
        if hasattr(x, "__dict__")
        else {}
    )
    for attr, value in extras.items():
        if contains_leaf(value):
            raise DSLUserCodeError(
                DiagId.CONTAINER_INVALID_RECORD,
                var=_display_path(path),
                type=type(x).__name__,
                detail=f"its attribute `{attr}` is not a dataclass field but holds a DSL value",
            )
    metadata = SimpleNamespace(
        kind="dataclass",
        type_str=f"{type(x).__module__}.{type(x).__qualname__}",
        cls=type(x),
        fields=fields,
        extras=extras,
    )
    return metadata, members


def _record_from_iterable(metadata: SimpleNamespace, children: Iterable[Any]) -> Any:
    """Rebuild a frozen record field by field (no ``__init__``/``__post_init__``)."""
    instance = object.__new__(metadata.cls)
    for name, value in zip(metadata.fields, children):
        object.__setattr__(instance, name, value)
    for name, value in getattr(metadata, "extras", {}).items():
        object.__setattr__(instance, name, value)
    return instance


_RECORD_NODE = NodeType("dataclass", _record_to_iterable, _record_from_iterable)
_TUPLE_NODE = NodeType(
    "tuple",
    lambda t: (SimpleNamespace(kind="tuple", length=len(t)), list(t)),
    lambda _, xs: tuple(xs),
)
_LIST_NODE = NodeType(
    "list",
    lambda l: (SimpleNamespace(kind="list", length=len(l)), list(l)),
    lambda _, xs: list(xs),
)


def _container_node_type(x: Any) -> Optional[NodeType]:
    """The node type of a container: a ``tuple``, a ``list`` or a frozen record."""
    if type(x) is tuple:
        return _TUPLE_NODE
    if type(x) is list:
        return _LIST_NODE
    if not isinstance(x, type) and is_frozen_dataclass(x):
        return _RECORD_NODE
    return None


def _child_path(metadata: SimpleNamespace, path: str, index: int) -> str:
    fields = getattr(metadata, "fields", None)
    if fields is not None and index < len(fields):
        return f"{path}.{fields[index]}"
    return f"{path}[{index}]"


def _display_path(path: str) -> str:
    return (path[1:] if path.startswith(".") else path) or "value"


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
            DiagId.CONTAINER_TOO_DEEP, var=root or "value", type=type(x).__name__
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
    if isinstance(x, ir.Value):
        wrapped = wrap_ir_value(x)
        entry = leaf_entry(type(wrapped))
        if entry is None:
            raise DSLRuntimeError(
                f"wrap_ir_value returned an unregistered {type(wrapped).__qualname__}"
            )
        return _flatten_leaf(wrapped, entry, return_ir_values)
    node_type = _container_node_type(x)
    if node_type is None:
        # A Python value, unless it hides DSL values the rebuild would lose.
        if not isinstance(x, type) and contains_leaf(x):
            if dataclasses.is_dataclass(x):
                raise DSLUserCodeError(
                    DiagId.CONTAINER_INVALID_RECORD,
                    var=_display_path(path),
                    type=type(x).__name__,
                    detail="it is a dataclass that is not frozen, so an update made on one path would be lost",
                )
            raise DSLUserCodeError(
                DiagId.CONTAINER_UNSUPPORTED,
                var=_display_path(path),
                type=type(x).__name__,
            )
        return [], [], Leaf(meta=x)
    if id(x) in active:
        raise DSLUserCodeError(
            DiagId.CONTAINER_TOO_DEEP, var=_display_path(path), type=type(x).__name__
        )
    active.add(id(x))
    node_metadata, children = (
        _record_to_iterable(x, path)
        if node_type is _RECORD_NODE
        else node_type.to_iterable(x)
    )
    child_values, child_attrs, child_trees, paths = [], [], [], []
    for i, child in enumerate(children):
        values, attrs, tree = _tree_flatten(
            child, return_ir_values, _child_path(node_metadata, path, i), active
        )
        child_values.append(values)
        child_attrs.append(attrs)
        child_trees.append(tree)
        child_path = _child_path(node_metadata, "", i)
        if isinstance(tree, Leaf):
            paths.append(child_path)
        else:
            paths.extend(child_path + p for p in tree.paths)
    active.discard(id(x))
    return (
        it.chain.from_iterable(child_values),
        it.chain.from_iterable(child_attrs),
        PyTreeDef(node_type, node_metadata, tuple(child_trees), tuple(paths)),
    )


def _value_count(treedef: PyTreeDef | Leaf) -> int:
    """How many entries of ``xs`` ``tree_unflatten`` consumes for ``treedef``."""
    count = 0
    for _, leaf in tree_leaves(treedef):
        if leaf.is_none or leaf.is_meta:
            continue
        count += leaf.node_metadata.num_values if leaf.node_metadata.ir_values else 1
    return count


def tree_unflatten(treedef: PyTreeDef | Leaf, xs: list[Any]) -> Any:
    """Rebuild the value ``treedef`` describes from ``xs``: leaves from their
    prototypes, Python-value slots verbatim, every container a fresh object."""
    _require_tree(treedef, "tree_unflatten")
    xs = list(xs)
    needed = _value_count(treedef)
    if len(xs) != needed:
        raise DSLRuntimeError(
            f"tree_unflatten got {len(xs)} value(s) for a tree of {needed}"
        )
    try:
        return _tree_unflatten(treedef, iter(xs))
    except RecursionError:
        raise DSLUserCodeError(
            DiagId.CONTAINER_TOO_DEEP, var="value", type="tree"
        ) from None


def _tree_unflatten(treedef: PyTreeDef | Leaf, xs: Iterator[Any]) -> Any:
    if isinstance(treedef, Leaf):
        if treedef.is_none:
            return None
        if treedef.is_meta:
            return treedef.meta
        metadata = treedef.node_metadata
        if not metadata.ir_values:
            return next(xs)
        values = [next(xs) for _ in range(metadata.num_values)]
        return leaf_entry(metadata.cls).from_ir_values(treedef.prototype, values)
    if isinstance(treedef, PyTreeDef):
        children = [_tree_unflatten(t, xs) for t in treedef.child_treedefs]
        return treedef.node_type.from_iterable(treedef.node_metadata, children)
    raise DSLRuntimeError(
        f"tree_unflatten expects a PyTreeDef or Leaf, got {treedef!r}"
    )


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
# Structural comparison (the join rule of staged regions)
# =============================================================================


def _meta_equal(lhs: Any, rhs: Any) -> bool:
    """Compare two Python values with ``==``, falling back to identity."""
    if lhs is rhs:
        return True
    try:
        return bool(lhs == rhs)
    except Exception:
        return False


def _prototype_equal(lhs: Any, rhs: Any) -> bool:
    """Compare two leaf prototypes: the type-stability rule of a join."""
    if isinstance(lhs, type) or isinstance(rhs, type):
        return lhs is rhs
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return len(lhs) == len(rhs) and all(
            _prototype_equal(a, b) for a, b in zip(lhs, rhs)
        )
    if type(lhs) is not type(rhs):
        return False
    # Address spaces compare by value; other registered leaves by class, their
    # MLIR types through ``ir_type_str``.
    return _meta_equal(lhs, rhs) if isinstance(lhs, (int, str)) else True


def _node_shape(tree: PyTreeDef) -> tuple:
    metadata = tree.node_metadata
    return (
        tree.node_type,
        getattr(metadata, "kind", None),
        getattr(metadata, "cls", None),
        getattr(metadata, "fields", []),
        len(tree.child_treedefs),
    )


def trees_equal(lhs: PyTreeDef | Leaf, rhs: PyTreeDef | Leaf) -> bool:
    """
    Check if two tree definitions are structurally equal.

    Leaves match when both are None, both Python values that compare equal, or both
    leaves of the same IR types and equal prototypes (same dtype, same
    ``(dtype, addrspace)``, same struct class, same registered class).
    """
    if isinstance(lhs, Leaf) and isinstance(rhs, Leaf):
        if lhs.is_none or rhs.is_none:
            return lhs.is_none == rhs.is_none
        if lhs.is_meta or rhs.is_meta:
            return lhs.is_meta == rhs.is_meta and _meta_equal(lhs.meta, rhs.meta)
        return lhs.ir_type_str == rhs.ir_type_str and _prototype_equal(
            lhs.prototype, rhs.prototype
        )
    if isinstance(lhs, PyTreeDef) and isinstance(rhs, PyTreeDef):
        return _node_shape(lhs) == _node_shape(rhs) and all(
            map(trees_equal, lhs.child_treedefs, rhs.child_treedefs)
        )
    return False


def leaf_type_name(prototype: Any) -> str:
    """Name the value a leaf prototype stands for: ``Int32``, ``Pointer[Float32]``
    (``Pointer[Float32, 3]`` outside space 0), ``Vector[Int32, 4]``, else the
    class name of the prototype."""
    if isinstance(prototype, type):
        return prototype.__name__
    if isinstance(prototype, tuple) and prototype and isinstance(prototype[0], type):
        if len(prototype) == 3:  # (Vector, dtype, lanes)
            return f"{prototype[0].__name__}[{prototype[1].__name__}, {prototype[2]}]"
        if len(prototype) == 2:  # (dtype, space) of a Pointer
            dtype, space = prototype
            return (
                f"Pointer[{dtype.__name__}]"
                if space == 0
                else f"Pointer[{dtype.__name__}, {space}]"
            )
    return type(prototype).__name__


def _tree_value_description(tree: PyTreeDef | Leaf) -> str:
    """Format a tree node or leaf for structure-change diagnostics."""
    if isinstance(tree, Leaf):
        if tree.is_none:
            return "`None`"
        if tree.is_meta:
            try:
                shown = repr(tree.meta)
            except Exception:
                shown = f"<{type(tree.meta).__name__}>"
            return f"a Python value `{shown}`"
        name = leaf_type_name(tree.prototype)
        if tree.ir_type_str:
            return f"`{name}` (type `{tree.ir_type_str}`)"
        return f"a `{name}` value"

    child_count = len(tree.child_treedefs)
    metadata = tree.node_metadata
    if metadata.kind in ("tuple", "list"):
        noun = "item" if child_count == 1 else "items"
        return f"`{metadata.kind}` with {child_count} {noun}"
    type_name = metadata.type_str.rsplit(".", 1)[-1]
    if metadata.fields:
        return f"`{type_name}` with fields {metadata.fields!r}"
    return f"`{type_name}`"


def describe_tree_difference(
    lhs: PyTreeDef | Leaf,
    rhs: PyTreeDef | Leaf,
    path: str,
) -> str:
    """Describe the first deterministic structural difference between two trees.

    The returned fragment is suitable for a diagnostic detail, for example
    ``"`state.pending` changed from `None` to `Int32` (type `i32`)"``. An
    empty string means the trees have the same structure.
    """
    _require_tree(lhs, "describe_tree_difference")
    _require_tree(rhs, "describe_tree_difference")
    if trees_equal(lhs, rhs):
        return ""
    if isinstance(lhs, PyTreeDef) and isinstance(rhs, PyTreeDef):
        if _node_shape(lhs) == _node_shape(rhs):
            for index, (lhs_child, rhs_child) in enumerate(
                zip(lhs.child_treedefs, rhs.child_treedefs)
            ):
                if not trees_equal(lhs_child, rhs_child):
                    child_path = _child_path(lhs.node_metadata, path, index)
                    return describe_tree_difference(lhs_child, rhs_child, child_path)
    return (
        f"`{path}` changed from {_tree_value_description(lhs)} "
        f"to {_tree_value_description(rhs)}"
    )


def check_tree_equal(lhs: PyTreeDef, rhs: PyTreeDef) -> int:
    """
    Check if two tree definitions are equal and return the index of first difference.

    The two trees must have the same number of children (the write_args of
    one region); the result is the index of the first child that differs, or
    -1 if they are completely equal.
    """
    _require_tree(lhs, "check_tree_equal")
    _require_tree(rhs, "check_tree_equal")
    if isinstance(lhs, Leaf) or isinstance(rhs, Leaf):
        return -1 if trees_equal(lhs, rhs) else 0
    if len(lhs.child_treedefs) != len(rhs.child_treedefs):
        raise DSLRuntimeError(
            "check_tree_equal expects trees with the same number of children, "
            f"got {len(lhs.child_treedefs)} and {len(rhs.child_treedefs)}"
        )
    for index, (l, r) in enumerate(zip(lhs.child_treedefs, rhs.child_treedefs)):
        if not trees_equal(l, r):
            return index
    return -1


# =============================================================================
# Built-in leaves
# =============================================================================


_add_entry(
    LeafEntry(
        Numeric,
        ir_types=lambda p: [p.mlir_type],
        ir_values=lambda v: [v.ir_value()],
        from_ir_values=lambda p, vs: p(vs[0]),
        marshal=lambda v: [type(v).marshal(v)],
        reference=False,
        prototype=lambda v: type(v),
        staged=lambda v: isinstance(v.value, ir.Value),
    )
)
_add_entry(
    LeafEntry(
        Pointer,
        ir_types=lambda p: [current_emitter().pointer_type(p[0], p[1])],
        ir_values=lambda v: [v.ir_value()],
        from_ir_values=lambda p, vs: Pointer(vs[0], dtype=p[0], space=p[1]),
        marshal=lambda v: [v.marshal()],
        reference=True,
        prototype=lambda v: (v._dtype, v._addrspace),
        staged=lambda v: v.is_staged,
        wrap=lambda v: (
            Pointer(v) if current_emitter().pointer_space(v.type) is not None else None
        ),
    )
)
_add_entry(
    LeafEntry(
        Vector,
        ir_types=lambda p: [current_emitter().vector_type(p[1], p[2])],
        ir_values=lambda v: [v.ir_value()],
        from_ir_values=lambda p, vs: Vector.from_ir(vs[0], dtype=p[1]),
        marshal=None,
        reference=False,
        # The dtype is part of the join rule: Vector[Int32] and Vector[Uint32]
        # share ``vector<N x i32>`` but are different values.
        prototype=lambda v: (Vector, v.dtype, v.lanes),
        staged=lambda v: True,
        wrap=lambda v: (
            Vector.from_ir(v)
            if current_emitter().vector_shape(v.type) is not None
            else None
        ),
    )
)
