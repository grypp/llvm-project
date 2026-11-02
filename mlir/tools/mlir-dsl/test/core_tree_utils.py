# RUN: %PYTHON %s 2>&1 | FileCheck %s
# tree_utils: the leaf registry and the flattening of staged values, with
# `Numeric` as the one built-in leaf. Runs wherever `util/tree_utils.py` is
# present (S3 with tree_utils, or S4). The emitter comes from the S1 context
# `active_dsl` with a stand-in record, as in S3_types.py.
import dataclasses
from collections import namedtuple
from types import SimpleNamespace

from mlir import ir
from mlir.dialects import func
from mlir.extras import types as T

from mlir.dsl.core.common import DSLRuntimeError, DSLUserCodeError, active_dsl
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.core.staging import _is_mlir_op_leaf, is_mlir_op
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith
from mlir.dsl.types.typing import Float32, Int32, Int64, Numeric
from mlir.dsl.util.tree_utils import (
    Leaf,
    LeafEntry,
    PyTreeDef,
    contains_leaf,
    describe_tree_difference,
    is_leaf,
    is_staged_leaf,
    leaf_entry,
    leaf_ir_types,
    register_container,
    register_leaf_protocol,
    registered_container,
    tree_flatten,
    tree_leaves,
    tree_unflatten,
    trees_equal,
)

STAND_IN = SimpleNamespace(
    name="S3",
    plugins=Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith)),
    _unavailable_context=lambda: {},
    is_mlir_op=_is_mlir_op_leaf,
)
text = []


def err(label, fn):
    try:
        fn()
        text.append(f"{label}: NO ERROR")
    except DSLUserCodeError as e:
        text.append(f"{label}: {e.code} | {e.message}")


# -- the registry needs no context
print(
    "REGISTRY:",
    leaf_entry(Int32).cls.__name__,
    leaf_entry(Numeric).reference,
    leaf_entry(int),
    leaf_entry(5),
)
print(
    "IS LEAF:",
    is_leaf(Int32),
    is_leaf(Int64(1)),
    is_leaf(int),
    is_leaf(5),
    is_leaf("s"),
)
print("STAGED:", is_staged_leaf(Int32(5)), is_staged_leaf(5))
print(
    "CONTAINS:",
    contains_leaf([1, (2, Int32(3))]),
    contains_leaf({"k": Float32(1.0)}),
    contains_leaf({"k": 1}),
    contains_leaf((1, 2)),
    contains_leaf(Int32),
)
cyclic = []
cyclic.append(cyclic)
print("CYCLIC:", contains_leaf(cyclic))
deep = []
for _ in range(50000):
    deep = [deep]
err("TOO DEEP", lambda: contains_leaf(deep))
err("TOO DEEP FLATTEN", lambda: tree_flatten(deep, return_ir_values=False, root="acc"))
# CHECK: REGISTRY: Numeric False None None
# CHECK: IS LEAF: True True False False False
# CHECK: STAGED: False False
# CHECK: CONTAINS: True True False False False
# CHECK: CYCLIC: False

# -- flattening without a context: Python values and None, a leaf needs one
values, attrs, leaf = tree_flatten(5, return_ir_values=False)
print(
    "META LEAF:", values, attrs, leaf.is_meta, leaf.is_none, leaf.meta, leaf.prototype
)
values, attrs, leaf = tree_flatten(None)
print("NONE LEAF:", values, attrs, leaf.is_none, leaf.is_meta, tree_leaves(leaf))
values, attrs, leaf = tree_flatten(Int32(5), return_ir_values=False)
print(
    "LEAF NO CONTEXT:",
    type(values[0]).__name__,
    attrs,
    leaf.prototype.__name__,
    leaf.ir_type_str,
    leaf.node_metadata.cls.__name__,
    leaf.node_metadata.num_values,
)
try:
    tree_flatten(Int32(5))
except DSLRuntimeError as e:
    print("IR VALUES NEED CONTEXT:", e.message)
# Containers (tuples, lists, frozen dataclasses) flatten into their leaves.
values, attrs, treedef = tree_flatten((Int32(1), 2), return_ir_values=False, root="acc")
text.append(
    f"CONTAINER: {[type(v).__name__ for v in values]} {attrs} {type(treedef).__name__} "
    f"{[(p, l.prototype.__name__ if l.prototype else None, l.meta) for p, l in tree_leaves(treedef)]}"
)
err("DICT CONTAINER", lambda: tree_flatten({"k": Int32(1)}, return_ir_values=False))
print(
    "PLAIN CONTAINER:", type(tree_flatten((1, 2), return_ir_values=False)[2]).__name__
)
# CHECK: META LEAF: [] [] True False 5 None
# CHECK: NONE LEAF: [] [] True False [('', Leaf(is_none=True, node_metadata=None, ir_type_str=None, prototype=None, meta=None))]
# CHECK: LEAF NO CONTEXT: Int32 [None] Int32 None Int32 1
# CHECK: IR VALUES NEED CONTEXT: tree_flatten(return_ir_values=True) needs an active MLIR context
# CHECK: PLAIN CONTAINER: PyTreeDef


# -- a sub-DSL's value model: leaf protocols and container registrations.
# A class speaking a protocol (`__pair__`) is a leaf through a predicate
# entry, synthesized once per class and cached; the class registry wins over a
# matching predicate (`MyInt` is a `Numeric`). A container registers by class
# (`Bag`) or by predicate (namedtuples), is consulted by `contains_leaf` and
# `is_mlir_op`, before the frozen-record node, and metadata naming a built-in
# kind joins with the built-in.
class Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __pair__(self):
        return [self.x, self.y]


class MyInt(Int32):
    def __pair__(self):
        return [self]


register_leaf_protocol(
    lambda cls: hasattr(cls, "__pair__"),
    ir_types=lambda p: [type(v).mlir_type for v in p.__pair__()],
    ir_values=lambda v: [x.ir_value() for x in v.__pair__()],
    from_ir_values=lambda p, vs: Pt(*(type(x)(v) for x, v in zip(p.__pair__(), vs))),
    staged=lambda v: any(is_staged_leaf(x) for x in v.__pair__()),
)


class Bag:
    def __init__(self, *items):
        self.items = list(items)


register_container(
    Bag,
    to_iterable=lambda bag, path: (SimpleNamespace(), bag.items),
    from_iterable=lambda _, items: Bag(*items),
)
register_container(
    lambda cls: issubclass(cls, tuple) and hasattr(cls, "_fields"),
    name="namedtuple",
    to_iterable=lambda x, path: (
        SimpleNamespace(kind="namedtuple", cls=type(x), fields=list(x._fields)),
        list(x),
    ),
    from_iterable=lambda md, children: md.cls._make(children),
)
Pair = namedtuple("Pair", "a b")
Pair2 = namedtuple("Pair2", "a b")
print(
    "PROTOCOL:",
    leaf_entry(Pt) is leaf_entry(Pt),
    leaf_entry(Pt).cls.__name__,
    leaf_entry(MyInt).cls.__name__,
    is_leaf(Pt(1, 2)),
    leaf_entry(Bag),
)
print(
    "CONTAINERS:",
    registered_container(Bag()).name,
    registered_container(Pair(1, 2)).name,
    registered_container((1, 2)),
    contains_leaf(Bag([Int32(1)])),
    contains_leaf(Bag(1)),
)
values, _, treedef = tree_flatten(
    Bag(1, Int32(2), Pair(Int32(3), 4)), return_ir_values=False
)
rebuilt = tree_unflatten(treedef, values)
print(
    "BAG TREE:",
    treedef.node_metadata.kind,
    [p for p, _ in tree_leaves(treedef)],
    type(rebuilt).__name__,
    type(rebuilt.items[2]).__name__,
    rebuilt.items[2].b,
)
pair_tree = tree_flatten(Pair(Int32(1), 2), return_ir_values=False)[2]
pair2_tree = tree_flatten(Pair2(Int32(1), 2), return_ir_values=False)[2]
print(
    "CLASS RULE:",
    trees_equal(pair_tree, tree_flatten(Pair(Int32(5), 2), return_ir_values=False)[2]),
    trees_equal(pair_tree, pair2_tree),
    describe_tree_difference(pair_tree, pair2_tree, "p"),
)


@dataclasses.dataclass(frozen=True)
class Boxed:
    items: tuple
    __boxed__ = True


register_container(
    lambda cls: getattr(cls, "__boxed__", False),
    name="boxed",
    to_iterable=lambda b, path: (SimpleNamespace(), list(b.items)),
    from_iterable=lambda _, items: Boxed(tuple(items)),
)
boxed_tree = tree_flatten(Boxed((Int32(1), 2)), return_ir_values=False)[2]
print(
    "BEFORE RECORD:",
    boxed_tree.node_type.name,
    boxed_tree.node_metadata.kind,
    boxed_tree.node_metadata.cls.__name__,
    [p for p, _ in tree_leaves(boxed_tree)],
)


class Watched(list):
    pass


register_container(
    Watched,
    to_iterable=lambda w, path: (SimpleNamespace(kind="list", length=len(w)), list(w)),
    from_iterable=lambda _, xs: list(xs),
)
watched_tree = tree_flatten(Watched([Int32(1)]), return_ir_values=False)[2]
print(
    "MIRROR:",
    trees_equal(watched_tree, tree_flatten([Int32(1)], return_ir_values=False)[2]),
    trees_equal(watched_tree, tree_flatten((Int32(1),), return_ir_values=False)[2]),
    type(tree_unflatten(watched_tree, [Int32(1)])).__name__,
)
# CHECK: PROTOCOL: True Pt Numeric True None
# CHECK: CONTAINERS: Bag namedtuple None True False
# CHECK: BAG TREE: Bag ['[0]', '[1]', '[2].a', '[2].b'] Bag Pair 4
# CHECK: CLASS RULE: True False `p` changed from `Pair` with fields ['a', 'b'] to `Pair2` with fields ['a', 'b']
# CHECK: BEFORE RECORD: boxed boxed Boxed ['[0]', '[1]']
# CHECK: MIRROR: True False list


class Marked(ir.Value):
    """A raw SSA value's subclass speaking the protocol: the predicate entry
    serves it before `wrap_ir_value` would claim it as an `Int32`."""

    def __init__(self, value):
        super().__init__(value)

    def __pair__(self):
        return [Int32(self)]


with ir.Context(), ir.Location.unknown(), active_dsl(STAND_IN):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp("tree", ([T.i32(), T.f32()], []))
        block = fop.add_entry_block()
        with ir.InsertionPoint(block):
            a = Int32(block.arguments[0])
            x = Float32(block.arguments[1])
            # CHECK-LABEL: func.func @tree(
            # CHECK-SAME: %[[A:[^:]+]]: i32, %[[X:[^:]+]]: f32)
            values, attrs, leaf = tree_flatten(a)
            text.append(
                f"STAGED LEAF: {values[0] == block.arguments[0]} {attrs} {leaf.ir_type_str} {leaf.prototype.__name__} {is_staged_leaf(a)}"
            )
            text.append(
                f"LEAF TYPES: {leaf_ir_types(leaf)} {leaf_ir_types(Leaf(is_none=True))} {leaf_ir_types(Leaf(meta=3))}"
            )
            text.append(f"LEAVES: {[(p, l.ir_type_str) for p, l in tree_leaves(leaf)]}")
            # A Python payload is materialized through the emitter's const.
            # CHECK: arith.constant 5 : i32
            values, attrs, leaf = tree_flatten(Int32(5))
            text.append(
                f"PYTHON PAYLOAD: {values[0].type} {leaf.ir_type_str} {str(attrs[0])}"
            )
            values, attrs, leaf = tree_flatten(x, return_ir_values=False)
            text.append(
                f"NO IR VALUES: {values[0] is x} {leaf.ir_type_str} {str(attrs[0])}"
            )
            entry = leaf_entry(Int32)
            restored = entry.from_ir_values(Int32, [block.arguments[0]])
            text.append(
                f"RESTORE: {type(restored).__name__} {restored.value == block.arguments[0]} {entry.ir_types(Int64)}"
            )
            _, _, marked = tree_flatten(
                Marked(block.arguments[0]), return_ir_values=False
            )
            text.append(
                f"PREDICATE BEFORE WRAP: {marked.node_metadata.cls.__name__} {marked.ir_type_str} "
                f"{is_mlir_op(Bag(a))} {is_mlir_op(Bag(Int32(1)))} {is_mlir_op(Pt(a, x))}"
            )
            # CHECK: return
            func.ReturnOp([])
    module.operation.verify()
    print(module)

print("\n".join(text))
# CHECK: TOO DEEP: CONTAINER_TOO_DEEP | `value` is nested deeper than Python's recursion limit allows
# With containers flattened by tree_flatten itself, the diagnostic names the
# `root` given to it.
# CHECK: TOO DEEP FLATTEN: CONTAINER_TOO_DEEP | `acc` is nested deeper than Python's recursion limit allows
# CHECK: CONTAINER: ['Int32'] [None] PyTreeDef [('[0]', 'Int32', None), ('[1]', None, 2)]
# CHECK: DICT CONTAINER: CONTAINER_UNSUPPORTED | `value` is a `dict` holding DSL values
# CHECK: STAGED LEAF: True [DictAttr({})] i32 Int32 True
# CHECK: LEAF TYPES: [IntegerType(i32)] [] []
# CHECK: LEAVES: [('', 'i32')]
# CHECK: PYTHON PAYLOAD: i32 i32 {}
# CHECK: NO IR VALUES: True f32 {}
# CHECK: RESTORE: Int32 True [IntegerType(i64)]
# CHECK: PREDICATE BEFORE WRAP: Marked i32 True False True
print("NAMES:", PyTreeDef.__name__, LeafEntry._fields[:4])
# CHECK: NAMES: PyTreeDef ('cls', 'ir_types', 'ir_values', 'from_ir_values')
print("S3_TREE", "OK")
# CHECK: S3_TREE OK
