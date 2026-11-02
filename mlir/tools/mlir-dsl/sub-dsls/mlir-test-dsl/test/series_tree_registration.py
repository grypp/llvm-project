# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# The value-model registration hooks, as the test DSL uses them:
# `register_leaf_protocol` makes every class speaking
# `__ir_types__`/`__ir_values__`/`__from_ir_values__` a leaf (`Complex`, user
# code, never registered by name) and `register_container` makes
# `typing.NamedTuple` a container (`Pair`). Both cross the `@jit` host boundary
# as their SSA values, print in `tree_leaves` and `describe_tree_difference`,
# and a leaf's `__ir_attributes__` lands on the entry's arguments.
from typing import NamedTuple

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.util import tree_utils as tu


class Complex:
    """A user class: two `Float32` attributes and the three protocol members."""

    def __init__(self, re, im):
        self.re = re
        self.im = im

    def __ir_types__(self):
        return [m.Float32.mlir_type, m.Float32.mlir_type]

    def __ir_values__(self):
        return [self.re.ir_value(), self.im.ir_value()]

    @classmethod
    def __from_ir_values__(cls, values):
        return cls(m.Float32(values[0]), m.Float32(values[1]))


class Pair(NamedTuple):
    re: m.Float32
    im: m.Int32


class Pair2(NamedTuple):
    re: m.Float32
    im: m.Int32


# --- the host boundary ------------------------------------------------------
# A protocol leaf is two `f32` arguments, a namedtuple its fields' types; the
# body gets a `Complex` rebuilt by `__from_ir_values__` and a `Pair` rebuilt by
# `_make`, so attribute access works on both.
@m.jit
def f(z: Complex, p: Pair) -> m.Float32:
    return z.re * z.im + p.re


# CHECK-LABEL: func.func @f(
# CHECK-SAME:    %{{[^:]+}}: f32, %{{[^:]+}}: f32, %{{[^:]+}}: f32, %{{[^:]+}}: i32) -> f32
# CHECK:         arith.mulf
# CHECK:         arith.addf
f(Complex(m.Float32(1.0), m.Float32(2.0)), Pair(m.Float32(3.0), m.Int32(4)))


# --- paths and the join rule --------------------------------------------------
# Inside a trace (a context is active): the leaf paths name the namedtuple's
# fields; a changed slot is described by its path; two namedtuple classes with
# the same fields are different containers (the `cls` rule).
@m.jit
def describe(z: Complex, p: Pair) -> m.Float32:
    _, _, tree = tu.tree_flatten((z, p), return_ir_values=False)
    print("LEAVES:", [(path, leaf.ir_type_str) for path, leaf in tu.tree_leaves(tree)])
    _, _, changed = tu.tree_flatten((z, Pair(p.re, None)), return_ir_values=False)
    print("DIFF:", tu.describe_tree_difference(tree, changed, "arg"))
    _, _, other = tu.tree_flatten((z, Pair2(*p)), return_ir_values=False)
    print("CLASS RULE:", tu.trees_equal(tree, other), end=" | ")
    print(tu.describe_tree_difference(tree, other, "arg"))
    return z.re


# CHECK: LEAVES: [('[0]', 'f32, f32'), ('[1].re', 'f32'), ('[1].im', 'i32')]
# CHECK: DIFF: `arg[1].im` changed from `Int32` (type `i32`) to `None`
# CHECK: CLASS RULE: False | `arg[1]` changed from `Pair` with fields ['re', 'im'] to `Pair2` with fields ['re', 'im']
# CHECK-LABEL: func.func @describe(
describe(Complex(m.Float32(1.0), m.Float32(2.0)), Pair(m.Float32(3.0), m.Int32(4)))


# --- per-value argument attributes -------------------------------------------
# A leaf's `__ir_attributes__` gives one dict per SSA value; the host boundary
# puts them on the entry's arguments.
class Lanes(Complex):
    def __ir_attributes__(self):
        i32 = ir.IntegerType.get_signless(32)
        return [
            ir.DictAttr.get({"test.lane": ir.IntegerAttr.get(i32, lane)})
            for lane in range(2)
        ]


@m.jit
def g(z: Lanes) -> m.Float32:
    return z.re + z.im


# CHECK-LABEL: func.func @g(
# CHECK-SAME:    %{{[^:]+}}: f32 {test.lane = 0 : i32}, %{{[^:]+}}: f32 {test.lane = 1 : i32}) -> f32
g(Lanes(m.Float32(1.0), m.Float32(2.0)))

# The protocol entry is resolved once per class and cached; a class without
# the members is no leaf; a plain tuple subclass is still no container.
entry = tu.leaf_entry(Complex)
print(
    "ENTRIES:",
    entry is tu.leaf_entry(Complex),
    entry.cls.__name__,
    tu.leaf_entry(Lanes).cls.__name__,
    tu.is_leaf(Pair),
    tu.registered_container(Pair(1, 2)).name,
    tu.registered_container((1, 2)),
)
# CHECK: ENTRIES: True Complex Lanes False namedtuple None
