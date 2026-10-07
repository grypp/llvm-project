# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# The dialect behind the scalar types is the DSL's `type_ops` plugin. `Int32(6) + a`
# emits `arith` under the test DSL's type ops (arith, vector, llvm); a DSL naming its own
# `TypeOpsPlugin` there emits another dialect. The plugin decides both the SSA
# type of a dtype (`mlir_type`) and the op behind each operator. The stand-in
# below models a tile dialect, where a scalar is a rank-0 tensor and the ops
# work on tensors; cuda_tile's rank-0 `!cuda_tile.tile<i32>` plugs in the same way.
from dataclasses import replace

import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith
from mlir.dsl.core.common import active_dsl
from mlir.dsl.plugins.type_ops import (
    arith as dsl_arith,
    llvm as dsl_llvm,
    vector as dsl_vector,
)
from mlir.dsl.plugins.type_ops import TypeOps


class RankZeroTensorTypeOps(TypeOps):
    """Scalars as rank-0 tensors: `Int32` is `tensor<i32>`."""

    name = "rank0_tensor"

    def mlir_type(self, dtype):
        return ir.RankedTensorType.get([], dtype.scalar_mlir_type)

    def scalar_type(self, mlir_type):
        if isinstance(mlir_type, ir.RankedTensorType) and mlir_type.rank == 0:
            return mlir_type.element_type
        return None

    def const(self, value, ty=None, *, signed=None, loc=None, ip=None):
        if isinstance(value, ir.Value):
            return value
        if ty is None:
            ty = type(value)
        if isinstance(value, m.Numeric):
            value = value.value
        if isinstance(value, ir.Value):
            return value
        tensor_type = self.mlir_type(ty)
        scalar = ir.IntegerAttr.get(ty.scalar_mlir_type, int(value))
        attr = ir.DenseElementsAttr.get_splat(tensor_type, scalar)
        return arith.constant(tensor_type, attr, loc=loc, ip=ip)


class TileLikeDSL(m.MlirTestDSL):
    # The DSL replaces one plugin of the record: this one behind the types; the
    # `func.func` entry and the rest stay the test DSL's (the entry takes tensors too).
    plugins = replace(
        m.MlirTestDSL.plugins,
        type_ops=RankZeroTensorTypeOps(
            scalars=dsl_arith, vectors=dsl_vector, memory=dsl_llvm
        ),
    )

    def pipeline(self):  # and the lowering of what it emits
        return ["my-tile-lowering", "reconcile-unrealized-casts"]


@m.jit
def on_arith(a: m.Int32) -> m.Int32:
    return m.Int32(6) + a


@TileLikeDSL.jit
def on_tiles(a: m.Int32) -> m.Int32:
    return m.Int32(6) + a


# The same source under two dialects.
# CHECK-LABEL: func.func @on_arith(
# CHECK-SAME:    %[[A:[^:]+]]: i32) -> i32
# CHECK:         %[[C:.+]] = arith.constant 6 : i32
# CHECK:         arith.addi %[[C]], %[[A]] : i32
on_arith(1)
# CHECK-LABEL: func.func @on_tiles(
# CHECK-SAME:    %[[T:[^:]+]]: tensor<i32>) -> tensor<i32>
# CHECK:         %[[CT:.+]] = arith.constant dense<6> : tensor<i32>
# CHECK:         arith.addi %[[CT]], %[[T]] : tensor<i32>
on_tiles(1)

# A role not replaced keeps the test DSL's plugin; the pipeline is the DSL's own.
base, tile = m.MlirTestDSL(), TileLikeDSL()
print(
    "TYPE_OPS:",
    type(base.plugins.type_ops).__name__,
    type(tile.plugins.type_ops).__name__,
    type(tile.plugins.func_entry).__name__,
)
print("PIPELINE:", tile._get_pipeline(None))
# CHECK: TYPE_OPS: TypeOps RankZeroTensorTypeOps Entry
# CHECK: PIPELINE: builtin.module(my-tile-lowering,reconcile-unrealized-casts)

# The types answer through the active DSL's `type_ops` plugin: `mlir_type` and
# the reverse lookup both go through it, so the same dtype is `i32` under one
# DSL and `tensor<i32>` under the other; `scalar_mlir_type` is the scalar behind
# a dtype under any dialect. Outside any DSL the types have no dialect to ask.
with ir.Context(), ir.Location.unknown():
    with active_dsl(base):
        print("ACTIVE:", m.Int32.mlir_type, m.Float16.mlir_type)
        print("SCALAR:", m.Int32.scalar_mlir_type)
        print(
            "LOOKUP:",
            m.Numeric.from_mlir_type(ir.IntegerType.get_signless(32)).__name__,
        )
    with active_dsl(tile):
        print(
            "TILE:",
            m.Int32.mlir_type,
            m.Numeric.from_mlir_type(m.Int32.mlir_type).__name__,
        )
    t = RankZeroTensorTypeOps(scalars=dsl_arith, vectors=dsl_vector, memory=dsl_llvm)
    print(
        "TILE_LOOKUP:",
        t.scalar_type(t.mlir_type(m.Float32)),
        t.scalar_type(ir.IntegerType.get_signless(32)),
    )
    try:
        m.Int32.mlir_type
        print("NO_DSL: no error")
    except m.DSLUserCodeError as e:
        print("NO_DSL:", e.diag_id.name)
# CHECK: ACTIVE: i32 f16
# CHECK: SCALAR: i32
# CHECK: LOOKUP: Int32
# CHECK: TILE: tensor<i32> Int32
# CHECK: TILE_LOOKUP: f32 None
# CHECK: NO_DSL: CALL_OUTSIDE_JIT


# =============================================================================
# Every part of the composer is optional
# =============================================================================
# A DSL whose type ops name only `scalars` has `Int32` and its operators, but
# no `Vector` and no `Pointer`: the first use of one is CALL_PLUGIN_REQUIRED,
# naming the missing part. The plugin's name lists the parts it has.
class ScalarOnlyDSL(m.MlirTestDSL):
    plugins = replace(
        m.MlirTestDSL.plugins,
        type_ops=TypeOps(scalars=dsl_arith),
        decorators=(),
        adapters=(),
    )


@ScalarOnlyDSL.jit
def scalars_only(a: m.Int32) -> m.Int32:
    return a * 2 + 1


@ScalarOnlyDSL.jit
def wants_vector(a: m.Int32) -> m.Int32:
    v = m.Vector([a, a])
    return v[0]


@ScalarOnlyDSL.jit
def wants_pointer(p: m.Pointer[m.Int32]) -> m.Int32:
    return p[0]


def part_error(fn, *args):
    try:
        fn(*args)
        print(f"{fn.__name__}: NO ERROR")
    except m.DSLUserCodeError as e:
        print(f"{fn.__name__}: {e.diag_id.name}: {e.message}")


# CHECK-LABEL: func.func @scalars_only(
# CHECK:         arith.muli
# CHECK:         arith.addi
scalars_only(3)
print("PARTS:", ScalarOnlyDSL().plugins.type_ops.name)
part_error(wants_vector, 3)
part_error(wants_pointer, np.zeros(4, np.int32))
# CHECK: PARTS: arith
# CHECK: wants_vector: CALL_PLUGIN_REQUIRED: `Vector` needs a `vectors` module in its type ops, which this DSL does not name.
# CHECK: wants_pointer: CALL_PLUGIN_REQUIRED: `Pointer` needs a `memory` module in its type ops, which this DSL does not name.
