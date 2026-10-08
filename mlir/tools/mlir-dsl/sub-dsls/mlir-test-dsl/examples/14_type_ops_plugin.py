# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The type-ops plugin behind the types: `Int32(6) + a` is `arith` on `i32`
under the test DSL, and a DSL naming its own `type_ops` plugin makes it
another dialect.

The scalar types are the core's; which IR they become is the tracing DSL's
`type_ops` plugin. It answers `mlir_type(dtype)` (the SSA type of a dtype),
`scalar_type(ir_type)` (the scalar an SSA type carries) and one method per
operator (`const`, `add`, `cmp`, ...), routing each to a dialect module
(`mlir.dsl.plugins.type_ops.arith`, `vector`, `llvm`). Below a stand-in tile dialect
makes every scalar a rank-0 `tensor`, as cuda_tile's rank-0
`!cuda_tile.tile<i32>` would, overriding only what differs.
`MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` shows the two IRs.
"""

import os
from dataclasses import replace

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith
from mlir.dsl.plugins.type_ops import (
    arith as dsl_arith,
    llvm as dsl_llvm,
    vector as dsl_vector,
)
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps


# (a) A type-ops plugin is the dialect's answer to the type system (the
# `type_ops` role of the record). Subclassing the `UpstreamDialectTypeOps` composer keeps `arith` ops, which also work on tensors; a real tile
# dialect overrides every operator with its own ops.
class RankZeroTensorTypeOps(UpstreamDialectTypeOps):
    name = "rank0-tensor"

    def mlir_type(self, dtype):
        return ir.RankedTensorType.get([], dtype.scalar_mlir_type)

    def scalar_type(self, mlir_type):
        if isinstance(mlir_type, ir.RankedTensorType) and mlir_type.rank == 0:
            return mlir_type.element_type
        return None

    def const(self, value, ty=None, *, signed=None, loc=None, ip=None):
        if isinstance(value, m.Numeric):
            ty, value = ty or type(value), value.value
        if isinstance(value, ir.Value):
            return value
        tensor_type = self.mlir_type(ty)
        scalar = ir.IntegerAttr.get(ty.scalar_mlir_type, int(value))
        attr = ir.DenseElementsAttr.get_splat(tensor_type, scalar)
        return arith.constant(tensor_type, attr, loc=loc, ip=ip)


# (b) A DSL names the plugin in its `Plugins` record; everything else of
# `MlirTestDSL` stays (its `func.func` entry takes tensors too).
class TileLikeDSL(m.MlirTestDSL):
    plugins = replace(
        m.MlirTestDSL.plugins,
        type_ops=RankZeroTensorTypeOps(
            scalars=dsl_arith, vectors=dsl_vector, memory=dsl_llvm
        ),
    )

    def pipeline(self):  # and the lowering of what it emits
        return ["my-tile-lowering", "reconcile-unrealized-casts"]


# (c) The same source under both dialects: `i32` and `arith.addi : i32` under
# `MlirTestDSL`, `tensor<i32>` and `arith.addi : tensor<i32>` under `TileLikeDSL`.
@m.jit
def on_arith(a: m.Int32) -> m.Int32:
    return m.Int32(6) + a


@TileLikeDSL.jit
def on_tiles(a: m.Int32) -> m.Int32:
    return m.Int32(6) + a


dry = os.environ.get("MLIR_DSL_DRYRUN") == "1"
assert dry or int(on_arith(1)) == 7
if dry:
    on_arith(1)
    # The stand-in has no real lowering, so its program is traced only.
    on_tiles(1)

base, tile = m.MlirTestDSL(), TileLikeDSL()
assert type(base.plugins.type_ops) is UpstreamDialectTypeOps
assert type(tile.plugins.type_ops) is RankZeroTensorTypeOps
assert tile.pipeline() == ["my-tile-lowering", "reconcile-unrealized-casts"]

# (d) Outside a trace there is no DSL and so no type-ops plugin: `Int32.mlir_type`
# is an error there, while a plugin instance answers directly;
# `scalar_mlir_type` is the scalar behind a dtype under any dialect.
with ir.Context(), ir.Location.unknown():
    assert str(m.Int32.scalar_mlir_type) == "i32"
    assert (
        str(
            UpstreamDialectTypeOps(
                scalars=dsl_arith, vectors=dsl_vector, memory=dsl_llvm
            ).mlir_type(m.Int32)
        )
        == "i32"
    )
    tile_ops = RankZeroTensorTypeOps(
        scalars=dsl_arith, vectors=dsl_vector, memory=dsl_llvm
    )
    assert str(tile_ops.mlir_type(m.Float32)) == "tensor<f32>"
    assert str(tile_ops.scalar_type(tile_ops.mlir_type(m.Float32))) == "f32"
    outside = None
    try:
        m.Int32.mlir_type
    except m.DSLUserCodeError as e:
        outside = e
    assert outside is not None, "Int32.mlir_type needs the tracing DSL's type_ops"

print("Type-ops plugin: passed")
