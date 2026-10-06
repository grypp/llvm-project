# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The dialect behind the types: `Int32(6) + a` is `arith` by default, and a
`DialectPlugin` with its own `OpEmitter` makes it another dialect.

The scalar types are the core's; which IR they become is the active dialect
plugin's. Its emitter answers `mlir_type(dtype)` (the SSA type of a dtype),
`scalar_type(ir_type)` (the scalar an SSA type carries) and one method per
operator (`const`, `add`, `cmp`, ...). Below a stand-in tile dialect makes
every scalar a rank-0 `tensor`, as cuda_tile's rank-0 `!cuda_tile.tile<i32>`
would. `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1` shows the two IRs.
"""

import os

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith


# (a) An emitter is the dialect's answer to the type system. Reusing
# `LlvmEmitter` keeps `arith` ops, which also work on tensors; a real tile
# dialect overrides every operator with its own ops.
class RankZeroTensorEmitter(m.LlvmEmitter):
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


# (b) The plugin carries the emitter and lowers the world it emits. A DSL
# listing it is not given the default `LlvmDialectPlugin`.
class RankZeroTensorDialect(m.DialectPlugin):
    name = "rank0_tensor"
    emitter = RankZeroTensorEmitter()
    # A dialect with its own emitter gets none of the defaults, so it also
    # names the host entry; the LLVM world's `func.func` takes tensors.
    host_gen_helper = m.LlvmHostGenHelper

    def pipeline_passes(self):
        return ["my-tile-lowering"]


class TileLikeDSL(m.MlirDSL):
    plugins = [RankZeroTensorDialect()]


# (c) The same source under both dialects: `i32` and `arith.addi : i32` under
# `MlirDSL`, `tensor<i32>` and `arith.addi : tensor<i32>` under `TileLikeDSL`.
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

base, tile = m.MlirDSL(), TileLikeDSL()
assert type(base.emitter) is m.LlvmEmitter
assert type(tile.emitter) is RankZeroTensorEmitter
assert [p.name for p in tile._dialect_plugins()] == ["rank0_tensor"]
assert tile._get_pipeline(None) == (
    "builtin.module(my-tile-lowering,reconcile-unrealized-casts)"
)

# (d) Outside any DSL the types answer with `LlvmEmitter`, so they also work
# standalone; `scalar_mlir_type` is the scalar behind a dtype under any dialect.
with ir.Context(), ir.Location.unknown():
    assert str(m.Int32.mlir_type) == "i32" and str(m.Int32.scalar_mlir_type) == "i32"
    emitter = RankZeroTensorEmitter()
    assert str(emitter.mlir_type(m.Float32)) == "tensor<f32>"
    assert str(emitter.scalar_type(emitter.mlir_type(m.Float32))) == "f32"

print("Dialect plugin: passed")
