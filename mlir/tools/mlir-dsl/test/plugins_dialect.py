# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# The dialect behind the scalar types is a plugin. `Int32(6) + a` emits `arith`
# under the core's default LLVM world; a `DialectPlugin` with its own
# `OpEmitter` decides both the SSA type of a dtype (`mlir_type`) and the op
# behind each operator. The stand-in below models a tile dialect, where a
# scalar is a rank-0 tensor and the ops work on tensors; cuda_tile's rank-0
# `!cuda_tile.tile<i32>` plugs in the same way.
import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith


class RankZeroTensorEmitter(m.LlvmEmitter):
    """Scalars as rank-0 tensors: `Int32` is `tensor<i32>`."""

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

# The default dialect is installed only when no listed dialect plugin brings
# an emitter. The dialect passes lead the pipeline, the core's one closes it.
base, tile = m.MlirDSL(), TileLikeDSL()
print("EMITTERS:", type(base.emitter).__name__, type(tile.emitter).__name__)
print("BASE:", [p.name for p in base._dialect_plugins()])
print("TILE:", [p.name for p in tile._dialect_plugins()])
print("PIPELINE:", tile._get_pipeline(None))
# CHECK: EMITTERS: LlvmEmitter RankZeroTensorEmitter
# CHECK: BASE: [{{.*}}'scf', 'llvm']
# CHECK: TILE: ['rank0_tensor']
# CHECK: PIPELINE: builtin.module(my-tile-lowering,reconcile-unrealized-casts)

# Outside any DSL the types answer with the arith emitter, so they work
# standalone; `scalar_mlir_type` is the scalar behind a dtype under any
# dialect, and the reverse lookup goes through the emitter as well.
with ir.Context(), ir.Location.unknown():
    print("STANDALONE:", m.Int32.mlir_type, m.Float16.mlir_type)
    print("SCALAR:", m.Int32.scalar_mlir_type)
    print("LOOKUP:", m.Numeric.from_mlir_type(ir.IntegerType.get_signless(32)).__name__)
    t = RankZeroTensorEmitter()
    print(
        "TILE_LOOKUP:",
        t.scalar_type(t.mlir_type(m.Float32)),
        t.scalar_type(ir.IntegerType.get_signless(32)),
    )
# CHECK: STANDALONE: i32 f16
# CHECK: SCALAR: i32
# CHECK: LOOKUP: Int32
# CHECK: TILE_LOOKUP: f32 None
