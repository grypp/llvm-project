# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
import mlir.mlir_dsl as m


@m.jit
def lanes(x: m.Float32) -> m.Float32:
    v = m.Vector([x, x + 1.0])
    doubled = v + m.Vector.splat(x, 2)
    return doubled.sum() + doubled[0]


# CHECK-LABEL: func.func @lanes(
# CHECK: vector.from_elements
# CHECK: vector.broadcast
# CHECK: arith.addf
# CHECK: vector.reduction
# CHECK: vector.extract
result = lanes(2.0)
if not m.is_mlir_op(result):
    assert float(result) == 13.0
    print("LANES: 13.0")
# EXEC: LANES: 13.0
