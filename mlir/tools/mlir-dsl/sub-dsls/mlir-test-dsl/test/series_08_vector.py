# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
import ctypes

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


# A vector loaded through a pointer has the pointer's dtype: `Uint32` lanes
# divide unsigned and store back through the same pointer.
@m.jit
def unsigned(p: m.Pointer[m.Uint32]) -> m.Float32:
    v = p.load(count=2)
    mask = m.Vector([True, False])
    w = p.masked_load(mask)
    print("LOADED:", v.dtype.__name__, w.dtype.__name__)
    (p + 2).store(v)
    (p + 4).masked_store(w, mask)
    return (v / 2)[0]


# CHECK: LOADED: Uint32 Uint32
# CHECK-LABEL: func.func @unsigned(
# CHECK: llvm.store
# CHECK: llvm.intr.masked.store
# CHECK: arith.uitofp
data = (ctypes.c_uint32 * 6)(4000000000, 2)
result = unsigned(ctypes.addressof(data))
if not m.is_mlir_op(result):
    assert float(result) == 2000000000.0
    print("UNSIGNED:", list(data))
# EXEC: UNSIGNED: [4000000000, 2, 4000000000, 2, 4000000000, 0]
