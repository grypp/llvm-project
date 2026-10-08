# RUN: env EMITC_DSL_DRYRUN=1 EMITC_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# The EmitC test sub-DSL: the same saxpy as the reference DSL's, with `+` and `*`
# answered by the EmitC dialect on unchanged types; a loop over a staged bound is
# still an `scf.for` over `llvm` loads and stores, because only the two scalar
# hooks were replaced; the record names no compiler, so a call returns the
# trace result.
import mlir.emitc_dsl as e


@e.jit
def saxpy(a: e.Float32, x: e.Float32, y: e.Float32) -> e.Float32:
    return a * x + y


# CHECK-LABEL: func.func @saxpy(
# CHECK-SAME:    %[[A:.+]]: f32, %[[X:.+]]: f32, %[[Y:.+]]: f32) -> f32
# CHECK:         %[[M:.+]] = emitc.mul %[[A]], %[[X]] : (f32, f32) -> f32
# CHECK:         %[[S:.+]] = emitc.add %[[M]], %[[Y]] : (f32, f32) -> f32
# CHECK:         return %[[S]] : f32
print("SAXPY:", type(saxpy(2.0, 3.0, 4.0)).__name__)
# CHECK: SAXPY: Float32


@e.jit
def saxpy_loop(
    a: e.Float32, x: e.Pointer[e.Float32], y: e.Pointer[e.Float32], n: e.Int32
):
    for i in range(n):
        y[i] = a * x[i] + y[i]


# CHECK-LABEL: func.func @saxpy_loop(
# CHECK:         scf.for
# CHECK:           llvm.load
# CHECK:           emitc.mul
# CHECK:           emitc.add
# CHECK:           llvm.store
host = e.Pointer(0, dtype=e.Float32, kind="host")  # never dereferenced: no compiler
saxpy_loop(2.0, host, host, 8)
dsl = e.EmitCTestDSL()
print("PLUGINS:", [p.name for p in dsl.plugins], dsl.plugins.compiler)
# CHECK: PLUGINS: ['emitc-scalars', 'scf', 'func'] None
