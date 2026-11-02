# RUN: env EMITC_DSL_DRYRUN=1 EMITC_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | sed -n '/^module/,/^}/p' | mlir-translate --mlir-to-cpp | FileCheck %s
# The traced module of the EmitC test sub-DSL is a complete product: cut out of
# the dry-run print, it goes through `mlir-translate --mlir-to-cpp` and comes
# back as a C function, with no DSL code involved past the trace.
import mlir.emitc_dsl as e


@e.jit
def saxpy(a: e.Float32, x: e.Float32, y: e.Float32) -> e.Float32:
    return a * x + y


saxpy(2.0, 3.0, 4.0)
# CHECK:      float saxpy(float v1, float v2, float v3) {
# CHECK-NEXT:   float v4 = v1 * v2;
# CHECK-NEXT:   float v5 = v4 + v3;
# CHECK-NEXT:   return v5;
# CHECK-NEXT: }
