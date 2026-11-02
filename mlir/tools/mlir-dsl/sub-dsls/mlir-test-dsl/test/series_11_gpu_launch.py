# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# GPU construction is target-independent and requires no CUDA device or AST rewrite.
import mlir.mlir_dsl as m


@m.kernel
def increment(p: m.Pointer[m.Int32]):
    p[0] = p[0] + 1


@m.jit
def launch(p: m.Pointer[m.Int32]):
    increment(p).launch(grid=[1, 1, 1], block=[1, 1, 1])


# CHECK: gpu.module
# CHECK: gpu.func @kernel_increment_
# CHECK: llvm.load
# CHECK: arith.addi
# CHECK: llvm.store
# CHECK: gpu.return
# CHECK: func.func @launch
# CHECK: gpu.launch_func
launch(0)
print("GPU: traced without a device")
# CHECK: GPU: traced without a device
