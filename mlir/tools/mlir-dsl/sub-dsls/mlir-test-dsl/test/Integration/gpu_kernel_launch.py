# RUN: env MLIR_DSL_ARCH=%cuda_chip MLIR_DSL_LIBS=%mlir_cuda_runtime %PYTHON %s | FileCheck %s
# REQUIRES: host-supports-jit, cuda-gpu
# A `@kernel` launched for real: the host entry is JIT'd, its global
# constructor loads the kernel binary (the engine is initialized), and the
# `gpu.launch_func` runs the kernel on the device's memory.
import torch

import mlir.mlir_dsl as m


@m.kernel
def axpy_kernel(
    n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
):
    for i in m.for_(n):
        y[i] = a * x[i] + y[i]


@m.jit
def axpy(n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
    axpy_kernel(n, a, x, y).launch(grid=[1, 1, 1], block=[1, 1, 1])


n = 64
x = torch.arange(n, dtype=torch.float32, device="cuda")
y = torch.full_like(x, 3.0)
axpy(n, 2.0, x, y)
torch.testing.assert_close(y, 2.0 * x + 3.0)
print("LAUNCHED:", int(y[n - 1].item()))
# CHECK: LAUNCHED: 129
axpy(n, 2.0, x, y)  # a second call launches again
print("AGAIN:", int(y[n - 1].item()))
# CHECK: AGAIN: 255
