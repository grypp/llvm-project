# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# RUN: %if host-supports-jit %{ %PYTHON %s %}
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPU kernels: `@m.kernel` bodies launched from a `@m.jit` host over CUDA tensors.

`@m.kernel` is not the core's: it is the decorator of the gpu kernels plugin
(`plugins/decorators/kernels/gpu_plugin.py`, listed in `MlirTestDSL`'s `decorators`).
A `@m.kernel` function becomes a `gpu.func` inside the module's `gpu.module`.
Calling it from a `@m.jit` host only prepares the launch; `.launch(grid=,
block=)` emits the synchronous `gpu.launch_func`. A kernel body is the host's
language: loads, stores and loops; the DSL exposes no thread indices (a
downstream kernels plugin brings its own), so these kernels run as one thread
over the whole vector. CUDA tensors adapt to device pointers without a copy, so
the kernel works on the caller's memory. The plugin reads `MLIR_DSL_ARCH`, the
chip MLIR's gpu lowering compiles for, only when a launch is compiled and never
interprets it, so the executing path needs it set for the visible device; a dry
run (MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1) traces the kernel and its launch on
any machine, GPU or not, with no target at all.
"""

import os

try:
    import torch
except ImportError:  # device memory comes from PyTorch
    torch = None

import mlir.mlir_dsl as m


@m.kernel
def axpy_kernel(
    n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
):
    for i in range(n):  # scf.for inside the gpu.func: the host's loop, unchanged
        y[i] = a * x[i] + y[i]


@m.jit
def axpy(n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
    # `axpy_kernel(...)` prepares the launch, `.launch` emits gpu.launch_func;
    # one thread runs the loop.
    axpy_kernel(n, a, x, y).launch(grid=[1, 1, 1], block=[1, 1, 1])


@m.kernel
def relu_kernel(n: m.Int32, x: m.Pointer[m.Float32], out: m.Pointer[m.Float32]):
    for i in range(n):  # the loop stops at `n`: the padding behind it is untouched
        out[i] = m.max(x[i], 0.0)


@m.jit
def relu(n: m.Int32, x: m.Pointer[m.Float32], out: m.Pointer[m.Float32]):
    relu_kernel(n, x, out).launch(grid=[1, 1, 1], block=[1, 1, 1])


def main():
    if os.environ.get("MLIR_DSL_DRYRUN"):
        # A dry run only traces and needs no target: bare addresses stand in for
        # device tensors, so the kernel and launch IR can be inspected anywhere.
        axpy(1000, 2.0, 0, 0)
        relu(1000, 0, 0)
        print("GPU kernels: passed")
        return
    if torch is None or not torch.cuda.is_available():
        print("GPU kernels: skipped (no CUDA device visible)")
        return
    if not os.environ.get("MLIR_DSL_ARCH"):
        print("GPU kernels: skipped (set MLIR_DSL_ARCH to the device's chip name)")
        return

    n = 1000
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.full_like(x, 3.0)
    axpy(n, 2.0, x, y)  # CUDA tensors adapt to device pointers, nothing is copied
    torch.testing.assert_close(y, 2.0 * x + 3.0)
    print(f"axpy over {n} elements on {os.environ['MLIR_DSL_ARCH']}: ok")

    # Padding behind the `n` elements proves the loop bound: the kernel leaves it
    # untouched. A prefix view is contiguous and adapts as-is.
    x_storage = torch.full((n + 256,), -1.0, device="cuda")
    out_storage = torch.full((n + 256,), -1234.0, device="cuda")
    x_storage[:n] = (torch.arange(n, device="cuda") % 7 - 3).float()
    relu(n, x_storage[:n], out_storage[:n])
    torch.testing.assert_close(out_storage[:n], torch.relu(x_storage[:n]))
    assert torch.all(out_storage[n:] == -1234.0)
    print("relu with padding: ok")
    print("GPU kernels: passed")


if __name__ == "__main__":
    main()
