# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPU kernels: `@m.kernel` bodies launched from a `@m.jit` host over CUDA tensors.

`@m.kernel` is not the core's: it is the decorator of the gpu kernels plugin
(`plugins/decorators/kernels/gpu/`, listed in `MlirTestDSL`'s `decorators`).
A `@m.kernel` function becomes a `gpu.func` inside the module's `gpu.module`.
Calling it from a `@m.jit` host only prepares the launch; `.launch(grid=,
block=)` emits the synchronous `gpu.launch_func`. CUDA tensors adapt to device
pointers without a copy, so the kernel works on the caller's memory. The plugin
only checks `MLIR_DSL_ARCH` (when the DSL is first used and at every launch) and
detects nothing, so this example sets it from the visible device before the
import. MLIR_DSL_DRYRUN=1
MLIR_DSL_PRINT_IR=1 MLIR_DSL_ARCH=sm_90 prints the traced IR on any machine,
GPU or not.
"""

import os

try:
    import torch
except ImportError:  # device memory comes from PyTorch
    torch = None

# The target is read when the DSL instance is created: fix it before the import.
if torch is not None and torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    os.environ.setdefault("MLIR_DSL_ARCH", f"sm_{major}{minor}")

import mlir.mlir_dsl as m

THREADS = 256


@m.kernel
def axpy_kernel(
    n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]
):
    # The gpu index helpers (`plugins/decorators/kernels/gpu/indices.py`) read the NVVM special registers as Int32.
    i = m.block_idx()[0] * m.block_dim()[0] + m.thread_idx()[0]
    if i < n:  # scf.if in the kernel: the last block is only partially filled
        y[i] = a * x[i] + y[i]


@m.jit
def axpy(n: m.Int32, a: m.Float32, x: m.Pointer[m.Float32], y: m.Pointer[m.Float32]):
    # `axpy_kernel(...)` prepares the launch, `.launch` emits gpu.launch_func.
    # The grid depends on the staged `n`, so it is promoted to i64 and guarded:
    # a zero-sized grid skips the launch at run time.
    grid = (n + THREADS - 1) // THREADS
    axpy_kernel(n, a, x, y).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1])


@m.kernel
def relu_kernel(n: m.Int32, x: m.Pointer[m.Float32], out: m.Pointer[m.Float32]):
    i = m.block_idx()[0] * m.block_dim()[0] + m.thread_idx()[0]
    if i < n:  # threads past `n` must not touch memory
        out[i] = m.max(x[i], 0.0)


@m.jit
def relu(n: m.Int32, x: m.Pointer[m.Float32], out: m.Pointer[m.Float32]):
    grid = (n + THREADS - 1) // THREADS
    relu_kernel(n, x, out).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1])


def main():
    if os.environ.get("MLIR_DSL_DRYRUN") and os.environ.get("MLIR_DSL_ARCH"):
        # A dry run only traces: bare addresses stand in for device tensors, so
        # the kernel and launch IR can be inspected without a GPU.
        axpy(1000, 2.0, 0, 0)
        relu(1000, 0, 0)
        print("GPU kernels: passed")
        return
    if torch is None or not torch.cuda.is_available():
        print("GPU kernels: skipped (no CUDA device visible)")
        return

    n = 1000  # not a multiple of THREADS: the last block is partially filled
    x = torch.arange(n, dtype=torch.float32, device="cuda")
    y = torch.full_like(x, 3.0)
    axpy(n, 2.0, x, y)  # CUDA tensors adapt to device pointers, nothing is copied
    torch.testing.assert_close(y, 2.0 * x + 3.0)
    print(f"axpy over {n} elements on {os.environ['MLIR_DSL_ARCH']}: ok")

    # Padding behind the `n` elements proves the bounds check: out-of-range
    # threads leave it untouched. A prefix view is contiguous and adapts as-is.
    x_storage = torch.full((n + THREADS,), -1.0, device="cuda")
    out_storage = torch.full((n + THREADS,), -1234.0, device="cuda")
    x_storage[:n] = (torch.arange(n, device="cuda") % 7 - 3).float()
    relu(n, x_storage[:n], out_storage[:n])
    torch.testing.assert_close(out_storage[:n], torch.relu(x_storage[:n]))
    assert torch.all(out_storage[n:] == -1234.0)
    print("relu with padding: ok")
    print("GPU kernels: passed")


if __name__ == "__main__":
    main()
