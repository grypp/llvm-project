# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pointers over host buffers: `m.Pointer[T]` arguments over NumPy memory.

A `Pointer[Float32]` parameter is one bare `!llvm.ptr`: a C-contiguous NumPy
array adapts to the address of its data, the length travels as its own
argument and the caller keeps owning the memory. `p[i]` is a load, `p[i] = v`
a store and `p + i` an offset pointer, each one `llvm.getelementptr`. Because
the length is a staged value, one compiled function serves every length.
"""

import os

import numpy as np

import mlir.mlir_dsl as m

F32 = m.Float32


@m.jit
def axpy(n: m.Int32, alpha: F32, x: m.Pointer[F32], y: m.Pointer[F32]):
    # `x` and `y` arrive as `!llvm.ptr` without a length: `n` is the caller's
    # promise about how far the buffers extend.
    for i in range(n):  # scf.for over the staged bound
        y[i] = alpha * x[i] + y[i]  # `x[i]`: gep + load; `y[i] = v`: gep + store


@m.jit
def sum_squares(n: m.Int32, x: m.Pointer[F32]) -> F32:
    acc = F32(0.0)  # rebound in the loop body: the f32 iter_arg of the scf.for
    for i in range(n):
        acc += x[i] * x[i]
    return acc  # the reduction leaves through the return value, not a buffer


@m.jit
def forward_diff(n: m.Int32, x: m.Pointer[F32], out: m.Pointer[F32]):
    nxt = x + 1  # a new Pointer one element further on (one gep); `x` is unchanged
    for i in range(n - 1):
        out[i] = nxt[i] - x[i]  # out[i] = x[i + 1] - x[i]


def check(label, actual, expected):
    # Under MLIR_DSL_DRYRUN=1 nothing runs: results are staged placeholders, the
    # buffers stay untouched and the cache counters stay at zero.
    if not os.environ.get("MLIR_DSL_DRYRUN"):
        np.testing.assert_allclose(np.float32(actual), expected, rtol=1e-6)
    print(f"{label}: ok")


def main():
    dsl = m.MlirTestDSL()
    for n in (0, 1, 17, 257):
        x = np.arange(n, dtype=np.float32)
        y = np.full(n, 3.0, dtype=np.float32)
        axpy(n, 2.0, x, y)  # writes into `y` in place; `x` is only read
        check(f"axpy n={n}", y, 2.0 * x + 3.0)
    # `n` is staged, so every length ran the same compiled function: built once
    # (or loaded from the on-disk cache of an earlier process), then three
    # in-memory hits.
    check("one compiled axpy", dsl.cache_misses + dsl.file_cache_hits, 1)
    check("reused for the other lengths", dsl.cache_hits, 3)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    check("sum of squares", sum_squares(4, x), 30.0)
    # A contiguous slice is just another address: the pointer starts at x[2].
    check("sum of squares over a slice", sum_squares(2, x[2:]), 25.0)

    out = np.zeros(3, dtype=np.float32)
    forward_diff(4, np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32), out)
    check("forward differences", out, [3.0, 5.0, 7.0])
    print("Pointers over host buffers: passed")


if __name__ == "__main__":
    main()
