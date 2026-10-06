# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vectors: fixed-width `m.Vector` values, lane-wise arithmetic, reductions.

A `Vector` is a `vector<NxT>` SSA value that exists only inside a trace: it is
built from a lane list or by broadcasting one scalar, combined elementwise with
other vectors, with typed scalars and with Python literals, carried through a
loop like any other staged value, and leaves the function only as scalars (one
lane or a reduction). Run with MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 to see
`vector.from_elements`, `vector.broadcast`, `arith.*` on `vector<4xf32>` and
`vector.reduction` in the traced IR instead of executing it.
"""

import numpy as np

import mlir.mlir_dsl as m


@m.jit
def lanes(x: m.Float32, y: m.Float32) -> tuple:
    # A lane list is one `vector.from_elements`; the typed lanes decide the
    # element dtype, so the literal `1` becomes a Float32 lane as well.
    v = m.Vector([x, y, x + y, 1])
    # `splat` broadcasts one scalar to a compile-time lane count.
    w = m.Vector.splat(y, 4)
    # Elementwise arithmetic needs equal vector types: vector * vector, then
    # + literal (a scalar constant broadcast to the lanes), then * typed scalar (broadcast).
    r = (v * w + 1.0) * x
    # Only scalars leave the trace: `sum()` is `vector.reduction <add>` and
    # `r[2]` is `vector.extract` at a compile-time lane index.
    return r.sum(), r[2]


@m.jit
def sum_in_lanes(chunks: m.Int32, p: m.Pointer[m.Float32]) -> m.Float32:
    # `acc` is the loop carry, so the `scf.for` iterates a `vector<4xf32>`:
    # four partial sums advance together, one horizontal reduction at the end.
    acc = m.Vector.splat(m.Float32(0.0), 4)
    for i in range(chunks):
        acc = acc + (p + i * 4).load(count=4)
    return acc.sum()


def check(label, got, want):
    # Under MLIR_DSL_DRYRUN=1 results are placeholders, so only print them.
    if not m.is_dynamic_expr(got):
        assert abs(float(got) - want) < 1e-6, (label, float(got), want)
    print(f"{label}: {got}")


def main():
    total, third = lanes(2.0, 3.0)
    # v = [2, 3, 5, 1], w = [3, 3, 3, 3]: r = (v * 3 + 1) * 2 = [14, 20, 32, 8]
    check("lanes sum", total, 74.0)
    check("lanes[2]", third, 32.0)

    data = np.arange(12, dtype=np.float32)
    check("sum_in_lanes(3 chunks)", sum_in_lanes(3, data), float(data.sum()))
    check("sum_in_lanes(0 chunks)", sum_in_lanes(0, data), 0.0)
    print("Vectors: passed")


if __name__ == "__main__":
    main()
