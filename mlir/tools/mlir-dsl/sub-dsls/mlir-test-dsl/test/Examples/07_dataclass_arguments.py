# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# RUN: %if host-supports-jit %{ %PYTHON %s %}
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dataclass arguments and carries: frozen records with staged and Meta fields.

A frozen dataclass argument is taken apart field by field: a field annotated
with a DSL type is a staged runtime value (a block argument of the function), a
field with a Python type is Meta, folded into the trace as a constant, so every
distinct value is its own compiled specialization. Tuples are
adapted element-wise the same way. Inside a loop, `dataclasses.replace` builds
the next carry of a record. A dataclass that holds DSL values but is not frozen
is rejected.
"""

import os
from dataclasses import dataclass, replace

import mlir.mlir_dsl as m


@dataclass(frozen=True)
class Affine:
    scale: m.Float32  # DSL type: staged, arrives as an f32 block argument
    bias: int  # Python type: Meta, folded into the trace as a constant


@dataclass(frozen=True)
class Stats:
    total: m.Int32
    count: m.Int32


@dataclass(frozen=True)
class Point:
    x: m.Float32
    y: m.Float32


@dataclass
class Mutable:  # not frozen: rejected as soon as it holds a DSL value
    scale: m.Float32


@m.jit
def affine(x: m.Float32, cfg: Affine) -> m.Float32:
    # cfg.scale is an SSA value; cfg.bias is the Python int of this specialization.
    return x * cfg.scale + cfg.bias


@m.jit
def accumulate(n: m.Int32) -> Stats:
    s = Stats(m.Int32(0), m.Int32(0))
    for i in range(n):
        # A frozen record never mutates: `replace` builds the next carry. The
        # scf.for carries the two leaves; `s` is rebuilt as a Stats afterwards.
        s = replace(s, total=s.total + i, count=s.count + 1)
    return s


@m.jit
def norm2(p: Point) -> m.Float32:
    # The record is adapted field by field and reaches the body as a Point.
    return p.x * p.x + p.y * p.y


@m.jit
def broken(cfg: Mutable) -> m.Float32:
    return cfg.scale


def check(label, got, want):
    if os.environ.get("MLIR_DSL_DRYRUN"):
        return  # results are `?` and nothing is compiled under DRYRUN
    assert got == want, (label, got, want)
    print(f"{label}: {got}")


def main():
    dsl = m.MlirTestDSL()

    def compiled():
        # A specialization is compiled, or loaded from the on-disk cache of an
        # earlier process (on by default, see 09): count both.
        return dsl.cache_misses + dsl.file_cache_hits

    check("affine(3, scale=2, bias=1)", affine(3.0, Affine(m.Float32(2.0), 1)), 7.0)
    before = compiled()
    # A new value of the staged field reuses the compiled function ...
    check("affine(4, scale=3, bias=1)", affine(4.0, Affine(m.Float32(3.0), 1)), 13.0)
    check("compilations after a new scale", compiled() - before, 0)
    # ... a new value of the Meta field is another specialization.
    check("affine(3, scale=2, bias=5)", affine(3.0, Affine(m.Float32(2.0), 5)), 11.0)
    check("compilations after a new bias", compiled() - before, 1)

    check("accumulate(5)", accumulate(5), Stats(total=10, count=5))
    check("norm2(3, 4)", norm2(Point(m.Float32(3.0), m.Float32(4.0))), 25.0)

    diag = None
    try:
        broken(Mutable(scale=m.Float32(2.0)))
    except m.DSLUserCodeError as e:
        diag = e.diag_id
    assert diag is m.DiagId.CONTAINER_INVALID_RECORD, diag
    print("non-frozen dataclass:", diag.name)
    print("Dataclass arguments and carries: passed")


if __name__ == "__main__":
    main()
