# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Staging: runtime vs Meta values, one trace per Meta specialization.

The parameter annotation decides what a value is. `a: m.Int32` is a staged
runtime value (an IR block argument); the unannotated `scale` is a Meta value,
a plain Python int that is folded into the trace and into the symbol name.
Watch `m.is_dynamic_expr` tell the two apart, a Python `if` on the Meta value
disappear from the IR, and the cache counters show one compiled function per
distinct Meta value. MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 prints the traced IR
(`func.func @scaled_2`, `@scaled_3`) instead of running.
"""

import os

import mlir.mlir_dsl as m

dsl = m.MlirDSL()


def describe(name, value):
    # An ordinary Python helper, called while `scaled` is traced: it sees the
    # staged `a` as an IR value and `scale` as the Python int it always was.
    print(f"  {name}: is_dynamic_expr={m.is_dynamic_expr(value)}")


@m.jit
def scaled(a: m.Int32, scale) -> m.Int32:
    # `a` is staged: every call shares the same IR. `scale` is Meta: its value
    # is known at trace time, so the symbol is `scaled_<scale>` and each new
    # value is a new specialization.
    describe("a", a)
    describe("scale", scale)
    # A Python `if` on a Meta value runs during the trace: only the taken
    # branch is traced, no `scf.if` is emitted.
    if scale == 1:
        result = a  # the IR is a bare `return %a`
    else:
        result = a * scale  # one `arith.muli` by a constant
    return result


def check(label, got, want):
    print(f"{label} = {got}")
    # DRYRUN traces but does not run: results print as `?`, so skip the asserts.
    if not os.environ.get("MLIR_DSL_DRYRUN"):
        assert got == want, (label, got, want)


def compiled():
    # A specialization is compiled, or loaded from the on-disk cache of an
    # earlier process (on by default, see 09): count both.
    return dsl.cache_misses + dsl.file_cache_hits


def main():
    # Outside a trace nothing is staged: a Python int is Meta.
    print(f"host: is_dynamic_expr={m.is_dynamic_expr(5)}")

    # First Meta value: trace and compile (a miss).
    check("scaled(5, 2)", scaled(5, 2), 10)
    # Same Meta value, another runtime value: same trace, nothing compiled (a hit).
    check("scaled(7, 2)", scaled(7, 2), 14)
    check("cache after scale=2 twice", (compiled(), dsl.cache_hits), (1, 1))

    # A new Meta value is a new symbol (`scaled_3`) and a new compile.
    check("scaled(5, 3)", scaled(5, 3), 15)
    check("cache after scale=3", (compiled(), dsl.cache_hits), (2, 1))

    # `scale == 1` folds the multiply away entirely.
    check("scaled(5, 1)", scaled(5, 1), 5)
    check("cache after scale=1", (compiled(), dsl.cache_hits), (3, 1))

    print("Staging: runtime vs Meta values: passed")


if __name__ == "__main__":
    main()
