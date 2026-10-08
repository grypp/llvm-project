# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics: user mistakes are `m.DSLUserCodeError`s with a stable `m.DiagId`.

Four deliberate mistakes, each caught and identified by `e.diag_id`: the
catalogue member that names the error class, independent of its wording.
`str(e)` is the full rendering: the `error[CODE]` headline, the source location
with a caret under the offending code, the category and `suggestion:` lines;
it is printed once in full. MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 changes
nothing here: all four are caught while tracing, before any IR is lowered.
"""

import mlir.mlir_dsl as m


@m.jit
def wrong_kind(x: m.Float32) -> m.Float32:
    return x * 2.0  # the annotation decides what the call may pass


@m.jit
def early_return(n: m.Int32) -> m.Int32:
    if n > 3:  # a staged condition: an `scf.if` region has no way to `return`
        return n
    return n + 1


@m.jit
def two_types(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = m.Float32(acc) + 1.0  # Int32 enters the loop, Float32 is yielded
    return acc


@m.jit
def capture_in_loop(n: m.Int32) -> m.Int32:
    k = m.Int32(2)

    def inc(x):
        return x + k  # captures `k`: fine outside a staged region ...

    acc = m.Int32(0)
    for i in range(n):
        acc = inc(acc)  # ... not inside an `scf.for` body
    return acc


def expect(diag, fn, *args):
    """Call `fn`, require the DSLUserCodeError `diag`, return it."""
    try:
        fn(*args)
    except m.DSLUserCodeError as e:
        assert e.diag_id is diag, (e.diag_id, diag)
        headline = next(ln for ln in str(e).splitlines() if ln.strip())
        print(f"{e.diag_id.name}: {headline}")
        return e
    raise AssertionError(f"{fn.__name__} did not raise {diag.name}")


def main():
    # A `str` where the annotation says Float32: rejected before tracing.
    expect(m.DiagId.ARG_ANNOTATION_MISMATCH, wrong_kind, "two")
    # The full rendering, once: location, caret span, category, suggestions.
    e = expect(m.DiagId.UNSUP_EARLY_EXIT, early_return, 10)
    print(str(e))
    expect(m.DiagId.TYPE_UNSTABLE_JOIN, two_types, 4)
    expect(m.DiagId.SCOPE_CLOSURE_CAPTURE, capture_in_loop, 4)
    print("Diagnostics: passed")


if __name__ == "__main__":
    main()
