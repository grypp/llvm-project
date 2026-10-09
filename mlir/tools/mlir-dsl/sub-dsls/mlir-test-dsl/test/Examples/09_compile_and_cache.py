# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# RUN: %if host-supports-jit %{ %PYTHON %s %}
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compilation and caches: `m.compile`, the in-memory cache, the on-disk cache.

`m.compile(f, *args)` traces, lowers and JITs `f` for representative arguments
and hands back the compiled function.  Plain calls go through the in-memory
cache, keyed by the traced module (`dsl.cache_hits`, `dsl.cache_misses`).  The
lowered module also lands on disk, in `MLIR_DSL_CACHE_DIR` (by default a
directory under `$TMPDIR`), so a second process skips the pass pipeline: this
file runs itself twice as that process.  `MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1`
prints the traced IR and compiles nothing, so every checked value below is
reported as `?`.
"""

import os
import subprocess
import sys
import tempfile

# Without MLIR_DSL_CACHE_DIR the lowered modules go to a default directory
# under $TMPDIR, where an earlier run already left them; with the file cache
# off, the in-memory counters below are the only cache at work in this
# process.  The DSL reads its MLIR_DSL_* variables on first use: set before import.
os.environ.setdefault("MLIR_DSL_DISABLE_FILE_CACHING", "1")

import mlir.mlir_dsl as m

DRYRUN = bool(os.environ.get("MLIR_DSL_DRYRUN"))


@m.jit
def axpb(x: m.Float32, a: m.Float32, b: int) -> m.Float32:
    # `b` is Meta: every value traces a different module (`axpb_1`, `axpb_2`,
    # ...), hence its own compiled function and its own cache entry.
    return a * x + b


def check(label, compute, expected):
    """Report one checked value; `compute` runs only outside DRYRUN, where
    there is compiled code to call and there are cache counters to read."""
    if DRYRUN:
        print(f"{label}: ?")
        return
    value = compute()
    assert value == expected, f"{label}: {value!r} != {expected!r}"
    print(f"{label}: {value}")


def child():
    # One process per run, same MLIR_DSL_CACHE_DIR.  The first finds the
    # directory empty, compiles and writes mlir_dsl_<module hash>.mlir (MLIR
    # bytecode); the second loads it and skips the pass pipeline, which
    # `dsl.file_cache_hits` counts.
    dsl = m.MlirTestDSL()
    entries = len(os.listdir(dsl.envar.cache_dir))
    result = axpb(2.0, 3.0, 1)
    label = f"  child: axpb(2, 3, 1) = {result}, entries found = {entries}, file hits"
    check(label, lambda: dsl.file_cache_hits, entries)


def main():
    if sys.argv[1:] == ["child"]:
        child()
        return
    dsl = m.MlirTestDSL()

    # m.compile: trace, lower and JIT for these argument types (Meta `b` baked
    # in) without calling.  The result binds a call like Python (the Meta
    # argument is still passed) and returns the raw scalar.  It compiles
    # unconditionally and leaves the in-memory cache alone: the counters see a
    # miss, nothing is kept.
    compiled = m.compile(axpb, 2.0, 3.0, 1)
    check("m.compile returns a", lambda: type(compiled).__name__, "JitCompiledFunction")
    check("compiled(2.0, 3.0, 1)", lambda: compiled(2.0, 3.0, 1), 7.0)

    # The in-memory cache is keyed by the traced module, not by the argument
    # values: other runtime values are a hit, another Meta value is a new
    # module and a miss.  (MLIR_DSL_NO_CACHE=1 turns every cache off.)
    first = axpb(2.0, 3.0, 1)  # miss: traced, lowered and JIT'd
    second = axpb(5.0, 3.0, 1)  # hit: the same module, nothing compiled
    third = axpb(2.0, 3.0, 2)  # miss: `axpb_2` is another module
    results = lambda: [float(v) for v in (first, second, third)]
    check("axpb(2, 3, 1), axpb(5, 3, 1), axpb(2, 3, 2)", results, [7.0, 16.0, 8.0])
    check(
        "in-memory (hits, misses)", lambda: (dsl.cache_hits, dsl.cache_misses), (1, 3)
    )

    # On disk: a process with MLIR_DSL_CACHE_DIR writes the lowered module of
    # every compile as mlir_dsl_<module hash>.mlir and later processes with the
    # same directory skip the pass pipeline.  (MLIR_DSL_DISABLE_FILE_CACHING=1
    # leaves the directory alone; MLIR_DSL_KEEP_IR=1 also saves the traced
    # module there as <function>.mlir, before any pass.)
    cache_dir = tempfile.mkdtemp()
    env = {**os.environ, "MLIR_DSL_CACHE_DIR": cache_dir}
    env["MLIR_DSL_DISABLE_FILE_CACHING"] = "0"
    for run in (1, 2):
        print(f"child run {run} with MLIR_DSL_CACHE_DIR={cache_dir}", flush=True)
        subprocess.run([sys.executable, __file__, "child"], env=env, check=True)
    print(f"on disk: {os.listdir(cache_dir)}")
    check("on-disk entries after two runs", lambda: len(os.listdir(cache_dir)), 1)
    print("Compilation and caches: passed")


if __name__ == "__main__":
    main()
