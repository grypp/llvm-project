# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
import inspect
import sys
import threading
import time
import mlir.mlir_dsl.mlir_dsl as impl
from mlir import ir
from mlir.dsl.util import profiler


# =============================================================================
# profiler: the default, deep and file report modes
# =============================================================================
def profiled_compile():
    profiler.begin_compile()
    profiler.profile_build(lambda: sum(range(1000)))()
    profiler.begin_mlir_phase()
    profiler.end_mlir_phase()
    profiler.finish_compile()


# The reports go to stderr as the phases end: keep stdout in step with them.
sys.stdout.flush()
profiler.configure("T_PROFILE_COMPILER", "1")
profiled_compile()
# The tracing section is emitted as soon as the build phase ends, the mlir
# section at the end of the compile; both name the variable.
# CHECK:      DSL compile profile — tracing  (T_PROFILE_COMPILER)
# CHECK:      build     : {{ *[0-9.]+}} ms   (IR trace + finalize; includes ast-build)
# CHECK-NOT:  deep mode
# CHECK:      DSL compile profile — mlir  (T_PROFILE_COMPILER)
# CHECK:      mlir      : {{ *[0-9.]+}} ms   (pass pipeline)
# CHECK-NOT:  per-pass
profiler.configure("T_PROFILE_COMPILER", "deep")
profiled_compile()
# CHECK:      DSL compile profile — tracing
# CHECK:      (deep mode: cProfile active — times inflated vs default mode)
# CHECK:      TOP FILES by self time
# CHECK:      WHICH FUNCTION is slow in 'build'
# CHECK:      mlir      : {{ *[0-9.]+}} ms   (pass pipeline; single-threaded for timing)
# CHECK:      (no MLIR per-pass report captured — pass pipeline did not run,
report_path = sys.argv[1]
profiler.configure("T_PROFILE_COMPILER", report_path)
profiled_compile()
with open(report_path, encoding="utf-8") as f:
    found = [line.strip() for line in f if "compile profile" in line]
    print("FILE:", found, flush=True)
# CHECK:      [T_PROFILE_COMPILER] compile profile written to {{.*}}
# CHECK:      FILE: ['DSL compile profile — tracing  (T_PROFILE_COMPILER)', 'DSL compile profile — mlir  (T_PROFILE_COMPILER)']
profiler.configure("T_PROFILE_COMPILER", "off")
profiled_compile()
print("OFF:", profiler.enabled(), profiler.deep(), flush=True)
# CHECK-NEXT: OFF: False False

# The deep report attributes frames by file with `region_of`: a file of a
# registered DSL package (the test DSL registers `mlir.mlir_dsl` at import)
# is `dsl: <file>`, the bindings under the `mlir` root are one bucket, Python's
# introspection modules another, and everything else is user / library code.
print(
    profiler.region_of(impl.__file__),
    profiler.region_of(ir.__file__),
    profiler.region_of(inspect.__file__),
    profiler.region_of(__file__),
    sep=" | ",
    flush=True,
)
# CHECK: dsl: mlir_dsl.py | mlir bindings (nanobind) | python introspection (inspect / linecache) | user / library code (outside DSL)


# =============================================================================
# profiler: two threads at one MLIR phase
# =============================================================================
# Claiming and releasing the phase are atomic. The thread `first` is parked
# right after its test of the phase, in the call the profiler makes next (the
# clock read of a release, the construction of the stderr capture of a deep
# claim), until `second` made the same call or is seen to wait for `first`.
# `race` gives the threads that got past the test, the errors raised and
# whether a thread is still running.
def race(call, owner, name):
    real = getattr(owner, name)
    parked, second_done = threading.Event(), threading.Event()
    calls, errors = [], []

    def parking(*args):
        calls.append(threading.current_thread().name)
        if calls == ["first"]:
            parked.set()
            second_done.wait(0.2)
        return real(*args)

    def run(done):
        try:
            call()
        except Exception as e:
            errors.append(type(e).__name__)
        finally:
            done.set()

    threads = [
        threading.Thread(target=run, args=(threading.Event(),), name="first"),
        threading.Thread(target=run, args=(second_done,), name="second"),
    ]
    setattr(owner, name, parking)
    try:
        threads[0].start()
        parked.wait(timeout=60)
        threads[1].start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        setattr(owner, name, real)
    return calls, errors, any(thread.is_alive() for thread in threads)


# Two threads end the open phase: it is released once, neither call fails.
profiler.configure("T_PROFILE_COMPILER", "1")
profiler.begin_compile()
profiler.begin_mlir_phase()
print("RELEASE:", *race(profiler.end_mlir_phase, time, "perf_counter"), flush=True)
profiler.finish_compile()
# CHECK:      RELEASE: ['first'] [] False
# Two threads begin a phase in deep mode: stderr is captured once.
profiler.configure("T_PROFILE_COMPILER", "deep")
profiler.begin_compile()
print(
    "CLAIM:",
    *race(profiler.begin_mlir_phase, profiler, "_capture_mlir_timing"),
    flush=True,
)
profiler.end_mlir_phase()
profiler.finish_compile()
profiler.configure("T_PROFILE_COMPILER", "off")
# CHECK:      CLAIM: ['first'] [] False
