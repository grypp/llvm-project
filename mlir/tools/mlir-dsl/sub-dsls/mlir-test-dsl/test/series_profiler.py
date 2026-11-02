# RUN: %PYTHON %s %t 2>&1 | FileCheck %s
import inspect
import sys
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
