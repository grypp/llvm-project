# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# REQUIRES: host-supports-jit
# The AST preprocessor plugin: a `BaseDSL` assembled from plugins gets native control
# flow from `ScfASTPreprocessorPlugin` (preprocessor + executors); its `closure_check`
# knob allows captures in staged regions; a DSL that preprocesses without a
# AST preprocessor plugin is a configuration error; a plugin subclass may swap the
# executors.
import mlir.mlir_dsl as m
from mlir import execution_engine, passmanager


def make(name, plugins, preprocess=True):
    class Custom(m.BaseDSL):
        _jit_arg_adapter_scope = "mlir"

        def __init__(self):
            super().__init__(
                name=name,
                dsl_package_name=["mlir", "dsl"],
                compiler_provider=m.Compiler(passmanager, execution_engine),
                pass_sm_arch_name="cubin-chip",
                preprocess=preprocess,
            )

    Custom.plugins = plugins
    Custom.__name__ = name
    return Custom


Lenient = make("MLIR_DSL", [m.ScfASTPreprocessorPlugin(closure_check=False)])


@Lenient.jit
def captured(a: m.Int32, n: m.Int32) -> m.Int32:
    def bump(x):
        return x + a  # a capture inside the staged loop below

    acc = m.Int32(0)
    for i in range(n):
        acc = bump(acc)
    return acc


# CHECK-LABEL: func.func @captured(
# CHECK:         scf.for
# CHECK:           arith.addi
# EXEC:          CAPTURED: 15
print("CAPTURED:", captured(5, 3))
print(
    "PREPROCESSOR:",
    type(Lenient().ast_preprocessor).__name__,
    Lenient().ast_preprocessor.closure_check,
)
# CHECK: PREPROCESSOR: ScfASTPreprocessorPlugin False
# EXEC:  PREPROCESSOR: ScfASTPreprocessorPlugin False


class CpuOnly(m.MlirDSL):
    plugins = []  # drops the optional plugins, not the language's AST preprocessor


# CHECK: DEFAULT PREPROCESSOR: ScfASTPreprocessorPlugin True []
# EXEC:  DEFAULT PREPROCESSOR: ScfASTPreprocessorPlugin True []
cpu = CpuOnly()
print(
    "DEFAULT PREPROCESSOR:",
    type(cpu.ast_preprocessor).__name__,
    cpu.ast_preprocessor.closure_check,
    [p.name for p in cpu.plugins],
)

try:
    make("NO_FRONTEND", [])()
    print("no error (unexpected)")
except m.DSLRuntimeError as e:
    # CHECK: NO PREPROCESSOR: the DSL preprocesses (preprocess=True) but lists no AST preprocessor plugin
    # EXEC:  NO PREPROCESSOR: the DSL preprocesses (preprocess=True) but lists no AST preprocessor plugin
    print("NO PREPROCESSOR:", e.message[:80])

Builders = make("MLIR_DSL", [], preprocess=False)


@Builders.jit
def explicit(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i, acc_in, acc_out in m.for_(0, n, 1, [acc]):
        m.yield_([acc_in + i])
    return acc_out


# Without an AST preprocessor plugin the explicit builders still work.
# CHECK-LABEL: func.func @explicit(
# CHECK:         scf.for
# EXEC:          EXPLICIT: 10
print("EXPLICIT:", explicit(5))


class CountingPreprocessor(m.ScfASTPreprocessorPlugin):
    """A preprocessor plugin subclass replacing one executor: it counts staged loops."""

    name = "counting_preprocessor"
    loops = 0

    def executors(self, dsl):
        executors = super().executors(dsl)
        inner = executors["loop_execute_range_dynamic"]

        def counting(*args, **kwargs):
            CountingPreprocessor.loops += 1
            return inner(*args, **kwargs)

        executors["loop_execute_range_dynamic"] = counting
        return executors


Counting = make("MLIR_DSL", [CountingPreprocessor()])


@Counting.jit
def two_loops(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    for j in range(n):
        acc += j
    return acc


r = two_loops(4)
# CHECK: LOOPS: 2 counting_preprocessor
# EXEC:  LOOPS: 2 counting_preprocessor
print("LOOPS:", CountingPreprocessor.loops, Counting().ast_preprocessor.name)
