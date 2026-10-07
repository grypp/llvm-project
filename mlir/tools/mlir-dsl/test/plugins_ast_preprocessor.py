# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# REQUIRES: host-supports-jit
# The AST preprocessor plugin: a `BaseDSL` assembled from a `Plugins` record
# gets native control flow from the `scf.ASTPreprocessor` plugin (preprocessor
# + executors); its `closure_check` knob allows captures in staged regions; a
# DSL that preprocesses without an `ast_preprocessor` plugin is a configuration
# error; a plugin subclass may swap the executors.
from dataclasses import replace

import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor import scf
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.func_entry import func
from mlir.dsl.plugins.type_ops import arith, llvm, vector
from mlir.dsl.plugins.type_ops import TypeOps


def make(name, ast_preprocessor=None, preprocess=True):
    class Custom(m.BaseDSL):
        # The record names the AST preprocessor next to the other roles.
        plugins = m.Plugins(
            type_ops=TypeOps(scalars=arith, vectors=vector, memory=llvm),
            func_entry=func.Entry(),
            ast_preprocessor=ast_preprocessor,
            compiler=execution_engine.Compiler(),
        )

        def pipeline(self):
            return list(m.LOWER_TO_LLVM)

        def __init__(self):
            super().__init__(
                name=name,
                dsl_package_name=["mlir", "mlir_dsl"],
                preprocess=preprocess,
            )

    Custom.__name__ = name
    return Custom


Lenient = make("MLIR_DSL", scf.ASTPreprocessor(closure_check=False))


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
    type(Lenient().plugins.ast_preprocessor).__name__,
    Lenient().plugins.ast_preprocessor.closure_check,
)
# CHECK: PREPROCESSOR: ASTPreprocessor False
# EXEC:  PREPROCESSOR: ASTPreprocessor False


class CpuOnly(m.MlirTestDSL):
    # Drops the decorator and adapter plugins, not the language's
    # AST preprocessor.
    plugins = replace(m.MlirTestDSL.plugins, decorators=(), adapters=())


# CHECK: DEFAULT PREPROCESSOR: ASTPreprocessor True []
# EXEC:  DEFAULT PREPROCESSOR: ASTPreprocessor True []
cpu = CpuOnly()
print(
    "DEFAULT PREPROCESSOR:",
    type(cpu.plugins.ast_preprocessor).__name__,
    cpu.plugins.ast_preprocessor.closure_check,
    [p.name for f in m.Plugins.FAMILIES for p in getattr(cpu.plugins, f)],
)

try:
    make("NO_FRONTEND", None)()
    print("no error (unexpected)")
except m.DSLRuntimeError as e:
    # CHECK: NO PREPROCESSOR: the DSL preprocesses (preprocess=True) but names no `ast_preprocessor`
    # EXEC:  NO PREPROCESSOR: the DSL preprocesses (preprocess=True) but names no `ast_preprocessor`
    print("NO PREPROCESSOR:", e.message[:80])

Builders = make("MLIR_DSL", None, preprocess=False)


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


class CountingPreprocessor(scf.ASTPreprocessor):
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


Counting = make("MLIR_DSL", CountingPreprocessor())


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
print("LOOPS:", CountingPreprocessor.loops, Counting().plugins.ast_preprocessor.name)
