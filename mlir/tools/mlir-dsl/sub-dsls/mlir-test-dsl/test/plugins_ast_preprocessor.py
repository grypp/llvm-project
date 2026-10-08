# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The AST preprocessor plugin: a `BaseDSL` assembled from a `Plugins` record
# gets native control flow from the `scf.ASTPreprocessor` plugin (preprocessor
# + executors); its `closure_check` knob allows captures in staged regions; a
# DSL without an `ast_preprocessor` plugin never rewrites (native control flow
# over a staged value fails, the explicit builders still work); a plugin
# subclass may swap the executors.
from dataclasses import replace

import mlir.mlir_dsl as m
from mlir.dsl.plugins.ast_preprocessor import scf
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import arith, llvm, vector
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps


def make(name, ast_preprocessor=None):
    class Custom(m.BaseDSL):
        # The record names the AST preprocessor next to the other roles.
        plugins = m.Plugins(
            type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
            ast_preprocessor=ast_preprocessor,
            compiler=execution_engine.Compiler(),
            decorators=[func.Jit()],
        )

        def pipeline(self):
            return list(m.LOWER_TO_LLVM)

        def __init__(self):
            super().__init__(name=name)

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


class JitOnly(m.MlirTestDSL):
    # Drops the gpu kernels plugin and the adapters, not the language's AST
    # preprocessor; `@jit` is the `func.Jit` decorator plugin and stays.
    plugins = replace(m.MlirTestDSL.plugins, decorators=[func.Jit()], adapters=())


# CHECK: DEFAULT PREPROCESSOR: ASTPreprocessor True ['func']
# EXEC:  DEFAULT PREPROCESSOR: ASTPreprocessor True ['func']
jit_only = JitOnly()
print(
    "DEFAULT PREPROCESSOR:",
    type(jit_only.plugins.ast_preprocessor).__name__,
    jit_only.plugins.ast_preprocessor.closure_check,
    [p.name for f in m.Plugins.FAMILIES for p in getattr(jit_only.plugins, f)],
)

# Naming the plugin is the switch: a record without one never rewrites, so a
# native loop over a staged bound is the plain-Python failure.
Builders = make("MLIR_DSL", None)


@Builders.jit
def native_without_plugin(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i
    return acc


try:
    native_without_plugin(5)
    print("NO PREPROCESSOR: no error (unexpected)")
except m.DSLUserCodeError as e:
    print("NO PREPROCESSOR:", Builders().enable_preprocessor, e.diag_id.name)
# CHECK: NO PREPROCESSOR: False PHASE_DYNAMIC_INDEX
# EXEC:  NO PREPROCESSOR: False PHASE_DYNAMIC_INDEX


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


# The rewrite recognises every decorator the record names: a third decorator
# plugin (`@task`, a `func.Jit` under another name) gets it too, so a native
# loop over a staged bound inside a `@task` function becomes `scf.for`.
class Tasks(func.Jit):
    name = "tasks"
    decorator_name = "task"


class WithTasks(m.BaseDSL):
    plugins = m.Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        ast_preprocessor=scf.ASTPreprocessor(),
        compiler=execution_engine.Compiler(),
        decorators=[func.Jit(), Tasks()],
    )

    def pipeline(self):
        return list(m.LOWER_TO_LLVM)

    def __init__(self):
        super().__init__(name="MLIR_DSL")


@WithTasks.task
def task_sum(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i
    return acc


# CHECK-LABEL: func.func @task_sum(
# CHECK:         scf.for
# EXEC:          TASK: 10
print("TASK:", task_sum(5))
print("DECORATORS:", sorted(WithTasks().preprocessor.decorator_names))
# CHECK: DECORATORS: ['jit', 'task']
# EXEC:  DECORATORS: ['jit', 'task']
