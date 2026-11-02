# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The AST preprocessor plugin: a `BaseDSL` assembled from a `Plugins` record
# gets native control flow from the `scf.ASTPreprocessor` plugin (preprocessor
# + executors); its `closure_check` knob allows captures in staged regions; a
# DSL without an `ast_preprocessor` plugin never rewrites (native control flow
# over a staged value fails, the explicit builders still work); a plugin
# subclass may swap the executors.
from dataclasses import replace
from typing import NamedTuple

import mlir.mlir_dsl as m
from mlir.dsl.core.diagnostics import DiagId
from mlir.dsl.plugins.ast_preprocessor import DSLPreprocessor, scf
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


# The value model is open: the test DSL registers `typing.NamedTuple` as a
# container and the `__ir_types__`/`__ir_values__`/`__from_ir_values__` protocol
# as a leaf, so a `Pair` carried through a staged loop becomes the `iter_args`
# of its fields and a `Complex` rebound in a staged `if` becomes two results.
class Pair(NamedTuple):
    re: m.Float32
    im: m.Int32


class Complex:
    def __init__(self, re, im):
        self.re = re
        self.im = im

    def __ir_types__(self):
        return [m.Float32.mlir_type, m.Float32.mlir_type]

    def __ir_values__(self):
        return [self.re.ir_value(), self.im.ir_value()]

    @classmethod
    def __from_ir_values__(cls, values):
        return cls(m.Float32(values[0]), m.Float32(values[1]))


@m.jit
def pair_loop(n: m.Int32) -> m.Float32:
    p = Pair(m.Float32(0.0), m.Int32(0))
    for i in range(n):
        p = Pair(p.re + 1.5, p.im + 1)
    return p.re


# CHECK-LABEL: func.func @pair_loop(
# CHECK:         scf.for {{.*}} iter_args(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) -> (f32, i32)
# CHECK:           scf.yield %{{.+}}, %{{.+}} : f32, i32
# EXEC:          PAIR_LOOP: 6.0
print("PAIR_LOOP:", pair_loop(4))


@m.jit
def swap_if(z: Complex, flag: m.Int32) -> m.Float32:
    if flag > 0:
        z = Complex(z.im, z.re)
    return z.re


# CHECK-LABEL: func.func @swap_if(
# CHECK-SAME:    %{{[^:]+}}: f32, %{{[^:]+}}: f32, %{{[^:]+}}: i32) -> f32
# CHECK:         scf.if %{{.+}} -> (f32, f32)
# CHECK:           scf.yield %{{.+}}, %{{.+}} : f32, f32
# EXEC:          SWAP_IF: 2.0
print("SWAP_IF:", swap_if(Complex(m.Float32(1.0), m.Float32(2.0)), 1))


# The rewrite's diagnostics are a table keyed by the reason a construct is
# rejected (`DSLPreprocessor.DIAGNOSTICS`): a preprocessor subclass overrides
# one entry, here the wording of `global`, and the visitors stay as they are;
# the shipped preprocessor keeps the shipped wording.
class StrictPreprocessor(DSLPreprocessor):
    DIAGNOSTICS = {
        **DSLPreprocessor.DIAGNOSTICS,
        "global": (
            DiagId.UNSUP_SYNTAX,
            {
                "what": "`global`",
                "detail": ": this DSL keeps module state out of compiled code",
            },
        ),
    }


class StrictASTPreprocessor(scf.ASTPreprocessor):
    name = "strict_preprocessor"
    preprocessor_class = StrictPreprocessor


Strict = make("MLIR_DSL", StrictASTPreprocessor())
counter = 0


@Strict.jit
def bumps(n: m.Int32) -> m.Int32:
    global counter
    counter += 1
    return n


@m.jit
def bumps_default(n: m.Int32) -> m.Int32:
    global counter
    counter += 1
    return n


for label, fn in (("STRICT GLOBAL:", bumps), ("DEFAULT GLOBAL:", bumps_default)):
    try:
        fn(1)
        print(label, "no error")
    except m.DSLUserCodeError as e:
        print(label, e.diag_id.name, "|", e.message)
# CHECK: STRICT GLOBAL: UNSUP_SYNTAX | `global` is not supported in a compiled function: this DSL keeps module state out of compiled code.
# EXEC:  STRICT GLOBAL: UNSUP_SYNTAX | `global` is not supported in a compiled function: this DSL keeps module state out of compiled code.
# CHECK: DEFAULT GLOBAL: UNSUP_SYNTAX | `global` is not supported in a compiled function: compiled code cannot assign to a module-level variable; pass the value in as an argument and return the updated value.
# EXEC:  DEFAULT GLOBAL: UNSUP_SYNTAX | `global` is not supported in a compiled function: compiled code cannot assign to a module-level variable; pass the value in as an argument and return the updated value.


# --- HELPERS: the module behind `__base_dsl__` -----------------------------
# The rewritten code imports one module as `__base_dsl__` and resolves every
# emitted helper call against it. The plugin's `helpers_module` names another
# one, which `make_preprocessor` hands to the preprocessor: here a module built
# in this script that re-exports the shipped helpers and replaces
# `assert_executor` with one that reports at trace time.
import ast
import contextlib
import copy
import sys
import types

from mlir.dsl.plugins.ast_preprocessor import helpers as shipped_helpers
from mlir.dsl.plugins.ast_preprocessor.preprocessor import create_module_attribute

trace_helpers = types.ModuleType("trace_helpers")
for helper_name in shipped_helpers.__all__:
    setattr(trace_helpers, helper_name, getattr(shipped_helpers, helper_name))


def assert_executor(test, msg=None):
    print("[assert]", msg)
    return shipped_helpers.assert_executor(test, msg)


def enter_function(name):
    print(f"[enter {name}]")


trace_helpers.assert_executor = assert_executor
trace_helpers.enter_function = enter_function
sys.modules["trace_helpers"] = trace_helpers

Reporting = make("MLIR_DSL", scf.ASTPreprocessor(helpers_module="trace_helpers"))


@Reporting.jit
def checked(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    assert n is not None, "n is bound"
    for i in range(n):
        acc = acc + i
    return acc


# CHECK:       [assert] n is bound
# CHECK-LABEL: func.func @checked(
# CHECK:         scf.for
# EXEC:        [assert] n is bound
# EXEC:        CHECKED: 10
print("CHECKED:", checked(5))
print("HELPERS:", Reporting().preprocessor.helpers_module)
# CHECK: HELPERS: trace_helpers
# EXEC:  HELPERS: trace_helpers


# --- Tracing: another preprocessor for the same function, later ------------
# A DSL may run a `@jit` function through another preprocessor later in the
# process (a tracing mode): `make_preprocessor` builds a preprocessor from a
# plugin, `preprocess_key` tells the DSL that the function must be rewritten
# again (the rewrite is cached per function otherwise), and the preprocessor's
# `parse_function_source`/`post_visit_function_body` let the tracing one serve
# a memoised parse and open every body with a call of its own helper, built
# with `create_module_attribute` against `__base_dsl__`.
class TracingPreprocessor(DSLPreprocessor):
    parsed = {}

    def parse_function_source(self, function_pointer):
        cached = TracingPreprocessor.parsed.get(function_pointer)
        if cached is None:
            cached = super().parse_function_source(function_pointer)
            TracingPreprocessor.parsed[function_pointer] = cached
        else:
            print(f"[parse cache hit {function_pointer.__name__}]")
        tree, file_name = cached
        return copy.deepcopy(tree), file_name

    def post_visit_function_body(self, func_def):
        enter = ast.Call(
            func=create_module_attribute("enter_function"),
            args=[ast.Constant(value=func_def.name)],
            keywords=[],
        )
        return [ast.Expr(enter)]


class TracingASTPreprocessor(scf.ASTPreprocessor):
    name = "tracing_preprocessor"
    preprocessor_class = TracingPreprocessor
    helpers_module = "trace_helpers"


class Switchable(m.BaseDSL):
    plugins = m.Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        ast_preprocessor=scf.ASTPreprocessor(),
        compiler=execution_engine.Compiler(),
        decorators=[func.Jit()],
    )

    def pipeline(self):
        return list(m.LOWER_TO_LLVM)

    def __init__(self):
        super().__init__(name="MLIR_DSL")
        self.plain_preprocessor = self.preprocessor
        self.tracing_preprocessor = TracingASTPreprocessor().make_preprocessor(
            self,
            self.preprocessor.client_module_name,
            decorator_names=sorted(self.preprocessor.decorator_names),
        )

    @contextlib.contextmanager
    def tracing(self):
        # Swapping the preprocessor changes `preprocess_key` (the preprocessor
        # class and its helpers module), so the next call rewrites again.
        self.preprocessor = self.tracing_preprocessor
        try:
            yield
        finally:
            self.preprocessor = self.plain_preprocessor


@Switchable.jit
def traced(
    n: m.Int32, call
) -> m.Int32:  # `call` is a Python value: each one is traced anew
    acc = m.Int32(0)
    for i in range(n):
        acc = acc + i
    return acc


switchable = Switchable()
# CHECK-NOT: [enter traced]
# CHECK:     PLAIN: ?
# EXEC:      PLAIN: 3
print("PLAIN:", traced(3, 1))
with switchable.tracing():
    # CHECK-NOT: [parse cache hit traced]
    # CHECK:     [enter traced]
    # CHECK:     TRACING: ?
    # EXEC:      [enter traced]
    # EXEC:      TRACING: 3
    print("TRACING:", traced(3, 2))
# CHECK-NOT: [enter traced]
# CHECK:     PLAIN AGAIN: ?
# EXEC:      PLAIN AGAIN: 3
print("PLAIN AGAIN:", traced(3, 3))
with switchable.tracing():
    # CHECK: [parse cache hit traced]
    # CHECK: [enter traced]
    # CHECK: TRACING AGAIN: ?
    # EXEC:  [parse cache hit traced]
    # EXEC:  [enter traced]
    # EXEC:  TRACING AGAIN: 3
    print("TRACING AGAIN:", traced(3, 4))
print(
    "KEYS:",
    switchable.preprocess_key(traced) != None,
    switchable.tracing_preprocessor.helpers_module,
)
# CHECK: KEYS: True trace_helpers
# EXEC:  KEYS: True trace_helpers
