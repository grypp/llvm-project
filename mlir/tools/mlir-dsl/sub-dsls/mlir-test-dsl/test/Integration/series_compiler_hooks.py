# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if system-windows %{ env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s printf 2>&1 | FileCheck %s --check-prefix=WINDOWS %} %else %{ true %}
# REQUIRES: host-supports-jit
# The compiler plugin's hooks for a DSL that reports its compiles its own way.
# The file cache is off: a lowered module another process cached under the
# same key (the compiler plugin is not part of it) would skip the pipeline,
# and with it the hooks this test watches.
# `compile(module, pipeline, **session_options)` hands every option it does
# not know to `remark_session(context, **options)`, so a subclass's session
# takes options of its own next to the remark settings; every exception out
# of the pipeline is raised as what `pipeline_failed(exc, session, pipeline)`
# returns (a DSL error passes through as it is; the parse failure keeps its
# message); `after_pipeline(session)` runs inside the session right after the
# passes, where it sees what the session collected. The same hooks run when
# a DSL names the subclass as its compiler. The `printf` mode is Windows only:
# there `jit()` makes the C stdout compiled code prints through unbuffered,
# so a line printed by compiled code keeps its place between Python's prints.
import ctypes
import dataclasses
import sys

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.core.diagnostics import DiagId
from mlir.dsl.core.remarks import error_diagnostics
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.compiler.jit_executor import lookup_packed_function

mode = sys.argv[1] if len(sys.argv) > 1 else ""
# The remarks the session renders go to stderr as they arrive; line-buffered
# stdout keeps the prints around them in program order.
sys.stdout.reconfigure(line_buffering=True)


class CountingCompiler(execution_engine.Compiler):
    """Records the forwarded `tag` option, blames a failing pass on the
    `pipeline` setting and reports what the session collected."""

    def remark_session(self, context, **options):
        print("SESSION_TAG:", options.pop("tag", None), sorted(options))
        return super().remark_session(context, **options)

    def pipeline_failed(self, exc, session, pipeline):
        if isinstance(exc, ir.MLIRError):
            return m.DSLUserCodeError(
                DiagId.CONFIG_INVALID,
                var="pipeline",
                detail=f"`{pipeline}` failed: {error_diagnostics(exc, session)}",
                suggestion="Pick a pipeline that lowers every operation of the module.",
            )
        return super().pipeline_failed(exc, session, pipeline)

    def after_pipeline(self, session):
        print(
            f"PIPELINE_DONE: {len(session.remarks)} remarks, {len(session.errors)} errors"
        )


class StrictCompiler(execution_engine.Compiler):
    """Promotes a remark the pipeline emitted to the DSL's own error."""

    def after_pipeline(self, session):
        if session.remarks:
            raise m.DSLUserCodeError(
                DiagId.CONFIG_INVALID,
                var="strict",
                detail=f"the pipeline reported: {session.remarks[0]['message']}",
            )


GOOD_SRC = "module { llvm.func @f(%a: i32) -> i32 { llvm.return %a : i32 } }"
# A pass that emits a remark at the module (the transform dialect's debug
# op), so the session has one record once the pipeline ran.
REMARK_SRC = """
module attributes {transform.with_named_sequence} {
  llvm.func @f(%a: i32) -> i32 { llvm.return %a : i32 }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %root, "pipeline ran" : !transform.any_op
    transform.yield
  }
}
"""
REMARK_PIPELINE = "builtin.module(transform-interpreter)"
BAD_SRC = (
    "module { func.func @g(%a: tensor<4xi32>) -> tensor<4xi32> {"
    " cf.br ^bb1(%a : tensor<4xi32>) ^bb1(%b: tensor<4xi32>): return %b : tensor<4xi32> } }"
)


def failure(fn):
    """The DiagId of the user error `fn` raises, the message of an internal one."""
    try:
        fn()
        return "no error"
    except m.DSLUserCodeError as e:
        return e.diag_id.name
    except m.DSLRuntimeError as e:
        return " ".join(e.message.split())


def windows_printf(compiler):
    """Compiled code printing through the C runtime between Python's prints."""
    src = """
    module {
      llvm.func @printf(!llvm.ptr, ...) -> i32
      llvm.mlir.global internal constant @fmt("from compiled code\\0A\\00")
      llvm.func @say() {
        %fmt = llvm.mlir.addressof @fmt : !llvm.ptr
        %r = llvm.call @printf(%fmt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
        llvm.return
      }
    }
    """
    with ir.Context():
        module = ir.Module.parse(src)
        engine = compiler.jit(module)
        say = lookup_packed_function(engine, "say")
        print("PY: before", flush=True)
        say((ctypes.c_void_p * 0)())
        print("PY: after", flush=True)


compiler = CountingCompiler()
if mode == "printf":
    windows_printf(compiler)
    raise SystemExit(0)
# WINDOWS:      PY: before
# WINDOWS-NEXT: from compiled code
# WINDOWS-NEXT: PY: after

with ir.Context() as ctx:
    # The forwarded option and the report after the passes, with and without
    # a remark emitted by the pipeline.
    compiler.compile(
        ir.Module.parse(GOOD_SRC), "builtin.module(canonicalize)", tag="t1"
    )
    compiler.compile(ir.Module.parse(REMARK_SRC), REMARK_PIPELINE, tag="t2")
    # CHECK:      SESSION_TAG: t1 []
    # CHECK-NEXT: PIPELINE_DONE: 0 remarks, 0 errors
    # CHECK-NEXT: SESSION_TAG: t2 []
    # CHECK-NEXT: remark{{.*}}pipeline ran
    # CHECK:      PIPELINE_DONE: 1 remarks, 0 errors

    # A failing pass is the user's `pipeline` setting, with the diagnostics
    # quoted; a pipeline that does not parse is still the internal error.
    try:
        compiler.compile(ir.Module.parse(BAD_SRC), "builtin.module(convert-cf-to-llvm)")
    except m.DSLUserCodeError as e:
        print("PASS_FAIL:", e.diag_id.name, "|", type(e.__cause__).__name__)
        print(str(e))
    print(
        "PARSE_FAIL:",
        failure(
            lambda: compiler.compile(
                ir.Module.parse(GOOD_SRC), "builtin.module(no-such-pass)"
            )
        ),
    )
    # CHECK:      SESSION_TAG: None []
    # CHECK-NEXT: PASS_FAIL: CONFIG_INVALID | MLIRError
    # CHECK:      error[CONFIG_INVALID]:{{.*}}`pipeline` has an invalid setting: `builtin.module(convert-cf-to-llvm)` failed:
    # CHECK:      failed to legalize operation 'cf.br'
    # CHECK:      suggestion:{{.*}}Pick a pipeline that lowers every operation of the module.
    # CHECK:      PARSE_FAIL: failed to parse the pass pipeline

    # A DSL error raised inside the session (here by `after_pipeline`) is
    # raised as it is, not wrapped as a pipeline failure.
    strict = StrictCompiler()
    try:
        strict.compile(ir.Module.parse(REMARK_SRC), REMARK_PIPELINE)
    except m.DSLUserCodeError as e:
        print("PROMOTED:", e.diag_id.name, "|", " ".join(e.message.split()))
    print(
        "LENIENT:",
        type(
            strict.compile(ir.Module.parse(GOOD_SRC), "builtin.module(canonicalize)")
        ).__name__,
    )
    # CHECK: PROMOTED: CONFIG_INVALID | `strict` has an invalid setting: the pipeline reported: pipeline ran.
    # CHECK: LENIENT: Module


# A DSL naming the subclass as its compiler reaches the same hooks from a
# decorated function: the session options the DSL passes are the remark
# settings (once for the DSL's own session around the trace, once for the
# compile), and the report follows the passes.
class HookedDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(m.MlirTestDSL.plugins, compiler=CountingCompiler())


@HookedDSL.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


print("RESULT:", twice(21))
# CHECK:      SESSION_TAG: None ['remark_filter', 'remark_output', 'remark_policy']
# CHECK-NEXT: SESSION_TAG: None ['remark_filter', 'remark_output', 'remark_policy']
# CHECK-NEXT: PIPELINE_DONE: 0 remarks, 0 errors
# CHECK-NEXT: RESULT: 42
