# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit
# The compile half of a call as hooks on the DSL. `_pop_call_keywords` takes
# the DSL's own call keywords out of the call (`label=`, `artifact=` and
# `fresh=` here) into `setup.options`; `_resolve_function_name` names the
# entry; `_disable_cache_reason` turns the cache off for one call;
# `_configure_compilation(setup)` is entered around the trace and the
# compile; `after_trace_result(trace, setup)` sees the finished trace;
# `emit_artifact(trace, setup)` ends the call with a value of its own (the
# module text: nothing is compiled, no engine is built) or hands the compile
# back as a `PendingCompile`, which the decorated function's wrapper runs
# once the call scope is left (inside it with `needs_call_scope`);
# `_call_result(jit_function, raw, trace)` builds the call's result from the
# raw slot and the trace. The file cache is off so every compile is this
# process's.
import contextlib
import sys

import mlir.mlir_dsl as m
from mlir.dsl.core.dsl import PendingCompile


class StepsDSL(m.MlirTestDSL):
    """Reserves `label=`, `artifact=` and `fresh=`; prints every hook."""

    @classmethod
    @contextlib.contextmanager
    def call_scope(cls, func):
        print("SCOPE: enter", func.__name__)
        try:
            yield
        finally:
            print("SCOPE: left", func.__name__)

    def _pop_call_keywords(self, kwargs):
        options = super()._pop_call_keywords(kwargs)
        for name in ("label", "artifact", "fresh"):
            options[name] = kwargs.pop(name, None)
        return options

    def _resolve_function_name(self, func, args, sig, options):
        if options["label"]:
            return f"{func.__name__}_{options['label']}"
        return super()._resolve_function_name(func, args, sig, options)

    def _disable_cache_reason(self, options):
        if options["fresh"]:
            return "the call asked for a fresh compile"
        return super()._disable_cache_reason(options)

    @contextlib.contextmanager
    def _configure_compilation(self, setup):
        given = sorted(name for name, value in setup.options.items() if value)
        print("CONFIGURE:", setup.function_name, given)
        yield
        print("CONFIGURED: left")

    def after_trace_result(self, trace, setup):
        hashed = trace.module_hash is not None
        print("TRACE DONE:", trace.function_name, "hashed:", hashed)

    def emit_artifact(self, trace, setup):
        artifact = setup.options["artifact"]
        if artifact == "text":
            return str(trace.module)
        if artifact in ("deferred", "in-scope"):

            def run():
                jit_function = self.compile_and_cache(
                    trace.module,
                    trace.module_hash,
                    trace.function_name,
                    trace.pipeline,
                    trace.sig,
                    setup.no_cache,
                    result_spec=trace.result_spec,
                    extra_link_libs=trace.extra_link_libs,
                )
                print("COMPILED:", trace.function_name)
                result = self.invoke_compiled(jit_function, trace)
                print("RAN:", result)
                return result

            return PendingCompile(run=run, needs_call_scope=artifact == "in-scope")
        return None

    def _call_result(self, jit_function, raw, trace):
        return f"{raw} from {type(trace.result).__name__}"


@StepsDSL.jit
def f(n: m.Int32) -> m.Int32:
    return n * 2


# `artifact="text"`: the module text is the call's result and the compiler
# plugin never imported the engine; `label=` named the entry.
text = f(1, label="label", artifact="text")
engine_imported = "mlir.execution_engine" in sys.modules
print("TEXT:", "func.func @f_label(" in text, "engine:", engine_imported)
# CHECK:      SCOPE: enter f
# CHECK-NEXT: CONFIGURE: f_label ['artifact', 'label']
# CHECK-NEXT: TRACE DONE: f_label hashed: True
# CHECK-NEXT: CONFIGURED: left
# CHECK-NEXT: SCOPE: left f
# CHECK-NEXT: TEXT: True engine: False

# The default path: trace, compile and run inside the configured scope, the
# result built by `_call_result`.
print("PLAIN:", f(2))
# CHECK-NEXT: SCOPE: enter f
# CHECK-NEXT: CONFIGURE: f []
# CHECK-NEXT: TRACE DONE: f hashed: True
# CHECK-NEXT: CONFIGURED: left
# CHECK-NEXT: SCOPE: left f
# CHECK-NEXT: PLAIN: 4 from Int32

# A deferred step runs once the call scope is left ...
print("DEFERRED:", f(3, artifact="deferred"))
# CHECK-NEXT: SCOPE: enter f
# CHECK-NEXT: CONFIGURE: f ['artifact']
# CHECK-NEXT: TRACE DONE: f hashed: True
# CHECK-NEXT: CONFIGURED: left
# CHECK-NEXT: SCOPE: left f
# CHECK-NEXT: COMPILED: f
# CHECK-NEXT: RAN: 6 from Int32
# CHECK-NEXT: DEFERRED: 6 from Int32

# ... or inside it when it says so.
print("IN SCOPE:", f(4, artifact="in-scope"))
# CHECK-NEXT: SCOPE: enter f
# CHECK-NEXT: CONFIGURE: f ['artifact']
# CHECK-NEXT: TRACE DONE: f hashed: True
# CHECK-NEXT: CONFIGURED: left
# CHECK-NEXT: COMPILED: f
# CHECK-NEXT: RAN: 8 from Int32
# CHECK-NEXT: SCOPE: left f
# CHECK-NEXT: IN SCOPE: 8 from Int32

# `fresh=` turns the cache off for the call: no hash, and a compile although
# the same module was compiled above; the plain call after it is a hit.
dsl = StepsDSL()
before = dsl.cache_misses, dsl.cache_hits
fresh = f(2, fresh=True)
print("FRESH:", fresh, dsl.cache_misses - before[0], dsl.cache_hits - before[1])
cached = f(2)
print("CACHED:", cached, dsl.cache_misses - before[0], dsl.cache_hits - before[1])
# CHECK:      CONFIGURE: f ['fresh']
# CHECK-NEXT: TRACE DONE: f hashed: False
# CHECK:      FRESH: 4 from Int32 1 0
# CHECK:      TRACE DONE: f hashed: True
# CHECK:      CACHED: 4 from Int32 1 1
