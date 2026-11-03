# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit
# The compile step as an object. `dsl.prepare_compile(...)` returns a
# `CompileStep` (the class `_compile_step_class` names) whose `run()`
# compiles under the module hash's lock (`dsl.module_locks`), builds through
# `build()` and publishes; `compile_and_cache` is `prepare_compile(...).run()`.
# Two threads over the same hash compile once: the second waits for the
# first, finds its function in the memory cache and takes it, its
# post-processors skipped. `build_engine=False` lowers the module without an
# engine: the handle holds the lowered module and refuses to be called. The
# entry is an explicit `llvm.func` driven through the DSL's compile driver,
# no decorator involved; the file cache is off so every compile is this
# process's.
import inspect
import threading
import time

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.core.dsl import CompileStep

SRC = "module { llvm.func @f(%a: i32) -> i32 { llvm.return %a : i32 } }"
PIPELINE = "builtin.module(canonicalize)"
sig = inspect.signature(lambda a: None)
builds = []
tags = []
# While set, a build waits until a second thread wants the same hash.
single_flight = threading.Event()


class WaitingStep(CompileStep):
    def build(self):
        if single_flight.is_set():
            deadline = time.monotonic() + 30
            while (
                dsl.module_locks.users(self.module_hash) < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
        builds.append(threading.current_thread().name)
        return super().build()


class SingleFlightDSL(m.MlirTestDSL):
    _compile_step_class = WaitingStep


dsl = SingleFlightDSL()


def tag(jit_function):
    jit_function.tagged = True
    tags.append(jit_function.function_name)


def prepare(module, module_hash, **options):
    return dsl.prepare_compile(
        module, module_hash, "f", PIPELINE, sig, False, **options
    )


# --- single flight -----------------------------------------------
# Each thread parses the text in a context of its own; the hashes are equal.
results = {}
hashes = {}


def compile_f(name):
    with ir.Context():
        module = ir.Module.parse(SRC)
        module_hash = hashes[name] = dsl.get_module_hash(module, "f", pipeline=PIPELINE)
        results[name] = prepare(module, module_hash, post_processors=[tag]).run()


single_flight.set()
threads = [threading.Thread(target=compile_f, args=(n,), name=n) for n in ("a", "b")]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
single_flight.clear()
first = results["a"]
print("BUILDS:", len(builds), "SAME:", first is results["b"])
print("TAGGED:", getattr(first, "tagged", False), "TAGS:", len(tags))
bound = first.capi_func is not None and first.engine is not None
print("BOUND:", bound, "HASHES:", hashes["a"] == hashes["b"], end=" ")
print("USERS:", dsl.module_locks.users(hashes["a"]), end=" ")
print("HITS:", dsl.cache_hits, "MISSES:", dsl.cache_misses)
# CHECK:      BUILDS: 1 SAME: True
# CHECK-NEXT: TAGGED: True TAGS: 1
# CHECK-NEXT: BOUND: True HASHES: True USERS: 0 HITS: 1 MISSES: 1

# --- no engine -----------------------------------------------------
# `build_engine=False`: the pipeline ran (`compile_module`), the handle has
# the lowered module and no engine, and calling it is the engine error.
with ir.Context():
    module = ir.Module.parse(SRC)
    handle = prepare(module, None, build_engine=False).run()
    print("LOWERED:", "llvm.func @f" in str(handle.ir_module), handle.engine is None)
    try:
        handle(1)
        print("NO_ENGINE: False")
    except m.DSLRuntimeError as e:
        print("NO_ENGINE:", "valid execution engine" in e.message)
# CHECK:      LOWERED: True True
# CHECK-NEXT: NO_ENGINE: True
