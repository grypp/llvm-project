# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit
# The executor hooks for a DSL whose entries carry more than the Python
# signature says. `JitExecutor(trailing_args=...)` appends parameters to the
# packed array, which reads `[user slots, trailing slots, result slot]`;
# `result_storage()` picks the object the entry stores its result through,
# `check_result(raw)` turns it into the call's value or the DSL's error, and
# `_before_call(args, kwargs)` runs first on every call. On the function side,
# `JitCompiledFunction.executor(**options)` builds the shared `JitModule` once
# through `_make_jit_module()` and every executor (the default one included)
# through `_make_executor(jit_module, **options)`; the compiler plugin's
# `load(..., func_type=...)` picks the class. The entry is an explicit
# `llvm.func` driven through the compiler plugin, no decorator involved.
import ctypes
import inspect
import threading

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.compiler.jit_executor import (
    JitCompiledFunction,
    JitExecutor,
    JitModule,
)

# `@entry(n, tail)` stores 42 through `tail` and returns the status 7 when `n`
# is negative, 0 otherwise. The Python signature names only `n`: `tail` is a
# trailing parameter the executor supplies, the status is the result slot.
SRC = """
module {
  llvm.func @entry(%n: i32, %tail: !llvm.ptr) -> i32 {
    %c42 = llvm.mlir.constant(42 : i32) : i32
    llvm.store %c42, %tail : i32, !llvm.ptr
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %neg = llvm.icmp "slt" %n, %c0 : i32
    %status = llvm.select %neg, %c7, %c0 : i1, i32
    llvm.return %status : i32
  }
}
"""

# The packed wrapper reads parameter `i` through slot `i`: the trailing slot
# points at the storage holding the pointer value, not at the pointee.
tail_cell = ctypes.c_int32(0)
tail_param = ctypes.c_void_p(ctypes.addressof(tail_cell))
counts = {"calls": 0, "direct": 0, "modules": 0, "executors": 0}
status_slots = []


class CountedModule(JitModule):
    def __init__(self, *args):
        super().__init__(*args)
        counts["modules"] += 1


class StatusExecutor(JitExecutor):
    """One status cell per thread; a nonzero status is the DSL's error."""

    def __init__(self, jit_module, **options):
        super().__init__(
            jit_module,
            result_ctype=ctypes.c_int32,
            trailing_args=[m.make_owning_c_pointer(tail_param)],
            **options,
        )
        self._cells = threading.local()

    def _before_call(self, args, kwargs):
        counts["calls"] += 1

    def result_storage(self):
        cell = getattr(self._cells, "status", None)
        if cell is None:
            cell = self._cells.status = ctypes.c_int32(-1)
        return cell

    def check_result(self, raw):
        status_slots.append(ctypes.addressof(raw))
        if raw.value != 0:
            raise m.DSLRuntimeError(f"status {raw.value}")
        return "ok"


class StatusFunction(JitCompiledFunction):
    def _make_jit_module(self):
        return CountedModule(self.engine, self.capi_func, self.execution_args)

    def _make_executor(self, jit_module, **options):
        counts["executors"] += 1
        return StatusExecutor(jit_module, **options)

    def _before_call(self, args, kwargs):
        counts["direct"] += 1


def proto(n: m.Int32):
    pass


def status_of(call):
    try:
        return call()
    except m.DSLRuntimeError as e:
        return e.message.split()[-1]


compiler = execution_engine.Compiler()
with ir.Context():
    module = ir.Module.parse(SRC)
    engine = compiler.jit(module)
    fn = compiler.load(
        module, engine, "entry", inspect.signature(proto), func_type=StatusFunction
    )
    run_a = fn.executor()
    run_b = fn.executor()
    ok, failed = status_of(lambda: run_a(1)), status_of(lambda: run_a(-1))
    print(f"STATUS: {ok} {failed} TAIL: {tail_cell.value}", end=" ")
    print(f"CALLS: {counts['calls']} MODULES: {counts['modules']}", end=" ")
    print(f"EXECUTORS: {counts['executors']}")
    # The per-thread status cell: the same address across one thread's calls,
    # another one on another thread; the two executors share one module.
    print("SAME_SLOT:", status_slots[0] == status_slots[1], end=" ")
    print("SHARED_MODULE:", run_a.jit_module is run_b.jit_module)
    other = []
    worker = threading.Thread(target=lambda: other.append(run_b(2)))
    worker.start()
    worker.join()
    print("DISTINCT_SLOTS:", status_slots[2] != status_slots[0], other)
    # A direct call goes through the function's own `_before_call` and its
    # default executor, built by the same factory.
    print("DIRECT:", fn(3), counts["direct"], counts["executors"], counts["calls"])
# CHECK:      STATUS: ok 7 TAIL: 42 CALLS: 2 MODULES: 1 EXECUTORS: 2
# CHECK-NEXT: SAME_SLOT: True SHARED_MODULE: True
# CHECK-NEXT: DISTINCT_SLOTS: True ['ok']
# CHECK-NEXT: DIRECT: ok 1 3 3

# Without hooks the result slot is the only extra slot and the raw result is
# the Python value: the default executor over the same entry, with the tail
# supplied through `executor(trailing_args=...)`.
with ir.Context():
    module = ir.Module.parse(SRC)
    engine = compiler.jit(module)
    plain = compiler.load(
        module,
        engine,
        "entry",
        inspect.signature(proto),
        result_ctype=ctypes.c_int32,
    )
    tail_cell.value = 0
    run = plain.executor(trailing_args=[tail_param and ctypes.addressof(tail_param)])
    print("PLAIN:", run(-5), run(5), tail_cell.value, type(run).__name__)
# CHECK: PLAIN: 7 0 42 JitExecutor
