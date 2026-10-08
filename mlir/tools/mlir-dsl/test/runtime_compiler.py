# RUN: rm -rf %t.dir && env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 MLIR_DSL_CACHE_DIR=%t.dir %PYTHON %s 2>&1 | FileCheck %s
# RUN: rm -rf %t.cache && env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=EXEC,MISS
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s corrupt 2>&1 | FileCheck %s --check-prefixes=EXEC,CORRUPT
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=EXEC,HIT
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=EXEC,OFF
# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 MLIR_DSL_JIT_CACHE_MAX_ELEMS=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=EXEC,ONE
# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 MLIR_DSL_NO_CACHE=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=EXEC,ZERO
# RUN: rm -rf %t.keep && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEP_IR=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_CACHE_DIR=%t.keep %PYTHON %s 2>&1 | FileCheck %s --check-prefix=KEEP
# RUN: rm -rf %t.after && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEPIR_AFTER_PASSES="builtin.module(convert-scf-to-cf)" MLIR_DSL_CACHE_DIR=%t.after %PYTHON %s 2>&1 | FileCheck %s --check-prefix=AFTER
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR_AFTER_PASSES=convert-scf-to-cf %PYTHON %s 2>&1 | FileCheck %s --check-prefix=PRINT
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PIPELINE="builtin.module(canonicalize{top-down=true})" MLIR_DSL_ARCH=sm_90 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=ENVPIPE
# REQUIRES: host-supports-jit
# The runtime and the compiler: the jit argument
# adapters (registry, numpy/torch/sequence/dataclass adapters, the `Pointer[T]`
# annotation, `ExecutionArgs` marshalling, the result decoder); the in-memory
# JIT cache (`JitCacheDict`, the counters, `JIT_CACHE_MAX_ELEMS`/`NO_CACHE`);
# the on-disk cache (a second process hits, a damaged entry is a miss with a
# warning and is rewritten, `DISABLE_FILE_CACHING` leaves the files alone);
# the pass pipeline as data (`<PREFIX>_PIPELINE`/`_ARCH`, the `pipeline=`
# keyword, parse/engine/pass failures as DSLRuntimeErrors); `KEEP_IR`,
# `KEEPIR_AFTER_PASSES` and `PRINT_IR_AFTER_PASSES`. The EXEC lines share one
# cache directory in sequence: miss, damaged entries (`corrupt`), hit, off.
import ctypes
import gc
import inspect
import logging
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.compiler.jit_executor import ExecutionArgs
from mlir.dsl.util.cache import JitCacheDict
from mlir.dsl.core.arguments import DefaultDataclassAdapter
from mlir.dsl.core.arguments import JitArgAdapterRegistry as R
from mlir.dsl.core.arguments import adapt_pointer_address
from mlir.dsl.util.logger import log

dsl = m.MlirTestDSL()
mode = sys.argv[1] if len(sys.argv) > 1 else ""
cache_dir = dsl.envar.cache_dir
warnings = []


class Capture(logging.Handler):
    def emit(self, rec):
        if rec.levelno >= logging.WARNING:
            warnings.append(rec.getMessage())


log().addHandler(Capture())
log().setLevel(logging.INFO)


def failure(fn):
    """The DiagId of the user error `fn` raises, the message of an internal one."""
    try:
        fn()
        return "no error"
    except m.DSLUserCodeError as e:
        return e.diag_id.name
    except m.DSLRuntimeError as e:
        return " ".join(e.message.split())


# --- the adapter registry ---------------------------------
# Adapters are keyed by type (common) or by (scope, type); a scoped adapter
# shadows the common one inside `using_scope`; a lazy registration by
# qualified name is promoted on the first lookup of an instance, which is how
# `torch.Tensor` is registered without importing torch; Python scalars have no
# adapter (they are Meta values); malformed registrations are internal errors.
class Handle:
    def __init__(self, v):
        self.v = v


class Third:
    pass


@R.register_jit_arg_adapter(Handle)
def adapt_handle(h):
    return m.Int32(h.v)


@R.register_jit_arg_adapter(Handle, scope="s1")
def adapt_handle_s1(h):
    return m.Int64(h.v)


@R.register_jit_arg_adapter(f"{__name__}.Third", lazy=True)
def adapt_third(t):
    return m.Int32(3)


with R.using_scope("s1"):
    scoped = R.get_registered_adapter(Handle(1)) is adapt_handle_s1
common = R.get_registered_adapter(Handle(1)) is adapt_handle
scalars = [R.get_registered_adapter(v) for v in (1, 2.5, "s", None)]
print(f"REGISTRY: {common} {scoped} {R.get_registered_adapter(object())} {scalars}")
lazy_before = Third in R.jit_arg_adapter_registry
lazy_named = f"{__name__}.Third" in R.lazy_jit_arg_adapter_registry
lazy_hit = R.get_registered_adapter(Third()) is adapt_third
# The lazy entry is keyed by qualified name until the first instance promotes
# it to the type-keyed registry.
print(
    f"LAZY: {lazy_before} {lazy_named} {lazy_hit} {Third in R.jit_arg_adapter_registry}"
)
dup = failure(lambda: R.register_jit_arg_adapter(Handle)(lambda h: h))
bad = failure(lambda: R.register_jit_arg_adapter("T", lazy=True))
print("MALFORMED:", dup, "|", bad)
# CHECK:      REGISTRY: True True None [None, None, None, None]
# CHECK-NEXT: LAZY: False True True True
# CHECK-NEXT: MALFORMED: {{.*}}already registered{{.*}} | {{.*}}must be fully-qualified

# --- the buffer adapters -----------------------------------------
# A C-contiguous numpy array (or CPU torch tensor) becomes, through the dlpack
# adapter, a host `Pointer` over its own storage, dtype from the DLPack type
# code, the buffer kept alive by the view, nothing copied; a non-contiguous
# buffer and an unmapped dtype are diagnostics naming the argument being
# adapted.
adapt_np = R.get_registered_adapter(np.zeros(1))
arr = np.arange(4, dtype=np.float32)
p = adapt_np(arr)
kinds = (np.bool_, np.int8, np.uint16, np.int64, np.float16, np.float64)
print(f"NUMPY: {type(p).__name__} {p.dtype.__name__} {p.kind}", end=" ")
keeps = p._keepalive is arr or getattr(p._keepalive, "tensor", None) is arr
print(p.address == arr.ctypes.data, keeps, end=" ")
print([adapt_np(np.zeros(1, d)).dtype.__name__ for d in kinds])
print("NUMPY_ERRORS:", failure(lambda: adapt_np(arr[::2])), end=" ")
print(failure(lambda: adapt_np(np.zeros((2, 3), np.float32).T)), end=" ")
print(failure(lambda: adapt_np(np.zeros(2, np.complex64))))
try:
    with R.using_argument("x", 1):
        adapt_np(arr[::2])
except m.DSLUserCodeError as e:
    print(str(e))
# CHECK:      NUMPY: Pointer Float32 host True True ['Boolean', 'Int8', 'Uint16', 'Int64', 'Float16', 'Float64']
# CHECK-NEXT: NUMPY_ERRORS: ARG_BUFFER_INVALID ARG_BUFFER_INVALID TYPE_UNKNOWN_DTYPE_NAME
# CHECK:      error[ARG_BUFFER_INVALID]:{{.*}}Argument `x` cannot be used as a `Pointer` argument: it is a `numpy.ndarray` that is not contiguous in memory
# CHECK:      suggestion:{{.*}}Pass one contiguous block of memory
# CHECK:      `np.ascontiguousarray(a)`, `t.contiguous()`

try:
    import torch
except ImportError:
    torch = None
if torch is None:
    print("TORCH: skipped")
else:
    t = torch.arange(4, dtype=torch.float32)
    adapt_torch = R.get_registered_adapter(t)
    tp = adapt_torch(t)
    strided = failure(lambda: adapt_torch(torch.zeros(8)[::2]))
    print(f"TORCH: {tp.dtype.__name__} {tp.kind} {tp.address == t.data_ptr()}", strided)
# CHECK: TORCH: {{Float32 host True ARG_BUFFER_INVALID|skipped}}

# --- containers, dataclasses and the `Pointer[T]` annotation ----------------
# The sequence adapter adapts every element and keeps the container type (a
# tuple is rebuilt); the default dataclass adapter casts the `Numeric`
# fields of a frozen record, adapts and checks its `Pointer[T]` fields (a
# float32 buffer under `Pointer[Int32]` is a mismatch, as for a parameter) and
# copies the Python-typed (Meta) ones; a non-frozen record holding DSL values
# is rejected; under a `Pointer[T]` annotation a bare address (int or
# `c_void_p`) becomes a Pointer of the annotated dtype and space, a buffer goes
# through the registry, a negative address and a dtype or space mismatch are
# diagnostics.
seq = R.get_registered_adapter((arr,))


@dataclass(frozen=True)
class Pair:
    buf: m.Pointer[m.Int32]
    n: int


t = seq((arr, [np.zeros(2, np.int32), 4], "x"))
pair = DefaultDataclassAdapter(Pair(np.zeros(2, np.int32), 3))
names = [type(x).__name__ for x in (t, t[0], t[1], t[1][0], pair, pair.buf)]
print("SEQUENCE:", *names, t[1][1], t[2], pair.n, pair.buf.dtype.__name__)
print("FIELD MISMATCH:", failure(lambda: DefaultDataclassAdapter(Pair(arr, 3))))
# CHECK: SEQUENCE: tuple Pointer list Pointer Pair Pointer 4 x 3 Int32
# CHECK: FIELD MISMATCH: ARG_ANNOTATION_MISMATCH


@dataclass(frozen=True, init=False)
class Quoted:
    buf: "m.Pointer[m.Float32]"  # string annotations are resolved
    n: "m.Int32"

    def __init__(self, buf):  # a custom constructor: the adapter rebuilds
        object.__setattr__(self, "buf", buf)  # field by field, no `__init__`
        object.__setattr__(self, "n", 7)


q = DefaultDataclassAdapter(Quoted(arr))
print(
    "QUOTED:", type(q.buf).__name__, q.buf.dtype.__name__, type(q.n).__name__, q.n.value
)
# CHECK: QUOTED: Pointer Float32 Int32 7


@dataclass(frozen=True)
class Config:
    n: m.Int32
    scale: m.Float32
    label: str  # no DSL annotation: a Meta field, copied as is


@dataclass
class MutableCfg:
    n: m.Int32


cfg = DefaultDataclassAdapter(Config(n=4, scale=2, label="x"))
default = R.get_registered_adapter(Config(1, 1.0, "a")) is DefaultDataclassAdapter
print(f"DATACLASS: {default} {type(cfg.n).__name__} {cfg.n.value}", end=" ")
print(f"{type(cfg.scale).__name__} {cfg.scale.value} {cfg.label}", end=" ")
print(failure(lambda: R.get_registered_adapter(MutableCfg(4))))
# CHECK: DATACLASS: True Int32 4 Float32 2.0 x CONTAINER_INVALID_RECORD

F32 = m.Pointer[m.Float32]
F32_S3 = m.Pointer[m.Float32, 3]


def show(p):
    return (p.dtype.__name__, p.space, p.kind, p.address)


print("ADDRESS:", show(adapt_pointer_address(4096, F32)), end=" ")
print(show(adapt_pointer_address(ctypes.c_void_p(4096), F32_S3)), end=" ")
print(show(adapt_pointer_address(arr, F32))[:3])
print("ADDRESS_ERRORS:", failure(lambda: adapt_pointer_address(-1, F32)), end=" ")
print(failure(lambda: adapt_pointer_address(np.zeros(2, np.float64), F32)), end=" ")
print(failure(lambda: adapt_pointer_address(m.Pointer(0, dtype=m.Float32), F32_S3)))
# CHECK:      ADDRESS: ('Float32', 0, 'unknown', 4096) ('Float32', 3, 'unknown', 4096) ('Float32', 0, 'host')
# CHECK-NEXT: ADDRESS_ERRORS: ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH


# --- `ExecutionArgs` --------------------------------------------
# The binder of a compiled function: positional/keyword/default arguments bind
# like Python, every runtime value is marshalled into one `c_void_p` slot
# (half precision as its bit pattern), a Meta value contributes none, binding
# mistakes and a dtype without a host representative are diagnostics.
def binder(fn):
    return ExecutionArgs(inspect.signature(fn), fn.__name__)


def read(slot, ctype):
    return ctype.from_address(slot.value).value


def halves(i8: m.Int8, u64: m.Uint64, f16: m.Float16, bf16: m.BFloat16, n):
    pass


def g(a: m.Int32, b: m.Int32 = 2, *, c: m.Int32 = 3):
    pass


def tiny(x: m.Float8E5M2):
    pass


u16 = ctypes.c_uint16
slots, _ = binder(halves).generate_execution_args((-5, 2**64 - 1, 1.5, 3.0, "x"), {})
i8, u64 = ctypes.c_int8, ctypes.c_uint64
print("SLOTS:", len(slots), read(slots[0], i8), read(slots[1], u64), end=" ")
print(hex(read(slots[2], u16)), hex(read(slots[3], u16)))
ea = binder(g)
print("BIND:", [int(v) for v in ea.get_rectified_args((1,), {"c": 9})], end=" ")
print(failure(lambda: ea.get_rectified_args((1,), {"z": 1})), end=" ")
print(failure(lambda: ea.get_rectified_args((1, 2), {"b": 5})), end=" ")
print(failure(lambda: binder(tiny).generate_execution_args((1.0,), {})))
# CHECK:      SLOTS: 4 -5 18446744073709551615 0x3e00 0x4040
# CHECK-NEXT: BIND: [1, 2, 9] CALL_ARGUMENTS CALL_ARGUMENTS ARG_UNSUPPORTED_TYPE

# --- scalars both ways -------------------------------------------
# `func.Jit` decodes the half types from their bit patterns
# (infinities, NaN and subnormals included); the widest integer and a half
# survive a round trip through compiled code.
from mlir.dsl.plugins.decorators.jit.func import _scalar_from_ctypes as dec

f16 = [dec(m.Float16, u16(bits)).value for bits in (0x3E00, 0xFC00, 0x0001)]
print(f"DECODE: {f16[:2]} {f16[2] == 2.0**-24}", end=" ")
print(math.isnan(dec(m.Float16, u16(0x7E00)).value), dec(m.BFloat16, u16(0x4040)).value)
# CHECK: DECODE: [1.5, -inf] True True 3.0


@m.jit
def id_u64(x: m.Uint64) -> m.Uint64:
    return x


@m.jit
def half_twice(x: m.Float16) -> m.Float16:
    return x * m.Float16(2.0)


print("ROUND_TRIP:", id_u64(2**64 - 1), half_twice(1.5), half_twice(65504.0))
# EXEC: ROUND_TRIP: 18446744073709551615 3.0 inf


# The `Pointer[T]` boundary: an array, an address, a `c_void_p` and a null
# pointer with `n == 0` all reach the same entry; the mistakes are diagnostics.
@m.jit
def fill(p: m.Pointer[m.Float32], n: m.Int32, v: m.Float32):
    for i in range(n):
        p[i] = v


buf = np.zeros(4, np.float32)
fill(buf, 4, 1.0)
fill(buf.ctypes.data, 2, 2.0)
fill(ctypes.c_void_p(buf.ctypes.data), 1, 3.0)
fill(0, 0, 9.0)
print("FILLED:", buf.tolist(), failure(lambda: fill(-1, 0, 0.0)), end=" ")
print(failure(lambda: fill(np.zeros(4, np.float64), 0, 0.0)), end=" ")
print(failure(lambda: fill(m.Pointer(0, dtype=m.Int32), 0, 0.0)))
# CHECK: FILLED: {{.*}} ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH
# EXEC:  FILLED: [3.0, 2.0, 1.0, 1.0] ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH ARG_ANNOTATION_MISMATCH


# --- the on-disk cache ---------------------------------------
# The lowered module of each module hash is written under `<PREFIX>_CACHE_DIR`
# as `mlir_dsl_<hash>.mlir`, bytecode plus a CRC32 trailer. The second process
# finds it, skips the pass pipeline and only builds the engine (a file hit is
# not an in-memory miss); a damaged entry is a miss with a logged warning and
# is rewritten valid; `DISABLE_FILE_CACHING` leaves the files alone. The
# second call in one process is an in-memory hit either way.
def entries():
    if not os.path.isdir(cache_dir):
        return []
    return sorted(p for p in os.listdir(cache_dir) if p.startswith("mlir_dsl_"))


if mode == "corrupt":
    for name in entries():
        path = os.path.join(cache_dir, name)
        with open(path, "r+b") as f:
            f.seek(os.path.getsize(path) // 2)
            byte = f.read(1)
            f.seek(-1, os.SEEK_CUR)
            f.write(bytes([byte[0] ^ 0xFF]))
    print("DAMAGED:", len(entries()) > 0)
# CORRUPT: DAMAGED: True


@m.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


before = (len(entries()), dsl.file_cache_hits, dsl.cache_misses, dsl.cache_hits)
seen = len(warnings)
results = twice(21), twice(4)
fh, mi = dsl.file_cache_hits - before[1], dsl.cache_misses - before[2]
mh = dsl.cache_hits - before[3]
print("FILE:", *results, "NEW_FILES:", len(entries()) - before[0], end=" ")
print(f"FILE_HITS: {fh} MISSES: {mi} MEM_HITS: {mh}", end=" ")
print("WARNINGS:", ["CRC32" in w for w in warnings[seen:]])
# MISS:    FILE: 42 8 NEW_FILES: 1 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: []
# CORRUPT: FILE: 42 8 NEW_FILES: 0 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: [True]
# HIT:     FILE: 42 8 NEW_FILES: 0 FILE_HITS: 1 MISSES: 0 MEM_HITS: 1 WARNINGS: []
# OFF:     FILE: 42 8 NEW_FILES: 0 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: []

# --- the in-memory cache -------------------------------------
# `JitCacheDict` maps module hash to compiled function: `max_elems` None is
# unlimited, N evicts least recently used (a `get` refreshes), 0 disables it;
# an entry tied to an `owner` goes when that object is collected.
# `<PREFIX>_JIT_CACHE_MAX_ELEMS` and `<PREFIX>_NO_CACHE` size the DSL's cache,
# so with one slot alternating between two functions recompiles every time.
lru = JitCacheDict(max_elems=2)
lru.set("a", "A")
lru.set("b", "B")
lru.get("a")
lru.set("c", "C")
print(f"LRU: {lru.get('a')} {lru.get('b')} {lru.get('c')} {len(lru)}")
body = lambda: None
tied = JitCacheDict()
tied.set("k", "K", owner=body)
alive = tied.get("k")
del body
gc.collect()
print("TIED:", alive, tied.get("k"), len(tied))
# CHECK:      LRU: A None C 2
# CHECK-NEXT: TIED: K None 0


@m.jit
def inc(n: m.Int32) -> m.Int32:
    return n + 1


@m.jit
def dec(n: m.Int32) -> m.Int32:
    return n - 1


before = (dsl.cache_hits, dsl.cache_misses)
results = inc(1), dec(1), inc(2), inc(3), dec(2)
hits, misses = dsl.cache_hits - before[0], dsl.cache_misses - before[1]
print("MEMORY:", *results, "MAX_ELEMS:", dsl.jit_cache.max_elems, end=" ")
print(f"HITS: {hits} MISSES: {misses} ENTRIES: {len(dsl.jit_cache)}")
# MISS: MEMORY: 2 0 3 4 1 MAX_ELEMS: None HITS: 3 MISSES: 2 ENTRIES: 6
# ONE:  MEMORY: 2 0 3 4 1 MAX_ELEMS: 1 HITS: 1 MISSES: 4 ENTRIES: 1
# ZERO: MEMORY: 2 0 3 4 1 MAX_ELEMS: 0 HITS: 0 MISSES: 5 ENTRIES: 0

# --- the pass pipeline as data ------------------------------
# `_get_pipeline`: the `pipeline=` call keyword as given; else
# `<PREFIX>_PIPELINE`, with the plugins' `pipeline_options()` (the gpu plugin's
# chip option when `<PREFIX>_ARCH` is set) merged into its option block; else
# the DSL's own `pipeline()`.
# A pipeline that does not parse is a DSLRuntimeError naming it, one that does
# not lower to the LLVM dialect fails when the engine is built, a failing pass
# quotes its diagnostics; the DSL is usable after each.
print("PIPELINE:", dsl._get_pipeline(None))
print("EXPLICIT:", dsl._get_pipeline("builtin.module(cse)"), end=" ")
NVVM = "builtin.module(gpu-lower-to-nvvm-pipeline)"
print(dsl.preprocess_pipeline(NVVM, {"cubin-chip": "sm_90"}), end=" ")
print(dsl.preprocess_pipeline("builtin.module(canonicalize)", {}))
# CHECK:        PIPELINE: builtin.module({{.*}}convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)
# CHECK-NEXT:   EXPLICIT: builtin.module(cse) builtin.module(gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90 }) builtin.module(canonicalize)
# ENVPIPE:      PIPELINE: builtin.module(canonicalize{top-down=true cubin-chip=sm_90 })
# ENVPIPE-NEXT: EXPLICIT: builtin.module(cse)


def attempt(label, fn):
    try:
        print(label, fn())
    except m.DSLRuntimeError as e:
        parse = "failed to parse the pass pipeline" in str(e)
        engine = "failed to create the execution engine" in str(e)
        print(label, "PARSE" if parse else "ENGINE" if engine else "OTHER", end=" ")
        print(e.context.get("pipeline", "-"))


core = dsl._get_pipeline(None)  # the DSL's own pipeline, given explicitly
attempt("EXPLICIT_GOOD:", lambda: twice(4, pipeline=core))
attempt("EXPLICIT_BAD:", lambda: twice(4, pipeline="builtin.module(no-such-pass)"))
attempt("EXPLICIT_NOLOWER:", lambda: twice(4, pipeline="builtin.module(cse)"))
attempt("AGAIN:", lambda: twice(5))
# EXEC: EXPLICIT_GOOD: 8
# EXEC: EXPLICIT_BAD: PARSE builtin.module(no-such-pass)
# EXEC: EXPLICIT_NOLOWER: ENGINE -
# EXEC: AGAIN: 10

compiler = execution_engine.Compiler()
BAD_SRC = (
    "module { func.func @g(%a: tensor<4xi32>) -> tensor<4xi32> {"
    " cf.br ^bb1(%a : tensor<4xi32>) ^bb1(%b: tensor<4xi32>): return %b : tensor<4xi32> } }"
)
with ir.Context() as ctx:
    try:
        compiler.compile(ir.Module.parse(BAD_SRC), "builtin.module(convert-cf-to-llvm)")
    except m.DSLRuntimeError as e:
        quoted = "failed to legalize operation 'cf.br'" in e.context["diagnostics"]
        print("PASS_FAIL:", e.context["pipeline"], quoted)
        print(str(e))
    empty = ir.Module.parse("module {}")
    print("OPT_LEVEL:", failure(lambda: compiler.jit(empty, opt_level=5)))
    bad_policy = dict(remark_filter=".*", remark_policy="sometimes")
    session = compiler.remark_session(ctx, **bad_policy)
    print("REMARK_CONFIG:", failure(session.__enter__))
# CHECK:      PASS_FAIL: builtin.module(convert-cf-to-llvm) True
# CHECK:      error[INTERNAL]:
# CHECK:      error:{{.*}}MLIR pass pipeline failed
# CHECK:      failed to legalize operation 'cf.br'
# CHECK:      OPT_LEVEL: CONFIG_INVALID
# CHECK-NEXT: REMARK_CONFIG: invalid remark configuration

# The compile cache key hashes the module bytecode with the pipeline and the
# extra libraries, not the function name.
with ir.Context():
    a = ir.Module.parse(
        "module { llvm.func @f(%a: i32) -> i32 { llvm.return %a : i32 } }"
    )
    base = dsl.get_module_hash(a, "f", pipeline="p")
    renamed = dsl.get_module_hash(a, "other_name", pipeline="p") == base
    print(f"HASH: {len(base)}", renamed, end=" ")
    print(dsl.get_module_hash(a, "f", pipeline="q") != base, end=" ")
    print(dsl.get_module_hash(a, "f", pipeline="p", extra_link_libs=("/x",)) != base)
# CHECK: HASH: 64 True True True


# --- IR dumps ---------------------------------------------------
# `KEEP_IR` saves the traced module as `<CACHE_DIR>/<function>.mlir` before
# any pass (`DEBUGINFO` adds locations) and records the path;
# `KEEPIR_AFTER_PASSES` runs the given passes and saves
# `<function>_after_pass.mlir`; `PRINT_IR_AFTER_PASSES` prints the result of
# the passes on a clone to stderr. All work under DRYRUN, which never reaches
# the compiler; without them nothing is written (the first RUN line).
@m.jit
def cumsum(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


print("CUMSUM:", cumsum(10))
# PRINT:     //===--- IR after passes: convert-scf-to-cf ---===
# PRINT:     func.func @cumsum(
# PRINT:     cf.cond_br
# PRINT-NOT: scf.for
# PRINT:     //===--- End of IR after passes ---===
# PRINT:     CUMSUM: ?
# EXEC:      CUMSUM: 45
listing = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
dumps = sorted(f for f in listing if f.startswith("cumsum"))
print("DUMPS:", dumps, os.path.basename(dsl.dump_mlir_path or "none"))
for name in dumps:
    text = open(os.path.join(cache_dir, name)).read()
    print(f"{name} scf.for: {'scf.for' in text} cf: {'cf.cond_br' in text}", end=" ")
    print(f"loc: {'loc(' in text} entry: {'func.func @cumsum(' in text}")
# CHECK:      DUMPS: [] none
# KEEP:       DUMPS: ['cumsum.mlir'] cumsum.mlir
# KEEP-NEXT:  cumsum.mlir scf.for: True cf: False loc: True entry: True
# AFTER:      DUMPS: ['cumsum_after_pass.mlir'] none
# AFTER-NEXT: cumsum_after_pass.mlir scf.for: False cf: True loc: False entry: True
