# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ rm -rf %t.cache && env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# RUN: %if host-supports-jit %{ env MLIR_DSL_REMARKS=".*" %PYTHON %s 2>&1 | FileCheck %s --check-prefix=REMARKS %}
# BaseDSL: the `@jit`/`@kernel` decorators and the
# singleton DSL instance (abstract bases included); the decorator hooks
# (`decorate`/`wrap`, `call_scope`, the rewrite key, the trace hooks); Meta
# specialisation and `mangle_name`; the host
# boundary (arguments restored in their Python shape, results packed into one
# `!llvm.struct` and unpacked again); `m.compile`; the `func.func` host entry; the
# `Plugins` record and the `Plugin` hooks; the in-memory cache counters and
# `collected_remarks`. Every error path is a DSL diagnostic. The EXEC line
# starts from an empty file cache so the counters are deterministic; the third
# RUN line turns the remark engine on for the `collected_remarks` checks.
import contextlib
import dataclasses
import gc
import inspect
import os

import numpy as np

import mlir.mlir_dsl as m
from mlir import ir
from mlir.dsl.plugins.compiler import execution_engine
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import arith, llvm, vector
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps

dsl = m.MlirTestDSL()


def report(label, fn, *args, **kwargs):
    """Print a call's result, or the DiagId of the DSL error it raises."""
    try:
        result = fn(*args, **kwargs)
    except m.DSLUserCodeError as e:
        print(label, "ERROR:", e.diag_id.name)
        return e
    print(label, "RESULT:", result)
    return result


# --- the decorators and the singleton ---------------------------------------
# The bare and the parenthesised forms; a `@jit` function called inside a
# trace is inlined (one entry, no `llvm.call`); a DSL that names no
# `ast_preprocessor` plugin leaves the code object alone (the rewrite has no
# per-function switch). One DSL instance per concrete subclass, `BaseDSL()`
# is the first one made.
@m.jit
def helper(a: m.Int32) -> m.Int32:
    return a * 3


@m.jit()
def outer(n: m.Int32) -> m.Int32:
    return helper(n) + 1


class NoRewriteDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(m.MlirTestDSL.plugins, ast_preprocessor=None)


@NoRewriteDSL.jit
def plus_two(n: m.Int32) -> m.Int32:
    return n + 2


class OtherDSL(m.MlirTestDSL):
    pass


# CHECK-LABEL: func.func @outer(
# CHECK-SAME:    %[[N:[^:]+]]: i32) -> i32
# CHECK-NOT:     llvm.call
# CHECK:         %[[M:.+]] = arith.muli %[[N]], %{{.+}} : i32
# CHECK:         %[[R:.+]] = arith.addi %[[M]], %{{.+}} : i32
# CHECK:         return %[[R]] : i32
# CHECK:         NESTED: ?
# EXEC:          NESTED: 7
print("NESTED:", outer(2))
code_before = plus_two.__wrapped__.__code__
print("NO_PREPROCESS:", plus_two(1), plus_two.__wrapped__.__code__ is code_before)
print(
    "SINGLETON:", m.MlirTestDSL() is dsl, m.BaseDSL() is dsl, OtherDSL() is OtherDSL()
)
# CHECK: NO_PREPROCESS: ? True
# CHECK: SINGLETON: True True True
# EXEC:  NO_PREPROCESS: 3 True


# An abstract base (`abstract=True`) is never instantiated itself: calling it
# returns the instance of one of its concrete subclasses (never its parent's
# or a sibling's), and before any exists it is an error.
class Family(m.MlirTestDSL, abstract=True):
    pass


class Concrete(Family):
    pass


try:
    Family()
    print("ABSTRACT_FIRST: instance (unexpected)")
except m.DSLRuntimeError as e:
    print("ABSTRACT_FIRST:", e.message)
concrete = Concrete()
print(
    "ABSTRACT:",
    Family() is concrete,
    Family() is not dsl,
    Family._abstract_dsl_base,
    Concrete._abstract_dsl_base,
)
# CHECK: ABSTRACT_FIRST: `Family` is an abstract DSL base: initialize a concrete subclass of it first
# CHECK: ABSTRACT: True True True False


# Only a plain Python function can be decorated; `@kernel` is inherited from
# MlirTestDSL, but using it needs the kernels decorator plugin in the record.
class JitOnlyDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(m.MlirTestDSL.plugins, decorators=[func.Jit()])


@JitOnlyDSL.kernel
def kernel_without_plugin(n: m.Int32):
    pass


report("NOT_FUNCTION", m.jit(), 3)
print(str(report("NO_PLUGIN_KERNEL", kernel_without_plugin, 1)))
# CHECK:      NOT_FUNCTION ERROR: CALL_NOT_CALLABLE
# CHECK-NEXT: NO_PLUGIN_KERNEL ERROR: CALL_PLUGIN_REQUIRED
# CHECK:      error[CALL_PLUGIN_REQUIRED]:{{.*}}`@kernel` needs the `Kernels` plugin, which this DSL does not name.
# CHECK:      suggestion:{{.*}}Name it: `plugins = Plugins(decorators=[Kernels()])`.


# The record owns the decorators: a record without a decorator plugin has no
# such attribute at all (an AttributeError that names the fix), two plugins
# cannot add the same decorator, and `without` drops a family member by name.
class NoDecorators(m.BaseDSL):
    plugins = m.Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith))


try:
    NoDecorators.jit
except AttributeError as e:
    print("NO_JIT:", str(e)[:78])
try:
    m.Plugins(decorators=[func.Jit(), func.Jit()])
except m.DSLRuntimeError as e:
    print("TWICE:", e.message)
print("WITHOUT:", [p.name for p in m.MlirTestDSL.plugins.without("gpu").decorators])
# CHECK: NO_JIT: `NoDecorators` has no attribute `jit`; its record names no decorator plugin
# CHECK: TWICE: `Jit` and `Jit` both add the decorator `@jit`; a record has one plugin per decorator
# CHECK: WITHOUT: ['func']

# `@jit(3)`: the decorators take no option, positional or keyword; a `plugins`
# attribute that is not a `Plugins` record is refused when the class is made.
try:
    m.jit(3)
    print("POSITIONAL: accepted (unexpected)")
except m.DSLUserCodeError as e:
    print("POSITIONAL:", e.diag_id.name)
try:

    class NotARecord(m.BaseDSL):
        plugins = [func.Jit()]

    print("NOT_A_RECORD: accepted (unexpected)")
except m.DSLRuntimeError as e:
    print("NOT_A_RECORD:", e.message)
# CHECK: POSITIONAL: CALL_ARGUMENTS
# CHECK: NOT_A_RECORD: `NotARecord.plugins` must be a `Plugins` record, not a `list`


# --- decorator options and the wrapper ---------------------------------------
# The options of a decorator are its plugin's: `decorate` takes them once the
# core stamped the function (the default takes none), and `wrap` sees the
# wrapper the user gets and may give it an API. The wrapper closes over the
# DSL class and the function only, no bound method, and looks the plugin up
# at call time.
class OptJit(func.Jit):
    def decorate(self, func, *, label=None):
        func._label = label
        return func

    def wrap(self, func, wrapper):
        def rename(prefix):
            func._symbol_prefix = prefix
            return wrapper

        wrapper.rename = rename
        return wrapper


class OptDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(m.MlirTestDSL.plugins, decorators=[OptJit()])

    def mangle_name(self, function_name, args, sig):
        prefix = getattr(self.traced_function, "_symbol_prefix", "")
        return super().mangle_name(prefix + function_name, args, sig)


@OptDSL.jit(label="renamed")
def f(n: m.Int32) -> m.Int32:
    return n + 1


def plain_option(n):
    pass


closure = inspect.getclosurevars(f).nonlocals
print(
    "OPTIONS:",
    f.__wrapped__._label,
    sorted(closure),
    any(map(inspect.ismethod, closure.values())),
)
print("RENAMED:", f.rename("renamed_")(1))
print(str(report("UNKNOWN_OPTION", m.jit(bogus=1), plain_option)))
# CHECK:       OPTIONS: renamed ['cls', 'func'] False
# CHECK-LABEL: func.func @renamed_f(
# CHECK:       RENAMED: ?
# CHECK:       UNKNOWN_OPTION ERROR: CALL_ARGUMENTS
# CHECK:       error[CALL_ARGUMENTS]:{{.*}}The call to `@jit` does not match its parameters: no option is named `bogus`.
# EXEC:        RENAMED: 2


# `call_scope` is the DSL's context around a call made outside any trace: the
# rewrite, the trace, the compile and the run happen inside it. A decorated
# function called inside the trace is launched into it and enters no scope of
# its own.
class ScopedDSL(m.MlirTestDSL):
    @classmethod
    @contextlib.contextmanager
    def call_scope(cls, func):
        print("SCOPE: enter", func.__name__)
        try:
            yield
        finally:
            print("SCOPE: exit", func.__name__)


@ScopedDSL.jit
def scoped_inner(n: m.Int32) -> m.Int32:
    return n * 2


@ScopedDSL.jit
def scoped_outer(n: m.Int32) -> m.Int32:
    return scoped_inner(n) + 1


# CHECK:       SCOPE: enter scoped_outer
# CHECK-NOT:   SCOPE:
# CHECK-LABEL: func.func @scoped_outer(
# CHECK-NOT:   SCOPE: enter
# CHECK:       SCOPE: exit scoped_outer
# CHECK-NEXT:  SCOPED: ?
# EXEC:        SCOPED: 7
print("SCOPED:", scoped_outer(3))


# A function is rewritten once per `preprocess_key`: a DSL whose rewrite has
# modes returns a key that changes with them, and the function is rewritten
# again from its original code; the same key leaves the code object alone.
REWRITE_MODE = ["a"]


class KeyedDSL(m.MlirTestDSL):
    def preprocess_key(self, func):
        return (super().preprocess_key(func), REWRITE_MODE[0])


@KeyedDSL.jit
def keyed(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


original_code = keyed.__wrapped__.__code__
keyed(3)
first_rewrite = keyed.__wrapped__.__code__
keyed(3)
same_key = keyed.__wrapped__.__code__
REWRITE_MODE[0] = "b"
keyed(3)
second_rewrite = keyed.__wrapped__.__code__
print(
    "REWRITE_KEY:",
    first_rewrite is not original_code,
    same_key is first_rewrite,
    second_rewrite is not first_rewrite,
    keyed.__wrapped__._original_code is original_code,
)
# CHECK: REWRITE_KEY: True True True True


# --- the trace hooks ------------------------------------------------------
# An entry with a `body_scope` wraps the traced body (arguments bound, the
# insertion point in the entry block); `_return_values` and `after_trace` know
# the function they finish; a call's `trace_hooks=` run on the finished module
# after the adapters added to it, outside the module's insertion point, and
# what they add is part of the hashed module.
class ScopedJit(func.Jit):
    @contextlib.contextmanager
    def body_scope(self, dsl, func, *, signature):
        print("BODY: enter", func.__name__, list(signature.parameters))
        yield
        print("BODY: exit", func.__name__)

    def after_trace(self, dsl, module, function_name):
        print("AFTER_TRACE:", function_name)


class TraceHooksDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(m.MlirTestDSL.plugins, decorators=[ScopedJit()])

    def _return_values(self, entry, result, sig, loc, *, func):
        print("RETURN_OF:", func.__name__, type(result).__name__)
        return super()._return_values(entry, result, sig, loc, func=func)


@TraceHooksDSL.jit
def hooked(n, k: m.Int32) -> m.Int32:
    print("BODY: run")
    return k + n


def stamp(dsl, module, function_name):
    try:
        ir.InsertionPoint.current
        print("STAMP: inside an insertion point (unexpected)")
    except ValueError:
        print("STAMP:", function_name, "outside the insertion point")
    module.operation.attributes["test.traced"] = ir.StringAttr.get(function_name)


# CHECK:       BODY: enter hooked ['n', 'k']
# CHECK-NEXT:  BODY: run
# CHECK-NEXT:  BODY: exit hooked
# CHECK-NEXT:  RETURN_OF: hooked Int32
# CHECK-NEXT:  AFTER_TRACE: hooked_3
# CHECK-NEXT:  STAMP: hooked_3 outside the insertion point
# CHECK:       module attributes {test.traced = "hooked_3"}
# CHECK-LABEL: func.func @hooked_3(
# CHECK:       HOOKED: ?
# CHECK:       BAD_HOOK ERROR: CALL_ARGUMENTS
# EXEC:        HOOKED: 8
print("HOOKED:", hooked(3, 5, trace_hooks=[stamp]))
print(str(report("BAD_HOOK", hooked, 3, 5, trace_hooks=[1])))


# --- Meta specialisation and `mangle_name` -----------------------
# Only the Meta arguments are folded into the symbol (a Python value under no
# DSL annotation, a default, a type by its name, a sequence element-wise); a
# staged argument (annotated, a DSL value, a buffer) leaves the name alone;
# unwanted characters and hex addresses are stripped, the name is capped at
# 180 characters and `_name_mangling_prefix` is prepended.
def one(a):
    pass


def two(a, b: m.Int32):
    pass


class PrefixedDSL(m.MlirTestDSL):
    _name_mangling_prefix = "pfx"


class Knob:
    pass


class FoldAllDSL(m.MlirTestDSL):
    # Every argument names the symbol, annotated scalars included (`_mangle_folds`),
    # and an instance of a user class is spelled by its class name (`_mangle_render`).
    def _mangle_folds(self, arg, annotation):
        return True

    def _mangle_render(self, arg, annotation):
        if isinstance(arg, Knob):
            return type(arg).__name__
        return super()._mangle_render(arg, annotation)


ONE = inspect.signature(one)
meta = [4, "hi there", (1, (2, 3)), m.Float32, None, True, 2.5]
print("META:", [dsl.mangle_name("one", (v,), ONE) for v in meta])
staged = [m.Float32(1.5), np.zeros(2, np.float32), (1, m.Int32(2))]
print("STAGED:", [dsl.mangle_name("one", (v,), ONE) for v in staged])
print("ANNOTATED:", dsl.mangle_name("two", (4, 3), inspect.signature(two)))
messy = "a'b-c[d]#e,f.g<h>i(j)k\"l:m{n}o=p%q?r@s;t/u"
print("CHARS:", dsl.mangle_name("one", (messy,), ONE), end=" ")
capped = len(dsl.mangle_name("one", ("x" * 500,), ONE))
print(dsl.mangle_name("one", (object(),), ONE), capped)
print("PREFIX:", PrefixedDSL().mangle_name("one", (7,), ONE))
fold_all = FoldAllDSL()
print(
    "FOLD_ALL:",
    fold_all.mangle_name("two", (4, 3), inspect.signature(two)),
    fold_all.mangle_name("one", (Knob(),), ONE),
)
# CHECK:      META: ['one_4', 'one_hi_there', 'one_1_2_3', 'one_Float32', 'one_None', 'one_True', 'one_25']
# CHECK-NEXT: STAGED: ['one', 'one', 'one']
# CHECK-NEXT: ANNOTATED: two_4
# CHECK-NEXT: CHARS: one_abcdefghijklmnopqrst_u one_object_object_at_ 180
# CHECK-NEXT: PREFIX: pfx_one_7
# CHECK-NEXT: FOLD_ALL: two_4_3 one_Knob


# The symbol of a traced entry follows the DSL's folding rule: the staged
# scalar `v` keeps its block argument and still names the specialisation.
@FoldAllDSL.jit
def folded(v: m.Int32, knob) -> m.Int32:
    return v + 1


# CHECK-LABEL: func.func @folded_5_Knob(
# CHECK-SAME:    %{{[^:]+}}: i32) -> i32
# CHECK:         FOLDED: ?
# EXEC:          FOLDED: 6
print("FOLDED:", folded(5, Knob()))


# The traced entries follow the same rules (a default is a Meta argument too):
# one symbol and one cache entry per specialisation, a repeat is a hit, and a
# `no_cache=True` call compiles without touching the cache.
@m.jit
def unrolled(n, k=2) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i * k
    return acc


def counters(label, before, *results):
    hits, misses = dsl.cache_hits - before[0], dsl.cache_misses - before[1]
    cached = len(dsl.jit_cache) - before[2]
    print(label, *results, f"HITS: {hits} MISSES: {misses} CACHED: {cached}")


# CHECK-LABEL: func.func @unrolled_3_2() -> i32
# CHECK-LABEL: func.func @unrolled_4_2() -> i32
# CHECK-LABEL: func.func @unrolled_3_5() -> i32
# EXEC:        SPECIALISED: 6 12 15 6 HITS: 1 MISSES: 3 CACHED: 3
# EXEC-NEXT:   NO_CACHE: 6 HITS: 0 MISSES: 1 CACHED: 0
gc.collect()
before = (dsl.cache_hits, dsl.cache_misses, len(dsl.jit_cache))
runs = unrolled(3), unrolled(4), unrolled(3, k=5), unrolled(3)
counters("SPECIALISED:", before, *runs)
gc.collect()
before = (dsl.cache_hits, dsl.cache_misses, len(dsl.jit_cache))
counters("NO_CACHE:", before, unrolled(3, no_cache=True))


# --- the host boundary: arguments --------------------------------
# The entry's block arguments reach the body in the shape of the Python
# arguments: an annotated numeric is wrapped in its type (a Python value is
# cast first), a tuple, a frozen dataclass and a `@struct` (one block argument
# per field) are rebuilt leaf by
# leaf, a Meta argument passes through and is folded into the name, keyword-
# only arguments follow the positional ones.
@dataclasses.dataclass(frozen=True)
class Cfg:
    scale: m.Int32
    name: str


@m.struct
class Pair:
    a: m.Int32
    b: m.Float32


@m.jit
def restore(a: m.Int32, pair: tuple, cfg: Cfg, flag, p: Pair, *, k: m.Int32):
    kinds = [type(x).__name__ for x in (a, *pair, cfg.scale, p, k)]
    print("TRACE:", kinds, cfg.name, flag)
    return a + pair[0] + cfg.scale + p.a + k


# CHECK:       TRACE: ['Int32', 'Int32', 'Float32', 'Int32', 'Pair', 'Int32'] n None
# CHECK-LABEL: func.func @restore_None(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[P0:[^:]+]]: i32, %[[P1:[^:]+]]: f32, %[[S:[^:]+]]: i32, %[[PA:[^:]+]]: i32, %[[PB:[^:]+]]: f32, %[[K:[^:]+]]: i32) -> i32
# CHECK:         %[[T0:.+]] = arith.addi %[[A]], %[[P0]] : i32
# CHECK:         %[[T1:.+]] = arith.addi %[[T0]], %[[S]] : i32
# CHECK:         %[[T2:.+]] = arith.addi %[[T1]], %[[PA]] : i32
# CHECK:         %[[T3:.+]] = arith.addi %[[T2]], %[[K]] : i32
# CHECK:         return %[[T3]] : i32
# CHECK:         RESTORED: ?
# EXEC:          RESTORED: 1113
cfg = Cfg(scale=m.Int32(10), name="n")
pair = (m.Int32(2), m.Float32(3.0))
print("RESTORED:", restore(1, pair, cfg, None, Pair(a=100, b=1.0), k=1000))


# A value the annotation cannot take is a diagnostic naming the parameter.
@m.jit
def scaled(x: m.Float32, n: m.Int64) -> m.Float32:
    return x * m.Float32(n)


print("CAST:", scaled(2.5, 3))
print(str(report("CAST", scaled, "two", 3)))
# CHECK:      CAST: ?
# EXEC:       CAST: 7.5
# CHECK:      CAST ERROR: ARG_ANNOTATION_MISMATCH
# CHECK:      error[ARG_ANNOTATION_MISMATCH]:{{.*}}Argument #1 `x` must be a `Float32`, but got a `str`.
# EXEC:       CAST ERROR: ARG_ANNOTATION_MISMATCH


# An abstract annotation (`m.Numeric`, `m.Integer`, `m.Float`) stages a DSL
# value of the family as it is: one definition, one compiled function per
# dtype. A Python number takes the dtype the DSL's `type_ops` plugin assigns
# (`infer_dtype`; the shipped one: the literal rule, every number `Float32`
# under `m.Float`, no dtype for a float under `m.Integer`). The core has no
# opinion of its own, so a number the plugin assigns no dtype is a diagnostic.
@m.jit
def generic(a: m.Numeric) -> m.Numeric:
    return a + a


@m.jit
def integral(a: m.Integer) -> m.Integer:
    return a + a


@m.jit
def floating(a: m.Float) -> m.Float:
    return a + a


# CHECK-LABEL: func.func @generic(
# CHECK-SAME:    %{{[^:]+}}: i16) -> i16
# CHECK:       GENERIC_I16 RESULT: ?
# CHECK-LABEL: func.func @generic(
# CHECK-SAME:    %{{[^:]+}}: i32) -> i32
# CHECK:       GENERIC_INT RESULT: ?
# CHECK-LABEL: func.func @generic(
# CHECK-SAME:    %{{[^:]+}}: f32) -> f32
# CHECK:       GENERIC_FLOAT RESULT: ?
# CHECK-LABEL: func.func @generic(
# CHECK-SAME:    %{{[^:]+}}: i1) -> i1
# CHECK:       GENERIC_BOOL RESULT: ?
# CHECK-LABEL: func.func @integral(
# CHECK-SAME:    %{{[^:]+}}: i32) -> i32
# CHECK:       INTEGRAL_INT RESULT: ?
# CHECK-LABEL: func.func @floating(
# CHECK-SAME:    %{{[^:]+}}: f32) -> f32
# CHECK:       FLOATING_INT RESULT: ?
# CHECK-LABEL: func.func @floating(
# CHECK-SAME:    %{{[^:]+}}: f16) -> f16
# CHECK:       FLOATING_F16 RESULT: ?
# CHECK:       INTEGRAL_FLOAT ERROR: ARG_ANNOTATION_MISMATCH
# CHECK:       error[ARG_ANNOTATION_MISMATCH]:{{.*}}Argument #1 `a` must be a `Integer`, but got a `float`.
# CHECK:       GENERIC_STR ERROR: ARG_ANNOTATION_MISMATCH
# EXEC:        GENERIC_I16 RESULT: 4
# EXEC-NEXT:   GENERIC_INT RESULT: 6
# EXEC-NEXT:   GENERIC_FLOAT RESULT: 5.0
# EXEC-NEXT:   GENERIC_BOOL RESULT: True
# EXEC-NEXT:   INTEGRAL_INT RESULT: 6
# EXEC-NEXT:   FLOATING_INT RESULT: 6.0
# EXEC-NEXT:   FLOATING_F16 RESULT: 3.0
# EXEC-NEXT:   INTEGRAL_FLOAT ERROR: ARG_ANNOTATION_MISMATCH
# EXEC:        GENERIC_STR ERROR: ARG_ANNOTATION_MISMATCH
report("GENERIC_I16", generic, m.Int16(2))
report("GENERIC_INT", generic, 3)
report("GENERIC_FLOAT", generic, 2.5)
report("GENERIC_BOOL", generic, True)
report("INTEGRAL_INT", integral, 3)
report("FLOATING_INT", floating, 3)
report("FLOATING_F16", floating, m.Float16(1.5))
print(str(report("INTEGRAL_FLOAT", integral, 2.5)))
print(str(report("GENERIC_STR", generic, "x")))


# Staging is decided by the annotation and the value, for every DSL built on
# BaseDSL: an unannotated Python value is a Meta argument (there is no
# `Constexpr` annotation and no per-DSL knob) and a DSL value stages under any
# annotation, so a bare `BaseDSL` behaves like `MlirTestDSL`.
class StrictDSL(m.BaseDSL):
    plugins = m.Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector, memory=llvm),
        compiler=execution_engine.Compiler(),
        decorators=[func.Jit()],
    )

    def pipeline(self):
        return list(m.LOWER_TO_LLVM)

    def __init__(self):
        super().__init__(name="MLIR_DSL")


@StrictDSL.jit
def strict(n):
    return m.Int32(1)


print(str(report("STRICT", strict, 4)))
# CHECK: STRICT RESULT: 1


# --- the host boundary: hooks ------------------------------------
# A DSL with a policy of its own overrides three hooks together:
# `_is_meta_at_boundary` says whether an argument is a Python value or a
# runtime value (the default: the annotation and the value decide; the symbol
# follows the verdict, so a runtime value is not folded into the name),
# `_flatten_jit_arg` gives a runtime value its SSA types and execution values,
# `_restore_argument` rebuilds it from the entry's block arguments. This DSL
# makes an unannotated `int` a runtime `i32` and knows `Span`, a class of the
# user's the core has no leaf for, as two `i32`s.
class Span:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi


class RuntimeByDefaultDSL(m.MlirTestDSL):
    def _is_meta_at_boundary(self, arg, spec, name, idx, func):
        if isinstance(arg, Span) or (
            type(arg) is int and spec is inspect.Parameter.empty
        ):
            return False
        return super()._is_meta_at_boundary(arg, spec, name, idx, func)

    def _flatten_jit_arg(self, arg, arg_name, arg_index, function_name, *, is_host):
        if isinstance(arg, Span):
            arg = (m.Int32(arg.lo), m.Int32(arg.hi))
        elif type(arg) is int:
            arg = m.Int32(arg)
        return super()._flatten_jit_arg(
            arg, arg_name, arg_index, function_name, is_host=is_host
        )

    def _restore_argument(self, arg, spec, block_args, start):
        if isinstance(arg, Span):
            return Span(m.Int32(block_args[start]), m.Int32(block_args[start + 1])), 2
        if type(arg) is int:
            return m.Int32(block_args[start]), 1
        return super()._restore_argument(arg, spec, block_args, start)


@RuntimeByDefaultDSL.jit
def runtime_int(n):
    return n + 1


@RuntimeByDefaultDSL.jit
def span_len(s: Span, k):
    print("SPAN_TRACE:", type(s).__name__, type(s.lo).__name__, type(k).__name__)
    return s.hi - s.lo + k


# CHECK-LABEL: func.func @runtime_int(
# CHECK-SAME:    %[[N:[^:]+]]: i32) -> i32
# CHECK:         arith.addi %[[N]], %{{.+}} : i32
# CHECK:         RUNTIME_INT: ?
# CHECK:         SPAN_TRACE: Span Int32 Int32
# CHECK-LABEL: func.func @span_len(
# CHECK-SAME:    %[[LO:[^:]+]]: i32, %[[HI:[^:]+]]: i32, %[[K:[^:]+]]: i32) -> i32
# CHECK:         %[[D:.+]] = arith.subi %[[HI]], %[[LO]] : i32
# CHECK:         arith.addi %[[D]], %[[K]] : i32
# CHECK:         SPAN: ?
# EXEC:          RUNTIME_INT: 4
# EXEC-NEXT:     SPAN_TRACE: Span Int32 Int32
# EXEC-NEXT:     SPAN: 107
print("RUNTIME_INT:", runtime_int(3))
print("SPAN:", span_len(Span(3, 10), 100))


# --- the host boundary: results ----------------------------------
# Several leaves (a tuple, a `@struct`, a nest of them) are packed into one
# `!llvm.struct` (`llvm.mlir.undef` + `llvm.insertvalue`) and unpacked into the
# same Python shape on the host, Meta slots restored verbatim; no return is a
# void entry; no value under an annotation and a `Pointer` result are errors.
@m.jit
def ret_tuple(n: m.Int32) -> tuple:
    return n + 1, m.Float32(n) * 0.5


@m.jit
def ret_nested(n: m.Int32):
    return (Pair(a=n, b=m.Float32(1.5)), (n * 3, m.Boolean(n > 2)), 5, None)


@m.jit
def ret_void(n: m.Int32):
    x = n + 1


@m.jit
def ret_none_annotated(n: m.Int32) -> m.Int32:
    pass


@m.jit
def ret_pointer(p: m.Pointer[m.Float32]):
    return p


# CHECK-LABEL: func.func @ret_tuple(
# CHECK-SAME:    %{{[^:]+}}: i32) -> !llvm.struct<(i32, f32)>
# CHECK:         %[[U:.+]] = llvm.mlir.undef : !llvm.struct<(i32, f32)>
# CHECK:         %[[I0:.+]] = llvm.insertvalue %{{.+}}, %[[U]][0] : !llvm.struct<(i32, f32)>
# CHECK:         %[[I1:.+]] = llvm.insertvalue %{{.+}}, %[[I0]][1] : !llvm.struct<(i32, f32)>
# CHECK:         return %[[I1]] : !llvm.struct<(i32, f32)>
# CHECK-LABEL: func.func @ret_nested(
# CHECK-SAME:    %{{[^:]+}}: i32) -> !llvm.struct<(i32, f32, i32, i1)>
# CHECK-LABEL: func.func @ret_void(
# CHECK-SAME:    %{{[^:]+}}: i32) attributes {llvm.emit_c_interface} {
# CHECK:         return
# CHECK:         VOID: None
# EXEC:          TUPLE: (Int32(5), Float32(2.0))
# EXEC-NEXT:     NESTED: (Pair(a=Int32(4), b=Float32(1.5)), (Int32(12), Boolean(True)), 5, None)
# EXEC-NEXT:     VOID: None
print("TUPLE:", repr(ret_tuple(4)))
print("NESTED:", repr(ret_nested(4)))
print("VOID:", ret_void(1))
print(str(report("NONE", ret_none_annotated, 1)))
print(str(report("POINTER", ret_pointer, np.zeros(2, np.float32))))
# CHECK: NONE ERROR: TYPE_RETURN_MISMATCH
# CHECK: error[TYPE_RETURN_MISMATCH]:{{.*}}This function returns `None`, which a compiled function cannot return while it declares a return type
# CHECK: suggestion:{{.*}}If the function returns nothing, remove its return type annotation.
# CHECK: POINTER ERROR: TYPE_RETURN_MISMATCH
# CHECK: error[TYPE_RETURN_MISMATCH]:{{.*}}This function returns a `Pointer`, which a compiled function cannot return (a `Pointer` result is memory: write through the pointer instead).
# EXEC: NONE ERROR: TYPE_RETURN_MISMATCH
# EXEC: POINTER ERROR: TYPE_RETURN_MISMATCH


# --- `m.compile` ------------------------------------------------
# Compiles for representative arguments without running and returns the
# compiled function: callable with matching arguments (positional or keyword),
# a Meta argument baked in, the in-memory cache left alone; the compiled
# function checks its own call; a function without `@jit` is rejected. DRYRUN
# returns the trace result instead, so this section runs on the EXEC line.
@m.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


def plain(n: m.Int32) -> m.Int32:
    return n


@dataclasses.dataclass(frozen=True)
class Scaled:
    n: m.Int32
    k: int  # a Meta field: part of the shape the function is compiled for


@m.jit
def scaled(s: Scaled) -> m.Int32:
    return s.n * s.k


if not dsl.envar.dryrun:
    # Entries die with their Python function; collect first so an unrelated
    # finalizer cannot run inside the window this section measures.
    gc.collect()
    entries = len(dsl.jit_cache)
    fn = m.compile(twice, 5)
    three = m.compile(unrolled, 3)
    print("COMPILE:", type(fn).__name__, fn.function_name, fn(5), fn(n=8))
    print("COMPILE_META:", three.function_name, three(3), len(dsl.jit_cache) - entries)
    report("TOO_MANY", fn, 1, 2)
    report("MISSING", fn)
    report("MISMATCH", fn, "five")
    print(str(report("PLAIN", m.compile, plain, 1)))
    # A compiled function remembers the shape of each argument (containers,
    # Meta fields, leaf types): a record with another Meta value or a tuple in
    # place of the record is a diagnostic, not a wrong call.
    by_three = m.compile(scaled, Scaled(2, 3))
    report("SHAPE_OK", by_three, Scaled(5, 3))
    report("SHAPE_META", by_three, Scaled(5, 4))
    report("SHAPE_TUPLE", by_three, (5, 3))
    # An abstract annotation pins the compiled function to the dtype the trace
    # chose: a Python number takes that dtype, a DSL value of another dtype of
    # the family is a diagnostic, not a mis-marshalled call.
    gen_i32 = m.compile(generic, 0)
    report("ABSTRACT_INT", gen_i32, 5)
    report("ABSTRACT_SAME", gen_i32, m.Int32(7))
    report("ABSTRACT_OTHER", gen_i32, m.Int16(1))
    report("ABSTRACT_FLOAT", gen_i32, m.Float32(1.5))
# EXEC:      COMPILE: JitCompiledFunction twice 10 16
# EXEC-NEXT: COMPILE_META: unrolled_3_2 6 0
# EXEC-NEXT: TOO_MANY ERROR: CALL_ARGUMENTS
# EXEC-NEXT: MISSING ERROR: CALL_ARGUMENTS
# EXEC-NEXT: MISMATCH ERROR: ARG_ANNOTATION_MISMATCH
# EXEC-NEXT: PLAIN ERROR: CALL_MISSING_JIT_DECORATOR
# EXEC:      error[CALL_MISSING_JIT_DECORATOR]:{{.*}}The function passed to `compile()` is a plain Python function
# EXEC:      suggestion:{{.*}}Add `@jit` above the definition of the function you pass to `compile()`.
# EXEC:      SHAPE_OK RESULT: 15
# EXEC-NEXT: SHAPE_META ERROR: ARG_ANNOTATION_MISMATCH
# EXEC-NEXT: SHAPE_TUPLE ERROR: ARG_ANNOTATION_MISMATCH
# EXEC-NEXT: ABSTRACT_INT RESULT: 10
# EXEC-NEXT: ABSTRACT_SAME RESULT: 14
# EXEC-NEXT: ABSTRACT_OTHER ERROR: ARG_ANNOTATION_MISMATCH
# EXEC-NEXT: ABSTRACT_FLOAT ERROR: ARG_ANNOTATION_MISMATCH

# A dtype class for a staged numeric parameter is `compile()`'s type-only
# placeholder: the function is compiled for the type, no value is marshalled,
# and it is called with values later (an abstract annotation is pinned to the
# class). The class must belong to the annotation, and outside `compile()` a
# class is a mismatched argument like any other value.
fn = m.compile(twice, m.Int32)
gen16 = m.compile(generic, m.Int16)
if dsl.envar.dryrun:
    print("PLACEHOLDER: ?")
else:
    print("PLACEHOLDER:", fn.function_name, fn(21), gen16(m.Int16(2)), gen16(3))
    report("PLACEHOLDER_OTHER", gen16, m.Int32(1))
print(str(report("PLACEHOLDER_MISMATCH", m.compile, twice, m.Float32)))
report("PLACEHOLDER_CALL", twice, m.Int32)
# CHECK-LABEL: func.func @twice(
# CHECK-SAME:    %{{[^:]+}}: i32) -> i32
# CHECK-LABEL: func.func @generic(
# CHECK-SAME:    %{{[^:]+}}: i16) -> i16
# CHECK:       PLACEHOLDER: ?
# CHECK:       PLACEHOLDER_MISMATCH ERROR: ARG_ANNOTATION_MISMATCH
# CHECK:       error[ARG_ANNOTATION_MISMATCH]:{{.*}}Argument #1 `n` must be a `Int32`, but got a `FloatMeta`.
# CHECK:       PLACEHOLDER_CALL ERROR: ARG_ANNOTATION_MISMATCH
# EXEC:        PLACEHOLDER: twice 42 4 6
# EXEC-NEXT:   PLACEHOLDER_OTHER ERROR: ARG_ANNOTATION_MISMATCH
# EXEC-NEXT:   PLACEHOLDER_MISMATCH ERROR: ARG_ANNOTATION_MISMATCH
# EXEC:        PLACEHOLDER_CALL ERROR: ARG_ANNOTATION_MISMATCH


# --- the host entry ------------------------------------------
# The host entry is `func.Jit`'s `func.func` with `llvm.emit_c_interface`;
# `convert-func-to-llvm` lowers it right after the `cf` lowering (`core_only.py`
# prints the pass list) and the packed `_mlir_<name>` wrapper calls it.
@m.jit
def cumsum_func(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


# CHECK-LABEL: func.func @cumsum_func(
# CHECK-SAME:    %[[N:[^:]+]]: i32) -> i32 attributes {llvm.emit_c_interface}
# CHECK:         %[[ACC:.+]] = scf.for %{{.+}} = %{{.+}} to %[[N]] step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (i32) : i32 {
# CHECK:         return %[[ACC]] : i32
# CHECK:         FUNC_ENTRY: ?
# EXEC:          FUNC_ENTRY: 45
print("FUNC_ENTRY:", cumsum_func(10))

# --- the `Plugin` protocol ---------------------------
# `install` binds a per-instance copy (the class-level instance stays
# unbound); the sub-DSL owns the pipeline (`pipeline()`) and names its
# plugins in the `Plugins` record (an add-on is a decorator or adapter plugin);
# `attach_to_module` edits the
# traced module before it is hashed; `after_lowering` sees the lowered module
# of the named entry before the engine is built; `wrap_compiled_function` replaces the
# compiled function, and a replacement that `prefers_python_args` is called
# with the Python arguments, also on a cache hit and from `compile()`. The
# engine's libraries (`<PREFIX>_LIBS`, the plugins' `shared_libs()`, the call's
# `extra_link_libs`) are made canonical, deduplicated and checked to exist.
calls = []


class PythonArgsWrapper:
    prefers_python_args = True

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        print("WRAPPER_CALL:", args, kwargs)
        return self.inner(*args, **kwargs)


class Recorder(m.AdapterPlugin):
    name = "recorder"

    def attach_to_module(self, dsl, module, function_name, sig, args, kwargs):
        calls.append(("attach", function_name, list(sig.parameters)))
        module.operation.attributes["recorder.entry"] = ir.StringAttr.get(function_name)

    def after_lowering(self, dsl, module, function_name):
        ops = sorted({op.operation.name for op in module.body.operations})
        calls.append(("lowered", function_name, ops))

    def wrap_compiled_function(self, dsl, jit_function):
        calls.append(("wrap", type(jit_function).__name__))
        return PythonArgsWrapper(jit_function)


class PluggedDSL(m.MlirTestDSL):
    plugins = dataclasses.replace(
        m.MlirTestDSL.plugins,
        adapters=[*m.MlirTestDSL.plugins.adapters, Recorder()],
    )

    def pipeline(self):
        return ["canonicalize", "cse", *super().pipeline()]


plugged = PluggedDSL()
recorder = plugged.plugins.named("recorder")
bound = (recorder is not PluggedDSL.plugins.adapters[-1], recorder.dsl is plugged)
print(
    "INSTALLED:",
    *bound,
    PluggedDSL.plugins.adapters[-1].dsl,
    plugged.plugins.adapters[-1].name,
)
print("PIPELINE:", plugged._get_pipeline(None))
here = os.path.realpath(__file__)
extra = dsl.get_shared_libs((__file__, here, os.path.relpath(__file__)))
print("LIBS:", extra.count(here), len(extra) - len(dsl.get_shared_libs()))
try:
    dsl.get_shared_libs(("/nonexistent/libfoo.so",))
except m.DSLRuntimeError as e:
    print("BAD_LIB:", e.message)
# CHECK:      INSTALLED: True True None recorder
# CHECK-NEXT: PIPELINE: builtin.module({{.*}}canonicalize,cse,convert-scf-to-cf,{{.*}}reconcile-unrealized-casts)
# CHECK-NEXT: LIBS: 1 1
# CHECK-NEXT: BAD_LIB: shared library not found: /nonexistent/libfoo.so


@PluggedDSL.jit
def plugged_twice(n: m.Int32) -> m.Int32:
    return n * 2


# CHECK:       module attributes {recorder.entry = "plugged_twice"}
# CHECK-LABEL: func.func @plugged_twice(
# CHECK:         HOOKS: ? [('attach', 'plugged_twice', ['n'])]
# EXEC:          WRAPPER_CALL: (21,) {}
# EXEC-NEXT:     HOOKS: 42 [('attach', 'plugged_twice', ['n']), ('lowered', 'plugged_twice', ['llvm.func']), ('wrap', 'JitCompiledFunction')]
print("HOOKS:", plugged_twice(21), calls)
del calls[:]
# EXEC:      WRAPPER_CALL: (4,) {}
# EXEC-NEXT: CACHED: 8 [('attach', 'plugged_twice', ['n'])]
# EXEC-NEXT: WRAPPER_CALL: (2,) {}
# EXEC-NEXT: COMPILED: PythonArgsWrapper 4
print("CACHED:", plugged_twice(4), calls)
if not dsl.envar.dryrun:
    compiled = m.compile(plugged_twice, 1)
    print("COMPILED:", type(compiled).__name__, compiled(2))


# --- `collected_remarks` ----------------------------------------
# Every compile replaces the list with that compile's records (it does not
# accumulate); a cached call re-traces, so a trace-time remark is collected
# again; without `<PREFIX>_REMARKS` nothing is collected. `remarks.py` covers
# the filter and the YAML stream.
REMARK = dict(category="mlir.dsl", function_name="remarked", message="n is staged")


@m.jit
def remarked(n: m.Int32) -> m.Int32:
    emit = ir.Location.current.emit_remark
    emit(ir.RemarkKind.ANALYSIS, "Decision", args=[("staged", "n")], **REMARK)
    return n + 1


@m.jit
def silent(n: m.Int32) -> m.Int32:
    return n + 3


def summary():
    return [
        (r["kind"], r["name"], r["category"], r["function"], r["args"]["staged"])
        for r in dsl.collected_remarks
    ]


print("REMARKED:", remarked(1), summary())
print("SILENT:", silent(1), summary())
hits = dsl.cache_hits
print("CACHED_REMARK:", remarked(5), summary(), dsl.cache_hits - hits)
print("KEYS:", sorted(dsl.collected_remarks[0]) if dsl.collected_remarks else "none")
# REMARKS: REMARKED: 2 [('analysis', 'Decision', 'mlir.dsl', 'remarked', 'n')]
# REMARKS: SILENT: 4 []
# REMARKS: CACHED_REMARK: 6 [('analysis', 'Decision', 'mlir.dsl', 'remarked', 'n')] 1
# REMARKS: KEYS: ['args', 'category', 'col', 'filename', 'full_category', 'function', 'kind', 'line', 'location', 'message', 'name', 'remark_id']
# EXEC:    REMARKED: 2 []
# EXEC:    SILENT: 4 []
# EXEC:    KEYS: none

# --- `LaunchConfig` and the public surface ----------------------------------
# Dimensions are padded with 1s to three entries, a tuple becomes a list, the
# cluster is kept only when given. Every name in `__all__` is importable and
# unique; the decorators are `MlirTestDSL`'s and `LaunchConfig` is the gpu
# kernels plugin's.
print("LAUNCH:", m.LaunchConfig(grid=4, block=(8, 8), cluster=(1, 2, 3), smem=1024))
names = list(m.__all__)
missing = [n for n in names if not hasattr(m, n)]
unique = len(names) == len(set(names))
print(f"API: {unique} {missing} {m.jit == m.MlirTestDSL.jit}", end=" ")
print(m.LaunchConfig.__module__ == "mlir.dsl.plugins.decorators.kernels.gpu_plugin")
# CHECK:      LAUNCH: LaunchConfig(cluster=[1, 2, 3], grid=[4, 1, 1], block=[8, 8, 1], smem=1024, async_deps=[])
# CHECK-NEXT: API: True [] True True
