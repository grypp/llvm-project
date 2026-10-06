# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC
# REQUIRES: host-supports-jit
# The composite leaf types (Design 4, 4b, 6): the `Pointer` API (annotation,
# host object, indexing, load/store, masked access, conversions, host-boundary
# adaptation, region carries) and its `POINTER_*`/`ARG_*` errors; `@struct`/
# `make_struct`/`Struct` semantics (fields, replace,
# nesting, options, marshalling, carries) and the `STRUCT_*` errors; the
# `Vector` API (construction, splat, broadcast rules, reduce/extract, carries)
# and its errors; the torch dtype bridge, which needs no torch.
import ctypes
import sys

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.plugins.thirdparty.pytorch import from_torch_dtype

I8, I32, I64, U8, U32 = m.Int8, m.Int32, m.Int64, m.Uint8, m.Uint32
F16, F32, B, V, P, S = m.Float16, m.Float32, m.Boolean, m.Vector, m.Pointer, m.Struct


def show(label, *values):
    print(f"{label}:", *(v if isinstance(v, str) else repr(v) for v in values))


def err(label, fn):
    try:
        fn()
        print(f"{label}: NO ERROR")
    except m.DSLUserCodeError as e:
        print(f"{label}: {e.diag_id.name if e.diag_id else '<free-form>'}")
        return e


def name(t):
    return t.__name__


buf = np.zeros(4, np.float32)


def staged_ptr(label, body):
    """Run `body(p, i)` in a preprocessor-less trace over (Pointer[Float32], Int32)."""

    @m.jit(preprocess=False)
    def f(p: P[F32], i: I32):
        return body(p, i)

    err(label, lambda: f(buf, 1))


# ===== Pointer =============================================================
# `Pointer[dtype]`/`Pointer[dtype, space]` is a `TypedPointer` record that
# declares the `!llvm.ptr`/`!llvm.ptr<N>` parameter type and compares by
# value; the host `Pointer` (built outside a trace) carries (address, dtype,
# space, kind, keepalive), compares and hashes by address, marshals as a
# `c_void_p` and refuses every operation that needs an SSA value.
tp = P[F32]
show("TYPED", repr(tp), name(tp.dtype), tp.space, tp == P[F32, 0], tp == P[I32, 1])
show("TYPED space", repr(P[I32, 1]), P[I32, 1].space, hash(tp) == hash(P[F32]), tp == 3)
# CHECK: TYPED: Pointer[Float32, 0] Float32 0 True False
# CHECK: TYPED space: Pointer[Int32, 1] 1 True False
for i, bad in enumerate((int, "Float32", (F32, -1), (F32, True), (F32, 1, 2), V)):
    err(f"bad subscript {i}", lambda: P.__class_getitem__(bad))
# CHECK-COUNT-6: bad subscript {{[0-5]}}: POINTER_BAD_SUBSCRIPT
hp, h2 = P(4096, dtype=m.Int16, kind="host"), P(4096, dtype=I8)
copy = P(P(4096, dtype=m.Int16, kind="host", keepalive=buf), dtype=F32)
show("HOST", hp.address, hp.kind, hp.space, name(hp.dtype), hp.is_staged, hp.alignment)
show("HOST str", str(hp), name(P(0).dtype), P(0).kind, P(0).alignment)
show("HOST eq", hp == P(4096), hp == P(4097), hash(hp) == hash(h2), hp == 4096)
show("HOST copy", copy.address, copy.kind, name(copy.dtype), copy._keepalive is buf)
show("HOST marshal", isinstance(hp.marshal(), ctypes.c_void_p))
# CHECK: HOST: 4096 host 0 Int16 False 2
# CHECK: HOST str: ptr<space=0, dtype=Int16> Int8 unknown 1
# CHECK: HOST eq: True False True False
# CHECK: HOST copy: 4096 host Float32 True
# CHECK: HOST marshal: True
err("Pointer(str)", lambda: P("0x1000"))
err("Pointer(-5)", lambda: P(-5))
err("host toint", lambda: hp.toint())
print(str(err("host ir_value", lambda: hp.ir_value())))
# CHECK: Pointer(str): ARG_UNSUPPORTED_TYPE
# CHECK: Pointer(-5): ARG_POINTER_NEGATIVE
# CHECK: host toint: CALL_OUTSIDE_JIT
# CHECK: host ir_value: <free-form>
# CHECK: error:{{.*}}A host pointer{{.*}}can only enter compiled code as an argument
# CHECK: suggestion:{{.*}}parameter annotated `Pointer[dtype]`


# Indexing and address arithmetic: `p[i]` is gep + load, `p[i] = v` gep +
# store; a Python/Meta index is a static gep, a staged Integer (or raw
# integer value) a dynamic one of its own width; `p + i`, `p - i`, `p += i`
# are geps too (no wrap flags); slices, tuples and floats are errors.
@m.jit
def index_kinds(p: P[I32], i: I32, j: I64, k) -> I32:
    (p + i)[1] = p[3] + p[j] + p[k]  # static, dynamic i64, Meta; dynamic i32
    q = p
    q += 2  # rebinds q to a new Pointer
    (q - 1)[0] = 7
    print("REBOUND:", q is p, name(q.dtype))
    return p[i.ir_value()]  # a raw integer ir.Value is a dynamic index too


# CHECK:       REBOUND: False Int32
# CHECK-LABEL: func.func @index_kinds_2(
# CHECK-SAME:    %[[P:[^:]+]]: !llvm.ptr, %[[I:[^:]+]]: i32, %[[J:[^:]+]]: i64) -> i32
# CHECK:         %[[G3:.+]] = llvm.getelementptr %[[P]][3] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.load %[[G3]] <alignment = 4> : !llvm.ptr -> i32
# CHECK:         llvm.getelementptr %[[P]][%[[J]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
# CHECK:         llvm.getelementptr %[[P]][2] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         %[[A:.+]] = llvm.getelementptr %[[P]][%[[I]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
# CHECK:         %[[A1:.+]] = llvm.getelementptr %[[A]][1] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.store %{{.+}}, %[[A1]] <alignment = 4> : i32, !llvm.ptr
# CHECK:         %[[Q:.+]] = llvm.getelementptr %[[P]][2] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.getelementptr %[[Q]][-1] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:         llvm.getelementptr %[[P]][%[[I]]] : (!llvm.ptr, i32) -> !llvm.ptr, i32
# CHECK-NOT:     inbounds
# EXEC:          RESULT: 7 [10, 7, 39, 13, 14, 15]
data = np.array([10, 11, 12, 13, 14, 15], np.int32)
print("RESULT:", index_kinds(data, 1, 4, 2), data.tolist())
staged_ptr("slice", lambda p, i: p[0:2])
staged_ptr("tuple", lambda p, i: p[0, 1])
staged_ptr("float offset", lambda p, i: p + 0.5)
# CHECK: slice: POINTER_INDEX_UNSUPPORTED
# CHECK: tuple: POINTER_INDEX_UNSUPPORTED
# CHECK: float offset: POINTER_INDEX_UNSUPPORTED


# `load`/`store`: the dtype's natural alignment unless `alignment=`; `count=`
# loads a `Vector`; a literal is coerced to the pointer dtype but a Numeric or
# Vector must already have it; fp8 goes through i8 and `arith.bitcast`;
# `masked_load`/`masked_store` take a Vector of Boolean lanes and an optional
# `pass_thru` for the masked-off lanes.
@m.jit
def load_store(p: P[F32], b: P[I8], q: P[m.Float8E5M2], n: I32) -> F32:
    a = p.load(alignment=16)
    v = (p + 1).load(count=2)  # vector<2xf32>
    (p + 3).store(2)  # int literal -> 2.0
    b.store(b.load() + I8(1))
    (q + 1).store(q.load())  # i8 load, bitcast to f8E5M2 and back
    mask = V([n > 0, False])
    w = p.masked_load(mask, V.splat(F32(-1.0), 2))  # lane 1 comes from pass_thru
    (p + 4).masked_store(w * 10.0, mask, alignment=16)
    print("TYPES:", name(type(a)), name(type(v)), v.lanes, p.alignment, q.alignment)
    return a + v.sum() + w.sum() + p.masked_load(mask)[0]


# CHECK:       TYPES: Float32 Vector 2 4 1
# CHECK-LABEL: func.func @load_store(
# CHECK-SAME:    %[[P:[^:]+]]: !llvm.ptr, %[[B:[^:]+]]: !llvm.ptr, %[[Q:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32) -> f32
# CHECK:         llvm.load %[[P]] <alignment = 16> : !llvm.ptr -> f32
# CHECK:         llvm.load %{{.+}} <alignment = 4> : !llvm.ptr -> vector<2xf32>
# CHECK:         %[[C2:.+]] = arith.constant 2.000000e+00 : f32
# CHECK:         llvm.store %[[C2]], %{{.+}} <alignment = 4> : f32, !llvm.ptr
# CHECK:         llvm.load %[[B]] <alignment = 1> : !llvm.ptr -> i8
# CHECK:         llvm.store %{{.+}}, %[[B]] <alignment = 1> : i8, !llvm.ptr
# CHECK:         %[[I8:.+]] = llvm.load %[[Q]] <alignment = 1> : !llvm.ptr -> i8
# CHECK:         %[[F8:.+]] = arith.bitcast %[[I8]] : i8 to f8E5M2
# CHECK:         %[[J8:.+]] = arith.bitcast %[[F8]] : f8E5M2 to i8
# CHECK:         llvm.store %[[J8]], %{{.+}} <alignment = 1> : i8, !llvm.ptr
# CHECK:         %[[MASK:.+]] = vector.from_elements %{{.+}}, %{{.+}} : vector<2xi1>
# CHECK:         %[[FILL:.+]] = vector.broadcast %{{.+}} : f32 to vector<2xf32>
# CHECK:         llvm.intr.masked.load(%[[P]], %[[MASK]], %[[FILL]]), alignment(4) : (!llvm.ptr, vector<2xi1>, vector<2xf32>) -> vector<2xf32>
# CHECK:         llvm.intr.masked.store(%{{.+}}, %{{.+}}, %[[MASK]]), alignment(16) : vector<2xf32>, vector<2xi1> into !llvm.ptr
# CHECK:         llvm.intr.masked.load(%[[P]], %[[MASK]]), alignment(4) : (!llvm.ptr, vector<2xi1>) -> vector<2xf32>
# EXEC:          RESULT: 7.0 [1.0, 2.0, 3.0, 2.0, 10.0, 9.0] 8 [60, 60]
f6 = np.array([1.0, 2.0, 3.0, 0.0, 9.0, 9.0], np.float32)
i8 = np.array([7], np.int8)
raw = np.array([60, 0], np.uint8)  # 0x3C is 1.0 in e5m2
r = load_store(f6, i8, raw.ctypes.data, 1)  # 1 + 5 + (1 - 1) + 1
print("RESULT:", r, f6.tolist(), int(i8[0]), raw.tolist())
staged_ptr("store Int32", lambda p, i: p.store(i))
staged_ptr("store vector dtype", lambda p, i: p.store(V([i, i])))
staged_ptr("store str", lambda p, i: p.store("x"))
staged_ptr("int mask", lambda p, i: p.masked_load(V([1, 2])))
staged_ptr("pass_thru", lambda p, i: p.masked_load(V([True]), V.splat(F32(0.0), 4)))
staged_ptr("masked dtype", lambda p, i: p.masked_store(V([i, i]), V([True, False])))
# CHECK: store Int32: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: store vector dtype: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: store str: ARG_NOT_NUMERIC
# CHECK: int mask: ARG_ANNOTATION_MISMATCH
# CHECK: pass_thru: ARG_ANNOTATION_MISMATCH
# CHECK: masked dtype: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED


# Conversions: `toint()` is `ptrtoint` (Int64 by default), `inttoptr` builds
# from a Python int (a constant), an Integer or a raw value, `tospace` is an
# `addrspacecast` (the same object for the current space), `p & mask` aligns
# through the integer round trip.
@m.jit
def conversions(p: P[F32], q: P[I32, 1], i: I32) -> I64:
    a = p.toint()
    r = m.inttoptr(a, 0, F32)  # Int64 -> !llvm.ptr
    r[0] = 1.0
    s = m.inttoptr(4096, 1, I8)  # a Python int: constant i64, space 1
    t = m.inttoptr(i, 0, I32)  # a staged Int32 address
    al = p & ~15
    print("CONV:", name(type(a)), name(type(p.toint(I32))), s.space, s.mlir_type)
    print("CONV space:", q.tospace(0).space, q.tospace(1) is q, al.space, t.space)
    return I64(a - al.toint() < 16) + I64(q.tospace(0).space)


# CHECK:       CONV: Int64 Int32 1 !llvm.ptr<1>
# CHECK-NEXT:  CONV space: 0 True 0 0
# CHECK-LABEL: func.func @conversions(
# CHECK-SAME:    %[[P:[^:]+]]: !llvm.ptr, %[[Q:[^:]+]]: !llvm.ptr<1>, %[[I:[^:]+]]: i32) -> i64
# CHECK:         %[[A:.+]] = llvm.ptrtoint %[[P]] : !llvm.ptr to i64
# CHECK:         %[[R:.+]] = llvm.inttoptr %[[A]] : i64 to !llvm.ptr
# CHECK:         llvm.getelementptr %[[R]][0] : (!llvm.ptr) -> !llvm.ptr, f32
# CHECK:         %[[C:.+]] = arith.constant 4096 : i64
# CHECK:         llvm.inttoptr %[[C]] : i64 to !llvm.ptr<1>
# CHECK:         llvm.inttoptr %[[I]] : i32 to !llvm.ptr
# CHECK:         %[[M:.+]] = arith.constant -16 : i64
# CHECK:         %[[AND:.+]] = arith.andi %{{.+}}, %[[M]] : i64
# CHECK:         llvm.inttoptr %[[AND]] : i64 to !llvm.ptr
# CHECK:         llvm.ptrtoint %[[P]] : !llvm.ptr to i32
# CHECK:         llvm.addrspacecast %[[Q]] : !llvm.ptr<1> to !llvm.ptr
# EXEC:          RESULT: 1 1.0
print("RESULT:", conversions(buf, P(0, dtype=I32, space=1), 0), float(buf[0]))
staged_ptr("inttoptr float", lambda p, i: m.inttoptr(1.5, 0, F32))
# CHECK: inttoptr float: ARG_NOT_NUMERIC


# The host boundary: a `Pointer[T]` parameter adapts a bare address (int,
# `c_void_p`: kind unknown), a numpy array (dtype checked) or a host
# `Pointer`; an unannotated array stages with its own dtype; the trace never
# sees the host payload.
@m.jit
def unannotated(p, n: I32):
    print("UNANNOTATED:", name(p.dtype), p.kind)
    p[1] = n


@m.jit
def addr_of(p: P[F32]) -> I64:
    print("KIND:", p.kind, p.is_staged, p.address)
    return p.toint()


# CHECK:       UNANNOTATED: Int32 unknown
# CHECK-LABEL: func.func @unannotated(
# CHECK-SAME:    %[[P:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32)
# CHECK:         llvm.store %[[N]], %{{.+}} <alignment = 4> : i32, !llvm.ptr
# CHECK:       KIND: unknown True None
# CHECK-LABEL: func.func @addr_of(
# EXEC:          ADDRESSES: [0, 9, 0] True True True True
ints = np.zeros(3, np.int32)
unannotated(ints, 9)
addr = buf.ctypes.data
same = lambda r: "?" if m.is_dynamic_expression(r) else r == addr
results = [addr_of(addr), addr_of(ctypes.c_void_p(addr)), addr_of(buf)]
results.append(addr_of(P(addr, dtype=F32)))
print("ADDRESSES:", ints.tolist(), *(same(r) for r in results))
bad_args = [
    ("negative", -1),
    ("float64 array", np.zeros(4, np.float64)),
    ("other space", P(addr, dtype=F32, space=1)),
    ("strided", np.zeros(8, np.float32)[::2]),
    ("device on host", P(addr, dtype=F32, kind="device")),
    ("str", "0x1000"),
    ("bool", True),
]
for label, arg in bad_args:
    err(f"arg {label}", lambda: addr_of(arg))
# CHECK: arg negative: ARG_POINTER_NEGATIVE
# CHECK: arg float64 array: ARG_ANNOTATION_MISMATCH
# CHECK: arg other space: ARG_ANNOTATION_MISMATCH
# CHECK: arg strided: ARG_BUFFER_NOT_CONTIGUOUS
# CHECK: arg device on host: ARG_DEVICE_BUFFER_ON_HOST
# CHECK: arg str: ARG_ANNOTATION_MISMATCH
# CHECK: arg bool: ARG_ANNOTATION_MISMATCH


# A Pointer is a reference leaf: a store base alone adds no iter_arg; a
# rebound pointer is carried as one `!llvm.ptr`; a join of two prototypes
# (dtype, space) is `TYPE_UNSTABLE_JOIN`.
@m.jit
def carry(out: P[I32], n: I32):
    for i in range(n):
        out[i] = i  # store only: no carry
    p = out
    for i in range(n):
        p[0] = p[0] * 10
        p += 1  # rebinding: one !llvm.ptr iter_arg
    if n > 2:
        p = p - n  # an `if` arm rebinding: scf.if yields the pointer
    p[0] = 7


# CHECK-LABEL: func.func @carry(
# CHECK-SAME:    %[[OUT:[^:]+]]: !llvm.ptr, %[[N:[^:]+]]: i32)
# CHECK:         scf.for %[[I:.+]] = %{{.+}} to %[[N]] step %{{.+}} : i32 {
# CHECK-NOT:     iter_args
# CHECK:         scf.for %{{.+}} = %{{.+}} to %[[N]] step %{{.+}} iter_args(%[[P:.+]] = %[[OUT]]) -> (!llvm.ptr) : i32 {
# CHECK:           %[[NEXT:.+]] = llvm.getelementptr %[[P]][1] : (!llvm.ptr) -> !llvm.ptr, i32
# CHECK:           scf.yield %[[NEXT]] : !llvm.ptr
# CHECK:         scf.if %{{.+}} -> (!llvm.ptr) {
# CHECK:           scf.yield %{{.+}} : !llvm.ptr
# CHECK:         } else {
# CHECK:           scf.yield %{{.+}} : !llvm.ptr
# EXEC:          RESULT: [7, 10, 20, 0] [0, 10, 7, 0]
a4, b4 = np.zeros(4, np.int32), np.zeros(4, np.int32)
carry(a4, 3)
carry(b4, 2)
print("RESULT:", a4.tolist(), b4.tolist())


@m.jit
def join_dtype(p: P[F32], q: P[I32], n: I32):
    r = p
    if n > 2:
        r = q  # another (dtype, space) prototype
    r[0] = 1


err("join dtype", lambda: join_dtype(buf, a4, 3))
# CHECK: join dtype: TYPE_UNSTABLE_JOIN


# ===== Structs ============================================================
# `@struct` is a frozen record of DSL-typed fields and a pytree, not an SSA
# aggregate: a struct argument arrives as one block argument per field, a
# field read is the field's own value, `replace` returns a copy, unpacking
# iterates the fields, struct and `Pointer` fields nest, a struct result is
# its leaves in the packed host result and a rebound struct is carried leaf by
# leaf. No `llvm.struct` appears anywhere; `make_struct` builds the same class
# as the decorator; a Python-typed field is `STRUCT_FIELD_TYPE`.
@m.struct
class Vec2:
    x: I32
    y: F32


@m.struct
class Inner:
    x: I32
    y: F32


@m.struct
class Outer:
    inner: Inner
    u: U8
    h: F16
    p: P[F32]


Pair = m.make_struct("Pair", lo=I32, hi=I32)


@m.jit
def vec2_ops(v: Vec2) -> F32:
    w = v.replace(y=v.y * 2)  # a copy; `v` is unchanged
    a, b = w  # the fields in declaration order
    print("VEC2:", w is v, w.x is v.x, name(type(a)), name(type(b)))
    return F32(a) + b + v.y


# CHECK:       VEC2: False True Int32 Float32
# CHECK-LABEL: func.func @vec2_ops(
# CHECK-SAME:    %[[X:[^:]+]]: i32, %[[Y:[^:]+]]: f32) -> f32
# CHECK:         %[[YS:.+]] = arith.mulf %[[Y]], %{{.+}} : f32
# CHECK:         %[[XF:.+]] = arith.sitofp %[[X]] : i32 to f32
# CHECK:         %[[S:.+]] = arith.addf %[[XF]], %[[YS]] : f32
# CHECK:         arith.addf %[[S]], %[[Y]] : f32
# CHECK-NOT:     llvm.{{(insert|extract)}}value
# CHECK:         RESULT: ?
# EXEC:          RESULT: 15.0
print("RESULT:", vec2_ops(Vec2(x=3, y=4.0)))  # 3 + 8 + 4


@m.jit
def nested(o: Outer, buf: P[F32]) -> tuple:
    o.p[0] = o.inner.y  # the pointer field is a Pointer[Float32]
    fresh = Outer(inner=Inner(x=1, y=2.0), u=200, h=0.5, p=buf)
    pr = Pair(lo=1, hi=2)
    print("NESTED:", name(type(o.inner)), name(type(o.u)), name(type(fresh.p)))
    return F32(o.u) + F32(o.h) + F32(fresh.inner.x) + fresh.h + buf[1], pr


# CHECK:       NESTED: Inner Uint8 Pointer
# CHECK-LABEL: func.func @nested(
# CHECK-SAME:    %[[IX:[^:]+]]: i32, %[[IY:[^:]+]]: f32, %[[U:[^:]+]]: i8, %[[H:[^:]+]]: f16, %[[OP:[^:]+]]: !llvm.ptr, %[[BUF:[^:]+]]: !llvm.ptr) -> !llvm.struct<(f32, i32, i32)>
# CHECK:         %[[OP0:.+]] = llvm.getelementptr %[[OP]][0] : (!llvm.ptr) -> !llvm.ptr, f32
# CHECK:         llvm.store %[[IY]], %[[OP0]] <alignment = 4> : f32, !llvm.ptr
# CHECK:         llvm.load %{{.+}} <alignment = 4> : !llvm.ptr -> f32
# CHECK:         llvm.mlir.undef : !llvm.struct<(f32, i32, i32)>
# CHECK:         return %{{.+}} : !llvm.struct<(f32, i32, i32)>
fbuf = np.array([0.0, 7.0], np.float32)
hptr = P(fbuf.ctypes.data, dtype=F32, kind="host", keepalive=fbuf)
o = Outer(inner=Inner(x=3, y=4.0), u=200, h=1.5, p=hptr)
total, pair = nested(o, fbuf)  # 200 + 1.5 + 1 + 0.5 + 7
if not m.is_dynamic_expression(total):
    print("RESULT:", total, repr(pair), fbuf.tolist())
# EXEC: RESULT: 210.0 Pair(lo=Int32(1), hi=Int32(2)) [4.0, 7.0]

# Host instances hold host values; a missing field is the zero of its type.
pair, v1 = Pair(lo=1, hi=2), Vec2(x=1)
show("HOST", repr(v1), repr(Outer().inner), Outer().p.address)
show("HOST class", isinstance(o, S), issubclass(Pair, S), Pair._field_names)
show("HOST fields", Vec2._field_names, tuple(pair), hasattr(v1, "scale"))
# CHECK: HOST: Vec2(x=Int32(1), y=Float32(0.0)) Inner(x=Int32(0), y=Float32(0.0)) 0
# CHECK: HOST class: True True ['lo', 'hi']
# CHECK: HOST fields: ['x', 'y'] (Int32(1), Int32(2)) False


# A struct rebound in a region is carried leaf by leaf; another class at a
# join changes the carried structure (`CONTAINER_STRUCTURE_CHANGED`).
@m.jit
def struct_carry(n: I32) -> I32:
    acc = Vec2(x=0, y=0.0)
    for i in range(n):
        acc = acc.replace(x=acc.x + i)
    if n > 2:
        acc = Vec2(x=acc.x * 10, y=1.0)  # a fresh instance of the same class
    return acc.x * 3


# CHECK-LABEL: func.func @struct_carry(
# CHECK:         scf.for {{.*}} iter_args(%[[AX:.+]] = %{{.+}}, %{{.+}} = %{{.+}}) -> (i32, f32) : i32 {
# CHECK:           arith.addi %[[AX]], %{{.+}} : i32
# CHECK:           scf.yield %{{.+}}, %{{.+}} : i32, f32
# CHECK:         scf.if %{{.+}} -> (i32, f32) {
# CHECK:           scf.yield %{{.+}}, %{{.+}} : i32, f32
# CHECK:         } else {
# CHECK:           scf.yield %{{.+}}, %{{.+}} : i32, f32
# CHECK-NOT:     llvm.{{(insert|extract)}}value
# EXEC:          RESULT: 180 3
print("RESULT:", struct_carry(4), struct_carry(2))  # 6*10*3, 1*3


@m.jit
def class_changes(n: I32) -> I32:
    acc = Vec2(x=n, y=0.0)
    if n > 2:
        acc = Inner(x=n, y=0.0)  # same fields, another class
    return acc.x


err("class changes", lambda: class_changes(3))
# CHECK: class changes: CONTAINER_STRUCTURE_CHANGED

# Declaration, construction and field coercion mistakes.
err("no fields", lambda: m.make_struct("E"))
err("python field", lambda: m.make_struct("E", n=int))
err("unexpected kwarg", lambda: Vec2(z=1))
err("positional", lambda: Vec2(1, 2.0))
err("field value", lambda: Vec2(x="a"))
err("nested kind", lambda: m.make_struct("N", v=Vec2)(v=3))
err("assign", lambda: setattr(o, "u", 1))
err("jit int arg", lambda: vec2_ops(3))
# CHECK: no fields: STRUCT_NO_FIELDS
# CHECK: python field: STRUCT_FIELD_TYPE
# CHECK: unexpected kwarg: STRUCT_UNEXPECTED_KWARG
# CHECK: positional: STRUCT_VALUE_TYPE
# CHECK: field value: ARG_NOT_NUMERIC
# CHECK: nested kind: ARG_ANNOTATION_MISMATCH
# CHECK: assign: STRUCT_FIELD_ASSIGNMENT
# CHECK: jit int arg: ARG_ANNOTATION_MISMATCH


@m.jit(preprocess=False)
def staged_struct_errors(v: Vec2, i: I32):
    err("staged arity", lambda: v.replace(x=(i, i)))
    err("staged assign", lambda: setattr(v, "x", i))


staged_struct_errors(Vec2(x=1), 2)
print(str(err("rendered", lambda: Vec2(1, 2.0))))
# CHECK: staged arity: STRUCT_FIELD_ARITY
# CHECK: staged assign: STRUCT_FIELD_ASSIGNMENT
# CHECK: error[STRUCT_VALUE_TYPE]:{{.*}}`Vec2(...)` takes keyword arguments naming its fields
# CHECK: suggestion:{{.*}}`Vec2(x=


# ===== Vectors ============================================================
# A lane list is one `vector.from_elements` (a typed lane or `dtype=` fixes
# the dtype of literals, else the literal rule); `splat` is `vector.broadcast`;
# a literal operand is a scalar constant broadcast to the lanes, a typed scalar goes through
# `vector.broadcast`, reverse operators swap the operands; integer `/` yields
# Float32 lanes (`sitofp`/`uitofp`); `sum()` is `vector.reduction <add>`;
# `v[i]` is `vector.extract` at a compile-time lane; iteration unrolls.
@m.jit
def vectors(a: F32, i: I32, u: U32) -> F32:
    v = V([a, 0, a])  # the typed lane makes `0` a Float32
    w = V.splat(i, 2)  # vector.broadcast
    k = V([1, True], dtype=I64)
    x = v * 2.0 + a  # literal: const + broadcast; typed scalar: broadcast
    y = 1.0 - x  # reverse: the constant is the left operand
    q = (w + 1) / 2  # Int32 lanes: sitofp both sides, divf, Float32 lanes
    r = V([u, u]) / U32(4)  # unsigned: uitofp
    print("VEC:", v.mlir_type, name(w.dtype), k.mlir_type, name(q.dtype), len(v))
    print("LANES:", name(type(v[-1])), [name(type(l)) for l in w])
    return y.sum() + q.sum() + r[0] + F32(k[1]) + v[-1]


# CHECK:       VEC: vector<3xf32> Int32 vector<2xi64> Float32 3
# CHECK-NEXT:  LANES: Float32 ['Int32', 'Int32']
# CHECK-LABEL: func.func @vectors(
# CHECK-SAME:    %[[A:[^:]+]]: f32, %[[I:[^:]+]]: i32, %[[U:[^:]+]]: i32) -> f32
# CHECK:         %[[Z:.+]] = arith.constant 0.000000e+00 : f32
# CHECK:         %[[V:.+]] = vector.from_elements %[[A]], %[[Z]], %[[A]] : vector<3xf32>
# CHECK:         %[[W:.+]] = vector.broadcast %[[I]] : i32 to vector<2xi32>
# CHECK:         vector.from_elements %{{.+}}, %{{.+}} : vector<2xi64>
# CHECK:         %[[S2:.+]] = arith.constant 2.000000e+00 : f32
# CHECK:         %[[C2:.+]] = vector.broadcast %[[S2]] : f32 to vector<3xf32>
# CHECK:         %[[M:.+]] = arith.mulf %[[V]], %[[C2]] : vector<3xf32>
# CHECK:         %[[AB:.+]] = vector.broadcast %[[A]] : f32 to vector<3xf32>
# CHECK:         %[[X:.+]] = arith.addf %[[M]], %[[AB]] : vector<3xf32>
# CHECK:         %[[S1:.+]] = arith.constant 1.000000e+00 : f32
# CHECK:         %[[C1:.+]] = vector.broadcast %[[S1]] : f32 to vector<3xf32>
# CHECK:         arith.subf %[[C1]], %[[X]] : vector<3xf32>
# CHECK:         %[[W1:.+]] = arith.addi %[[W]], %{{.+}} : vector<2xi32>
# CHECK:         arith.sitofp %[[W1]] : vector<2xi32> to vector<2xf32>
# CHECK:         arith.divf %{{.+}}, %{{.+}} : vector<2xf32>
# CHECK:         arith.uitofp %{{.+}} : vector<2xi32> to vector<2xf32>
# CHECK:         vector.reduction <add>, %{{.+}} : vector<3xf32> into f32
# CHECK:         vector.extract %{{.+}}[1] : i64 from vector<2xi64>
# CHECK:         vector.extract %[[V]][2] : f32 from vector<3xf32>
# EXEC:          RESULT: 1.0
print("RESULT:", vectors(1.5, 3, 8))  # -7.5 + 4 + 2 + 1 + 1.5


# A Vector is a registered leaf: rebound in a region it is carried as one
# `vector<NxT>` (`load(count=)` wraps into one); joins need equal MLIR types;
# it has no host representation.
@m.jit
def vector_carry(n: I32, a: F32, p: P[F32]) -> F32:
    acc = V.splat(a, 4)
    for i in range(n):
        acc = acc + (p + i * 4).load(count=4)
    if n > 2:
        acc = acc * 2.0
    return acc.sum()


# CHECK-LABEL: func.func @vector_carry(
# CHECK:         scf.for {{.*}} iter_args(%[[ACC:.+]] = %{{.+}}) -> (vector<4xf32>) : i32 {
# CHECK:           %[[L:.+]] = llvm.load %{{.+}} <alignment = 4> : !llvm.ptr -> vector<4xf32>
# CHECK:           arith.addf %[[ACC]], %[[L]] : vector<4xf32>
# CHECK:           scf.yield %{{.+}} : vector<4xf32>
# CHECK:         scf.if %{{.+}} -> (vector<4xf32>) {
# EXEC:          RESULT: 140.0
arr = np.arange(12, dtype=np.float32)
print("RESULT:", vector_carry(3, 1.0, arr))  # (4 + 66) * 2


@m.jit
def join_lanes(n: I32, a: F32) -> F32:
    v = V.splat(a, 4)
    if n > 2:
        v = V.splat(a, 2)  # vector<2xf32> vs vector<4xf32>
    return v.sum()


@m.jit
def return_vector(a: F32):
    return V.splat(a, 2)  # no host representation


err("join lanes", lambda: join_lanes(3, 1.0))
err("return vector", lambda: return_vector(1.0))
# CHECK: join lanes: TYPE_UNSTABLE_JOIN
# CHECK: return vector: TYPE_RETURN_MISMATCH


def staged_vec(label, body):
    @m.jit(preprocess=False)
    def f(a: I32, g: F32):
        return body(a, g, V([a, a]))

    err(label, lambda: f(1, 2.0))


staged_vec("float literal", lambda a, g, v: v + 1.5)
staged_vec("scalar dtype", lambda a, g, v: v + g)
staged_vec("lane count", lambda a, g, v: v + V.splat(a, 3))
staged_vec("mixed lanes", lambda a, g, v: V([a, g]))
staged_vec("lane oob", lambda a, g, v: v[2])
staged_vec("lane staged", lambda a, g, v: v[a])
staged_vec("empty", lambda a, g, v: V([]))
staged_vec("splat staged lanes", lambda a, g, v: V.splat(a, a))
staged_vec("wrap scalar", lambda a, g, v: V(a.ir_value()))
staged_vec("bool", lambda a, g, v: bool(v))
# CHECK: float literal: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: scalar dtype: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: lane count: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: mixed lanes: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: lane oob: ARG_ANNOTATION_MISMATCH
# CHECK: lane staged: PHASE_DYNAMIC_INDEX
# CHECK: empty: CALL_MISSING_ARG
# CHECK: splat staged lanes: PHASE_REQUIRES_CONSTANT
# CHECK: wrap scalar: TYPE_UNSUPPORTED_MLIR_TYPE
# CHECK: bool: PHASE_DYNAMIC_TO_STATIC_BOOL

# ===== The torch dtype bridge needs no torch ==============================
# `from_torch_dtype` matches a `torch.dtype` by its printed name, so the
# `torch.Tensor` adapter can type a tensor without importing torch.
ft = lambda s: name(from_torch_dtype(s))
show("TORCH", ft("torch.float32"), ft("bool"), ft("bfloat16"), ft("float8_e4m3fn"))
show("TORCH narrow", ft("torch.float4_e2m1fn_x2"), ft("int4"), "torch" in sys.modules)
err("from_torch_dtype(complex64)", lambda: from_torch_dtype("torch.complex64"))
# CHECK: TORCH: Float32 Boolean BFloat16 Float8E4M3FN
# CHECK: TORCH narrow: Float4E2M1FN Int4 False
# CHECK: from_torch_dtype(complex64): TYPE_UNKNOWN_DTYPE_NAME

# Known defects, reported separately and not asserted here: `inttoptr(p, ...)`
# with a Pointer operand builds invalid IR instead of `ARG_NOT_NUMERIC`; `p - q`
# on two Pointers leaks Python's TypeError; `Cls.mlir_type` outside a context
# leaks the bindings' RuntimeError; a `Vector`-typed struct field is accepted at
# declaration; another `@struct` class is accepted for a struct-annotated
# parameter; `grid_constant` on a `Pointer[T]` parameter is dropped.
