# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The scalar type system: the dtype catalogue the DSL defines (MLIR
# spelling, width, ctypes/NumPy names, lookups, the open metaclass), the
# literal and mixed-type promotion rules, the operator-to-op selection for
# signed/unsigned/float operands (arithmetic, comparisons, bitwise, shifts),
# the cast table, Boolean semantics, Meta and `GridConstant` parameters,
# scalar marshalling and the `TYPE_*`/`ARG_*` error and warning paths. The
# semantics of the emitted ops are MLIR's business and are not checked here.
import ctypes
import warnings
from typing import Annotated as A

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.core.common import active_dsl
from mlir import ir
from mlir.dsl.plugins.decorators.kernels.gpu import GridConstant as GC

I8, I16, I32, I64 = m.Int8, m.Int16, m.Int32, m.Int64
U8, U32, U64 = m.Uint8, m.Uint32, m.Uint64
F16, BF16, F32, TF32, F64 = m.Float16, m.BFloat16, m.Float32, m.TFloat32, m.Float64
B, FP8 = m.Boolean, m.Float8E4M3FN
back, npd, lit = m.Numeric.from_mlir_type, m.from_numpy_dtype, m.as_numeric


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


# --- The dtype catalogue: every dtype's MLIR spelling, width, byte size,
# ctypes width and NumPy name, as `Name=mlir/width/bytes/ctype/numpy`.
def row(dt):
    ct = "-" if dt.ctype is None else f"c{ctypes.sizeof(dt.ctype) * 8}"
    return f"{name(dt)}={dt.mlir_type}/{dt.width}/{dt.bytes}/{ct}/{dt._np_dtype_name}"


def family(label, pred):
    dts = sorted(filter(pred, m.ALL_DTYPES), key=lambda t: (t.width, name(t)))
    print(f"{label}:", *(row(dt) for dt in dts))


# The MLIR spelling of a dtype is the tracing DSL's `type_ops` answer, so the
# catalogue is read with the test DSL active.
with ir.Context(), active_dsl(m.MlirTestDSL()):
    family("SIGNED", lambda t: t.is_integer and t.signed and t is not B)
    family("UNSIGNED", lambda t: t.is_integer and not t.signed)
    family("FLOAT", lambda t: t.is_float and t.width >= 16)
    family("NARROW", lambda t: t.is_float and t.width < 16)
    print("BOOL:", row(B), len(m.ALL_DTYPES))
# CHECK: SIGNED: Int2=i2/2/1/-/None Int4=i4/4/1/-/None Int8=i8/8/1/c8/int8 Int16=i16/16/2/c16/int16 Int32=i32/32/4/c32/int32 Int64=i64/64/8/c64/int64 Int128=i128/128/16/-/None
# CHECK: UNSIGNED: Uint8=i8/8/1/c8/uint8 Uint16=i16/16/2/c16/uint16 Uint32=i32/32/4/c32/uint32 Uint64=i64/64/8/c64/uint64 Uint128=i128/128/16/-/None
# CHECK: FLOAT: BFloat16=bf16/16/2/c16/bfloat16 Float16=f16/16/2/c16/float16 Float32=f32/32/4/c32/float32 TFloat32=tf32/32/4/-/None Float64=f64/64/8/c64/float64
# CHECK: NARROW: Float4E2M1FN=f4E2M1FN/4/1/-/None Float6E2M3FN=f6E2M3FN/6/1/-/None Float6E3M2FN=f6E3M2FN/6/1/-/None Float8E3M4=f8E3M4/8/1/-/None Float8E4M3=f8E4M3/8/1/-/None Float8E4M3B11FNUZ=f8E4M3B11FNUZ/8/1/-/None Float8E4M3FN=f8E4M3FN/8/1/-/None Float8E4M3FNUZ=f8E4M3FNUZ/8/1/-/None Float8E5M2=f8E5M2/8/1/-/None Float8E5M2FNUZ=f8E5M2FNUZ/8/1/-/None Float8E5M3FNU=f8E5M3FNU/8/1/-/None Float8E8M0FNU=f8E8M0FNU/8/1/-/None
# CHECK: BOOL: Boolean=i1/1/1/c8/bool_ 30

# The lookups: `from_mlir_type` inverts `mlir_type` (a signless integer maps
# to the signed dtype of its width, the explicitly signed/unsigned MLIR types
# to their own); `dtype()` indexes the class names, `from_numpy_dtype()` the
# NumPy spellings (strings or `np.dtype`s); the abstract bases are not dtypes.
with ir.Context(), active_dsl(m.MlirTestDSL()):
    si32, ui8 = ir.IntegerType.get_signed(32), ir.IntegerType.get_unsigned(8)
    show("FROM_MLIR", name(back(U8.mlir_type)), name(back(FP8.mlir_type)))
    show("FROM_MLIR si/ui", name(back(si32)), name(back(ui8)))
show("LOOKUP", m.dtype("Uint128") is m.Uint128, m.dtype("Boolean") is B)
show("NP", npd("bool") is B, npd(np.dtype("int16")) is I16, npd("bfloat16") is BF16)
show("ABSTRACT", m.Numeric.is_abstract, I32.is_abstract, m.Integer in m.ALL_DTYPES)
# CHECK: FROM_MLIR: Int8 Float8E4M3FN
# CHECK: FROM_MLIR si/ui: Int32 Uint8
# CHECK: LOOKUP: True True
# CHECK: NP: True True True
# CHECK: ABSTRACT: True False False

# --- Metaclass facts: kind, ranges, float format, `recast_width` (the family
# change the promotion rule relies on) and `Cls.isinstance`: a Numeric of
# exactly this dtype or a Python scalar of the right kind (bool is an integer).
rc = lambda t, w: name(t.recast_width(w))
isi = I32.isinstance
show("META", type(I32) is m.IntegerMeta, type(F32) is m.FloatMeta)
show("KIND", issubclass(B, m.Integer), I32.is_same_kind(U8), I32.is_same_kind(F32))
show("BYTES", m.Int4.n_bytes(4), I32.n_bytes(10), m.Int128.max == 2**127 - 1)
show("RANGE", I8.min, I8.max, U8.max, m.Int4.min, m.Int4.max, B.width, B.signed)
show("FORMAT", F16.exponent_width, F16.mantissa_width, BF16.mantissa_width)
show("FORMAT tf32", TF32.mantissa_width, FP8.exponent_width, FP8.mantissa_width)
show("RECAST", rc(I8, 64), rc(BF16, 64), rc(F32, 16))
show("ISINSTANCE", isi(I32(1)), isi(U32(1)), isi(5), isi(True), isi(5.0), isi("x"))
# CHECK: META: True True
# CHECK: KIND: True True False
# CHECK: BYTES: 4 40 True
# CHECK: RANGE: -128 127 255 -8 7 1 True
# CHECK: FORMAT: 5 10 7
# CHECK: FORMAT tf32: 10 4 3
# CHECK: RECAST: Int64 Float64 Float16
# CHECK: ISINSTANCE: True False True True False False


# --- The metaclass is open: a sub-DSL dtype registers itself in ALL_DTYPES and
# both lookups and promotes by the same rules.
class Int24(
    m.Integer,
    metaclass=m.IntegerMeta,
    width=24,
    signed=True,
    mlir_type=lambda: ir.IntegerType.get_signless(24),
):
    ...


x24 = Int24(5)
with ir.Context(), active_dsl(m.MlirTestDSL()):
    show("SUBDSL", Int24 in m.ALL_DTYPES, m.dtype("Int24") is Int24, Int24.bytes)
    show("SUBDSL range", Int24.min, Int24.max, Int24.ctype, Int24._np_dtype_name)
    show("SUBDSL lookup", back(ir.IntegerType.get_signless(24)) is Int24)
show("SUBDSL arith", x24 + Int24(1), x24 + 1, x24 + I8(1), x24 + I64(1), x24 < 6)
# CHECK: SUBDSL: True True 3
# CHECK: SUBDSL range: -8388608 8388607 None None
# CHECK: SUBDSL lookup: True
# CHECK: SUBDSL arith: Int24(6) Int32(6) Int24(6) Int64(6) Boolean(True)

# --- The literal rule: bool -> Boolean, int -> Int32 (Int64 when it does not
# fit), float -> Float32; a Numeric passes through.
show("LIT", lit(True), lit(7), lit(2**31 - 1), lit(-(2**31)), lit(2.5), lit(U8(3)))
show("LIT wide", lit(2**31), lit(-(2**31) - 1))
# CHECK: LIT: Boolean(True) Int32(7) Int32(2147483647) Int32(-2147483648) Float32(2.5) Uint8(3)
# CHECK: LIT wide: Int64(2147483648) Int64(-2147483649)

# --- The mixed-type rule (`_binary_op_type_promote`), folded on payloads:
# integers of mixed signedness pick unsigned when its width >= the signed
# width, else signed; any float recasts the integer side to the IEEE float of
# max(width) and the wider float wins (Float32 over TFloat32 at equal width);
# a Boolean is a 1-bit signed integer, except Boolean op Boolean, which
# computes in Int32 and re-wraps as Boolean; `/` on integers is Float32.
INTS = [I8, I32, I64, U8, U32, U64]
FLOATS = [F16, BF16, F32, TF32, F64]
show("INT+int", *(dt(1) + 2 for dt in INTS))
show("INT+float", *(dt(1) + 0.5 for dt in INTS))
show("INT+bool", *(dt(1) + True for dt in INTS))
show("FLOAT+int", *(dt(1.0) + 2 for dt in FLOATS))
show("FLOAT+bool", *(dt(1.0) + True for dt in FLOATS))
# CHECK: INT+int: Int32(3) Int32(3) Int64(3) Int32(3) Uint32(3) Uint64(3)
# CHECK: INT+float: Float32(1.5) Float32(1.5) Float64(1.5) Float32(1.5) Float32(1.5) Float64(1.5)
# CHECK: INT+bool: Int8(2) Int32(2) Int64(2) Uint8(2) Uint32(2) Uint64(2)
# CHECK: FLOAT+int: Float32(3.0) Float32(3.0) Float32(3.0) Float32(3.0) Float64(3.0)
# CHECK: FLOAT+bool: Float16(2.0) BFloat16(2.0) Float32(2.0) TFloat32(2.0) Float64(2.0)
t = B(True)
show("WIDE", I32(1) + 2**40, U32(1) + 2**40, F32(1.0) + 2**40, 2**40 + I32(1))
show("MIX", I32(1) + U32(2), I32(1) + U8(2), I8(1) + U32(2), I16(1) + I8(2))
show("MIX wide", I64(1) + U32(2), I32(1) + U64(2), U8(1) + I8(2))
show("IF", I32(1) + F32(2.0), I64(1) + F32(2.0), I8(1) + F16(2.0), I32(1) + FP8(2.0))
show("IF narrow", I16(1) + BF16(2.0), FP8(1.0) + FP8(2.0))
show("FF", F16(1.0) + F32(2.0), F32(1.0) + TF32(2.0), F16(1.0) + BF16(2.0))
show("FF tf32", TF32(1.0) + F16(2.0), BF16(1.0) + F64(2.0))
show("BOOL", t + True, t + t, t * B(False), t + 2, t + 2.5, I32(1) + t, True + F16(1.0))
show("DIV", I32(7) / 2, I32(8) / 2, F64(1.0) / 3, 2 + U8(1))
# CHECK: WIDE: Int64(1099511627777) Int64(1099511627777) Float64(1099511627777.0) Int64(1099511627777)
# CHECK: MIX: Uint32(3) Int32(3) Uint32(3) Int16(3)
# CHECK: MIX wide: Int64(3) Uint64(3) Uint8(3)
# CHECK: IF: Float32(3.0) Float64(3.0) Float16(3.0) Float32(3.0)
# CHECK: IF narrow: Float16(3.0) Float8E4M3FN(3.0)
# CHECK: FF: Float32(3.0) Float32(3.0) Float16(3.0)
# CHECK: FF tf32: TFloat32(3.0) Float64(3.0)
# CHECK: BOOL: Int32(2) Boolean(True) Boolean(False) Int32(3) Float32(3.5) Int32(2) Float16(2.0)
# CHECK: DIV: Float32(3.5) Float32(4.0) Float64(0.3333333333333333) Int32(3)
# Two different narrow formats, or an integer whose width has no IEEE float
# (Int8 + fp8), have no common type: the user converts explicitly.
err("NARROW fp8+fp8", lambda: FP8(1.0) + m.Float8E5M2(2.0))
err("NARROW Int8+fp8", lambda: I8(1) + m.Float8E5M2(2.0))
# CHECK: NARROW fp8+fp8: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK: NARROW Int8+fp8: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED


# --- On staged values the same rule picks the widening op from the source
# signedness and kind (`extui`/`extsi`/`extf`/`sitofp`); a literal becomes a
# constant of the promoted dtype; equal-width signed/unsigned needs no op.
@m.jit
def staged_promote(a: I64, f: F32, u: U32, h: F16, b: I8) -> F64:
    x = a + f  # Int64 + Float32 -> Float64
    y = u + a  # Uint32 + Int64 -> Int64: zero-extend the unsigned side
    z = h + 2  # Float16 + Int32 literal -> Float32: extf, f32 constant
    w = b + a  # Int8 + Int64 -> Int64: sign-extend
    v = u + 1  # Uint32 + Int32 literal -> Uint32: equal width, unsigned wins
    return x + F64(y) + F64(z) + F64(w) + F64(v)


# CHECK-LABEL: func.func @staged_promote(
# CHECK-SAME:    %[[A:[^:]+]]: i64, %[[F:[^:]+]]: f32, %[[U:[^:]+]]: i32, %[[H:[^:]+]]: f16, %[[B:[^:]+]]: i8) -> f64
# CHECK:         %[[AF:.+]] = arith.sitofp %[[A]] : i64 to f64
# CHECK:         %[[FF:.+]] = arith.extf %[[F]] : f32 to f64
# CHECK:         arith.addf %[[AF]], %[[FF]] : f64
# CHECK:         %[[UW:.+]] = arith.extui %[[U]] : i32 to i64
# CHECK:         arith.addi %[[UW]], %[[A]] : i64
# CHECK:         %[[HW:.+]] = arith.extf %[[H]] : f16 to f32
# CHECK:         %[[C2:.+]] = arith.constant 2.000000e+00 : f32
# CHECK:         arith.addf %[[HW]], %[[C2]] : f32
# CHECK:         %[[BW:.+]] = arith.extsi %[[B]] : i8 to i64
# CHECK:         arith.addi %[[BW]], %[[A]] : i64
# CHECK:         %[[C1:.+]] = arith.constant 1 : i32
# CHECK:         arith.addi %[[U]], %[[C1]] : i32
# EXEC:          RESULT: 4294967299.0
print("RESULT:", staged_promote(1, 0.5, 2**32 - 1, 1.5, -3))


# --- Operator-to-op selection (`plugins/type_ops/arith.py`, through the
# `UpstreamDialectTypeOps` plugin): the promoted dtype
# picks the signed, unsigned or float form; `//` on floats is `divf` +
# `math.floor`; `-x` on an integer is `0 - x`; `~x` is `xor(x, -1)`; `>>` is
# arithmetic for signed and logical for unsigned dtypes; `**` is `math.powf`
# after promotion (two staged integers are an error, below).
@m.jit
def int_ops(a: I32, b: I32, u: U32, v: U32) -> I32:
    s = (a // b) + (a % b) + (a >> b) + (-a) + abs(b) + (a & b) + (~a)
    return s + I32((u // v) + (u % v) + (u >> v))


# CHECK-LABEL: func.func @int_ops(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32, %[[U:[^:]+]]: i32, %[[V:[^:]+]]: i32) -> i32
# CHECK:         arith.floordivsi %[[A]], %[[B]] : i32
# CHECK:         arith.remsi %[[A]], %[[B]] : i32
# CHECK:         arith.shrsi %[[A]], %[[B]] : i32
# CHECK:         %[[Z:.+]] = arith.constant 0 : i32
# CHECK:         arith.subi %[[Z]], %[[A]] : i32
# CHECK:         math.absi %[[B]] : i32
# CHECK:         arith.andi %[[A]], %[[B]] : i32
# CHECK:         %[[ONES:.+]] = arith.constant -1 : i32
# CHECK:         arith.xori %[[A]], %[[ONES]] : i32
# CHECK:         arith.divui %[[U]], %[[V]] : i32
# CHECK:         arith.remui %[[U]], %[[V]] : i32
# CHECK:         arith.shrui %[[U]], %[[V]] : i32
# EXEC:          RESULT: -1073741833
print("RESULT:", int_ops(7, 2, 2**32 - 4, 2))  # -6 + (3221225469 as i32)


@m.jit
def float_ops(a: F32, b: F32, n: I32) -> F32:
    return (a // b) + (a % b) + (-a) + abs(b) + a**b + a**n + a**0.5


# CHECK-LABEL: func.func @float_ops(
# CHECK-SAME:    %[[A:[^:]+]]: f32, %[[B:[^:]+]]: f32, %[[N:[^:]+]]: i32) -> f32
# CHECK:         %[[Q:.+]] = arith.divf %[[A]], %[[B]] : f32
# CHECK:         math.floor %[[Q]] : f32
# CHECK:         arith.remf %[[A]], %[[B]] : f32
# CHECK:         arith.negf %[[A]] : f32
# CHECK:         math.absf %[[B]] : f32
# CHECK:         math.powf %[[A]], %[[B]] : f32
# CHECK:         %[[NF:.+]] = arith.sitofp %[[N]] : i32 to f32
# CHECK:         math.powf %[[A]], %[[NF]] : f32
# CHECK:         %[[CH:.+]] = arith.constant 5.000000e-01 : f32
# CHECK:         math.powf %[[A]], %[[CH]] : f32
# EXEC:          RESULT: 82.0
print("RESULT:", float_ops(4.0, 2.0, 3))  # 2 + 0 - 4 + 2 + 16 + 64 + 2


# Comparisons yield `Boolean`; the predicate follows the promoted dtype:
# `cmpi s*`, `cmpi u*`, ordered `cmpf o*` except `!=`, unordered so that NaN
# compares unequal as in Python.
@m.jit
def compare(a: I32, u: U32, f: F32) -> I32:
    r = I32(a < 10) + I32(a >= u)  # Int32 vs Uint32 -> unsigned compare
    return r + I32(u <= u) + I32(f > 2.5) + I32(f == a) + I32(f != f)


# CHECK-LABEL: func.func @compare(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[U:[^:]+]]: i32, %[[F:[^:]+]]: f32) -> i32
# CHECK:         arith.cmpi slt, %[[A]], %{{.+}} : i32
# CHECK:         arith.cmpi uge, %[[A]], %[[U]] : i32
# CHECK:         arith.cmpi ule, %[[U]], %[[U]] : i32
# CHECK:         arith.cmpf ogt, %[[F]], %{{.+}} : f32
# CHECK:         %[[AF:.+]] = arith.sitofp %[[A]] : i32 to f32
# CHECK:         arith.cmpf oeq, %[[F]], %[[AF]] : f32
# CHECK:         arith.cmpf une, %[[F]], %[[F]] : f32
# EXEC:          RESULT: 4 2
print("RESULT:", compare(-1, 1, float("nan")), compare(20, 1, 2.0))
show("CMP fold", I32(3) < 4, F32(float("nan")) != 1.0, 5 > I32(3), I32(3) < U8(4))
# CHECK: CMP fold: Boolean(True) Boolean(True) Boolean(True) Boolean(True)

# --- Payload folding (`_binary_op` on two Python values) and the Python
# protocol of a compile-time Numeric (`bool`, `index`, `hash`, `to()`).
a7 = I32(7)
show("FOLD", a7 + 3, a7 // 2, a7 % 2, a7 / 2, a7**2, -a7, abs(I32(-7)))
show("FOLD bits", a7 & 3, a7 << 2, I32(-8) >> 1, ~U8(5), U8(250) + U8(3), 1 << a7)
show("PROTO", bool(a7), bool(F32(0.0)), [0, 1, 2, 3, 4, 5, 6, 70][a7], str(a7))
show("PROTO hash", hash(I32(3)) == hash(U32(3)), hash(I32(3)) == hash(I32(3)))
show("PROTO to", a7.to(F32), I32(300).to(I8), F32(2.9).to(int), a7.to(I32) is a7)
# CHECK: FOLD: Int32(10) Int32(3) Int32(1) Float32(3.5) Int32(49) Int32(-7) Int32(7)
# CHECK: FOLD bits: Int32(3) Int32(28) Int32(-4) Uint8(250) Uint8(253) Int32(128)
# CHECK: PROTO: True False 70 7
# CHECK: PROTO hash: False True
# CHECK: PROTO to: Float32(7.0) Int8(44) 2 True
try:
    I32(1) + "x"  # a non-numeric operand is left to Python's own TypeError
except TypeError:
    print("PY TYPEERROR")
# CHECK: PY TYPEERROR


# --- Boolean: `and`/`or`/`not` go through the executor contract
# (`__dsl_and__`/`__dsl_or__`/`__dsl_not__`): a select on `x != 0` (`cmpf
# une` for floats) with an `arith.andi` fast path for Boolean and Boolean;
# `-Boolean` is a user error.
@m.jit
def logic(a: I32, b: I32, f: F32, p: B, q: B) -> I32:
    x = a and b  # select(a != 0, b, a)
    y = a or b  # select(a != 0, a, b)
    z = not a  # a == 0
    w = p and q  # Boolean and Boolean: one andi on i1
    g = f or 1.0  # floats test with `cmpf une`
    return x * 1000 + y * 100 + I32(z) * 10 + I32(w) + I32(g)


# CHECK-LABEL: func.func @logic(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32, %[[F:[^:]+]]: f32, %[[P:[^:]+]]: i1, %[[Q:[^:]+]]: i1) -> i32
# CHECK:         %[[T0:.+]] = arith.cmpi ne, %[[A]], %{{.+}} : i32
# CHECK:         arith.select %[[T0]], %[[B]], %[[A]] : i32
# CHECK:         %[[T1:.+]] = arith.cmpi ne, %[[A]], %{{.+}} : i32
# CHECK:         arith.select %[[T1]], %[[A]], %[[B]] : i32
# CHECK:         arith.cmpi eq, %[[A]], %{{.+}} : i32
# CHECK:         arith.andi %[[P]], %[[Q]] : i1
# CHECK:         %[[TF:.+]] = arith.cmpf une, %[[F]], %{{.+}} : f32
# CHECK:         arith.select %[[TF]], %[[F]], %{{.+}} : f32
# EXEC:          RESULT: 712 7302
print("RESULT:", logic(0, 7, 0.0, True, True), logic(3, 7, 2.0, True, False))
show("DSL_BOOL", I32(5).__dsl_and__(3), I32(0).__dsl_or__(3), B(True).__dsl_not__())
show("DSL_BOOL float", F32(-0.1).__dsl_bool__(), F32(0.0).__dsl_not__())
err("-Boolean", lambda: -B(True))
# CHECK: DSL_BOOL: Int32(3) Int32(3) False
# CHECK: DSL_BOOL float: Boolean(True) True
# CHECK: -Boolean: <free-form>


# --- Casts: a dtype constructor picks the `arith` cast from the source and
# target signedness and kind (Boolean zero-extends, `Boolean(x)` compares
# with zero); `cast()` accepts an abstract target the value already
# satisfies; a raw `ir.Value` of a scalar type wraps into any dtype. Payload
# conversions run in Python with the C-cast wrap.
@m.jit
def casts(a: I32, f: F32, p: B) -> I64:
    x = I64(a)  # signed widening: extsi
    y = I8(a)  # narrowing: trunci
    z = I64(U32(a))  # same width: only the dtype changes; then extui
    v = I32(f)  # fptosi
    w = U32(f)  # fptoui
    i = I32(p)  # i1 zero-extends: True is 1, never -1
    g = I64(F32(p) + F64(f))  # uitofp, extf, then fptosi
    b = B(a)  # cmpi ne 0
    c = m.cast(a, m.Integer)  # already an Integer: no op
    r = I64(a.ir_value())  # a raw i32 value wraps into any dtype
    return x + I64(y) + z + I64(v) + I64(w) + I64(i) + g + I64(b) + I64(c) + r


# CHECK-LABEL: func.func @casts(
# CHECK-SAME:    %[[A:[^:]+]]: i32, %[[F:[^:]+]]: f32, %[[P:[^:]+]]: i1) -> i64
# CHECK:         arith.extsi %[[A]] : i32 to i64
# CHECK:         arith.trunci %[[A]] : i32 to i8
# CHECK:         arith.extui %[[A]] : i32 to i64
# CHECK:         arith.fptosi %[[F]] : f32 to i32
# CHECK:         arith.fptoui %[[F]] : f32 to i32
# CHECK:         arith.extui %[[P]] : i1 to i32
# CHECK:         %[[PF:.+]] = arith.uitofp %[[P]] : i1 to f32
# CHECK:         %[[FD:.+]] = arith.extf %[[F]] : f32 to f64
# CHECK:         %[[PD:.+]] = arith.extf %[[PF]] : f32 to f64
# CHECK:         %[[G:.+]] = arith.addf %[[PD]], %[[FD]] : f64
# CHECK:         arith.fptosi %[[G]] : f64 to i64
# CHECK:         arith.cmpi ne, %[[A]], %{{.+}} : i32
# CHECK:         arith.extsi %[[A]] : i32 to i64
# EXEC:          RESULT: 1268
print("RESULT:", casts(300, 7.9, True))  # 300+44+300+7+7+1+8+1+300+300
show("FOLD cast", I32(3.7), I32(-3.7), I8(I32(300)), U8(I32(-1)), m.Int4(I32(9)))
show("FOLD cast wide", U64(I32(-1)), B(-0.5), F16(3), B(I32(0)), I32(True))
show("FOLD cast()", m.cast(5, I32), m.cast(I32(5), m.Integer), m.cast(2.5, F16))
err("cast abstract", lambda: m.cast(F32(1.0), m.Integer))
# CHECK: FOLD cast: Int32(3) Int32(-3) Int8(44) Uint8(255) Int4(-7)
# CHECK: FOLD cast wide: Uint64(18446744073709551615) Boolean(True) Float16(3.0) Boolean(False) Int32(1)
# CHECK: FOLD cast(): Int32(5) Int32(5) Float16(2.5)
# CHECK: cast abstract: <free-form>


# --- A parameter whose annotation is not a DSL type (`int`, `str`, none) is
# a Meta value: no block argument, the trace sees the Python value, which
# folds into constants and into the mangled symbol, so each value is its own
# specialization; a Python `if`/`range` on it is decided/unrolled at trace
# time. (A `Numeric` instance is always staged: there is no `Constexpr`.)
@m.jit
def meta(a: I32, n: int, mode: str, scale) -> I32:
    print("META:", type(n).__name__, type(mode).__name__, type(scale).__name__)
    acc = a * scale
    for i in range(n):  # unrolled
        acc = acc + i
    if mode == "double":  # decided at trace time
        acc = acc * 2
    return acc


# CHECK:       META: int str int
# CHECK-LABEL: func.func @meta_3_double_2(
# CHECK-SAME:    %[[A:[^:]+]]: i32) -> i32
# CHECK-NOT:     scf.
# CHECK:         %[[C2:.+]] = arith.constant 2 : i32
# CHECK:         %[[M:.+]] = arith.muli %[[A]], %[[C2]] : i32
# CHECK-COUNT-3: arith.addi
# CHECK:         arith.muli %{{.+}}, %{{.+}} : i32
# CHECK-NOT:     scf.
# CHECK:         return
# EXEC:          META: int str int
# EXEC:          RESULT: 26
print("RESULT:", meta(5, 3, "double", 2))  # (5*2 + 0 + 1 + 2) * 2
# CHECK-LABEL: func.func @meta_1_single_2(
# EXEC:          RESULT: 10
print("RESULT:", meta(5, 1, "single", 2))


# `GridConstant[T]` is `Annotated[T, grid_constant]` and tags the argument
# `{cuda.grid_constant}`; other `Annotated` metadata contributes nothing.
@m.jit
def grid(a: GC[I32], b: A[F32, m.grid_constant], c: A[I32, "doc"]) -> F32:
    return F32(a + c) + b


# CHECK-LABEL: func.func @grid(
# CHECK-SAME:    %[[A:[^:]+]]: i32 {cuda.grid_constant}, %[[B:[^:]+]]: f32 {cuda.grid_constant}, %[[C:[^:]+]]: i32) -> f32
# EXEC:          RESULT: 3.5
# (Numeric arguments: a plain `int` under `Annotated[...]` is a known defect.)
print("RESULT:", grid(I32(1), F32(0.5), I32(2)))


# --- Out-of-range literals: an explicit construction narrows with the C-cast
# wrap and warns (`DSLWarning`, a `WarnId.TYPE_*`); in-range values, the
# bit-pattern idioms (`Int32(0xFFFFFFFF)`, `Uint8(-1)`, the unsigned range of
# the width) and Numeric-to-Numeric wrapping are silent; a float literal into
# a float warns only when it collapses to inf or 0 (never for the non-IEEE
# formats); NaN/inf into an integer is an error.
def construct(label, fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    codes = ",".join(w.message.warn_id.name for w in caught) or "-"
    print(f"{label}: {value!r} {codes}")


construct("Int8(255)", lambda: I8(255))
construct("Int8(256)", lambda: I8(256))
construct("Uint8(-1)", lambda: U8(-1))
construct("Uint8(256)", lambda: U8(256))
construct("Int32(2**40)", lambda: I32(2**40))
construct("Int32(3e9)", lambda: I32(3e9))
construct("Float32(1e40)", lambda: F32(1e40))
construct("Float32(1e-50)", lambda: F32(1e-50))
construct("BFloat16(1e40)", lambda: BF16(1e40))
construct("Int8(Int32(300))", lambda: I8(I32(300)))
# CHECK:      Int8(255): Int8(-1) -
# CHECK-NEXT: Int8(256): Int8(0) TYPE_INT_LITERAL_OUT_OF_RANGE
# CHECK-NEXT: Uint8(-1): Uint8(255) -
# CHECK-NEXT: Uint8(256): Uint8(0) TYPE_INT_LITERAL_OUT_OF_RANGE
# CHECK-NEXT: Int32(2**40): Int32(0) TYPE_INT_LITERAL_OUT_OF_RANGE
# CHECK-NEXT: Int32(3e9): Int32(-1294967296) TYPE_FLOAT_TO_INT_OUT_OF_RANGE
# CHECK-NEXT: Float32(1e40): Float32(1e+40) TYPE_FLOAT_LITERAL_OVERFLOW
# CHECK-NEXT: Float32(1e-50): Float32(1e-50) TYPE_FLOAT_LITERAL_UNDERFLOW
# CHECK-NEXT: BFloat16(1e40): BFloat16(1e+40) -
# CHECK-NEXT: Int8(Int32(300)): Int8(44) -
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    I8(256)
w0 = caught[0].message
show("WARNING", type(w0).__name__, isinstance(w0, UserWarning), w0.code)
print(str(w0))
err("Int32(nan)", lambda: I32(float("nan")))
# CHECK:      WARNING: DSLWarning True TYPE_INT_LITERAL_OUT_OF_RANGE
# CHECK:      warning[TYPE_INT_LITERAL_OUT_OF_RANGE]:{{.*}} The Python integer 256 does not fit in `Int8` (range [-128, 127]).
# CHECK:      suggestion:{{.*}}mask to the type width
# CHECK:      Int32(nan): <free-form>

# --- `TYPE_*`/`ARG_*` user errors are catalogued `DSLUserCodeError`s, never
# builtin exceptions, rendered with the user's frame and suggestions.
err("Int32(str)", lambda: I32("5"))
err("dtype(float32)", lambda: m.dtype("float32"))
err("from_numpy_dtype(complex64)", lambda: npd("complex64"))
with ir.Context(), active_dsl(m.MlirTestDSL()):
    err("from_mlir_type(index)", lambda: back(ir.IndexType.get()))
err("align(6)", lambda: m.align(6))
# CHECK: Int32(str): ARG_NOT_NUMERIC
# CHECK: dtype(float32): TYPE_UNKNOWN_DTYPE_NAME
# CHECK: from_numpy_dtype(complex64): TYPE_UNKNOWN_DTYPE_NAME
# CHECK: from_mlir_type(index): TYPE_UNSUPPORTED_MLIR_TYPE
# CHECK: align(6): ARG_INVALID_ALIGNMENT
print(str(err("rendered", lambda: I8(1) * FP8(1.0))))
# CHECK:      rendered: TYPE_IMPLICIT_PROMOTION_UNSUPPORTED
# CHECK:      error[TYPE_IMPLICIT_PROMOTION_UNSUPPORTED]:{{.*}} `Int8` and `Float8E4M3FN` cannot be combined by `mul` without an explicit conversion
# CHECK:      -->{{.*}}types_numeric.py:[[#@LINE-3]]
# CHECK:      = category: usage (types)
# CHECK:      suggestion:{{.*}}Convert one operand explicitly


@m.jit
def staged_errors(a: I32, b: I32, f: F32) -> I32:
    err("int ** int", lambda: a**b)  # compiled code has no integer power
    err("int ** 2", lambda: a**2)
    err("float ** int", lambda: f**b)
    err("to(int)", lambda: a.to(int))  # needs a compile-time value
    return a


# CHECK: int ** int: TYPE_INT_POW_UNSUPPORTED
# CHECK: int ** 2: TYPE_INT_POW_UNSUPPORTED
# CHECK: float ** int: NO ERROR
# CHECK: to(int): PHASE_REQUIRES_CONSTANT
# CHECK: func.func @staged_errors(
staged_errors(1, 2, 1.5)


# --- Scalar marshalling (`Numeric.marshal`): a dtype with a ctypes
# representative passes through an owning `c_void_p` (a payload widens to the
# dtype); Float16/BFloat16 travel as bit patterns the DSL computes itself
# (f16 rounds and saturates to inf, bf16 truncates); the staged-only dtypes
# are full Numerics in a trace but raise `ARG_UNSUPPORTED_TYPE` at the boundary.
def ms(dt, v):
    c = dt.marshal(v)._keepalive
    return f"{name(dt)}={ctypes.sizeof(c)}:{c.value!r}"


f16 = lambda v: hex(F16._to_ctype(v).value)
bf16 = lambda v: hex(BF16._to_ctype(v).value)
show("MARSHAL", ms(I8, -5), ms(U8, 250), ms(B, True), ms(F32, 2), ms(I32, I32(9)))
show("F16", f16(1.0), f16(65504.0), f16(2.0**-24), f16(1e10), f16(float("nan")))
show("BF16", bf16(1.0), bf16(3.14159), bf16(1e-40), bf16(float("nan")), bf16(-1e40))
# CHECK: MARSHAL: Int8=1:-5 Uint8=1:250 Boolean=1:True Float32=4:2.0 Int32=4:9
# CHECK: F16: 0x3c00 0x7bff 0x1 0x7c00 0x7e00
# CHECK: BF16: 0x3f80 0x4049 0x1 0x7fc0 0xff80
for dt in (m.Int4, m.Int128, TF32, FP8):
    err(f"marshal {name(dt)}", lambda: dt.marshal(dt(1), arg_name="x"))
# CHECK: marshal Int4: ARG_UNSUPPORTED_TYPE
# CHECK: marshal Int128: ARG_UNSUPPORTED_TYPE
# CHECK: marshal TFloat32: ARG_UNSUPPORTED_TYPE
# CHECK: marshal Float8E4M3FN: ARG_UNSUPPORTED_TYPE

# Known defects, reported separately and not asserted here: widening a signed
# staged integer to a wider unsigned dtype emits `extui` instead of `extsi`;
# `ARG_UNSUPPORTED_TYPE` for a staged-only `@jit` parameter names `value`
# instead of the parameter; a plain `int` under `Annotated[Int32, ...]` is
# rejected (`ARG_UNSUPPORTED_TYPE`) where a bare `Int32` annotation casts it.
