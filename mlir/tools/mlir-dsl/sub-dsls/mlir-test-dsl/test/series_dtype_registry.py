# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
# The scalar type system: the dtype catalogue the DSL defines (MLIR
# spelling, width, ctypes/NumPy names, lookups, the open metaclass), the
# literal and mixed-type promotion rules, the operator-to-op selection for
# signed/unsigned/float operands (arithmetic, comparisons, bitwise, shifts),
# the cast table, Boolean semantics, Meta and `Annotated` parameters,
# scalar marshalling and the `TYPE_*`/`ARG_*` error and warning paths. The
# semantics of the emitted ops are MLIR's business and are not checked here.
import ctypes
import warnings
from typing import Annotated as A

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.core.common import active_dsl
from mlir import ir

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
