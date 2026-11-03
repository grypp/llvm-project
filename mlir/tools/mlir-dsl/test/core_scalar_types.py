# RUN: %PYTHON %s 2>&1 | FileCheck %s
# S3: the scalar types over block arguments of a hand-built func.func. The
# types reach their emitter through `current_emitter()`, which reads the
# active DSL's `plugins.type_ops`; no BaseDSL exists yet, so the S1 context
# `active_dsl` is entered with a stand-in carrying that record (and the two
# names the error path reads). The IR is checked first, the text after it.
import os
import warnings
from types import SimpleNamespace

from mlir import ir
from mlir.dialects import func
from mlir.extras import types as T

from mlir.dsl.core.common import DSLUserCodeError, active_dsl
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith
from mlir.dsl.types.typing import (
    ALL_DTYPES,
    Boolean,
    Float,
    Float32,
    Int8,
    Int16,
    Int32,
    Int64,
    Integer,
    IntegerMeta,
    Numeric,
    as_numeric,
    cast,
)

STAND_IN = SimpleNamespace(
    name="S3",
    plugins=Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith)),
    _unavailable_context=lambda: {},
)
text = []


def err(label, fn):
    try:
        fn()
        text.append(f"{label}: NO ERROR")
    except DSLUserCodeError as e:
        text.append(f"{label}: {e.code or '<free-form>'} | {e.message}")


with ir.Context(), ir.Location.unknown(), active_dsl(STAND_IN):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp("types", ([T.i32(), T.i32(), T.f32(), T.i64()], []))
        block = fop.add_entry_block()
        with ir.InsertionPoint(block):
            a = Int32(block.arguments[0])
            b = Int32(block.arguments[1])
            x = Float32(block.arguments[2])
            w = Int64(block.arguments[3])
            text.append(
                f"WRAPPED: {type(a).__name__} {type(x).__name__} {type(w).__name__} {a} {x!r}"
            )
            # CHECK-LABEL: func.func @types(
            # CHECK-SAME: %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32, %[[X:[^:]+]]: f32, %[[W:[^:]+]]: i64)

            # -- same dtype, and a literal promoted to the operand's dtype
            # CHECK: arith.addi %[[A]], %[[B]] : i32
            # CHECK: %[[C1:.+]] = arith.constant 1 : i32
            # CHECK: arith.addi %[[A]], %[[C1]] : i32
            # CHECK: arith.floordivsi %[[A]], %[[B]] : i32
            # CHECK: arith.remsi %[[A]], %[[B]] : i32
            # CHECK: %[[C25:.+]] = arith.constant 2.500000e+00 : f32
            # CHECK: arith.mulf %[[X]], %[[C25]] : f32
            r = a + b
            r2 = a + 1
            a // b
            a % b
            r5 = x * 2.5
            text.append(
                f"RESULTS: {type(r).__name__} {type(r2).__name__} {type(r5).__name__} {r}"
            )
            # -- `/` on integers is a float, as in Python
            # CHECK: %[[AF:.+]] = arith.sitofp %[[A]] : i32 to f32
            # CHECK: %[[BF:.+]] = arith.sitofp %[[B]] : i32 to f32
            # CHECK: arith.divf %[[AF]], %[[BF]] : f32
            text.append(f"TRUEDIV: {type(a / b).__name__}")
            # -- promotion: Int32 beside Float32 -> Float32, Int32 beside Int64 -> Int64
            # CHECK: %[[AF2:.+]] = arith.sitofp %[[A]] : i32 to f32
            # CHECK: arith.addf %[[AF2]], %[[X]] : f32
            # CHECK: %[[AW:.+]] = arith.extsi %[[A]] : i32 to i64
            # CHECK: arith.addi %[[AW]], %[[W]] : i64
            # CHECK: %[[C2:.+]] = arith.constant 2 : i32
            # CHECK: arith.subi %[[C2]], %[[A]] : i32
            text.append(
                f"PROMOTED: {type(a + x).__name__} {type(a + w).__name__} {type(2 - a).__name__}"
            )
            # Float64 exists from the registry PR on: the literal promotes to it.
            text.append(f"INT64 + FLOAT LITERAL: {type(w + 1.5).__name__}")
            # -- comparisons are Boolean
            # CHECK: arith.cmpi slt, %[[A]], %[[B]] : i32
            # CHECK: arith.cmpf une, %[[X]], %[[X]] : f32
            # CHECK: %[[C3:.+]] = arith.constant 3 : i32
            # CHECK: arith.cmpi eq, %[[A]], %[[C3]] : i32
            text.append(
                f"COMPARE: {type(a < b).__name__} {type(x != x).__name__} {type(a == 3).__name__}"
            )
            # -- conversions by construction
            # CHECK: arith.sitofp %[[A]] : i32 to f32
            # CHECK: arith.fptosi %[[X]] : f32 to i32
            # CHECK: arith.extsi %[[A]] : i32 to i64
            # CHECK: arith.trunci %[[W]] : i64 to i32
            # CHECK: %[[Z:.+]] = arith.constant 0 : i32
            # CHECK: arith.cmpi ne, %[[A]], %[[Z]] : i32
            text.append(
                "CONVERT: "
                + " ".join(
                    type(v).__name__
                    for v in (Float32(a), Int32(x), Int64(a), Int32(w), Boolean(a))
                )
            )
            # CHECK: arith.sitofp %[[A]] : i32 to f32
            # CHECK: arith.fptosi %[[X]] : f32 to i32
            text.append(
                f"TO: {type(a.to(Float32)).__name__} {type(x.to(Int32)).__name__} {a.to(Int32) is a}"
            )
            # -- unary, bitwise and shifts
            # CHECK: %[[Z2:.+]] = arith.constant 0 : i32
            # CHECK: arith.subi %[[Z2]], %[[A]] : i32
            # CHECK: arith.negf %[[X]] : f32
            # CHECK: math.absi %[[A]] : i32
            # CHECK: math.absf %[[X]] : f32
            # CHECK: %[[C3B:.+]] = arith.constant 3 : i32
            # CHECK: arith.andi %[[A]], %[[C3B]] : i32
            # CHECK: arith.ori %[[A]], %[[B]] : i32
            # CHECK: arith.xori %[[A]], %[[B]] : i32
            # CHECK: %[[C2B:.+]] = arith.constant 2 : i32
            # CHECK: arith.shli %[[A]], %[[C2B]] : i32
            # CHECK: %[[C1B:.+]] = arith.constant 1 : i32
            # CHECK: arith.shrsi %[[A]], %[[C1B]] : i32
            # CHECK: %[[M1:.+]] = arith.constant -1 : i32
            # CHECK: arith.xori %[[A]], %[[M1]] : i32
            text.append(
                "UNARY/BITS: "
                + " ".join(
                    type(v).__name__
                    for v in (
                        -a,
                        -x,
                        abs(a),
                        abs(x),
                        a & 3,
                        a | b,
                        a ^ b,
                        a << 2,
                        a >> 1,
                        ~a,
                    )
                )
            )
            # -- pow: floats through math.powf, a runtime int ** int is an error
            # CHECK: %[[C2F:.+]] = arith.constant 2.000000e+00 : f32
            # CHECK: math.powf %[[X]], %[[C2F]] : f32
            text.append(f"POW: {type(x ** 2).__name__}")
            err("INT POW", lambda: a**b)
            # -- a raw ir.Value on the right takes the dtype claiming its type
            # CHECK: arith.addi %[[A]], %[[B]] : i32
            text.append(
                f"RAW RHS: {type(a + block.arguments[1]).__name__} {type(as_numeric(block.arguments[3])).__name__}"
            )
            # -- staged values cannot become Python values
            err("INT OF STAGED", lambda: int(a))
            err("BOOL OF STAGED", lambda: bool(a))
            err("INDEX OF STAGED", lambda: [1, 2, 3][a])
            err("TO INT OF STAGED", lambda: a.to(int))
            err("NEG BOOLEAN", lambda: -(a < b))
            # -- the abstract families: a Python number takes the plugin's dtype
            text.append(
                "CAST ABSTRACT: "
                + " ".join(
                    repr(v)
                    for v in (
                        cast(5, Numeric),
                        cast(2.5, Numeric),
                        cast(True, Numeric),
                        cast(2**40, Numeric),
                        cast(3, Float),
                        cast(True, Integer),
                    )
                )
            )
            text.append(
                f"CAST KEEPS: {cast(a, Integer) is a} {cast(x, Numeric) is x} {type(cast(a, Float32)).__name__}"
            )
            err("CAST FLOAT TO INTEGER", lambda: cast(2.5, Integer))
            err("CAST STAGED FLOAT TO INTEGER", lambda: cast(x, Integer))
            text.append(
                f"INFER: {STAND_IN.plugins.type_ops.infer_dtype(7, Numeric).__name__} {STAND_IN.plugins.type_ops.infer_dtype(2.5, Integer)}"
            )
            # -- type-level answers through the emitter
            text.append(
                f"MLIR TYPES: {Int32.mlir_type} {Int64.mlir_type} {Float32.mlir_type} {Boolean.mlir_type}"
            )
            text.append(
                f"FROM MLIR TYPE: {Numeric.from_mlir_type(T.i64()).__name__} {Numeric.from_mlir_type(T.bool()).__name__}"
            )
            text.append(
                f"FROM MLIR TYPE F64: {Numeric.from_mlir_type(T.f64()).__name__}"
            )
            err("INT32 OF F64 VALUE", lambda: Int32(arith.const(1.0, T.f64())))
            # CHECK: return
            func.ReturnOp([])
    module.operation.verify()
    print(module)

# -- folding: two Python payloads never touch the emitter (no DSL is active here)
print(
    "FOLD:",
    Int32(5) + 3,
    Int32(7) // 2,
    Int32(7) % 4,
    Float32(1.0) / 4,
    Int32(5) / 2,
    Int32(5) < 7,
    Int32(-1) + Int64(1),
    -Int32(4),
)
print(
    "FOLD TYPES:",
    type(Int32(5) + 3).__name__,
    type(Int32(5) / 2).__name__,
    type(Int32(5) < 7).__name__,
    type(Int32(-1) + Int64(1)).__name__,
    type(Int32(1) + 2.0).__name__,
)
print(
    "PYTHON PROTOCOL:",
    int(Int32(5)),
    bool(Int32(0)),
    float(Int32(2)),
    [10, 20, 30][Int32(1)],
    Int32(5) == 5,
    Int32(5) != 5,
    hash(Int32(1)) == hash(Int32(1)),
)
print(
    "CONSTRUCT:",
    repr(Int32(True)),
    repr(Int32(3.7)),
    repr(Int32(-3.7)),
    repr(Boolean(2)),
    repr(Boolean(0.0)),
    repr(Float32(3)),
    repr(Float32(Int32(2))),
    repr(Int64(Int32(-1))),
    repr(Int32(Float32(5.9))),
)
print(
    "ABSTRACT META:",
    Numeric.is_abstract,
    Integer.is_abstract,
    Float.is_abstract,
    Int32.is_abstract,
    Int32.width,
    Int32.signed,
    Int32.bytes,
    Boolean.width,
    Float32.is_float,
    Int64.is_integer,
)
print(
    "RANGES:",
    Int32.min,
    Int32.max,
    Int64.min,
    Int64.max,
    Boolean.min,
    Boolean.max,
    Int32.zero,
    Float32.zero,
    str(Int32),
    str(Float32),
)
names = {d.__name__ for d in ALL_DTYPES}
print("DTYPES:", [n for n in ("Boolean", "Float32", "Int32", "Int64") if n in names])
print(
    "AS NUMERIC:",
    repr(as_numeric(3)),
    repr(as_numeric(2.0)),
    repr(as_numeric(True)),
    repr(as_numeric(2**40)),
    as_numeric(Int32(1)).__class__.__name__,
)
# CHECK: FOLD: 8 3 3 0.25 2.5 True 0 -4
# CHECK: FOLD TYPES: Int32 Float32 Boolean Int64 Float32
# CHECK: PYTHON PROTOCOL: 5 False 2.0 20 True False True
# CHECK: CONSTRUCT: Int32(1) Int32(3) Int32(-3) Boolean(True) Boolean(False) Float32(3.0) Float32(2.0) Int64(-1) Int32(5)
# CHECK: ABSTRACT META: True True True False 32 True 4 1 True True
# Boolean is declared signed with width 1, so IntegerMeta.min/max answer -1/0
# (its exact range for construction is (0, 1)); recorded as it is.
# CHECK: RANGES: -2147483648 2147483647 -9223372036854775808 9223372036854775807 -1 0 0 0.0 Int32 Float32
# CHECK: DTYPES: ['Boolean', 'Float32', 'Int32', 'Int64']
# CHECK: AS NUMERIC: Int32(3) Float32(2.0) Boolean(True) Int64(1099511627776) Int32

# -- the literal warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    vals = [
        Int32(2**40),
        Int32(1e12),
        Float32(1e40),
        Float32(1e-50),
        Int32(0xFFFFFFFF),
    ]
print("WARNED:", [w.message.code for w in caught])
print("WARNED VALUES:", [repr(v) for v in vals])
# CHECK: WARNED: ['TYPE_INT_LITERAL_OUT_OF_RANGE', 'TYPE_FLOAT_TO_INT_OUT_OF_RANGE', 'TYPE_FLOAT_LITERAL_OVERFLOW', 'TYPE_FLOAT_LITERAL_UNDERFLOW']
# CHECK: WARNED VALUES: ['Int32(0)', 'Int32(-727379968)', 'Float32(1e+40)', 'Float32(1e-50)', 'Int32(-1)']


# The warnings module attributes a literal warning to the author's line,
# through the DSL frames between it and the range check (`cast` here), so its
# filters and once-per-location deduplication see the author's call.
def warn_through_cast():
    return cast(2**40, Int32)


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warn_through_cast()
print(
    "WARN AT:",
    os.path.basename(caught[0].filename),
    caught[0].lineno == warn_through_cast.__code__.co_firstlineno + 1,
)
# CHECK: WARN AT: core_scalar_types.py True

# -- errors outside a trace, and errors that need no trace
with ir.Context():
    err("MLIR TYPE OUTSIDE", lambda: Int32.mlir_type)
    print("SCALAR MLIR TYPE OUTSIDE:", Int32.scalar_mlir_type, Boolean.scalar_mlir_type)
err("CAST ABSTRACT OUTSIDE", lambda: cast(5, Numeric))
err("NAN TO INT", lambda: Int32(float("nan")))
err("STR TO INT", lambda: Int32("7"))
err("STR TO FLOAT", lambda: Float32("7"))
err("STR TO BOOLEAN", lambda: Boolean("7"))
err("STR AS NUMERIC", lambda: as_numeric("7"))
print("\n".join(text))
# CHECK: SCALAR MLIR TYPE OUTSIDE: i32 i1
# CHECK: WRAPPED: Int32 Float32 Int64 ? Float32(<mlir.{{.*}}BlockArgument object at {{.*}}>)
# CHECK: RESULTS: Int32 Int32 Float32 ?
# CHECK: TRUEDIV: Float32
# CHECK: PROMOTED: Float32 Int64 Int32
# CHECK: INT64 + FLOAT LITERAL: Float64
# CHECK: COMPARE: Boolean Boolean Boolean
# CHECK: CONVERT: Float32 Int32 Int64 Int32 Boolean
# CHECK: TO: Float32 Int32 True
# CHECK: UNARY/BITS: Int32 Float32 Int32 Float32 Int32 Int32 Int32 Int32 Int32 Int32
# CHECK: POW: Float32
# CHECK: INT POW: TYPE_INT_POW_UNSUPPORTED | `**` between two runtime integers is not supported: compiled code has no integer power operation.
# CHECK: RAW RHS: Int32 Int64
# CHECK: INT OF STAGED: PHASE_REQUIRES_CONSTANT | `int()` requires a value known at compile time, but it received a runtime value.
# CHECK: BOOL OF STAGED: PHASE_DYNAMIC_TO_STATIC_BOOL | A runtime value is used where plain Python needs a true/false answer
# CHECK: INDEX OF STAGED: PHASE_DYNAMIC_INDEX | Cannot use a runtime value as a list index or for loop range in plain Python code
# CHECK: TO INT OF STAGED: PHASE_REQUIRES_CONSTANT | `int(Int32)` requires a value known at compile time, but it received a runtime value.
# CHECK: NEG BOOLEAN: <free-form> | The operator `-` is not supported for `Boolean`.
# CHECK: CAST ABSTRACT: Int32(5) Float32(2.5) Boolean(True) Int64(1099511627776) Float32(3.0) Boolean(True)
# CHECK: CAST KEEPS: True True Float32
# CHECK: CAST FLOAT TO INTEGER: <free-form> | Cannot cast a `float` to the abstract type `Integer`: the DSL's `type_ops` plugin assigns it no dtype.
# CHECK: CAST STAGED FLOAT TO INTEGER: <free-form> | Cannot cast a `Float32` to the abstract type `Integer`: it is not a Python number.
# CHECK: INFER: Int32 None
# CHECK: MLIR TYPES: i32 i64 f32 i1
# CHECK: FROM MLIR TYPE: Int64 Boolean
# CHECK: FROM MLIR TYPE F64: Float64
# CHECK: INT32 OF F64 VALUE: NO ERROR
# CHECK: MLIR TYPE OUTSIDE: CALL_OUTSIDE_JIT | `a DSL type's MLIR type or op` was called from plain Python, but it can only be used inside a function decorated with `@jit`.
# CHECK: CAST ABSTRACT OUTSIDE: <free-form> | Cannot cast a `int` to the abstract type `Numeric`: no DSL is tracing, so there is no `type_ops` plugin to ask.
# CHECK: NAN TO INT: <free-form> | Cannot convert the float `nan` to `Int32`: it is not a finite number.
# CHECK: STR TO INT: ARG_NOT_NUMERIC | Argument `Int32(...)` expects a numeric value, but this call passes a value of type `str`.
# CHECK: STR TO FLOAT: ARG_NOT_NUMERIC | Argument `Float32(...)` expects a numeric value, but this call passes a value of type `str`.
# CHECK: STR TO BOOLEAN: ARG_NOT_NUMERIC | Argument `Boolean(...)` expects a numeric value, but this call passes a value of type `str`.
# CHECK: STR AS NUMERIC: ARG_NOT_NUMERIC | Argument `value` expects a numeric value, but this call passes a value of type `str`.


# -- the dtype registry: `register=False` keeps a dtype out of every lookup
# (it is still a working type), a dict-backed abstract base gets no `value`
# slot (the payload lives in its `__dict__`), and a dtype re-declared under a
# registered name supersedes the shipped one as the anchor the literal rule and
# promotion resolve. Last in the file: the supersession is process-global.
class Int8View(
    Integer, metaclass=IntegerMeta, width=8, signed=True, mlir_type=T.i8, register=False
): ...


with ir.Context():
    print(
        "UNREGISTERED:",
        Int8View in ALL_DTYPES,
        Numeric.from_mlir_type(T.i8()) is Int8,
        repr(Int8View(3) + Int8View(4)),
    )
# CHECK: UNREGISTERED: False True Int8View(7)


class Scalar(Numeric, is_abstract=True):
    __slots__ = ("__dict__",)


class Int24(
    Scalar,
    Integer,
    metaclass=IntegerMeta,
    width=24,
    signed=True,
    mlir_type=lambda: ir.IntegerType.get_signless(24),
): ...


x = Int24(5)
x.note = 1
print(
    "SLOTS:",
    Int24.__slots__,
    [k.__name__ for k in Int24.__mro__ if "value" in getattr(k, "__slots__", ())],
    x.note == 1 and x.value == 5,
    Int32.__slots__,
)
# CHECK: SLOTS: () [] True ('value',)


class Int32(Integer, metaclass=IntegerMeta, width=32, signed=True, mlir_type=T.i32): ...


print(
    "ANCHORS:",
    as_numeric(5).dtype is Int32,
    Int8.recast_width(32) is Int32,
    (Int16(1) + Int32(2)).dtype is Int32,
    type(Int8(7) / 2) is Float32,
    repr(Boolean(True) + Boolean(False)),
)
# CHECK: ANCHORS: True True True True Boolean(True)
print("S3_TYPES", "OK")
# CHECK: S3_TYPES OK
