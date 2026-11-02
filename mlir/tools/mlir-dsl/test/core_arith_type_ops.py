# RUN: %PYTHON %s 2>&1 | FileCheck %s
# S2: the `arith` op module on raw `ir.Value`s inside a hand-built module, the
# way mlir/test/python/dialects/*.py drive a dialect. No DSL type is involved:
# the dtype-shaped arguments (`int_to_int`'s destination) are stand-ins with
# the three attributes the function reads; the real dtypes arrive in S3.
import operator
from types import SimpleNamespace

from mlir import ir
from mlir.dialects import func
from mlir.extras import types as T

from mlir.dsl.core.common import DSLRuntimeError, DSLUserCodeError
from mlir.dsl.plugins.type_ops import arith

text = []

with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        fop = func.FuncOp(
            "ops", ([T.i32(), T.i32(), T.f32(), T.f32(), T.f16(), T.bool()], [])
        )
        block = fop.add_entry_block()
        with ir.InsertionPoint(block):
            a, b, x, y, h, p = block.arguments
            # CHECK-LABEL: func.func @ops(
            # CHECK-SAME: %[[A:[^:]+]]: i32, %[[B:[^:]+]]: i32, %[[X:[^:]+]]: f32, %[[Y:[^:]+]]: f32, %[[H:[^:]+]]: f16, %[[P:[^:]+]]: i1)

            # -- const: the literal rule, an explicit type, the unsigned re-encoding, a splat
            # CHECK: arith.constant 7 : i32
            # CHECK: arith.constant true
            # CHECK: arith.constant 2.500000e+00 : f32
            # CHECK: arith.constant 1 : i32
            # CHECK: arith.constant 1.000000e+00 : f32
            # CHECK: arith.constant 2 : i32
            # CHECK: arith.constant -1 : i32
            # CHECK: arith.constant dense<3> : vector<4xi32>
            arith.const(7)
            arith.const(True)
            arith.const(2.5)
            c1 = arith.const(1, T.i32())
            arith.const(1, T.f32())
            arith.const(2.7, T.i32())
            arith.const(0xFFFFFFFF, T.i32(), signed=False)
            arith.const(3, T.vector(4, T.i32()))
            text.append(f"CONST PASSTHROUGH: {arith.const(a) is a}")

            # -- the binary operators, integer and float
            # CHECK: arith.addi %[[A]], %[[B]] : i32
            # CHECK: arith.addf %[[X]], %[[Y]] : f32
            # CHECK: arith.subi %[[A]], %[[B]] : i32
            # CHECK: arith.subf %[[X]], %[[Y]] : f32
            # CHECK: arith.muli %[[A]], %[[B]] : i32
            # CHECK: arith.mulf %[[X]], %[[Y]] : f32
            arith.add(a, b)
            arith.add(x, y)
            arith.sub(a, b)
            arith.sub(x, y)
            arith.mul(a, b)
            arith.mul(x, y)
            # CHECK: %[[AF:.+]] = arith.sitofp %[[A]] : i32 to f32
            # CHECK: %[[BF:.+]] = arith.sitofp %[[B]] : i32 to f32
            # CHECK: arith.divf %[[AF]], %[[BF]] : f32
            # CHECK: arith.divf %[[X]], %[[Y]] : f32
            arith.truediv(a, b, signed=True)
            arith.truediv(x, y)
            # CHECK: arith.floordivsi %[[A]], %[[B]] : i32
            # CHECK: arith.divui %[[A]], %[[B]] : i32
            # CHECK: %[[Q:.+]] = arith.divf %[[X]], %[[Y]] : f32
            # CHECK: math.floor %[[Q]] : f32
            arith.floordiv(a, b, signed=True)
            arith.floordiv(a, b, signed=False)
            arith.floordiv(x, y)
            # CHECK: arith.remsi %[[A]], %[[B]] : i32
            # CHECK: arith.remui %[[A]], %[[B]] : i32
            # CHECK: arith.remf %[[X]], %[[Y]] : f32
            arith.mod(a, b, signed=True)
            arith.mod(a, b, signed=False)
            arith.mod(x, y)
            # CHECK: math.powf %[[X]], %[[Y]] : f32
            # CHECK: math.fpowi %[[X]], %[[A]] : f32, i32
            # CHECK: %[[AF2:.+]] = arith.sitofp %[[A]] : i32 to f32
            # CHECK: math.powf %[[AF2]], %[[X]] : f32
            arith.pow(x, y)
            arith.pow(x, a)
            arith.pow(a, x, signed=True)
            try:
                arith.pow(a, b)
            except DSLUserCodeError as e:
                text.append(f"INT POW: {e.code}")

            # -- unary, bitwise, shifts
            # CHECK: %[[Z:.+]] = arith.constant 0 : i32
            # CHECK: arith.subi %[[Z]], %[[A]] : i32
            # CHECK: arith.negf %[[X]] : f32
            # CHECK: math.absi %[[A]] : i32
            # CHECK: math.absf %[[X]] : f32
            arith.neg(a)
            arith.neg(x)
            arith.abs(a)
            arith.abs(x)
            # CHECK: arith.andi %[[A]], %[[B]] : i32
            # CHECK: arith.ori %[[A]], %[[B]] : i32
            # CHECK: arith.xori %[[A]], %[[B]] : i32
            # CHECK: arith.shli %[[A]], %[[B]] : i32
            # CHECK: arith.shrsi %[[A]], %[[B]] : i32
            # CHECK: arith.shrui %[[A]], %[[B]] : i32
            arith.and_(a, b)
            arith.or_(a, b)
            arith.xor(a, b)
            arith.shl(a, b)
            arith.shr(a, b, signed=True)
            arith.shr(a, b, signed=False)

            # -- comparisons: a predicate name or the operator function
            # CHECK: arith.cmpi slt, %[[A]], %[[B]] : i32
            # CHECK: arith.cmpi ult, %[[A]], %[[B]] : i32
            # CHECK: arith.cmpi sge, %[[A]], %[[B]] : i32
            # CHECK: arith.cmpi eq, %[[A]], %[[B]] : i32
            # CHECK: arith.cmpf olt, %[[X]], %[[Y]] : f32
            # CHECK: arith.cmpf une, %[[X]], %[[Y]] : f32
            arith.cmp("lt", a, b, signed=True)
            arith.cmp("lt", a, b, signed=False)
            arith.cmp(operator.ge, a, b)
            arith.cmp("eq", a, b)
            arith.cmp("lt", x, y)
            arith.cmp("ne", x, y)
            try:
                arith.cmp("spaceship", a, b)
            except DSLRuntimeError as e:
                text.append(f"BAD PREDICATE: {e.message}")

            # -- conversions
            # CHECK: arith.sitofp %[[A]] : i32 to f32
            # CHECK: arith.uitofp %[[A]] : i32 to f32
            # CHECK: arith.uitofp %[[P]] : i1 to f32
            # CHECK: arith.fptosi %[[X]] : f32 to i32
            # CHECK: arith.fptoui %[[X]] : f32 to i32
            # CHECK: arith.fptosi %[[X]] : f32 to i32
            arith.itofp(a, True, T.f32())
            arith.itofp(a, False, T.f32())
            arith.itofp(p, True, T.f32())
            arith.fptoi(x, True, T.i32())
            arith.fptoi(x, False, T.i32())
            arith.fptoi(x, None, T.i32())
            # CHECK: arith.truncf %[[X]] : f32 to f16
            # CHECK: arith.extf %[[H]] : f16 to f32
            # CHECK: %[[HW:.+]] = arith.extf %[[H]] : f16 to f32
            # CHECK: arith.truncf %[[HW]] : f32 to bf16
            arith.cvtf(x, T.f16())
            arith.cvtf(h, T.f32())
            arith.cvtf(h, T.bf16())
            text.append(f"CVTF SAME TYPE: {arith.cvtf(x, T.f32()) is x}")
            # int_to_int reads `.signed`, `.width`, `.mlir_type` of the destination dtype.
            i64 = SimpleNamespace(signed=True, width=64, mlir_type=T.i64())
            i8 = SimpleNamespace(signed=True, width=8, mlir_type=T.i8())
            u64 = SimpleNamespace(signed=False, width=64, mlir_type=T.i64())
            i32 = SimpleNamespace(signed=True, width=32, mlir_type=T.i32())
            # CHECK: arith.extsi %[[A]] : i32 to i64
            # CHECK: arith.trunci %[[A]] : i32 to i8
            # CHECK: arith.extui %[[A]] : i32 to i64
            # CHECK: arith.extui %[[P]] : i1 to i32
            # CHECK: arith.extui %[[A]] : i32 to i64
            arith.int_to_int(a, i64)
            arith.int_to_int(a, i8)
            arith.int_to_int(a, u64)
            arith.int_to_int(p, i32)
            arith.int_to_int(
                a, i64, src_signed=False
            )  # unsigned -> wider signed: extui
            text.append(f"INT_TO_INT SAME WIDTH: {arith.int_to_int(a, i32) is a}")
            # CHECK: return
            func.ReturnOp([])

    # -- the type helpers
    text.append(
        f"RECAST: {arith.recast_type(T.vector(4, T.i32()), T.f32())} {arith.recast_type(T.i32(), T.f32())}"
    )
    text.append(
        f"SCALAR: {arith.is_scalar(T.i32())} {arith.is_scalar(T.vector(4, T.i32()))}"
    )
    text.append(
        f"ELEMENT: {arith.element_type(T.vector(4, T.f16()))} {arith.element_type(T.i8())}"
    )
    text.append(
        f"KINDS: {arith.is_float_type(T.bf16())} {arith.is_float_type(T.i32())} {arith.is_integer_like_type(T.index())} {arith.is_integer_like_type(T.f32())}"
    )
    try:
        arith.const("seven")
    except DSLRuntimeError as e:
        text.append(f"CONST STR: {e.message}")

    module.operation.verify()
    print(module)

print("\n".join(text))
# CHECK: CONST PASSTHROUGH: True
# CHECK: INT POW: TYPE_INT_POW_UNSUPPORTED
# CHECK: BAD PREDICATE: cmp: unsupported predicate 'spaceship'
# CHECK: CVTF SAME TYPE: True
# CHECK: INT_TO_INT SAME WIDTH: True
# CHECK: RECAST: vector<4xf32> f32
# CHECK: SCALAR: True False
# CHECK: ELEMENT: f16 i8
# CHECK: KINDS: True False True False
# CHECK: CONST STR: <class 'str'> is not supported
print("S2_ARITH", "OK")
# CHECK: S2_ARITH OK
