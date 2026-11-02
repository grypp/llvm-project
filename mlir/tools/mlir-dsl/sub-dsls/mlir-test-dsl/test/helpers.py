# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
import ctypes
import functools
import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith
from mlir.dsl.core.user_op import dsl_user_op
from mlir.extras import types as T

def report(fn, *args, **kwargs):
    try:
        print("RESULT:", fn(*args, **kwargs))
    except m.DSLUserCodeError as e:
        print("ERROR:", e.diag_id.name)
    except m.DSLRuntimeError as e:
        print("INTERNAL:", e.message)

# An argument the MLIR bindings reject while a user op builds its op (an
# operand that is not a `Value`, an attribute value of the wrong kind, one that
# does not fit the type) reaches the author as `ARG_REJECTED_BY_OP` naming the
# builder, through a `DSLUserCodeError` that keeps the builtin base of the raw
# error: `except ValueError`, `except TypeError` and `except ir.MLIRError` in
# the author's code keep working. Any other error propagates as it is.
@dsl_user_op
def add_operand(a, b, *, loc=None, ip=None):
    return arith.AddIOp(a, b, loc=loc, ip=ip).result


@dsl_user_op
def constant_of(ty, value, *, loc=None, ip=None):
    return arith.ConstantOp(ty, value, loc=loc, ip=ip).result


@dsl_user_op
def own_check(a, *, loc=None, ip=None):
    raise ValueError("not a binding error")


@m.jit
def operand_not_a_value(a: m.Int32) -> m.Int32:
    return m.Int32(add_operand(a.ir_value(), "x"))


@m.jit
def attribute_wrong_kind(a: m.Int32) -> m.Int32:
    return m.Int32(constant_of(T.i32(), "x"))


@m.jit
def attribute_does_not_fit(a: m.Int32) -> m.Int32:
    return m.Int32(constant_of(T.i32(), 1.5))


@m.jit
def unrelated(a: m.Int32) -> m.Int32:
    return m.Int32(own_check(a.ir_value()))


for label, raw, fn, api in (
    ("OPERAND:", ValueError, operand_not_a_value, "add_operand"),
    ("ATTR KIND:", TypeError, attribute_wrong_kind, "constant_of"),
    ("ATTR VALUE:", ir.MLIRError, attribute_does_not_fit, "constant_of"),
):
    try:
        fn(1)
        print(label, "no error")
    except raw as e:
        print(
            label,
            getattr(e, "code", None),
            f"`{api}` rejected" in str(getattr(e, "message", "")),
            isinstance(e, m.DSLUserCodeError),
        )
try:
    unrelated(1)
except ValueError as e:
    print("UNCHANGED:", type(e) is ValueError, e)
# CHECK: OPERAND: ARG_REJECTED_BY_OP True True
# CHECK: ATTR KIND: ARG_REJECTED_BY_OP True True
# CHECK: ATTR VALUE: ARG_REJECTED_BY_OP True True
# CHECK: UNCHANGED: True not a binding error
