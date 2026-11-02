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


@m.jit
def stored(p: m.Pointer[m.Int32], x: m.Int32):
    print("STORE:", p.store(x))
    p.store(x, ordering="acquire")  # the verifier rejects an acquire store


# A statement builder (`Pointer.store`) returns the op it built, so the wrapper
# verifies it in every mode and names the builder; the `-> None` annotation
# hands the caller None, not the op.
# CHECK:      STORE: None
# CHECK-NEXT: INTERNAL: Operation verification failed in 'store'
storage = (ctypes.c_int32 * 1)(0)
report(stored, ctypes.addressof(storage), 1)


# A sub-DSL's root class replaces the wrapper of every operator and the driver
# behind the binary ones through two class attributes: a class body that sets
# `user_op` has its inherited operators re-wrapped with it at class creation,
# and `_binary_op` is resolved per call through the instance. The shipped
# `Int32` keeps the shipped wrapper and driver.
counts = {"user_op": 0, "ops": []}


def counting_op(fn):
    wrapped = dsl_user_op(fn)

    @functools.wraps(wrapped)
    def counted(*args, **kwargs):
        counts["user_op"] += 1
        return wrapped(*args, **kwargs)

    return counted


def recording_binary_op(op, *args, **kwargs):
    counts["ops"].append(op.__name__)
    return m.Numeric._binary_op(op, *args, **kwargs)


class Counting(m.Numeric, is_abstract=True):
    user_op = staticmethod(counting_op)
    _binary_op = staticmethod(recording_binary_op)


class CInt32(
    Counting,
    m.Integer,
    metaclass=m.IntegerMeta,
    width=32,
    signed=True,
    mlir_type=T.i32,
    register=False,
): ...


print("BINOP:", repr(CInt32(1) + 2), repr(CInt32(3) < 4), *counts["ops"])
print("USER_OP calls:", counts["user_op"])
m.Int32(1) + 2
print("SHIPPED untouched:", counts["user_op"], len(counts["ops"]))
# CHECK:      BINOP: CInt32(3) Boolean(True) add lt
# CHECK-NEXT: USER_OP calls: 2
# CHECK-NEXT: SHIPPED untouched: 2 2
# The re-wrap takes an operator from the dtype classes only (`__or__` is
# `Integer`'s, not the metaclass's `type.__or__`), and the class keeps
# `Numeric.__hash__` although `__eq__` was injected into its body.
print(
    "OR/HASH:",
    repr(CInt32(6) | 3),
    repr(3 | CInt32(6)),
    hash(CInt32(1)) == hash(CInt32(1)),
)
# CHECK-NEXT: OR/HASH: CInt32(7) CInt32(7) True
