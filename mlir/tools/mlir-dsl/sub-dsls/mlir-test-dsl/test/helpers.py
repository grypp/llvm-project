# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_VERIFY_TRACE=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=DEBUGINFO
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUG=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=DEBUG
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


# =============================================================================
# dsl_user_op: source locations (MLIR's traceback locations), verification
# =============================================================================
# Locations come from MLIR's traceback locations, which `BaseDSL` turns on
# under DEBUGINFO (depth 1): an op carries the user line and column range that
# built it; the frames of the DSL packages, the bindings and the standard
# library are skipped, and DEBUG keeps the DSL frames instead so an op is
# attributed to the DSL line that built it. An explicit `loc=` wins over the
# traceback; the name of a `NameLoc` in `Location.current` wraps either.
# A helper in a module of a DSL package: its frames are skipped by the walk.
# (The exec'd module has no file on disk, so the test registers its pseudo
# filename the way a package directory is registered.)
LIB_SRC = "def lib_twice(x):\n    return x + x\n"
lib_namespace = {"__name__": "mlir.dsl._test_lib"}
exec(compile(LIB_SRC, "<dsl-lib>", "exec"), lib_namespace)
lib_twice = lib_namespace["lib_twice"]
ir._globals.register_traceback_file_exclusion("<dsl-lib>")


@m.jit
def located(a: m.Int32, x: m.Float32) -> m.Float32:
    b = a + 1
    print("LOC addi:", b.value.owner.location)
    # DEBUGINFO: LOC addi: loc("located"("{{.*}}helpers.py":[[#@LINE-2]]:8 to :13))
    # DEBUG:     LOC addi: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC addi: loc(unknown)
    r = lib_twice(x)
    print("LOC lib:", r.value.owner.location)
    # The op was built through a helper in a DSL package: the user line is
    # its caller, except under MLIR_DSL_DEBUG=1, which names the DSL line
    # that built the op (the scalar type ops).
    # DEBUGINFO: LOC lib: loc("located"("{{.*}}helpers.py":[[#@LINE-5]]:8 to :20))
    # DEBUG:     LOC lib: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC lib: loc(unknown)
    y = a + x
    print("LOC promoted:", y.value.owner.operands[0].owner.location)
    # The `sitofp` that operand promotion emits inside the DSL gets the user
    # line too: every op built during the statement does, whichever DSL frame
    # builds it.
    # DEBUGINFO: LOC promoted: loc("located"("{{.*}}helpers.py":[[#@LINE-5]]:8 to :13))
    # DEBUG:     LOC promoted: loc("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}}))
    # CHECK:     LOC promoted: loc(unknown)
    c = b.__add__(1, loc=ir.Location.name("given"))  # a caller's `loc=` wins
    print("LOC explicit:", c.value.owner.location)
    # DEBUGINFO: LOC explicit: loc("given")
    # DEBUG:     LOC explicit: loc("given")
    # CHECK:     LOC explicit: loc("given")
    with ir.Location.name("scope", childLoc=ir.Location.current):
        # A `NameLoc` in `Location.current` names a scope: its name wraps the
        # location of every op built inside, the ones given by `loc=` too.
        b2 = a + 1
        print("LOC scoped:", b2.value.owner.location)
        # DEBUGINFO: LOC scoped: loc("scope"("located"("{{.*}}helpers.py":[[#@LINE-2]]:13 to :18)))
        # DEBUG:     LOC scoped: loc("scope"("{{.*}}"("{{.*}}/mlir/dsl/{{.*}}.py":{{.*}})))
        # CHECK:     LOC scoped: loc("scope")
        c2 = b.__add__(1, loc=ir.Location.name("given2"))
        print("LOC scoped explicit:", c2.value.owner.location)
        # DEBUGINFO: LOC scoped explicit: loc("scope"("given2"))
        # DEBUG:     LOC scoped explicit: loc("scope"("given2"))
        # CHECK:     LOC scoped explicit: loc("given2")
    return m.Float32(c) + r


report(located, 1, 4.0)
# DEBUGINFO: RESULT: ?
# DEBUG:     RESULT: ?
# CHECK:     RESULT: ?


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
