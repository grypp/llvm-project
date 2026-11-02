# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
# RUN: %if host-supports-jit %{ %PYTHON %s 2>&1 | FileCheck %s --check-prefix=EXEC %}
import ctypes
import inspect
import math
import mlir.mlir_dsl as m
from mlir.dsl.plugins.compiler.jit_executor import ExecutionArgs


def failure(fn):
    """The DiagId of the user error `fn` raises, the message of an internal one."""
    try:
        fn()
        return "no error"
    except m.DSLUserCodeError as e:
        return e.diag_id.name
    except m.DSLRuntimeError as e:
        return " ".join(e.message.split())


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
