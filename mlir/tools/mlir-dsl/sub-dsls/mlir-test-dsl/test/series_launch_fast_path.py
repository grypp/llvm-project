# RUN: %PYTHON %s 2>&1 | FileCheck %s
# REQUIRES: host-supports-jit
# The launch fast path: a Python scalar passed for a concrete `Numeric`
# parameter that the dtype's constructor would only box is packed straight
# into its slot (`ArgMeta.scalar_packers`, one `scalar_fast_packer` per such
# slot); a value outside the packer's window (an out-of-range int, NaN, an
# infinity) takes the constructor path with its conversions and warnings, so
# the two paths agree byte for byte and the warnings are kept.
import ctypes
import inspect
import math
import warnings

import numpy as np

import mlir.mlir_dsl as m
from mlir.dsl.plugins.compiler.jit_executor import ExecutionArgs, scalar_fast_packer
from mlir.dsl.types import typing as t
from mlir.dsl.util.tree_utils import leaf_entry


def fill(p: m.Pointer[m.Float32], n: m.Int32, v: m.Float32):
    pass


def slot_bytes(slot, ctype):
    return ctypes.string_at(slot.value, ctypes.sizeof(ctype))


def read(slot, ctype):
    return ctype.from_address(slot.value).value


# The binder: a packer for each concrete `Numeric` slot, none for the pointer.
binder = ExecutionArgs(inspect.signature(fill), "fill")
# The pointer parameter takes a bare address under `Pointer[Float32]` (no
# buffer adapter is involved; none is registered without a DSL instance).
buf = np.zeros(4, np.float32).ctypes.data
packers = binder._meta.scalar_packers
print("PACKERS:", *[packer is not None for packer in packers])
# CHECK: PACKERS: False True True

# The slot bytes of (7, 1.5) are those the leaf marshal of the cast values
# produces.
slots, _ = binder.generate_execution_args((buf, 7, 1.5), {})
full_n = leaf_entry(m.Int32).marshal(t.cast(7, m.Int32))[0]
full_v = leaf_entry(m.Float32).marshal(t.cast(1.5, m.Float32))[0]
print(
    "BYTES:",
    slot_bytes(slots[1], ctypes.c_int32) == slot_bytes(full_n, ctypes.c_int32),
    end=" ",
)
print(slot_bytes(slots[2], ctypes.c_float) == slot_bytes(full_v, ctypes.c_float))
# CHECK: BYTES: True True

# The windows: an `int` inside the exact range, a `float` that is zero or
# between the smallest subnormal and the largest finite value of the format.
int_type, pack_int = packers[1]
float_type, pack_float = packers[2]
print("WINDOW:", int_type.__name__, pack_int(2**31 - 1) is not None, end=" ")
print(pack_int(2**31), pack_int(-(2**31) - 1), float_type.__name__, end=" ")
print(pack_float(0.0) is not None, pack_float(3.0e38) is not None, end=" ")
print(
    pack_float(float("nan")), pack_float(math.inf), pack_float(1e39), pack_float(1e-46)
)
# CHECK: WINDOW: int True None None float True True None None None None

# Which dtypes get a packer: `bool` for the width-1 integer, none for an
# abstract annotation, for a float format outside the probe table and for a
# dtype without a host representation.
kinds = [
    (scalar_fast_packer(dtype) or (None,))[0]
    for dtype in (m.Boolean, m.Numeric, m.BFloat16, m.Float8E5M2, m.Float16, m.Uint64)
]
print("KINDS:", [kind.__name__ if kind else None for kind in kinds])
# CHECK: KINDS: ['bool', None, None, None, 'float', 'int']

# Out of the window: an `int` past the exact range takes the constructor
# path, which wraps like a C cast (silently inside the unsigned range of the
# width, the `Int32(0xFFFFFFFF)` idiom, with a warning beyond it); NaN under
# `Float32` takes it too (and stays NaN); a `bool` under `Int32` is not the
# packer's type and is folded by the constructor.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    args = (buf, 2**32, float("nan"))
    slots, _ = binder.generate_execution_args(args, {})
codes = [w.message.code for w in caught]
print(
    "WARNED:", "TYPE_INT_LITERAL_OUT_OF_RANGE" in codes, read(slots[1], ctypes.c_int32)
)
print("NAN_FULL:", math.isnan(read(slots[2], ctypes.c_float)), pack_float(float("nan")))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    slots, _ = binder.generate_execution_args((buf, 2**31, 1.0), {})
print("WRAPPED:", read(slots[1], ctypes.c_int32), len(caught))
slots, _ = binder.generate_execution_args((buf, True, 1.0), {})
print("BOOL_FULL:", read(slots[1], ctypes.c_int32))
# CHECK: WARNED: True 0
# CHECK: NAN_FULL: True None
# CHECK: WRAPPED: -2147483648 0
# CHECK: BOOL_FULL: 1


# The smoke user: a compiled reduction called three times before compilation caching is introduced with a Python
# `int` for its `Int32` parameter.
@m.jit
def total(p: m.Pointer[m.Int32], n: m.Int32) -> m.Int32:
    for i, acc, result in m.for_(n, iter_args=[m.Int32(0)]):
        m.yield_(acc + p[i])
    return result


data = np.arange(1000, dtype=np.int32)
results = {total(data.ctypes.data, 1000) for _ in range(3)}
print("SUM:", *sorted(results))
# CHECK: SUM: 499500
