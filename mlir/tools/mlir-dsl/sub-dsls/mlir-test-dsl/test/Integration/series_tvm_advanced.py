# RUN: %PYTHON %s
# REQUIRES: host-supports-jit, tvm_ffi
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise nested tuple decoding, packed calls and stream constraints."""
import ctypes

import tvm_ffi
from tvm_ffi import libinfo
from mlir import execution_engine, ir
from mlir.dsl.core.common import DSLUserCodeError
from mlir.dsl.plugins.adapters.tvm_ffi import (
    CallProvider,
    DirectCallProvider,
    DynamicParamPackCallProvider,
    NopCallProvider,
    attach_ffi_func,
    spec,
)


class CheckedOnly(CallProvider):
    def __call__(self, block, context):
        with ir.InsertionPoint(block):
            DirectCallProvider("unused").store_result(context.raw_result, None)
        return block


with ir.Context(), ir.Location.unknown():
    pair = spec.TupleParam(
        "pair", [spec.Var("value", "int32"), spec.ConstInt("tag", 7)]
    )
    module = ir.Module.create()
    attach_ffi_func(module, "tuple", [pair], CheckedOnly())
    assert module.operation.verify()
    engine = execution_engine.ExecutionEngine(
        module, shared_libs=[libinfo.find_libtvm_ffi()]
    )
    check = tvm_ffi.Function.__from_mlir_packed_safe_call__(
        engine.raw_lookup("__tvm_ffi_tuple"), keep_alive_object=engine
    )
    assert check((3, 7)) is None
    try:
        check((3, 8))
    except Exception:
        pass
    else:
        raise AssertionError("a nested compile-time constant must be checked")

    packed = ir.Module.parse(
        "module { llvm.func @target(!llvm.ptr) attributes {llvm.linkage = #llvm.linkage<external>} }"
    )
    attach_ffi_func(
        packed,
        "packed",
        [spec.Var("n", "int32")],
        DynamicParamPackCallProvider("target"),
    )
    assert packed.operation.verify()
    assert "llvm.alloca" in str(packed)
    assert "llvm.call @target" in str(packed)

    # A handle whose pointee the callee takes by value: `pass_pointee_as`
    # names the aggregate's type and the packing provider forwards the handle
    # itself, loaded as that aggregate under `struct_call`, instead of packing
    # the pointer into a stack slot.
    pointee = ir.Module.parse(
        """
        module {
          llvm.func @sum_pair(%p: !llvm.struct<(i32, i32)>, %out: !llvm.ptr) {
            %x = llvm.extractvalue %p[0] : !llvm.struct<(i32, i32)>
            %y = llvm.extractvalue %p[1] : !llvm.struct<(i32, i32)>
            %s = llvm.add %x, %y : i32
            llvm.store %s, %out : i32, !llvm.ptr
            llvm.return
          }
        }
        """
    )
    attach_ffi_func(
        pointee,
        "sum_pair",
        [
            spec.Var(
                "pair",
                "handle",
                pass_pointee_as=lambda b: b.struct_type(
                    fields=[b.i32_type, b.i32_type]
                ),
            ),
            spec.Var("out", "handle"),
        ],
        DynamicParamPackCallProvider("sum_pair", struct_call=True),
    )
    assert pointee.operation.verify()
    assert "!llvm.ptr -> !llvm.struct<(i32, i32)>" in str(pointee), str(pointee)
    assert "llvm.call @sum_pair" in str(pointee)
    engine = execution_engine.ExecutionEngine(
        pointee, shared_libs=[libinfo.find_libtvm_ffi()]
    )
    sum_pair = tvm_ffi.Function.__from_mlir_packed_safe_call__(
        engine.raw_lookup("__tvm_ffi_sum_pair"), keep_alive_object=engine
    )

    class Pair(ctypes.Structure):
        _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]

    pair_value = Pair(7, 11)
    out = ctypes.c_int32(0)
    sum_pair(
        ctypes.c_void_p(ctypes.addressof(pair_value)),
        ctypes.c_void_p(ctypes.addressof(out)),
    )
    assert out.value == 18, out.value

    handles = ir.Module.create()
    attach_ffi_func(
        handles,
        "handles",
        [spec.Stream("stream"), spec.CudaEvent("event")],
        NopCallProvider(),
    )
    assert handles.operation.verify()
    try:
        attach_ffi_func(
            ir.Module.create(),
            "no_device",
            [spec.EnvStream("stream")],
            NopCallProvider(),
        )
    except DSLUserCodeError as error:
        assert "EnvStream" in str(error)
    else:
        raise AssertionError("an environment stream needs a device tensor")
print("nested tuples, packed calls and stream constraints: passed")
