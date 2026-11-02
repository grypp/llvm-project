# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Call providers: the calling convention between the wrapper and the callee.

:func:`attach_ffi_func` decodes and checks the ``TVMFFIAny`` arguments and
then hands the bound values to a :class:`CallProvider`, which emits the call
into the DSL's own function. :class:`DirectCallProvider` forwards the values
as plain operands and publishes one scalar result (mlir.dsl's host entries);
:class:`DynamicParamPackCallProvider` packs them into stack structs and
passes a ``void**`` array; :class:`NopCallProvider` emits nothing (tests).
"""

from collections.abc import Callable
from typing import Any, Optional, Union

from ..... import ir
from .....dialects import func as func_dialect
from .....dialects import llvm
from ....core.common import DSLRuntimeError, DSLUserCodeError
from .diagnostics import TvmFfiDiagId
from . import spec
from .tvm_ffi_builder import CallContext, CallProvider, TVMFFIBuilder, TVMFFITypeIndex

# The compile-time kinds: asserted by the wrapper, never forwarded.
_CONST_PARAMS = (spec.ConstNone, spec.ConstInt, spec.ConstBool, spec.ConstFloat)
# The kinds whose ``var`` holds an opaque handle.
_HANDLE_PARAMS = (spec.DataPointer,)


class NopCallProvider(CallProvider):
    """Emits no call: the wrapper only checks its arguments (for tests)."""

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        return current_block


class DirectCallProvider(CallProvider, TVMFFIBuilder):
    """Forward the decoded bindings as plain call operands, in parameter order.

    .. code-block:: c

        int32_t __tvm_ffi_f(void* handle, TVMFFIAny* args, int32_t n, TVMFFIAny* result) {
            // ... args[i] decoded and checked by the function builder ...
            T ret = f(arg0, arg1, ...);   // llvm.call, or func.call for a func.func callee
            result->type_index = kTVMFFIInt; result->v_int64 = ret;   // one scalar, or None
            return 0;
        }

    This is the convention of a host entry that takes its scalars and
    ``!llvm.ptr`` buffers one by one: a ``Var`` forwards its (already
    narrowed) value, a ``DataPointer``/``Stream``/... its handle, a ``Tensor``
    its data pointer followed by its symbolic shape and stride variables, a
    ``Shape`` its symbolic dimensions; ``Const*`` parameters are asserted by
    the wrapper and not forwarded.

    :param target_func: The symbol of the callee
    :param result_type: The callee's single scalar result type (an integer or a
        float type), or None for a ``void`` callee: the wrapper then returns
        ``None``
    :param result_signed: Whether an integer result widens with ``sext`` (else
        ``zext``)
    :param callee_kind: ``"llvm"`` emits ``llvm.call``; ``"func"`` emits
        ``func.call`` for a ``func.func`` callee (an ``llvm.call`` to one does
        not verify)
    :raises DSLRuntimeError: An unknown ``callee_kind``
    """

    def __init__(
        self,
        target_func: str,
        *,
        result_type: Optional[ir.Type] = None,
        result_signed: bool = True,
        callee_kind: str = "llvm",
    ) -> None:
        TVMFFIBuilder.__init__(self)
        self.target_func = target_func
        self.result_type = result_type
        self.result_signed = result_signed
        if callee_kind not in ("llvm", "func"):
            raise DSLRuntimeError(
                f"DirectCallProvider: unknown callee kind `{callee_kind}`"
            )
        self.callee_kind = callee_kind

    def call_operands(self, context: CallContext) -> list[ir.Value]:
        """The callee operands, in the order of the (flattened) parameters.

        :raises DSLUserCodeError: ``UNSUP_PARAM`` for an unknown kind
        """
        operands: list[ir.Value] = []
        for param in context.params:
            if isinstance(param, _CONST_PARAMS):
                continue
            if isinstance(param, spec.Var):
                operands.append(context.matched_var_binding[param])
            elif isinstance(param, _HANDLE_PARAMS):
                operands.append(context.matched_var_binding[param.var])
            elif isinstance(param, spec.Tensor):
                operands.append(context.matched_var_binding[param.data])
                for dim in param.shape:
                    if isinstance(dim, spec.Var):
                        operands.append(context.matched_var_binding[dim])
                if param.strides is not None:
                    for stride in param.strides:
                        if isinstance(stride, spec.Var):
                            operands.append(context.matched_var_binding[stride])
            elif isinstance(param, spec.Shape):
                for dim in param.shape:
                    if isinstance(dim, spec.Var):
                        operands.append(context.matched_var_binding[dim])
            else:
                raise DSLUserCodeError(
                    TvmFfiDiagId.UNSUP_PARAM,
                    detail=f"unsupported parameter type {type(param).__name__}",
                )
        return operands

    def store_result(self, raw_result: ir.Value, value: Optional[ir.Value]) -> None:
        """Write ``value`` into the ``TVMFFIAny`` result slot (``None`` for no value).

        The slot is ``{i32 type_index, i32 zero_padding, i64 v_int64 | f64
        v_float64}``: integers widen to ``i64`` (``i1`` becomes a ``bool``),
        floats to ``f64``.

        :raises DSLUserCodeError: ``UNSUP_PARAM`` for a result that is
            neither an integer nor a float type
        """
        type_index_ptr = self.getelementptr(raw_result, [0, 0], self.tvm_ffi_any_type)
        padding_ptr = self.getelementptr(raw_result, [0, 1], self.tvm_ffi_any_type)
        payload_ptr = self.getelementptr(raw_result, [0, 2], self.tvm_ffi_any_type)
        llvm.store(self.i32(0), padding_ptr)
        if value is None:
            llvm.store(self.i32(TVMFFITypeIndex.kTVMFFINone), type_index_ptr)
            llvm.store(self.i64(0), payload_ptr)
            return
        value_type = value.type.maybe_downcast()
        if isinstance(value_type, ir.IntegerType):
            width = value_type.width
            if width == 1:
                type_index = TVMFFITypeIndex.kTVMFFIBool
                payload = llvm.zext(self.i64_type, value)
            else:
                type_index = TVMFFITypeIndex.kTVMFFIInt
                if width == 64:
                    payload = value
                elif self.result_signed:
                    payload = llvm.sext(self.i64_type, value)
                else:
                    payload = llvm.zext(self.i64_type, value)
        elif isinstance(value_type, ir.F64Type):
            type_index = TVMFFITypeIndex.kTVMFFIFloat
            payload = value
        elif isinstance(value_type, (ir.F32Type, ir.F16Type, ir.BF16Type)):
            type_index = TVMFFITypeIndex.kTVMFFIFloat
            payload = llvm.fpext(self.f64_type, value)
        else:
            raise DSLUserCodeError(
                TvmFfiDiagId.UNSUP_PARAM,
                detail=f"unsupported result type {value_type}",
            )
        llvm.store(self.i32(type_index), type_index_ptr)
        llvm.store(payload, payload_ptr)

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        operands = self.call_operands(context)
        with ir.InsertionPoint(current_block):
            result: Optional[ir.Value]
            if self.callee_kind == "func":
                call = func_dialect.CallOp(
                    [self.result_type] if self.result_type is not None else [],
                    self.target_func,
                    operands,
                )
                result = call.results[0] if self.result_type is not None else None
            else:
                call_result = llvm.call(
                    result=self.result_type,
                    callee=self.target_func,
                    callee_operands=operands,
                    op_bundle_sizes=[],
                    op_bundle_operands=[],
                )
                result = call_result if self.result_type is not None else None
            self.store_result(context.raw_result, result)
        return current_block
