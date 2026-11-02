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
_HANDLE_PARAMS = (
    spec.Stream,
    spec.EnvStream,
    spec.CudaEvent,
    spec.CudaGraph,
    spec.CudaGraphNode,
    spec.DataPointer,
)


def _flatten_tuple_params(params: list[spec.Param]) -> list[spec.Param]:
    """``params`` with every :class:`spec.TupleParam` replaced by its elements."""
    flattened = []
    for param in params:
        if isinstance(param, spec.TupleParam):
            flattened.extend(_flatten_tuple_params(param.params))
        else:
            flattened.append(param)
    return flattened


class NopCallProvider(CallProvider):
    """Emits no call: the wrapper only checks its arguments (for tests)."""

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        return current_block


class DynamicParamPackCallProvider(CallProvider, TVMFFIBuilder):
    """Pack the arguments into stack structs and call ``target_func`` with them.

    .. code-block:: c

        void call(Tensor0 t0, Tensor1 t1) {
            // packed arguments
            void** packed_args[] = {&t0, &t1};
            // call target
            target_func(packed_args);
        }

    A :class:`spec.Tensor` packs as ``{data, *symbolic shape, *symbolic
    strides}``, a :class:`spec.Var` or handle as a one-field struct, a
    :class:`spec.Shape` as its symbolic dimensions; ``Const*`` parameters are
    not forwarded.

    :param target_func: The callee's symbol
    :param include_num_args: Also pass the number of packed arguments (``i32``)
    :param struct_call: Load the structs back and pass them by value instead of
        the ``void**`` array
    :param flatten_tuple_params: Flatten :class:`spec.TupleParam` into its
        elements (the only supported setting)
    :raises DSLRuntimeError: ``flatten_tuple_params=False``
    """

    def __init__(
        self,
        target_func: str,
        include_num_args: bool = False,
        struct_call: bool = False,
        flatten_tuple_params: bool = True,
    ) -> None:
        TVMFFIBuilder.__init__(self)
        self.target_func = target_func
        self.include_num_args = include_num_args
        self.struct_call = struct_call
        self.flatten_tuple_params = flatten_tuple_params
        self.float4x2_dtype = spec.tvm_ffi.dtype("float4_e2m1fnx2")

        if not self.flatten_tuple_params:
            raise DSLRuntimeError("flatten_tuple_params=False is not supported yet")

    def get_callee_struct_for_param_tensor(
        self,
        param: spec.Tensor,
        current_block: ir.Block,
        data: ir.Value,
        shape: list[ir.Value],
        strides: list[ir.Value],
        flatten_struct: ir.Type,
    ) -> ir.Type:
        """Hook for a subclass to change the struct type a tensor is passed as.

        :return: ``flatten_struct``, the ``{data, *shape, *strides}`` struct
        """
        return flatten_struct

    def pack_param_tensor(
        self, current_block: ir.Block, context: CallContext, param: spec.Tensor
    ) -> tuple[ir.Type, ir.Value]:
        """Pack a tensor as ``{data, *symbolic shape, *symbolic strides}``."""
        map_shape_value = lambda _, value: value
        map_stride_value = lambda _, value: value

        if param.map_tensor_dtype_f4x2_to_f4 and param.dtype == self.float4x2_dtype:
            # A float4x2 tensor packs two float4 per byte: the callee sees the
            # stride-1 dimension doubled and every other stride doubled.
            stride_one_index = spec.stride_one_index_of(param.shape, param.strides)

            def _make_f4x2_mapper(
                double_at_stride_one: bool,
            ) -> Callable[[int, ir.Value], ir.Value]:
                def mapper(index: int, value: ir.Value) -> ir.Value:
                    if (index == stride_one_index) == double_at_stride_one:
                        with ir.InsertionPoint(current_block):
                            return self.mul(value, self.integer_constant(value.type, 2))
                    return value

                return mapper

            map_shape_value = _make_f4x2_mapper(double_at_stride_one=True)
            map_stride_value = _make_f4x2_mapper(double_at_stride_one=False)

        data = context.matched_var_binding[param.data]
        shape = [
            map_shape_value(index, context.matched_var_binding[dim])
            for index, dim in enumerate(param.shape)
            if isinstance(dim, spec.Var)
        ]
        strides = []
        if param.strides is not None:
            strides = [
                map_stride_value(index, context.matched_var_binding[dim])
                for index, dim in enumerate(param.strides)
                if isinstance(dim, spec.Var)
            ]
        flatten_struct, alloca = self.pack_values_to_alloca(
            current_block, context.entry_block, [data, *shape, *strides]
        )
        callee_struct = self.get_callee_struct_for_param_tensor(
            param, current_block, data, shape, strides, flatten_struct
        )
        return callee_struct, alloca

    def pack_param_var(
        self, current_block: ir.Block, context: CallContext, param: spec.Var
    ) -> tuple[ir.Type, ir.Value]:
        """Pack a scalar or handle as a one-field struct.

        :return: The value's type (or the ``alternate_ir_type_fetch_func``
            type) and the ``alloca``
        """
        value: ir.Value = context.matched_var_binding[param]
        _, alloca = self.pack_values_to_alloca(
            current_block, context.entry_block, [value]
        )
        if param.alternate_ir_type_fetch_func is not None:
            return (param.alternate_ir_type_fetch_func(self), alloca)
        return (value.type, alloca)

    def pack_param_shape(
        self, current_block: ir.Block, context: CallContext, param: spec.Shape
    ) -> tuple[ir.Type, ir.Value]:
        """Pack the symbolic dimensions of a shape."""
        dynamic_args = [
            context.matched_var_binding[dim]
            for dim in param.shape
            if isinstance(dim, spec.Var)
        ]
        return self.pack_values_to_alloca(
            current_block, context.entry_block, dynamic_args
        )

    def pack_params(
        self, current_block: ir.Block, context: CallContext
    ) -> list[tuple[ir.Type, ir.Value]]:
        """Pack every forwarded parameter.

        :return: ``(struct type, alloca)`` per parameter, in order
        :raises DSLUserCodeError: ``UNSUP_PARAM`` for an unknown kind
        """
        if self.flatten_tuple_params:
            flattened_params = _flatten_tuple_params(context.params)
        else:
            flattened_params = context.params

        packed_params = []
        for param in flattened_params:
            if isinstance(param, spec.Tensor):
                packed_params.append(
                    self.pack_param_tensor(current_block, context, param)
                )
            elif isinstance(param, spec.Var):
                packed_params.append(self.pack_param_var(current_block, context, param))
            elif isinstance(param, spec.Shape):
                packed_params.append(
                    self.pack_param_shape(current_block, context, param)
                )
            elif isinstance(param, _HANDLE_PARAMS):
                packed_params.append(
                    self.pack_param_var(current_block, context, param.var)
                )
            elif isinstance(param, _CONST_PARAMS):
                continue
            else:
                raise DSLUserCodeError(
                    TvmFfiDiagId.UNSUP_PARAM,
                    detail=f"Unsupported parameter type: {type(param)}",
                )
        return packed_params

    def generate_llvm_call(
        self,
        current_block: ir.Block,
        call_operands: list[ir.Value],
        context: CallContext,
    ) -> ir.Block:
        """Emit ``llvm.call @target_func(call_operands)`` (no result)."""
        with ir.InsertionPoint(current_block):
            llvm.call(
                result=None,
                callee=self.target_func,
                callee_operands=call_operands,
                op_bundle_sizes=[],
                op_bundle_operands=[],
            )
        return current_block

    def load_to_call_operands(
        self,
        struct_type: Union[ir.Type, tuple[ir.Type]],
        alloca: Union[ir.Value, tuple[ir.Value]],
    ) -> list[ir.Value]:
        """Load the packed struct(s) back as by-value call operands."""
        if isinstance(struct_type, tuple) != isinstance(alloca, tuple):
            raise DSLRuntimeError(
                "load_to_call_operands: struct_type and alloca must both be "
                "single values or both be tuples"
            )
        if isinstance(struct_type, tuple):
            return [
                llvm.load(struct_type[i], alloca[i]) for i in range(len(struct_type))
            ]
        return [llvm.load(struct_type, alloca)]

    def __call__(self, current_block: ir.Block, context: CallContext) -> ir.Block:
        packed_params = self.pack_params(current_block, context)

        if self.struct_call:
            call_operands = []
            with ir.InsertionPoint(current_block):
                for struct_type, alloca in packed_params:
                    call_operands += self.load_to_call_operands(struct_type, alloca)
        else:
            # One more alloca holds the struct pointers: the ``void**`` array.
            all_values: list[Any] = []
            for _, value in packed_params:
                if isinstance(value, tuple):
                    all_values.extend(value)
                else:
                    all_values.append(value)
            _, packed_args_value = self.pack_values_to_alloca(
                current_block, context.entry_block, all_values
            )
            call_operands = [packed_args_value]
            if self.include_num_args:
                with ir.InsertionPoint(current_block):
                    call_operands.append(self.i32(len(all_values)))

        return self.generate_llvm_call(current_block, call_operands, context)


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
        for param in _flatten_tuple_params(context.params):
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
