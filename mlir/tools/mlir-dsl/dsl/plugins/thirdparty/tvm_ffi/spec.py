# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Parameter specifications of a TVM-FFI function.

A ``spec`` describes what the wrapper built by :func:`attach_ffi_func` expects
in each ``TVMFFIAny`` argument slot and what it hands to the
:class:`CallProvider`: a :class:`Var` is one scalar or handle, a
:class:`Tensor` a DLPack tensor whose shape and strides may bind symbolic
:class:`Var` dimensions (the same ``Var`` in two places asserts equality), a
:class:`Shape` a tuple of integers, the ``Const*`` kinds compile-time values
the wrapper asserts and does not forward, :class:`TupleParam` a nested
``ffi.Array`` and :class:`EnvStream` the framework's current stream, which has
no argument slot. :func:`signature` renders the declared interface as text
for error messages.

The ``tvm_ffi`` package is imported on first use only (``dtype`` parsing and
the device table), never at import of this module.
"""

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ....core.common import DSLUserCodeError
from ....core.diagnostics import DiagId

if TYPE_CHECKING:
    from ..... import ir
    from .mlir_builder import MLIRBuilder


class _LazyTvmFfi:
    """The ``tvm_ffi`` package, imported on first attribute access.

    Importing the builder must not import ``tvm_ffi`` (it pulls in torch when
    present and the package is optional); the specs and decoders need it only
    when a function is actually declared or attached.
    """

    def __getattr__(self, name: str) -> Any:
        import tvm_ffi as module

        return getattr(module, name)


tvm_ffi = _LazyTvmFfi()


class DefaultConfig:
    """The default device type of :class:`Tensor` specs, as a context manager.

    ``with DefaultConfig(device_type="cpu"): ...`` makes every ``Tensor``
    declared inside expect CPU tensors; the initial default is ``"cuda"``.

    :param device_type: The device type (``"cpu"``, ``"cuda"``, ...); None
        copies the current default
    """

    _current: Optional["DefaultConfig"] = None
    _old_current: Optional["DefaultConfig"] = None
    device_type: str

    def __init__(self, *, device_type: Optional[str] = None) -> None:
        if device_type is None:
            device_type = DefaultConfig.current().device_type  # type: ignore[union-attr]
        self.device_type = device_type

    def __enter__(self) -> "DefaultConfig":
        self._old_current = DefaultConfig._current
        DefaultConfig._current = self
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        DefaultConfig._current = self._old_current

    @classmethod
    def current(cls) -> Optional["DefaultConfig"]:
        """:return: The configuration in effect."""
        return cls._current

    @classmethod
    def _set_init_default_config(cls) -> None:
        """Install the initial default (``"cuda"``) without copying from one."""
        current = cls.__new__(cls)
        current.device_type = "cuda"
        cls._current = current


DefaultConfig._set_init_default_config()


class Param(ABC):
    """Base class of every parameter specification."""


class Var(Param):
    """One scalar or handle argument: an integer, a float, a bool or a pointer.

    An integer ``Var`` is range-checked against its ``dtype`` and narrowed
    from the ``int64`` slot; a float one is read from the ``float64`` slot (an
    int or bool argument converts); a ``"handle"`` is an opaque pointer.

    :param name: The parameter name, used in error messages
    :param dtype: The data type, a ``tvm_ffi.dtype`` or its spelling
        (``"int32"``, ``"float16"``, ``"handle"``, ...)
    :param divisibility: Assert that the integer value is a multiple of this
    :param alternate_ir_type_fetch_func: Returns the IR type a packing call
        provider forwards the value as when it is not the value's own type
    """

    name: str
    dtype: "tvm_ffi.dtype"
    divisibility: Optional[int]
    alternate_ir_type_fetch_func: Optional[Callable[["MLIRBuilder"], "ir.Type"]] = None

    def __init__(
        self,
        name: str,
        dtype: Union[str, "tvm_ffi.dtype"],
        *,
        divisibility: Optional[int] = None,
        alternate_ir_type_fetch_func: Optional[
            Callable[["MLIRBuilder"], "ir.Type"]
        ] = None,
    ) -> None:
        self.name = name
        self.dtype = tvm_ffi.dtype(dtype)
        self.divisibility = divisibility
        self.alternate_ir_type_fetch_func = alternate_ir_type_fetch_func


class Shape(Param):
    """A tuple of integers (an ``ffi.Shape`` or ``ffi.Array`` argument).

    :param name: The parameter name
    :param shape: One entry per dimension: an int asserts that value, a
        :class:`Var` binds (or, when already bound, asserts) the dimension
    """

    name: str
    shape: list[Union[int, Var]]

    def __init__(self, name: str, shape: list[Union[int, Var]]) -> None:
        self.name = name
        self.shape = shape


class Tensor(Param):
    """A DLPack tensor argument (``ffi.Tensor`` or ``DLTensor*``).

    The wrapper checks ``ndim``, ``dtype`` and the device type, binds or
    asserts every symbolic dimension and stride, and asserts contiguity when
    no ``strides`` are declared. ``data`` is the ``Var`` the call provider
    reads the (byte-offset adjusted) data pointer from; ``device_id`` the one
    bound to the tensor's device index.

    :param name: The parameter name
    :param shape: One int or :class:`Var` per dimension
    :param dtype: The element type, a ``tvm_ffi.dtype`` or its spelling
    :param device_type: The device type name; None takes
        :meth:`DefaultConfig.current`
    :param device_id: The ``Var`` bound to the device index; None creates
        ``<name>.device.index``
    :param strides: One int or :class:`Var` per dimension; None asserts a
        contiguous layout instead (a dimension of extent 1 is exempt)
    :param map_tensor_dtype_f4x2_to_f4: Hand a ``float4_e2m1fnx2`` tensor to the
        callee as the ``float4_e2m1fn`` tensor it packs
        (:func:`create_map_tensor_dtype_f4x2_to_f4_spec`)
    :param data_alignment: Assert that the data pointer is a multiple of this
        many bytes
    """

    name: str
    shape: list[Union[int, Var]]
    dtype: "tvm_ffi.dtype"
    strides: Optional[list[Var]]
    dlpack_device_type: int
    device_id: Var
    map_tensor_dtype_f4x2_to_f4: bool
    data_alignment: Optional[int]

    def __init__(
        self,
        name: str,
        shape: Sequence[Union[int, Var]],
        dtype: Union[str, "tvm_ffi.dtype"],
        *,
        device_type: Optional[str] = None,
        device_id: Optional[Var] = None,
        strides: Optional[Sequence[Var]] = None,
        map_tensor_dtype_f4x2_to_f4: bool = False,
        data_alignment: Optional[int] = None,
    ) -> None:
        self.name = name
        self.data = Var(name + ".data", tvm_ffi.dtype("handle"))
        self.shape: list[Union[int, Var]] = list(shape)
        self.dtype = tvm_ffi.dtype(dtype)
        self.strides: Optional[list[Var]] = (
            list(strides) if strides is not None else None
        )
        self.data_alignment = data_alignment

        if device_type is None:
            device_type = DefaultConfig.current().device_type  # type: ignore[union-attr]

        example_device = tvm_ffi.device(device_type, 0)
        self.dlpack_device_type = example_device.dlpack_device_type()
        self.device_type_name = example_device.type
        if device_id is None:
            self.device_id = Var(name + ".device.index", tvm_ffi.dtype("int32"))
        else:
            self.device_id = device_id
        self.map_tensor_dtype_f4x2_to_f4 = map_tensor_dtype_f4x2_to_f4


class _HandleParam(Param):
    """A parameter decoded as an opaque handle (``var`` holds the pointer).

    :param name: The parameter name
    """

    name: str
    var: Var

    def __init__(self, name: str) -> None:
        self.name = name
        self.var = Var(name, tvm_ffi.dtype("handle"))


class Stream(_HandleParam):
    """A stream handle argument (an opaque pointer, an int or None)."""


class CudaEvent(_HandleParam):
    """A CUDA event handle argument.

    TVM-FFI has a native stream type but no event type, so an event travels
    as an opaque handle (or its integer address) like the graph kinds below.
    """


class CudaGraph(_HandleParam):
    """A CUDA graph handle argument (opaque handle or integer address)."""


class CudaGraphNode(_HandleParam):
    """A CUDA graph node handle argument (opaque handle or integer address)."""


class EnvStream(_HandleParam):
    """The framework's current stream, obtained with ``TVMFFIEnvGetStream``.

    It has no argument slot and is absent from :func:`signature`; the wrapper
    queries it for the device of the first non-CPU :class:`Tensor` parameter,
    so a function declaring an ``EnvStream`` needs such a tensor.
    """


class DataPointer(_HandleParam):
    """A raw data pointer argument.

    Unlike a ``"handle"`` :class:`Var` it carries an address space, and an
    integer argument is accepted as the address (``torch.Tensor.data_ptr()``).

    :param name: The parameter name
    :param address_space: The LLVM address space of the pointer the callee
        receives; None or 0 is the generic one
    """

    address_space: Optional[int]

    def __init__(self, name: str, address_space: Optional[int] = None) -> None:
        super().__init__(name)
        self.address_space = address_space


class ConstNone(Param):
    """A compile-time ``None``: the slot must hold ``None``; nothing is forwarded.

    :param name: The parameter name
    """

    name: str

    def __init__(self, name: str) -> None:
        self.name = name


class ConstInt(Param):
    """A compile-time int: the slot must hold exactly ``value``; not forwarded.

    :param name: The parameter name
    :param value: The value the function was compiled for
    """

    name: str
    value: int

    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = int(value)


class ConstBool(Param):
    """A compile-time bool: the slot must hold a bool equal to ``value``.

    An int argument is rejected even when its value would match.

    :param name: The parameter name
    :param value: The value the function was compiled for
    """

    name: str
    value: bool

    def __init__(self, name: str, value: bool) -> None:
        self.name = name
        self.value = bool(value)


class ConstFloat(Param):
    """A compile-time float: the slot must hold a float equal to ``value``.

    :param name: The parameter name
    :param value: The value the function was compiled for
    """

    name: str
    value: float

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = float(value)


class TupleParam(Param):
    """An ``ffi.Array`` argument whose elements are decoded by ``params``.

    :param name: The parameter name
    :param params: One specification per element, nested tuples included
    """

    name: str
    params: list[Param]

    def __init__(self, name: str, params: list[Param]) -> None:
        self.name = name
        self.params = params


_HANDLE_PARAM_NAMES: dict[type, str] = {
    Stream: "Stream",
    CudaEvent: "CudaEvent",
    CudaGraph: "CudaGraph",
    CudaGraphNode: "CudaGraphNode",
    DataPointer: "DataPointer",
}


def _format_dims(dims: Sequence[Union[int, Var]]) -> str:
    """``[n, 128]``: symbolic dimensions by name, static ones by value."""
    return "[" + ", ".join(d.name if isinstance(d, Var) else str(d) for d in dims) + "]"


def format_param_type(param: Param) -> str:
    """Render the type of ``param`` for :func:`signature`, nested tuples included.

    :param param: The parameter to format
    :return: ``int32``, ``Tensor([n, 128], float32)``, ``Shape([n, m])``,
        ``Int(4)``, ``Tuple[int32, DataPointer]``, ...
    :raises DSLUserCodeError: ``UNSUP_TVM_FFI_PARAM`` for an unknown kind
    """
    if isinstance(param, Var):
        return str(param.dtype)
    if isinstance(param, Tensor):
        return f"Tensor({_format_dims(param.shape)}, {param.dtype})"
    if isinstance(param, Shape):
        return f"Shape({_format_dims(param.shape)})"
    for handle_kind, handle_name in _HANDLE_PARAM_NAMES.items():
        if isinstance(param, handle_kind):
            return handle_name
    if isinstance(param, ConstNone):
        return "None"
    if isinstance(param, ConstInt):
        return f"Int({param.value})"
    if isinstance(param, ConstBool):
        return f"Bool({param.value})"
    if isinstance(param, ConstFloat):
        return f"Float({param.value})"
    if isinstance(param, TupleParam):
        return f"Tuple[{', '.join(format_param_type(p) for p in param.params)}]"
    raise DSLUserCodeError(
        DiagId.UNSUP_TVM_FFI_PARAM,
        detail=f"Unsupported parameter type: {type(param)}",
    )


def signature(name: str, params: list[Param]) -> str:
    """Render the declared interface, ``name(p0: type0, p1: type1)``.

    :class:`EnvStream` parameters have no argument slot and are left out.

    :param name: The function name
    :param params: The parameter specifications in argument order
    :return: The signature text used in the wrapper's error messages
    :raises DSLUserCodeError: ``UNSUP_TVM_FFI_PARAM`` for an unknown kind
    """
    param_strs = [
        f"{param.name}: {format_param_type(param)}"  # type: ignore[attr-defined]
        for param in params
        if not isinstance(param, EnvStream)
    ]
    return f"{name}({', '.join(param_strs)})"


def create_map_tensor_dtype_f4x2_to_f4_spec(f4_tensor_spec: Tensor) -> Tensor:
    """The ``float4_e2m1fnx2`` spec that accepts a packed ``float4_e2m1fn`` tensor.

    The stride-1 dimension and every other stride halve (two nibbles per
    byte); the packing call provider doubles them back with
    ``map_tensor_dtype_f4x2_to_f4``.

    :param f4_tensor_spec: A :class:`Tensor` spec of dtype ``float4_e2m1fn``
    :return: The equivalent ``float4_e2m1fnx2`` spec
    :raises DSLUserCodeError: ``UNSUP_TVM_FFI_PARAM`` when the spec is not a
        float4 tensor, has no stride-1 dimension or a static extent, stride or
        divisibility that is not even
    """
    if f4_tensor_spec.dtype != tvm_ffi.dtype("float4_e2m1fn"):
        raise DSLUserCodeError(
            DiagId.UNSUP_TVM_FFI_PARAM, detail="f4_tensor_spec must be a float4 tensor"
        )

    def find_stride_one_index() -> int:
        if f4_tensor_spec.strides is None:
            return len(f4_tensor_spec.shape) - 1
        for i, stride in enumerate(f4_tensor_spec.strides):
            if isinstance(stride, int) and stride == 1:
                return i
        raise DSLUserCodeError(
            DiagId.UNSUP_TVM_FFI_PARAM, detail="Cannot find dimension with stride=1"
        )

    stride_one_index = find_stride_one_index()

    def divisibility_divide_by_2(value: Var) -> Optional[int]:
        if value.divisibility is None:
            return None
        if value.divisibility % 2 != 0:
            raise DSLUserCodeError(
                DiagId.UNSUP_TVM_FFI_PARAM,
                detail="Dimension with stride=1 must be divisible by 2",
            )
        return value.divisibility // 2

    def halve(index: int, value: Union[int, Var], what: str) -> Union[int, Var]:
        if isinstance(value, int):
            if value % 2 != 0:
                raise DSLUserCodeError(
                    DiagId.UNSUP_TVM_FFI_PARAM,
                    detail=f"Dimension {index} with {what} must be even",
                )
            return value // 2
        return Var(
            value.name, value.dtype, divisibility=divisibility_divide_by_2(value)
        )

    def map_shape(index: int, value: Union[int, Var]) -> Union[int, Var]:
        if index == stride_one_index:
            return halve(index, value, "stride=1")
        return value

    def map_stride(index: int, value: Union[int, Var]) -> Union[int, Var]:
        if index != stride_one_index:
            return halve(index, value, "stride != 1")
        return value

    new_shape = [map_shape(i, x) for i, x in enumerate(f4_tensor_spec.shape)]
    if f4_tensor_spec.strides is not None:
        new_strides = [map_stride(i, x) for i, x in enumerate(f4_tensor_spec.strides)]
    else:
        new_strides = None

    return Tensor(
        f4_tensor_spec.name,
        new_shape,
        dtype=tvm_ffi.dtype("float4_e2m1fnx2"),
        strides=new_strides,  # type: ignore[arg-type]
        map_tensor_dtype_f4x2_to_f4=True,
        device_type=f4_tensor_spec.device_type_name,
        device_id=f4_tensor_spec.device_id,
        data_alignment=f4_tensor_spec.data_alignment,
    )
