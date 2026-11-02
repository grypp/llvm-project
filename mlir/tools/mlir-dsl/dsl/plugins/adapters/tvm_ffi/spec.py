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
from .diagnostics import TvmFfiDiagId

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


class _HandleParam(Param):
    """A parameter decoded as an opaque handle (``var`` holds the pointer).

    :param name: The parameter name
    """

    name: str
    var: Var

    def __init__(self, name: str) -> None:
        self.name = name
        self.var = Var(name, tvm_ffi.dtype("handle"))


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


_HANDLE_PARAM_NAMES: dict[type, str] = {}


def format_param_type(param: Param) -> str:
    """Render the type of ``param`` for :func:`signature`, nested tuples included.

    :param param: The parameter to format
    :return: ``int32``, ``Tensor([n, 128], float32)``, ``Shape([n, m])``,
        ``Int(4)``, ``Tuple[int32, DataPointer]``, ...
    :raises DSLUserCodeError: ``UNSUP_PARAM`` for an unknown kind
    """
    if isinstance(param, Var):
        return str(param.dtype)
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
    raise DSLUserCodeError(
        TvmFfiDiagId.UNSUP_PARAM,
        detail=f"Unsupported parameter type: {type(param)}",
    )


def signature(name: str, params: list[Param]) -> str:
    """Render the declared interface, ``name(p0: type0, p1: type1)``.

    :class:`EnvStream` parameters have no argument slot and are left out.

    :param name: The function name
    :param params: The parameter specifications in argument order
    :return: The signature text used in the wrapper's error messages
    :raises DSLUserCodeError: ``UNSUP_PARAM`` for an unknown kind
    """
    param_strs = [
        f"{param.name}: {format_param_type(param)}"  # type: ignore[attr-defined]
        for param in params
    ]
    return f"{name}({', '.join(param_strs)})"
