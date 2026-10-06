# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DLPack tensors as ``Pointer`` arguments.

:class:`DlpackTensor` describes any object with ``__dlpack__`` and
``__dlpack_device__`` (NumPy, PyTorch, CuPy, JAX, ...): data pointer, shape,
strides, element type and device, read through the ``_mlirDslDlpack``
extension, which imports the tensor with nanobind's ndarray and owns the
producer's lease while the descriptor lives. :class:`DlpackPlugin` registers
the protocol adapter that turns such an argument into a contiguous ``Pointer``
of the matching dtype at the host boundary. This follows the
``TensorDescriptor`` of DkgDSL's runtime.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

from ....core.common import DSLUserCodeError
from ....core.diagnostics import DiagId
from ....core.plugin import Plugin
from ....runtime.jit_arg_adapters import JitArgAdapterRegistry, _check_contiguous
from ....types.typing import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Numeric,
    Pointer,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from ....util.logger import log

__all__ = ["DlpackPlugin", "DlpackTensor", "available", "speaks_dlpack"]

_EXTENSION = "mlir._mlir_libs._mlirDslDlpack"

# (DLPack type code, bits) -> DSL element type; lanes must be 1.
_DTYPES: dict[tuple[int, int], type[Numeric]] = {
    (0, 8): Int8,
    (0, 16): Int16,
    (0, 32): Int32,
    (0, 64): Int64,
    (1, 8): Uint8,
    (1, 16): Uint16,
    (1, 32): Uint32,
    (1, 64): Uint64,
    (2, 16): Float16,
    (2, 32): Float32,
    (2, 64): Float64,
    (4, 16): BFloat16,
    (6, 8): Boolean,
}

# DLPack device type -> the ``Pointer`` kind of the host payload.
_DEVICE_KIND = {1: "host", 2: "device", 3: "host", 13: "device"}


def available() -> bool:
    """Whether the ``_mlirDslDlpack`` extension is part of the build."""
    return importlib.util.find_spec(_EXTENSION) is not None


def speaks_dlpack(obj: object) -> bool:
    """Whether ``obj`` implements the DLPack protocol."""
    return hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__")


class DlpackTensor:
    """The metadata of a DLPack tensor, valid while this object lives.

    :param tensor: Any object with ``__dlpack__``
    :raises DSLUserCodeError: ``TYPE_UNKNOWN_DTYPE_NAME`` for an element type
        the DSL has no numeric type for
    """

    def __init__(self, tensor: Any) -> None:
        from mlir._mlir_libs import _mlirDslDlpack

        self.tensor = tensor
        self._view = _mlirDslDlpack.TensorView(tensor)
        key = (self._view.dtype_code, self._view.dtype_bits)
        dtype = _DTYPES.get(key) if self._view.dtype_lanes == 1 else None
        if dtype is None:
            raise DSLUserCodeError(
                DiagId.TYPE_UNKNOWN_DTYPE_NAME,
                name=f"dlpack(code={key[0]}, bits={key[1]}, "
                f"lanes={self._view.dtype_lanes})",
            )
        self.dtype: type[Numeric] = dtype
        log().debug("DlpackTensor created [%s]", self)

    @property
    def data_ptr(self) -> int:
        """The address of the first element, on the host or the device."""
        return self._view.data_ptr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._view.shape

    @property
    def strides(self) -> tuple[int, ...]:
        """The strides in elements."""
        return self._view.strides

    @property
    def rank(self) -> int:
        return self._view.ndim

    @property
    def device(self) -> str:
        """``"host"``, ``"device"`` (CUDA) or ``"unknown"``."""
        return _DEVICE_KIND.get(self._view.device_type, "unknown")

    @property
    def device_id(self) -> int:
        return self._view.device_id

    @property
    def is_contiguous(self) -> bool:
        """Whether the elements form one row-major block."""
        expected = 1
        for extent, stride in zip(reversed(self.shape), reversed(self.strides)):
            if extent > 1 and stride != expected:
                return False
            expected *= extent
        return True

    @property
    def size_in_bytes(self) -> int:
        return math.prod(self.shape) * self.dtype.width // 8

    def pointer(self) -> Pointer:
        """A host ``Pointer`` over the data, keeping this descriptor alive."""
        return Pointer(
            self.data_ptr, dtype=self.dtype, kind=self.device, keepalive=self
        )

    def __repr__(self) -> str:
        shape = "x".join(map(str, self.shape))
        return f"tensor<{shape}x{self.dtype.__name__}>_{self.device}"


def _convert_dlpack(arg: Any) -> Pointer:
    """The protocol adapter: a contiguous DLPack tensor becomes a ``Pointer``."""
    tensor = DlpackTensor(arg)
    _check_contiguous(arg, tensor.is_contiguous)
    return tensor.pointer()


class DlpackPlugin(Plugin):
    """Adapts every argument speaking the DLPack protocol to a ``Pointer``."""

    name = "dlpack"

    @classmethod
    def available(cls) -> bool:
        return available()

    def install(self, dsl: Any) -> None:
        super().install(dsl)
        # The protocol registry is process-wide: one registration serves
        # every DSL instance. Type-keyed adapters (NumPy, PyTorch) win first.
        JitArgAdapterRegistry.register_protocol_adapter(speaks_dlpack, _convert_dlpack)
