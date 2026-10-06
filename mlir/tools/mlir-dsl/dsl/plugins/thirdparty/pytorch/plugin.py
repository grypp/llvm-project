# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The PyTorch plugin: ``torch.Tensor`` arguments as ``Pointer`` payloads.

``PyTorchPlugin.install`` registers the lazy ``torch.Tensor`` adapter (by
qualified name, so ``torch`` is never imported by the DSL); a tensor then
reaches a ``Pointer[T]`` parameter through ``data_ptr()``, its dtype mapped by
name (``dtypes.from_torch_dtype``) and its device deciding the payload kind.
The DSL copies nothing: the caller owns the memory. ``MlirDSL`` lists the
plugin when ``torch`` is importable.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ....core.plugin import Plugin
from ....runtime.jit_arg_adapters import JitArgAdapterRegistry, _check_contiguous
from ....types.typing import Pointer
from .dtypes import from_torch_dtype

__all__ = ["PyTorchPlugin", "available"]

TORCH_TENSOR_QUALNAME = "torch.Tensor"


def available() -> bool:
    """Whether a DSL class should list ``PyTorchPlugin()``: ``torch`` is importable."""
    return importlib.util.find_spec("torch") is not None


# ``torch.device.type`` -> host payload kind; anything else is unchecked.
_TORCH_DEVICE_KIND = {"cpu": "host", "cuda": "device"}


def _convert_torch_tensor(arg: Any) -> Pointer:
    """Adapt a PyTorch tensor, host or CUDA, to a ``Pointer``.

    Reads ``data_ptr()``, ``dtype`` (mapped by name), ``device.type`` and
    ``is_contiguous()`` only; a non-contiguous tensor is rejected.
    """
    _check_contiguous(arg, arg.is_contiguous())
    return Pointer(
        arg.data_ptr(),
        dtype=from_torch_dtype(arg.dtype),
        kind=_TORCH_DEVICE_KIND.get(arg.device.type, "unknown"),
        keepalive=arg,
    )


class PyTorchPlugin(Plugin):
    """Registers the ``torch.Tensor`` argument adapter when installed."""

    name = "pytorch"

    @classmethod
    def available(cls) -> bool:
        return available()

    def install(self, dsl: Any) -> None:
        # The lazy registry is process-wide and keyed by qualified name: one
        # registration serves every DSL instance.
        if (
            TORCH_TENSOR_QUALNAME
            not in JitArgAdapterRegistry.lazy_jit_arg_adapter_registry
        ):
            JitArgAdapterRegistry.register_jit_arg_adapter(
                TORCH_TENSOR_QUALNAME, lazy=True
            )(_convert_torch_tensor)
