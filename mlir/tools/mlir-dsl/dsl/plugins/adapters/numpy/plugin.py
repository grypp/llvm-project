# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The numpy adapter plugin: ``numpy.ndarray`` arguments as host ``Pointer`` values.

``NumpyPlugin.register`` registers the type-keyed ``numpy.ndarray`` adapter: a
C-contiguous array becomes a host ``Pointer`` over its data, the dtype mapped
by name (``from_numpy_dtype``), the array kept alive for the call; no shape or
stride crosses the boundary, the length travels as its own argument. The plugin
is available when ``numpy`` is importable (``available()``) and imports it only
when a DSL naming the plugin is constructed. A DSL that does not name it
rejects an array with ``ARG_UNSUPPORTED_TYPE``; the core registers no host
buffer type of its own.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ....core.arguments import JitArgAdapterRegistry, _check_contiguous
from ....core.plugin import AdapterPlugin
from ....types.typing import Pointer, from_numpy_dtype

__all__ = ["NumpyPlugin", "available"]


def available() -> bool:
    """Whether ``numpy`` is importable (the plugin's ``available()``)."""
    return importlib.util.find_spec("numpy") is not None


def _convert_numpy_array(arg: Any) -> Pointer:
    """Adapt a C-contiguous numpy array to a host ``Pointer`` over its data: the
    dtype from the array (else ``TYPE_UNKNOWN_DTYPE_NAME``), the array kept
    alive for the call; no shape or stride crosses the boundary."""
    _check_contiguous(arg, arg.flags.c_contiguous)
    return Pointer(
        arg.ctypes.data,
        dtype=from_numpy_dtype(arg.dtype),
        kind="host",
        keepalive=arg,
    )


class NumpyPlugin(AdapterPlugin):
    """Adapts ``numpy.ndarray`` arguments to host ``Pointer`` addresses."""

    name = "numpy"

    @classmethod
    def available(cls) -> bool:
        return available()

    def register(self, dsl: Any) -> None:
        # The type-keyed registry is process-wide: one registration serves
        # every DSL instance. numpy is imported here, by the first DSL that
        # names the plugin, never by the core.
        import numpy as np

        if np.ndarray not in JitArgAdapterRegistry.jit_arg_adapter_registry:
            JitArgAdapterRegistry.register_jit_arg_adapter(np.ndarray)(
                _convert_numpy_array
            )
