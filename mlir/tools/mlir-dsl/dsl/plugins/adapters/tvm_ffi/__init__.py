# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helper tool to build TVM-FFI functions using MLIR.

``spec`` declares the interface of a function (scalars, tensors with symbolic
shapes, handles, compile-time constants); :func:`attach_ffi_func` adds an
``llvm.func @__tvm_ffi_<name>`` with the TVM-FFI ABI to a module that decodes
and checks the ``TVMFFIAny`` arguments and hands them to a
:class:`CallProvider`, which emits the call into the DSL's own function. The
``tvm_ffi`` package is needed only to call the result; the builder itself is
plain MLIR emission. The DSL-side use is :class:`TvmFfiPlugin` (``plugin.py``).
"""

from . import spec
from .diagnostics import TvmFfiDiagId
from .call_provider import (
    DirectCallProvider,
    DynamicParamPackCallProvider,
    NopCallProvider,
)
from .plugin import (
    NumericToTVMFFIDtype,
    TvmFfiJitCompiledFunction,
    TvmFfiPlugin,
    available,
    tvm_ffi_symbol,
)
from .spec import Param, Var
from .tvm_ffi_builder import (
    CallContext,
    CallProvider,
    TVMFFIBuilder,
    TVMFFIFunctionBuilder,
    TVMFFITypeIndex,
    attach_ffi_func,
    rename_tvm_ffi_function,
)

__all__ = [
    "TvmFfiDiagId",
    "NumericToTVMFFIDtype",
    "TvmFfiJitCompiledFunction",
    "TvmFfiPlugin",
    "available",
    "tvm_ffi_symbol",
    "CallContext",
    "CallProvider",
    "DirectCallProvider",
    "DynamicParamPackCallProvider",
    "NopCallProvider",
    "Param",
    "TVMFFIBuilder",
    "TVMFFIFunctionBuilder",
    "TVMFFITypeIndex",
    "Var",
    "attach_ffi_func",
    "rename_tvm_ffi_function",
    "spec",
]
