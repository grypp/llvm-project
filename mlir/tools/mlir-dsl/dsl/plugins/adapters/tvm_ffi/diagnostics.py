# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The tvm_ffi plugin's diagnostics, rendered ``error[tvm_ffi:CODE]``.

The export (``spec``, ``tvm_ffi_builder``, ``call_provider``) raises
``UNSUP_PARAM`` for a parameter or result the TVM-FFI ABI wrapper cannot carry;
the wrapped call raises ``CALL_REJECTED`` when the wrapper refuses the
arguments at run time. The core catalog knows nothing of them.
"""

import enum

from ....core.diagnostics import UNSUPPORTED, USAGE, DiagCatalog, classify

__all__ = ["TvmFfiDiagId"]


class TvmFfiDiagId(DiagCatalog, enum.Enum):
    """Member name == stable code, value == (message, fixes)."""

    namespace = enum.nonmember("tvm_ffi")

    UNSUP_PARAM = (
        "The TVM-FFI export does not support this parameter: {detail}.",
        (
            "Exported parameters are DSL scalars (`Int32`, `Float32`, ...), `Pointer[T]` "
            "buffers and compile-time int/bool/float/None values.",
        ),
    )
    CALL_REJECTED = (
        "The TVM-FFI call of `{function_name}` rejected its arguments: {detail}",
        (
            "Check the argument count and types against the function signature; a "
            "compile-time argument must repeat the value the function was compiled with.",
        ),
    )


classify(
    TvmFfiDiagId,
    UNSUPPORTED,
    "a parameter the TVM-FFI export cannot carry",
    "UNSUP_PARAM",
)
classify(TvmFfiDiagId, USAGE, "arguments", "CALL_REJECTED")
