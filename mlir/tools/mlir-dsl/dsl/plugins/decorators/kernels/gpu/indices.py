# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``gpu`` dialect's op library: the kernel-body index helpers and the
grid-constant annotation marker.

``thread_idx``/``block_idx``/``block_dim``/``grid_dim`` read the NVVM special
registers as ``Int32`` inside a ``gpu.func`` body; ``Annotated[T, grid_constant]``
(``GridConstant[T]``) tags a kernel argument as a grid constant. The kernel
function, its container and its launch are the gpu kernels plugin's
(``plugins/decorators/kernels/gpu/__init__.py``); this module emits ops only and
is not a plugin.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias, TypeVar

from ...... import ir

try:
    from ......dialects import gpu, nvvm
except ImportError:  # the gpu bindings are not built into this MLIR
    gpu = nvvm = None  # type: ignore[assignment]
from .....core.common import DSLUserCodeError
from .....core.diagnostics import DiagId
from .....core.user_op import dsl_user_op
from .....types.typing import Int32
from ..launch import KernelsPlugin

__all__ = [
    "GridConstant",
    "block_dim",
    "block_idx",
    "grid_constant",
    "grid_dim",
    "thread_idx",
]


# =============================================================================
# Kernel-argument annotation markers
# =============================================================================

TY = TypeVar("TY")


class _GridConstantMarker:
    """``Annotated`` marker that tags a kernel argument as a grid constant.

    ``BaseDSL._extract_annotation_markers`` turns it into one
    ``{cuda.grid_constant}`` argument attribute on the generated kernel.
    """

    def __extract_mlir_attributes__(self) -> list:
        return [ir.DictAttr.get({"cuda.grid_constant": ir.UnitAttr.get()})]


# The one marker instance: ``Annotated[Int32, grid_constant]``.
grid_constant: _GridConstantMarker = _GridConstantMarker()

# ``GridConstant[T]`` spells ``Annotated[T, grid_constant]``.
GridConstant: TypeAlias = Annotated[TY, grid_constant]


# =============================================================================
# Kernel-body index helpers
# =============================================================================

_SREGS = {
    "thread_idx": "tid",
    "block_idx": "ctaid",
    "block_dim": "ntid",
    "grid_dim": "nctaid",
}


def _in_kernel_body() -> bool:
    """Whether the current insertion point sits inside a ``gpu.func``."""
    if ir.Context.current is None:
        return False
    try:
        op = ir.InsertionPoint.current.block.owner
    except ValueError:
        return False
    while op is not None:
        if op.operation.name == gpu.GPUFuncOp.OPERATION_NAME:
            return True
        op = op.operation.parent
    return False


def _sreg_indices(api: str, *, loc: Any, ip: Any) -> tuple[Int32, Int32, Int32]:
    """Read the ``x``/``y``/``z`` special register of ``api`` as ``Int32``."""
    if not _in_kernel_body():
        raise DSLUserCodeError(
            DiagId.CALL_OUTSIDE_JIT,
            api=f"{api}()",
            decorator=f"@{KernelsPlugin.decorator_name}",
        )
    x, y, z = (
        Int32(getattr(nvvm, f"read_ptx_sreg_{_SREGS[api]}_{axis}")(loc=loc, ip=ip))
        for axis in "xyz"
    )
    return x, y, z


@dsl_user_op
def thread_idx(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` index of this thread in its block (``%tid``)."""
    return _sreg_indices("thread_idx", loc=loc, ip=ip)


@dsl_user_op
def block_idx(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` index of this block in the grid (``%ctaid``)."""
    return _sreg_indices("block_idx", loc=loc, ip=ip)


@dsl_user_op
def block_dim(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` size of a block in threads (``%ntid``)."""
    return _sreg_indices("block_dim", loc=loc, ip=ip)


@dsl_user_op
def grid_dim(*, loc: Any = None, ip: Any = None) -> tuple[Int32, Int32, Int32]:
    """The ``(x, y, z)`` size of the grid in blocks (``%nctaid``)."""
    return _sreg_indices("grid_dim", loc=loc, ip=ip)
