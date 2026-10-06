# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Torch dtype <-> DSL Numeric type conversion utilities.

Two directions: :func:`dtype` maps a DSL type to the ``torch.dtype`` of the
same name, :func:`from_torch_dtype` maps a ``torch.dtype`` (or its name) back.
``torch`` is imported lazily, inside :func:`dtype`, so the module imports
without it; :func:`from_torch_dtype` works on the dtype's name and needs no
torch at all (the ``torch.Tensor`` argument adapter relies on that).
"""

from typing import Any, Type

from ....core.common import DSLRuntimeError
from ....types.typing import (
    BFloat16,
    Boolean,
    Float4E2M1FN,
    Float8E4M3B11FNUZ,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Float8E8M0FNU,
    Int2,
    Int4,
    Numeric,
    TFloat32,
    from_numpy_dtype,
)

__all__ = ["dtype", "from_torch_dtype"]


def dtype(ty: Type[Numeric]) -> Any:
    """Return the ``torch.dtype`` corresponding to the DSL type ``ty``.

    Types torch spells as the lower-cased DSL name (``Int32`` -> ``torch.int32``,
    ``Float16`` -> ``torch.float16``) resolve by name; the others go through an
    explicit table (``Boolean`` -> ``torch.bool``, ``TFloat32`` -> ``torch.float32``,
    the FP8/FP4 family). Entries torch added recently (``float8_e8m0fnu``,
    ``float4_e2m1fn_x2``) are present only when the installed torch has them.

    :param ty: A DSL ``Numeric`` class such as ``Float32``.
    :return: The ``torch.dtype``.
    :raises DSLRuntimeError: When torch has no dtype for ``ty``.
    """
    import torch

    torch_dtype = getattr(torch, ty.__name__.lower(), None)

    torch_type_map = {
        Boolean: torch.bool,
        # TFloat32 is just alias of float32
        TFloat32: torch.float32,
        Float8E5M2: torch.float8_e5m2,
        Float8E5M2FNUZ: torch.float8_e5m2fnuz,
        Float8E4M3FN: torch.float8_e4m3fn,
        Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
        Float8E4M3FNUZ: torch.float8_e4m3fnuz,
    }

    # float8_e8m0fnu / float4_e2m1fn_x2 are introduced in later versions of torch
    if hasattr(torch, "float8_e8m0fnu"):
        torch_type_map[Float8E8M0FNU] = torch.float8_e8m0fnu
    if hasattr(torch, "float4_e2m1fn_x2"):
        torch_type_map[Float4E2M1FN] = torch.float4_e2m1fn_x2

    if torch_dtype is None:
        torch_dtype = torch_type_map.get(ty)

    if torch_dtype is None:
        raise DSLRuntimeError(f"{ty} is not supported by torch")
    return torch_dtype


# torch dtype names that are not spelled as NumPy spells them.
_TORCH_NAME_TO_DTYPE: dict[str, Type[Numeric]] = {
    "bool": Boolean,
    "bfloat16": BFloat16,
    "int2": Int2,
    "int4": Int4,
    "float8_e5m2": Float8E5M2,
    "float8_e5m2fnuz": Float8E5M2FNUZ,
    "float8_e4m3fn": Float8E4M3FN,
    "float8_e4m3fnuz": Float8E4M3FNUZ,
    "float8_e8m0fnu": Float8E8M0FNU,
    "float4_e2m1fn_x2": Float4E2M1FN,
}


def from_torch_dtype(torch_dtype: Any) -> Type[Numeric]:
    """Return the DSL type corresponding to a ``torch.dtype`` (or its name).

    The dtype is matched by name, so no torch import is needed: a
    ``torch.dtype`` prints as ``torch.<name>`` and the prefix is stripped.
    Names torch shares with NumPy (``float32``, ``int8``, ...) resolve through
    :func:`from_numpy_dtype`; a name the DSL has no type for is
    ``TYPE_UNKNOWN_DTYPE_NAME``.

    :param torch_dtype: A ``torch.dtype`` or its name (``"float32"``, ``"torch.float32"``).
    :return: The DSL ``Numeric`` class.
    """
    name = str(torch_dtype).removeprefix("torch.")
    ty = _TORCH_NAME_TO_DTYPE.get(name)
    if ty is not None:
        return ty
    return from_numpy_dtype(name)
