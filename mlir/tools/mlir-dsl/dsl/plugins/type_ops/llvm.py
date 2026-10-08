# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``llvm`` dialect: ``!llvm.ptr`` memory ops and pointer casts.

The loads, stores and address arithmetic behind the core ``Pointer`` (with the
legalisation the ``llvm`` dialect needs for sub-byte float element types) and
the casts between pointers, integers and address spaces. An op module, not a
plugin: the ``UpstreamDialectTypeOps`` composer routes its memory hooks here, and any
DSL may call these functions directly."""

from typing import Any, Optional

from .... import ir
from ....dialects import llvm
from ...core.common import DSLUserCodeError
from . import arith as _arith

__all__ = [
    "addrspacecast",
    "inttoptr",
    "load",
    "pointer_space",
    "pointer_type",
    "ptr_add",
    "ptrtoint",
    "store",
]

# ``llvm.getelementptr``'s sentinel for "this index is the next dynamic operand".
MLIR_DYNAMIC_INDEX = -(2**31)


# The LLVM dialect knows only the floating-point types LLVM itself has (f16,
# bf16, f32, f64, f80, f128): its load, store and getelementptr verifiers and
# ``llvm.bitcast`` reject the others (tf32, the fp8/fp6/fp4 dtypes). Scalar
# loads and stores of those go through the same-width integer type and an
# ``arith.bitcast`` (which the arith lowering turns into the LLVM cast);
# vectors pass through unchanged, and ``getelementptr`` strides by the integer.
_LLVM_FLOAT_WIDTHS = frozenset({16, 32, 64, 80, 128})


def _legalized_int_type(mlir_type: ir.Type) -> Optional[ir.Type]:
    """The same-width signless integer for a scalar float type the LLVM
    dialect does not accept, else None."""
    if (
        isinstance(mlir_type, ir.FloatType)
        and mlir_type.width not in _LLVM_FLOAT_WIDTHS
    ):
        return ir.IntegerType.get_signless(mlir_type.width)
    return None


def _gep(
    base: ir.Value,
    elem_type: ir.Type,
    *,
    static_indices: Optional[list] = None,
    dynamic_indices: Optional[list] = None,
    loc: object = None,
    ip: object = None,
) -> ir.Value:
    """Helper for LLVM getelementptr operations."""
    if static_indices is None:
        static_indices = []
    if dynamic_indices is None:
        dynamic_indices = []

    return llvm.getelementptr(
        base.type,
        base,
        dynamic_indices,
        static_indices,
        _legalized_int_type(elem_type) or elem_type,
        no_wrap_flags="None",
        loc=loc,
        ip=ip,
    )


def _legalized_llvm_load(
    result_type: ir.Type, addr: ir.Value, **kwargs: Any
) -> ir.Value:
    """``llvm.load`` with legalization for scalar subword float types.

    Vector types pass through untouched.
    """
    if not isinstance(result_type, ir.VectorType):
        int_type = _legalized_int_type(result_type)
        if int_type is not None:
            result = llvm.load(int_type, addr, **kwargs)
            loc = kwargs.get("loc")
            ip = kwargs.get("ip")
            return _arith.bitcast(result, result_type, loc=loc, ip=ip)
    return llvm.load(result_type, addr, **kwargs)


def _legalized_llvm_store(value: ir.Value, addr: ir.Value, **kwargs: Any) -> Any:
    """``llvm.store`` with legalization for scalar subword float types.

    Vector types pass through untouched.
    """
    if not isinstance(value.type, ir.VectorType):
        int_type = _legalized_int_type(value.type)
        if int_type is not None:
            loc = kwargs.get("loc")
            ip = kwargs.get("ip")
            value = _arith.bitcast(value, int_type, loc=loc, ip=ip)
    return llvm.store(value, addr, **kwargs)


def _atomic_ordering(ordering: Any) -> Any:
    """Map the ``ordering`` keyword of ``load``/``store`` to the LLVM attribute."""
    if ordering is None or ordering == "not_atomic":
        return None
    if isinstance(ordering, llvm.AtomicOrdering):
        return ordering
    if isinstance(ordering, str) and ordering in llvm.AtomicOrdering.__members__:
        return llvm.AtomicOrdering[ordering]
    raise DSLUserCodeError(
        f"`{ordering!r}` is not a memory ordering.",
        suggestion="Use one of `not_atomic`, `unordered`, `monotonic`, `acquire`, "
        "`release`, `acq_rel`, `seq_cst`.",
    )


def pointer_type(dtype: Any, address_space: int) -> ir.Type:
    """``!llvm.ptr`` (opaque) in ``address_space``; the element type is metadata."""
    return llvm.PointerType.get(address_space)


def pointer_space(mlir_type: ir.Type) -> Optional[int]:
    """The address space of an ``!llvm.ptr`` type, else None."""
    if isinstance(mlir_type, llvm.PointerType):
        return llvm.PointerType(mlir_type).address_space
    return None


def ptr_add(
    ptr: ir.Value, dtype: Any, index: Any, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``ptr + index`` elements of ``dtype``: one ``llvm.getelementptr``."""
    if isinstance(index, ir.Value):
        return _gep(
            ptr,
            dtype.scalar_mlir_type,
            static_indices=[MLIR_DYNAMIC_INDEX],
            dynamic_indices=[index],
            loc=loc,
            ip=ip,
        )
    return _gep(
        ptr, dtype.scalar_mlir_type, static_indices=[int(index)], loc=loc, ip=ip
    )


def load(
    ptr: ir.Value,
    dtype: Any,
    *,
    lanes: Optional[int] = None,
    mask: Optional[ir.Value] = None,
    pass_thru: Optional[ir.Value] = None,
    alignment: Optional[int] = None,
    volatile: bool = False,
    invariant: bool = False,
    invariant_group: bool = False,
    ordering: Any = None,
    syncscope: Optional[str] = None,
    loc: Any = None,
    ip: Any = None,
) -> ir.Value:
    """A scalar, a ``vector<lanes x T>`` or a masked vector load through ``ptr``."""
    elem = dtype.scalar_mlir_type
    res_ty = elem if lanes is None else ir.VectorType.get([lanes], elem)
    if mask is not None:
        return llvm.intr_masked_load(
            res_ty, ptr, mask, alignment=alignment, pass_thru=pass_thru, loc=loc, ip=ip
        )
    return _legalized_llvm_load(
        res_ty,
        ptr,
        alignment=alignment,
        volatile_=volatile,
        nontemporal=False,
        invariant=invariant,
        invariant_group=invariant_group,
        ordering=_atomic_ordering(ordering),
        syncscope=syncscope,
        loc=loc,
        ip=ip,
    )


def store(
    ptr: ir.Value,
    value: ir.Value,
    *,
    mask: Optional[ir.Value] = None,
    alignment: Optional[int] = None,
    volatile: bool = False,
    invariant_group: bool = False,
    ordering: Any = None,
    syncscope: Optional[str] = None,
    loc: Any = None,
    ip: Any = None,
) -> None:
    """A scalar, vector or masked vector store through ``ptr``."""
    if mask is not None:
        llvm.intr_masked_store(value, ptr, mask, alignment=alignment, loc=loc, ip=ip)
        return
    _legalized_llvm_store(
        value,
        ptr,
        alignment=alignment,
        volatile_=volatile,
        nontemporal=False,
        invariant_group=invariant_group,
        ordering=_atomic_ordering(ordering),
        syncscope=syncscope,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Pointer casts
# =============================================================================


def inttoptr(
    value: ir.Value, address_space: int, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``llvm.inttoptr``: an integer address as an ``!llvm.ptr`` in ``address_space``."""
    return llvm.inttoptr(llvm.PointerType.get(address_space), value, loc=loc, ip=ip)


def ptrtoint(
    ptr: ir.Value, int_type: ir.Type, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``llvm.ptrtoint``: the address of ``ptr`` as an integer of ``int_type``."""
    return llvm.ptrtoint(int_type, ptr, loc=loc, ip=ip)


def addrspacecast(
    ptr: ir.Value, address_space: int, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``llvm.addrspacecast``: ``ptr`` re-qualified in ``address_space``."""
    return llvm.addrspacecast(llvm.PointerType.get(address_space), ptr, loc=loc, ip=ip)
