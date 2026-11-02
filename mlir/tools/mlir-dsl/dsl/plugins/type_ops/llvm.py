# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``llvm`` dialect: ``!llvm.ptr`` memory ops and pointer casts.

The loads, stores and address arithmetic behind the core ``Pointer`` (with the
legalisation the ``llvm`` dialect needs for the float element types it lacks)
and the casts between pointers, integers and address spaces. An op module, not a
plugin: the ``UpstreamDialectTypeOps`` composer routes its memory hooks here, and any
DSL may call these functions directly."""

from typing import Any, Optional

from .... import ir
from ....dialects import builtin, llvm
from ...core.common import DSLUserCodeError
from . import arith as _arith

__all__ = [
    "addrspacecast",
    "gep",
    "inttoptr",
    "legalized_load",
    "legalized_store",
    "load",
    "pointer_space",
    "pointer_type",
    "ptr_add",
    "ptrtoint",
    "storage_type",
    "store",
]

# ``llvm.getelementptr``'s sentinel for "this index is the next dynamic operand".
MLIR_DYNAMIC_INDEX = -(2**31)


# The LLVM dialect knows only the floating-point types LLVM itself has (f16,
# bf16, f32, f64, f80, f128): its load, store and getelementptr verifiers and
# ``llvm.bitcast`` reject the others (tf32, the fp8/fp6/fp4 dtypes). Scalar
# loads and stores of those go through a signless integer, the storage type
# (``storage_type``): the 32-bit container for tf32, the same-width integer
# for the narrow formats. The conversion between a value and its storage is an
# ``arith.bitcast`` (which the arith lowering turns into the LLVM cast); tf32
# and i32 differ in width, which ``arith.bitcast`` rejects, so that pair goes
# through ``builtin.unrealized_conversion_cast``, for a later lowering to
# reconcile. Vectors pass through unchanged, and ``getelementptr`` strides by
# the storage type.
_LLVM_FLOAT_WIDTHS = frozenset({16, 32, 64, 80, 128})


def storage_type(mlir_type: ir.Type) -> Optional[ir.Type]:
    """The signless integer the LLVM dialect stores a scalar of ``mlir_type``
    as, or None when the type is legal there.

    ``tf32`` is stored in its 32-bit container; the other float types the
    dialect lacks (the fp8, fp6 and fp4 formats) in the integer of their
    width. Any DSL that lays out memory for the ``llvm`` dialect strides and
    loads by the same rule.
    """
    if isinstance(mlir_type, ir.FloatTF32Type):
        return ir.IntegerType.get_signless(32)
    if (
        isinstance(mlir_type, ir.FloatType)
        and mlir_type.width not in _LLVM_FLOAT_WIDTHS
    ):
        return ir.IntegerType.get_signless(mlir_type.width)
    return None


_legalized_int_type = storage_type


def _from_storage(
    value: ir.Value, result_type: ir.Type, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``value``, loaded as its storage integer, as a ``result_type``."""
    if isinstance(result_type, ir.FloatTF32Type):
        return builtin.unrealized_conversion_cast(
            [result_type], [value], loc=loc, ip=ip
        )
    return _arith.bitcast(value, result_type, loc=loc, ip=ip)


def _to_storage(
    value: ir.Value, storage: ir.Type, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``value`` as its storage integer ``storage``, for a store."""
    if isinstance(value.type, ir.FloatTF32Type):
        return builtin.unrealized_conversion_cast([storage], [value], loc=loc, ip=ip)
    return _arith.bitcast(value, storage, loc=loc, ip=ip)


def gep(
    base: ir.Value,
    elem_type: ir.Type,
    *,
    static_indices: Optional[list] = None,
    dynamic_indices: Optional[list] = None,
    loc: object = None,
    ip: object = None,
) -> ir.Value:
    """``llvm.getelementptr`` over elements of ``elem_type``, striding by its
    storage type when the dialect has none for it."""
    if static_indices is None:
        static_indices = []
    if dynamic_indices is None:
        dynamic_indices = []

    return llvm.getelementptr(
        base.type,
        base,
        dynamic_indices,
        static_indices,
        storage_type(elem_type) or elem_type,
        no_wrap_flags="None",
        loc=loc,
        ip=ip,
    )


def legalized_load(result_type: ir.Type, addr: ir.Value, **kwargs: Any) -> ir.Value:
    """``llvm.load`` of a scalar or vector ``result_type`` through ``addr``.

    A scalar the dialect cannot load (``storage_type``) is loaded as its
    storage integer and converted back; vector types pass through untouched.
    ``kwargs`` are ``llvm.load``'s (``alignment``, ``volatile_``, ...).
    """
    if not isinstance(result_type, ir.VectorType):
        storage = storage_type(result_type)
        if storage is not None:
            loaded = llvm.load(storage, addr, **kwargs)
            return _from_storage(
                loaded, result_type, loc=kwargs.get("loc"), ip=kwargs.get("ip")
            )
    return llvm.load(result_type, addr, **kwargs)


def legalized_store(value: ir.Value, addr: ir.Value, **kwargs: Any) -> Any:
    """``llvm.store`` of a scalar or vector ``value`` through ``addr``; returns
    the store op.

    A scalar the dialect cannot store (``storage_type``) is converted to its
    storage integer first; vector types pass through untouched. ``kwargs``
    are ``llvm.store``'s.
    """
    if not isinstance(value.type, ir.VectorType):
        storage = storage_type(value.type)
        if storage is not None:
            value = _to_storage(
                value, storage, loc=kwargs.get("loc"), ip=kwargs.get("ip")
            )
    return llvm.store(value, addr, **kwargs)


_gep = gep
_legalized_llvm_load = legalized_load
_legalized_llvm_store = legalized_store


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
        return gep(
            ptr,
            dtype.scalar_mlir_type,
            static_indices=[MLIR_DYNAMIC_INDEX],
            dynamic_indices=[index],
            loc=loc,
            ip=ip,
        )
    return gep(ptr, dtype.scalar_mlir_type, static_indices=[int(index)], loc=loc, ip=ip)


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
    return legalized_load(
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
) -> Any:
    """A scalar, vector or masked vector store through ``ptr``; returns the
    store op, for the caller's ``dsl_user_op`` wrapper to verify."""
    if mask is not None:
        return llvm.intr_masked_store(
            value, ptr, mask, alignment=alignment, loc=loc, ip=ip
        )
    return legalized_store(
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
