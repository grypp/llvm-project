# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The LLVM world's emitter: MLIR's builtin types with ``arith``, ``math``,
``vector`` and ``llvm`` ops, everything that lowers to the LLVM dialect."""

from __future__ import annotations

from typing import Any

from ..... import ir
from .....dialects import llvm, vector
from ....core.mlir_op import OpEmitter
from . import arith as _arith
from . import memory as _memory

__all__ = ["LlvmEmitter"]


class LlvmEmitter(OpEmitter):
    """Scalars as ``i32``/``f32``/..., vectors as ``vector<N x T>``, pointers as
    ``!llvm.ptr``; the ops of ``arith.py`` and the ``vector``/``llvm`` dialects."""

    # -- scalar ops ----------------------------------------------------------

    def const(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.const(*args, **kwargs)

    def add(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.add(*args, **kwargs)

    def sub(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.sub(*args, **kwargs)

    def mul(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.mul(*args, **kwargs)

    def truediv(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.truediv(*args, **kwargs)

    def floordiv(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.floordiv(*args, **kwargs)

    def mod(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.mod(*args, **kwargs)

    def pow(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.pow(*args, **kwargs)

    def neg(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.neg(*args, **kwargs)

    def abs(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.abs(*args, **kwargs)

    def and_(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.and_(*args, **kwargs)

    def or_(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.or_(*args, **kwargs)

    def xor(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.xor(*args, **kwargs)

    def shl(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.shl(*args, **kwargs)

    def shr(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.shr(*args, **kwargs)

    def cmp(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.cmp(*args, **kwargs)

    def select(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.select(*args, **kwargs)

    def cast(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.cast(*args, **kwargs)

    def bitcast(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.bitcast(*args, **kwargs)

    def cvtf(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.cvtf(*args, **kwargs)

    def fptoi(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.fptoi(*args, **kwargs)

    def itofp(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.itofp(*args, **kwargs)

    def int_to_int(self, *args: Any, **kwargs: Any) -> Any:
        return _arith.int_to_int(*args, **kwargs)

    def minmax(self, *args: Any, **kwargs: Any) -> Any:
        return _arith._minmax(*args, **kwargs)

    # -- vector ops ----------------------------------------------------------

    def from_elements(
        self, vec_type: ir.Type, elements: list, *, loc=None, ip=None
    ) -> Any:
        return vector.from_elements(vec_type, elements, loc=loc, ip=ip)

    def broadcast(
        self, vec_type: ir.Type, scalar: ir.Value, *, loc=None, ip=None
    ) -> Any:
        return vector.broadcast(vec_type, scalar, loc=loc, ip=ip)

    def extract(self, vec: ir.Value, lane: int, *, loc=None, ip=None) -> Any:
        return vector.extract(vec, [], [lane], loc=loc, ip=ip)

    def reduce(self, kind: str, vec: ir.Value, *, loc=None, ip=None) -> Any:
        elem = ir.VectorType(vec.type).element_type
        combining = getattr(vector.CombiningKind, kind.upper())
        return vector.reduction(elem, combining, vec, loc=loc, ip=ip)

    # -- pointer types and memory ops ----------------------------------------

    def pointer_type(self, dtype: Any, address_space: int) -> ir.Type:
        return _memory.pointer_type(dtype, address_space)

    def pointer_space(self, mlir_type: ir.Type) -> int | None:
        return _memory.pointer_space(mlir_type)

    def load(self, *args: Any, **kwargs: Any) -> Any:
        return _memory.load(*args, **kwargs)

    def store(self, *args: Any, **kwargs: Any) -> Any:
        return _memory.store(*args, **kwargs)

    def ptr_add(self, *args: Any, **kwargs: Any) -> Any:
        return _memory.ptr_add(*args, **kwargs)

    def inttoptr(
        self, value: ir.Value, address_space: int, *, loc=None, ip=None
    ) -> Any:
        return llvm.inttoptr(llvm.PointerType.get(address_space), value, loc=loc, ip=ip)

    def ptrtoint(self, ptr: ir.Value, int_type: ir.Type, *, loc=None, ip=None) -> Any:
        return llvm.ptrtoint(int_type, ptr, loc=loc, ip=ip)

    def addrspacecast(
        self, ptr: ir.Value, address_space: int, *, loc=None, ip=None
    ) -> Any:
        return llvm.addrspacecast(
            llvm.PointerType.get(address_space), ptr, loc=loc, ip=ip
        )
