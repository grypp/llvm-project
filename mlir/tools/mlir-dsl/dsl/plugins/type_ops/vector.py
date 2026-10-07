# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``vector`` dialect: the ops behind the core ``Vector``.

``from_elements``, ``broadcast``, ``extract`` and ``reduce`` over
``vector<N x T>`` values; the element-wise arithmetic of a ``Vector`` goes
through the scalar ops of ``arith`` on vector operands. An op module, not a
plugin: the ``TypeOps`` composer routes its vector hooks here."""

from typing import Any

from .... import ir
from ....dialects import vector

__all__ = ["broadcast", "extract", "from_elements", "reduce"]


def from_elements(
    vec_type: ir.Type, elements: list, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``vector.from_elements``: a ``vec_type`` value from its scalar ``elements``."""
    return vector.from_elements(vec_type, elements, loc=loc, ip=ip)


def broadcast(
    vec_type: ir.Type, scalar: ir.Value, *, loc: Any = None, ip: Any = None
) -> ir.Value:
    """``vector.broadcast``: ``scalar`` splat into a ``vec_type`` value."""
    return vector.broadcast(vec_type, scalar, loc=loc, ip=ip)


def extract(vec: ir.Value, lane: int, *, loc: Any = None, ip: Any = None) -> ir.Value:
    """``vector.extract``: the scalar at the static position ``lane``."""
    return vector.extract(vec, [], [lane], loc=loc, ip=ip)


def reduce(kind: str, vec: ir.Value, *, loc: Any = None, ip: Any = None) -> ir.Value:
    """``vector.reduction`` of ``vec`` with the combining kind ``kind``
    (``"add"``, ``"mul"``, ``"minsi"``, ... as ``vector.CombiningKind`` names)."""
    elem = ir.VectorType(vec.type).element_type
    combining = getattr(vector.CombiningKind, kind.upper())
    return vector.reduction(elem, combining, vec, loc=loc, ip=ip)
