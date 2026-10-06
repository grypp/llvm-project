# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``scf`` dialect: builders, executors and :class:`ScfDialectPlugin`."""

from .builders import WhileLoopContext, for_, if_, while_, yield_
from .executors import LoopUnroll, all_, and_, any_, in_, not_, or_
from .plugin import ScfDialectPlugin

__all__ = [
    "LoopUnroll",
    "ScfDialectPlugin",
    "WhileLoopContext",
    "all_",
    "and_",
    "any_",
    "for_",
    "if_",
    "in_",
    "not_",
    "or_",
    "while_",
    "yield_",
]
