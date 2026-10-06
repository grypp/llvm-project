# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``scf`` dialect plugin: control flow regions and their lowering."""

from __future__ import annotations

from ....core.plugin import DialectPlugin

__all__ = ["ScfDialectPlugin"]


class ScfDialectPlugin(DialectPlugin):
    """``scf.for``/``scf.if``/``scf.while`` regions: the explicit builders
    (``for_``, ``if_``, ``while_``, ``yield_``), the executors the ``scf`` AST
    preprocessor plugin drives, and the lowering of ``scf`` and ``cf`` to the
    LLVM dialect. ``BaseDSL`` installs it by default, ahead of the LLVM world,
    because the ``scf`` lowering creates ``arith`` ops for induction variables."""

    name = "scf"

    def pipeline_passes(self) -> list[str]:
        return ["convert-scf-to-cf", "convert-cf-to-llvm"]
