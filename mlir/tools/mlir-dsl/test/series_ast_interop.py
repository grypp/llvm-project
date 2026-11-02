# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A direct dialect builder receives raw operands after preprocessing.

The factory simultaneously checks captured values and a default argument whose
source-level name is local to the factory. No staged control flow is needed.
"""
import mlir.mlir_dsl as m
from mlir.dialects import arith


def make_scaled(factor):
    offset = 3

    @m.jit
    def scaled(x: m.Int32, scale=factor) -> m.Int32:
        # `arith.addi` expects ir.Values; the rewrite unwraps both x operands.
        doubled = m.Int32(arith.addi(x, x))
        return doubled * scale + offset

    return scaled


scaled = make_scaled(4)
scaled(5)
# CHECK: func.func @scaled
# CHECK: arith.addi
# CHECK: arith.muli
# CHECK: arith.addi
# CHECK: return
print("direct-builder operands and captured defaults: passed")
# CHECK: direct-builder operands and captured defaults: passed
