# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.mlir_dsl as m


@m.jit
def increment_to(n: m.Int32) -> m.Int32:
    i = m.Int32(0)
    while i < n:
        i += 1
    return i


increment_to(4)
# CHECK: func.func @increment_to
# CHECK: scf.while
# CHECK: scf.condition
# CHECK: arith.addi
# CHECK: scf.yield
# CHECK: return
