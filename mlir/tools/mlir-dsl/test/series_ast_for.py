# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.mlir_dsl as m


@m.jit
def sum_to(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    # An ordinary Python iterable still runs during tracing.
    for offset in (1, 2):
        acc += offset
    return acc


sum_to(5)
# CHECK: func.func @sum_to
# CHECK: scf.for
# CHECK: arith.addi
# CHECK: scf.yield
# CHECK: return
