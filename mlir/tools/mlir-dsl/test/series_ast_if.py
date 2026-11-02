# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.mlir_dsl as m


@m.jit
def positive(x: m.Int32) -> m.Int32:
    result = m.Int32(0)
    if x > 0:
        result = x
    else:
        result = -x
    return result


positive(-3)
# CHECK: func.func @positive
# CHECK: scf.if
# CHECK: scf.yield
# CHECK: else
# CHECK: scf.yield
# CHECK: return
