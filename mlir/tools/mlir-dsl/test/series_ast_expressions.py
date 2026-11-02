# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.mlir_dsl as m


@m.jit
def select_inside(x: m.Int32) -> m.Int32:
    inside = 0 < x < 10
    valid = inside and not (x == 5)
    return x if valid else m.Int32(0)


select_inside(4)
# CHECK: func.func @select_inside
# CHECK: arith.cmpi
# CHECK: arith.cmpi
# CHECK: scf.if
# CHECK: scf.yield
# CHECK: return
