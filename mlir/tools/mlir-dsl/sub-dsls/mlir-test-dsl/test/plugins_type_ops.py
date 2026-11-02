# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
from dataclasses import replace
import mlir.mlir_dsl as m
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith as dsl_arith, llvm as dsl_llvm
# The plugin decides the width of an address per address space
# =============================================================================
# `Pointer.toint()` and the constant a Python int becomes in `inttoptr` take
# the dtype the type-ops plugin answers for the pointer's address space
# (`address_dtype`); the `OpEmitter` default answers nothing, which the types
# resolve to `Int64`. A plugin over a target whose addresses are 32 bits wide
# in one address space narrows that space alone.
class NarrowSpace3TypeOps(UpstreamDialectTypeOps):
    name = "narrow3"

    def address_dtype(self, address_space):
        return m.Int32 if address_space == 3 else None


class Space3DSL(m.MlirTestDSL):
    plugins = replace(
        m.MlirTestDSL.plugins,
        type_ops=NarrowSpace3TypeOps(
            scalars=dsl_arith, memory=dsl_llvm
        ),
    )


@Space3DSL.jit
def narrow_address(a: m.Int32) -> m.Int32:
    narrow = m.inttoptr(16, 3, m.Int32)  # a 32-bit address in space 3
    wide = m.inttoptr(16, 0, m.Int32)  # the default: 64 bits
    print("ADDRESS:", type(narrow.toint()).__name__, type(wide.toint()).__name__)
    return narrow.toint() + a


# CHECK:       ADDRESS: Int32 Int64
# CHECK-LABEL: func.func @narrow_address(
# CHECK:         %[[C:.+]] = arith.constant 16 : i32
# CHECK:         %[[N:.+]] = llvm.inttoptr %[[C]] : i32 to !llvm.ptr<3>
# CHECK:         %[[W:.+]] = arith.constant 16 : i64
# CHECK:         llvm.inttoptr %[[W]] : i64 to !llvm.ptr
# CHECK:         llvm.ptrtoint %[[N]] : !llvm.ptr<3> to i32
narrow_address(1)
