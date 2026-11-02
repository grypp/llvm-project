# RUN: rm -rf %t.keep && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEP_IR=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_CACHE_DIR=%t.keep %PYTHON %s 2>&1 | FileCheck %s --check-prefix=KEEP
# RUN: rm -rf %t.after && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEPIR_AFTER_PASSES="builtin.module(convert-scf-to-cf)" MLIR_DSL_CACHE_DIR=%t.after %PYTHON %s 2>&1 | FileCheck %s --check-prefix=AFTER
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR_AFTER_PASSES=convert-scf-to-cf %PYTHON %s 2>&1 | FileCheck %s --check-prefix=PRINT
import os
import mlir.mlir_dsl as m

dsl = m.MlirTestDSL()
cache_dir = dsl.envar.cache_dir


# --- IR dumps ---------------------------------------------------
# `KEEP_IR` saves the traced module as `<CACHE_DIR>/<dsl>_<function>.mlir`
# before any pass (`DEBUGINFO` adds locations) and records the path;
# `KEEPIR_AFTER_PASSES` runs the given passes and saves
# `<dsl>_<function>_after_pass.mlir`; `PRINT_IR_AFTER_PASSES` prints the result
# of the passes on a clone to stderr. The dumps are written atomically, like
# the file cache, and named like its entries. All work under DRYRUN, which
# never reaches the compiler; without them nothing is written (the first RUN
# line).
@m.jit
def cumsum(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


print("CUMSUM:", cumsum(10))
# PRINT:     //===--- IR after passes: convert-scf-to-cf ---===
# PRINT:     func.func @cumsum(
# PRINT:     cf.cond_br
# PRINT-NOT: scf.for
# PRINT:     //===--- End of IR after passes ---===
# PRINT:     CUMSUM: ?
# EXEC:      CUMSUM: 45
listing = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
dumps = sorted(f for f in listing if f.startswith("mlir_dsl_cumsum"))
print("DUMPS:", dumps, os.path.basename(dsl.dump_mlir_path or "none"))
for name in dumps:
    text = open(os.path.join(cache_dir, name)).read()
    print(f"{name} scf.for: {'scf.for' in text} cf: {'cf.cond_br' in text}", end=" ")
    print(f"loc: {'loc(' in text} entry: {'func.func @cumsum(' in text}")
# CHECK:      DUMPS: [] none
# KEEP:       DUMPS: ['mlir_dsl_cumsum.mlir'] mlir_dsl_cumsum.mlir
# KEEP-NEXT:  mlir_dsl_cumsum.mlir scf.for: True cf: False loc: True entry: True
# AFTER:      DUMPS: ['mlir_dsl_cumsum_after_pass.mlir'] none
# AFTER-NEXT: mlir_dsl_cumsum_after_pass.mlir scf.for: False cf: True loc: False entry: True
