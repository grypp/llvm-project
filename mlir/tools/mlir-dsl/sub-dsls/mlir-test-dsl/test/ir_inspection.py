# RUN: rm -rf %t.one && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEP_IR=1 MLIR_DSL_CACHE_DIR=%t.one %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ONE
# RUN: rm -rf %t.matrix && env MLIR_DSL_DRYRUN=1 MLIR_DSL_KEEP_IR=raw,clean,after-canonicalize MLIR_DSL_CACHE_DIR=%t.matrix %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,MATRIX
# The IR dumps of `build_module` and its verification hook. `KEEP_IR=1` saves
# the raw module (before any pass) as `<CACHE_DIR>/<dsl>_<function>.mlir`, as
# it always did; a comma-separated list of stages saves one file per stage,
# the raw module under the same name and every other stage on a clone, the
# module itself untouched: `clean` (canonicalize and cse) as `<function>_clean`,
# `after-<pass>` as `<function>_after_<pass>`. The paths are recorded by stage
# in `dump_mlir_paths` (`dump_mlir_path`/`dump_mlir_raw_path` is the raw one).
# An unknown stage is a `CONFIG_INVALID` diagnostic naming the variable. Last
# in `build_module`, `_verify_module` runs the verifier; a DSL with an
# invariant of its own on the traced module checks it there.
import os

import mlir.mlir_dsl as m
from mlir import ir

dsl = m.MlirTestDSL()
cache_dir = dsl.envar.cache_dir


def report(label, fn, *args, **kwargs):
    """Print a call's result, or the DiagId of the DSL error it raises."""
    try:
        result = fn(*args, **kwargs)
    except m.DSLUserCodeError as e:
        print(label, "ERROR:", e.diag_id.name)
        return e
    print(label, "RESULT:", result)
    return result


# --- the dump matrix ---------------------------------------------------------
# A loop, a duplicated product (cse folds it, canonicalize alone keeps both)
# and two literal additions (canonicalize folds them into one constant).
@m.jit
def shape(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    x = n * 3
    y = n * 3
    return acc + x + y + 1 + 1


print("SHAPE:", shape(4))
listing = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
dumps = sorted(f for f in listing if f.startswith("mlir_dsl_shape"))
print(
    "DUMPS:",
    dumps,
    os.path.basename(dsl.dump_mlir_path),
    os.path.basename(dsl.dump_mlir_raw_path),
    list(dsl.dump_mlir_paths),
)
for stage, path in dsl.dump_mlir_paths.items():
    print(f"--- {stage}: {os.path.basename(path)}")
    print(open(path).read())
print("--- end of dumps")
# CHECK:         SHAPE: ?
# ONE:           DUMPS: ['mlir_dsl_shape.mlir'] mlir_dsl_shape.mlir mlir_dsl_shape.mlir ['raw']
# ONE-LABEL:     --- raw: mlir_dsl_shape.mlir
# ONE:             func.func @shape(
# ONE:             scf.for
# ONE-LABEL:     --- end of dumps
# MATRIX:        DUMPS: ['mlir_dsl_shape.mlir', 'mlir_dsl_shape_after_canonicalize.mlir', 'mlir_dsl_shape_clean.mlir'] mlir_dsl_shape.mlir mlir_dsl_shape.mlir ['raw', 'clean', 'after-canonicalize']
# MATRIX-LABEL:  --- raw: mlir_dsl_shape.mlir
# MATRIX:          func.func @shape(
# MATRIX:          scf.for
# MATRIX:          arith.muli
# MATRIX:          arith.muli
# MATRIX:          arith.constant 1 : i32
# MATRIX:          arith.addi
# MATRIX:          arith.constant 1 : i32
# MATRIX:          arith.addi
# MATRIX-NOT:      arith.constant 2 : i32
# MATRIX-LABEL:  --- clean: mlir_dsl_shape_clean.mlir
# MATRIX:          %[[C2:.+]] = arith.constant 2 : i32
# MATRIX:          %[[M:.+]] = arith.muli
# MATRIX-NOT:      arith.muli
# MATRIX:          arith.addi %{{.+}}, %[[M]] : i32
# MATRIX:          arith.addi %{{.+}}, %[[M]] : i32
# MATRIX:          arith.addi %{{.+}}, %[[C2]] : i32
# MATRIX-LABEL:  --- after-canonicalize: mlir_dsl_shape_after_canonicalize.mlir
# MATRIX:          %[[C2:.+]] = arith.constant 2 : i32
# MATRIX:          arith.muli
# MATRIX:          arith.muli
# MATRIX:          arith.addi %{{.+}}, %[[C2]] : i32
# MATRIX-LABEL:  --- end of dumps

# An unknown stage names the variable; the module is traced again on the
# next call, so the setting is restored afterwards.
saved = dsl.envar.keep_ir
dsl.envar.keep_ir = "raw,bogus"
print(str(report("BOGUS", shape, 2)))
dsl.envar.keep_ir = saved
# CHECK:      BOGUS ERROR: CONFIG_INVALID
# CHECK:      error[CONFIG_INVALID]:{{.*}}`MLIR_DSL_KEEP_IR` has an invalid setting: `bogus` is not a stage; expected `1`, or a comma-separated list of `raw`, `clean`, `after-<pass>` and `after-passes`.


# --- the verification hook ---------------------------------------------------
# `_verify_module` runs last in `build_module`, after the dumps and the trace
# hooks added to the module; this DSL refuses a module carrying `test.marker`
# on top of the verifier's own checks.
class StrictVerifyDSL(m.MlirTestDSL):
    def _verify_module(self, module):
        super()._verify_module(module)
        if "test.marker" in module.operation.attributes:
            raise m.DSLRuntimeError("the module carries a `test.marker` attribute")


def mark(dsl, module, function_name):
    module.operation.attributes["test.marker"] = ir.UnitAttr.get()


@StrictVerifyDSL.jit
def verified(n: m.Int32) -> m.Int32:
    return n + 1


print("VERIFIED:", verified(1))
try:
    verified(1, trace_hooks=[mark])
    print("MARKED: accepted (unexpected)")
except m.DSLRuntimeError as e:
    print("MARKED:", e.message)
# CHECK:      VERIFIED: ?
# CHECK-NEXT: MARKED: the module carries a `test.marker` attribute
