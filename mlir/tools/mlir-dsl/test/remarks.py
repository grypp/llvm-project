# RUN: env MLIR_DSL_REMARKS=".*" %PYTHON %s 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_REMARKS=".*" MLIR_DSL_REMARKS_POLICY=final %PYTHON %s 2>&1 | FileCheck %s
# RUN: env MLIR_DSL_REMARKS="other" %PYTHON %s 2>&1 | FileCheck %s --check-prefix=OFF
# RUN: %PYTHON %s 2>&1 | FileCheck %s --check-prefix=OFF
# RUN: rm -f %t.yaml && env MLIR_DSL_REMARKS=".*" MLIR_DSL_REMARKS_OUTPUT=%t.yaml %PYTHON %s 2>&1 | FileCheck %s --check-prefix=STREAMED
# RUN: rm -f %t.bitstream && env MLIR_DSL_REMARKS=".*" MLIR_DSL_REMARKS_OUTPUT=%t.bitstream %PYTHON %s 2>&1 | FileCheck %s --check-prefix=BITSTREAM
# RUN: env MLIR_DSL_REMARKS=".*" MLIR_DSL_REMARKS_POLICY=sometimes %PYTHON %s 2>&1 | FileCheck %s --check-prefix=BADPOLICY
# REQUIRES: host-supports-jit
# The DSL's wiring of the upstream RemarkEngine, not the engine's
# own filter/policy/format semantics (mlir/test/python/ir/remarks.py): the
# `<PREFIX>_REMARKS` filter, `_REMARKS_POLICY` and `_REMARKS_OUTPUT` settings
# reach the per-compile session; without an output path the remarks are
# collected as records in `dsl.collected_remarks` (the last compile's) and
# rendered to stderr in the DSL's diagnostic style; with one they stream to
# the file whose extension picks the format and are neither rendered nor
# collected; `policy=final` remarks are flushed while the session still owns
# the engine; a bad policy is a DSLRuntimeError. The session spans the whole
# compile, so a remark emitted at trace time (here from user code, where a
# sub-DSL would report a decision) lands in the same stream as the passes'.
import os

import mlir.mlir_dsl as m
from mlir import ir

dsl = m.MlirTestDSL()
OUTPUT = os.environ.get("MLIR_DSL_REMARKS_OUTPUT", "")


@m.jit
def twice(n: m.Int32) -> m.Int32:
    ir.Location.current.emit_remark(
        ir.RemarkKind.ANALYSIS,
        "TraceDecision",
        category="mlir.dsl",
        function_name="twice",
        message="n is staged",
        args=[("staged", "n")],
    )
    return n * 2


@m.jit
def thrice(n: m.Int32) -> m.Int32:
    return n * 3


def collected():
    """The user-facing fields of every record (each also carries the engine's
    `RemarkId` argument)."""
    fields = ("kind", "name", "category", "function")
    return [
        tuple(r[f] for f in fields) + (r["args"]["Remark"], r["args"]["staged"])
        for r in dsl.collected_remarks
    ]


# The remark is rendered as it arrives, before the result; the record keeps
# the kind, name, category, function and the user arguments.
try:
    print("RESULT:", twice(21))
except m.DSLRuntimeError as e:
    print("CONFIG ERROR:", e.message, "|", e.cause)
    raise SystemExit(0)
print("COLLECTED:", collected())
# CHECK:      remark[analysis]:{{.*}}TraceDecision | Category:mlir.dsl | Function=twice | Remark="n is staged", RemarkId={{[0-9]+}}, staged=n
# CHECK:      RESULT: 42
# CHECK-NEXT: COLLECTED: [('analysis', 'TraceDecision', 'mlir.dsl', 'twice', 'n is staged', 'n')]

# A filter that matches nothing, or no filter at all (the default): nothing
# is rendered or collected.
# OFF-NOT:  remark[
# OFF:      RESULT: 42
# OFF-NEXT: COLLECTED: []

# With an output path the remarks stream to the file (`.yaml` / `.bitstream`
# by extension) and are neither rendered nor collected.
# STREAMED-NOT:  remark[
# STREAMED:      RESULT: 42
# STREAMED-NEXT: COLLECTED: []
# BITSTREAM-NOT: remark[
# BITSTREAM:     RESULT: 42
if OUTPUT.endswith(".yaml"):
    with open(OUTPUT, encoding="utf-8") as f:
        print("YAML:", " | ".join(line.strip() for line in f if line.strip()))
elif OUTPUT:
    with open(OUTPUT, "rb") as f:
        print("MAGIC:", f.read(4), os.path.getsize(OUTPUT) > 4)
# STREAMED:      YAML: --- !Analysis | {{.*}}Name: {{ *}}TraceDecision{{.*}}Function: {{ *}}twice
# BITSTREAM:     MAGIC: b'RMRK' True

# `collected_remarks` holds the records of the last compile only.
print("RESULT:", thrice(1), "| COLLECTED:", collected())
# CHECK:      RESULT: 3 | COLLECTED: []
# OFF:        RESULT: 3 | COLLECTED: []

# A policy the DSL does not know is reported as a DSL configuration error
# before the engine is touched.
# BADPOLICY-NOT: RESULT
# BADPOLICY:     CONFIG ERROR: invalid remark configuration: unknown remark policy 'sometimes'; expected 'all' or 'final' | None
