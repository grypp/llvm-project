# RUN: %PYTHON %s
# Shared rendering and catalog customization work without AST/compiler plugins.
import enum
import warnings
from mlir.dsl.core.common import DSLRuntimeError, DSLUserCodeError, report_warning
from mlir.dsl.core.diagnostics import DiagCatalog, DiagId, WarnId, classify, USAGE
from mlir.dsl.core.diagnostics import format_labeled_text, format_diagnostic_headline
class Own(DiagCatalog, enum.Enum):
    CALL_OUTSIDE_JIT = ("Own message for {api}.", ("Own fix.",))
original = DiagId.CALL_OUTSIDE_JIT.subcategory
classify(Own, USAGE, "own launch", "CALL_OUTSIDE_JIT")
assert Own.CALL_OUTSIDE_JIT.subcategory == "own launch"
assert DiagId.CALL_OUTSIDE_JIT.subcategory == original
error = DSLUserCodeError(Own.CALL_OUTSIDE_JIT, api="launch")
assert "Own message for launch" in str(error) and "Own fix" in str(error)
internal = str(DSLRuntimeError("backend failed", context={"phase": "lowering"}, suggestion=["Retry.", ""]))
assert "phase: lowering" in internal and "Retry." in internal
assert "SPILL" in format_diagnostic_headline("warning", "spilled", code="SPILL", namespace="be")
assert any("detail" in line for line in format_labeled_text("note", "detail"))
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always")
    report_warning(WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE, value=300, type="Int8", min=-128, max=127, wrapped=44, mask=255)
assert len(seen) == 1 and seen[0].filename == __file__
