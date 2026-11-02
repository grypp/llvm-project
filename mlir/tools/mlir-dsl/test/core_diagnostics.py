# RUN: %PYTHON %s 2>&1 | FileCheck %s
# S1: the exception classes, the diagnostic catalogs, the warning channel and
# the active-DSL context. Pure Python: nothing here needs the MLIR bindings.
import os
import sys
import warnings
from types import SimpleNamespace

from mlir.dsl.core.common import (
    DSLBaseError,
    DSLRuntimeError,
    DSLUserCodeError,
    DSLWarning,
    active_dsl,
    get_current_dsl,
    get_current_env_manager,
    in_trace,
    report_warning,
)
from mlir.dsl.core.diagnostics import (
    META_VALUE,
    STAGED_VALUE,
    DiagCatalog,
    DiagId,
    WarnId,
    find_user_source_location,
)

# --- the catalogs -----------------------------------------------------------
print("CATALOG:", len(DiagId) >= 16, len(WarnId) >= 4)  # the seed's members, at least
print(
    "CATALOG MIXIN:",
    isinstance(DiagId.CALL_OUTSIDE_JIT, DiagCatalog),
    DiagId.namespace == "",
)
print(
    "CODE:",
    DiagId.TYPE_INT_POW_UNSUPPORTED.code,
    DiagId.TYPE_INT_POW_UNSUPPORTED.prefix,
)
print(
    "PREFIXES:",
    {"ARG", "CALL", "CONFIG", "CONTAINER", "PHASE", "SCOPE", "TYPE"}
    <= {d.prefix for d in DiagId},
)
print("WORDS:", META_VALUE, "|", STAGED_VALUE)
message, fixes = DiagId.PHASE_DYNAMIC_TO_STATIC_BOOL.fill()
print("INJECTED:", message.split(" is used")[0], "|", len(fixes), "fixes")
message, fixes = DiagId.ARG_ANNOTATION_MISMATCH.fill(num=1)
print("MISSING FIELD:", message)
# CHECK: CATALOG: True True
# CHECK: CATALOG MIXIN: True True
# CHECK: CODE: TYPE_INT_POW_UNSUPPORTED TYPE
# CHECK: PREFIXES: True
# CHECK: WORDS: Python value | runtime value
# CHECK: INJECTED: A runtime value | 2 fixes
# CHECK: MISSING FIELD: Argument #1 `{arg_name}` must be {expected}, but got {got}.

# --- a user error from a catalog member, rendered -----------------------------
e = DSLUserCodeError(DiagId.CALL_OUTSIDE_JIT, api="Int32.mlir_type", decorator="@jit")
print(
    "USER:",
    type(e).__name__,
    e.code,
    e.diag_id is DiagId.CALL_OUTSIDE_JIT,
    e._is_internal,
)
print("USER IS BASE:", isinstance(e, DSLBaseError), isinstance(e, Exception))
print("USER LOCATION:", os.path.basename(e.filename), e.line, e.col is not None)
print(str(e))
# CHECK: USER: DSLUserCodeError CALL_OUTSIDE_JIT True False
# CHECK: USER IS BASE: True True
# CHECK: USER LOCATION: core_diagnostics.py {{[0-9]+}} True
# CHECK: error[CALL_OUTSIDE_JIT]:{{.*}} `Int32.mlir_type` was called from plain Python,
# CHECK: -->{{.*}}core_diagnostics.py:{{[0-9]+}}
# CHECK: suggestion:{{.*}} Move this call into a

# --- a free-form user error with an explicit location, context and fixes ------
e = DSLUserCodeError(
    "Scope Error: `a` escapes its scope",
    filename="user.py",
    lineno=42,
    suggestion=["Define it before the loop.", "Or return it."],
    context={"var": "a", "kind": "loop"},
)
print("FREE:", e.code, e.diag_id, e.filename, e.line)
print(str(e))
# CHECK: FREE: None None user.py 42
# CHECK: error:{{.*}} Scope Error: `a` escapes its scope
# CHECK: -->{{.*}}user.py:42
# CHECK: note:{{.*}} var: a
# CHECK: note:{{.*}} kind: loop
# CHECK: suggestion:{{.*}} Define it before the loop.
# CHECK: suggestion:{{.*}} Or return it.

# --- template fields on a plain string are the DSL author's mistake ---------
try:
    DSLUserCodeError("plain text", var="x")
    print("FIELDS ON STRING: no error")
except DSLRuntimeError as err:
    print(
        "FIELDS ON STRING:",
        type(err).__name__,
        "fields ['var']" if "['var']" in err.message else err.message,
    )
# CHECK: FIELDS ON STRING: DSLRuntimeError fields ['var']

# --- an internal error: the INTERNAL envelope, cause and free-text context ---
e = DSLRuntimeError(
    "the pipeline produced no module",
    context="after lowering",
    cause=ValueError("inner"),
)
print(
    "INTERNAL:",
    e._is_internal,
    e.code if hasattr(e, "code") else "<no code attr>",
    e.line,
)
print(str(e))
# CHECK: INTERNAL: True <no code attr> None
# CHECK: error[INTERNAL]:
# CHECK: the pipeline produced no module
# CHECK: Caused exception: inner

# --- a warning through the standard warnings module -------------------------
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    report_warning(
        WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE,
        value=2**40,
        type="Int32",
        min=-(2**31),
        max=2**31 - 1,
        wrapped=0,
        mask=0xFFFFFFFF,
    )
w = caught[0].message
print(
    "WARNING:",
    len(caught),
    type(w).__name__,
    caught[0].category is DSLWarning,
    issubclass(DSLWarning, UserWarning),
)
print(
    "WARNING ID:",
    w.code,
    w.warn_id is WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE,
    os.path.basename(w.filename),
)
print(str(w))
# CHECK: WARNING: 1 DSLWarning True True
# CHECK: WARNING ID: TYPE_INT_LITERAL_OUT_OF_RANGE True core_diagnostics.py
# CHECK: warning[TYPE_INT_LITERAL_OUT_OF_RANGE]:{{.*}} The Python integer 1099511627776 does not
# CHECK: -->{{.*}}core_diagnostics.py:{{[0-9]+}}
# CHECK: suggestion:{{.*}} Use a wider integer type
# CHECK: suggestion:{{.*}} If you meant the wrap-around

# A free-form warning, and fields on a plain string.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warnings.warn(DSLWarning("just saying", filename="user.py", lineno=7))
print(
    "FREE WARNING:", caught[0].message.code, str(caught[0].message).replace("\n", " / ")
)
try:
    DSLWarning("plain", x=1)
except DSLRuntimeError as err:
    print("WARNING FIELDS ON STRING:", type(err).__name__)
# CHECK: FREE WARNING: None{{.*}}warning:{{.*}}just saying{{.*}}user.py:7
# CHECK: WARNING FIELDS ON STRING: DSLRuntimeError

# --- the active-DSL context ---------------------------------------------------
print("OUTSIDE:", get_current_dsl(), in_trace(), get_current_env_manager())
dsl = SimpleNamespace(name="S1", envar=SimpleNamespace(prefix="S1"))
with active_dsl(dsl):
    print(
        "INSIDE:",
        get_current_dsl() is dsl,
        in_trace(),
        get_current_env_manager() is dsl.envar,
    )
    inner = SimpleNamespace(name="inner", envar=None)
    with active_dsl(inner):
        print("NESTED:", get_current_dsl().name)
    print("RESTORED:", get_current_dsl().name)
print("AFTER:", get_current_dsl(), in_trace())
try:
    with active_dsl(dsl):
        raise DSLRuntimeError("escaping")
except DSLRuntimeError as err:
    print("ESCAPED:", err._dsl_env_manager is dsl.envar, get_current_dsl())
# CHECK: OUTSIDE: None False None
# CHECK: INSIDE: True True True
# CHECK: NESTED: inner
# CHECK: RESTORED: S1
# CHECK: AFTER: None False
# CHECK: ESCAPED: True None

# --- the user-frame finder skips DSL modules by module name ------------------
fn, line, col, end_col = find_user_source_location()
print(
    "LOCATION HERE:",
    os.path.basename(fn),
    line,
    col is not None and end_col is not None,
)
# A function whose module is named inside the DSL package is skipped: the
# location reported is this file's call, not the fake module's line.
fake_src = "def inner():\n    return find_user_source_location()\n"
fake_globals = {
    "__name__": "mlir.dsl.fake",
    "find_user_source_location": find_user_source_location,
}
exec(compile(fake_src, "/nowhere/fake_dsl_module.py", "exec"), fake_globals)


def call_through_dsl_frame():
    # Both halves evaluate on this one line: the location the finder reports
    # must be this frame's line, not the fake module's.
    return fake_globals["inner"](), sys._getframe().f_lineno


(fn2, line2, _, _), call_line = call_through_dsl_frame()
print("LOCATION THROUGH DSL FRAME:", os.path.basename(fn2), line2 == call_line)
# CHECK: LOCATION HERE: core_diagnostics.py {{[0-9]+}} True
# CHECK: LOCATION THROUGH DSL FRAME: core_diagnostics.py True

print("S1_DIAG", "OK")
# CHECK: S1_DIAG OK
