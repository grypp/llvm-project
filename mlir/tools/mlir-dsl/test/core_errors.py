# RUN: env MLIR_DSL_WARNINGS_IGNORE=1 %PYTHON %s 2>&1 | FileCheck %s
# Error identity. A DSL decides the class of the errors the core raises for
# the author's mistakes (`user_error_class`), substitutes its own catalog ids
# for the core's (`diag_id_aliases`) and handles a `NameError` met while
# tracing (`on_trace_name_error`); a DSL that sets none of them raises today's
# `DSLUserCodeError` with the core's ids. `<PREFIX>_WARNINGS_IGNORE` silences
# the DSL's own warnings (`_warning_category`), not the process's.
import enum
import warnings

from mlir.dsl.core.common import DSLUserCodeError, DSLWarning, _user_error
from mlir.dsl.core.diagnostics import USAGE, DiagCatalog, DiagId, classify
from mlir.dsl.core.dsl import BaseDSL
from mlir.dsl.core.plugin import Plugins
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith
from mlir.dsl.types.typing import Int32


class StrictDiagId(DiagCatalog, enum.Enum):
    """A sub-DSL's own catalog, rendered as `error[strict:CODE]`."""

    namespace = enum.nonmember("strict")
    BAD_CALL = (
        "`{function_name}` was not called as declared: {detail}.",
        ("Call it with one value per parameter.",),
    )


classify(StrictDiagId, USAGE, "calls", "BAD_CALL")


class StrictError(DSLUserCodeError):
    """The class every core-raised error of `StrictErrors` arrives as."""


class StrictWarning(UserWarning):
    """The category `StrictErrors` silences under WARNINGS_IGNORE."""


class StrictErrors(BaseDSL):
    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith),
        decorators=[func.Jit()],
    )
    user_error_class = StrictError
    diag_id_aliases = {DiagId.CALL_ARGUMENTS: StrictDiagId.BAD_CALL}
    _warning_category = StrictWarning

    def __init__(self):
        super().__init__(name="MLIR_DSL")

    def on_trace_name_error(self, func, error, filename, lineno):
        raise StrictError(
            f"`{error.name}` is not defined in `{func.__name__}`",
            filename=filename,
            lineno=lineno,
            cause=error,
            suggestion="Define it before the call, or pass it in.",
        )


class Plain(BaseDSL):
    plugins = Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith),
        decorators=[func.Jit()],
    )

    def __init__(self):
        super().__init__(name="MLIR_DSL")


@StrictErrors.jit
def twice(a: Int32) -> Int32:
    return a + a


@StrictErrors.jit
def unbound(a: Int32) -> Int32:
    return a + never_set  # noqa: F821 - the trace-time NameError under test


@Plain.jit
def plain_twice(a: Int32) -> Int32:
    return a + a


@Plain.jit
def plain_unbound(a: Int32) -> Int32:
    return a + never_set  # noqa: F821


def headline(e):
    """The `error...` line of the rendering (its first line is blank)."""
    return str(e).splitlines()[1]


def report(label, thunk):
    try:
        thunk()
        print(label, "no error")
    except DSLUserCodeError as e:
        print(label, type(e).__name__, e.code, type(e) is DSLUserCodeError)
        print("  ", headline(e))


# A wrong call: the core's CALL_ARGUMENTS arrives as the DSL's class carrying
# the DSL's own id, the fields of the raise site filling the alias's template.
report("CALL:", lambda: twice(1, 2))
# CHECK:      CALL: StrictError BAD_CALL False
# CHECK-NEXT: {{.*}}error[strict:BAD_CALL]:{{.*}} `twice` was not called as declared: 2 positional and 0 keyword argument(s) do not bind

# The same at decoration time, through the class: no instance is active.
report("DECORATOR:", lambda: StrictErrors.jit(3))
# CHECK:      DECORATOR: StrictError BAD_CALL False
# CHECK-NEXT: {{.*}}error[strict:BAD_CALL]:{{.*}} `@jit` was not called as declared: it takes no option, but 1 positional value(s) were given.

# A NameError met while tracing is the DSL's to report.
report("NAME:", lambda: unbound(1))
# CHECK:      NAME: StrictError None False
# CHECK-NEXT: {{.*}}error:{{.*}} `never_set` is not defined in `unbound`

# A DSL that sets nothing: today's class and ids, exactly.
report("PLAIN CALL:", lambda: plain_twice(1, 2))
# CHECK:      PLAIN CALL: DSLUserCodeError CALL_ARGUMENTS True
# CHECK-NEXT: {{.*}}error[CALL_ARGUMENTS]:{{.*}} The call to `plain_twice` does not match its parameters
report("PLAIN NAME:", lambda: plain_unbound(1))
# CHECK:      PLAIN NAME: DSLUserCodeError SCOPE_UNBOUND_NAME_IN_TRACE True
# CHECK-NEXT: {{.*}}error[SCOPE_UNBOUND_NAME_IN_TRACE]:{{.*}} `never_set` is read here

# The factory the raise sites use: the active DSL's tables (none here), a
# given DSL's, and an id the DSL does not alias.
e = _user_error(DiagId.CALL_ARGUMENTS, function_name="f", detail="x")
print("FACTORY outside:", type(e).__name__, e.code)
e = _user_error(DiagId.CALL_ARGUMENTS, StrictErrors, function_name="f", detail="x")
print("FACTORY given:", type(e).__name__, e.code)
e = StrictErrors.user_error(DiagId.CALL_NOT_CALLABLE, decorator="@jit", got="a `str`")
print("FACTORY unaliased:", type(e).__name__, e.code)
# CHECK: FACTORY outside: DSLUserCodeError CALL_ARGUMENTS
# CHECK: FACTORY given: StrictError BAD_CALL
# CHECK: FACTORY unaliased: StrictError CALL_NOT_CALLABLE

# WARNINGS_IGNORE: the DSL's own warnings and the category a DSL names are
# silenced; a warning of another library is not. (Both DSLs were constructed
# by the calls above; the filters are installed then.)
StrictErrors(), Plain()
with warnings.catch_warnings(record=True) as caught:
    warnings.warn(DSLWarning("the DSL's own warning"))
    warnings.warn(StrictWarning("the category the DSL names"))
    warnings.warn("a warning of another library", UserWarning)
print("WARNINGS:", [type(w.message).__name__ for w in caught])
# CHECK: WARNINGS: ['UserWarning']
