# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s 2>&1 | FileCheck %s
# Diagnostics, the DSL-owned half only: the DiagId/WarnId catalogue
# invariants and `fill`; DSLUserCodeError attributes, locations and rendering;
# the author-frame finder and sub-DSL registration; the code frame renderer;
# one rendered error per catalogue category from a real trigger; the
# DSLRuntimeError internal envelope; report_warning / DSLWarning. Each program
# catches its error and prints the rendering, so the pipeline exits 0 under
# lit's pipefail. ANSI colours are matched with {{.*}} or stripped.
import importlib.util
import inspect
import os
import re
import sys
import tempfile
import warnings

import mlir.mlir_dsl as m
from mlir.dsl.plugins.compiler.execution_engine import OptLevel
from mlir.dsl.core.common import (
    DSLRuntimeError,
    DSLUserCodeError,
    DSLUserCodeRuntimeError,
    DSLUserCodeTypeError,
    DSLWarning,
    active_dsl,
    report_warning,
)
from mlir.dsl.core.diagnostics import (
    META_VALUE,
    STAGED_VALUE,
    DiagId,
    WarnId,
    _CATEGORIES,
    _is_dsl_module,
    register_dsl_package,
    render_code_frame,
    render_user_diagnostic,
)
from mlir.dsl.plugins.decorators.kernels.gpu_plugin import GpuDiagId, check_arch
from mlir.dsl.plugins.adapters.tvm_ffi.diagnostics import TvmFfiDiagId

HERE = os.path.abspath(__file__)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TMP = tempfile.mkdtemp()
CATALOGS = (DiagId, WarnId, GpuDiagId, TvmFfiDiagId)
with open(HERE, encoding="utf-8") as f:
    SRC = f.read().splitlines()


def report(fn, *args):
    """Call `fn`; print the rendered DSLUserCodeError, or NO ERROR."""
    try:
        fn(*args)
    except m.DSLUserCodeError as e:
        print(str(e))
    else:
        print("NO ERROR from", getattr(fn, "__name__", fn))


def headline(exc):
    """The first non-blank line of a rendering, colours stripped."""
    return next(ln for ln in ANSI.sub("", str(exc)).splitlines() if ln.strip())


def show(label, frame):
    """Print a code frame line by line with repr, colours stripped."""
    if frame is None:
        print(label, "-> None")
        return
    lines = ANSI.sub("", frame).splitlines()
    print(label, "->", len(lines), "lines")
    for line in lines:
        print(label, repr(line))


def span(err):
    """The source text an error's column span covers, when it is in this file."""
    if err.filename != HERE or err.col is None:
        return None
    return SRC[err.line - 1][err.col : err.end_col]


def at_mark(diag, **kw):
    """A DSLUserCodeError with an explicit location: this file, MARK_LINE."""
    return DSLUserCodeError(diag, filename=HERE, lineno=MARK_LINE, **kw)


def load_module(name, body):
    """Import `body` as module `name` from a file under TMP."""
    path = os.path.join(TMP, name.replace(".", "_") + ".py")
    with open(path, "w", encoding="utf-8") as out:
        out.write(body)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


MARK_LINE = inspect.currentframe().f_lineno  # explicit-location errors point here
print("mark line:", MARK_LINE)
# CHECK: mark line: [[#MARK:]]


# --- The catalogue: shape, prefixes, namespaces, classification, fill -------
def well_formed(d):
    return len(d.value) == 2 and d.message.strip() and all(x.strip() for x in d.fix)


print("malformed:", [d.name for cat in CATALOGS for d in cat if not well_formed(d)])
# CHECK: malformed: []
for cat in CATALOGS:
    prefixes = " ".join(sorted({d.prefix for d in cat}))
    categories = " / ".join(sorted({d.category for d in cat}))
    print(cat.__name__, repr(cat.namespace), "|", prefixes, "|", categories)
# CHECK: DiagId '' | ARG CALL CONFIG CONTAINER PHASE POINTER SCOPE STRUCT TYPE UNSUP | not zero-cost / unsupported / usage
# CHECK: WarnId '' | TYPE | warning
# CHECK: GpuDiagId 'gpu' | CONFIG LAUNCH | usage
# CHECK: TvmFfiDiagId 'tvm_ffi' | CALL UNSUP | unsupported / usage
all_codes = {(cat.namespace, d.name) for cat in CATALOGS for d in cat}
print("unclassified:", sorted(all_codes ^ set(_CATEGORIES)))
# CHECK: unclassified: []
d = DiagId.TYPE_UNSTABLE_JOIN
print("accessors:", d.code, d.prefix, d.category, "|", d.subcategory, "|", len(d.fix))
# CHECK: accessors: TYPE_UNSTABLE_JOIN TYPE not zero-cost | a type or structure differs between paths of a runtime for/while/if | 3

# `fill`: fields substituted, the fixes returned as a tuple; `{meta}`/`{staged}`
# are injected from the module constants; a forgotten field renders as its
# placeholder, never a KeyError; every template fills with no fields at all.
MUT, REQ = DiagId.PHASE_MUTATE_PYTHON, DiagId.PHASE_REQUIRES_CONSTANT
msg, fixes = MUT.fill(var="acc")
print("fill:", msg)
# CHECK: fill: `acc` is a Python value, but it is changed inside a for/while/if controlled by a runtime value. Only a runtime value can change there
print("fixes:", type(fixes).__name__, len(fixes), "|", fixes[0])
# CHECK: fixes: tuple 2 | Create `acc` as a runtime value of the matching type before the for/while/if, e.g. `acc = Int32(0)`
print("constants:", META_VALUE, "|", STAGED_VALUE, "|", REQ.fill(what="`c`")[0])
# CHECK: constants: Python value | runtime value | `c` requires a value known at compile time, but it received a runtime value.
print("missing:", d.fill(var="x")[0])
# CHECK: missing: `x` has type `{old_type}` on one path and `{new_type}` on another.


def fills(d):
    try:
        d.fill()
    except Exception:  # noqa: BLE001 -- the test reports, not raises
        return False
    return True


print("fill() failures:", [d.name for cat in CATALOGS for d in cat if not fills(d)])
# CHECK: fill() failures: []

# --- DSLUserCodeError: attributes, locations, rendering ---------------------
# Catalogue form with an explicit location: the fields fill the templates, the
# catalogue fixes become `suggestion`, the stable code is recorded.
e = at_mark(d, var="count", old_type="Int32", new_type="Float32")
print("diag_id:", e.diag_id.name, "| code:", e.code, "| member:", e.diag_id is d)
# CHECK: diag_id: TYPE_UNSTABLE_JOIN | code: TYPE_UNSTABLE_JOIN | member: True
print("message:", e.message)
# CHECK: message: `count` has type `Int32` on one path and `Float32` on another. `count` must have one type wherever these paths come back together.
print("suggestion:", len(e.suggestion), "|", e.suggestion[1])
# CHECK: suggestion: 3 | If a conversion is needed, convert explicitly in each branch, e.g. `count = Float32(count)`
print("location:", e.filename == HERE, e.line == MARK_LINE, e.col, e.end_col)
# CHECK: location: True True None None
print("str is the rendering:", str(e) == render_user_diagnostic(e))
# CHECK: str is the rendering: True
print(str(e))
# CHECK:      error[TYPE_UNSTABLE_JOIN]:{{.*}} `count` has type `Int32` on one path
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#MARK]]
# CHECK:      {{[0-9]+}} | MARK_LINE = inspect.currentframe().f_lineno
# CHECK-NEXT: |{{.*}}^
# CHECK:      = category: not zero-cost (a type or structure differs between paths of a runtime for/while/if)
# CHECK:      suggestion:{{.*}}Make every assignment to `count` produce the same type.
# CHECK:      suggestion:{{.*}}If a conversion is needed
# CHECK:      suggestion:{{.*}}If one path sets `count` to `None`

# An explicit 0-based column span is shown 1-based and underlined.
print(str(at_mark(DiagId.ARG_INVALID_ALIGNMENT, col_offset=12, end_col_offset=24)))
# CHECK:      error[ARG_INVALID_ALIGNMENT]
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#MARK]]:13
# CHECK:      |{{.*}}^^^^^^^^^^^^{{[^^]*$}}

# `cause` and a dict `context` become `= note:` lines after the category.
ctx = {"region": "for", "depth": 2}
print(str(at_mark(MUT, var="acc", context=ctx, cause=ValueError("boom"))))
# CHECK:      = category: not zero-cost (a Python value changes inside a runtime for/while/if)
# CHECK-NEXT: = note:{{.*}}Caused exception: boom
# CHECK-NEXT: = note:{{.*}}region: for
# CHECK-NEXT: = note:{{.*}}depth: 2
# CHECK-NEXT: suggestion:{{.*}}Create `acc` as a runtime value

# Free-form string form: no code, no fixes, no category line. Template fields
# with a string are a programming error, reported as an internal error.
e = at_mark("Custom message here")
print("plain:", e.diag_id, e.code, e.suggestion, "category" in str(e), "|", headline(e))
# CHECK: plain: None None None False | error: Custom message here
try:
    at_mark("Plain", var="x")
except DSLRuntimeError as err:
    print("misuse:", err.message)
# CHECK: misuse: DSLUserCodeError received template fields ['var'] but the first argument is a plain string, not a DiagId.

# The subclasses user code may already catch as RuntimeError / TypeError.
RT, TE = DSLUserCodeRuntimeError, DSLUserCodeTypeError
BASES = [(RT, RuntimeError), (TE, TypeError), (RT, DSLUserCodeError)]
print("subclasses:", [issubclass(c, b) for c, b in BASES])
# CHECK: subclasses: [True, True, True]
try:
    raise TE(DiagId.ARG_NOT_NUMERIC, arg_name="x", arg_type="str")
except TypeError as err:
    print("caught as TypeError:", err.code, "|", err.message)
# CHECK: caught as TypeError: ARG_NOT_NUMERIC | Argument `x` expects a numeric value, but this call passes a value of type `str`.


# --- The author frame: find_user_source_location, register_dsl_package ------
# No location given: the nearest frame outside the DSL package, the standard
# library and `<...>` pseudo-files is the author's, with its column span.
def raise_here():
    raise DSLUserCodeError(DiagId.ARG_INVALID_ALIGNMENT)


try:
    raise_here()
except DSLUserCodeError as err:
    print("auto:", err.line == raise_here.__code__.co_firstlineno + 1, "|", span(err))
# CHECK: auto: True | DSLUserCodeError(DiagId.ARG_INVALID_ALIGNMENT)

# A sub-DSL's module is author code until its package is registered; then its
# frames are skipped and its errors point at the author's call.
HELPER = (
    "import mlir.dsl as m\n"
    "from mlir.dsl.core.diagnostics import find_user_source_location\n\n\n"
    "def where():\n    return find_user_source_location()\n\n\n"
    "def fail():\n    raise m.DSLUserCodeError(m.DiagId.ARG_INVALID_ALIGNMENT)\n"
)
sub, sub_path = load_module("mysubdsl.helpers", HELPER)
loc = sub.where()
print("unregistered:", loc[0] == sub_path, loc[1])
# CHECK: unregistered: True 6
register_dsl_package("mysubdsl")
loc = sub.where()
print("registered:", loc[0] == HERE, loc[1] == inspect.currentframe().f_lineno - 1)
# CHECK: registered: True True
try:
    sub.fail()
except DSLUserCodeError as err:
    print("sub-DSL error:", span(err))
# CHECK: sub-DSL error: sub.fail()
NAMES = ("mlir.dsl", "mlir.dsl.core.dsl", "mysubdsl.helpers", "mlir.dslx", "mlir")
print("is_dsl_module:", [_is_dsl_module(n) for n in NAMES])
# CHECK: is_dsl_module: [True, True, True, False, False]

# --- render_code_frame: gutter, context, caret span, degraded forms ---------
snippet = os.path.join(TMP, "snippet.py")
with open(snippet, "w", encoding="utf-8") as f:
    f.write("def f():\n    x = 1 + 2\n    return x\n\n")
long = os.path.join(TMP, "long.py")
with open(long, "w", encoding="utf-8") as f:
    f.write("".join(f"v{i} = {i}\n" for i in range(1, 13)))

# A span `[col, end_col)` is shown 1-based, under the error line and up to two
# context lines; an end past the line is clipped.
show("span", render_code_frame(snippet, 2, 8, 99))
# CHECK:      span -> 5 lines
# CHECK-NEXT: span ' --> {{.*}}snippet.py:2:9'
# CHECK-NEXT: span '  |'
# CHECK-NEXT: span '1 | def f():'
# CHECK-NEXT: span '2 |     x = 1 + 2'
# CHECK-NEXT: span '  |         ^^^^^'
# No column: one caret under the first non-blank character and no `:col`.
show("nocol", render_code_frame(snippet, 2))
# CHECK:      nocol ' --> {{.*}}snippet.py:2'
# CHECK:      nocol '  |     ^'
# The gutter widens with the line number.
show("wide", render_code_frame(long, 12, 0, 3))
# CHECK:      wide -> 6 lines
# CHECK-NEXT: wide '  --> {{.*}}long.py:12:1'
# CHECK-NEXT: wide '   |'
# CHECK-NEXT: wide '10 | v10 = 10'
# CHECK-NEXT: wide '11 | v11 = 11'
# CHECK-NEXT: wide '12 | v12 = 12'
# CHECK-NEXT: wide '   | ^^^'
# A blank error line keeps just the location line; no frame without a file.
show("blank", render_code_frame(snippet, 4, 0, 1))
# CHECK:      blank -> 1 lines
# CHECK-NEXT: blank ' --> {{.*}}snippet.py:4:1'
show("nofile", render_code_frame(None, 2, 0, 1))
# CHECK:      nofile -> None


# --- One rendered error per catalogue category, each from a real trigger ----
# PHASE: a Python value changed inside a staged loop.
@m.jit
def phase(n: m.Int32) -> m.Int32:
    acc = 0
    for i in range(n):
        acc += i
    return acc


report(phase, 10)
# CHECK:      error[PHASE_MUTATE_PYTHON]:{{.*}} `acc` is a Python value
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#@LINE-7]]
# CHECK:      {{[0-9]+}} |     acc = 0
# CHECK-NEXT: {{[0-9]+}} |     for i in range(n):
# CHECK-NEXT: |{{.*}}^
# CHECK:      = category: not zero-cost (a Python value changes inside a runtime for/while/if)
# CHECK:      = note:{{.*}}region: for
# CHECK:      suggestion:{{.*}}Create `acc` as a runtime value


# SCOPE: a value born in a loop body read after the loop.
@m.jit
def scope(n: m.Int32) -> m.Int32:
    for i in range(n):
        y = i * 2
    return y


report(scope, 3)
# CHECK:      error[SCOPE_REGION_LOCAL_ESCAPES]:{{.*}} `y` is read here, but it was created inside a for body
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#@LINE-5]]:12
# CHECK:      {{[0-9]+}} |     return y
# CHECK-NEXT: |{{.*}}^
# CHECK:      = category: not zero-cost (a variable is not set on every path of a runtime for/while/if)
# CHECK:      suggestion:{{.*}}Read `y` inside the for body that creates it.


# UNSUP: a `for ... else`.
@m.jit
def unsup(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    else:
        acc += 1
    return acc


report(unsup, 3)
# CHECK:      error[UNSUP_SYNTAX]:{{.*}} A `for`/`while` loop with an `else:` clause is not supported in a compiled function: put the `else:` code after the loop.
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#@LINE-9]]:5
# CHECK:      {{[0-9]+}} |     for i in range(n):
# CHECK-NEXT: |{{.*}}^^^^^^^^^^^^^^^^^
# CHECK:      = category: unsupported (a Python construct the DSL does not compile yet)
# CHECK:      suggestion:{{.*}}Compiled functions accept a subset of Python: rewrite this part with a


# TYPE, ARG, CALL, CONFIG, STRUCT, POINTER and the gpu plugin's CONFIG: direct
# triggers go through a lambda so the author frame is the `report` line.
def plain(n):
    return n


@m.struct
class Vec:
    x: m.Int32
    y: m.Int32


report(lambda: m.dtype("nope"))
# CHECK:      error[TYPE_UNKNOWN_DTYPE_NAME]:{{.*}} `nope` is not the name of a DSL data type.
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#@LINE-2]]:16
# CHECK:      {{[0-9]+}} | report(lambda: m.dtype("nope"))
# CHECK-NEXT: |{{.*}}^^^^^^^^^^^^^^^
# CHECK:      = category: usage (types)
report(lambda: m.align(3))
# CHECK:      error[ARG_INVALID_ALIGNMENT]:{{.*}} The value given to `align()` is not a positive power of 2.
# CHECK:      = category: usage (arguments)
report(lambda: m.compile(plain, 3))
# CHECK:      error[CALL_MISSING_JIT_DECORATOR]:{{.*}} The function passed to `compile()` is a plain Python function
# CHECK:      = category: usage (compiling and reusing functions)
report(lambda: OptLevel(5))
# CHECK:      error[CONFIG_INVALID]:{{.*}} `opt-level` has an invalid setting: the optimization level must be an integer between 0 and 3, but got 5.
# CHECK:      = category: usage (compile options)
report(lambda: Vec(z=1))
# CHECK:      error[STRUCT_CONSTRUCTION]:{{.*}} `Vec(...)` cannot be built: unexpected keyword argument(s) ['z']
# CHECK:      = category: usage (structs)
report(lambda: m.Pointer[1, 2, 3])
# CHECK:      error[POINTER_BAD_SUBSCRIPT]:{{.*}} `Pointer[1, 2, 3]` is not a valid pointer annotation.
# CHECK:      = category: usage (pointers)
report(lambda: check_arch("", var="MY_DSL_ARCH"))
# CHECK:      error[gpu:CONFIG_MISSING_ARCH]:{{.*}} No target chip is set for the gpu kernels this function launches: `MY_DSL_ARCH` must name the chip MLIR's gpu lowering compiles for.
# CHECK:      suggestion:{{.*}}Set the environment variable `MY_DSL_ARCH=<chip>` before the DSL is first used.
# CHECK-NOT:  NO ERROR

# --- DSLRuntimeError: the internal-error envelope ---------------------------
print(str(DSLRuntimeError("Leaf registry has no entry for i7")))


# CHECK:      error[INTERNAL]:{{.*}} The compiler hit an internal DSL problem while compiling your code.
# CHECK-NEXT: note:{{.*}}This is a bug in the DSL, not a mistake in your kernel.
# CHECK-NEXT: error:{{.*}}Leaf registry has no entry for i7
# CHECK-NEXT: suggestion:{{.*}}Please report this with the snippet above and your kernel.
# CHECK-NEXT: suggestion:{{.*}}Re-run with MLIR_DSL_SHOW_STACKTRACE=1 to include the full technical detail.
# Raised while a DSL is active: the hint names that DSL's prefix.
class MyPrefixDSL(m.BaseDSL):
    def __init__(self):
        super().__init__(name="MY_DSL")


try:
    with active_dsl(MyPrefixDSL()):
        raise DSLRuntimeError("boom")
except DSLRuntimeError as err:
    print(str(err))
# CHECK: suggestion:{{.*}}Re-run with MY_DSL_SHOW_STACKTRACE=1
# The verifier variant summarises MLIR's message, recovers the compiler
# source frame from its `"file.py":line:col` location and drops the IR dump.
kernel = os.path.join(TMP, "kernel.py")
with open(kernel, "w", encoding="utf-8") as f:
    f.write("import mlir.mlir_dsl as m\n\n\n@m.jit\n")
    f.write("def kern(a: m.Int32, b: m.Float32) -> m.Int32:\n    return a + b\n")
cause = Exception(
    "Verification failed:\n"
    f"loc(\"{kernel}\":6:12): error: 'arith.addi' op requires the same type "
    "for all operands and results\n"
    'see current operation: %0 = "arith.addi"(%arg0, %arg1) : (i32, f32) -> i32'
)
print(str(DSLRuntimeError("IR verification failed", cause=cause)))
# CHECK:      error[INTERNAL]:{{.*}} The compiler could not build valid IR for this code.
# CHECK:      -->{{.*}}kernel.py:6:12
# CHECK:      > 6 |     return a + b
# CHECK:      error:{{.*}}IR verification failed:{{.*}}'arith.addi' op requires the same type for all operands and
# CHECK-NEXT: results
# CHECK-NOT:  see current operation
# CHECK-NEXT: suggestion:{{.*}}Check the source location above

# --- report_warning / DSLWarning --------------------------------------------
# A catalogue warning flows through the `warnings` module as a UserWarning
# subclass (so it is filterable and deduplicated there), carries the WarnId
# and the author's location, and renders as a `warning[CODE]` block.
OUT_OF_RANGE = dict(value=300, type="Int8", min=-128, max=127, wrapped=44)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    WARN_LINE = inspect.currentframe().f_lineno + 1
    report_warning(WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE, mask=0xFF, **OUT_OF_RANGE)
w, msg = caught[0], caught[0].message
print("warning:", len(caught), type(msg).__name__, issubclass(w.category, UserWarning))
# CHECK: warning: 1 DSLWarning True
print("warn_id:", msg.warn_id.name, msg.code, msg.warn_id.category, len(msg.suggestion))
# CHECK: warn_id: TYPE_INT_LITERAL_OUT_OF_RANGE TYPE_INT_LITERAL_OUT_OF_RANGE warning 2
LOC = (HERE, WARN_LINE)  # the DSLWarning's own and the warnings-module location
print("located:", (msg.filename, msg.line) == LOC, (w.filename, w.lineno) == LOC)
# CHECK: located: True True
print(str(msg))
# CHECK:      warning[TYPE_INT_LITERAL_OUT_OF_RANGE]:{{.*}} The Python integer 300 does not fit in `Int8` (range [-128, 127]). Its high bits were dropped, so the kernel will use 44 without any error.
# CHECK-NEXT: -->{{.*}}diagnostics.py:[[#WARN_LINE:]]
# CHECK:      [[#WARN_LINE]] |     report_warning(WarnId.TYPE_INT_LITERAL_OUT_OF_RANGE
# CHECK-NEXT: |{{.*}}^^^^
# CHECK:      = category: warning (numeric literal out of range)
# CHECK:      suggestion:{{.*}}Use a wider integer type that holds 300, e.g. `Int64(300)` or `Uint64(300)`.
# CHECK:      suggestion:{{.*}}mask to the type width
# CHECK-NEXT: first, e.g. `Int8(300 & 0xFF)`.
# CHECK-NOT:  error

# An explicit location is used as given (no readable source: just `-->`); the
# free-form string form has no code or category; fields with a string are a
# programming error.
OVERFLOW = dict(value=1e10, type="Float16", max=65504.0, wrapped="inf")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    report_warning(
        WarnId.TYPE_FLOAT_LITERAL_OVERFLOW, filename="/x.py", lineno=7, **OVERFLOW
    )
    report_warning("Something worth noting")
print(str(caught[0].message))
# CHECK:      warning[TYPE_FLOAT_LITERAL_OVERFLOW]:{{.*}} The Python float 10000000000.0 is outside the finite range of `Float16` (largest finite value 65504), so it became inf
# CHECK-NEXT: -->{{.*}}/x.py:7
# CHECK-NEXT: = category: warning (numeric literal out of range)
w = caught[1].message
print("plain:", w.warn_id, w.code, "|", headline(w))
# CHECK: plain: None None | warning: Something worth noting
try:
    DSLWarning("text", value=1)
except DSLRuntimeError as err:
    print("misuse:", err.message)
# CHECK: misuse: DSLWarning received template fields ['value'] but the first argument is a plain string, not a WarnId.

# The real triggers: each out-of-range literal emits its WarnId once, located
# at the literal; in-range literals are silent.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m.Int8(300)
    m.Float16(1e10)
    m.Int32(3.0e10)
    m.Float16(1e-10)
    m.Int8(5)
    m.Float32(1.5)
print("triggered:", [w.message.code for w in caught])
# CHECK: triggered: ['TYPE_INT_LITERAL_OUT_OF_RANGE', 'TYPE_FLOAT_LITERAL_OVERFLOW', 'TYPE_FLOAT_TO_INT_OUT_OF_RANGE', 'TYPE_FLOAT_LITERAL_UNDERFLOW']
print(str(caught[1].message))
# CHECK:      warning[TYPE_FLOAT_LITERAL_OVERFLOW]
# CHECK:      {{[0-9]+}} |     m.Float16(1e10)
# CHECK-NEXT: |{{.*}}^^^^^^^^^^^^^^^
