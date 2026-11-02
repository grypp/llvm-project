# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""User-facing DSL error catalog and diagnostic renderer.

One entry per author-facing mistake.  The enum **member name** is the stable
error code (e.g. ``TYPE_UNSTABLE_JOIN``) -- there are no hand-written numbers to
keep in sync.  The member **value** is ``(message_template, (fix_line, ...))``,
written in the author's words, never in compiler/IR terms::

    raise DSLUserCodeError(DiagId.TYPE_UNSTABLE_JOIN, filename=fn, lineno=ln,
                           var="count", old_type="Int32", new_type="Float32")

The exception consumes the ``DiagId`` directly (see ``DiagCatalog.fill``) and
renders through ``render_user_diagnostic`` (code frame + suggestions), so an
error raised while tracing looks the same as one raised while reading code.

Auto-doc readiness: every entry exposes ``.code`` (the name), ``.category`` /
``.subcategory`` (the author-facing classification), ``.message`` and
``.fix``, so a doc generator just iterates ``for d in DiagId: ...``.

The module has three parts: the DSL-package registry that tells user frames
from DSL frames (``register_dsl_package``, ``find_user_source_location``), the
renderer (``render_code_frame``, ``render_user_diagnostic`` and the internal
error envelope), and the catalogs (``DiagId``, ``WarnId``, the ``DiagCatalog``
mix-in that namespaced plugin catalogs such as ``GpuDiagId`` reuse, and the
author-facing classification at the end).
"""

import enum
import sys
import types
from typing import Any

__all__ = [
    "DiagCatalog",
    "DiagId",
    "WarnId",
    "META_VALUE",
    "STAGED_VALUE",
    "register_dsl_package",
    "find_user_source_location",
    "render_user_diagnostic",
]

# The author-facing words for the two kinds of values (a Python value and an
# MLIR op, ``is_mlir_op`` in the code). Defined once here and injected into
# every message as {meta}/{staged} (see DiagCatalog.fill), so the wording is
# consistent and can be changed in one place.
META_VALUE = "Python value"
STAGED_VALUE = "runtime value"


# =============================================================================
# DSL package registry (which frames are "inside the DSL")
# =============================================================================

_DSL_PACKAGES: list[str] = ["mlir.dsl"]


def register_dsl_package(prefix: str) -> None:
    """Treat modules under ``prefix`` as DSL-internal when locating user code.

    ``find_user_source_location`` walks out of every frame whose module name is
    a registered package (or a submodule of one), so a sub-DSL registers its own
    package once at import and its diagnostics keep pointing at the author's
    line, not at the sub-DSL's.  Registering a prefix twice is harmless.
    """
    if prefix and prefix not in _DSL_PACKAGES:
        _DSL_PACKAGES.append(prefix)


def _is_dsl_module(mod: str) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in _DSL_PACKAGES)


# =============================================================================
# Renderer
# =============================================================================


def find_user_source_location() -> (
    tuple[str | None, int | None, int | None, int | None]
):
    """Best-effort author location: ``(filename, line, col, end_col)``.

    Walks out to the nearest call-stack frame *outside* the DSL package (frames
    classified by module name -- robust across source / build-tree / installed
    layouts), so a diagnostic raised deep inside the DSL still points at the
    author's own line.  On Python 3.11+ the column span of the instruction the
    author frame is suspended at is recovered via ``co_positions`` so the caret
    can underline it.  Returns ``(None, None, None, None)`` when no author frame
    is on the stack.
    """
    try:
        frame: types.FrameType | None = sys._getframe(1)
    except Exception:  # noqa: BLE001
        return None, None, None, None
    stdlib: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())
    try:
        while frame is not None:
            mod = frame.f_globals.get("__name__", "") or ""
            top = mod.split(".", 1)[0]
            fn = frame.f_code.co_filename
            is_internal = _is_dsl_module(mod) or top in stdlib or fn.startswith("<")
            if not is_internal:
                line, col, end_col = frame.f_lineno, None, None
                try:  # exact column span of the suspended instruction (3.11+)
                    positions = list(frame.f_code.co_positions())  # type: ignore[attr-defined]
                    idx = frame.f_lasti // 2
                    if 0 <= idx < len(positions):
                        sl, el, sc, ec = positions[idx]
                        if sl is not None:
                            line = sl
                            if sc is not None and el == sl:
                                col, end_col = sc, ec
                except Exception:  # noqa: BLE001 -- best-effort column
                    pass
                return fn, line, col, end_col
            frame = frame.f_back
    finally:
        del frame
    return None, None, None, None


def render_user_diagnostic(err: Any) -> str:
    """Seed stand-in for the shared renderer: the headline, the location, the
    notes and the suggestions as plain lines (the renderer PR replaces it)."""
    is_warning = getattr(err, "_severity", "error") == "warning"
    severity = "warning" if is_warning else "error"
    code = "INTERNAL" if err._is_internal else getattr(err, "code", None)
    headline = f"{severity}[{code}]" if code else severity
    parts = [f"{headline}: {err.message}"]
    if err.filename and err.line:
        parts.append(f"--> {err.filename}:{err.line}")
    cause = err._generate_cause()
    if cause:
        parts.append(f"  = note: {cause}")
    if isinstance(err.context, dict):
        parts.extend(f"  = note: {k}: {v}" for k, v in err.context.items())
    elif err.context:
        parts.append(f"  = note: {err.context}")
    fixes = err.suggestion
    if not isinstance(fixes, (list, tuple)):
        fixes = [fixes]
    parts.extend(f"  suggestion: {s}" for s in fixes if s)
    return "\n".join(parts)


# =============================================================================
# Catalog machinery
# =============================================================================


class _MissingField:
    """Renders a not-supplied ``{field}`` back as the literal placeholder.

    A diagnostic must never crash the compile it is trying to explain, so a
    template referencing a field the call site forgot degrades to showing the
    raw ``{field}`` (caught by tests) instead of raising ``KeyError``.
    """

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __format__(self, spec: str) -> str:
        return "{" + self.key + (":" + spec if spec else "") + "}"

    def __str__(self) -> str:
        return "{" + self.key + "}"


class _SafeFields(dict):
    """Template fields that render a missing key as :class:`_MissingField`."""

    def __missing__(self, key: str) -> "_MissingField":
        return _MissingField(key)


class DiagCatalog:
    """Shared behaviour for the diagnostic catalogs (:class:`DiagId` errors and
    :class:`WarnId` warnings, and the namespaced catalogs of plugins and
    sub-DSLs).

    Each catalog is an ``enum.Enum`` whose member **name** is the stable code
    and whose value is ``(message_template, (fix_line, ...))``.  This mix-in
    supplies the code/category/message/fix accessors and ``fill`` (template
    substitution).  A catalog other than the base one sets ``namespace`` so its
    codes render as ``error[<namespace>:CODE]``; inside an ``Enum`` body the
    attribute must be written ``namespace = enum.nonmember("gpu")`` so the
    enum machinery does not turn it into a member.
    """

    # ``name`` and ``value`` are supplied by ``enum.Enum`` at runtime (this
    # mix-in is combined with ``enum.Enum`` in DiagId/WarnId); declared here so
    # the accessor bodies below type-check.
    name: str
    value: Any  # the (message_template, (fix_line, ...)) tuple, per enum member

    # Rendered as ``error[<namespace>:CODE]``; empty for the base catalog.
    namespace: str = ""

    @property
    def code(self) -> str:
        """The stable, human-readable code shown to the user (the name)."""
        return self.name

    @property
    def prefix(self) -> str:
        """The name prefix (PHASE / TYPE / ...), the catalog's internal grouping."""
        return self.name.split("_", 1)[0]

    @property
    def message(self) -> str:
        """The message template, with ``{field}`` placeholders."""
        return self.value[0]

    @property
    def fix(self) -> tuple[str, ...]:
        """The fix templates, rendered as ``suggestion:`` lines."""
        return self.value[1]

    def fill(self, **fields: Any) -> tuple[str, tuple[str, ...]]:
        """Fill the templates and return ``(message, fixes)``.

        No code suffix -- the stable code (``.code``) is shown separately in the
        ``error[CODE]:`` / ``warning[CODE]:`` header.  The two
        value-phase names :data:`META_VALUE` / :data:`STAGED_VALUE` are always
        available as ``{meta}`` / ``{staged}``; a field a call site forgot
        degrades to the literal ``{field}`` instead of crashing (see
        :class:`_SafeFields`)."""
        fields.setdefault("meta", META_VALUE)
        fields.setdefault("staged", STAGED_VALUE)
        # Optional enrichment field: templates may reference ``{detail}`` to show
        # what concretely changed (e.g. old vs new structure/type); default empty
        # so call sites that omit it render cleanly.
        fields.setdefault("detail", "")
        safe = _SafeFields(fields)
        message = self.message.format_map(safe)
        fixes = tuple(f.format_map(safe) for f in self.fix)
        return message, fixes


# =============================================================================
# Error catalog
# =============================================================================


class DiagId(DiagCatalog, enum.Enum):
    """User-facing diagnostics: member name == stable code, value == (message, fix).

    Naming convention ``<CATEGORY>_<rest>`` -- CATEGORY is one of
    PHASE / TYPE / SCOPE / CONTAINER / UNSUP / ARG / CALL / CONFIG / STRUCT /
    POINTER and is what ``.prefix`` returns.  Keep the prefix when adding
    entries.  Internal/compiler errors do not live here -- raise
    ``DSLRuntimeError`` for those.
    """

    # --- PHASE ---
    PHASE_DYNAMIC_INDEX = (
        "Cannot use a {staged} as a list index or for loop range in plain Python code; "
        "only a Python `int` works there.",
        (
            "For a loop: put `for ... in range(...)` inside a function decorated with "
            "`@jit` (with the preprocessor enabled), so it becomes a for controlled by "
            "a {staged}, or build the loop with `for_(...)`.",
            "For an index: if it is known at compile time, keep it a Python `int`, "
            "i.e. a {meta}, instead of a {staged}.",
        ),
    )
    PHASE_DYNAMIC_TO_STATIC_BOOL = (
        "A {staged} is used where plain Python needs a true/false answer, for example "
        "`if x:` in a function that is not compiled or has preprocessing disabled, or "
        "`sorted()` comparing {staged}s; only a {meta} can be used there.",
        (
            "Decorate the function that contains the `if`/`while` with `@jit` and keep "
            "the preprocessor enabled, so it becomes an if/while controlled by a "
            "{staged}, or build it with `if_(...)`/`while_(...)`.",
            "If the condition is decided at compile time, make it a {meta}.",
        ),
    )
    PHASE_REQUIRES_CONSTANT = (
        "{what} requires a value known at compile time, but it received a "
        "{staged}{detail}.",
        (
            "Make the value a Python constant: a literal, or a value computed only "
            "from Python values (a {meta}).",
            "If the value is only known when the kernel runs, use the runtime form "
            "instead: a plain `if`/`while` on the staged value, or `x != 0` in place "
            "of `bool(x)`.",
        ),
    )

    # --- TYPE ---
    TYPE_RETURN_MISMATCH = (
        "This function returns {got}, which a compiled function cannot return{detail}.",
        (
            "Return a DSL numeric such as `Int32`/`Float32`, a `@struct` instance, or "
            "a tuple or frozen dataclass of them.",
            "A kernel cannot return a value: write its results through a `Pointer` "
            "argument instead.",
            "If the function returns nothing, remove its return type annotation.",
        ),
    )
    TYPE_IMPLICIT_PROMOTION_UNSUPPORTED = (
        "`{lhs_type}` and `{rhs_type}` cannot be combined by `{op}` without an "
        "explicit conversion: the DSL does not pick a common type for them.",
        (
            "Convert one operand explicitly so both have one type, e.g. "
            "`{lhs_type}(b)` or `{rhs_type}(a)`.",
        ),
    )
    TYPE_UNSUPPORTED_MLIR_TYPE = (
        "An MLIR value of type `{mlir_type}` reached the DSL, but no DSL type claims "
        "that MLIR type.",
        (
            "Register a leaf class for this type with `register_leaf(...)`.",
            "Convert the value to a supported type before it enters the DSL.",
        ),
    )
    TYPE_UNKNOWN_DTYPE_NAME = (
        "`{name}` is not the name of a DSL data type.",
        (
            "Use one of the DSL's type names, e.g. `Int32`, `Float32`, `Boolean`, or "
            "the numpy spelling `float32`.",
            "For a numpy array, use a dtype the DSL maps, e.g. `np.float32` or "
            "`np.int32`.",
        ),
    )
    TYPE_INT_POW_UNSUPPORTED = (
        "`**` between two runtime integers is not supported: compiled code has no "
        "integer power operation.",
        (
            "Cast one operand to a float, e.g. `Float32(a) ** b`.",
            "If both operands are known at compile time, compute the power with Python "
            "ints before the kernel.",
        ),
    )

    # --- SCOPE ---
    SCOPE_UNBOUND_NAME_IN_TRACE = (
        "`{var}` is read here, but it was never given a value on the path taken to "
        "this point{detail}. Every path that reaches this read must set it first.",
        (
            "Set `{var}` before the for/while/if, even if every branch or iteration "
            "assigns it later.",
            "Set it before any compile-time `if` as well: a branch on a {meta} "
            "that is not taken does not assign the variables inside it.",
        ),
    )

    # --- CONTAINER ---
    CONTAINER_UNSUPPORTED = (
        "`{var}` is a `{type}` holding DSL values, but compiled code carries DSL "
        "values only in tuples, lists and frozen dataclasses.",
        (
            "Use a tuple, a list or a frozen dataclass (`@dataclass(frozen=True)`, "
            "`@struct`) for `{var}`; a dict, set, namedtuple or tuple subclass of "
            "DSL values is not supported.",
        ),
    )
    CONTAINER_TOO_DEEP = (
        "`{var}` is nested deeper than Python's recursion limit allows (or it "
        "contains itself), so it cannot be flattened.",
        (
            "Flatten the value yourself or raise `sys.setrecursionlimit`; a container "
            "that refers back to itself is not supported.",
        ),
    )
    # preprocessor) ---

    # --- ARG ---
    ARG_ANNOTATION_MISMATCH = (
        "Argument #{num} `{arg_name}` must be {expected}, but got {got}.",
        (
            "Pass a value that matches `{arg_name}`'s annotation.",
            "If the value you pass is the intended one, change `{arg_name}`'s "
            "annotation to that type.",
        ),
    )
    ARG_NOT_NUMERIC = (
        "Argument `{arg_name}` expects a numeric value, but this call passes a value "
        "of type `{arg_type}`.",
        (
            "Pass an `int`, a `float`, or a DSL numeric such as `Int32(...)` for "
            "`{arg_name}`.",
        ),
    )

    # --- CALL ---
    CALL_ARGUMENTS = (
        "The call to `{function_name}` does not match its parameters: {detail}.",
        (
            "Pass exactly one value for every parameter without a default, by "
            "position or by its exact name, and nothing else.",
            "Keyword-only parameters such as loop options are passed by name, e.g. "
            "`range(n, unroll=2)`.",
        ),
    )
    CALL_NOT_CALLABLE = (
        "`{decorator}` was applied to {got}, which is not a plain Python function.",
        (
            "Apply `{decorator}` directly above a `def`.",
            "To compile a method, decorate the method itself; to compile a callable "
            "object, decorate its `__call__`.",
        ),
    )
    CALL_OUTSIDE_JIT = (
        "`{api}` was called from plain Python, but it can only be used inside a "
        "function decorated with `{decorator}`.",
        (
            "Move this call into a function decorated with `{decorator}`, then call "
            "that function.",
        ),
    )
    CALL_PLUGIN_REQUIRED = (
        "`{name}` needs {plugin}, which this DSL does not name.",
        ("Name it: `{fix}`.",),
    )

    # --- CONFIG ---
    CONFIG_INVALID = (
        "`{var}` has an invalid setting: {detail}.",
        ("Set `{var}` to a supported value, or leave it unset for the default.",),
    )


# =============================================================================
# Warning catalog
# =============================================================================


class WarnId(DiagCatalog, enum.Enum):
    """Author-facing **warnings** (non-fatal) -- a separate catalog from the
    :class:`DiagId` errors so the two namespaces never collide.

    Same shape as ``DiagId`` (member name == stable code, value ==
    ``(message, fix)``) and rendered through the same block, but with a yellow
    ``warning[CODE]`` header.  Raised via ``report_warning`` /
    ``DSLWarning(WarnId.X, ...)``.
    """

    TYPE_INT_LITERAL_OUT_OF_RANGE = (
        "The Python integer {value} does not fit in `{type}` (range [{min}, {max}]). "
        "Its high bits were dropped, so the kernel will use {wrapped} without any "
        "error.",
        (
            "Use a wider integer type that holds {value}, e.g. `Int64({value})` or "
            "`Uint64({value})`.",
            "If you meant the wrap-around (a mask or other bit pattern), mask to the "
            "type width first, e.g. `{type}({value} & 0x{mask:X})`.",
        ),
    )

    TYPE_FLOAT_TO_INT_OUT_OF_RANGE = (
        "The Python float {value} does not fit in `{type}` (range [{min}, {max}]), so "
        "it became {result}. The kernel will use that value without any error, and it "
        "can differ between machines that compile the kernel.",
        (
            "Clamp the value into range before converting, e.g. `{type}(max({min}, "
            "min({max}, {value})))`.",
            "Use a wider integer type such as `Int64` if {value} fits in it.",
        ),
    )

    TYPE_FLOAT_LITERAL_OVERFLOW = (
        "The Python float {value} is outside the finite range of `{type}` (largest "
        "finite value {max:g}), so it became {wrapped}, which the kernel will use "
        "without any error.",
        (
            "Use a wider float type that holds {value}, e.g. `Float64({value})`.",
            "If you meant infinity, write it explicitly, e.g. `{type}(math.inf)` or "
            "`{type}(-math.inf)`.",
        ),
    )

    TYPE_FLOAT_LITERAL_UNDERFLOW = (
        "The Python float {value} is closer to zero than the smallest nonzero `{type}` "
        "value ({tiny:g}), so it became {wrapped}, which the kernel will use without "
        "any error.",
        ("Use a wider float type that holds {value}, e.g. `Float64({value})`.",),
    )
