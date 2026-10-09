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

import ast
import enum
import linecache
import re
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

__all__ = [
    "Colors",
    "DiagCatalog",
    "DiagId",
    "WarnId",
    "META_VALUE",
    "STAGED_VALUE",
    "NOT_ZERO_COST",
    "UNSUPPORTED",
    "USAGE",
    "WARNING",
    "classify",
    "register_dsl_package",
    "find_user_source_location",
    "render_code_frame",
    "render_user_diagnostic",
]

# The author-facing words for the two kinds of values (a Python value and an
# MLIR op, ``is_mlir_op`` in the code). Defined once here and injected into
# every message as {meta}/{staged} (see DiagCatalog.fill), so the wording is
# consistent and can be changed in one place.
META_VALUE = "Python value"
STAGED_VALUE = "runtime value"


class Colors:
    """Shared ANSI color codes for DSL and compiler diagnostics."""

    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# An MLIR location as printed in a verifier message: `"file.py":line:col`. The
# internal-error envelope recovers a source frame from it.
_PY_LOC_RE = re.compile(r'"(?P<file>[^"]+?\.py)":(?P<line>\d+):(?P<col>\d+)')

# Wrap width of the labelled lines, and the context shown around a compiler
# source location.
_COMPILER_DIAG_TEXT_WIDTH = 100
_COMPILER_CONTEXT_LINES = 2

# The name of the environment prefix quoted in the internal-error envelope when
# the error carries no env manager (``active_dsl`` attaches the DSL's).
_DEFAULT_ENV_PREFIX = "MLIR_DSL"


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


def render_code_frame(
    filename: str | None,
    line: int | None,
    col: int | None = None,
    end_col: int | None = None,
) -> str | None:
    """Code frame: ``--> file:line:col`` + gutter + source lines + caret.

    Shows the error line preceded by up to two lines of context. The ``^``
    underlines ``[col, end_col)`` on the error line when a column span is known,
    otherwise it points at the first non-blank character of that line.
    Best-effort -- returns just the ``-->`` location line if the source cannot
    be read.
    """
    if not filename or not line:
        return None
    frame = _format_user_source_frame(filename, line, col, end_col)
    return "\n".join(frame) if frame else None


def render_user_diagnostic(err: Any) -> str:
    """Render a DSL user diagnostic or warning.

    Used by ``DSLBaseError`` and ``DSLWarning`` so AST pre-processing, tracing,
    runtime errors, and warnings share one source-frame and suggestion format.
    """
    parts = []
    cause_text = err._generate_cause()
    code = getattr(err, "code", None)
    frame = render_code_frame(
        err.filename, err.line, getattr(err, "col", None), getattr(err, "end_col", None)
    )

    if getattr(err, "_is_internal", False):
        return _format_internal_error_diagnostic(err, frame, cause_text)

    diag = getattr(err, "diag_id", None) or getattr(err, "warn_id", None)
    severity = "warning" if getattr(err, "_severity", "error") == "warning" else "error"
    parts.append(
        _format_diagnostic_headline(
            severity,
            err.message,
            code=code or "",
            namespace=getattr(diag, "namespace", "") if diag is not None else "",
            bold=True,
            leading_newline=True,
        )
    )

    if frame:
        parts.append(frame)

    if diag is not None:
        parts.extend(
            _format_user_labeled_text(
                "category", f"{diag.category} ({diag.subcategory})"
            )
        )

    if cause_text:
        parts.extend(_format_user_labeled_text("note", cause_text))
    if err.context:
        if isinstance(err.context, dict):
            for key, value in err.context.items():
                parts.extend(_format_user_labeled_text("note", f"{key}: {value}"))
        else:
            parts.extend(_format_user_labeled_text("note", str(err.context)))
    if err.suggestion:
        fixes = (
            err.suggestion
            if isinstance(err.suggestion, (list, tuple))
            else [err.suggestion]
        )
        for s in fixes:
            parts.extend(_format_labeled_text("suggestion", str(s)))

    parts.append("")
    return "\n".join(parts)


def _format_internal_error_diagnostic(
    err: Any, frame: str | None, cause_text: str
) -> str:
    """Format internal compiler errors with a user-facing bug-report envelope."""
    # Internal errors (compiler bugs) get a "please report" envelope instead of
    # a "here's your mistake + fix" block -- they are not the author's fault.
    # Keep the same headline/source-frame/reason/suggestion grammar as compiler
    # diagnostics so backend failures and internal DSL failures are scannable in
    # the same way.
    is_verifier_error = _is_internal_verifier_error(err.message, cause_text)
    headline = (
        "The compiler could not build valid IR for this code."
        if is_verifier_error
        else "The compiler hit an internal DSL problem while compiling your code."
    )
    parts = [
        _format_diagnostic_headline(
            "error",
            headline,
            code="INTERNAL",
            bold=True,
            leading_newline=True,
        )
    ]
    frame = frame or _internal_error_source_frame_from_cause(cause_text)
    if frame:
        parts.append(frame)

    if is_verifier_error:
        summary = _brief_internal_error(err.message)
        verifier_detail = _brief_verifier_cause(cause_text)
        if verifier_detail:
            summary = f"{summary}: {verifier_detail}"
        parts.extend(_format_labeled_text("error", summary))
    else:
        parts.extend(
            _format_labeled_text(
                "note", "This is a bug in the DSL, not a mistake in your kernel."
            )
        )
        if err.message:
            parts.extend(
                _format_labeled_text("error", _brief_internal_error(err.message))
            )
        if cause_text:
            parts.extend(_format_internal_cause(cause_text))

    if is_verifier_error:
        parts.extend(
            _format_labeled_text(
                "suggestion",
                "Check the source location above for invalid primitive arguments, "
                "types, or address spaces. If the code looks valid, report this "
                "with the snippet above and your kernel.",
            )
        )
    else:
        parts.extend(
            _format_labeled_text(
                "suggestion",
                "Please report this with the snippet above and your kernel.",
            )
        )
    prefix = getattr(getattr(err, "_dsl_env_manager", None), "prefix", None)
    parts.extend(
        _format_labeled_text(
            "suggestion",
            f"Re-run with {prefix or _DEFAULT_ENV_PREFIX}_SHOW_STACKTRACE=1 to "
            "include the full technical detail.",
        )
    )
    parts.append("")
    return "\n".join(parts)


def _is_internal_verifier_error(message: str, cause_text: str) -> bool:
    """Whether an internal error is the MLIR verifier rejecting the built IR."""
    return "IR verification failed" in message or "Verification failed:" in cause_text


def _brief_internal_error(message: str) -> str:
    """One-line summary of an internal error message."""
    if "IR verification failed" in message:
        return "IR verification failed"
    return message.strip()


def _internal_error_source_frame_from_cause(cause_text: str) -> str | None:
    """Source frame for the first ``"file.py":line:col`` location in a cause."""
    loc_match = _PY_LOC_RE.search(cause_text)
    if not loc_match:
        return None
    frame = _format_compiler_source_frame(
        loc_match.group("file"),
        int(loc_match.group("line")),
        int(loc_match.group("col")),
    )
    return "\n".join(frame) if frame else None


def _clean_internal_cause_line(line: str) -> str:
    """Strip the ``Caused exception:`` / ``error:`` scaffolding and the MLIR
    location from one line of a cause, returning ``""`` for a bare label."""
    line = line.strip()
    if not line or line in {"error:", "note:"}:
        return ""
    if line.startswith("Caused exception: "):
        line = line.removeprefix("Caused exception: ").strip()
    if line == "Verification failed:":
        return ""

    loc_match = _PY_LOC_RE.search(line)
    if loc_match:
        line = line[loc_match.end() :].lstrip("): ")
    return line.strip()


def _brief_verifier_cause(cause_text: str) -> str:
    """The first informative line of a verifier failure, without the IR dump."""
    details: list[str] = []
    for line in cause_text.splitlines():
        cleaned = _clean_internal_cause_line(line)
        if not cleaned:
            continue
        if cleaned.startswith("see current operation"):
            continue
        if cleaned.startswith(": ("):
            continue
        if cleaned not in details:
            details.append(cleaned)

    return details[0] if details else ""


def _format_internal_cause(cause_text: str) -> list[str]:
    return _format_labeled_multiline_text("error", cause_text)


def _nearest_function_name(lines: list[str], line_no: int) -> str | None:
    """Name of the innermost function containing ``line_no`` of ``lines``.

    Function locations commonly point at the first decorator rather than the
    ``def`` line, so the containing ``ast`` node is resolved first (the
    smallest span wins) and a decorated function is not mislabeled as the one
    preceding it. Sources that do not parse fall back to the nearest ``def``
    above the line.
    """
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        candidates: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start_line = min(
                (decorator.lineno for decorator in node.decorator_list),
                default=node.lineno,
            )
            end_line = node.end_lineno or node.lineno
            if start_line <= line_no <= end_line:
                candidates.append((end_line - start_line, node.name))
        if candidates:
            return min(candidates)[1]

    # Retain the old best-effort behavior for incomplete or generated sources
    # that cannot be parsed as a complete Python module.
    for idx in range(min(line_no - 1, len(lines) - 1), -1, -1):
        match = re.match(
            r"\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\(",
            lines[idx],
        )
        if match:
            return match.group(1)
    return None


def _read_source_lines(file_path: str | Path) -> list[str]:
    """Source lines of ``file_path`` (via ``linecache``, then the file system);
    empty when the file cannot be read."""
    filename = str(file_path)
    try:
        lines = linecache.getlines(filename)
    except Exception:  # noqa: BLE001
        lines = []
    if lines:
        return [line.rstrip("\n") for line in lines]
    try:
        return Path(filename).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _display_column(col: int | None, *, col_zero_based: bool) -> int | None:
    """1-based column for the ``file:line:col`` location line."""
    if col is None:
        return None
    return col + 1 if col_zero_based else col


def _caret_column(source_line: str, col: int | None, *, col_zero_based: bool) -> int:
    """0-based column of the caret: ``col``, or the first non-blank character
    of the line when no column is known."""
    if col is None:
        return len(source_line) - len(source_line.lstrip())
    return max(col if col_zero_based else col - 1, 0)


def _caret_span(
    source_line: str,
    col: int | None,
    end_col: int | None,
    *,
    col_zero_based: bool,
) -> int:
    """Number of carets: the ``[col, end_col)`` span clipped to the line, and
    one when the span is unknown or empty."""
    if col is None or end_col is None:
        return 1
    start = _caret_column(source_line, col, col_zero_based=col_zero_based)
    end = end_col if col_zero_based else end_col - 1
    end = min(end, len(source_line))
    return max(1, end - start) if end > start else 1


def _format_source_location(
    filename: str,
    line: int,
    col: int | None,
    *,
    absolute_path: bool,
    col_zero_based: bool,
) -> str:
    """``path:line[:col]`` with the column shown 1-based."""
    display_path = str(Path(filename).resolve()) if absolute_path else filename
    display_col = _display_column(col, col_zero_based=col_zero_based)
    loc = f"{display_path}:{line}"
    if display_col is not None:
        loc += f":{display_col}"
    return loc


def _format_user_source_frame(
    filename: str,
    line: int,
    col: int | None,
    end_col: int | None,
) -> list[str]:
    """Lines of the user code frame: ``-->`` location, gutter, up to two
    context lines, the error line and the caret line (0-based ``col``).

    Only the ``-->`` line is produced when the source cannot be read, the line
    is out of range, or it is blank.
    """
    source_lines = _read_source_lines(filename)
    loc = _format_source_location(
        filename,
        line,
        col,
        absolute_path=False,
        col_zero_based=True,
    )
    width = len(str(line))
    pad = " " * width
    lines = [f"{pad}{Colors.BLUE}-->{Colors.RESET} {loc}"]
    if not (1 <= line <= len(source_lines)):
        return lines
    source_line = source_lines[line - 1]
    if not source_line.strip():
        return lines
    caret_col = _caret_column(source_line, col, col_zero_based=True)
    span = _caret_span(source_line, col, end_col, col_zero_based=True)
    caret = f"{Colors.RED}{'^' * span}{Colors.RESET}"
    lines.append(f"{pad} |")
    for current in range(max(1, line - 2), line):
        context_line = source_lines[current - 1]
        lines.append(f"{current:>{width}} | {context_line}")
    lines.append(f"{line} | {source_line}")
    lines.append(f"{pad} | {' ' * caret_col}{caret}")
    return lines


def _format_compiler_source_frame(filename: str, line: int, col: int) -> list[str]:
    """Lines of the compiler-location frame used by the internal-error
    envelope: absolute path, enclosing function name, two context lines on
    each side and a ``>`` marker on the error line (1-based ``col``)."""
    source_lines = _read_source_lines(filename)
    loc = _format_source_location(
        filename, line, col, absolute_path=True, col_zero_based=False
    )
    lines = ["", f"  --> {loc}"]
    fn_name = _nearest_function_name(source_lines, line)
    if fn_name:
        lines.append(f"      in function `{Colors.GREEN}{fn_name}{Colors.RESET}(...)`:")
    if not source_lines:
        return lines

    start = max(1, line - _COMPILER_CONTEXT_LINES)
    end = min(len(source_lines), line + _COMPILER_CONTEXT_LINES)
    width = len(str(end))
    lines.append("   |")
    for current in range(start, end + 1):
        source_line = source_lines[current - 1]
        prefix = ">" if current == line else " "
        lines.append(f"{prefix} {current:{width}d} | {source_line}")
        if current == line:
            caret_col = _caret_column(source_line, col, col_zero_based=False)
            lines.append(f"  {' ' * width} | {' ' * caret_col}^")
    return lines


_DIAGNOSTIC_LABEL_COLORS = {
    "error": Colors.RED,
    "warning": Colors.YELLOW,
    "remark": Colors.CYAN,
    "suggestion": Colors.GREEN,
    "note": Colors.BLUE,
}


def _diagnostic_label_prefixes(
    label: str, *, marker: str = "", prefix: str = "  "
) -> tuple[str, str]:
    """``(plain, colored)`` prefixes of a labelled line, e.g. ``"  = note: "``;
    the plain one sizes the wrap width and continuation indent."""
    plain_prefix = f"{prefix}{marker}{label}: "
    color = _DIAGNOSTIC_LABEL_COLORS.get(label, "")
    colored_label = f"{marker}{label}:"
    colored_prefix = (
        f"{prefix}{color}{colored_label}{Colors.RESET} " if color else plain_prefix
    )
    return plain_prefix, colored_prefix


def _diagnostic_label_indent(label: str, *, marker: str = "") -> str:
    """Continuation indent aligning wrapped text under a label's text."""
    plain_prefix, _ = _diagnostic_label_prefixes(label, marker=marker)
    return " " * len(plain_prefix)


def _format_labeled_text(
    label: str, text: str, *, marker: str = "", wrap: bool = True
) -> list[str]:
    """``label: text`` lines, wrapped at :data:`_COMPILER_DIAG_TEXT_WIDTH`
    with continuation lines indented under the text."""
    plain_prefix, colored_prefix = _diagnostic_label_prefixes(label, marker=marker)
    if not wrap:
        if not text:
            return [colored_prefix.rstrip()]
        return [colored_prefix + text]

    wrapped = textwrap.wrap(
        text,
        width=max(20, _COMPILER_DIAG_TEXT_WIDTH - len(plain_prefix)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [colored_prefix.rstrip()]

    lines = [colored_prefix + wrapped[0]]
    lines.extend(
        _diagnostic_label_indent(label, marker=marker) + line for line in wrapped[1:]
    )
    return lines


def _format_user_labeled_text(label: str, text: str) -> list[str]:
    """``= label: text``, unwrapped: the form used under a user code frame."""
    return _format_labeled_text(label, text, marker="= ", wrap=False)


def _format_labeled_block(label: str, heading: str, body: str) -> list[str]:
    """A labelled heading followed by a multi-line body indented under it."""
    lines = _format_labeled_text(label, heading)
    indent = _diagnostic_label_indent(label)
    width = max(20, _COMPILER_DIAG_TEXT_WIDTH - len(indent))
    for raw_line in body.splitlines():
        if not raw_line:
            lines.append(indent.rstrip())
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(indent + line for line in wrapped)
    return lines


def _format_labeled_multiline_text(label: str, text: str) -> list[str]:
    """Label ``text``, as one wrapped line or as heading plus indented body."""
    text = text.strip()
    if "\n" not in text:
        return _format_labeled_text(label, text)
    heading, body = text.split("\n", 1)
    return _format_labeled_block(label, heading, body)


def _diagnostic_marker(severity: str, code: str = "", namespace: str = "") -> str:
    """``severity``, ``severity[CODE]`` or ``severity[namespace:CODE]``."""
    marker = severity
    if code:
        marker = (
            f"{severity}[{namespace}:{code}]" if namespace else f"{severity}[{code}]"
        )
    return marker


def _format_diagnostic_headline(
    severity: str,
    message: str,
    *,
    code: str = "",
    namespace: str = "",
    bold: bool = False,
    leading_newline: bool = False,
) -> str:
    """The first line of a diagnostic: the colored marker and the message."""
    marker = _diagnostic_marker(severity, code, namespace)
    color = _DIAGNOSTIC_LABEL_COLORS.get(severity, "")
    style = f"{color}{Colors.BOLD if bold else ''}"
    prefix = "\n" if leading_newline else ""
    if style:
        return f"{prefix}{style}{marker}:{Colors.RESET} {message}"
    return f"{prefix}{marker}: {message}"


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
    def category(self) -> str:
        """Author-facing category ("not zero-cost", "unsupported", "usage",
        "warning"); see ``_CATEGORIES``."""
        return _CATEGORIES[(self.namespace, self.name)][0]

    @property
    def subcategory(self) -> str:
        """Author-facing subcategory, a short phrase naming the mistake."""
        return _CATEGORIES[(self.namespace, self.name)][1]

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
    PHASE_MUTATE_PYTHON = (
        "`{var}` is a {meta}, but it is changed inside a for/while/if controlled by a "
        "{staged}{detail}. Only a {staged} can change there: the code inside "
        "is compiled once, not run once per iteration or per taken branch.",
        (
            "Create `{var}` as a {staged} of the matching type before the "
            "for/while/if, e.g. `{var} = Int32(0)`, `Float32(0.0)`, or "
            "`Boolean(False)`, and update it inside.",
            "If `{var}` is a fixed setting, assign it once before the for/while/if and "
            "do not change it inside.",
        ),
    )
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
    TYPE_UNSTABLE_JOIN = (
        "`{var}` has type `{old_type}` on one path and `{new_type}` on "
        "another{detail}. `{var}` must have one type wherever these paths come back "
        "together.",
        (
            "Make every assignment to `{var}` produce the same type.",
            "If a conversion is needed, convert explicitly in each branch, e.g. `{var} "
            "= Float32({var})`, or change the type before the for/while/if so every "
            "path has the new type.",
            "If one path sets `{var}` to `None`, a tuple, or an object, use a number "
            "or `Boolean` flag instead, or decide the branch at compile time with an "
            "`if` on a {meta}.",
        ),
    )
    TYPE_CONDITIONAL_BRANCH_MISMATCH = (
        "The two branches of this `x if cond else y` expression produce different "
        "types. The compiler cannot know at compile time which branch runs, so both "
        "branches must produce the same type.",
        (
            "Convert one branch so both produce the same type, e.g. `Float32(x) if "
            "cond else y`.",
        ),
    )
    TYPE_LOOP_BOUND_NOT_INT = (
        "The loop's `{name}` is a `{dtype}`, but a loop's start, stop, and step must "
        "all be integers.",
        (
            "Use an integer for `{name}`, e.g. `2` instead of `2.0`.",
            "Convert the value to an integer before the loop, e.g. `int(x)` for a "
            "Python value.",
            "For a fractional step, loop over an integer count and scale inside the "
            "body, e.g. `for i in range(n): x = i * 0.5`.",
        ),
    )
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
    SCOPE_CLOSURE_CAPTURE = (
        "Function `{func_name}` captures variable `{var_name}` from the enclosing "
        "scope, but `{func_name}` is used inside a for/while/if controlled by a "
        "{staged}, where captured variables are not supported.",
        (
            "Pass `{var_name}` to `{func_name}` as an argument.",
            "Define `{func_name}` inside the body of the for/while/if.",
        ),
    )
    SCOPE_REGION_LOCAL_ESCAPES = (
        "`{var}` is read here, but it was created inside a {region} body (at "
        "{birth_file}:{birth_line}) and belongs to that body. The body is compiled "
        "once, so this read outside it would keep one pass's value even when the body "
        "runs zero times or takes a different path.",
        (
            "Read `{var}` inside the {region} body that creates it.",
            "Create `{var}` as a {staged} before the {region}, e.g. `{var} = "
            "Int32(0)`, and assign it inside so the value is carried on every path.",
        ),
    )

    # --- CONTAINER ---
    CONTAINER_STRUCTURE_CHANGED = (
        "`{var}` has a different structure at the end of this `{op_type}` than at the "
        "start{detail}. Every path through the `{op_type}` must leave `{var}` with the "
        "same type and structure (same fields, same number of parts).",
        (
            "Assign `{var}` a value of the same type and structure on every branch and "
            "every iteration of the `{op_type}`.",
            "If `{var}` is meant to change structure, do it in a compile-time for/if "
            "(one whose bounds or condition are {meta}s).",
        ),
    )
    CONTAINER_UNSUPPORTED = (
        "`{var}` is a `{type}` holding DSL values, but compiled code carries DSL "
        "values only in tuples, lists and frozen dataclasses.",
        (
            "Use a tuple, a list or a frozen dataclass (`@dataclass(frozen=True)`, "
            "`@struct`) for `{var}`; a dict, set, namedtuple or tuple subclass of "
            "DSL values is not supported.",
        ),
    )
    CONTAINER_INVALID_RECORD = (
        "`{var}` is a `{type}` that compiled code cannot carry: {detail}.",
        (
            "A record carried through compiled code is a frozen dataclass or "
            "`@struct` whose fields all have values: declare it "
            "`@dataclass(frozen=True)`, assign every field (an `init=False` one in "
            "`__post_init__`), and keep DSL values in fields, not in other attributes.",
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

    # --- UNSUP: constructs the DSL does not compile (found by the AST
    # preprocessor) ---
    UNSUP_SYNTAX = (
        "{what} is not supported in a compiled function{detail}.",
        (
            "Compiled functions accept a subset of Python: rewrite this part with a "
            "supported form, or do it in plain Python outside the compiled function "
            "and pass the result in.",
        ),
    )
    UNSUP_EARLY_EXIT = (
        "Early exit ({kind}) is not allowed in {where}. The `{kind}` sits inside a "
        "for/while/if controlled by a {staged}, and that block always runs to its "
        "end when the kernel runs, so it cannot leave early. (An `if`/`while` whose "
        "condition is a {meta} runs in Python and may exit early.)",
        (
            "If the condition is decided at compile time, make it a {meta} (a "
            "plain Python value, not a staged one); the `{kind}` then runs in Python "
            "and is allowed.",
            "If the condition is only decided when the kernel runs, replace the "
            "`{kind}` with a runtime `Boolean` flag: set it where the `{kind}` is "
            "(e.g. `done = Boolean(True)`) and wrap the code that must be skipped in "
            "`if not done:`.",
        ),
    )
    UNSUP_NO_SOURCE = (
        "The source of `{func}` is not available (for example it was defined in a REPL "
        "or through `exec()`), so it cannot be compiled.",
        ("Save the function to a `.py` file and import it from there.",),
    )

    # --- ARG ---
    ARG_ANNOTATION_MISMATCH = (
        "Argument #{num} `{arg_name}` must be {expected}, but got {got}.",
        (
            "Pass a value that matches `{arg_name}`'s annotation.",
            "If the value you pass is the intended one, change `{arg_name}`'s "
            "annotation to that type.",
        ),
    )
    ARG_INVALID_ALIGNMENT = (
        "The value given to `align()` is not a positive power of 2.",
        ("Pass a positive power of 2 to `align()`, e.g. `align(16)`.",),
    )
    ARG_NOT_NUMERIC = (
        "Argument `{arg_name}` expects a numeric value, but this call passes a value "
        "of type `{arg_type}`.",
        (
            "Pass an `int`, a `float`, or a DSL numeric such as `Int32(...)` for "
            "`{arg_name}`.",
        ),
    )
    ARG_UNSUPPORTED_TYPE = (
        "Argument #{num} `{arg_name}` of `{function_name}` has type `{arg_type}`, "
        "which cannot be passed into compiled code{detail}.",
        (
            "Pass a DSL numeric such as `Int32`, a pointer, a host buffer one of the "
            "DSL's adapter plugins accepts (a numpy array or a torch tensor with the "
            "dlpack plugin), or a "
            "tuple or frozen record of them for `{arg_name}`.",
            "To pass a custom class, register an adapter for it with "
            "`@register_jit_arg_adapter(YourClass)`.",
        ),
    )
    ARG_BUFFER_INVALID = (
        "Argument `{arg_name}` cannot be used as a `Pointer` argument: {detail}.",
        (
            "Pass one contiguous block of memory on the device the function runs on, "
            "e.g. `np.ascontiguousarray(a)`, `t.contiguous()`, or `t.cpu()` for a "
            "host function.",
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
    CALL_MISSING_JIT_DECORATOR = (
        "The function passed to `compile()` is a plain Python function (not "
        "decorated with the DSL's `@jit`).",
        ("Add `@jit` above the definition of the function you pass to `compile()`.",),
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
    CALL_NEVER_ISSUED = (
        "`{function_name}` was called but never issued. Calling a `{decorator}` "
        "function only prepares the call; it runs when the prepared call is issued.",
        (
            "Issue it, e.g. `{function_name}(...){issue}`.",
            "If the call is not needed, remove it.",
        ),
    )
    CALL_ALREADY_ISSUED = (
        "`{function_name}` is issued twice from one prepared call; "
        "`{function_name}(...)` runs once.",
        ("Call `{function_name}(...)` again for a second issue, one `{issue}` each.",),
    )
    CALL_META_VALUE_MISMATCH = (
        "Argument #{num} `{arg_name}` is a compile-time (Meta) value: this compiled "
        "`{function_name}` was specialized for {expected}, but is called with {got}.",
        (
            "Call the `@jit` function itself; it compiles one specialization per Meta "
            "value and picks the right one.",
            "Or `compile(...)` the function again with this value.",
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

    # --- STRUCT ---
    STRUCT_DEFINITION = (
        "`{name}` is not a valid `@struct`: {detail}.",
        (
            "Declare at least one field annotated with a DSL type such as `Int32` or "
            "`Float32`, a `Pointer[T]` or another `@struct` class; keep compile-time "
            "settings as plain class attributes.",
        ),
    )
    STRUCT_CONSTRUCTION = (
        "`{name}(...)` cannot be built: {detail}.",
        (
            "Construct a `@struct` by field, e.g. `Vec2(x=Float32(1.0), "
            "y=Float32(2.0))`, one DSL value per field.",
        ),
    )
    STRUCT_FIELD_ASSIGNMENT = (
        "`{name}.{field}` cannot be assigned: a `@struct` instance is immutable.",
        ("Create an updated copy instead: `v = v.replace({field}=...)`.",),
    )

    # --- POINTER ---
    POINTER_INDEX_UNSUPPORTED = (
        "A `Pointer` cannot be indexed with a {kind}; only a single integer offset is "
        "supported.",
        (
            "Use `load(count=...)` / `store(...)` for several elements, or index one "
            "element at a time, e.g. `p[i]`.",
        ),
    )
    POINTER_BAD_SUBSCRIPT = (
        "`Pointer[{args}]` is not a valid pointer annotation.",
        (
            "Write `Pointer[dtype]` or `Pointer[dtype, space]`, e.g. "
            "`Pointer[Float32]` or `Pointer[Float32, 1]`.",
        ),
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


# =============================================================================
# Author-facing classification
# =============================================================================

# Author-facing classification of every catalog entry: (category, subcategory).
# The category names the rule the author broke; the subcategory says how.
# Rendered under the source frame as "= category: <category> (<subcategory>)".
# "not zero-cost" = the code would only work if the Python interpreter ran while
# the compiled code runs; "unsupported" = a Python construct the DSL does not
# compile (yet); "usage" = a wrong call, launch, option or type.
NOT_ZERO_COST = "not zero-cost"
UNSUPPORTED = "unsupported"
USAGE = "usage"
WARNING = "warning"

# Keyed by ``(namespace, code)``: the base catalogs use the empty namespace, a
# plugin's or sub-DSL's catalog classifies its own codes through ``classify``.
_CATEGORIES: dict[tuple[str, str], tuple[str, str]] = {}


def classify(catalog: type, category: str, subcategory: str, *codes: str) -> None:
    """Classify ``codes`` of ``catalog`` (a ``DiagCatalog`` enum) under
    ``category``/``subcategory``, so they render like the base ones; every code
    must be a member of the catalog. Called once per catalog, next to its
    definition (the gpu plugin's ``GpuDiagId``, the tvm_ffi plugin's
    ``TvmFfiDiagId``)."""
    members = getattr(catalog, "__members__", {})
    for code in codes:
        if code not in members:
            # Imported here: ``common`` imports this module.
            from .common import DSLRuntimeError

            raise DSLRuntimeError(
                f"{catalog.__name__} has no member {code!r} to classify",
                context={"catalog": catalog.__name__, "code": code},
            )
        _CATEGORIES[(catalog.namespace, code)] = (category, subcategory)


def _classify(category: str, subcategory: str, *codes: str) -> None:
    for code in codes:
        _CATEGORIES[("", code)] = (category, subcategory)


_classify(
    NOT_ZERO_COST,
    "a Python value changes inside a runtime for/while/if",
    "PHASE_MUTATE_PYTHON",
)
_classify(
    NOT_ZERO_COST,
    "a type or structure differs between paths of a runtime for/while/if",
    "TYPE_UNSTABLE_JOIN",
    "TYPE_CONDITIONAL_BRANCH_MISMATCH",
    "CONTAINER_STRUCTURE_CHANGED",
    "CONTAINER_UNSUPPORTED",
    "CONTAINER_INVALID_RECORD",
    "CONTAINER_TOO_DEEP",
)
_classify(
    NOT_ZERO_COST,
    "a variable is not set on every path of a runtime for/while/if",
    "SCOPE_UNBOUND_NAME_IN_TRACE",
    "SCOPE_REGION_LOCAL_ESCAPES",
)
_classify(
    NOT_ZERO_COST,
    "a Python-only operation is applied to a runtime value",
    "PHASE_REQUIRES_CONSTANT",
    "PHASE_DYNAMIC_INDEX",
    "PHASE_DYNAMIC_TO_STATIC_BOOL",
)
_classify(
    UNSUPPORTED,
    "a Python construct the DSL does not compile yet",
    "UNSUP_SYNTAX",
    "UNSUP_EARLY_EXIT",
    "UNSUP_NO_SOURCE",
    "SCOPE_CLOSURE_CAPTURE",
)
_classify(
    USAGE,
    "arguments",
    "ARG_ANNOTATION_MISMATCH",
    "ARG_UNSUPPORTED_TYPE",
    "ARG_NOT_NUMERIC",
    "ARG_INVALID_ALIGNMENT",
    "ARG_BUFFER_INVALID",
    "CALL_ARGUMENTS",
    "CALL_META_VALUE_MISMATCH",
    "TYPE_LOOP_BOUND_NOT_INT",
)
_classify(
    USAGE,
    "compiling and reusing functions",
    "CALL_MISSING_JIT_DECORATOR",
    "CALL_NOT_CALLABLE",
    "CALL_OUTSIDE_JIT",
    "CALL_NEVER_ISSUED",
    "CALL_ALREADY_ISSUED",
    "CALL_PLUGIN_REQUIRED",
    "TYPE_RETURN_MISMATCH",
)
_classify(
    USAGE,
    "compile options",
    "CONFIG_INVALID",
)
_classify(
    USAGE,
    "types",
    "TYPE_IMPLICIT_PROMOTION_UNSUPPORTED",
    "TYPE_INT_POW_UNSUPPORTED",
    "TYPE_UNSUPPORTED_MLIR_TYPE",
    "TYPE_UNKNOWN_DTYPE_NAME",
)
_classify(
    USAGE,
    "structs",
    "STRUCT_DEFINITION",
    "STRUCT_CONSTRUCTION",
    "STRUCT_FIELD_ASSIGNMENT",
)
_classify(
    USAGE,
    "pointers",
    "POINTER_INDEX_UNSUPPORTED",
    "POINTER_BAD_SUBSCRIPT",
)
_classify(
    WARNING,
    "numeric literal out of range",
    "TYPE_INT_LITERAL_OUT_OF_RANGE",
    "TYPE_FLOAT_TO_INT_OUT_OF_RANGE",
    "TYPE_FLOAT_LITERAL_OVERFLOW",
    "TYPE_FLOAT_LITERAL_UNDERFLOW",
)
