# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optimization remarks of one compile.

The remarks are the upstream ``RemarkEngine``'s (``Context.enable_remarks``);
a :class:`RemarkSession` owns the engine and a diagnostic handler for one
compile, streams the remarks to a file (``REMARKS_OUTPUT``) or collects and
renders them from its callback (``remarks``), and quotes the pass errors the
handler saw when a compile fails (``error_diagnostics``).
"""

import re
import sys
from typing import Any

from ... import ir
from . import diagnostics as _diagnostics
from .common import DSLRuntimeError
from ..util.logger import log

__all__ = ["REMARK_POLICIES", "RemarkSession", "error_diagnostics", "remarks_available"]

# The ``REMARKS_POLICY`` values a DSL accepts (``BaseDSL`` checks the setting).
REMARK_POLICIES: tuple[str, ...] = ("all", "final")


# ``Remark::print`` spells the kind as a bracketed prefix; the headline drops it
# and carries the kind as the diagnostic code instead.
_REMARK_KIND_RE = re.compile(r"^\[(Passed|Missed|Failure|Analysis|Unknown)\]\s*")


def remarks_available() -> bool:
    """Whether the ``mlir`` bindings in use carry the remark engine API
    (``Context.enable_remarks``)."""
    return hasattr(ir.Context, "enable_remarks")


def _remark_format(remark_output: str) -> Any:
    """The ``ir.RemarkFormat`` of a ``REMARKS_OUTPUT`` path, from its suffix."""
    if remark_output.endswith(".bitstream"):
        return ir.RemarkFormat.BITSTREAM
    return ir.RemarkFormat.YAML


def _remark_policy(remark_policy: str) -> Any:
    """The ``ir.RemarkPolicy`` named by a ``REMARKS_POLICY`` value, or None."""
    if not remarks_available():
        return None
    return {"all": ir.RemarkPolicy.ALL, "final": ir.RemarkPolicy.FINAL}.get(
        remark_policy
    )


# How many wrapping locations (``NameLoc``, ``CallSiteLoc`` under
# ``LOC_TRACEBACKS``, ``FusedLoc``) are peeled before giving up.
_MAX_LOCATION_DEPTH = 8


def _user_location(loc: ir.Location) -> tuple[str | None, int | None, int | None]:
    """The ``(filename, line, col)`` of the ``FileLineColLoc`` inside ``loc``,
    unwrapping the ``NameLoc`` of ``BaseDSL.get_ir_location`` and the
    ``CallSiteLoc`` chain of ``LOC_TRACEBACKS``; ``(None, None, None)`` when
    there is none."""
    for _ in range(_MAX_LOCATION_DEPTH):
        if isinstance(loc, ir.FileLineColLoc):
            return loc.filename, loc.start_line, loc.start_col
        if isinstance(loc, ir.NameLoc):
            loc = loc.child_loc
        elif isinstance(loc, ir.CallSiteLoc):
            loc = loc.callee
        elif isinstance(loc, ir.FusedLoc) and loc.locations:
            loc = loc.locations[0]
        else:
            break
    return None, None, None


def _remark_record(remark: Any) -> dict[str, Any]:
    """The structured view of an ``ir.Remark`` kept in ``RemarkSession.remarks``
    (``dsl.collected_remarks``): kind, name, category, full_category,
    function, message and location, the user source position (filename, line,
    col), remark_id and args."""
    filename, line, col = _user_location(remark.location)
    kind = str(remark.kind).rsplit(".", 1)[-1].lower()
    return {
        "kind": "failed" if kind == "failure" else kind,
        "name": remark.remark_name,
        "category": remark.category_name,
        "full_category": remark.full_category_name,
        "function": remark.function_name,
        "message": remark.message,
        "location": str(remark.location),
        "filename": filename,
        "line": line,
        "col": col,
        "remark_id": remark.remark_id,
        "args": dict(remark.args),
    }


def _diagnostic_remark_record(diagnostic: Any) -> dict[str, Any]:
    """The record of a ``DiagnosticSeverity.REMARK`` diagnostic: the engine's
    ``emit`` form, when another owner enabled it without a streamer."""
    message = str(diagnostic.message)
    match = _REMARK_KIND_RE.match(message)
    kind = match.group(1).lower() if match else "unknown"
    filename, line, col = _user_location(diagnostic.location)
    return {
        "kind": "failed" if kind == "failure" else kind,
        "name": "",
        "category": "",
        "full_category": "",
        "function": "",
        "message": message,
        "location": str(diagnostic.location),
        "filename": filename,
        "line": line,
        "col": col,
        "remark_id": 0,
        "args": {},
    }


def _render_remark(record: dict[str, Any]) -> str:
    """Render one remark record as ``remark[kind]: ...`` over its code frame."""
    message = _REMARK_KIND_RE.sub("", record["message"])
    lines = [
        _diagnostics._format_diagnostic_headline("remark", message, code=record["kind"])
    ]
    frame = _diagnostics.render_code_frame(
        record["filename"], record["line"], record["col"]
    )
    if frame:
        lines.append(frame)
    return "\n".join(lines)


def error_diagnostics(exc: ir.MLIRError, session: "RemarkSession") -> str:
    """The error diagnostics of a failed pass pipeline, one per line: those
    the pass manager captured into ``exc`` (its own handler sees them first)
    and any the session's handler recorded."""
    lines = [
        f"{diagnostic.location}: {diagnostic.message}"
        for diagnostic in getattr(exc, "error_diagnostics", ())
    ]
    lines.extend(session.errors)
    return "\n".join(lines)


class RemarkSession:
    """Own the context's remark engine and diagnostic handler for one compile.

    ``with session:`` enables the engine (``REMARKS`` filter; ``REMARKS_POLICY``;
    a ``REMARKS_OUTPUT`` path streams YAML/bitstream, otherwise the remarks
    come back through a callback, are kept as records in ``remarks`` and are
    rendered to stderr), attaches a diagnostic handler that quotes pass errors
    and renders ``emit``-form remarks, and finalizes on exit so ``policy="final"``
    remarks are flushed while the handler is still attached. A session whose
    context already carries an engine (an enclosing session, or the user's own)
    leaves the engine alone and only collects.
    """

    def __init__(
        self,
        context: ir.Context,
        *,
        remark_filter: str = "",
        remark_policy: str = "all",
        remark_output: str = "",
    ) -> None:
        self.context = context
        self.remark_filter = remark_filter
        self.remark_policy = remark_policy
        self.remark_output = remark_output
        # True while this session owns the engine it enabled.
        self.enabled = False
        self.remarks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._handler: Any = None

    def enable(self) -> None:
        """Enable the engine on the context, unless no filter is set, the
        binding lacks the API, or another owner already enabled it.

        :raises DSLRuntimeError: Unknown policy/format or unwritable output path
        """
        if not self.remark_filter or self.enabled:
            return
        if not remarks_available():
            log().debug(
                "remarks requested [%s] but Context.enable_remarks is not available",
                self.remark_filter,
            )
            return
        if self.context.remarks_enabled:
            # Someone else owns the engine; its remarks reach us as diagnostics.
            return
        policy = _remark_policy(self.remark_policy)
        if policy is None:
            raise DSLRuntimeError(
                "invalid remark configuration: unknown remark policy "
                f"'{self.remark_policy}'; expected 'all' or 'final'",
                context={
                    "filter": self.remark_filter,
                    "policy": self.remark_policy,
                    "output": self.remark_output,
                },
            )
        try:
            if self.remark_output:
                self.context.enable_remarks(
                    policy=policy,
                    output_file=self.remark_output,
                    format=_remark_format(self.remark_output),
                    all_filter=self.remark_filter,
                )
            else:
                self.context.enable_remarks(
                    policy=policy,
                    all_filter=self.remark_filter,
                    callback=self._on_remark,
                )
        except ValueError as exc:
            raise DSLRuntimeError(
                "invalid remark configuration",
                context={
                    "filter": self.remark_filter,
                    "policy": self.remark_policy,
                    "output": self.remark_output,
                },
                cause=exc,
            ) from exc
        self.enabled = True

    def _on_remark(self, remark: Any) -> None:
        """The engine callback: record and render one ``ir.Remark``."""
        record = _remark_record(remark)
        self.remarks.append(record)
        print(_render_remark(record), file=sys.stderr)

    def _handle(self, diagnostic: Any) -> bool:
        """The diagnostic handler: ``emit``-form remarks are recorded (and
        rendered unless streaming to a file) and consumed; errors emitted
        outside the pass manager's own capture (e.g. while finalizing) are
        kept in ``errors`` for the failure report and passed on."""
        if diagnostic.severity == ir.DiagnosticSeverity.REMARK:
            record = _diagnostic_remark_record(diagnostic)
            self.remarks.append(record)
            if not self.remark_output:
                print(_render_remark(record), file=sys.stderr)
            return True
        if diagnostic.severity == ir.DiagnosticSeverity.ERROR:
            self.errors.append(str(diagnostic.message))
        return False

    def finalize(self) -> None:
        """Flush postponed remarks and release the engine."""
        if self.enabled:
            self.context.finalize_remarks()
            self.enabled = False

    def __enter__(self) -> "RemarkSession":
        self.enable()
        self._handler = self.context.attach_diagnostic_handler(self._handle)
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        try:
            self.finalize()
        finally:
            if self._handler is not None:
                self._handler.detach()
                self._handler = None
        return False
