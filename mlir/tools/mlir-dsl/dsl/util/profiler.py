# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Profiling: the compile-phase profiler and the call stopwatch.

Two knobs, two sections. ``<PREFIX>_PROFILE_COMPILER`` profiles the compiler
(how long the DSL takes to turn a function into a module), aggregated per
compile into one report; ``<PREFIX>_JIT_TIME_PROFILING`` is the coarse
stopwatch :func:`timer` that logs the wall time of each wrapped call (the
build, compile and engine steps of ``BaseDSL`` and the argument marshalling
and packed call of the JIT executor) as one line each. Neither profiles the
generated kernel.

The compile profiler has two modes, chosen by the env var value (see
``configure``):

* default: wall time of the ``ast-build`` (with parse/visit split), ``build``,
  ``mlir`` and ``jit`` phases via ``time.perf_counter``. No instrumentation
  beyond the clock reads, so these numbers are trustworthy for fast-or-slow
  comparisons.
* ``deep``: additionally attributes the ``build`` phase with cProfile — a
  per-file self-time table (DSL tracing internals *and* the user/library code
  run during tracing, e.g. schedule validation) plus a per-region
  which-function drill — and captures MLIR's per-pass timing report. Both
  distort the phase times (cProfile overhead; pass timing forces the pass
  manager single-threaded), so deep mode answers *where* time goes, not
  *how much* — the report says so.

MLIR writes the pass-timing report to C-level stderr, which
``contextlib.redirect_stderr`` cannot see, so deep mode captures it by
redirecting fd 2.

The report is bright red and is emitted incrementally: the tracing
(ast-build/build) section is printed the moment the build phase finishes, so a
later failure in the MLIR pipeline or the kernel launch cannot hide it; the
MLIR/JIT section follows at the end of the compile (``finish_compile``), which
runs from the compile driver's ``finally`` so it is emitted even when a
compile has no MLIR phase at all (in-memory cache hit, DRYRUN).

Every entry point is a no-op while the profiler is off. Configuration comes
from ``EnvironmentVarManager`` via ``configure``; this module never reads the
environment itself.
"""

from __future__ import annotations

import atexit
import cProfile
import functools
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, ParamSpec, TextIO, TypeVar

from .logger import log

_P = ParamSpec("_P")
_T = TypeVar("_T")

_enabled = False
_deep = False
_var_name = ""
_output_path: str | None = None
_append = False  # the first report truncates the output file, later ones append

_phase_seconds: defaultdict[str, float] = defaultdict(float)
_phase_counts: Counter[str] = Counter()
_mlir_reports: list[str] = []
_build_stats: dict | None = None
_mlir_capture: Any = None  # open capture context during the MLIR phase, else None
_mlir_started_at = 0.0
# The tracing (ast-build/build) numbers are reported the moment the build phase
# finishes so a later failure (MLIR pipeline or kernel launch) cannot hide them.
# This flag stops ``report_and_reset`` from printing those same phases again.
_tracing_reported = False
# Nesting depth of compiles (begin_compile/finish_compile) and of build
# phases. A compile can start another one while tracing (a traced body may
# compile a callee); only the outermost records and reports, so phase times
# are never counted twice into the same report.
_compile_depth = 0
_build_depth = 0

_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})

# The whole report is wrapped in bright red so it stands out in a wall of test
# output. A single leading code colors every line until the trailing reset
# (color persists across newlines in a terminal).
_RED = "\033[91m"
_RESET = "\033[0m"


def configure(var_name: str, value: str | None) -> None:
    """Apply the ``<PREFIX>_PROFILE_COMPILER`` value read by EnvironmentVarManager.

    Value grammar: off values disable; on values report phase times to stderr;
    ``deep`` / ``deep:<path>`` enable the cProfile and per-pass MLIR drill;
    any other value is a path to write reports to. ``None`` (variable unset)
    leaves the state untouched so one DSL's manager cannot clobber another's
    setting.
    """
    global _enabled, _deep, _var_name, _output_path, _append, _tracing_reported
    if value is None:
        return
    text = value.strip()
    lowered = text.lower()
    _var_name = var_name
    _append = False
    _deep = False
    _output_path = None
    _tracing_reported = False
    if lowered in _OFF_VALUES:
        _enabled = False
    elif lowered in _ON_VALUES:
        _enabled = True
    elif lowered == "deep" or lowered.startswith("deep:"):
        _enabled = True
        _deep = True
        _output_path = text[len("deep:") :].strip() or None
    else:
        _enabled = True
        _output_path = text


def enabled() -> bool:
    """Whether compile profiling is active."""
    return _enabled


def deep() -> bool:
    """Whether the cProfile / per-pass-MLIR attribution mode is active."""
    return _deep


def start(name: str) -> float:
    """Start timing phase ``name``; pass the returned stamp to ``stop``."""
    return time.perf_counter() if _enabled else 0.0


def stop(name: str, started_at: float) -> None:
    """Record the wall time of phase ``name`` since its ``start`` stamp."""
    if not _enabled:
        return
    _phase_seconds[name] += time.perf_counter() - started_at
    _phase_counts[name] += 1


def timed(name: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Decorate a function so every call is recorded as phase ``name``."""

    def decorate(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            started_at = start(name)
            try:
                return fn(*args, **kwargs)
            finally:
                stop(name, started_at)

        return wrapper

    return decorate


def profile_build(build_fn: Callable[[], _T]) -> Callable[[], _T]:
    """Wrap the zero-arg IR-build closure to record it as the ``build`` phase.

    Returns ``build_fn`` unchanged while the profiler is off. In deep mode the
    wrapper also runs the closure under cProfile for the report's
    which-function drill.
    """
    if not _enabled:
        return build_fn

    @functools.wraps(build_fn)
    def profiled_build() -> _T:
        global _build_stats, _build_depth
        # A build can nest (a traced body may start another compile); only the
        # outermost records, so its wall time is not counted twice.
        if _build_depth:
            return build_fn()
        profile = cProfile.Profile() if _deep else None
        started_at = start("build")
        _build_depth += 1
        try:
            return profile.runcall(build_fn) if profile else build_fn()
        finally:
            _build_depth -= 1
            stop("build", started_at)
            if profile is not None:
                profile.create_stats()
                _build_stats = profile.stats
            # Emit the tracing numbers now, before the MLIR pipeline or the
            # kernel launch runs, so a failure there cannot swallow them.
            report_tracing()

    return profiled_build


def begin_mlir_phase() -> None:
    """Start timing the MLIR pass pipeline (and, in deep mode, capture its report).

    Paired with ``end_mlir_phase``. A no-op while the profiler is off or when a
    phase is already open, so the call sites stay flat statements rather than a
    ``with`` block.
    """
    global _mlir_capture, _mlir_started_at
    if not _enabled or _mlir_capture is not None:
        return
    _mlir_capture = _capture_mlir_timing() if _deep else nullcontext()
    _mlir_capture.__enter__()
    _mlir_started_at = time.perf_counter()


def end_mlir_phase() -> None:
    """Finish the MLIR phase started by ``begin_mlir_phase``.

    Idempotent: safe to call twice (success path) or when no phase is open
    (profiler off, or a failure before ``begin_mlir_phase``), so it can also run
    from the caller's ``finally`` to guarantee the captured fd is restored even
    when the pass pipeline raises. Reporting happens at the end of the compile
    (``finish_compile``), after the JIT engine phase that follows the pipeline.
    """
    global _mlir_capture
    if _mlir_capture is None:
        return
    _phase_seconds["mlir"] += time.perf_counter() - _mlir_started_at
    _phase_counts["mlir"] += 1
    capture, _mlir_capture = _mlir_capture, None
    capture.__exit__(None, None, None)


def begin_compile() -> None:
    """Mark the start of one compile; pairs with ``finish_compile``.

    Compiles can nest (a traced body may start another compile); the report
    and reset happen only when the outermost compile finishes, so the nested
    compile's phases fold into the enclosing report.
    """
    global _compile_depth
    if not _enabled:
        return
    _compile_depth += 1


def finish_compile() -> None:
    """Report and reset at the end of the outermost compile.

    Runs from the compile driver's ``finally``, so every compile reports —
    including ones that never reach the MLIR pipeline (in-memory cache hit,
    DRYRUN, a failure while tracing). Without this per-compile boundary,
    phases recorded by such a compile leaked into the next compile's report
    or were cleared unprinted.
    """
    global _compile_depth
    if not _enabled or _compile_depth == 0:
        return
    _compile_depth -= 1
    if _compile_depth == 0:
        report_and_reset()


@contextmanager
def _capture_mlir_timing() -> Iterator[None]:
    """Capture MLIR's pass-timing report from C-level stderr (fd 2).

    The pass manager writes the ``enable_timing`` report straight to the C
    ``stderr`` stream, which Python-level redirection cannot intercept, so
    fd 2 is redirected into a temp file for the duration. Captured output
    that is not part of a timing report (warnings, etc.) is re-emitted.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        try:
            os.dup2(tmp.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            tmp.seek(0)
            text = tmp.read().decode("utf-8", "replace")
            for report in _extract_mlir_timing(text):
                _mlir_reports.append(report)
                text = text.replace(report, "")
            if text.strip():
                sys.stderr.write(text)
                sys.stderr.flush()


# MLIR's report title varies across versions: "Pass execution timing report"
# (upstream) vs "Execution time report".
_MLIR_TIMING_HEADER = re.compile(r"execution tim(?:e|ing) report", re.IGNORECASE)


def _extract_mlir_timing(text: str) -> list[str]:
    """Extract MLIR execution-timing report blocks from captured stderr."""
    lines = text.splitlines()
    reports: list[str] = []
    i = 0
    while i < len(lines):
        if not _MLIR_TIMING_HEADER.search(lines[i]):
            i += 1
            continue
        first = i
        while first > 0 and ("===" in lines[first - 1] or not lines[first - 1].strip()):
            first -= 1
        j, seen_rows, blanks = i, False, 0
        while j < len(lines):
            line = lines[j]
            if (
                re.match(r"\s*[\d.]+\s*\(", line)
                or "Wall Time" in line
                or "Execution Time" in line
            ):
                seen_rows = True
            if not line.strip():
                blanks += 1
                if seen_rows and blanks >= 1 and j > i + 3:
                    break
            else:
                blanks = 0
            j += 1
        reports.append("\n".join(lines[first:j]))
        i = j
    return reports


def _region_of(file_name: str) -> str:
    """Coarse attribution bucket for a cProfile frame, by file location only."""
    if "/_mlir" in file_name or file_name.endswith("_ops_gen.py"):
        return "dsl: mlir-op-construction (nanobind)"
    if file_name.endswith(("inspect.py", "linecache.py")):
        return "python introspection (inspect / linecache)"
    if os.sep.join(("mlir", "dsl", "")) in file_name:
        return f"dsl: {os.path.basename(file_name)}"
    return "user / library code (outside DSL)"


def report_tracing() -> None:
    """Emit the tracing (ast-build/build) numbers the moment the build finishes.

    Called from ``profile_build`` right after the IR trace completes — before
    the MLIR pipeline or the kernel launch runs — so a failure in either cannot
    hide the tracing profile. Idempotent within one compile: the phases printed
    here are skipped by ``report_and_reset`` via ``_tracing_reported``.
    """
    global _tracing_reported
    if not _enabled or _tracing_reported or not _phase_counts.get("build"):
        return
    out, opened = _open_report_output()
    _write_tracing_report(out)
    _close_report_output(out, opened)
    _tracing_reported = True


def report_and_reset() -> None:
    """Write any not-yet-reported phases (tracing fallback + MLIR/JIT), then reset.

    Writes to the ``configure`` output path (truncated on the first report of
    the process, appended thereafter) or to stderr. Runs at the end of every
    compile (``finish_compile``) and is also registered at process exit as a
    fallback for paths that record phases without a compile boundary.
    Tracing is normally emitted by ``report_tracing`` as soon as the build phase
    ends; it is only written here as a fallback for paths that recorded phases
    without going through ``profile_build``.
    """
    global _build_stats, _tracing_reported
    if not _enabled or not _phase_seconds:
        return

    out, opened = _open_report_output()
    # Fallback for paths that recorded a build without going through
    # ``profile_build`` (``report_tracing`` handles the normal case).
    if not _tracing_reported and _phase_counts.get("build"):
        _write_tracing_report(out)
        _tracing_reported = True
    # The MLIR/JIT section always closes the report; under DRYRUN / a failure
    # before the pipeline ran it shows zero-time phases and the "did not run"
    # note.
    _write_mlir_report(out)
    _close_report_output(out, opened)

    _phase_seconds.clear()
    _phase_counts.clear()
    _mlir_reports.clear()
    _build_stats = None
    _tracing_reported = False


_RULE = "=" * 64


def _open_report_output() -> tuple[TextIO, TextIO | None]:
    """Open the report destination: the ``configure`` path, else stderr.

    Returns ``(out, opened)`` where ``opened`` is the file handle to close
    afterwards (``None`` when writing to stderr). The path is truncated on the
    first report of the process and appended to thereafter.
    """
    global _append
    out: TextIO = sys.stderr
    opened: TextIO | None = None
    if _output_path:
        try:
            opened = open(_output_path, "a" if _append else "w", encoding="utf-8")
            _append = True
            out = opened
        except OSError as error:
            out.write(
                f"[{_var_name}] cannot write {_output_path!r} ({error}); using stderr\n"
            )
    return out, opened


def _close_report_output(out: TextIO, opened: TextIO | None) -> None:
    """Flush stderr / close the report file opened by ``_open_report_output``."""
    if opened is not None:
        opened.close()
        sys.stderr.write(f"[{_var_name}] compile profile written to {_output_path}\n")
        sys.stderr.flush()
    else:
        out.flush()


def _write_tracing_report(out: TextIO) -> None:
    """Write the red-colored ast-build/build section (plus deep build drill)."""
    out.write(_RED)
    out.write(f"\n{_RULE}\n  DSL compile profile — tracing  ({_var_name})\n{_RULE}\n")
    _emit_phase(
        out, "ast-build", f"({_phase_counts['ast-build']} functions; subset of build)"
    )
    _emit_phase(out, "build", "(IR trace + finalize; includes ast-build)")
    pure = _phase_seconds["build"] - _phase_seconds["ast-build"]
    out.write(f"      pure trace (build - ast-build): {pure * 1e3:10.1f} ms\n")
    if _deep:
        out.write(
            "      (deep mode: cProfile active — times inflated vs default mode)\n"
        )
    out.write(_RULE + "\n")
    if _deep and _build_stats:
        _write_file_breakdown(out, _build_stats)
        _write_build_drill(out, _build_stats)
    out.write(_RESET)
    out.flush()


def _write_mlir_report(out: TextIO) -> None:
    """Write the red-colored MLIR/JIT-phase section (plus deep per-pass timing)."""
    out.write(_RED)
    out.write(f"\n{_RULE}\n  DSL compile profile — mlir  ({_var_name})\n{_RULE}\n")
    note = "(pass pipeline; single-threaded for timing)" if _deep else "(pass pipeline)"
    _emit_phase(out, "mlir", note)
    _emit_phase(out, "jit", "(execution engine + entry lookup)")
    out.write(_RULE + "\n")
    if _deep:
        _write_mlir_reports(out)
    out.write(_RESET)
    out.flush()


def _emit_phase(out: TextIO, name: str, note: str) -> None:
    """Write one phase row plus any ``name/<sub>`` breakdown rows."""
    out.write(f"  {name:<10}: {_phase_seconds[name] * 1e3:10.1f} ms   {note}\n")
    for key in sorted(_phase_seconds):
        if key.startswith(name + "/"):
            sub = key.split("/", 1)[1]
            out.write(f"      {sub:<20}{_phase_seconds[key] * 1e3:10.1f} ms\n")


def _file_label(file_name: str) -> str:
    """Short, disambiguated label for a cProfile source file.

    ``~`` is cProfile's marker for built-in / C functions. Other files show
    their last two path components (so two ``typing.py`` at different paths
    don't collide) tagged ``dsl`` or ``user`` per :func:`_region_of`, matching
    the report's goal of surfacing both DSL tracing and user/library code.
    """
    if file_name == "~":
        return "[C   ] <built-in / C methods>"
    tag = "dsl " if _region_of(file_name).startswith("dsl") else "user"
    short = os.sep.join(file_name.split(os.sep)[-2:])
    return f"[{tag}] {short}"


def _write_file_breakdown(out: TextIO, stats: dict, limit: int = 15) -> None:
    """Write the per-file self-time table for the build phase.

    Aggregates the cProfile self time of ``stats`` by source file so the report
    shows *where* the build wall time goes across every layer — DSL tracing
    internals AND the user/library code executed during tracing (e.g.
    schedule validation), which the per-region drill below lumps into a
    single "outside DSL" bucket.
    """
    self_by_file: defaultdict[str, float] = defaultdict(float)
    for (file_name, _line, _func), (_, _, self_time, _, _) in stats.items():
        self_by_file[file_name] += self_time
    total = sum(self_by_file.values()) or 1.0
    out.write(
        "  TOP FILES by self time (build phase — DSL tracing + user/library code;\n"
        "  cProfile self time, so absolute values are inflated vs default mode):\n"
    )
    for file_name, self_time in sorted(self_by_file.items(), key=lambda kv: -kv[1])[
        :limit
    ]:
        share = 100 * self_time / total
        out.write(
            f"    {self_time * 1e3:9.1f} ms  {share:5.1f}%  {_file_label(file_name)}\n"
        )
    out.write(_RULE + "\n")


def _write_build_drill(out: TextIO, stats: dict) -> None:
    """Write the per-region function drill of the build phase from ``stats``.

    The first table groups self time by :func:`_region_of` bucket with the
    hottest functions of each; the second lists the top functions by
    cumulative time, which surfaces orchestrators whose own code is cheap.
    """
    out.write(
        "  WHICH FUNCTION is slow in 'build'  (cProfile self time"
        " — build time above is inflated):\n"
    )
    # cProfile stats: {(file, line, func): (prim_calls, calls, self, cum, callers)}
    regions: dict[str, list[tuple[float, int, str]]] = {}
    for (file_name, _line, func_name), (
        _,
        calls,
        self_time,
        _,
        _,
    ) in stats.items():
        label = f"{os.path.basename(file_name)}:{func_name}"
        regions.setdefault(_region_of(file_name), []).append((self_time, calls, label))
    total = sum(t for rows in regions.values() for t, _, _ in rows) or 1.0
    by_time = sorted(regions.items(), key=lambda kv: -sum(t for t, _, _ in kv[1]))
    for region, rows in by_time:
        region_time = sum(t for t, _, _ in rows)
        share = 100 * region_time / total
        out.write(f"    {region:<34}{region_time * 1e3:9.1f} ms  {share:5.1f}%\n")
        for self_time, calls, label in sorted(rows, reverse=True)[:6]:
            if self_time * 1e3 < 1.0:
                break
            out.write(
                f"        {self_time * 1e3:8.1f} ms  {calls:>10,} calls  {label}\n"
            )
    out.write(_RULE + "\n")

    # Cumulative view (function + callees): surfaces orchestrators whose own
    # code is cheap but whose callees are not.
    out.write("  TOP BY CUMULATIVE time (function + callees; finds orchestrators):\n")
    by_cumulative = sorted(stats.items(), key=lambda kv: -kv[1][3])[:15]
    for rank, (
        (file_name, _line, func_name),
        (prim_calls, _, _, cum_time, _),
    ) in enumerate(by_cumulative):
        if rank > 0 and cum_time * 1e3 < 5.0:
            break
        tag = "dsl" if _region_of(file_name).startswith("dsl") else "lib"
        out.write(
            f"    {cum_time * 1e3:9.1f} ms  {prim_calls:>9,} calls  "
            f"[{tag}] {os.path.basename(file_name)}:{func_name}\n"
        )
    out.write(_RULE + "\n")


def _write_mlir_reports(out: TextIO) -> None:
    """Write the captured MLIR pass-timing reports, or the did-not-run note."""
    if _mlir_reports:
        out.write("  MLIR per-pass timing (--mlir-timing):\n")
        for report in _mlir_reports:
            out.write(report + "\n")
    else:
        out.write("  (no MLIR per-pass report captured — pass pipeline did not run,\n")
        out.write("   e.g. DRYRUN; ast-build/build above are still valid.)\n")
    out.write(_RULE + "\n")


# =============================================================================
# The call stopwatch (``<PREFIX>_JIT_TIME_PROFILING``)
# =============================================================================


def timer(*dargs: Any, enable: bool = True) -> Any:
    """Log the wall time of every call of the decorated function, in microseconds.

    ``BaseDSL`` wraps its build, compile and engine steps with it and the JIT
    executor its marshalling and packed call, when ``JIT_TIME_PROFILING`` is
    set; each call is one ``[JIT-TIMER]`` info line.
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enable:
                return func(*args, **kwargs)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            spend_us = (time.perf_counter() - start) * 1e6
            label = getattr(func, "__name__", None) or f"C API Function: {func}"
            log().info("[JIT-TIMER] %s | Execution Time: %.2f us", label, spend_us)
            return result

        return func_wrapper

    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    return decorator


# Fallback for phases recorded outside a begin/finish_compile boundary.
atexit.register(report_and_reset)
