# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Environment-variable configuration of a DSL instance.

An :class:`EnvironmentVarManager` reads the ``{prefix}_*`` variables of one
DSL once, at construction, and holds them as typed attributes. Each setting is
declared in the class body as ``attribute: type = env_var("SUFFIX", ...)``;
the annotation picks the reader (bool, int, ``int | None`` or str), and
``affects_compile`` says whether the setting enters the JIT cache key
(:meth:`EnvVarSpec.cache_key_str`). A sub-DSL subclasses the manager and
redeclares or adds settings; ``BaseDSL`` constructs one per instance under the
DSL's name as prefix.

Malformed values raise :class:`DSLRuntimeError` from the typed readers: they
describe the process environment, not the author's kernel.
"""

import inspect
import os
import types
import warnings
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Callable, Union, get_args, get_origin

from ..util import phase_profiler
from ..util.logger import setup_log
from .common import DSLRuntimeError, DSLWarning


@dataclass(frozen=True)
class EnvVar:
    """One setting on an :class:`EnvironmentVarManager`.

    Written in the class body as ``attribute: type = env_var(...)``, so the
    attribute, its type and where its value comes from are stated together and
    exactly once. ``source`` is either the environment variable's suffix -- the
    value is read from ``{prefix}_{source}`` -- or a function of the manager
    that computes it, for a setting with no variable of its own. How to read
    the environment follows from the annotation.

    ``affects_compile`` says whether the setting is part of the JIT cache key;
    it is required, so a setting cannot be added without answering.
    """

    source: str | Callable[[Any], Any]
    affects_compile: bool = field(kw_only=True)
    default: Any = None
    read_as: str | None = None
    attribute: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) and (
            self.default is not None or self.read_as is not None
        ):
            raise DSLRuntimeError(
                "A computed setting takes neither `default` nor `read_as`: it has no "
                "environment variable to fall back from or to be read under."
            )

    def __set_name__(self, owner: type, name: str) -> None:
        object.__setattr__(self, "attribute", name)

    @property
    def key_name(self) -> str:
        return self.read_as or self.attribute

    def resolve(
        self, manager: Any, prefix: str, parser: Callable[..., Any] | None
    ) -> Any:
        """Value this setting takes for ``manager``."""
        if not isinstance(self.source, str):
            return self.source(manager)
        if parser is None:
            raise DSLRuntimeError(
                f"Setting `{self.attribute}` is read from `{prefix}_{self.source}` "
                "but no environment-variable reader was supplied for it."
            )
        # A default that reads other settings is written as a function of the
        # manager. There is no ambiguity to resolve: parser_for_type admits only
        # bool, int and str, so a callable is never itself a legitimate default.
        default = self.default(manager) if callable(self.default) else self.default
        return parser(f"{prefix}_{self.source}", default)


def env_var(
    source: str | Callable[[Any], Any],
    *,
    affects_compile: bool,
    default: Any = None,
    read_as: str | None = None,
) -> Any:
    """Declare a setting, as the default of an annotated class attribute.

    ``source`` is the variable's suffix -- the value is read from
    ``{prefix}_{source}`` -- or a function of the manager, for a setting
    computed rather than read. Either way it may use anything declared above.

    Returns ``Any``, the way :func:`dataclasses.field` does, so the declaration
    can carry the attribute's real type. ``default`` may itself be a function
    of the manager when it depends on a setting declared above.
    """
    return EnvVar(
        source,
        affects_compile=affects_compile,
        default=default,
        read_as=read_as,
    )


def _annotated_type(owner: type, attr: str) -> Any:
    """Declared type of ``attr``, searched up ``owner``'s MRO."""
    for klass in owner.__mro__:
        annotation = inspect.get_annotations(klass).get(attr)
        if annotation is not None:
            return annotation
    raise DSLRuntimeError(
        f"{owner.__name__}.{attr} has no type annotation, so the parser for its "
        f"environment variable cannot be determined. Annotate it on the class."
    )


def parser_for_type(annotation: Any) -> Callable[..., Any]:
    """Return the environment reader for a setting's annotation.

    ``bool`` reads with :func:`get_bool_env_var`, ``int`` with
    :func:`get_int_env_var`, ``int | None`` with
    :func:`get_int_or_none_env_var` and ``str`` / ``str | None`` with
    :func:`get_str_env_var`; any other annotation is a declaration error.
    """
    base, optional = annotation, False
    if get_origin(annotation) in (types.UnionType, Union):
        members = [a for a in get_args(annotation) if a is not type(None)]
        optional = len(members) < len(get_args(annotation))
        if len(members) == 1:
            base = members[0]
    if base is bool:
        return get_bool_env_var
    if base is int:
        return get_int_or_none_env_var if optional else get_int_env_var
    if base is str:
        return get_str_env_var
    raise DSLRuntimeError(
        f"No environment-variable reader for {annotation!r}. Give the setting a "
        f"bool, int or str annotation, or compute it from the manager instead."
    )


class EnvVarSpec:
    """Turns the settings declared in a class body into a spec.

    ``_ENV_VAR_SPEC`` is what a class declares plus everything it inherits, in
    declaration order. A subclass redeclaring an attribute replaces the
    inherited declaration and keeps its position.
    """

    _ENV_VAR_SPEC: tuple[EnvVar, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        composed = {entry.attribute: entry for entry in cls._ENV_VAR_SPEC}
        composed.update(
            {v.attribute: v for v in vars(cls).values() if isinstance(v, EnvVar)}
        )
        cls._ENV_VAR_SPEC = tuple(composed.values())

    def _apply_env_var_spec(self, prefix: str) -> None:
        """Read every declared setting under ``prefix`` onto ``self``.

        Settings are resolved in declaration order, so a computed setting or
        a callable default may read the settings declared above it.
        """
        for entry in type(self)._ENV_VAR_SPEC:
            parser = (
                parser_for_type(_annotated_type(type(self), entry.attribute))
                if isinstance(entry.source, str)
                else None
            )
            setattr(self, entry.attribute, entry.resolve(self, prefix, parser))

    def cache_key_str(self) -> str:
        """Return the settings that are part of the JIT cache key.

        One ``key=value;`` item per ``affects_compile`` setting whose value is
        not ``None``, ordered by key name so the string is stable across
        declaration order and subclassing.
        """
        rendered = []
        for entry in sorted(self._ENV_VAR_SPEC, key=lambda e: e.key_name):
            if not entry.affects_compile:
                continue
            value = getattr(self, entry.key_name)
            if value is None:
                continue
            rendered.append(f"{entry.key_name}={_render_cache_key_value(value)};")
        return "".join(rendered)


def _render_cache_key_value(value: object) -> str:
    """``repr`` of a cache-key value, deterministic for sets.

    A computed setting may hold a set; its iteration order is randomized per
    process, so it is rendered as a sorted tuple.
    """
    if isinstance(value, (set, frozenset)):
        return repr(tuple(sorted(value, key=repr)))
    return repr(value)


# =============================================================================
# Environment Variable Helpers
# =============================================================================


def get_str_env_var(var_name: str, default_value: str | None = None) -> str | None:
    """
    Get the string value of an environment variable.
    """
    value = os.getenv(var_name)
    return value if value is not None else default_value


_BOOL_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def get_bool_env_var(var_name: str, default_value: bool = False) -> bool:
    """
    Get the value of a boolean environment variable.

    Recognized values (case-insensitive, surrounding whitespace ignored):
      * Truthy:   ``1``, ``true``, ``yes``, ``on``
      * Falsy:    ``0``, ``false``, ``no``, ``off``

    An unset variable, or one whose value is empty (or whitespace only),
    returns ``default_value``.

    Raises:
        DSLRuntimeError: if the variable is set to any other value.
    """
    raw = get_str_env_var(var_name)
    if raw is None:
        return default_value
    normalized = raw.strip().lower()
    if normalized == "":
        return default_value
    if normalized in _BOOL_TRUE_VALUES:
        return True
    if normalized in _BOOL_FALSE_VALUES:
        return False
    raise DSLRuntimeError(
        f"Invalid value for environment variable {var_name}={raw!r}. "
        f"Expected a boolean (case-insensitive): "
        f"{sorted(_BOOL_TRUE_VALUES) + sorted(_BOOL_FALSE_VALUES)} "
        f"or empty/unset to use the default ({default_value!r})."
    )


def get_int_env_var(var_name: str, default_value: int = 0) -> int:
    """
    Get the value of an integer environment variable.

    Surrounding whitespace is ignored. An unset variable or one with an
    empty value returns ``default_value``. Negative integers (e.g.
    ``-5``) are accepted.

    Raises:
        DSLRuntimeError: if the variable is set to a value that is not a
            valid base-10 integer.
    """
    raw = get_str_env_var(var_name)
    if raw is None:
        return default_value
    stripped = raw.strip()
    if stripped == "":
        return default_value
    try:
        return int(stripped)
    except ValueError:
        raise DSLRuntimeError(
            f"Invalid value for environment variable {var_name}={raw!r}. "
            f"Expected a base-10 integer, or empty/unset to use the "
            f"default ({default_value!r})."
        ) from None


def get_int_or_none_env_var(
    var_name: str, default_value: int | None = None
) -> int | None:
    """
    Get the value of an integer-or-``None`` environment variable.

    Recognized values (case-insensitive, surrounding whitespace ignored):
      * ``"none"``                       returns ``None``
      * any base-10 integer literal      returns that integer (negatives accepted)

    An unset variable or one with an empty value returns ``default_value``.

    Raises:
        DSLRuntimeError: if the variable is set to anything else.
    """
    raw = get_str_env_var(var_name)
    if raw is None:
        return default_value
    normalized = raw.strip().lower()
    if normalized == "":
        return default_value
    if normalized == "none":
        return None
    try:
        return int(normalized)
    except ValueError:
        raise DSLRuntimeError(
            f"Invalid value for environment variable {var_name}={raw!r}. "
            f"Expected a base-10 integer, the literal 'none', or "
            f"empty/unset to use the default ({default_value!r})."
        ) from None


def has_env_var(var_name: str) -> bool:
    """
    Check if an environment variable is set.
    """
    return os.getenv(var_name) is not None


class LogEnvironmentManager(EnvVarSpec):
    """The logging settings, applied to the DSL logger at construction.

    Reads ``{prefix}_LOG_TO_CONSOLE``, ``{prefix}_LOG_TO_FILE`` (written to
    ``{prefix}.log``), ``{prefix}_LOG_LEVEL`` and ``{prefix}_JIT_TIME_PROFILING``,
    then calls :func:`setup_log`. Profiling needs a console to report to, so
    it opens one at INFO level when the user enabled none. A ``LOG_LEVEL`` with
    no sink enabled is reported as a warning, since it would have no effect.
    """

    jit_time_profiling: bool = env_var(
        "JIT_TIME_PROFILING", affects_compile=False, default=False
    )
    log_to_console: bool = env_var(
        "LOG_TO_CONSOLE", affects_compile=False, default=False
    )
    log_to_file: bool = env_var("LOG_TO_FILE", affects_compile=False, default=False)
    log_level: int = env_var("LOG_LEVEL", affects_compile=False, default=1)

    def __init__(self, prefix: str = "DSL") -> None:
        self.prefix = prefix

        self._apply_env_var_spec(prefix)

        # Profiling opens the console on its own only when the user did not.
        console_for_profiling = not self.log_to_console and self.jit_time_profiling
        if console_for_profiling:
            self.log_to_console = True
            self.log_level = 20  # info level

        setup_log(
            prefix,
            self.log_to_console,
            self.log_to_file,
            f"{prefix}.log",
            self.log_level,
        )

        if (
            has_env_var(f"{prefix}_LOG_LEVEL")
            and not self.log_to_console
            and not self.log_to_file
        ):
            # setup_log has just disabled the DSL logger, so the notice cannot
            # go through it; a DSLWarning reaches the user instead.
            warnings.warn(
                DSLWarning(
                    f"{prefix}_LOG_LEVEL was set, but neither logging to file "
                    f"({prefix}_LOG_TO_FILE) nor logging to console "
                    f"({prefix}_LOG_TO_CONSOLE) is enabled, so it has no effect.",
                    suggestion=[
                        f"Set {prefix}_LOG_TO_CONSOLE=1 or {prefix}_LOG_TO_FILE=1 "
                        "to see the log.",
                        f"Unset {prefix}_LOG_LEVEL if logging is not wanted.",
                    ],
                ),
                stacklevel=3,
            )


class EnvironmentVarManager(LogEnvironmentManager):
    """Manages environment variables for configuration options.

    Printing options:
    - [DSL_NAME]_LOG_TO_CONSOLE: Print logging to stderr (default: False)
    - [DSL_NAME]_PRINT_IR: Print generated IR (default: False)
    - [DSL_NAME]_PRINT_IR_AFTER_PASSES: Print the IR after applying the given passes to stderr (default: ""). Example: PRINT_IR_AFTER_PASSES="canonicalize,cse"

    File options:
    - [DSL_NAME]_CACHE_DIR: Root for IR dumps and the file cache (default: unset,
      meaning $TMPDIR/<user>/<dsl_name lower>_cache)
    - [DSL_NAME]_LOG_TO_FILE: Store all logging into [DSL_NAME].log (default: False)
    - [DSL_NAME]_KEEP_IR: Save the generated IR into a file under CACHE_DIR (default: False)
    - [DSL_NAME]_KEEPIR_AFTER_PASSES: Save generated IR after applying the passes into a file (default: ""). Example: KEEPIR_AFTER_PASSES="canonicalize,cse"

    Other options:
    - [DSL_NAME]_DEBUG: Master debug switch for DSL developers (default: False).
      When True, raises the default of DEBUGINFO and SHOW_STACKTRACE. These
      defaults remain independently overridable by their own env vars.
    - [DSL_NAME]_SHOW_STACKTRACE: Show full stack traces on failure (default: False)
    - [DSL_NAME]_DEBUGINFO: Attach source locations to every op (default: DEBUG)
    - [DSL_NAME]_VERIFY_TRACE: Verify every op as it is built while tracing (default: False)
    - [DSL_NAME]_LOG_LEVEL: Logging level to set, for LOG_TO_CONSOLE or LOG_TO_FILE (default: 1).
    - [DSL_NAME]_DRYRUN: Generates IR only (default: False)
    - [DSL_NAME]_ARCH: GPU architecture, e.g. "sm_90a" (default: None, no GPU target)
    - [DSL_NAME]_AST_PREPROCESSOR: Run the AST preprocessor on decorated functions (default: True)
    - [DSL_NAME]_WARNINGS_IGNORE: Ignore warnings (default: False)
    - [DSL_NAME]_JIT_TIME_PROFILING: Whether or not to profile the IR generation/compilation/execution time (default: False)
    - [DSL_NAME]_ENABLE_PASS_PROFILING: Print the pass manager's timing report (default: False)
    - [DSL_NAME]_PROFILE_COMPILER: Profile the DSL compiler itself, not the generated
        kernel: reports ast-build/build/mlir phase wall times. "deep" additionally runs
        a cProfile drill-down of the build phase and per-pass MLIR timing (inflates the
        phase times). Any other non-switch value is a file path for the report
        (default: off)
    - [DSL_NAME]_NO_CACHE: Disable JIT cache (default: False)
    - [DSL_NAME]_DISABLE_FILE_CACHING: Disable file caching (default: False)
    - [DSL_NAME]_JIT_CACHE_MAX_ELEMS: Capacity of the in-memory JIT cache, LRU eviction (default: unlimited)
    - [DSL_NAME]_LIBS: Path to dependent shared libraries (default: None)
    - [DSL_NAME]_LOC_TRACEBACKS: Maximum depth of location tracebacks (default: 0)
    - [DSL_NAME]_PIPELINE: MLIR pipeline, replacing the composed default (default: None)
    - [DSL_NAME]_COMPILER_OPT: Compact compiler option string handed to the compiler (default: "")
    - [DSL_NAME]_ENABLE_TVM_FFI: Also export compiled functions under the TVM-FFI ABI (default: False)
    - [DSL_NAME]_REMARKS: Regular expression over remark categories to emit (default: "", remarks off)
    - [DSL_NAME]_REMARKS_POLICY: Remark policy, "all" or "final" (default: "all")
    - [DSL_NAME]_REMARKS_OUTPUT: File the remarks stream to; ".yaml" or ".bitstream" picks the format (default: "")
    """

    # Master switch for DSL developers: raises the default of a curated set of
    # diagnostic settings below, each still overridable by its own variable.
    debug: bool = env_var("DEBUG", affects_compile=True, default=False)
    print_ir: bool = env_var("PRINT_IR", affects_compile=False, default=False)
    # Selects between a full traceback and the formatted message on the exception
    # path.
    show_stacktrace: bool = env_var(
        "SHOW_STACKTRACE", affects_compile=False, default=lambda mgr: mgr.debug
    )
    enable_pass_profiling: bool = env_var(
        "ENABLE_PASS_PROFILING", affects_compile=False, default=False
    )
    ast_preprocessor: bool = env_var(
        "AST_PREPROCESSOR", affects_compile=True, default=True
    )
    # Emits an ir.Location per op; keyed because it changes the module.
    debuginfo: bool = env_var(
        "DEBUGINFO", affects_compile=True, default=lambda mgr: mgr.debug
    )
    # Verifies each op as dsl_user_op builds it, so a malformed op is reported
    # at the Python line that built it rather than by the pass manager.
    verify_trace: bool = env_var("VERIFY_TRACE", affects_compile=False, default=False)
    # Governs whether results are cached, not what is compiled.
    no_cache: bool = env_var("NO_CACHE", affects_compile=False, default=False)
    # Root for IR dumps and the file cache.
    cache_dir: str | None = env_var("CACHE_DIR", affects_compile=False, default=None)
    # Saves the raw IR before any passes; dumped from build_module, which runs
    # before the cache is consulted, so it cannot be served stale.
    keep_ir: bool = env_var("KEEP_IR", affects_compile=False, default=False)
    # The one dump that is not free of the artifact: it runs its pipeline on the
    # module itself rather than a clone, and does so after the hash is taken, so
    # what gets cached under that hash is the mutated module.
    keep_ir_after_passes: str = env_var(
        "KEEPIR_AFTER_PASSES", affects_compile=True, default=""
    )
    # Runs its pipeline on a clone, leaving the module untouched.
    print_ir_after_passes: str = env_var(
        "PRINT_IR_AFTER_PASSES", affects_compile=False, default=""
    )
    # All three reach Compiler.compile; keyed because a cache hit emits no
    # remarks.
    remarks: str = env_var("REMARKS", affects_compile=True, default="")
    remarks_policy: str = env_var("REMARKS_POLICY", affects_compile=True, default="all")
    remarks_output: str = env_var("REMARKS_OUTPUT", affects_compile=True, default="")
    dryrun: bool = env_var("DRYRUN", affects_compile=True, default=False)
    # Stored under _arch because the public spelling belongs to the arch property;
    # read_as sends the key through that property.
    _arch: str | None = env_var(
        "ARCH", affects_compile=True, default=None, read_as="arch"
    )
    warnings_ignore: bool = env_var(
        "WARNINGS_IGNORE", affects_compile=False, default=False
    )
    # Governs whether results are cached, not what is compiled.
    disable_file_caching: bool = env_var(
        "DISABLE_FILE_CACHING", affects_compile=False, default=False
    )
    # Capacity of the in-memory JIT cache; None is unlimited, 0 disables it.
    jit_cache_max_elems: int | None = env_var(
        "JIT_CACHE_MAX_ELEMS", affects_compile=False, default=None
    )
    # Export every compiled function under the TVM-FFI ABI as well (the
    # ``tvm_ffi`` plugin); keyed because it adds a function to the module.
    enable_tvm_ffi: bool = env_var(
        "ENABLE_TVM_FFI", affects_compile=True, default=False
    )
    compiler_opt: str = env_var("COMPILER_OPT", affects_compile=True, default="")
    pipeline: str | None = env_var("PIPELINE", affects_compile=True, default=None)
    # MLIR runtime libraries linked by the JIT.
    shared_libs: str | None = env_var("LIBS", affects_compile=True)
    loc_tracebacks: int = env_var("LOC_TRACEBACKS", affects_compile=True, default=0)

    def __init__(self, prefix: str = "DSL") -> None:
        super().__init__(prefix)

        phase_profiler.configure(
            f"{prefix}_PROFILE_COMPILER", get_str_env_var(f"{prefix}_PROFILE_COMPILER")
        )

    @property
    def missing_shared_libs_message(self) -> str:
        """Explain how to configure this DSL's runtime libraries."""
        return (
            f"{self.prefix}_LIBS environment variable is not set. Set "
            f"{self.prefix}_LIBS explicitly for this DSL runtime."
        )

    @property
    def arch(self) -> str | None:
        """GPU architecture from ``{prefix}_ARCH``, or ``None`` when it is unset.

        There is no detection: ``None`` means no GPU target, and a kernel
        launch reports it naming the variable.
        """
        return self._arch

    @arch.setter
    def arch(self, value: str) -> None:
        self._arch = value

    @arch.deleter
    def arch(self) -> None:
        """Reset to the construction-time state (``{prefix}_ARCH``), so
        ``mock.patch.object`` teardown restores the default."""
        self._arch = get_str_env_var(f"{self.prefix}_ARCH")


# These variables are read directly in ``__init__``, so their suffixes are not
# discoverable from string-backed ``EnvVar`` entries.
_COMPUTED_PREFIX_ENV_VAR_SUFFIXES: frozenset[str] = frozenset({"PROFILE_COMPILER"})


@cache
def _prefixed_env_var_suffixes() -> tuple[str, ...]:
    """Return suffixes read under an ``EnvironmentVarManager`` prefix."""
    suffixes = {
        entry.source
        for entry in EnvironmentVarManager._ENV_VAR_SPEC
        if isinstance(entry.source, str)
    }
    suffixes.update(_COMPUTED_PREFIX_ENV_VAR_SUFFIXES)
    return tuple(sorted(suffixes))
