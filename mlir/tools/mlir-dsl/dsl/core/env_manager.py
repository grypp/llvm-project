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

A malformed value is the user's mistake: the typed readers raise
:class:`DSLUserCodeError` ``CONFIG_INVALID`` naming the variable. A malformed
declaration (an annotation without a reader, a bad default) is the DSL
author's and raises :class:`DSLRuntimeError`.
"""

import inspect
import os
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Union, get_args, get_origin

from .common import DSLRuntimeError, DSLUserCodeError
from .diagnostics import DiagId


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
        DSLUserCodeError: ``CONFIG_INVALID`` if the variable is set to any
            other value.
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
    accepted = ", ".join(sorted(_BOOL_TRUE_VALUES) + sorted(_BOOL_FALSE_VALUES))
    raise DSLUserCodeError(
        DiagId.CONFIG_INVALID,
        var=var_name,
        detail=f"`{raw}` is not a boolean; expected one of {accepted} "
        f"(case-insensitive), or empty for the default `{default_value!r}`",
    )


class EnvironmentVarManager(EnvVarSpec):
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
    - [DSL_NAME]_SHOW_STACKTRACE: Show full stack traces on failure (default: DEBUG)
    - [DSL_NAME]_DEBUGINFO: Attach source locations to every op (default: DEBUG)
    - [DSL_NAME]_VERIFY_TRACE: Verify every op as it is built while tracing (default: False)
    - [DSL_NAME]_LOG_LEVEL: Logging level to set, for LOG_TO_CONSOLE or LOG_TO_FILE (default: 1).
    - [DSL_NAME]_DRYRUN: Generates IR only (default: False)
    - [DSL_NAME]_ARCH: Target chip for the plugins that need one, read by the gpu kernels plugin only when a launch is compiled (default: None, no target)
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
    - [DSL_NAME]_REMARKS: Regular expression over remark categories to emit (default: "", remarks off)
    - [DSL_NAME]_REMARKS_POLICY: Remark policy, "all" or "final" (default: "all")
    - [DSL_NAME]_REMARKS_OUTPUT: File the remarks stream to; ".yaml" or ".bitstream" picks the format (default: "")
    """

    # Master switch for DSL developers: raises the default of a curated set of
    # diagnostic settings below, each still overridable by its own variable.
    debug: bool = env_var("DEBUG", affects_compile=True, default=False)
    print_ir: bool = env_var("PRINT_IR", affects_compile=False, default=False)
    # Emits an ir.Location per op; keyed because it changes the module.
    debuginfo: bool = env_var(
        "DEBUGINFO", affects_compile=True, default=lambda mgr: mgr.debug
    )
    # Governs whether results are cached, not what is compiled.
    no_cache: bool = env_var("NO_CACHE", affects_compile=False, default=False)
    dryrun: bool = env_var("DRYRUN", affects_compile=True, default=False)
    warnings_ignore: bool = env_var(
        "WARNINGS_IGNORE", affects_compile=False, default=False
    )
    pipeline: str | None = env_var("PIPELINE", affects_compile=True, default=None)

    def __init__(self, prefix: str = "DSL") -> None:
        self._apply_env_var_spec(prefix)
        self.prefix = prefix
