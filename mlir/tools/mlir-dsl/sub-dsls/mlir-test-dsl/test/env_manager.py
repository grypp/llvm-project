# RUN: env PYTHONUNBUFFERED=1 MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_PIPELINE=canonicalize,cse MLIR_DSL_PROFILE_COMPILER=1 %PYTHON %s 2>&1 | FileCheck %s
# The environment manager: one manager per prefix, every
# `{prefix}_{SUFFIX}` read once at construction with the reader its annotation
# selects; the documented defaults; the `affects_compile` settings forming the
# JIT cache key; the DEBUG master switch; prefix isolation; malformed values
# rejected as `CONFIG_INVALID` (a DSLUserCodeError naming the variable) and
# malformed declarations as DSLRuntimeError; the `arch` property; sub-DSL
# EnvVarSpec composition; PROFILE_COMPILER and the logging
# settings. The RUN line configures the DSL's own MLIR_DSL prefix.
import logging
import os
import tempfile
import warnings
from typing import Optional
from unittest import mock

import mlir.mlir_dsl as m
from mlir.dsl.core.common import DSLRuntimeError, DSLUserCodeError, DSLWarning
from mlir.dsl.core.env_manager import (
    EnvironmentVarManager,
    EnvVarSpec,
    _prefixed_env_var_suffixes,
    _render_cache_key_value,
    env_var,
    get_bool_env_var,
    get_int_env_var,
    get_int_or_none_env_var,
    get_str_env_var,
    parser_for_type,
)
from mlir.dsl.util import profiler
from mlir.dsl.util.logger import log

os.chdir(tempfile.mkdtemp())  # LOG_TO_FILE writes <prefix>.log into the cwd
SPEC = EnvironmentVarManager._ENV_VAR_SPEC
GROUPS = [
    ("jit_time_profiling", "log_to_console", "log_to_file", "log_level"),
    ("debug", "debuginfo", "show_stacktrace", "print_ir", "verify_trace", "dryrun"),
    ("ast_preprocessor", "enable_pass_profiling", "warnings_ignore"),
    ("no_cache", "cache_dir", "disable_file_caching", "jit_cache_max_elems"),
    ("keep_ir", "keep_ir_after_passes", "print_ir_after_passes", "loc_tracebacks"),
    ("remarks", "remarks_policy", "remarks_output"),
    ("arch", "pipeline", "shared_libs", "enable_tvm_ffi"),
]


def clear(prefix):
    for name in [k for k in os.environ if k.startswith(prefix + "_")]:
        del os.environ[name]


def parse(table):
    """`NAME=value` words of a table into a dict."""
    return dict(word.split("=", 1) for word in table.split())


def manager(prefix, **vars):
    """A fresh manager for `prefix` with exactly `vars` set under it."""
    clear(prefix)
    os.environ.update({f"{prefix}_{k}": v for k, v in vars.items()})
    return EnvironmentVarManager(prefix)


def key(prefix, **vars):
    return manager(prefix, **vars).cache_key_str()


def show(env):
    """Every setting of `env`, grouped, with repr so the type is visible."""
    for names in GROUPS:
        print(" ".join(f"{n}={getattr(env, n)!r}" for n in names))


def read(reader, value, *args):
    """`reader` applied to T_VAR holding `value` (None: unset)."""
    if value is None:
        os.environ.pop("T_VAR", None)
    else:
        os.environ["T_VAR"] = value
    try:
        return repr(reader("T_VAR", *args))
    except DSLUserCodeError as e:
        return e.diag_id.name


def reads(reader, values, *args):
    return [read(reader, v, *args) for v in values]


def bad(suffix, value):
    """Construct a manager with one malformed variable; print the outcome."""
    try:
        manager("T_BAD", **{suffix: value})
    except (DSLRuntimeError, DSLUserCodeError) as e:
        print(f"{suffix}={value!r}:", e.message)
    except Exception as e:  # noqa: BLE001 -- the test reports, not raises
        print(f"{suffix}={value!r}: BUILTIN {type(e).__name__}")
    else:
        print(f"{suffix}={value!r}: accepted")


# --- The DSL's own prefix, from the RUN line --------------------------------
env = EnvironmentVarManager("MLIR_DSL")
print("MLIR_DSL:", env.dryrun, env.debuginfo, env.pipeline, env.arch, env.print_ir)
# CHECK:      MLIR_DSL: True True canonicalize,cse None False
print("key:", env.cache_key_str())
# CHECK-NEXT: key: ast_preprocessor=True;{{.*}}debuginfo=True;dryrun=True;{{.*}}pipeline='canonicalize,cse';
# The DSL instance reads the same prefix; PIPELINE replaces its core pass list.
dsl = m.MlirTestDSL()
print("dsl:", dsl.envar.prefix, dsl.envar.dryrun, dsl._get_pipeline(None))
# CHECK:      dsl: MLIR_DSL True canonicalize,cse


@m.jit
def cumsum(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc


# PROFILE_COMPILER: the tracing report is written to stderr as soon as the
# build ends, the mlir one even though DRYRUN never reaches the pass pipeline.
print("RESULT:", cumsum(10))
# CHECK:      DSL compile profile {{.*}} tracing {{.*}}(MLIR_DSL_PROFILE_COMPILER)
# CHECK:      ast-build
# CHECK:      build
# CHECK:      DSL compile profile {{.*}} mlir {{.*}}(MLIR_DSL_PROFILE_COMPILER)
# CHECK:      RESULT: ?


# Its values: off/on spellings, `deep[:path]` adds the cProfile drill, any
# other value is a report path; an unset variable leaves another DSL's
# setting alone. Not an `env_var` field: read directly by __init__.
def profile(value):
    manager("T_PROF", **({} if value is None else {"PROFILE_COMPILER": value}))
    return profiler.enabled(), profiler.deep()


print("off:", {profile(v) for v in ["off", "0", "false", "no", ""]})
# CHECK: off: {(False, False)}
print("on:", {profile(v) for v in ["on", "1", "true", "yes", " On "]})
# CHECK: on: {(True, False)}
print("deep:", profile("deep"), profile("deep:/r.txt"), "| path:", profile("/r.txt"))
# CHECK: deep: (True, True) (True, True) | path: (True, False)
is_field = any(e.key_name == "profile_compiler" for e in SPEC)
print("unset keeps:", profile(None), "| field:", is_field)
# CHECK: unset keeps: (True, False) | field: False

# --- Defaults, under a prefix with nothing set ------------------------------
show(manager("T_DEF"))
# CHECK:      jit_time_profiling=False log_to_console=False log_to_file=False log_level=1
# CHECK-NEXT: debug=False debuginfo=False show_stacktrace=False print_ir=False verify_trace=False dryrun=False
# CHECK-NEXT: ast_preprocessor=True enable_pass_profiling=False warnings_ignore=False
# CHECK-NEXT: no_cache=False cache_dir=None disable_file_caching=False jit_cache_max_elems=None
# CHECK-NEXT: keep_ir=False keep_ir_after_passes='' print_ir_after_passes='' loc_tracebacks=0
# CHECK-NEXT: remarks='' remarks_policy='all' remarks_output=''
# CHECK-NEXT: arch=None pipeline=None shared_libs=None enable_tvm_ffi=False
# Every variable read under the prefix (the spec plus PROFILE_COMPILER), all
# documented on the class.
suffixes = _prefixed_env_var_suffixes()
print("suffixes:", len(SPEC), len(suffixes), " ".join(suffixes))
# CHECK: suffixes: 28 29 ARCH AST_PREPROCESSOR CACHE_DIR DEBUG DEBUGINFO DISABLE_FILE_CACHING DRYRUN ENABLE_PASS_PROFILING ENABLE_TVM_FFI JIT_CACHE_MAX_ELEMS JIT_TIME_PROFILING KEEPIR_AFTER_PASSES KEEP_IR LIBS LOC_TRACEBACKS LOG_LEVEL LOG_TO_CONSOLE LOG_TO_FILE NO_CACHE PIPELINE PRINT_IR PRINT_IR_AFTER_PASSES PROFILE_COMPILER REMARKS REMARKS_OUTPUT REMARKS_POLICY SHOW_STACKTRACE VERIFY_TRACE WARNINGS_IGNORE
doc = EnvironmentVarManager.__doc__
print("documented:", all(f"[DSL_NAME]_{s}" in doc for s in suffixes))
# CHECK: documented: True

# --- Every variable parsed with the reader its annotation selects -----------
# Booleans in every spelling (case-insensitive), strings verbatim, base-10
# integers, `int | None` as an integer here (the literal `none` below).
PARSE = """JIT_TIME_PROFILING=1 LOG_TO_CONSOLE=1 LOG_TO_FILE=1 LOG_LEVEL=40 DEBUG=true
PRINT_IR=on SHOW_STACKTRACE=0 ENABLE_PASS_PROFILING=yes AST_PREPROCESSOR=false
DEBUGINFO=no VERIFY_TRACE=True NO_CACHE=ON CACHE_DIR=/some/cache/dir KEEP_IR=1
KEEPIR_AFTER_PASSES=canonicalize,cse PRINT_IR_AFTER_PASSES=cse REMARKS=llvm-.*
REMARKS_POLICY=final REMARKS_OUTPUT=/some/remarks.yaml DRYRUN=1 ARCH=sm_90a
WARNINGS_IGNORE=1 DISABLE_FILE_CACHING=1 JIT_CACHE_MAX_ELEMS=16 ENABLE_TVM_FFI=1
PIPELINE=canonicalize LIBS=/a.so:/b.so LOC_TRACEBACKS=3"""
env = manager("T_PARSE", **parse(PARSE))
show(env)
# CHECK:      jit_time_profiling=True log_to_console=True log_to_file=True log_level=40
# CHECK-NEXT: debug=True debuginfo=False show_stacktrace=False print_ir=True verify_trace=True dryrun=True
# CHECK-NEXT: ast_preprocessor=False enable_pass_profiling=True warnings_ignore=True
# CHECK-NEXT: no_cache=True cache_dir='/some/cache/dir' disable_file_caching=True jit_cache_max_elems=16
# CHECK-NEXT: keep_ir=True keep_ir_after_passes='canonicalize,cse' print_ir_after_passes='cse' loc_tracebacks=3
# CHECK-NEXT: remarks='llvm-.*' remarks_policy='final' remarks_output='/some/remarks.yaml'
# CHECK-NEXT: arch='sm_90a' pipeline='canonicalize' shared_libs='/a.so:/b.so' enable_tvm_ffi=True
# Values are read once, at construction: a later change to the process
# environment is seen by a fresh manager only.
os.environ["T_PARSE_DRYRUN"] = "0"
print("read once:", env.dryrun, EnvironmentVarManager("T_PARSE").dryrun)
# CHECK: read once: True False

# The typed readers behind the annotations: whitespace is ignored, unset or
# empty gives the default, anything else is a DSLRuntimeError naming the
# variable, the raw value and the expectation.
TRUE, FALSE = ["1", "TRUE", "Yes", " on "], ["0", "False", "NO", "\toff\n"]
print("bool:", reads(get_bool_env_var, TRUE), reads(get_bool_env_var, FALSE))
# CHECK: bool: ['True', 'True', 'True', 'True'] ['False', 'False', 'False', 'False']
print("bool bad:", reads(get_bool_env_var, ["2", "y", "t", "enabled"]))
# CHECK: bool bad: ['CONFIG_INVALID', 'CONFIG_INVALID', 'CONFIG_INVALID', 'CONFIG_INVALID']
print("int:", reads(get_int_env_var, ["42", " -7 ", "+3", "007"]))
# CHECK: int: ['42', '-7', '3', '7']
print("int bad:", reads(get_int_env_var, ["1.0", "0x1F", "1e3", "none", "1 2"]))
# CHECK: int bad: ['CONFIG_INVALID', 'CONFIG_INVALID', 'CONFIG_INVALID', 'CONFIG_INVALID', 'CONFIG_INVALID']
print("int|None:", reads(get_int_or_none_env_var, ["8", "none", " NONE ", "null"]))
# CHECK: int|None: ['8', 'None', 'None', 'CONFIG_INVALID']
print("str:", reads(get_str_env_var, ["  x ", ""]), read(get_str_env_var, None, "dflt"))
# CHECK: str: ["'  x '", "''"] 'dflt'
defaults = [read(get_bool_env_var, None, True), read(get_int_env_var, "  ", 9)]
print("unset/empty -> default:", defaults, read(get_int_or_none_env_var, "", 5))
# CHECK: unset/empty -> default: ['True', '9'] 5
os.environ["T_VAR"] = " Maybe "
try:
    get_bool_env_var("T_VAR", True)
except DSLUserCodeError as e:
    print("message:", e.message)
# CHECK: message: `T_VAR` has an invalid setting: ` Maybe ` is not a boolean; expected one of 1, on, true, yes, 0, false, no, off (case-insensitive), or empty for the default `True`.

# --- Malformed values: CONFIG_INVALID naming the variable, never a builtin --
bad("LOG_LEVEL", "abc")
# CHECK: LOG_LEVEL='abc': `T_BAD_LOG_LEVEL` has an invalid setting: `abc` is not a base-10 integer; expected one, or empty for the default `1`.
bad("JIT_CACHE_MAX_ELEMS", "unlimited")
# CHECK: JIT_CACHE_MAX_ELEMS='unlimited': `T_BAD_JIT_CACHE_MAX_ELEMS` has an invalid setting: `unlimited` is not a base-10 integer or `none`; expected one of those, or empty for the default `None`.
bad("DEBUGINFO", "nope")  # a dependent default is validated through its own variable
# CHECK: DEBUGINFO='nope': `T_BAD_DEBUGINFO` has an invalid setting: `nope` is not a boolean
bad("DEBUG", "   ")  # whitespace-only is "unset", not an error
# CHECK: DEBUG='   ': accepted
bad("REMARKS_POLICY", "sometimes")  # strings accept anything; the consumer validates
# CHECK: REMARKS_POLICY='sometimes': accepted
try:
    manager("T_BAD", DEBUG="maybe")  # rendered as a user diagnostic with its fix
except m.DSLUserCodeError as e:
    print(str(e))
# CHECK:      error[CONFIG_INVALID]:{{.*}} `T_BAD_DEBUG` has an invalid setting: `maybe` is not a boolean
# CHECK:      suggestion:{{.*}}Set `T_BAD_DEBUG` to a supported value, or leave it unset for the default.
clear("T_BAD")

# Declarations: a computed setting takes no default or read_as; a setting
# needs a bool, int or str annotation (optionally `| None`) for its reader.
try:
    env_var(lambda mgr: 1, affects_compile=False, default=3)
except DSLRuntimeError as e:
    print("computed with default:", e.message)
# CHECK: computed with default: A computed setting takes neither `default` nor `read_as`


class Unannotated(EnvVarSpec):
    tile = env_var("TILE", affects_compile=True, default=1)

    def __init__(self):
        self._apply_env_var_spec("T_DECL")


class Unreadable(EnvVarSpec):
    ratio: float = env_var("RATIO", affects_compile=True, default=1.0)

    def __init__(self):
        self._apply_env_var_spec("T_DECL")


for cls in (Unannotated, Unreadable):
    try:
        cls()
    except DSLRuntimeError as e:
        print(cls.__name__ + ":", e.message)
# CHECK: Unannotated: Unannotated.tile has no type annotation, so the parser for its environment variable cannot be determined. Annotate it on the class.
# CHECK: Unreadable: No environment-variable reader for <class 'float'>. Give the setting a bool, int or str annotation, or compute it from the manager instead.


def reader(annotation):
    try:
        return parser_for_type(annotation).__name__
    except DSLRuntimeError:
        return "DSLRuntimeError"


ANNOTATIONS = (bool, int, str, int | None, Optional[int], str | None, list[int])
print("readers:", [reader(a) for a in ANNOTATIONS])
# CHECK: readers: ['get_bool_env_var', 'get_int_env_var', 'get_str_env_var', 'get_int_or_none_env_var', 'get_int_or_none_env_var', 'get_str_env_var', 'DSLRuntimeError']

# --- The JIT cache key: `affects_compile` settings, sorted, None skipped -----
keys = sorted(e.key_name for e in SPEC)
in_key = {e.key_name for e in SPEC if e.affects_compile}
print("in key:", " ".join(k for k in keys if k in in_key))
# CHECK: in key: arch ast_preprocessor debug debuginfo dryrun enable_tvm_ffi keep_ir_after_passes loc_tracebacks pipeline remarks remarks_output remarks_policy shared_libs
print("not in key:", " ".join(k for k in keys if k not in in_key))
# CHECK: not in key: cache_dir disable_file_caching enable_pass_profiling jit_cache_max_elems jit_time_profiling keep_ir log_level log_to_console log_to_file no_cache print_ir print_ir_after_passes show_stacktrace verify_trace warnings_ignore
base = key("T_KEY")
print("defaults:", base)
# CHECK: defaults: ast_preprocessor=True;debug=False;debuginfo=False;dryrun=False;enable_tvm_ffi=False;keep_ir_after_passes='';loc_tracebacks=0;remarks='';remarks_output='';remarks_policy='all';
# Settings that only govern caching, printing or logging never enter it, so a
# change to them cannot miss the cache.
NON_COMPILE = """PRINT_IR=1 NO_CACHE=1 LOG_LEVEL=20 CACHE_DIR=/x KEEP_IR=1 VERIFY_TRACE=1
PRINT_IR_AFTER_PASSES=cse JIT_CACHE_MAX_ELEMS=4 WARNINGS_IGNORE=1 SHOW_STACKTRACE=1
ENABLE_PASS_PROFILING=1 DISABLE_FILE_CACHING=1 JIT_TIME_PROFILING=1"""
print("unchanged by non-compile settings:", key("T_KEY", **parse(NON_COMPILE)) == base)
# CHECK: unchanged by non-compile settings: True
# Each compile setting changes the key; together they sit at their sorted
# positions, `arch`, `pipeline` and `shared_libs` present only when set.
COMPILE = """ARCH=sm_90a AST_PREPROCESSOR=0 DEBUGINFO=1 DRYRUN=1
ENABLE_TVM_FFI=1 KEEPIR_AFTER_PASSES=cse LOC_TRACEBACKS=2 PIPELINE=cse REMARKS=llvm-.*
REMARKS_OUTPUT=/r.yaml REMARKS_POLICY=final LIBS=/l.so"""
for suffix, value in parse(COMPILE).items():
    k = key("T_KEY", **{suffix: value})
    print(
        f"{suffix} ->", k != base, [item for item in k.split(";") if item not in base]
    )
# CHECK: ARCH -> True ["arch='sm_90a'"]
# CHECK: AST_PREPROCESSOR -> True ['ast_preprocessor=False']
# CHECK: DEBUGINFO -> True ['debuginfo=True']
# CHECK: DRYRUN -> True ['dryrun=True']
# CHECK: ENABLE_TVM_FFI -> True ['enable_tvm_ffi=True']
# CHECK: KEEPIR_AFTER_PASSES -> True ["keep_ir_after_passes='cse'"]
# CHECK: LOC_TRACEBACKS -> True ['loc_tracebacks=2']
# CHECK: PIPELINE -> True ["pipeline='cse'"]
# CHECK: REMARKS -> True ["remarks='llvm-.*'"]
# CHECK: REMARKS_OUTPUT -> True ["remarks_output='/r.yaml'"]
# CHECK: REMARKS_POLICY -> True ["remarks_policy='final'"]
# CHECK: LIBS -> True ["shared_libs='/l.so'"]
print("all:", key("T_KEY", **parse(COMPILE)))
# CHECK: all: arch='sm_90a';ast_preprocessor=False;debug=False;debuginfo=True;dryrun=True;enable_tvm_ffi=True;keep_ir_after_passes='cse';loc_tracebacks=2;pipeline='cse';remarks='llvm-.*';remarks_output='/r.yaml';remarks_policy='final';shared_libs='/l.so';
# DEBUG enters the key itself and through the default it raises.
print("debug:", key("T_KEY", DEBUG="1"))
# CHECK: debug: ast_preprocessor=True;debug=True;debuginfo=True;dryrun=False;
# Values render with repr, sets made deterministic.
print("render:", *[_render_cache_key_value(v) for v in ("x", 3, True, {2, 1})])
# CHECK: render: 'x' 3 True (1, 2)


# --- The DEBUG master switch ------------------------------------------------
# `{prefix}_DEBUG` raises the default of DEBUGINFO and SHOW_STACKTRACE; each
# stays overridable by its own variable.
def debug(**vars):
    env = manager("T_DBG", **vars)
    return env.debug, env.debuginfo, env.show_stacktrace


ON = dict(DEBUG="1")
print("off:", debug(), "| on:", debug(**ON), "| empty:", debug(DEBUG=""))
# CHECK: off: (False, False, False) | on: (True, True, True) | empty: (False, False, False)
print("override:", debug(**ON, DEBUGINFO="0"), debug(**ON, SHOW_STACKTRACE="0"))
# CHECK: override: (True, False, True) (True, True, False)
print("alone:", debug(DEBUGINFO="1"), debug(SHOW_STACKTRACE="1"))
# CHECK: alone: (False, True, False) (False, False, True)

# --- The `arch` property ----------------------------------------------------
# `{prefix}_ARCH` or None (no detection); assignable on the instance, `del`
# re-reads the variable (so `mock.patch.object` teardown works); the key
# reads it through the property.
env = manager("T_ARCH")
print("unset:", env.arch, "arch=" in env.cache_key_str())
# CHECK: unset: None False
env.arch = "sm_80"
print("set:", env.arch, env.cache_key_str().split(";")[0])
# CHECK: set: sm_80 arch='sm_80'
del env.arch
print("deleted:", env.arch)
# CHECK: deleted: None
env = manager("T_ARCH", ARCH="sm_90a")
with mock.patch.object(env, "arch", "sm_120"):
    print("patched:", env.arch, env.cache_key_str().split(";")[0])
# CHECK: patched: sm_120 arch='sm_120'
os.environ["T_ARCH_ARCH"] = "sm_75"
print("after patch:", env.arch, "| fresh:", EnvironmentVarManager("T_ARCH").arch)
# CHECK: after patch: sm_90a | fresh: sm_75

# --- Prefix isolation -------------------------------------------------------
# A manager reads only `{prefix}_*`: another prefix, a prefix of it and the
# DSL's own MLIR_DSL see none of its variables; messages quote the prefix;
# variables are case-sensitive.
a = manager("A_DSL", DRYRUN="1", ARCH="sm_80", PIPELINE="cse", LOC_TRACEBACKS="5")
b, short = EnvironmentVarManager("B_DSL"), EnvironmentVarManager("A")
for env in (a, b, short):
    print(env.prefix, env.dryrun, env.arch, env.pipeline, env.loc_tracebacks)
# CHECK:      A_DSL True sm_80 cse 5
# CHECK-NEXT: B_DSL False None None 0
# CHECK-NEXT: A False None None 0
print("a key:", a.cache_key_str())
# CHECK: a key: arch='sm_80';ast_preprocessor=True;debug=False;debuginfo=False;dryrun=True;enable_tvm_ffi=False;keep_ir_after_passes='';loc_tracebacks=5;pipeline='cse';
print("b == short == defaults:", b.cache_key_str() == short.cache_key_str() == base)
# CHECK: b == short == defaults: True
d = m.MlirTestDSL().envar
print("MlirTestDSL:", d.prefix, d.dryrun, d.arch, d.pipeline, d.loc_tracebacks)
# CHECK: MlirTestDSL: MLIR_DSL True None canonicalize,cse 0
os.environ["lower_DRYRUN"] = "1"
lo, up = EnvironmentVarManager("lower"), EnvironmentVarManager("LOWER")
print(
    "case:", lo.dryrun, up.dryrun, "| default prefix:", EnvironmentVarManager().prefix
)
# CHECK: case: True False | default prefix: DSL


# --- EnvVarSpec composition for a sub-DSL -----------------------------------
# A subclass redeclares a setting in place, adds string-backed, computed (a
# function of the manager) and `read_as` settings; the base class is
# unchanged and the key follows the subclass's declarations.
class SubEnv(EnvironmentVarManager):
    dryrun: bool = env_var("DRYRUN", affects_compile=False, default=True)
    tile: int = env_var("TILE", affects_compile=True, default=128)
    backend: str | None = env_var("BACKEND", affects_compile=True)
    label: str = env_var(lambda mgr: f"{mgr.prefix}-{mgr.tile}", affects_compile=True)
    _target: str = env_var(
        "TARGET", affects_compile=True, default="host", read_as="target"
    )

    @property
    def target(self) -> str:
        return self._target.upper()


class SubSub(SubEnv):
    tile: int = env_var("TILE", affects_compile=True, default=256)


names = [e.attribute for e in SubEnv._ENV_VAR_SPEC]
same_slot = names.index("dryrun") == [e.attribute for e in SPEC].index("dryrun")
print("spec:", len(names) - len(SPEC), same_slot, names[-4:])
# CHECK: spec: 4 True ['tile', 'backend', 'label', '_target']
sub = SubEnv("T_SUB")
print("defaults:", sub.dryrun, sub.tile, sub.backend, sub.label, sub.target)
# CHECK: defaults: True 128 None T_SUB-128 HOST
print("base untouched:", manager("T_SUB").dryrun, "dryrun" in sub.cache_key_str())
# CHECK: base untouched: False False
print("key:", sub.cache_key_str())
# CHECK: key: ast_preprocessor=True;debug=False;debuginfo=False;enable_tvm_ffi=False;keep_ir_after_passes='';label='T_SUB-128';loc_tracebacks=0;remarks='';remarks_output='';remarks_policy='all';target='HOST';tile=128;
os.environ.update(
    {"T_SUB_TILE": "64", "T_SUB_BACKEND": "cuda", "T_SUB_TARGET": "device"}
)
sub = SubEnv("T_SUB")
print("from env:", sub.tile, sub.backend, sub.label, sub.target)
# CHECK: from env: 64 cuda T_SUB-64 DEVICE
print("key:", sub.cache_key_str())
# CHECK: key: {{.*}}backend='cuda';{{.*}}label='T_SUB-64';{{.*}}target='DEVICE';tile=64;
print("subsub:", SubSub("T_SS").tile, SubSub("T_SS").label)
# CHECK: subsub: 256 T_SS-256


# --- The logging settings ---------------------------------------------------
# No sink: the logger named after the prefix is off; LOG_TO_CONSOLE opens a
# stderr handler at LOG_LEVEL (1: everything), LOG_TO_FILE a `<prefix>.log`
# handler; a LOG_LEVEL with no sink is reported as a DSLWarning;
# JIT_TIME_PROFILING opens the console at INFO when the user did not.
def handlers():
    return [type(h).__name__ for h in log().handlers]


manager("T_L1")
lg = log()
print("no sink:", lg.name, lg.level, handlers(), lg.isEnabledFor(logging.CRITICAL))
# CHECK: no sink: T_L1 51 [] False
manager("T_L2", LOG_TO_CONSOLE="1")
print("console:", log().level, handlers())
# CHECK: console: 1 ['StreamHandler']
log().debug("debug line")
# CHECK: - T_L2 - DEBUG - [<module>] - debug line
manager("T_L3", LOG_TO_FILE="1", LOG_LEVEL="20")
fh = log().handlers[0]
print("file:", log().level, handlers(), os.path.basename(fh.baseFilename))
# CHECK: file: 20 ['FileHandler'] T_L3.log
log().info("to the file")
fh.close()
with open("T_L3.log", encoding="utf-8") as f:
    print("file content:", "- T_L3 - INFO - [<module>] - to the file" in f.read())
# CHECK: file content: True
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    manager("T_L4", LOG_LEVEL="20")
print("no sink warning:", len(caught), isinstance(caught[0].message, DSLWarning))
# CHECK: no sink warning: 1 True
print(str(caught[0].message))
# CHECK: warning:{{.*}} T_L4_LOG_LEVEL was set, but neither logging to file (T_L4_LOG_TO_FILE) nor logging to console (T_L4_LOG_TO_CONSOLE) is enabled, so it has no effect.
# CHECK: suggestion:{{.*}}Set T_L4_LOG_TO_CONSOLE=1 or T_L4_LOG_TO_FILE=1 to see the log.
env = manager("T_L5", JIT_TIME_PROFILING="1")
print("profiling console:", env.log_to_console, env.log_level, handlers(), log().level)
# CHECK: profiling console: True 20 ['StreamHandler'] 20
env = manager("T_L6", JIT_TIME_PROFILING="1", LOG_TO_CONSOLE="1", LOG_LEVEL="1")
print("user console kept:", env.log_level, log().level)
# CHECK: user console kept: 1 1
