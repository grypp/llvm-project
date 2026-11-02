# RUN: %PYTHON %s 2>&1 | FileCheck %s
# S1: the environment-variable settings of a DSL instance. The variables are
# set from inside the test so the RUN line stays plain; a manager reads them
# once, at construction.
import os

from mlir.dsl.core.common import DSLRuntimeError, DSLUserCodeError
from mlir.dsl.core.env_manager import (
    EnvVar,
    EnvVarSpec,
    EnvironmentVarManager,
    env_var,
    get_bool_env_var,
    get_str_env_var,
    parser_for_type,
)

for k in list(os.environ):
    if k.startswith(("MLIR_DSL_", "OTHER_", "MY_")):
        del os.environ[k]
os.environ["MLIR_DSL_DEBUG"] = "1"
os.environ["MLIR_DSL_PRINT_IR"] = " Yes "
os.environ["MLIR_DSL_DRYRUN"] = "off"
os.environ["MLIR_DSL_NO_CACHE"] = ""  # empty: the default

# --- the shipped settings ----------------------------------------------------
m = EnvironmentVarManager("MLIR_DSL")
SEED_SETTINGS = (
    "debug",
    "print_ir",
    "debuginfo",
    "no_cache",
    "dryrun",
    "warnings_ignore",
)
spec = [e.attribute for e in type(m)._ENV_VAR_SPEC]
print("SPEC:", [n for n in SEED_SETTINGS if n in spec])
keyed = [e.attribute for e in type(m)._ENV_VAR_SPEC if e.affects_compile]
print("KEYED:", [n for n in SEED_SETTINGS if n in keyed])
print(
    "VALUES:", m.debug, m.print_ir, m.debuginfo, m.no_cache, m.dryrun, m.warnings_ignore
)
other = EnvironmentVarManager("OTHER")
print(
    "OTHER:", other.debug, other.print_ir, other.debuginfo, other.no_cache, other.dryrun
)
# CHECK: SPEC: ['debug', 'print_ir', 'debuginfo', 'no_cache', 'dryrun', 'warnings_ignore']
# CHECK: KEYED: ['debug', 'debuginfo', 'dryrun']
# CHECK: VALUES: True True True False False False
# CHECK: OTHER: False False False False False

# --- a malformed value is the user's mistake ----------------------------------
os.environ["MLIR_DSL_DRYRUN"] = "maybe"
try:
    EnvironmentVarManager("MLIR_DSL")
    print("INVALID: no error")
except DSLUserCodeError as e:
    print("INVALID:", e.code)
    print("INVALID MESSAGE:", e.message)
del os.environ["MLIR_DSL_DRYRUN"]
# CHECK: INVALID: CONFIG_INVALID
# CHECK: INVALID MESSAGE: `MLIR_DSL_DRYRUN` has an invalid setting: `maybe` is not a boolean

# --- the readers --------------------------------------------------------------
print(
    "BOOL:",
    get_bool_env_var("MLIR_DSL_UNSET", True),
    get_bool_env_var("MLIR_DSL_PRINT_IR"),
)
print(
    "STR:",
    get_str_env_var("MLIR_DSL_PRINT_IR"),
    get_str_env_var("MLIR_DSL_UNSET", "dflt"),
)
print(
    "PARSERS:",
    parser_for_type(bool).__name__,
    parser_for_type(str).__name__,
    parser_for_type(str | None).__name__,
)
try:
    print("PARSER INT:", parser_for_type(int).__name__)
except DSLRuntimeError as e:
    print("PARSER INT:", e.message.split(".")[0])
# CHECK: BOOL: True True
# CHECK: STR:  Yes  dflt
# CHECK: PARSERS: get_bool_env_var get_str_env_var get_str_env_var
# CHECK: PARSER INT: get_int_env_var


# --- a sub-DSL's manager: redeclare, add, compute ------------------------------
class MyEnv(EnvironmentVarManager):
    # Redeclared: keeps its position, new default and cache-key answer.
    debug: bool = env_var("DEBUG", affects_compile=False, default=True)
    arch: str | None = env_var("ARCH", affects_compile=True, default=None)
    # Computed from the settings above; no variable of its own.
    label: str = env_var(
        lambda mgr: f"{mgr.arch or 'host'}:{int(mgr.debug)}", affects_compile=False
    )
    renamed: str = env_var(
        "LIB_PATH", affects_compile=False, default="", read_as="libs"
    )


os.environ["MY_ARCH"] = "sm_90"
mine = MyEnv("MY")
my_spec = [e.attribute for e in MyEnv._ENV_VAR_SPEC]
print(
    "MY SPEC:",
    [n for n in SEED_SETTINGS + ("arch", "label", "renamed") if n in my_spec],
)
print(
    "MY VALUES:", mine.debug, mine.debuginfo, mine.arch, mine.label, repr(mine.renamed)
)
print(
    "MY KEYS:",
    [
        (e.attribute, e.key_name)
        for e in MyEnv._ENV_VAR_SPEC
        if e.attribute in ("debug", "renamed")
    ],
)
print(
    "BASE UNCHANGED:",
    [
        e.affects_compile
        for e in EnvironmentVarManager._ENV_VAR_SPEC
        if e.attribute == "debug"
    ],
    EnvironmentVarManager("MY").debug,
)
# CHECK: MY SPEC: ['debug', 'print_ir', 'debuginfo', 'no_cache', 'dryrun', 'warnings_ignore', 'arch', 'label', 'renamed']
# CHECK: MY VALUES: True True sm_90 sm_90:1 ''
# CHECK: MY KEYS: [('debug', 'debug'), ('renamed', 'libs')]
# CHECK: BASE UNCHANGED: [True] False

# --- declaration mistakes are the DSL author's -------------------------------
try:

    class BadComputed(EnvVarSpec):
        x: str = env_var(lambda mgr: "v", affects_compile=False, default="d")

except DSLRuntimeError as e:
    print("BAD COMPUTED:", e.message.split(":")[0])


class NoAnnotation(EnvironmentVarManager):
    plain = env_var("PLAIN", affects_compile=False, default=False)


try:
    NoAnnotation("MY")
except DSLRuntimeError as e:
    print("NO ANNOTATION:", e.message.split(",")[0])


class IntSetting(EnvironmentVarManager):
    count: int = env_var("COUNT", affects_compile=True, default=1)


try:
    print("INT SETTING:", IntSetting("MY").count)
except DSLRuntimeError as e:
    print("INT SETTING:", e.message.split(".")[0])
spec_debug = next(e for e in MyEnv._ENV_VAR_SPEC if e.attribute == "debug")
print("ENVVAR:", isinstance(spec_debug, EnvVar), spec_debug.source)
# CHECK: BAD COMPUTED: A computed setting takes neither `default` nor `read_as`
# CHECK: NO ANNOTATION: NoAnnotation.plain has no type annotation
# CHECK: INT SETTING: 1
# CHECK: ENVVAR: True DEBUG

print("S1_ENV", "OK")
# CHECK: S1_ENV OK
