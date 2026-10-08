# -*- Python -*-

import os
import subprocess

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner of the mlir.mlir_dsl sub-DSL:
# the IR-level tests here and the compile-and-run tests under Integration/.

# name: The name of this test suite.
config.name = "MLIR-DSL-MlirTestDSL"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py", ".test"]

# excludes: A list of directories and files to exclude from the testsuite.
config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.mlir_dsl_obj_root

# The package is part of the Python bindings; nothing to test without them.
if not config.enable_bindings_python:
    config.unsupported = True

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%PYTHON", config.python_executable))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

# FileCheck, not, count and the MLIR tools.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# The `mlir` Python package of this build.
llvm_config.with_environment(
    "PYTHONPATH",
    [os.path.join(config.mlir_obj_root, "python_packages", "mlir_core")],
    append_path=True,
)

# ASan does not play well with the Python interpreter.
config.environment["ASAN_OPTIONS"] = "detect_leaks=0"

# MLIR_DSL_KEEP_IR dumps and the file cache go under MLIR_DSL_CACHE_DIR: keep
# them out of the source tree and of the shared system temp dir.
config.environment["MLIR_DSL_CACHE_DIR"] = os.path.join(
    config.test_exec_root, "Output", "mlir_dsl_cache"
)


def have_host_jit_feature_support(feature_name):
    """Whether the host can run the ExecutionEngine (asked of mlir-runner)."""
    mlir_runner_exe = lit.util.which("mlir-runner", config.llvm_tools_dir)
    if not mlir_runner_exe:
        return False
    try:
        mlir_runner_cmd = subprocess.Popen(
            [mlir_runner_exe, "--host-supports-" + feature_name],
            stdout=subprocess.PIPE,
        )
    except OSError:
        print("could not exec mlir-runner")
        return False
    mlir_runner_out = mlir_runner_cmd.stdout.read().decode("ascii")
    mlir_runner_cmd.wait()
    return "true" in mlir_runner_out


if config.enable_execution_engine and have_host_jit_feature_support("jit"):
    config.available_features.add("host-supports-jit")

# The TVM-FFI export test needs the optional ``tvm_ffi`` package of the test
# interpreter. Probed from a neutral cwd: importing tvm_ffi from inside its own
# package directory fails on a stdlib name clash.
try:
    _probe = subprocess.run(
        [config.python_executable, "-c", "import tvm_ffi"],
        capture_output=True,
        cwd=config.test_exec_root if os.path.isdir(config.test_exec_root) else None,
        timeout=120,
    )
    if _probe.returncode == 0:
        config.available_features.add("tvm_ffi")
except Exception:
    pass
