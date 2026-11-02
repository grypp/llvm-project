# -*- Python -*-

import os

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner of the mlir.dsl Python DSL.

# name: The name of this test suite.
config.name = "MLIR-DSL"

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

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

# FileCheck.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# The `mlir` Python package of this build.
llvm_config.with_environment(
    "PYTHONPATH",
    [os.path.join(config.mlir_obj_root, "python_packages", "mlir_core")],
    append_path=True,
)

# ASan does not play well with the Python interpreter.
config.environment["ASAN_OPTIONS"] = "detect_leaks=0"
