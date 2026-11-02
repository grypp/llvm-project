# RUN: rm -rf %t && env MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 %PYTHON %s %t 2>&1 | FileCheck %s
# The framework and the test sub-DSL relocate together with their bindings
# package, like every other `mlir.*` module: a project that ships the bindings
# under its own name (`<pkg>._mlir`) installs `dsl/` and `mlir_dsl/` next to
# them, and nothing in either names the `mlir` package by its string.
#
# The layout built here under %t is `relocated._mlir`: a copy of `dsl/` and
# `mlir_dsl/` (so the code runs under its new module names and from its new
# files) plus a meta path finder that serves every other `relocated._mlir.<x>`
# with the real `mlir.<x>` module object. The extensions are compiled for the
# `mlir` prefix and are initialised once, so plain `mlir` stays importable and
# is never imported a second time; `mlir.dsl` is not imported at all.
#
# Two checks are behavioural and fail with module names as string literals:
# the registry (`packages:`, and with it the author frame a diagnostic points
# at) and the profiler's `dsl:` bucket for a relocated file. The derivations
# on the bindings side (`user_op`, `dlpack`) resolve to the plain name under
# this alias, so the static scan at the end is what pins those.
import ast
import importlib
import importlib.abc
import importlib.util
import os
import pathlib
import shutil
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
PREFIX = "relocated._mlir"
COPIED = ("dsl", "mlir_dsl")


class BindingsAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``relocated._mlir.<x>`` is the real ``mlir.<x>`` module, except for the
    packages copied under the alias, which the path finder serves from disk."""

    def __init__(self):
        self._specs = {}

    def find_spec(self, name, path=None, target=None):
        head, dot, tail = name.partition(PREFIX + ".")
        if head or not dot or tail.split(".", 1)[0] in COPIED:
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        real = importlib.import_module("mlir." + spec.name[len(PREFIX) + 1 :])
        self._specs[spec.name] = real.__spec__
        return real

    def exec_module(self, module):
        # module_from_spec stamped the alias spec on the real module.
        module.__spec__ = self._specs.pop(module.__spec__.name)


def package_dir(name):
    """The directory of package ``name``, located without importing it."""
    return pathlib.Path(importlib.util.find_spec(name).submodule_search_locations[0])


pkg = root / "relocated" / "_mlir"
pkg.mkdir(parents=True)
(root / "relocated" / "__init__.py").touch()
(pkg / "__init__.py").touch()
for name in COPIED:
    shutil.copytree(
        package_dir(f"mlir.{name}"),
        pkg / name,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
sys.path.insert(0, str(root))
sys.meta_path.insert(0, BindingsAlias())

m = importlib.import_module("relocated._mlir.mlir_dsl")
diagnostics = importlib.import_module("relocated._mlir.dsl.core.diagnostics")
profiler = importlib.import_module("relocated._mlir.dsl.util.profiler")
dlpack = importlib.import_module("relocated._mlir.dsl.plugins.adapters.dlpack")
ir = importlib.import_module("relocated._mlir.ir")

# The framework registers itself under the name it was imported by, the test
# DSL under its own package; nothing imported the plain `mlir.dsl` and the
# bindings were not initialised twice.
# CHECK: packages: ['relocated._mlir.dsl', 'relocated._mlir.mlir_dsl']
# CHECK-NEXT: plain framework imported: False
# CHECK-NEXT: one bindings module: True
print("packages:", diagnostics._DSL_PACKAGES, flush=True)
print("plain framework imported:", "mlir.dsl" in sys.modules, flush=True)
print("one bindings module:", ir is sys.modules["mlir.ir"], flush=True)


@m.jit
def f(a: m.Int32) -> m.Int32:
    return a + 1


f(3)
# CHECK: func.func @f(

# A diagnostic walks out of the relocated framework's frames to the author's
# line, and a relocated framework file is attributed to the framework.
try:
    m.dtype("nope")
except m.DSLUserCodeError as e:
    print("author frame:", os.path.basename(e.filename), flush=True)
# CHECK: author frame: relocation.py
print("region:", profiler.region_of(str(pkg / "dsl" / "core" / "dsl.py")), flush=True)
# CHECK: region: dsl: dsl.py

# The adapter finds its extension next to the bindings it was imported with.
print("dlpack:", dlpack.available(), dlpack.DlpackTensor(np.zeros(4, np.float32)))
# CHECK: dlpack: True tensor<4xFloat32>_host


def module_names(tree):
    """Where ``tree`` names the ``mlir`` package by its string: an absolute
    import, a string constant outside a docstring, or the name handed to
    ``import_module``/``find_spec``."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    docstrings.add(id(stmt.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mlir" or alias.name.startswith("mlir."):
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "mlir" or node.module.startswith("mlir."):
                yield node.lineno, f"from {node.module}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings and node.value.startswith("mlir."):
                yield node.lineno, repr(node.value)
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func
            name = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else getattr(callee, "id", "")
            )
            first = node.args[0]
            if name in ("import_module", "find_spec") and isinstance(
                first, ast.Constant
            ):
                if first.value == "mlir":
                    yield node.lineno, f"{name}('mlir')"


hits = []
for directory in (pkg / "dsl", pkg / "mlir_dsl", package_dir("mlir.emitc_dsl")):
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(directory.parent).as_posix()
        tree = ast.parse(path.read_text(), str(path))
        hits.extend(f"{rel}:{line} {what}" for line, what in module_names(tree))
print("module names as strings:", hits, flush=True)
# CHECK: module names as strings: []
