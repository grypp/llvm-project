# RUN: %PYTHON %s | FileCheck %s
# Package code raises DSLUserCodeError for user
# mistakes and DSLRuntimeError for internal invariants, never a bare builtin
# exception. Every `raise` under the package is inspected syntactically, so
# comments and strings can neither hide nor fake a hit; a bare `raise` and a
# `raise DSLRuntimeError(...) from exc` pass.
import ast
import os

import mlir.dsl
import mlir.mlir_dsl

FORBIDDEN = {
    "TypeError",
    "ValueError",
    "RuntimeError",
    "NotImplementedError",
    "AttributeError",
}


def raised_name(node: ast.Raise):
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


roots = [
    os.path.dirname(os.path.abspath(mlir.dsl.__file__)),
    os.path.dirname(os.path.abspath(mlir.mlir_dsl.__file__)),
]
scanned = 0
violations = 0
for root in roots:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            scanned += 1
            # A `__getattr__` must raise AttributeError for an unknown name, at
            # module level (PEP 562) or on a class or metaclass (the data model,
            # so `getattr(x, name, default)` keeps its default); that is the
            # protocol, not an internal invariant.
            protocol = [
                (n.lineno, n.end_lineno)
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
            ]
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise):
                    name = raised_name(node)
                    if name == "AttributeError" and any(
                        a <= node.lineno <= b for a, b in protocol
                    ):
                        continue
                    if name in FORBIDDEN:
                        violations += 1
                        print(
                            f"VIOLATION {os.path.relpath(path, root)}:{node.lineno} raise {name}"
                        )

print(f"SCANNED {scanned} files, {violations} violations")
print("OK" if violations == 0 else "FAIL")
# CHECK-NOT: VIOLATION
# CHECK:     SCANNED {{[1-9][0-9]*}} files
# CHECK-NOT: VIOLATION
# CHECK:     OK
