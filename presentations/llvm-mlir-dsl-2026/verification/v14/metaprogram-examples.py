# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Current-source verification of three presentation metaprogram examples."""

import hashlib
import importlib
import pathlib
import sys

import numpy as np
import mlir.mlir_dsl as m


@m.jit
def scale(x: m.Int32, factor):
    return x * factor


class Op:
    def apply(self, x):
        raise NotImplementedError


class AddN(Op):
    def __init__(self, n): self.n = n
    def apply(self, x): return x + self.n


class MulN(Op):
    def __init__(self, k): self.k = k
    def apply(self, x): return x * self.k


@m.jit
def pipeline(x: m.Int32, ops):
    for op in ops.values():
        x = op.apply(x)
    return x


@m.jit
def fused(arr: m.Pointer[m.Float32], ops):
    for i in m.range(4):
        x = arr[i]
        for op in ops.values():
            x = op.apply(x)
        arr[i] = x


def scalar_value(result):
    # The current ABI may return a Python scalar or its scalar wrapper.
    # This harness normalizes only that host result; no traced code is changed.
    value = result.value if hasattr(result, "value") else result
    assert isinstance(value, (int, np.integer)), (type(result), type(value))
    return int(value)


def main():
    print("Python:", sys.version)
    print("NumPy:", np.__version__)
    print("DSL:", type(m.MlirTestDSL()).__name__)
    for name in (
        "mlir.mlir_dsl.mlir_dsl", "mlir.dsl.core.dsl", "mlir.dsl.core.plugin",
        "mlir.dsl.plugins.ast_preprocessor.scf",
    ):
        module = importlib.import_module(name)
        path = pathlib.Path(module.__file__).resolve()
        print("SOURCE:", name, path, hashlib.sha256(path.read_bytes()).hexdigest())

    result = scale(4, 3)
    assert scalar_value(result) == 12, result
    print(f"PASS scale(4, 3) = 12; host result type: {type(result).__name__}")

    ops = {"shift": AddN(2), "scale": MulN(3)}
    result = pipeline(4, ops)
    assert scalar_value(result) == 18, result
    print(f"PASS pipeline(4, ops) = 18; host result type: {type(result).__name__}")

    data = np.arange(4, dtype=np.float32)
    returned = fused(data, ops)
    expected = np.array([6, 9, 12, 15], dtype=np.float32)
    np.testing.assert_array_equal(data, expected)
    print(f"PASS fused(data, ops): {data.tolist()}; returns {returned!r}")
    print("PASS: all three current-source metaprogram examples")


if __name__ == "__main__":
    main()
