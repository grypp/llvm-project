# RUN: env TRACE_DSL_PRINT_IR=1 %PYTHON %s 2>&1 | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace existing MLIR builders without DSL types or a compiler."""

from mlir import ir
from mlir.dialects import arith, gpu
from mlir.dsl import BaseDSL, DSLRuntimeError, DSLUserCodeError, Plugins
from mlir.dsl.core.common import get_current_dsl
from mlir.dsl.plugins.decorators.jit import func


# ===----------------------------------------------------------------------===
# ===----               DSL Implementation                         ====----===
# ===----------------------------------------------------------------------===


class PrintDSL(BaseDSL):
    plugins = Plugins(decorators=[func.Jit()])

    def __init__(self):
        super().__init__(name="TRACE_DSL")


# ===----------------------------------------------------------------------===
# ===----               DSL Test Cases                             ====----===
# ===----------------------------------------------------------------------===


@PrintDSL.jit
def hello():
    gpu.printf("hello\n")


# CHECK-LABEL: func.func @hello()
# CHECK: gpu.printf "hello
# CHECK: return
assert hello() is None


@PrintDSL.jit()
def repeat(count, *, text="hello"):
    for _ in range(count):
        gpu.printf(text)


# CHECK-LABEL: func.func @repeat_2()
# CHECK: gpu.printf "again"
# CHECK-NEXT: gpu.printf "again"
# CHECK-NEXT: return
repeat(2, text="again")


@PrintDSL.jit
def print_parts(*texts, **named):
    for text in texts:
        gpu.printf(text)
    for text in named.values():
        gpu.printf(text)


# CHECK-LABEL: func.func @print_parts
# CHECK: gpu.printf "left"
# CHECK-NEXT: gpu.printf "right"
# CHECK-NEXT: gpu.printf "last"
# CHECK-NEXT: return
print_parts("left", "right", end="last")


@PrintDSL.jit
def print_value(value):
    gpu.printf("value=%d\n", value)


@PrintDSL.jit
def nested():
    value = arith.constant(ir.IntegerType.get_signless(32), 7)
    print_value(value)


# CHECK-LABEL: func.func @nested()
# CHECK: %[[VALUE:.+]] = arith.constant 7 : i32
# CHECK-NEXT: gpu.printf "value=%d{{.*}}", %[[VALUE]] : i32
# CHECK-NEXT: return
nested()


def expect_error(label, fn):
    try:
        fn()
    except (DSLUserCodeError, DSLRuntimeError) as exc:
        print(label, type(exc).__name__)
    else:
        raise AssertionError(f"{label}: expected an error")
    assert get_current_dsl() is None


@PrintDSL.jit
def invalid_result():
    return object()


expect_error("NOT_FUNCTION", lambda: PrintDSL.jit(7))
expect_error("MISSING_ARGUMENT", repeat)
expect_error("EXTRA_ARGUMENT", lambda: hello(7))
expect_error("RESULT", invalid_result)
expect_error("DUPLICATE", lambda: Plugins(decorators=[func.Jit(), func.Jit()]))
# CHECK: NOT_FUNCTION DSLUserCodeError
# CHECK-NEXT: MISSING_ARGUMENT DSLUserCodeError
# CHECK-NEXT: EXTRA_ARGUMENT DSLUserCodeError
# CHECK-NEXT: RESULT DSLUserCodeError
# CHECK-NEXT: DUPLICATE DSLRuntimeError

print("SINGLETON", PrintDSL() is PrintDSL())
print("TRACE RESTORED", get_current_dsl() is None)
# CHECK: SINGLETON True
# CHECK-NEXT: TRACE RESTORED True
