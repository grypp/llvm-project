# RUN: %PYTHON %s | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Decorator hooks, nested calls, and abstract DSL families."""

import contextlib
import inspect

from mlir.dsl import BaseDSL, Plugins
from mlir.dsl.core.common import DSLRuntimeError
from mlir.dsl.plugins.decorators.jit import func


# ===----------------------------------------------------------------------===
# ===----               DSL Implementation                         ====----===
# ===----------------------------------------------------------------------===


events = []


class Entry(func.Jit):
    def decorate(self, function, **options):
        function.marker = options.pop("marker", None)
        return super().decorate(function, **options)

    def wrap(self, function, wrapper):
        wrapper.marker = function.marker
        return wrapper

    @contextlib.contextmanager
    def body_scope(self, dsl, function, *, signature):
        events.append("body-enter")
        yield
        events.append("body-exit")


class Family(BaseDSL, abstract=True):
    plugins = Plugins(decorators=[Entry()])

    def __init__(self):
        super().__init__(name="CORE_CONTRACT")

    @classmethod
    @contextlib.contextmanager
    def call_scope(cls, function):
        events.append("call-enter")
        yield
        events.append("call-exit")


class Concrete(Family):
    pass


# ===----------------------------------------------------------------------===
# ===----               DSL Test Cases                             ====----===
# ===----------------------------------------------------------------------===


# Declaring a concrete subclass does not initialize the family.
try:
    Family()
except DSLRuntimeError:
    print("ABSTRACT: refused before a concrete instance")
else:
    raise AssertionError("abstract family instantiated")


@Concrete.jit
def inner():
    events.append("nested")


@Concrete.jit(marker="public")
def outer():
    inner()


assert outer.marker == "public"
# Resolve the instance at call time; the wrapper must not retain a bound method.
assert all(not inspect.ismethod(cell.cell_contents) for cell in outer.__closure__)
outer()
assert events == [
    "call-enter",
    "body-enter",
    "nested",
    "body-exit",
    "call-exit",
], events
assert Family() is Concrete()
print("TRACE CONTRACT: decorator options, scopes, nested call and ownership")
# CHECK: ABSTRACT: refused before a concrete instance
# CHECK-NEXT: TRACE CONTRACT: decorator options, scopes, nested call and ownership
