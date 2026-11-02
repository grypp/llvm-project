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

from mlir import ir
from mlir.dsl import Int32
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith

class RuntimeDefaults(Concrete):
    plugins = Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith), decorators=[func.Jit()])
    def _is_meta_at_boundary(self, value, annotation, name, index, function):
        if isinstance(value, int):
            return False
        return super()._is_meta_at_boundary(value, annotation, name, index, function)
    def _flatten_jit_arg(self, value, *args, **kwargs):
        return super()._flatten_jit_arg(Int32(value) if isinstance(value, int) else value, *args, **kwargs)
    def _restore_argument(self, value, annotation, block_args, start):
        if isinstance(value, int):
            return Int32(block_args[start]), 1
        return super()._restore_argument(value, annotation, block_args, start)
    def _mangle_folds(self, value, annotation):
        return True
    def _mangle_render(self, value, annotation):
        return "selected"

@RuntimeDefaults.jit
def runtime_argument(value):
    assert isinstance(value, Int32)
    assert isinstance(value.ir_value(), ir.Value)

runtime_argument(7)
assert RuntimeDefaults().mangle_name("f", (7,), inspect.signature(lambda value: None)) == "f_selected"
print("BOUNDARY CONTRACT: runtime default, restoration and symbol policy")
# CHECK-NEXT: BOUNDARY CONTRACT: runtime default, restoration and symbol policy

from mlir.dsl.core.common import DSLUserCodeError
class DomainError(DSLUserCodeError):
    pass
class DomainDSL(Concrete):
    user_error_class = DomainError
    def on_trace_name_error(self, function, error, filename, lineno):
        raise LookupError("domain name policy") from error

@DomainDSL.jit
def needs_argument(value):
    pass
try:
    needs_argument()
except DomainError:
    pass
else:
    raise AssertionError("the core bypassed the DSL error class")

@DomainDSL.jit
def missing_name():
    unresolved_core_contract_name
try:
    missing_name()
except LookupError as error:
    assert str(error) == "domain name policy"
else:
    raise AssertionError("the core bypassed the trace error hook")
print("ERROR CONTRACT: custom class and trace name policy")
# CHECK-NEXT: ERROR CONTRACT: custom class and trace name policy

forwarded_options = []
class BoundaryOptionsDSL(RuntimeDefaults):
    def _check_unsupported_jit_arg(self, *, boundary_marker=None, **kwargs):
        if boundary_marker is not None:
            forwarded_options.append(boundary_marker)
        return super()._check_unsupported_jit_arg(**kwargs)

@BoundaryOptionsDSL.jit
def bind_device_argument(value: Int32):
    _, _, _, device_args = BoundaryOptionsDSL().bind_arguments(
        lambda device_value: None,
        "device_entry",
        (value,),
        {},
        boundary_marker="device policy",
    )
    assert len(device_args.values) == len(device_args.types) == len(device_args.attributes) == 1

bind_device_argument(7)
assert forwarded_options == ["device policy"]
print("DEVICE BOUNDARY CONTRACT: custom options reach validation")
# CHECK-NEXT: DEVICE BOUNDARY CONTRACT: custom options reach validation

from mlir.dsl.plugins.ast_preprocessor import scf
factory_calls = []
rewrite_mode = ["a"]
class Factory(scf.ASTPreprocessor):
    def make_preprocessor(self, dsl, package, *, decorator_names):
        factory_calls.append(tuple(decorator_names))
        return super().make_preprocessor(dsl, package, decorator_names=decorator_names)

class KeyedDSL(Concrete):
    plugins = Plugins(type_ops=UpstreamDialectTypeOps(scalars=arith), ast_preprocessor=Factory(), decorators=[func.Jit()])
    def preprocess_key(self, function):
        return super().preprocess_key(function), rewrite_mode[0]

@KeyedDSL.jit
def keyed(value: Int32):
    result = value + 1

original = keyed.__wrapped__.__code__
keyed(3)
first = keyed.__wrapped__.__code__
keyed(4)
assert keyed.__wrapped__.__code__ is first
rewrite_mode[0] = "b"
keyed(5)
assert keyed.__wrapped__.__code__ is not first
assert first is not original
assert keyed.__wrapped__._original_code is original
assert factory_calls == [("jit",)]
print("PREPROCESS CONTRACT: factory and keyed rewrite from original code")
# CHECK-NEXT: PREPROCESS CONTRACT: factory and keyed rewrite from original code

from mlir.dsl.util.cache import JitCacheDict
class Cache(JitCacheDict):
    pass
class CacheDSL(RuntimeDefaults):
    _jit_cache_class = Cache
    salt = b"first"
    def _cache_key_extra(self):
        return self.salt

cache_dsl = CacheDSL()
assert isinstance(cache_dsl.jit_cache, Cache)
with ir.Context(), ir.Location.unknown():
    module = ir.Module.create()
    first_hash = cache_dsl.get_module_hash(module, "empty")
    cache_dsl.salt = b"second"
    second_hash = cache_dsl.get_module_hash(module, "empty")
assert first_hash != second_hash
print("CACHE CONTRACT: cache class and extra identity")
# CHECK-NEXT: CACHE CONTRACT: cache class and extra identity

class UntracedInspectionDSL(Concrete):
    pass
class OtherInspectionDSL(Concrete):
    pass
inspection = UntracedInspectionDSL()
assert inspection.dump_mlir_raw_path is None
assert inspection.dump_mlir_paths == {}
inspection.dump_mlir_paths["marker"] = "only this instance"
assert OtherInspectionDSL().dump_mlir_paths == {}
print("INSPECTION CONTRACT: empty state before the first trace")
# CHECK-NEXT: INSPECTION CONTRACT: empty state before the first trace
