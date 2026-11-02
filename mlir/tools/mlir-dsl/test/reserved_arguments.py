# RUN: %PYTHON %s 2>&1 | FileCheck %s
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reserved framework keywords cannot bind to user parameters, even as None or False."""

from mlir.dsl import BaseDSL, DSLUserCodeError, Plugins
from mlir.dsl.core.common import get_current_dsl
from mlir.dsl.plugins.decorators.jit import func


# ===----------------------------------------------------------------------===
# ===----               DSL Implementation                         ====----===
# ===----------------------------------------------------------------------===


class TestDSL(BaseDSL):
    plugins = Plugins(decorators=[func.Jit()])

    def __init__(self):
        super().__init__(name="RESERVED_ARGUMENTS_DSL")


# ===----------------------------------------------------------------------===
# ===----               DSL Test Cases                             ====----===
# ===----------------------------------------------------------------------===


entered = []


@TestDSL.jit
def takes_pipeline(*, pipeline):
    entered.append("pipeline")


@TestDSL.jit
def takes_container_attrs(*, container_attrs):
    entered.append("container_attrs")


@TestDSL.jit
def takes_no_cache(*, no_cache):
    entered.append("no_cache")


@TestDSL.jit
def takes_extra_link_libs(*, extra_link_libs):
    entered.append("extra_link_libs")


@TestDSL.jit
def takes_compile_only(*, compile_only):
    entered.append("compile_only")


def rejected(name, function, value):
    try:
        function(**{name: value})
    except DSLUserCodeError as error:
        assert error.diag_id.name == "CALL_ARGUMENTS"
        print("RESERVED:", name, repr(value))
    else:
        raise AssertionError(f"reserved keyword {name} bound to a user parameter")
    assert not entered
    assert get_current_dsl() is None


rejected("pipeline", takes_pipeline, None)
rejected("container_attrs", takes_container_attrs, {})
rejected("no_cache", takes_no_cache, False)
rejected("extra_link_libs", takes_extra_link_libs, ())
rejected("compile_only", takes_compile_only, False)
rejected("no_cache", takes_no_cache, None)
rejected("compile_only", takes_compile_only, None)
# CHECK: RESERVED: pipeline None
# CHECK-NEXT: RESERVED: container_attrs {}
# CHECK-NEXT: RESERVED: no_cache False
# CHECK-NEXT: RESERVED: extra_link_libs ()
# CHECK-NEXT: RESERVED: compile_only False
# CHECK-NEXT: RESERVED: no_cache None
# CHECK-NEXT: RESERVED: compile_only None


@TestDSL.jit
def ordinary(*, text, flag):
    entered.append((text, flag))


ordinary(text="plain Meta", flag=False)
ordinary(text="another Meta", flag=None)
assert entered == [("plain Meta", False), ("another Meta", None)]
assert get_current_dsl() is None
print("ORDINARY META:", entered)
# CHECK-NEXT: ORDINARY META: [('plain Meta', False), ('another Meta', None)]
