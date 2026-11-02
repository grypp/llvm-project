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
def takes_reserved(
    *,
    pipeline=None,
    container_attrs=None,
    no_cache=None,
    extra_link_libs=None,
    compile_only=None,
):
    entered.append("reserved")


def rejected(name, value):
    try:
        # Defaults keep missing arguments from masking reserved-name failures.
        takes_reserved(**{name: value})
    except DSLUserCodeError as error:
        assert error.diag_id.name == "CALL_ARGUMENTS"
        print("RESERVED:", name, repr(value))
    else:
        raise AssertionError(f"reserved keyword {name} bound to a user parameter")
    assert not entered
    assert get_current_dsl() is None


rejected("pipeline", None)
rejected("container_attrs", {})
rejected("no_cache", False)
rejected("extra_link_libs", ())
rejected("compile_only", False)
rejected("no_cache", None)
rejected("compile_only", None)
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
