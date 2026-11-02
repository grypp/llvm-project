# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_DEBUGINFO=1 MLIR_DSL_VERIFY_TRACE=1 %PYTHON %s 2>&1 | FileCheck %s
# Settings composition, diagnostics, and explicit source-location propagation.
import os
import mlir.mlir_dsl as m
from mlir import ir
from mlir.dialects import arith
from mlir.dsl.core.env_manager import EnvironmentVarManager, env_var
from mlir.dsl.core.user_op import dsl_user_op


class Settings(EnvironmentVarManager):
    count: int = env_var("COUNT", affects_compile=True, default=7)


os.environ["SERIES_COUNT"] = "11"
assert Settings("SERIES").count == 11
os.environ["SERIES_COUNT"] = "bad"
try:
    Settings("SERIES")
except m.DSLUserCodeError as exc:
    assert exc.code == "CONFIG_INVALID"
    assert "SERIES_COUNT" in str(exc)
else:
    raise AssertionError("an invalid integer setting was accepted")
print("SETTING: invalid value diagnosed")
# CHECK: SETTING: invalid value diagnosed


@dsl_user_op
def user_add(a, b, *, loc=None):
    result = arith.AddIOp(a, b, loc=loc).result
    assert "caller-location" in str(result.owner.location)
    return result


@m.jit
def add(n: m.Int32) -> m.Int32:
    return m.Int32(
        user_add(n.ir_value(), n.ir_value(), loc=ir.Location.name("caller-location"))
    )


add(3)
print("LOCATION: explicit caller location preserved")
# CHECK: LOCATION: explicit caller location preserved
