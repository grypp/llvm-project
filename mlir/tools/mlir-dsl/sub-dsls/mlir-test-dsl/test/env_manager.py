# RUN: %PYTHON %s 2>&1 | FileCheck %s
import logging
import os
from mlir.dsl.core.env_manager import EnvironmentVarManager
from mlir.dsl.util.logger import log

os.environ['KEEP_LOG_LOG_TO_CONSOLE'] = '1'
os.environ['KEEP_LOG_LOG_LEVEL'] = '20'
EnvironmentVarManager('KEEP_LOG')
EnvironmentVarManager('QUIET_LOG')
print('KEPT:', log().name, logging.getLogger('QUIET_LOG').level)
# CHECK: KEPT: KEEP_LOG 51
os.environ.pop('KEEP_LOG_LOG_TO_CONSOLE')
os.environ.pop('KEEP_LOG_LOG_LEVEL')
EnvironmentVarManager('KEEP_LOG')
print('OFF:', log().name, log().level)
# CHECK: OFF: KEEP_LOG 51
