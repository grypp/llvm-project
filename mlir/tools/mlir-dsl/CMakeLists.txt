# Python DSL infrastructure and optional feature implementations.
declare_mlir_python_sources(MLIRPythonSources.DSL
  ADD_TO_PARENT MLIRPythonSources)
declare_mlir_python_sources(MLIRPythonSources.DSL.Python
  ADD_TO_PARENT MLIRPythonSources.DSL)

declare_mlir_python_sources(MLIRPythonSources.DSL.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources.DSL.Python
  SOURCES
    dsl/__init__.py
    dsl/core/__init__.py
    dsl/core/arguments.py
    dsl/core/common.py
    dsl/core/diagnostics.py
    dsl/core/dsl.py
    dsl/core/env_manager.py
    dsl/core/mlir_op.py
    dsl/core/plugin.py
    dsl/core/staging.py
    dsl/plugins/__init__.py
    dsl/types/__init__.py
    dsl/types/typing.py
    dsl/util/__init__.py
    dsl/util/tree_utils.py
)

declare_mlir_python_sources(MLIRPythonSources.DSL.decorators
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources.DSL.Python
  SOURCES
    dsl/plugins/decorators/__init__.py
)

declare_mlir_python_sources(MLIRPythonSources.DSL.decorators.jit
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources.DSL.Python
  SOURCES
    dsl/plugins/decorators/jit/__init__.py
    dsl/plugins/decorators/jit/func.py
)

declare_mlir_python_sources(MLIRPythonSources.DSL.typeops
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources.DSL.Python
  SOURCES
    dsl/plugins/type_ops/__init__.py
    dsl/plugins/type_ops/arith.py
    dsl/plugins/type_ops/upstream_dialects.py
)

declare_mlir_python_sources(MLIRPythonSources.DSL.MlirTestDSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sub-dsls/mlir-test-dsl/dsl"
  SOURCES __init__.py mlir_dsl.py
)
add_mlir_python_modules(MLIRPythonModulesMlirTestDSL
  ROOT_PREFIX "${MLIRPythonModules_ROOT_PREFIX}/mlir_dsl"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/mlir_dsl"
  DECLARED_SOURCES MLIRPythonSources.DSL.MlirTestDSL
)

declare_mlir_python_sources(MLIRPythonSources.DSL.EmitCTestDSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sub-dsls/emitc-test-dsl/dsl"
  SOURCES __init__.py emitc_dsl.py
)
add_mlir_python_modules(MLIRPythonModulesEmitCTestDSL
  ROOT_PREFIX "${MLIRPythonModules_ROOT_PREFIX}/emitc_dsl"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/emitc_dsl"
  DECLARED_SOURCES MLIRPythonSources.DSL.EmitCTestDSL
)

if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(sub-dsls/mlir-test-dsl/test)
  add_subdirectory(sub-dsls/emitc-test-dsl/test)
endif()
