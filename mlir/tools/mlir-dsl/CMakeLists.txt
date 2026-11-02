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
    dsl/core/common.py
    dsl/core/diagnostics.py
    dsl/core/dsl.py
    dsl/core/plugin.py
    dsl/plugins/__init__.py
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

if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
