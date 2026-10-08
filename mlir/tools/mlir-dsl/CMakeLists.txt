# The Python DSL base layer ``mlir.dsl`` (``dsl/``: the core, the dialect op
# modules, the role plugins and the plugin families) and its test sub-DSL
# ``mlir.mlir_dsl`` (``sub-dsls/mlir-dsl/dsl/``). A file is installed into the
# ``mlir`` Python package at its path relative to ROOT_DIR: the core's directory
# therefore carries the package name, while the sub-DSL has a module target of
# its own that installs it under ``mlir_dsl/``. This directory is added from
# mlir/python after MLIRPythonSources is declared.
declare_mlir_python_sources(MLIRPythonSources.DSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources
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
    dsl/core/remarks.py
    dsl/core/staging.py
    dsl/core/user_op.py
    dsl/plugins/__init__.py
    dsl/plugins/ast_preprocessor/__init__.py
    dsl/plugins/ast_preprocessor/helpers.py
    dsl/plugins/ast_preprocessor/preprocessor.py
    dsl/plugins/ast_preprocessor/scf/__init__.py
    dsl/plugins/ast_preprocessor/scf/builders.py
    dsl/plugins/ast_preprocessor/scf/executors.py
    dsl/plugins/compiler/__init__.py
    dsl/plugins/compiler/execution_engine.py
    dsl/plugins/compiler/jit_executor.py
    dsl/plugins/adapters/__init__.py
    dsl/plugins/adapters/dlpack/__init__.py
    dsl/plugins/adapters/dlpack/plugin.py
    dsl/plugins/adapters/tvm_ffi/__init__.py
    dsl/plugins/adapters/tvm_ffi/call_provider.py
    dsl/plugins/adapters/tvm_ffi/diagnostics.py
    dsl/plugins/adapters/tvm_ffi/mlir_builder.py
    dsl/plugins/adapters/tvm_ffi/plugin.py
    dsl/plugins/adapters/tvm_ffi/spec.py
    dsl/plugins/adapters/tvm_ffi/tvm_ffi_builder.py
    dsl/plugins/decorators/__init__.py
    dsl/plugins/decorators/jit/__init__.py
    dsl/plugins/decorators/jit/func.py
    dsl/plugins/decorators/kernels/__init__.py
    dsl/plugins/decorators/kernels/gpu/__init__.py
    dsl/plugins/decorators/kernels/gpu/indices.py
    dsl/plugins/decorators/kernels/launch.py
    dsl/plugins/type_ops/__init__.py
    dsl/plugins/type_ops/arith.py
    dsl/plugins/type_ops/llvm.py
    dsl/plugins/type_ops/upstream_dialects.py
    dsl/plugins/type_ops/vector.py
    dsl/types/__init__.py
    dsl/types/typing.py
    dsl/types/vector.py
    dsl/util/__init__.py
    dsl/util/cache.py
    dsl/util/logger.py
    dsl/util/profiler.py
    dsl/util/tree_utils.py
)

# The ``dlpack`` adapter plugin imports DLPack tensors through nanobind's
# ndarray; the extension needs no C API and no library of its own.
declare_mlir_python_extension(MLIRPythonExtension.DSL.Dlpack
  MODULE_NAME _mlirDslDlpack
  ADD_TO_PARENT MLIRPythonSources.DSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dsl/plugins/adapters/dlpack/csrc"
  SOURCES
    DlpackPlugin.cpp
)

# The test sub-DSL: ``mlir.mlir_dsl``, a sibling package of the core. Its
# sources live in ``sub-dsls/mlir-dsl/dsl/``; a module target of its own
# installs them under the package name, ``<prefix>/mlir_dsl/``, so the
# directory name is free (``MLIRPythonModules`` depends on this target).
declare_mlir_python_sources(MLIRPythonSources.DSL.MlirTestDSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sub-dsls/mlir-dsl/dsl"
  SOURCES
    __init__.py
    mlir_dsl.py
)
add_mlir_python_modules(MLIRPythonModulesMlirTestDSL
  ROOT_PREFIX "${MLIRPythonModules_ROOT_PREFIX}/mlir_dsl"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/mlir_dsl"
  DECLARED_SOURCES MLIRPythonSources.DSL.MlirTestDSL
)

if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(sub-dsls/mlir-dsl/test)
endif()
