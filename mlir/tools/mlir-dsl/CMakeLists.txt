# The Python DSL base layer ``mlir.dsl`` (``dsl/``) and its reference sub-DSL
# ``mlir.mlir_dsl`` (``sub-dsls/mlir-dsl/mlir_dsl/``). A file is installed into
# the ``mlir`` Python package at its path relative to ROOT_DIR, so each package
# directory carries the package's name; this directory is added from
# mlir/python after MLIRPythonSources is declared.
declare_mlir_python_sources(MLIRPythonSources.DSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    dsl/__init__.py
    dsl/compiler/__init__.py
    dsl/compiler/compiler.py
    dsl/compiler/jit_executor.py
    dsl/compiler/remarks.py
    dsl/core/__init__.py
    dsl/core/common.py
    dsl/core/diagnostics.py
    dsl/core/dsl.py
    dsl/core/env_manager.py
    dsl/core/executor.py
    dsl/core/mlir_op.py
    dsl/core/plugin.py
    dsl/core/user_op.py
    dsl/plugins/__init__.py
    dsl/plugins/ast_preprocessor/__init__.py
    dsl/plugins/ast_preprocessor/helpers.py
    dsl/plugins/ast_preprocessor/preprocessor.py
    dsl/plugins/ast_preprocessor/scf.py
    dsl/plugins/dialects/__init__.py
    dsl/plugins/dialects/gpu.py
    dsl/plugins/dialects/llvm/__init__.py
    dsl/plugins/dialects/llvm/arith.py
    dsl/plugins/dialects/llvm/emitter.py
    dsl/plugins/dialects/llvm/entry.py
    dsl/plugins/dialects/llvm/math.py
    dsl/plugins/dialects/llvm/memory.py
    dsl/plugins/dialects/llvm/plugin.py
    dsl/plugins/dialects/scf/__init__.py
    dsl/plugins/dialects/scf/builders.py
    dsl/plugins/dialects/scf/executors.py
    dsl/plugins/dialects/scf/plugin.py
    dsl/plugins/thirdparty/__init__.py
    dsl/plugins/thirdparty/dlpack/__init__.py
    dsl/plugins/thirdparty/dlpack/plugin.py
    dsl/plugins/thirdparty/pytorch/__init__.py
    dsl/plugins/thirdparty/pytorch/dtypes.py
    dsl/plugins/thirdparty/pytorch/plugin.py
    dsl/plugins/thirdparty/tvm_ffi/__init__.py
    dsl/plugins/thirdparty/tvm_ffi/call_provider.py
    dsl/plugins/thirdparty/tvm_ffi/mlir_builder.py
    dsl/plugins/thirdparty/tvm_ffi/plugin.py
    dsl/plugins/thirdparty/tvm_ffi/spec.py
    dsl/plugins/thirdparty/tvm_ffi/tvm_ffi_builder.py
    dsl/runtime/__init__.py
    dsl/runtime/jit_arg_adapters.py
    dsl/types/__init__.py
    dsl/types/typing.py
    dsl/types/vector.py
    dsl/util/__init__.py
    dsl/util/cache_helpers.py
    dsl/util/cache_key.py
    dsl/util/logger.py
    dsl/util/phase_profiler.py
    dsl/util/timing.py
    dsl/util/tree_utils.py
)

# The ``dlpack`` plugin imports DLPack tensors through nanobind's ndarray; the
# extension needs no C API and no library of its own.
declare_mlir_python_extension(MLIRPythonExtension.DSL.Dlpack
  MODULE_NAME _mlirDslDlpack
  ADD_TO_PARENT MLIRPythonSources.DSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dsl/plugins/thirdparty/dlpack/csrc"
  SOURCES
    DlpackPlugin.cpp
)

# The reference sub-DSL: ``mlir.mlir_dsl``, a sibling package of the core.
declare_mlir_python_sources(MLIRPythonSources.DSL.MlirDSL
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sub-dsls/mlir-dsl"
  ADD_TO_PARENT MLIRPythonSources.DSL
  SOURCES
    mlir_dsl/__init__.py
    mlir_dsl/mlir_dsl.py
)

if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(sub-dsls/mlir-dsl/test)
endif()
