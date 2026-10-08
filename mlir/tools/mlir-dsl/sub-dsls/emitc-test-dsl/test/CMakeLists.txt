llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  MLIR_ENABLE_EXECUTION_ENGINE
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_TOOLS_DIR"
  "MLIR_BINARY_DIR"
  "CMAKE_CURRENT_SOURCE_DIR"
  "CMAKE_CURRENT_BINARY_DIR"
)

set(MLIR_DSL_TEST_DEPENDS
  FileCheck count not
  MLIRPythonModules
  MLIRPythonModulesEmitCTestDSL
  mlir-translate
)
if(MLIR_ENABLE_EXECUTION_ENGINE)
  # Provides the `host-supports-jit` feature probe.
  list(APPEND MLIR_DSL_TEST_DEPENDS mlir-runner)
endif()

add_lit_testsuite(check-mlir-dsl-emitc-test-dsl "Running the mlir.emitc_dsl sub-DSL tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_DSL_TEST_DEPENDS}
)
set_target_properties(check-mlir-dsl-emitc-test-dsl PROPERTIES FOLDER "MLIR/Tests")

add_lit_testsuites(MLIR_DSL_EMITC_TEST_DSL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_DSL_TEST_DEPENDS})

# `check-mlir-dsl` covers the sub-DSL tests too.
add_dependencies(check-mlir-dsl check-mlir-dsl-emitc-test-dsl)
