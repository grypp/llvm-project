# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``compiler`` role: ``execution_engine.Compiler`` runs the pipeline on
MLIR's pass manager and executes through the ``ExecutionEngine``;
``jit_executor`` binds the packed entry it produces into a callable. Nothing
here is imported until a DSL names the plugin (``Plugins(compiler=
execution_engine.Compiler())``), so the engine's bindings load on first use."""
