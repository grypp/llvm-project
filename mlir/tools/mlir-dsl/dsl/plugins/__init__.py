# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The plugins a DSL is assembled from: one folder per role, one per family.

A plugin fills or extends a core role; a module emits ops. The folders mirror
the fields of the ``Plugins`` record a DSL names on its class:

* roles, one plugin each: ``type_ops/`` (the ``TypeOps`` composer over dialect
  modules), ``func_entry/`` (``func.Entry``), ``ast_preprocessor/``
  (``scf.ASTPreprocessor``), ``compiler/`` (``execution_engine.Compiler``);
* families, any number each: ``decorators/`` (``kernels/gpu.Kernels`` adds
  ``@kernel`` and its launcher), ``adapters/`` (the host boundary: ``pytorch``
  and ``dlpack`` turn host objects into arguments, ``tvm_ffi`` exposes the
  compiled entry through another ABI).

Each folder also ships the op modules its plugin emits through (``type_ops/``:
``arith``, ``vector``, ``llvm``; ``ast_preprocessor/scf/``: the
``scf`` builders and executors; ``decorators/kernels/gpu/``: the kernel-body
index ops). Those are modules, not plugins: a plugin folder owns the ops it
emits and never imports another folder's ops. A dialect a DSL only emits ops
from needs no plugin at all. Importing this package imports no
concrete plugin; it re-exports the bases of ``mlir.dsl.core.plugin``.
"""

from ..core.plugin import (
    ASTPreprocessorPlugin,
    AdapterPlugin,
    CompilerPlugin,
    DecoratorPlugin,
    FuncEntryPlugin,
    Plugin,
    Plugins,
    TypeOpsPlugin,
)

__all__ = [
    "ASTPreprocessorPlugin",
    "AdapterPlugin",
    "CompilerPlugin",
    "DecoratorPlugin",
    "FuncEntryPlugin",
    "Plugin",
    "Plugins",
    "TypeOpsPlugin",
]
