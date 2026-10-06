# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plugins of ``mlir.dsl``, in three families.

``dialects/`` are the IR worlds (``llvm``, ``scf``, ``gpu``): what the types
and control flow emit and how it lowers. ``ast_preprocessor/`` maps Python
syntax onto them (the rewrite and the ``scf`` plugin). ``thirdparty/`` adapts
external packages (``pytorch``, ``dlpack``, ``tvm_ffi``). A plugin is selected
by listing it on a DSL class; importing this package pulls in none of them and
the core imports none at import time. Every plugin is usable by any
``BaseDSL`` subclass; ``MlirDSL`` is one assembly of them.
"""

from ..core.plugin import ASTPreprocessorPlugin, DialectPlugin, Plugin

__all__ = ["ASTPreprocessorPlugin", "DialectPlugin", "Plugin"]
