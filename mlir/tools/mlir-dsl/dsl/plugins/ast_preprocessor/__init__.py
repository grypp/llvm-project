# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AST preprocessor plugins: how Python syntax maps onto a dialect.

``preprocessor.py`` is the rewrite (``DSLPreprocessor``) and ``helpers.py``
the callbacks the rewritten code calls; ``scf.py`` is the plugin that stages
native control flow as ``scf``. Without an ``ASTPreprocessorPlugin`` a DSL
does not rewrite and uses the explicit builders.
"""

from .helpers import range
from .preprocessor import DSLPreprocessor
from .scf import ScfASTPreprocessorPlugin

__all__ = ["DSLPreprocessor", "ScfASTPreprocessorPlugin", "range"]
