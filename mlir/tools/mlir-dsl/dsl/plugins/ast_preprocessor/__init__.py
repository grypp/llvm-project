# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``ast_preprocessor`` role: how Python syntax maps onto a dialect.

``preprocessor.py`` is the rewrite (``DSLPreprocessor``) and ``helpers.py``
the callbacks the rewritten code calls; ``scf/`` is the plugin that stages
native control flow as ``scf`` (``scf/__init__.py`` holds ``ASTPreprocessor``,
``scf/builders.py`` the explicit builders, ``scf/executors.py`` the executors). Without an ``ast_preprocessor`` plugin a DSL does not rewrite
and uses the explicit builders.
"""

from .helpers import range
from .preprocessor import DSLPreprocessor
from .scf import ASTPreprocessor

__all__ = ["ASTPreprocessor", "DSLPreprocessor", "range"]
