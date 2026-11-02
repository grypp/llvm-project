# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``type_ops`` role: the MLIR types of the core dtypes and the ops behind
their operators (``core.plugin.TypeOpsPlugin``).

The shipped plugin is :class:`UpstreamDialectTypeOps` (``upstream_dialects.py``),
a composer over the op modules beside it, each emitting one in-tree MLIR
dialect: ``arith.py`` (scalars; ``math`` behind pow, abs and floor),
``vector.py`` (vectors) and ``llvm.py`` (pointers and memory). A DSL names it
as ``type_ops=UpstreamDialectTypeOps(scalars=arith, vectors=vector,
memory=llvm)``, any part optional, or brings its own ``TypeOpsPlugin``.
"""

from .upstream_dialects import UpstreamDialectTypeOps

__all__ = ["UpstreamDialectTypeOps"]
