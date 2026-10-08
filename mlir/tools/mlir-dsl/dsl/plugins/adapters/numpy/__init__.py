# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""numpy plugin of ``mlir.dsl``: ``ndarray`` arguments as host pointers."""

from .plugin import NumpyPlugin, available

__all__ = ["NumpyPlugin", "available"]
