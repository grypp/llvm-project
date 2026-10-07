# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PyTorch plugin of ``mlir.dsl``: tensor arguments and the dtype bridge."""

from .dtypes import dtype, from_torch_dtype
from .plugin import PyTorchPlugin, available

__all__ = ["PyTorchPlugin", "available", "dtype", "from_torch_dtype"]
