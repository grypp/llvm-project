# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``dlpack`` plugin: any object speaking the DLPack protocol as a ``Pointer``."""

from .plugin import DlpackPlugin, DlpackTensor, available, speaks_dlpack

__all__ = ["DlpackPlugin", "DlpackTensor", "available", "speaks_dlpack"]
