# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernels: decorator plugins adding ``@kernel`` and its launcher to a DSL.

The core knows one decorator, ``@jit``. A DSL with kernels lists a kernels
plugin in ``Plugins(decorators=[...])``: ``launch.py`` is the generic part (the
``kernel`` decorator, the deferred :class:`KernelLauncher`, :class:`LaunchConfig`,
the per-trace bookkeeping and the entry protocol a target implements), ``gpu/``
the CUDA target over the ``gpu`` dialect (``gpu.Kernels`` in ``gpu/__init__.py``,
the kernel-body index ops in ``gpu/indices.py``). A third decorator
follows the same shape: a ``DecoratorPlugin`` whose ``decorators`` returns it
and whose launcher says what calling the decorated function does.
"""
