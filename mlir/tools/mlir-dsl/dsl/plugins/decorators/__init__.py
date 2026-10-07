# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The ``decorators`` family: plugins adding a decorator and its launcher
(``core.plugin.DecoratorPlugin``). ``kernels/`` ships ``@kernel``: the generic
launcher in ``kernels/launch.py`` and the CUDA target ``kernels/gpu/``
(``gpu.Kernels`` in ``gpu/__init__.py``, the index ops in ``gpu/indices.py``).
A third decorator is another ``DecoratorPlugin`` here.
"""
