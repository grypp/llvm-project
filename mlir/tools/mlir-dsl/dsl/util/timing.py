# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wall-clock timing of calls (``<PREFIX>_JIT_TIME_PROFILING``)."""

import functools
from time import perf_counter
from typing import Any

from .logger import log

__all__ = ["timer"]


def timer(*dargs: Any, enable: bool = True) -> Any:
    """Log the wall time of every call of the decorated function, in microseconds."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enable:
                return func(*args, **kwargs)
            start = perf_counter()
            result = func(*args, **kwargs)
            spend_us = (perf_counter() - start) * 1e6
            label = getattr(func, "__name__", None) or f"C API Function: {func}"
            log().info("[JIT-TIMER] %s | Execution Time: %.2f us", label, spend_us)
            return result

        return func_wrapper

    if len(dargs) == 1 and callable(dargs[0]):
        return decorator(dargs[0])
    return decorator
