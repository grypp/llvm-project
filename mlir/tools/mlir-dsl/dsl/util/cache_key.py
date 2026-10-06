# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The compile cache key: a hash of the module, the pipeline, the compile-affecting
environment and the identity of the MLIR binaries in use."""

import functools
import hashlib
import io
import os
from collections.abc import Sequence
from pathlib import Path

from ... import _mlir_libs, ir

__all__ = ["module_cache_key", "toolchain_identity"]


@functools.cache
def toolchain_identity() -> tuple[tuple[str, int, int], ...]:
    """Identify the MLIR binaries in use as ``(path, size, mtime_ns)`` triples.

    Folded into the compile key so a rebuilt toolchain never serves a stale
    engine; the files are stat'ed once, never read.
    """
    binaries: list[tuple[str, int, int]] = []
    for directory in _mlir_libs.__path__:
        for path in sorted(Path(directory).glob("*")):
            if path.is_file() and (
                ".so" in path.name or path.suffix in (".pyd", ".dylib")
            ):
                stat = path.stat()
                binaries.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(binaries)


def module_cache_key(
    module: ir.Module,
    pipeline: str,
    cache_key_str: str = "",
    *,
    extra: Sequence[str] = (),
) -> str:
    """Hash of the module bytecode, the pipeline, the compile-affecting
    environment (``EnvVarSpec.cache_key_str``) and the toolchain identity."""
    s = io.BytesIO()
    module.operation.write_bytecode(s)
    s.write(b"\0pipeline\0")
    s.write(pipeline.encode())
    s.write(b"\0env\0")
    s.write(cache_key_str.encode())
    for item in extra:
        s.write(b"\0extra\0")
        s.write(os.fsencode(item))
    hash_obj = hashlib.sha256(repr(toolchain_identity()).encode())
    hash_obj.update(s.getvalue())
    return hash_obj.hexdigest()
