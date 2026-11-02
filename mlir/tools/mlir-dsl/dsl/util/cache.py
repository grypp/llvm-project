# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The compile cache: its key, the in-memory level and the on-disk level.

A compiled function is cached under a module hash (``BaseDSL.get_module_hash``:
the module bytecode, the pipeline, the compile-affecting environment and
:func:`toolchain_identity`, so a rebuilt toolchain never serves a stale
engine). The in-memory level, :class:`JitCacheDict`, maps the hash to the
``JitCompiledFunction`` for this process. The on-disk level keeps the *lowered*
module of each hash as bytecode with a CRC32 trailer under the cache directory
(``<PREFIX>_CACHE_DIR``), so a later process skips the pass pipeline and only
builds its ``ExecutionEngine``. :func:`save_ir` and :func:`load_ir` are its
bytecode writer and reader; ``BaseDSL._save_ir`` writes the textual ``KEEP_IR``
dumps into the same directory.
"""

import functools
import threading
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
from ... import _mlir_libs, ir
from ..core.common import DSLRuntimeError

if TYPE_CHECKING:
    from ..plugins.compiler.jit_executor import JitCompiledFunction

__all__ = [
    "JitCacheDict",
    "toolchain_identity",
]


# =============================================================================
# The key: the identity of the toolchain
# =============================================================================


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


# =============================================================================
# The in-memory level: module hash -> compiled function, for this process
# =============================================================================


class JitCacheDict:
    """A dictionary of :class:`JitCompiledFunction` objects keyed by module hash.

    An entry registered with an ``owner`` is dropped when that object is
    garbage collected, so compiled functions do not leak. With
    ``max_elems`` set the dictionary evicts least-recently-used entries.

    Every operation runs under the dictionary's own re-entrant lock, so the
    cache is consistent across threads and an owner's finalizer, which the
    collector may run from inside any cache call, can delete its entry
    without deadlocking. A displaced value (overwritten, evicted, deleted or
    cleared) is released only after the lock is, because its finalizer runs
    arbitrary code and must not run under the cache lock.

    :param max_elems: Capacity; ``None`` is unlimited, ``0`` disables the cache
    """

    def __init__(self, max_elems: int | None = None) -> None:
        self._dict: OrderedDict[Any, tuple[Any, weakref.finalize | None]] = (
            OrderedDict()
        )
        self.max_elems = max_elems
        self._lock = threading.RLock()

    def get(self, key: Any) -> Any | None:
        """The cached value for ``key`` (moved to most-recently-used), or None."""
        if self.max_elems == 0:
            return None
        with self._lock:
            entry = self._dict.get(key)
            if entry is None:
                return None
            if self.max_elems is not None:
                self._dict.move_to_end(key, last=True)
            return entry[0]

    def set(self, key: Any, value: Any, owner: Any = None) -> None:
        """Store ``value`` under ``key``, tied to ``owner``'s lifetime."""
        if self.max_elems == 0:
            return
        if value is owner:
            raise DSLRuntimeError(
                "value and owner cannot be the same object to avoid circular references"
            )

        def _remove_entry(
            k: Any, self_ref: weakref.ref[JitCacheDict] = weakref.ref(self)
        ) -> None:
            # Called from GC/finalizer; be defensive and avoid raising.
            self_obj = self_ref()
            if self_obj is not None:
                self_obj.delete(k)

        # The entries this store displaces (the previous value of ``key`` and
        # the evicted ones) stay referenced until the lock is released.
        displaced: list[tuple[Any, weakref.finalize | None]] = []
        with self._lock:
            # Detach the existing finalizer for this key so that collection of
            # the old owner cannot remove the new entry.
            if key in self._dict:
                displaced.append(self._dict.pop(key))
            self._dict[key] = (
                value,
                (
                    None
                    if owner is None
                    else weakref.finalize(owner, _remove_entry, key)
                ),
            )
            if self.max_elems is not None:
                while len(self._dict) > self.max_elems:
                    displaced.append(self._dict.popitem(last=False)[1])
            for _, finalize in displaced:
                if finalize is not None:
                    finalize.detach()

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._dict

    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)

    def delete(self, key: Any) -> None:
        """Drop ``key`` if present, detaching its finalizer."""
        with self._lock:
            entry = self._dict.pop(key, None)
            if entry is not None and entry[1] is not None:
                entry[1].detach()
        # ``entry`` keeps the value alive until here, past the lock.

    def clear(self) -> None:
        """Drop every entry, detaching the finalizers."""
        with self._lock:
            displaced = list(self._dict.values())
            self._dict.clear()
            for _, finalize in displaced:
                if finalize is not None:
                    finalize.detach()
