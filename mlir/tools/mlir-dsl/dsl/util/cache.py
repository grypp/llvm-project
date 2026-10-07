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
import getpass
import io
import os
import sys
import tempfile
import uuid
import weakref
import zlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ... import _mlir_libs, ir
from ..core.common import DSLRuntimeError
from .logger import log

if TYPE_CHECKING:
    from ..plugins.compiler.jit_executor import JitCompiledFunction

__all__ = [
    "JitCacheDict",
    "dump_cache_to_path",
    "get_default_generated_ir_path",
    "load_cache_from_path",
    "read_bytecode_and_check_crc32",
    "toolchain_identity",
    "write_bytecode_with_crc32",
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

    :param max_elems: Capacity; ``None`` is unlimited, ``0`` disables the cache
    """

    def __init__(self, max_elems: int | None = None) -> None:
        self._dict: OrderedDict[
            Any, tuple[Any, weakref.finalize | None]
        ] = OrderedDict()
        self.max_elems = max_elems

    def get(self, key: Any) -> Any | None:
        """The cached value for ``key`` (moved to most-recently-used), or None."""
        if self.max_elems == 0:
            return None
        value = self._dict.get(key)
        if value is None:
            return None
        obj, _ = value
        if self.max_elems is not None:
            self._dict.move_to_end(key, last=True)
        return obj

    def set(self, key: Any, value: Any, owner: Any = None) -> None:
        """Store ``value`` under ``key``, tied to ``owner``'s lifetime."""
        if self.max_elems == 0:
            return
        if value is owner:
            raise DSLRuntimeError(
                "value and owner cannot be the same object to avoid circular references"
            )

        # Detach any existing finalizer for this key so that collection of the
        # old value cannot accidentally remove or interfere with the new entry.
        old = self._dict.get(key)
        if old is not None:
            _, old_finalize = old
            if old_finalize is not None:
                old_finalize.detach()

        def _remove_entry(
            k: Any, self_ref: weakref.ref[JitCacheDict] = weakref.ref(self)
        ) -> None:
            # Called from GC/finalizer; be defensive and avoid raising.
            self_obj = self_ref()
            if self_obj is not None:
                self_obj.delete(k)

        self._dict[key] = (
            value,
            (None if owner is None else weakref.finalize(owner, _remove_entry, key)),
        )
        if self.max_elems is not None:
            self._dict.move_to_end(key, last=True)
            while len(self._dict) > self.max_elems:
                _, (_, finalize) = self._dict.popitem(last=False)
                if finalize is not None:
                    finalize.detach()

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def delete(self, key: Any) -> None:
        """Drop ``key`` if present, detaching its finalizer."""
        entry = self._dict.pop(key, None)
        if entry is not None:
            _, finalize = entry
            if finalize is not None:
                finalize.detach()

    def clear(self) -> None:
        """Drop every entry, detaching the finalizers."""
        for _, finalize in self._dict.values():
            if finalize is not None:
                finalize.detach()
        self._dict.clear()


# =============================================================================
# The on-disk level: the lowered module of each hash, across processes
# =============================================================================


def get_current_user() -> str:
    """The current user's name, for a per-user cache directory under the temp
    dir; the numeric uid (or ``"user"``) when there is no login database."""
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 -- no login database inside some sandboxes
        return str(os.getuid()) if hasattr(os, "getuid") else "user"


def normalize_path(path: str | Path) -> Path:
    """Expand ``~``, resolve symlinks and relative segments.

    :param path: The path to normalize
    :return: The absolute, resolved ``Path``
    """
    return Path(os.path.realpath(os.path.expanduser(str(path))))


def get_default_generated_ir_path(dsl_name: str = "MLIR_DSL") -> str:
    """The cache directory: IR dumps and the file cache go here.

    :param dsl_name: The DSL's name (``BaseDSL.name``); ``<dsl_name>_CACHE_DIR``
        names the directory, and the lowered name is used for the default one
    :return: ``$<dsl_name>_CACHE_DIR`` if set, else
        ``$TMPDIR/<user>/<dsl_name lower>_cache``, created on demand
        (``$TMPDIR/<dsl_name lower>_cache`` when the per-user directory cannot
        be created)
    """
    if cache_dir := os.environ.get(f"{dsl_name}_CACHE_DIR"):
        return cache_dir
    tmp_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    dir_name = f"{dsl_name.lower()}_cache"

    def get_reusable_temp_dir(name: str) -> str:
        p = tmp_dir / get_current_user() / name
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    try:
        return get_reusable_temp_dir(dir_name)
    except Exception as e:  # noqa: BLE001 -- fall back rather than fail a compile
        fallback = str(tmp_dir / dir_name)
        log().warning(
            "Could not determine user or create cache directory, using fallback path %s. Error: %s",
            fallback,
            e,
        )
        return fallback


def write_bytecode_with_crc32(f: io.BufferedIOBase, module: ir.Module) -> None:
    """Write the module's bytecode followed by its CRC32 (4 bytes, native
    byte order) to ``f``.

    :param f: The binary file to write to
    :param module: The module to serialize
    """
    s = io.BytesIO()
    module.operation.write_bytecode(s)
    content = s.getvalue()
    crc = zlib.crc32(content)
    s.write(crc.to_bytes(4, sys.byteorder))
    f.write(s.getvalue())


def read_bytecode_and_check_crc32(f: io.BufferedReader) -> ir.Module:
    """Read bytecode written by :func:`write_bytecode_with_crc32` and verify it.

    The module is parsed into the current context (the compile's), so it can
    go straight into an ``ExecutionEngine``.

    :param f: The binary file holding the bytecode and its CRC32 trailer
    :return: The parsed module
    :raises DSLRuntimeError: The file is too short or the checksum mismatches
    """
    content = f.read()
    if len(content) < 4:
        raise DSLRuntimeError(
            f"File {f.name} does not contain enough data for CRC32 checksum."
        )
    bytecode = content[:-4]
    crc_appended = content[-4:]
    crc_appended_int = int.from_bytes(crc_appended, sys.byteorder)
    crc_computed = zlib.crc32(bytecode)
    if crc_appended_int != crc_computed:
        raise DSLRuntimeError(
            f"CRC32 checksum mismatch! Expected {crc_computed}, got {crc_appended_int}"
        )
    return ir.Module.parse(bytecode)


def load_ir(file: str, bytecode_reader: Callable[..., Any] | None = None) -> ir.Module:
    """Load a module saved by :func:`save_ir` into the current context.

    :param file: The path of a ``.mlir`` bytecode file written by :func:`save_ir`
    :param bytecode_reader: Parses the open binary file into a module (e.g.
        :func:`read_bytecode_and_check_crc32`); ``ir.Module.parse`` by default
    :return: The module
    :raises DSLRuntimeError: ``file`` is not a ``.mlir`` path
    """
    if ".mlir" not in file:
        raise DSLRuntimeError(
            "generated IR is loaded from `.mlir` files only", context={"file": file}
        )
    with open(file, "rb") as f:
        if bytecode_reader:
            return bytecode_reader(f)
        return ir.Module.parse(f.read())


def save_ir(
    dsl_name: str,
    module: ir.Module,
    fname: str,
    output_dir: str | None = None,
    bytecode_writer: Callable[..., Any] | None = None,
) -> Path:
    """Save a module as bytecode, ``<output_dir>/<dsl_name lower>_<fname>.mlir``,
    atomically: the file is written in a private temporary directory next to
    its destination and moved into place with ``os.replace``, so a reader
    never sees a partial write.

    :param dsl_name: The DSL's name, the file name prefix
    :param module: The module to save
    :param fname: The label of the file (a module hash)
    :param output_dir: The directory, defaults to :func:`get_default_generated_ir_path`
    :param bytecode_writer: Writes the open binary file itself (e.g.
        ``lambda f: write_bytecode_with_crc32(f, module)``); plain bytecode by default
    :return: The path of the saved file
    """
    initial_name = f"{dsl_name.lower()}_{fname}.mlir"
    save_path = normalize_path(
        output_dir if output_dir else get_default_generated_ir_path(dsl_name)
    )
    save_fname = save_path / initial_name
    # A per-write temporary directory (pid + random id) so concurrent writers
    # of the same file never collide; an abnormal exit may leave it behind.
    temp_dir = os.path.join(save_path, f"tmp.pid_{os.getpid()}_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=False)
    temp_fname = os.path.join(temp_dir, initial_name)
    with open(temp_fname, "wb") as f:
        if bytecode_writer:
            bytecode_writer(f)
        else:
            module.operation.write_bytecode(f)
    # os.replace is atomic on POSIX when it succeeds.
    os.replace(temp_fname, save_fname)
    os.rmdir(temp_dir)
    log().debug("Generated IR saved into %s", save_fname)
    return save_fname


def load_cache_from_path(
    dsl_name: str,
    file: str,
    path: str | None = None,
    bytecode_reader: Callable[..., Any] | None = None,
) -> ir.Module | None:
    """Load the lowered module cached under ``file`` (the module hash).

    :param dsl_name: The DSL's name, the file name prefix
    :param file: The module hash the entry was dumped under
    :param path: The cache directory, defaults to :func:`get_default_generated_ir_path`
    :param bytecode_reader: See :func:`load_ir`
    :return: The module, or None when there is no entry or it cannot be read
        (a corrupted entry is a miss, logged as a warning)
    """
    if path is None:
        path = get_default_generated_ir_path(dsl_name)
    if not os.path.exists(path):
        return None
    ret = None
    try:
        file = f"{dsl_name.lower()}_{file}.mlir"
        if os.path.exists(os.path.join(path, file)):
            ret = load_ir(os.path.join(path, file), bytecode_reader=bytecode_reader)
    except Exception as e:  # noqa: BLE001 -- a bad cache entry is a miss, not a failure
        log().warning(
            "%s failed with loading generated IR cache for %s: %s", dsl_name, file, e
        )
    return ret


def dump_cache_to_path(
    dsl_name: str,
    jit_function: "JitCompiledFunction",
    file: str,
    path: str | None = None,
    bytecode_writer: Callable[..., Any] | None = None,
) -> None:
    """Dump the lowered module of ``jit_function`` under ``file`` (the module hash).

    :param dsl_name: The DSL's name, the file name prefix
    :param jit_function: The compiled function whose ``ir_module`` is dumped
    :param file: The module hash
    :param path: The cache directory, defaults to :func:`get_default_generated_ir_path`
    :param bytecode_writer: See :func:`save_ir`; a failed dump is logged, never raised
    """
    log().info("JIT cache : dumping [%s] file=[%s]", dsl_name, file)
    if path is None:
        path = get_default_generated_ir_path(dsl_name)
    try:
        os.makedirs(path, exist_ok=True)
        save_ir(
            dsl_name,
            jit_function.ir_module,
            file,
            output_dir=path,
            bytecode_writer=bytecode_writer,
        )
    except Exception as e:  # noqa: BLE001 -- a failed dump must not fail the compile
        log().warning(
            "%s failed with dumping generated IR cache for %s: %s", dsl_name, file, e
        )
