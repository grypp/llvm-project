# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This module provides jit cache load/dump helper functions.

The in-memory cache (``JitCacheDict`` in ``jit_executor``) maps the module hash
to the ``JitCompiledFunction``; the file cache under the cache directory keeps
the *lowered* module of each hash as bytecode with a CRC32 trailer, so a later
process skips the pass pipeline and only builds its ``ExecutionEngine``.
"""

import getpass
import functools
import hashlib
import io
import os
import random
import sys
import tempfile
from collections.abc import Sequence
import time
import uuid
import zlib
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from ... import ir
from ..core.common import DSLRuntimeError
from ..util.logger import log
from ..compiler.jit_executor import JitCacheDict, JitCompiledFunction

__all__ = [
    "JitCacheDict",
    "dump_cache_to_path",
    "get_current_user",
    "get_default_file_dump_root",
    "get_default_generated_ir_path",
    "load_cache_from_path",
    "load_ir",
    "make_unique_filename",
    "normalize_path",
    "read_bytecode_and_check_crc32",
    "save_ir",
    "write_bytecode_with_crc32",
]


# =============================================================================
# Jit Cache Helper functions
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


@lru_cache(maxsize=1)
def get_default_file_dump_root() -> Path:
    """The root for user-requested file dumps: the working directory at first
    use, fixed for the process."""
    return Path.cwd()


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


def load_ir(
    file: str,
    asBytecode: bool = False,
    bytecode_reader: Callable[..., Any] | None = None,
) -> tuple[str, ir.Module]:
    """Load generated IR from a ``.mlir`` file into the current context.

    :param file: The path of a file written by :func:`save_ir`
    :param asBytecode: Open the file in binary mode (bytecode), defaults to False
    :param bytecode_reader: Parses the open binary file into a module (e.g.
        :func:`read_bytecode_and_check_crc32`); ``ir.Module.parse`` by default
    :return: The label the file was saved under (the part of the stem after
        ``<dsl>_``) and the module
    :raises DSLRuntimeError: ``file`` is not a ``.mlir`` path
    """
    if ".mlir" not in file:
        raise DSLRuntimeError(
            "generated IR is loaded from `.mlir` files only", context={"file": file}
        )
    func_name = file.split(".mlir")[0].split("dsl_")[-1]
    with open(file, "rb" if asBytecode else "r") as f:
        if bytecode_reader:
            module = bytecode_reader(f)
        else:
            module = ir.Module.parse(f.read())
    return func_name, module


def make_unique_filename(fpath: Path, new_ext: str | None = None) -> Path:
    """A sibling of ``fpath`` whose stem carries a 16-hex-digit tag derived
    from the path, the time and a random number.

    :param fpath: The path to derive the name from
    :param new_ext: Replaces the suffix (``".txt"``), defaults to ``fpath``'s
    :return: ``<dir>/<stem>_<tag><ext>``
    """
    random_part = random.randint(0, 999999)
    timestamp = time.time()
    hash_input = f"{fpath}_{timestamp}_{random_part}".encode()
    hash_code = hashlib.md5(hash_input).hexdigest()[:16]
    stem_with_hash = f"{fpath.stem}_{hash_code}"
    return fpath.with_name(stem_with_hash).with_suffix(new_ext or fpath.suffix)


def save_ir(
    dsl_name: str,
    module: ir.Module,
    fname: str,
    output_dir: str | None = None,
    as_bytecode: bool = False,
    bytecode_writer: Callable[..., Any] | None = None,
    enable_debug_info: bool = True,
) -> Path:
    """Save generated IR as ``<output_dir>/<dsl_name lower>_<fname>.mlir``,
    atomically: the file is written in a private temporary directory next to
    its destination and moved into place with ``os.replace``, so a reader
    never sees a partial write.

    :param dsl_name: The DSL's name, the file name prefix
    :param module: The module to save
    :param fname: The label of the file (a function name or a module hash)
    :param output_dir: The directory, defaults to :func:`get_default_generated_ir_path`
    :param as_bytecode: Write bytecode instead of text, defaults to False
    :param bytecode_writer: Writes the open binary file itself (e.g.
        ``lambda f: write_bytecode_with_crc32(f, module)``); plain bytecode by default
    :param enable_debug_info: Print locations in the textual form, defaults to True
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
    if as_bytecode:
        with open(temp_fname, "wb") as f:
            if bytecode_writer:
                bytecode_writer(f)
            else:
                module.operation.write_bytecode(f)
    else:
        with open(temp_fname, "w") as f:
            print(module.operation.get_asm(enable_debug_info=enable_debug_info), file=f)
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
            _, ret = load_ir(
                os.path.join(path, file),
                asBytecode=True,
                bytecode_reader=bytecode_reader,
            )
    except Exception as e:  # noqa: BLE001 -- a bad cache entry is a miss, not a failure
        log().warning(
            "%s failed with loading generated IR cache for %s: %s", dsl_name, file, e
        )
    return ret


def dump_cache_to_path(
    dsl_name: str,
    jit_function: JitCompiledFunction,
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
            as_bytecode=True,
            bytecode_writer=bytecode_writer,
        )
    except Exception as e:  # noqa: BLE001 -- a failed dump must not fail the compile
        log().warning(
            "%s failed with dumping generated IR cache for %s: %s", dsl_name, file, e
        )
