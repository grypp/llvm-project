# RUN: rm -rf %t.cache && env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s 2>&1 | FileCheck %s --check-prefix=MISS
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s corrupt 2>&1 | FileCheck %s --check-prefix=CORRUPT
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache %PYTHON %s 2>&1 | FileCheck %s --check-prefix=HIT
# RUN: env MLIR_DSL_CACHE_DIR=%t.cache MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefix=OFF
# REQUIRES: host-supports-jit
import logging
import os
import sys
import mlir.mlir_dsl as m
from mlir.dsl.util.logger import log

dsl = m.MlirTestDSL()
mode = sys.argv[1] if len(sys.argv) > 1 else ""
cache_dir = dsl.envar.cache_dir
warnings = []


class Capture(logging.Handler):
    def emit(self, rec):
        if rec.levelno >= logging.WARNING:
            warnings.append(rec.getMessage())


log().addHandler(Capture())
log().setLevel(logging.INFO)


# --- the on-disk cache ---------------------------------------
# The lowered module of each module hash is written under `<PREFIX>_CACHE_DIR`
# as `mlir_dsl_<hash>.mlir`, bytecode plus a CRC32 trailer. The second process
# finds it, skips the pass pipeline and only builds the engine (a file hit is
# not an in-memory miss); a damaged entry is a miss with a logged warning and
# is rewritten valid; `DISABLE_FILE_CACHING` leaves the files alone. The
# second call in one process is an in-memory hit either way.
def entries():
    if not os.path.isdir(cache_dir):
        return []
    return sorted(p for p in os.listdir(cache_dir) if p.startswith("mlir_dsl_"))


if mode == "corrupt":
    for name in entries():
        path = os.path.join(cache_dir, name)
        with open(path, "r+b") as f:
            f.seek(os.path.getsize(path) // 2)
            byte = f.read(1)
            f.seek(-1, os.SEEK_CUR)
            f.write(bytes([byte[0] ^ 0xFF]))
    print("DAMAGED:", len(entries()) > 0)
# CORRUPT: DAMAGED: True


@m.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


before = (len(entries()), dsl.file_cache_hits, dsl.cache_misses, dsl.cache_hits)
seen = len(warnings)
results = twice(21), twice(4)
fh, mi = dsl.file_cache_hits - before[1], dsl.cache_misses - before[2]
mh = dsl.cache_hits - before[3]
print("FILE:", *results, "NEW_FILES:", len(entries()) - before[0], end=" ")
print(f"FILE_HITS: {fh} MISSES: {mi} MEM_HITS: {mh}", end=" ")
print("WARNINGS:", ["CRC32" in w for w in warnings[seen:]])
# MISS:    FILE: 42 8 NEW_FILES: 1 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: []
# CORRUPT: FILE: 42 8 NEW_FILES: 0 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: [True]
# HIT:     FILE: 42 8 NEW_FILES: 0 FILE_HITS: 1 MISSES: 0 MEM_HITS: 1 WARNINGS: []
# OFF:     FILE: 42 8 NEW_FILES: 0 FILE_HITS: 0 MISSES: 1 MEM_HITS: 1 WARNINGS: []
