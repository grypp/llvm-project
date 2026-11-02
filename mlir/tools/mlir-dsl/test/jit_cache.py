# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,MANY
# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 MLIR_DSL_JIT_CACHE_MAX_ELEMS=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ONE
# REQUIRES: host-supports-jit
# The in-memory JIT cache is the DSL's to key and to build. `_cache_key_extra`
# adds a DSL's own bytes to the module hash (`get_module_hash`), so a change of
# what its compiled artifact depends on is a cache miss and restoring it a hit;
# `_jit_cache_class` names the cache class `BaseDSL.__init__` builds, so a DSL
# can observe or replace the cache policy. The file cache is off so the
# counters are those of this process; the second RUN line gives the cache one
# slot, so every other store evicts.
import gc

import mlir.mlir_dsl as m
from mlir.dsl.util.cache import JitCacheDict

SALT = [b""]
events = []


class LoggingCache(JitCacheDict):
    """The shipped cache, recording every lookup and store."""

    def get(self, key):
        value = super().get(key)
        events.append("hit" if value is not None else "miss")
        return value

    def set(self, key, value, owner=None):
        before = len(self)
        super().set(key, value, owner=owner)
        events.append("store" if len(self) > before else "store, one evicted")


class SaltedDSL(m.MlirTestDSL):
    _jit_cache_class = LoggingCache

    def _cache_key_extra(self):
        return SALT[0]


dsl = SaltedDSL()


@SaltedDSL.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2


@SaltedDSL.jit
def thrice(n: m.Int32) -> m.Int32:
    return n * 3


def window(label, *calls):
    """Run the calls and print the cache counters and events they caused."""
    gc.collect()
    before = (dsl.cache_hits, dsl.cache_misses)
    del events[:]
    results = [fn(arg) for fn, arg in calls]
    hits, misses = dsl.cache_hits - before[0], dsl.cache_misses - before[1]
    print(label, *results, f"HITS: {hits} MISSES: {misses}", events)


print("CACHE:", type(dsl.jit_cache).__name__, isinstance(dsl.jit_cache, JitCacheDict))
window("FIRST:", (twice, 2), (twice, 3))
SALT[0] = b"salt"
window("SALTED:", (twice, 4))
SALT[0] = b""
window("RESTORED:", (twice, 5))
window("SECOND:", (thrice, 2), (twice, 6))
# CHECK:      CACHE: LoggingCache True
# CHECK-NEXT: FIRST: 4 6 HITS: 1 MISSES: 1 ['miss', 'store', 'hit']
# MANY-NEXT:  SALTED: 8 HITS: 0 MISSES: 1 ['miss', 'store']
# MANY-NEXT:  RESTORED: 10 HITS: 1 MISSES: 0 ['hit']
# MANY-NEXT:  SECOND: 6 12 HITS: 1 MISSES: 1 ['miss', 'store', 'hit']
# ONE-NEXT:   SALTED: 8 HITS: 0 MISSES: 1 ['miss', 'store, one evicted']
# ONE-NEXT:   RESTORED: 10 HITS: 0 MISSES: 1 ['miss', 'store, one evicted']
# ONE-NEXT:   SECOND: 6 12 HITS: 0 MISSES: 2 ['miss', 'store, one evicted', 'miss', 'store, one evicted']
