# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,MISS
# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 MLIR_DSL_JIT_CACHE_MAX_ELEMS=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ONE
# RUN: env MLIR_DSL_DISABLE_FILE_CACHING=1 MLIR_DSL_NO_CACHE=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ZERO
# REQUIRES: host-supports-jit
import gc
import threading
import mlir.mlir_dsl as m
from mlir.dsl.util.cache import JitCacheDict

dsl = m.MlirTestDSL()
# --- the in-memory cache -------------------------------------
# `JitCacheDict` maps module hash to compiled function: `max_elems` None is
# unlimited, N evicts least recently used (a `get` refreshes), 0 disables it;
# an entry tied to an `owner` goes when that object is collected.
# `<PREFIX>_JIT_CACHE_MAX_ELEMS` and `<PREFIX>_NO_CACHE` size the DSL's cache,
# so with one slot alternating between two functions recompiles every time.
lru = JitCacheDict(max_elems=2)
lru.set("a", "A")
lru.set("b", "B")
lru.get("a")
lru.set("c", "C")
print(f"LRU: {lru.get('a')} {lru.get('b')} {lru.get('c')} {len(lru)}")
body = lambda: None
tied = JitCacheDict()
tied.set("k", "K", owner=body)
alive = tied.get("k")
del body
gc.collect()
print("TIED:", alive, tied.get("k"), len(tied))
# CHECK:      LRU: A None C 2
# CHECK-NEXT: TIED: K None 0

# --- the cache lock ------------------------------------------
# Every operation runs under the cache's own re-entrant lock. A displaced
# value (overwritten, evicted, deleted, cleared) is released after the lock,
# so a finalizer that runs arbitrary code never runs under it.
under_lock = []


class Recorder:
    def __del__(self):
        under_lock.append(guarded._lock._is_owned())


guarded = JitCacheDict(max_elems=2)
guarded.set("k", Recorder())
guarded.set("k", Recorder())  # overwritten
guarded.set("l", Recorder())
guarded.set("m", Recorder())  # evicts "k"
guarded.delete("l")
guarded.clear()
print("UNDER LOCK:", sorted(set(under_lock)), len(under_lock))
# CHECK-NEXT: UNDER LOCK: [False] 4
# An owner collected inside a cache call (collection can happen at any point)
# deletes its entry through the finalizer; the lock is re-entrant, so this
# does not deadlock.
owner = lambda: None
reentrant = JitCacheDict()
reentrant.set("o", "O", owner=owner)
with reentrant._lock:
    del owner
    gc.collect()
    print("REENTRANT:", reentrant.get("o"), len(reentrant))
# CHECK-NEXT: REENTRANT: None 0
# Mixed get/set/delete from eight threads keep the cache within its capacity
# and raise nothing.
shared = JitCacheDict(max_elems=4)
errors = []


def churn(seed):
    try:
        for i in range(300):
            k = (seed + i) % 6
            shared.set(k, i)
            shared.get((k + 1) % 6)
            if i % 3 == 0:
                shared.delete((k + 2) % 6)
            assert len(shared) <= 4
    except Exception as e:  # noqa: BLE001 -- the test reports, not raises
        errors.append(repr(e))


threads = [threading.Thread(target=churn, args=(s,)) for s in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("THREADS:", errors, len(shared) <= 4)
# CHECK-NEXT: THREADS: [] True


@m.jit
def inc(n: m.Int32) -> m.Int32:
    return n + 1


@m.jit
def dec(n: m.Int32) -> m.Int32:
    return n - 1


before = (dsl.cache_hits, dsl.cache_misses)
results = inc(1), dec(1), inc(2), inc(3), dec(2)
hits, misses = dsl.cache_hits - before[0], dsl.cache_misses - before[1]
print("MEMORY:", *results, "MAX_ELEMS:", dsl.jit_cache.max_elems, end=" ")
print(f"HITS: {hits} MISSES: {misses} ENTRIES: {len(dsl.jit_cache)}")
# MISS: MEMORY: 2 0 3 4 1 MAX_ELEMS: None HITS: 3 MISSES: 2 ENTRIES: 2
# ONE:  MEMORY: 2 0 3 4 1 MAX_ELEMS: 1 HITS: 1 MISSES: 4 ENTRIES: 1
# ZERO: MEMORY: 2 0 3 4 1 MAX_ELEMS: 0 HITS: 0 MISSES: 5 ENTRIES: 0
