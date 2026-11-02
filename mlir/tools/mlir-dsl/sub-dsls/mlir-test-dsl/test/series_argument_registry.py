# RUN: %PYTHON %s 2>&1 | FileCheck %s
import mlir.mlir_dsl as m
from mlir.dsl.core.arguments import JitArgAdapterRegistry as R


def failure(fn):
    """The DiagId of the user error `fn` raises, the message of an internal one."""
    try:
        fn()
        return "no error"
    except m.DSLUserCodeError as e:
        return e.diag_id.name
    except m.DSLRuntimeError as e:
        return " ".join(e.message.split())


# --- the adapter registry ---------------------------------
# Adapters are keyed by type (common) or by (scope, type); a scoped adapter
# shadows the common one inside `using_scope`; a lazy registration by
# qualified name is promoted on the first lookup of an instance, which is how
# `torch.Tensor` is registered without importing torch; Python scalars have no
# adapter (they are Meta values); malformed registrations are internal errors.
class Handle:
    def __init__(self, v):
        self.v = v


class Third:
    pass


@R.register_jit_arg_adapter(Handle)
def adapt_handle(h):
    return m.Int32(h.v)


@R.register_jit_arg_adapter(Handle, scope="s1")
def adapt_handle_s1(h):
    return m.Int64(h.v)


@R.register_jit_arg_adapter(f"{__name__}.Third", lazy=True)
def adapt_third(t):
    return m.Int32(3)


with R.using_scope("s1"):
    scoped = R.get_registered_adapter(Handle(1)) is adapt_handle_s1
common = R.get_registered_adapter(Handle(1)) is adapt_handle
scalars = [R.get_registered_adapter(v) for v in (1, 2.5, "s", None)]
print(f"REGISTRY: {common} {scoped} {R.get_registered_adapter(object())} {scalars}")
lazy_before = Third in R.jit_arg_adapter_registry
lazy_named = f"{__name__}.Third" in R.lazy_jit_arg_adapter_registry
lazy_hit = R.get_registered_adapter(Third()) is adapt_third
# The lazy entry is keyed by qualified name until the first instance promotes
# it to the type-keyed registry.
print(
    f"LAZY: {lazy_before} {lazy_named} {lazy_hit} {Third in R.jit_arg_adapter_registry}"
)
dup = failure(lambda: R.register_jit_arg_adapter(Handle)(lambda h: h))
bad = failure(lambda: R.register_jit_arg_adapter("T", lazy=True))
print("MALFORMED:", dup, "|", bad)
# CHECK:      REGISTRY: True True None [None, None, None, None]
# CHECK-NEXT: LAZY: False True True True
# CHECK-NEXT: MALFORMED: {{.*}}already registered{{.*}} | {{.*}}must be fully-qualified

# --- one registry, published copy-on-write -------------------------------
# A writer replaces a read-only table whole under the registry's lock: a
# reader iterating the common table while two threads register disjoint types
# sees a snapshot (no "dictionary changed size during iteration") and no
# registration is lost; the published tables refuse item assignment; a lazy
# entry promoted by two threads at once is promoted once.
import threading

types_a = [type(f"A{i}", (), {}) for i in range(200)]
types_b = [type(f"B{i}", (), {}) for i in range(200)]
start = threading.Event()
reader_errors = []


def register_all(types):
    start.wait()
    for t in types:
        R.register_jit_arg_adapter(t)(lambda h: h)


def read_loop():
    start.wait()
    try:
        for _ in range(500):
            for _ in R.jit_arg_adapter_registry:
                pass
            R.get_registered_adapter(Handle(1))
    except RuntimeError as e:
        reader_errors.append(str(e))


threads = [
    threading.Thread(target=register_all, args=(types_a,)),
    threading.Thread(target=register_all, args=(types_b,)),
    threading.Thread(target=read_loop),
]
for t in threads:
    t.start()
start.set()
for t in threads:
    t.join(timeout=60)
all_present = all(t in R.jit_arg_adapter_registry for t in types_a + types_b)
print(
    "RACE_FREE:",
    not reader_errors and not any(t.is_alive() for t in threads),
    all_present,
)
try:
    R.jit_arg_adapter_registry[Handle] = adapt_handle
    frozen = False
except TypeError:
    frozen = True
print("FROZEN:", frozen)


class Fourth:
    pass


@R.register_jit_arg_adapter(f"{__name__}.Fourth", lazy=True)
def adapt_fourth(t):
    return m.Int32(4)


size_before = len(R.jit_arg_adapter_registry)
gate = threading.Barrier(2)
promoted = []


def promote():
    gate.wait()
    promoted.append(R.get_registered_adapter(Fourth()))


promoters = [threading.Thread(target=promote) for _ in range(2)]
for t in promoters:
    t.start()
for t in promoters:
    t.join(timeout=60)
print(
    "PROMOTED_ONCE:",
    promoted == [adapt_fourth, adapt_fourth],
    len(R.jit_arg_adapter_registry) - size_before == 1,
)
# CHECK:      RACE_FREE: True True
# CHECK-NEXT: FROZEN: True
# CHECK-NEXT: PROMOTED_ONCE: True True

# --- a scoped record policy ------------------------------------------------
# A sub-DSL's adapter plugin installs a record policy for the DSL's own
# scope: `accepts` replaces the frozen-record rule there, so a non-frozen
# dataclass argument is adapted (here into a frozen twin) under the DSL, at
# the registry and at its `@jit` boundary alike, while outside the scope the
# default rule still refuses it.
import dataclasses

from mlir.dsl.core.plugin import AdapterPlugin
from mlir.dsl.plugins.decorators.jit import func
from mlir.dsl.plugins.type_ops import UpstreamDialectTypeOps, arith


@dataclasses.dataclass
class Record:
    x: m.Int32
    y: m.Int32


@dataclasses.dataclass(frozen=True)
class FrozenRecord:
    x: m.Int32
    y: m.Int32


def adapt_record(rec):
    return FrozenRecord(m.Int32(rec.x), m.Int32(rec.y))


class RecordPolicy(AdapterPlugin):
    name = "records"

    def register(self, dsl):
        R.set_default_dataclass_adapter(
            adapt_record, accepts=dataclasses.is_dataclass, scope="rec"
        )


class RecDSL(m.BaseDSL):
    _jit_arg_adapter_scope = "rec"
    plugins = m.Plugins(
        type_ops=UpstreamDialectTypeOps(scalars=arith),
        decorators=[func.Jit()],
        adapters=[RecordPolicy()],
    )

    def __init__(self):
        super().__init__(name="REC_DSL")


@RecDSL.jit
def total(r: Record) -> m.Int32:
    return r.x + r.y


RecDSL()  # constructing the DSL runs its adapter plugin's `register`
with R.using_scope("rec"):
    inside = type(R.get_registered_adapter(Record(1, 2))(Record(1, 2))).__name__
outside = failure(lambda: R.get_registered_adapter(Record(1, 2)))
print("SCOPED_RECORD:", inside, outside, "|", failure(lambda: total(Record(1, 2))))
# CHECK: SCOPED_RECORD: FrozenRecord CONTAINER_INVALID_RECORD | no error
