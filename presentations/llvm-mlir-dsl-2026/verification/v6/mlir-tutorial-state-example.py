from dataclasses import dataclass, replace
import mlir.dsl as m

@dataclass(frozen=True)
class State:
    total: m.Int32

@m.jit
def sum_record(n: m.Int32) -> m.Int32:
    state = State(m.Int32(0))
    for i in range(n):
        state = replace(state, total=state.total + i)
    return state.total

@dataclass
class MutableState:
    total: m.Int32

@m.jit
def read_state(state):
    return state.total

print('FROZEN:', sum_record(10))
try:
    read_state(MutableState(m.Int32(0)))
except m.DSLUserCodeError as e:
    print('MUTABLE:', 'CONTAINER_DATACLASS_NOT_FROZEN' in str(e))
