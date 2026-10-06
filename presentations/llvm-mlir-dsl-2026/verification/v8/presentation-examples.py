# Tutorial v8 displayed-example verification, 2026-10-06.
# Code excerpts are copied from deck.json. Harness calls/checks are appended.
# Diagnostic call corrected to a string: the host boundary accepts 1.5 as Int32.
# Uses the current built mlir.mlir_dsl package; no implementation changes.
import mlir.mlir_dsl as m
import numpy as np

failures = []

def check(name, call, expected):
    try:
        got = call()
        if hasattr(got, "value"):
            got = got.value
        assert got == expected, (got, expected)
        print(f"PASS {name}: {got!r}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}")



# v7-07

import mlir.mlir_dsl as m

@m.jit
def scale(x: m.Int32, factor):
    return x * factor

scale(4, 3)

check("scale", lambda: scale(4, 3), 12)


# v7-20

@m.jit
def affine(x: m.Int32, scale, bias):
    return x * scale + bias

affine(7, 3, 4)  # 25

check("affine", lambda: affine(7, 3, 4), 25)


# v7-21

class Op:
    def apply(self, x):
        raise NotImplementedError

class AddN(Op):
    def __init__(self, n): self.n = n
    def apply(self, x): return x + self.n

class MulN(Op):
    def __init__(self, k): self.k = k
    def apply(self, x): return x * self.k

@m.jit
def pipeline(x: m.Int32, ops):
    for op in ops.values():
        x = op.apply(x)
    return x

ops = {"shift": AddN(2), "scale": MulN(3)}
pipeline(4, ops)  # 18

# Captured: (x + 2) * 3

check("pipeline", lambda: pipeline(4, ops), 18)


# v7-33

@m.jit(preprocess=False)
def sum_upto(n: m.Int32):
    acc = m.Int32(0)
    for i, current, result in m.for_(
            0, n, 1, [acc]):
        m.yield_([current + i])
    return result

check("explicit sum_upto", lambda: sum_upto(10), 45)


# v7-34

@m.jit
def sum_upto(n: m.Int32):
    acc = m.Int32(0)
    for i in range(n):
        acc += i
    return acc

sum_upto(10)  # 45

check("native sum_upto", lambda: sum_upto(10), 45)


# v7-35

@m.jit
def fused(arr: m.Pointer[m.Float32],
          ops):
    for i in m.range(4):
        x = arr[i]
        for op in ops.values():
            x = op.apply(x)
        arr[i] = x

# [0, 1, 2, 3] -> [6, 9, 12, 15]

data = np.arange(4, dtype=np.float32)
def run_fused():
    fused(data, ops)
    return data.tolist()
check("fused NumPy", run_fused, [6.0, 9.0, 12.0, 15.0])


# v7-26

@m.jit
def expects_int(x: m.Int32):
    return x



try:
    expects_int("oops")
except m.DSLUserCodeError as e:
    if e.diag_id is m.DiagId.ARG_ANNOTATION_MISMATCH:
        print("PASS expects_int(str): ARG_ANNOTATION_MISMATCH")
        print(str(e))
    else:
        failures.append("expects_int diagnostic")
        print("FAIL expects_int diagnostic:", e.diag_id.name)
else:
    failures.append("expects_int diagnostic")
    print("FAIL expects_int diagnostic: accepted string")

if failures:
    raise SystemExit("Failed: " + ", ".join(failures))
print("Presentation examples: passed")
