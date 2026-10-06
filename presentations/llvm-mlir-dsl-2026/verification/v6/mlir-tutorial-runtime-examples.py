import numpy as np
import mlir.dsl as m

@m.jit
def affine(x: m.Int32, scale, bias) -> m.Int32:
    return x * scale + bias

class Op:
    def apply(self, x):
        raise NotImplementedError

class AddN(Op):
    def __init__(self, n):
        self.n = n
    def apply(self, x):
        return x + self.n

class MulN(Op):
    def __init__(self, k):
        self.k = k
    def apply(self, x):
        return x * self.k

@m.jit
def pipeline(x: m.Int32, ops: m.Constexpr) -> m.Int32:
    for op in ops.values():
        x = op.apply(x)
    return x

@m.jit
def fused(arr: m.Pointer[m.Float32], ops: m.Constexpr):
    for tx in m.range(4):
        x = arr[tx]
        for op in ops.values():
            x = op.apply(x)
        arr[tx] = x

print('AFFINE:', affine(7, 3, 4))
ops = {'shift': AddN(2), 'scale': MulN(3)}
print('POLYMORPHIC:', pipeline(4, ops))
a = np.arange(4, dtype=np.float32)
fused(a, ops)
print('FUSED:', a)
compiled = m.compile(affine, 7, 3, 4)
print('COMPILED:', compiled(9, 3, 4))
