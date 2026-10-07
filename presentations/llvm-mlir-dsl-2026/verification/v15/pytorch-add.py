import torch

a = torch.arange(4, dtype=torch.float32)
b = torch.full_like(a, 10.0)
out = torch.empty_like(a)
# Expected: tensor([10., 11., 12., 13.])

import mlir.mlir_dsl as m
P = m.Pointer[m.Float32]

@m.jit
def add(n: m.Int32, a: P, b: P,
        out: P):
    for i in range(n):
        out[i] = a[i] + b[i]

add(a.numel(), a, b, out)
print(out)
