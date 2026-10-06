import mlir.dsl as m

@m.jit
def twice(n: m.Int32) -> m.Int32:
    return n * 2

compiled = m.compile(twice, 21)
print('COMPILE:', type(compiled).__name__)
print('EXECUTE:', int(compiled(7)))
print('HAS_ARRAY:', hasattr(m, 'Array'))
