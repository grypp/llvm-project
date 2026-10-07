# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structs: immutable records of DSL-typed fields with `@m.struct` and `m.make_struct`.

A struct is a frozen record and a pytree, not an aggregate SSA value: a field
read is the field's own value, `s.replace(...)` returns a copy and leaves `s`
untouched, `a, b = s` unpacks the fields, and structs nest. At every boundary
a struct flattens to its fields: a struct argument is one function argument
per field, a returned struct comes back as a host instance, and a returned
tuple of leaves is packed into one host result and unpacked again. No dialect
type is involved, so structs work under every `type_ops` plugin (every dialect).
"""

import mlir.mlir_dsl as m


@m.struct
class Vec2:
    x: m.Float32  # fields are DSL types only; `float` would be STRUCT_DEFINITION
    y: m.Float32


@m.struct
class Particle:
    pos: Vec2  # structs nest; the fields flatten to four f32 arguments
    vel: Vec2


# The decorator and `make_struct` build the same kind of class.
Span = m.make_struct("Span", lo=m.Int32, hi=m.Int32)


@m.jit
def step(p: Particle, dt: m.Float32) -> Particle:
    # `p` arrives as four f32 block arguments; `p.pos.x` is one of them.
    x = p.pos.x + p.vel.x * dt
    y = p.pos.y + p.vel.y * dt
    # `replace` returns a new record; `p` itself never changes.
    return p.replace(pos=Vec2(x=x, y=y))


@m.jit
def norm2(v: Vec2) -> m.Float32:
    x, y = v  # unpacking iterates the fields in declaration order
    return x * x + y * y


@m.jit
def around(n: m.Int32, radius: m.Int32) -> Span:
    return Span(lo=n - radius, hi=n + radius)  # a struct is an ordinary result


@m.jit
def divmod_(n: m.Int32, d: m.Int32) -> tuple:
    # A tuple of leaves is returned as ONE packed host result and unpacked
    # back into a Python tuple on the host.
    return n // d, n - (n // d) * d


def check(label, result, pick, want):
    if m.is_mlir_op(result):  # DRYRUN hands back the traced value: `?`
        print(f"{label}: ?")
        return
    got = pick(result)
    assert got == want, (label, got, want)
    print(f"{label}: {got}")


def main():
    p = Particle(pos=Vec2(x=1.0, y=2.0), vel=Vec2(x=0.5, y=-1.0))
    q = step(p, 2.0)  # a host struct in, a host struct out
    check("step", q, lambda s: (tuple(s.pos), tuple(s.vel)), ((2.0, 0.0), (0.5, -1.0)))
    check("norm2", norm2(Vec2(x=3.0, y=4.0)), float, 25.0)
    check("around", around(10, 3), tuple, (7, 13))
    quot, rem = divmod_(17, 5)  # one struct result, two host values
    check("divmod", (quot, rem), tuple, (3, 2))
    print("Structs: passed")


if __name__ == "__main__":
    main()
