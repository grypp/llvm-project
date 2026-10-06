# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit scf builders: the loops and branches of 03, written by hand.

With `@m.jit(preprocess=False)` nothing rewrites the Python source: `for_`,
`yield_`, `if_` and `while_` build `scf.for`, `scf.if` and `scf.while`
directly, and loop carries are threaded by hand. These builders are the
modular layer the preprocessor targets when it rewrites native `for`/`if`/
`while`, and a sub-DSL may use them directly. Each function carries its
native equivalent in a comment.
"""

import mlir.mlir_dsl as m


@m.jit(preprocess=False)
def sum_of_squares(n: m.Int32) -> m.Int32:
    # Native form:  acc = m.Int32(0)
    #               for i in range(n): acc = acc + i * i
    # `for_(start, stop, step, iter_args)` yields `(iv, carry_in, carry_out)`
    # for one carry; the body must end with `yield_` of the next carry value,
    # and `carry_out` is the scf.for result, readable after the loop.
    # (`for_(n)` alone yields just `iv` and needs no `yield_`.)
    for i, acc_in, acc_out in m.for_(0, n, 1, [m.Int32(0)]):
        m.yield_(acc_in + i * i)
    return acc_out


@m.jit(preprocess=False)
def clamp(x: m.Int32, lo: m.Int32, hi: m.Int32) -> m.Int32:
    # Native form:  y = lo if x < lo else x
    #               return hi if y > hi else y
    # `if_(cond, then, else, return_types)` runs each callable under its arm's
    # insertion point; the arm's value is cast to `return_types` and yielded,
    # and the scf.if result comes back typed (one type -> one value).
    y = m.if_(x < lo, lambda: lo, lambda: x, return_types=[m.Int32])
    return m.if_(y > hi, lambda: hi, lambda: y, return_types=[m.Int32])


@m.jit(preprocess=False)
def ceil_log2(n: m.Int32) -> m.Int32:
    # Native form:  p, k = m.Int32(1), m.Int32(0)
    #               while p < n: p, k = p * 2, k + 1
    # `while_(inputs, cond)` traces `cond` into the before block; the `with`
    # binds the after-block carries, the body ends with `yield_` of the next
    # carries, and `.results` restores the scf.while results in `inputs` shape.
    loop = m.while_([m.Int32(1), m.Int32(0)], lambda p, k: p < n)
    with loop as (p, k):
        m.yield_([p * 2, k + 1])
    return loop.results[1]


def check(name, got, want):
    if m.is_dynamic_expr(got):  # DRYRUN traces only: no value to compare
        return
    assert int(got) == want, (name, int(got), want)
    print(f"{name} = {want}")


def main():
    for n in (0, 1, 5):
        check(f"sum_of_squares({n})", sum_of_squares(n), sum(i * i for i in range(n)))
    for x in (-3, 4, 12):
        check(f"clamp({x}, 0, 10)", clamp(x, 0, 10), min(max(x, 0), 10))
    for n in (1, 2, 9, 1000):
        check(f"ceil_log2({n})", ceil_log2(n), (n - 1).bit_length())
    print("Explicit scf builders: passed")


if __name__ == "__main__":
    main()
