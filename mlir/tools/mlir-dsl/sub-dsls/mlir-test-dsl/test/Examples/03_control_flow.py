# RUN: env MLIR_DSL_DRYRUN=1 %PYTHON %s
# RUN: %if host-supports-jit %{ %PYTHON %s %}
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native control flow: `for`/`if`/`while` become `scf` regions or run in Python.

The preprocessor looks at a loop's bound or a branch's condition: a staged
value (derived from a DSL-typed argument) turns the statement into an
`scf.for`, `scf.if` or `scf.while` whose carries are the names the body
assigns; a Meta value (plain Python) leaves it a Python statement that runs at
trace time, so Meta loops unroll and only Meta `if`s may `return`, `break` or
`continue`. Run with MLIR_DSL_DRYRUN=1 MLIR_DSL_PRINT_IR=1 to see which
statements became `scf` regions and which folded into straight-line IR.
"""

import mlir.mlir_dsl as m


@m.jit
def fib(n: m.Int32) -> m.Int32:
    a, b = m.Int32(0), m.Int32(1)
    for _ in range(n):  # staged bound: scf.for; `a` and `b`, stored in the
        a, b = b, a + b  # body, are its two iter_args (scf.yield b, a + b)
    return a


@m.jit
def power(x: m.Float32, k) -> m.Float32:
    acc = m.Float32(1.0)
    for _ in range(k):  # Meta bound: a Python loop at trace time; the body is
        acc = acc * x  # traced `k` times, so the IR is `k` chained arith.mulf
    return acc


@m.jit
def magnitude(x: m.Int32) -> m.Int32:
    if x < 0:  # staged condition: scf.if; `r` is stored in both arms, so it
        r = -x  # is the op's result, yielded by each region
    else:
        r = x
    return r


@m.jit
def halvings(n: m.Int32) -> m.Int32:
    steps = m.Int32(0)
    while n > 1:  # staged condition: scf.while carrying `n` and `steps`; the
        n = n // 2  # before-region recomputes the condition from the carries
        steps += 1
    return steps


@m.jit
def odd_sum(x: m.Int32, limit) -> m.Int32:
    if limit <= 0:  # Meta condition owning a `return`: stays a Python `if`,
        return -x  # decided at trace time (staged: UNSUP_EARLY_EXIT)
    acc = x
    last = 0  # read after the loop, so initialised before it: a name born
    for i in range(limit):  # inside a loop body may not be read after it
        if i % 2 == 0:  # Meta `if`s inside a Meta loop: `continue` and
            continue  # `break` are Python's, nothing of them reaches the IR
        if i > 7:
            break
        acc += i
        last = i
    return acc + last


def check(name, got, want):
    if not m.is_mlir_op(got):  # under MLIR_DSL_DRYRUN the result is `?`
        assert got == want, (name, got, want)
    print(f"{name} = {got}")


def main():
    check("fib(10)", fib(10), 55)
    check("power(2.0, 10)", power(2.0, 10), 1024.0)
    check("magnitude(-7)", magnitude(-7), 7)
    check("magnitude(7)", magnitude(7), 7)
    check("halvings(1000)", halvings(1000), 9)
    check("odd_sum(100, 10)", odd_sum(100, 10), 123)  # 100 + 1+3+5+7, last 7
    check("odd_sum(100, 0)", odd_sum(100, 0), -100)
    print("Native control flow: passed")


if __name__ == "__main__":
    main()
