# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_FILTER_STACKTRACE=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,FILTERED
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_FILTER_STACKTRACE=0 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,UNFILTERED
# RUN: env MLIR_DSL_DRYRUN=1 MLIR_DSL_SHOW_STACKTRACE=1 %PYTHON %s 2>&1 | FileCheck %s --check-prefixes=CHECK,UNFILTERED
# The traceback of an error raised by a decorated call: with `FILTER_STACKTRACE`
# (on by default, off under `DEBUG` or `SHOW_STACKTRACE`, the third RUN line)
# the wrapper of the call drops the DSL's own frames, the bindings' and the
# standard library's, so the error reads as the user's code; the region
# function the rewrite generates for a staged `for`/`if`/`while` keeps the
# frame of the statement that raised, and the region call sitting at the line
# of the rewritten statement is folded into the user's frame at that line.
# Without the filter every frame is shown.
import sys
import traceback

import mlir.mlir_dsl as m


@m.jit
def plain(n: m.Int32) -> m.Int32:
    return n + int(None)


@m.jit
def staged_for(n: m.Int32) -> m.Int32:
    acc = m.Int32(0)
    for i in range(n):
        acc += int(None)
    return acc


@m.jit
def staged_if(n: m.Int32) -> m.Int32:
    if n > 1:
        n = n + int(None)
    return n


@m.jit
def staged_while(n: m.Int32) -> m.Int32:
    while n < 10:
        n = n + int(None)
    return n


def show(label, fn, *args):
    """Call `fn` and print the traceback of the error it raises."""
    try:
        fn(*args)
    except TypeError:
        print(f"--- {label}")
        traceback.print_exc(file=sys.stdout)


# CHECK-LABEL:  --- PLAIN
# CHECK-NEXT:   Traceback (most recent call last):
# CHECK-NEXT:     File "{{.*}}stacktrace.py", line {{[0-9]+}}, in show
# UNFILTERED:     File "{{.*}}{{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}core{{[\\/]}}dsl.py", line {{[0-9]+}}, in jit_wrapper
# UNFILTERED:     File "{{.*}}{{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}core{{[\\/]}}dsl.py", line {{[0-9]+}}, in trace_body
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in plain
# CHECK-NEXT:       return n + int(None)
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:        TypeError: int() argument must be
show("PLAIN", plain, 1)

# CHECK-LABEL:  --- FOR
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in show
# UNFILTERED:     in jit_wrapper
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in staged_for
# CHECK-NEXT:       for i in range(n):
# UNFILTERED:     File "{{.*}}{{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}plugins{{[\\/]}}ast_preprocessor{{[\\/]}}scf{{[\\/]}}executors.py", line {{[0-9]+}}, in _loop_execute_range_dynamic
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in loop_body_{{[0-9]+}}
# CHECK-NEXT:       acc += int(None)
# CHECK:        TypeError: int() argument must be
show("FOR", staged_for, 3)

# The region call `if_region_N` sits at the `if` line, like the user's frame
# there: filtered, the two are one frame.
# CHECK-LABEL:  --- IF
# CHECK:          File "{{.*}}stacktrace.py", line [[#IF_LINE:]], in staged_if
# CHECK-NEXT:       if n > 1:
# UNFILTERED:     File "{{.*}}stacktrace.py", line [[#IF_LINE]], in if_region_{{[0-9]+}}
# UNFILTERED-NEXT:  if n > 1:
# FILTERED-NOT:   in if_region_
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in then_block_{{[0-9]+}}
# CHECK-NEXT:       n = n + int(None)
# CHECK:        TypeError: int() argument must be
show("IF", staged_if, 3)

# CHECK-LABEL:  --- WHILE
# CHECK:          File "{{.*}}stacktrace.py", line [[#WHILE_LINE:]], in staged_while
# CHECK-NEXT:       while n < 10:
# UNFILTERED:     File "{{.*}}stacktrace.py", line [[#WHILE_LINE]], in while_region_{{[0-9]+}}
# FILTERED-NOT:   in while_region_
# FILTERED-NOT:   {{[\\/]}}mlir{{[\\/]}}dsl{{[\\/]}}
# CHECK:          File "{{.*}}stacktrace.py", line {{[0-9]+}}, in while_after_block_{{[0-9]+}}
# CHECK-NEXT:       n = n + int(None)
# CHECK:        TypeError: int() argument must be
# CHECK:        --- end
show("WHILE", staged_while, 3)
print("--- end")
