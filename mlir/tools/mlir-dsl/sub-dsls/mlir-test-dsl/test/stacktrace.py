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
