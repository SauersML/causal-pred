"""Process-pool helper for embarrassingly-parallel work.

gamfit's Rust core dispatches via ``rayon`` (its linear algebra is
``faer``, not OpenBLAS / MKL), so ``RAYON_NUM_THREADS`` is the only
env var that actually constrains its thread fan-out.  Set it inside
each worker so K parallel processes do not all spawn cpu_count rayon
threads on a cpu_count-core box.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from joblib import Parallel, delayed


def cpu_count() -> int:
    """Return the number of CPUs the current process can actually run on.

    ``os.cpu_count()`` returns the kernel-visible CPU count, which on
    cgrouped containers (AoU JupyterLab, Kubernetes pods, etc.) can
    over- or under-report depending on Python build flags.
    ``sched_getaffinity`` asks the kernel for the affinity mask of the
    current process -- the CPUs we are *actually allowed to run on*.
    That's the right number for BLAS thread + worker heuristics: hand
    more threads than that to BLAS or rayon and the OS oversubscribes,
    so context-switch overhead eats the gain.

    This pipeline only ever runs on Linux (AoU Researcher Workbench),
    so we use sched_getaffinity unconditionally -- no fallback path.
    """
    return max(1, len(os.sched_getaffinity(0)))


def _worker(func: Callable[..., Any], args: tuple, threads: int) -> Any:
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(threads)))
    return func(*args)


def parallel_call(
    func: Callable[..., Any],
    arg_tuples: Sequence[tuple],
    *,
    n_workers: int,
    threads_per_worker: int,
) -> list:
    """Run ``func(*args)`` for each ``args`` in ``arg_tuples`` in parallel.

    Results are returned in input order.  Uses joblib's loky backend so
    workers are persistent across calls within the same parent process.
    """
    if not arg_tuples:
        return []
    return list(
        Parallel(n_jobs=max(1, int(n_workers)), backend="loky")(
            delayed(_worker)(func, args, threads_per_worker) for args in arg_tuples
        )
    )
