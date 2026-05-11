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
    return max(1, os.cpu_count() or 1)


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
