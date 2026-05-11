"""Process-pool helper for embarrassingly-parallel work.

Single canonical path: joblib's loky backend with BLAS threads pinned per
worker.  Pinning matters because the gamfit Rust core grabs as many BLAS
threads as it can find; without per-worker pinning, K parallel fits ask
for K * cpu_count threads on cpu_count cores and the scheduler thrashes.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from joblib import Parallel, delayed


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _pin_blas_threads(threads: int) -> None:
    val = str(max(1, int(threads)))
    for var in _BLAS_ENV_VARS:
        os.environ[var] = val


def _worker(func: Callable[..., Any], args: tuple, threads: int) -> Any:
    _pin_blas_threads(threads)
    return func(*args)


def parallel_call(
    func: Callable[..., Any],
    arg_tuples: Sequence[tuple],
    *,
    n_workers: int,
    threads_per_worker: int,
) -> list:
    """Run ``func(*args)`` for each ``args`` in ``arg_tuples`` in parallel.

    Results are returned in input order.  Uses joblib's loky backend, so
    workers are persistent across calls within the same parent process.
    """
    if not arg_tuples:
        return []
    n_workers = max(1, int(n_workers))
    threads_per_worker = max(1, int(threads_per_worker))
    return list(
        Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_worker)(func, args, threads_per_worker) for args in arg_tuples
        )
    )
