"""Validation framework.

Public interface
----------------

``known_edge_recovery(edge_probs, ground_truth_edges, n_permute=1000,
                      rng=None) -> dict``
    Recovery rate of ground-truth edges versus a permutation null.

``nagelkerke_r2(y_true, p_pred) -> float``

``calibration_metrics(y_true, p_pred, n_bins=10) -> dict``
    Brier score, expected calibration error, reliability bins.

``bootstrap_metric(fn, data, n_boot=500, rng=None) -> (mean, lo, hi)``
"""

from .known_edges import known_edge_recovery  # noqa: F401
from .metrics import (  # noqa: F401
    nagelkerke_r2,
    calibration_metrics,
    bootstrap_metric,
    time_dependent_auc,
    brier_score,
)
