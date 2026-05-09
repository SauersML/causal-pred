"""Tests for :mod:`causal_pred.benchmarks` and ``scripts/benchmark.py``."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from causal_pred.data.synthetic import simulate  # noqa: E402
from causal_pred.benchmarks import (  # noqa: E402
    _surv_at_times,
    _surv_metrics,
    _train_test_indices,
    run_kaplan_meier,
    run_cox_ph,
    run_naive_logistic,
    run_mr_ivw,
    run_causal_pred,
)


# ---- per-baseline smoke tests -----------------------------------------------

_REQUIRED_SURVIVAL_KEYS = {
    "nagelkerke_at_10y",
    "time_dep_auc",
    "ibs",
    "ibs_km",
    "scaled_brier",
    "runtime_s",
    "model",
    "evaluation",
    "n_train",
    "n_test",
}


@pytest.fixture(scope="module")
def n500_data():
    return simulate(n=500, rng=np.random.default_rng(7))


def test_km_runs(n500_data):
    t0 = time.perf_counter()
    out = run_kaplan_meier(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    assert out["evaluation"] == "held_out"
    assert out["n_train"] + out["n_test"] == n500_data.n
    td = out["time_dep_auc"]
    assert set(td) == {"times", "auc", "integrated_auc"}
    # KM is marginal: td-AUC must be 0.5 because every held-out row receives
    # the same train-set survival curve.
    assert 0.8 < out["scaled_brier"] < 1.2
    for a in td["auc"]:
        assert abs(a - 0.5) < 1e-6


def test_cox_runs(n500_data):
    t0 = time.perf_counter()
    out = run_cox_ph(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    assert out["evaluation"] == "held_out"
    assert out["n_train"] + out["n_test"] == n500_data.n
    # Cox should do materially better than KM on this structured data.
    assert out["scaled_brier"] < 0.95
    # IBS must be finite and in a sensible range.
    assert 0.0 < out["ibs"] < 0.5
    # Time-dependent AUC at t=10y should be > 0.5.
    td = out["time_dep_auc"]
    idx10 = td["times"].index(10.0)
    assert td["auc"][idx10] > 0.55


def test_naive_logistic_runs(n500_data):
    t0 = time.perf_counter()
    out = run_naive_logistic(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    assert out["evaluation"] == "held_out"
    assert out["n_train"] + out["n_test"] == n500_data.n
    assert out["n_train_determined"] <= out["n_train"]
    assert out["n_test_determined"] <= out["n_test"]
    assert np.isfinite(out["nagelkerke_at_10y"])
    td = out["time_dep_auc"]
    idx10 = td["times"].index(10.0)
    assert td["auc"][idx10] > 0.55


def test_benchmark_split_is_stratified_and_disjoint(n500_data):
    train_idx, test_idx = _train_test_indices(n500_data.event)

    assert train_idx.size + test_idx.size == n500_data.n
    assert set(train_idx).isdisjoint(set(test_idx))
    assert 0 < n500_data.event[test_idx].sum() < test_idx.size


def test_survival_interpolation_uses_exact_requested_times():
    grid = np.array([0.0, 10.0, 20.0])
    surv = np.array([[1.0, 0.8, 0.2], [1.0, 0.6, 0.0]])

    out = _surv_at_times(surv, grid, [5.0, 15.0])

    np.testing.assert_allclose(out, [[0.9, 0.5], [0.8, 0.3]])


def test_survival_metrics_exclude_early_censoring_and_use_time_specific_auc(
    monkeypatch,
):
    from causal_pred import benchmarks

    time_arr = np.array([4.0, 8.0, 12.0, 16.0])
    event_arr = np.array([0, 1, 0, 0])
    t_grid = np.array([5.0, 10.0, 15.0])
    survival = np.array(
        [
            [0.96, 0.90, 0.82],
            [0.92, 0.70, 0.50],
            [0.98, 0.86, 0.65],
            [0.99, 0.80, 0.60],
        ]
    )
    seen = {}

    def fake_nagelkerke(y, p):
        seen["r2_y"] = np.asarray(y).copy()
        seen["r2_p"] = np.asarray(p).copy()
        return 0.25

    def fake_time_dependent_auc(*, time, event, risk_score, eval_times):
        seen["auc_risk"] = np.asarray(risk_score).copy()
        return {
            "times": list(eval_times),
            "auc": [0.6, 0.7, 0.8],
            "integrated_auc": 0.7,
        }

    def fake_brier_score(*, time, event, survival_pred, eval_times):
        return {"ibs": 0.1, "ibs_km": 0.2, "scaled_brier": 0.5}

    monkeypatch.setattr(benchmarks, "nagelkerke_r2", fake_nagelkerke)
    monkeypatch.setattr(benchmarks, "time_dependent_auc", fake_time_dependent_auc)
    monkeypatch.setattr(benchmarks, "brier_score", fake_brier_score)

    out = _surv_metrics(time_arr, event_arr, survival, t_grid)

    np.testing.assert_array_equal(seen["r2_y"], np.array([1, 0, 0]))
    np.testing.assert_allclose(seen["r2_p"], 1.0 - survival[[1, 2, 3], 1])
    np.testing.assert_allclose(seen["auc_risk"], 1.0 - survival)
    assert out["nagelkerke_n_used"] == 3
    assert out["nagelkerke_n_indeterminate"] == 1


def test_mr_ivw_runs():
    out = run_mr_ivw()
    assert out["model"] == "mr_ivw"
    # Cached OpenGWAS MR estimates should still enrich known edges strongly.
    assert out["edge_auprc"] > 0.5
    # AUROC should also be above chance.
    assert out["edge_auroc"] > 0.55
    assert out["n_tests"] > 0
    # The Bonferroni set should be reasonable (not everything, not nothing).
    assert 0 < out["significant_edges"] < out["n_tests"]


def test_causal_pred_runs_with_gamfit():
    data = simulate(n=120, rng=np.random.default_rng(11))
    out = run_causal_pred(
        data,
        mcmc_iter=5,
        mcmc_chains=1,
        gam_samples=2,
        rng=np.random.default_rng(12),
    )

    assert out["model"] == "causal_pred"
    assert out["backend"] == "gamfit"
    assert out["evaluation"] == "held_out"
    assert out["n_train"] + out["n_test"] == data.n
    assert out["mcmc_n_samples"] > 0
    assert out["parent_sets"]
    assert np.isfinite(out["ibs"])
