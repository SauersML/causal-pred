"""Tests for the validation framework (known-edge recovery + metrics)."""

from __future__ import annotations

import numpy as np
import pytest

from causal_pred.data.nodes import (
    ALL_GROUND_TRUTH_EDGES,
    NODE_INDEX,
    NODE_NAMES,
    adjacency_from_edges,
)
from causal_pred.data.synthetic import simulate
from causal_pred.validation.known_edges import known_edge_recovery
from causal_pred.validation.metrics import (
    bootstrap_metric,
    brier_score,
    calibration_metrics,
    nagelkerke_r2,
    time_dependent_auc,
)


# ---------------------------------------------------------------------------
# 1 + 2: known_edge_recovery.
# ---------------------------------------------------------------------------

def test_known_edge_recovery_perfect():
    """Probabilities = ground-truth adjacency: AUPRC == 1, AUROC ~ 1, all
    edges above every threshold, permutation p-values small."""
    A = adjacency_from_edges(ALL_GROUND_TRUTH_EDGES).astype(float)
    rng = np.random.default_rng(0)
    out = known_edge_recovery(
        A, ALL_GROUND_TRUTH_EDGES, NODE_NAMES,
        n_permute=500, rng=rng,
    )
    assert out["auprc"] == pytest.approx(1.0, abs=1e-9)
    assert out["auroc"] == pytest.approx(1.0, abs=1e-9)
    assert out["observed_recovery"][0.5] == pytest.approx(1.0)
    assert out["observed_recovery"][0.9] == pytest.approx(1.0)
    for tau in out["thresholds"]:
        assert out["recovery_pvalue"][tau] <= 0.01, (tau, out)


def test_known_edge_recovery_null():
    """Uniform-random edge_probs: AUROC ~ 0.5, AUPRC ~ density,
    permutation p-values are roughly uniform."""
    p = len(NODE_NAMES)
    np.random.default_rng(0)
    # Average p-values across several independent random matrices.
    pvals = []
    aurocs = []
    auprcs = []
    density = len(ALL_GROUND_TRUTH_EDGES) / (p * (p - 1))
    for trial in range(20):
        trng = np.random.default_rng(1000 + trial)
        M = trng.uniform(0.0, 1.0, size=(p, p))
        np.fill_diagonal(M, np.nan)
        out = known_edge_recovery(
            M, ALL_GROUND_TRUTH_EDGES, NODE_NAMES,
            n_permute=200, rng=trng,
        )
        aurocs.append(out["auroc"])
        auprcs.append(out["auprc"])
        # Average the recovery p-values across the supplied thresholds.
        pvals.append(np.mean(list(out["recovery_pvalue"].values())))

    assert 0.35 < np.mean(aurocs) < 0.65, np.mean(aurocs)
    assert abs(np.mean(auprcs) - density) < 0.05, (np.mean(auprcs), density)
    assert np.mean(pvals) > 0.05
    # Uniform-ish: mean should be near 0.5, within 0.15.
    assert abs(np.mean(pvals) - 0.5) < 0.15, np.mean(pvals)


def test_known_edge_recovery_scores_only_eligible_edges():
    names = ("a", "b", "c", "d")
    scores = np.zeros((4, 4), dtype=float)
    scores[0, 1] = 0.9
    scores[2, 3] = 0.9
    eligible = np.zeros((4, 4), dtype=bool)
    eligible[0, 1] = True
    eligible[0, 2] = True
    eligible[1, 2] = True

    out = known_edge_recovery(
        scores,
        [("a", "b"), ("c", "d")],
        names,
        eligible_edges=eligible,
        n_permute=20,
        rng=np.random.default_rng(0),
    )

    assert out["n_valid_ground_truth_edges"] == 1
    assert out["null_model"]["n_usable_cells"] == 3
    assert out["null_model"]["type"] == "eligible_value_permutation"
    assert "degree" not in out["null_model"]["type"]
    assert out["observed_recovery"][0.5] == pytest.approx(1.0)
    assert out["per_edge"][("c", "d")]["masked"]


# ---------------------------------------------------------------------------
# 3 + 4: Nagelkerke R^2 extremes.
# ---------------------------------------------------------------------------

def test_nagelkerke_r2_zero_model():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=1000).astype(float)
    r2 = nagelkerke_r2(y, np.full_like(y, y.mean()))
    assert abs(r2) < 1e-8


def test_nagelkerke_r2_perfect():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=1000).astype(float)
    r2 = nagelkerke_r2(y, y)
    assert r2 > 0.999


# ---------------------------------------------------------------------------
# 5: Brier decomposition sums.
# ---------------------------------------------------------------------------

def test_brier_decomposition_uses_binned_reliability():
    rng = np.random.default_rng(1)
    n = 2000
    # Somewhat miscalibrated predictions so that reliability > 0.
    p_true = rng.uniform(0.0, 1.0, size=n)
    y = rng.binomial(1, p_true).astype(float)
    p_pred = np.clip(p_true + rng.normal(0, 0.05, size=n), 0.01, 0.99)
    out = calibration_metrics(y, p_pred, n_bins=10, strategy="quantile")
    bins = out["reliability"]
    expected = (
        np.sum(
            bins["bin_counts"]
            * (bins["mean_predicted"] - bins["mean_observed"]) ** 2
        )
        / n
    )
    assert out["brier_decomposition"]["reliability"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6: ECE monotone.
# ---------------------------------------------------------------------------

def test_ece_monotone():
    rng = np.random.default_rng(2)
    n = 5000
    p_true = rng.uniform(0.0, 1.0, size=n)
    y = rng.binomial(1, p_true).astype(float)
    # Perfectly calibrated: p_pred = p_true.
    calib = calibration_metrics(y, p_true, n_bins=10, strategy="quantile")
    # Over-confident: push probabilities toward 0/1.
    p_over = np.where(p_true > 0.5,
                       np.clip(p_true + 0.3, 0.0, 1.0),
                       np.clip(p_true - 0.3, 0.0, 1.0))
    over = calibration_metrics(y, p_over, n_bins=10, strategy="quantile")
    assert calib["ece"] < 0.03
    assert over["ece"] > calib["ece"] + 0.05


# ---------------------------------------------------------------------------
# 7: Hosmer-Lemeshow p-values.
# ---------------------------------------------------------------------------

def test_hl_chi2():
    rng = np.random.default_rng(3)
    n = 2000
    p_true = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p_true).astype(float)
    well = calibration_metrics(y, p_true, n_bins=10, strategy="quantile")
    assert well["hl_pvalue"] > 0.05, well
    # Badly calibrated: flip probabilities.
    bad = calibration_metrics(y, 1.0 - p_true, n_bins=10, strategy="quantile")
    assert bad["hl_pvalue"] < 0.01, bad


# ---------------------------------------------------------------------------
# 8: Time-dependent AUC on synthetic data.
# ---------------------------------------------------------------------------

def test_time_dependent_auc_sanity():
    d = simulate(n=2000, rng=np.random.default_rng(42))
    bmi = d.X[:, NODE_INDEX["bmi"]]
    pgs = d.X[:, NODE_INDEX["pgs_t2d"]]
    risk = (bmi - bmi.mean()) / bmi.std() + (pgs - pgs.mean()) / pgs.std()
    eval_times = np.array([5.0, 10.0, 15.0])
    out = time_dependent_auc(d.time, d.event, risk, eval_times)
    # At tau = 10 years, informed risk score beats 0.6.
    assert out["auc"][1] > 0.6, out

    rng = np.random.default_rng(0)
    rand_risk = rng.normal(size=d.n)
    out_rand = time_dependent_auc(d.time, d.event, rand_risk, eval_times)
    assert abs(out_rand["auc"][1] - 0.5) < 0.05, out_rand


def test_time_dependent_auc_uses_time_specific_scores():
    time = np.array([1.0, 2.0, 8.0, 9.0])
    event = np.array([1, 1, 0, 0])
    eval_times = np.array([3.0, 7.0])
    scores = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.9],
            [0.1, 0.8],
        ]
    )

    out = time_dependent_auc(time, event, scores, eval_times)

    assert out["auc"][0] == pytest.approx(1.0)
    assert out["auc"][1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9: IPCW Brier reduces to unweighted on uncensored data.
# ---------------------------------------------------------------------------

def test_brier_ipcw_sanity():
    d = simulate(n=500, censoring_rate=0.0, rng=np.random.default_rng(0))
    # All events observed.
    assert np.all(d.event == 1)
    eval_times = np.linspace(1.0, float(np.max(d.time)) - 1.0, 5)
    # Use KM as the survival predictor -- this is the "baseline" branch
    # inside brier_score, so its IPCW Brier should equal the unweighted
    # Brier when there is no censoring.
    from causal_pred.validation.metrics import _km_estimator, _km_eval
    km_t, km_S = _km_estimator(d.time, d.event)
    S_km = np.broadcast_to(_km_eval(km_t, km_S, eval_times),
                            (d.n, eval_times.size)).copy()

    out = brier_score(d.time, d.event, S_km, eval_times)
    # Unweighted Brier (G == 1 everywhere because no censoring).
    unweighted = np.empty(eval_times.size)
    for j, tau in enumerate(eval_times):
        case = (d.time <= tau) & (d.event == 1)
        ctrl = d.time > tau
        term = np.where(case, (0.0 - S_km[:, j]) ** 2, 0.0) + \
               np.where(ctrl, (1.0 - S_km[:, j]) ** 2, 0.0)
        unweighted[j] = float(np.mean(term))
    np.testing.assert_allclose(out["brier"], unweighted, atol=1e-10)


def test_integrated_brier_uses_eval_grid_interval():
    time = np.array([1.0, 3.0, 5.0])
    event = np.array([1, 1, 1])
    eval_times = np.array([1.0, 3.0])
    survival_pred = np.full((3, 2), 0.5)

    out = brier_score(time, event, survival_pred, eval_times)

    np.testing.assert_allclose(out["brier"], np.array([0.25, 0.25]))
    assert out["ibs"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 10: Bootstrap CI coverage.
# ---------------------------------------------------------------------------

def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(11)
    true_mean = 2.5
    hits = 0
    n_trials = 50
    for _ in range(n_trials):
        x = rng.normal(loc=true_mean, scale=1.0, size=200)
        out = bootstrap_metric(lambda a: float(np.mean(a)), x,
                                n_boot=200, ci=0.95, rng=rng)
        if out["lo"] <= true_mean <= out["hi"]:
            hits += 1
    assert hits / n_trials >= 0.9, hits / n_trials
