"""Tests for the distributional survival GAM (gam-library backend).

The survival GAM delegates to the ``gam`` Python library (PyO3 bindings
to SauersML/gam's Rust engine).  These tests keep the surface compact:
we sanity-check the library is importable, that a small fit completes
within the laptop's budget, shapes are correct, and BMA weights behave.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

try:
    import gam as _gam_lib

    _HAS_GAM = True
except Exception:  # pragma: no cover -- env without gam
    _gam_lib = None
    _HAS_GAM = False

needs_gam = pytest.mark.skipif(not _HAS_GAM, reason="gam library not installed")


# ---------------------------------------------------------------------------
# 0. Library-present smoke test
# ---------------------------------------------------------------------------


@needs_gam
def test_gam_import():
    info = _gam_lib.build_info()
    assert info.get("available") is True, info
    # Must expose the capabilities we rely on.
    caps = info.get("capabilities", [])
    assert "fit" in caps and "predict" in caps and "summary" in caps, caps


# ---------------------------------------------------------------------------
# 1. Fit smoke test (< 60 s on small data)
# ---------------------------------------------------------------------------


@needs_gam
def test_fit_smoke(medium_data):
    """Fit on ~300 rows with 3 covariates and check runtime + monotonicity."""
    from causal_pred.data.nodes import NODE_INDEX
    from causal_pred.gam.survival import fit_survival_gam

    d = medium_data
    # Cap rows for the laptop's sake.
    n_keep = 300
    keep = slice(0, n_keep)

    parents = ("BMI", "HbA1c", "age")  # 3 continuous covariates
    cols = [NODE_INDEX[p] for p in parents]
    X = d.X[keep][:, cols]
    time = d.time[keep]
    event = d.event[keep]

    t0 = _time.perf_counter()
    gam_model = fit_survival_gam(
        time,
        event,
        X,
        columns=parents,
        n_samples=100,
        rng=np.random.default_rng(7),
    )
    elapsed = _time.perf_counter() - t0
    assert elapsed < 60.0, f"fit took {elapsed:.1f}s (limit 60)"

    # Monotonicity of S(t|x).
    t_grid = np.linspace(0.5, np.max(time) * 0.9, 25)
    S = gam_model.predict_survival(X[:4], t_grid)
    assert S.shape[2] == t_grid.size
    assert np.all(S >= 0.0) and np.all(S <= 1.0 + 1e-9)
    diffs = np.diff(S, axis=-1)
    assert np.all(diffs <= 1e-9), f"survival not monotone (max +dS = {diffs.max():.2e})"

    # Diagnostics dict has the library-level fields.
    diag = gam_model.diagnostics
    for key in (
        "reml_iterations",
        "reml_score",
        "edf_total",
        "sigma_residual",
        "n_events",
        "converged",
        "formula",
    ):
        assert key in diag, f"missing diagnostic {key!r}"
    assert bool(diag["converged"]) is True


# ---------------------------------------------------------------------------
# 2. Predict-shape conventions
# ---------------------------------------------------------------------------


@needs_gam
def test_predict_shape(small_data):
    """Check (n_samples, n_new, n_t) shape contracts."""
    from causal_pred.data.nodes import NODE_INDEX
    from causal_pred.gam.survival import fit_survival_gam

    d = small_data
    n_keep = 300
    parents = ("BMI", "age")
    cols = [NODE_INDEX[p] for p in parents]
    X = d.X[:n_keep][:, cols]
    time = d.time[:n_keep]
    event = d.event[:n_keep]

    n_post = 37  # unusual number to catch hard-coded defaults
    gam_model = fit_survival_gam(
        time,
        event,
        X,
        columns=parents,
        n_samples=n_post,
        rng=np.random.default_rng(1),
    )

    X_new = X[:4]
    t_grid = np.array([1.0, 2.0, 5.0, 10.0])

    S = gam_model.predict_survival(X_new, t_grid)
    assert S.shape == (n_post, 4, 4), S.shape

    hz = gam_model.predict_hazard(X_new, t_grid)
    assert hz.shape == (n_post, 4, 4), hz.shape

    med = gam_model.predict_median_survival(X_new)
    assert med.shape == (n_post, 4), med.shape
    assert np.all(med > 0)


# ---------------------------------------------------------------------------
# 3. BMA weights: 0.9/0.1 gives a curve close to the 0.9 model
# ---------------------------------------------------------------------------


@needs_gam
def test_bma_weights(small_data):
    from causal_pred.data.nodes import NODE_INDEX
    from causal_pred.gam.survival import bma_survival

    d = small_data
    # Cap rows to respect the laptop budget.
    n_keep = 300
    X_full = d.X[:n_keep]
    time = d.time[:n_keep]
    event = d.event[:n_keep]

    set_A = (NODE_INDEX["BMI"], NODE_INDEX["HbA1c"])  # strong predictors
    set_B = (NODE_INDEX["ancestry_PC1"],)  # weak predictor
    parent_sets = [set_A, set_B]
    weights = np.array([0.9, 0.1])
    t_grid = np.array([2.0, 5.0, 10.0])
    X_eval = X_full[:10]

    out = bma_survival(
        parent_sets,
        weights,
        time,
        event,
        X_full,
        d.columns,
        t_grid,
        X_eval=X_eval,
        n_samples=50,
        rng=np.random.default_rng(23),
    )
    S_bma = out["S_bma"]
    S_A = out["per_model_mean"][0]
    S_B = out["per_model_mean"][1]
    err_A = float(np.mean(np.abs(S_bma - S_A)))
    err_B = float(np.mean(np.abs(S_bma - S_B)))
    assert err_A < err_B, (
        f"BMA closer to B than A (weights [0.9, 0.1]): {err_A:.3f} vs {err_B:.3f}"
    )
    # Structural/parametric decomposition is non-negative.
    assert np.all(out["var_parametric"] >= -1e-12)
    assert np.all(out["var_structural"] >= -1e-12)
    total = out["var_parametric"] + out["var_structural"]
    assert np.allclose(total, out["var_total"], atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Censoring sanity: fit completes at 0% and ~50% censoring
# ---------------------------------------------------------------------------


@needs_gam
def test_censoring_sanity():
    """Fit runs cleanly whether censoring rate is 0 or ~50%.

    We don't assert parameter equality -- the library (v0.1.15) exposes
    only the 'standard' model class, so the wrapper uses a complete-case
    AFT workaround and biases are expected.  This test only checks that
    both censoring regimes produce finite, in-range survival curves.
    """
    from causal_pred.data.synthetic import simulate
    from causal_pred.data.nodes import NODE_INDEX
    from causal_pred.gam.survival import fit_survival_gam

    parents = ("BMI",)  # single smooth is enough here
    col = [NODE_INDEX[p] for p in parents]

    for censoring_rate, seed in [(0.0, 10), (0.5, 11)]:
        d = simulate(
            n=300, censoring_rate=censoring_rate, rng=np.random.default_rng(seed)
        )
        X = d.X[:, col]
        gam_model = fit_survival_gam(
            d.time,
            d.event,
            X,
            columns=parents,
            n_samples=50,
            rng=np.random.default_rng(seed + 1),
        )
        t_grid = np.array([2.0, 5.0, 10.0])
        S = gam_model.predict_survival(X[:5], t_grid)
        assert np.all(np.isfinite(S))
        assert np.all(S >= 0.0) and np.all(S <= 1.0 + 1e-9)
