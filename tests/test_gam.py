"""Tests for the distributional survival GAM (gamfit-library backend).

The survival GAM delegates to the ``gamfit`` Python library (PyO3
bindings to SauersML/gam's Rust engine). These tests keep the surface
compact: we sanity-check the library is importable, that a small fit
completes within the laptop's budget, shapes are correct, and BMA
weights behave.
"""

from __future__ import annotations

import time as _time

import gamfit
import numpy as np


# ---------------------------------------------------------------------------
# 0. Library-present smoke test
# ---------------------------------------------------------------------------


def test_gam_import():
    info = gamfit.build_info()
    # Must expose the capabilities we rely on.
    caps = info.get("capabilities", [])
    assert "fit" in caps and "predict" in caps and "summary" in caps, caps


# ---------------------------------------------------------------------------
# 1. Fit smoke test (< 60 s on small data)
# ---------------------------------------------------------------------------


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
    progress_messages = []
    gam_model = fit_survival_gam(
        time,
        event,
        X,
        columns=parents,
        n_uncertainty_slices=100,
        progress=progress_messages.append,
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
    assert any(message.startswith("fit start ") for message in progress_messages)
    assert any(message.startswith("fit complete") for message in progress_messages)


# ---------------------------------------------------------------------------
# 2. Predict-shape conventions
# ---------------------------------------------------------------------------


def test_predict_shape(small_data):
    """Check (n_uncertainty_slices, n_new, n_t) shape contracts."""
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
        n_uncertainty_slices=n_post,
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


def test_predict_survival_mean_uses_chunked_surface():
    from causal_pred.gam.survival import _SubmodelFit, _predict_survival_matrix

    n_new = 11
    t_grid = np.arange(7, dtype=float)
    backend_t_grid = 1.0 + t_grid
    X = np.zeros((n_new, 1), dtype=float)

    class _Prediction:
        def survival_at(self, _times):
            raise ValueError(
                "dense survival curves are limited to diagnostic subsets"
            )

        def survival_at_chunks(self, times, *, people_chunk=50000, time_grid_chunk=64):
            assert np.array_equal(times, backend_t_grid)
            yield slice(0, 5), slice(0, 3), np.full((5, 3), 0.9)
            yield slice(0, 5), slice(3, 7), np.full((5, 4), 0.8)
            yield slice(5, 11), slice(0, 7), np.full((6, 7), 0.7)

    class _Model:
        def predict(self, df_new):
            assert df_new.shape[0] == n_new
            return _Prediction()

    fit = _SubmodelFit(
        model=_Model(),
        columns=("x",),
        kinds=("continuous",),
        n_train=10,
        n_events=3,
        formula="Surv(entry, exit, event) ~ s(x, type=ps, knots=10)",
        survival_likelihood="location-scale",
        train_summary={},
    )
    S = _predict_survival_matrix(fit, X, t_grid)

    assert S.shape == (n_new, t_grid.size)
    np.testing.assert_allclose(S[:5, :3], 0.9)
    np.testing.assert_allclose(S[:5, 3:], 0.8)
    np.testing.assert_allclose(S[5:, :], 0.7)


# ---------------------------------------------------------------------------
# 3. BMA weights: 0.9/0.1 gives a curve close to the 0.9 model
# ---------------------------------------------------------------------------


def test_bma_weights(small_data, monkeypatch):
    from causal_pred.data.nodes import NODE_INDEX
    import causal_pred.gam.survival as survival_mod

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

    class _FixedSurvivalFit:
        def __init__(self, offset: float):
            self.offset = float(offset)

        def predict_survival_mean(self, X_new, t_grid):
            row_term = (
                np.arange(X_new.shape[0], dtype=float).reshape(-1, 1) * 0.001
            )
            time_term = np.asarray(t_grid, dtype=float).reshape(1, -1) * 0.01
            return 0.9 - self.offset - row_term - time_term

        def predict_survival_variance(self, X_new, t_grid):
            return np.zeros((X_new.shape[0], len(t_grid)), dtype=float)

    def _fixed_fit(_time, _event, _X, columns, **_kwargs):
        offsets = {("BMI", "HbA1c"): 0.0, ("ancestry_PC1",): 0.2}
        return _FixedSurvivalFit(offsets[tuple(columns)])

    monkeypatch.setattr(survival_mod, "fit_survival_gam", _fixed_fit)

    out = survival_mod.bma_survival(
        parent_sets,
        weights,
        time,
        event,
        X_full,
        d.columns,
        t_grid,
        X_eval=X_eval,
        n_uncertainty_slices=50,
    )
    S_bma = out["survival_mean"]
    S_A = out["per_set_mean"][0]
    S_B = out["per_set_mean"][1]
    err_A = float(np.mean(np.abs(S_bma - S_A)))
    err_B = float(np.mean(np.abs(S_bma - S_B)))
    assert err_A < err_B, (
        f"BMA closer to B than A (weights [0.9, 0.1]): {err_A:.3f} vs {err_B:.3f}"
    )
    # Structural/parametric decomposition is non-negative.
    assert np.all(out["variance_parametric"] >= -1e-12)
    assert np.all(out["variance_structural"] >= -1e-12)
    total = out["variance_parametric"] + out["variance_structural"]
    assert np.allclose(total, out["variance_total"], atol=1e-9)


def test_bma_uses_gamfit_within_model_variance(monkeypatch):
    import causal_pred.gam.survival as survival_mod

    class _VarianceFit:
        def predict_survival_mean(self, X_new, t_grid):
            return np.full((X_new.shape[0], len(t_grid)), 0.4, dtype=float)

        def predict_survival_variance(self, X_new, t_grid):
            return np.full((X_new.shape[0], len(t_grid)), 0.04, dtype=float)

    monkeypatch.setattr(
        survival_mod,
        "fit_survival_gam",
        lambda *_args, **_kwargs: _VarianceFit(),
    )

    out = survival_mod.bma_survival(
        [(0,)],
        np.array([1.0]),
        np.ones(4),
        np.ones(4),
        np.zeros((4, 1)),
        ("x0",),
        np.array([1.0, 2.0]),
        X_eval=np.zeros((2, 1)),
        n_uncertainty_slices=3,
    )

    np.testing.assert_allclose(out["variance_parametric"], 0.04)


# ---------------------------------------------------------------------------
# 4. Censoring sanity: fit completes at 0% and ~50% censoring
# ---------------------------------------------------------------------------


def test_censoring_sanity():
    """Fit runs cleanly whether censoring rate is 0 or ~50%.

    We don't assert parameter equality; this test only checks that both
    censoring regimes produce finite, in-range survival curves.
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
            n_uncertainty_slices=50,
        )
        t_grid = np.array([2.0, 5.0, 10.0])
        S = gam_model.predict_survival(X[:5], t_grid)
        assert np.all(np.isfinite(S))
        assert np.all(S >= 0.0) and np.all(S <= 1.0 + 1e-9)
