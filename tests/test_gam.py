"""Tests for the survival GAM (gamfit-library backend)."""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


class _FakeSummary:
    def to_dict(self):
        return {
            "reml_iterations": 3,
            "reml_score": 12.5,
            "edf_total": 2.0,
            "sigma_residual": None,
            "deviance": 4.0,
        }


class _FakePrediction:
    def __init__(self, df_new, with_uncertainty: bool):
        self.n_rows = int(df_new.shape[0])
        self.with_uncertainty = bool(with_uncertainty)

    def survival_at(self, times):
        horizon = np.asarray(times, dtype=float).reshape(1, -1) - 1.0
        row = np.arange(self.n_rows, dtype=float).reshape(-1, 1)
        return np.exp(-0.03 * horizon) * (1.0 - 0.0001 * row)

    def survival_se_at(self, times):
        if not self.with_uncertainty:
            return None
        return np.full((self.n_rows, len(times)), 0.02, dtype=float)


class _FakeModel:
    def summary(self):
        return _FakeSummary()

    def predict(self, df_new, *, with_uncertainty=False):
        return _FakePrediction(df_new, with_uncertainty)


def test_gam_import():
    info = gamfit.build_info()
    caps = info.get("capabilities", [])
    assert "fit" in caps and "predict" in caps and "summary" in caps, caps


def test_fit_smoke(monkeypatch):
    """Fit wrapper calls gamfit location-scale and returns uncertainty slices."""
    import causal_pred.gam.survival as survival_mod

    seen = {}

    def fake_fit(df, formula, *, survival_likelihood, baseline_target, config):
        seen["df_columns"] = tuple(df.columns)
        seen["formula"] = formula
        seen["survival_likelihood"] = survival_likelihood
        seen["baseline_target"] = baseline_target
        seen["config"] = dict(config)
        seen["df"] = df.copy()
        return _FakeModel()

    monkeypatch.setattr(survival_mod.gam, "fit", fake_fit)
    monkeypatch.setattr(
        survival_mod.gam,
        "build_info",
        lambda: {"version": "test-gamfit"},
    )

    time = np.array([1.0, 2.0, 4.0, 8.0])
    event = np.array([1, 0, 1, 0])
    X = np.column_stack(
        [
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([20.0, 25.0, 30.0, 35.0]),
        ]
    )
    progress_messages = []
    gam_model = survival_mod.fit_survival_gam(
        time,
        event,
        X,
        columns=("sex", "BMI"),
        n_uncertainty_slices=9,
        progress=progress_messages.append,
    )

    assert seen["survival_likelihood"] == "location-scale"
    assert seen["baseline_target"] == "linear"
    assert seen["config"] == {"noise_formula": "1"}
    assert seen["df_columns"] == ("entry", "exit", "event", "sex", "BMI")
    assert seen["formula"] == "Surv(entry, exit, event) ~ sex + BMI"
    np.testing.assert_allclose(seen["df"][["sex", "BMI"]].mean(axis=0), 0.0)
    np.testing.assert_allclose(seen["df"][["sex", "BMI"]].std(axis=0, ddof=0), 1.0)

    t_grid = np.array([0.5, 2.0, 5.0])
    S = gam_model.predict_survival(X[:2], t_grid)
    S_se = gam_model.predict_survival_se(X[:2], t_grid)
    assert S.shape == (9, 2, 3)
    assert S_se.shape == (2, 3)
    np.testing.assert_allclose(S_se, 0.02)
    assert not np.allclose(S[0], S[-1])
    assert np.all(S >= 0.0) and np.all(S <= 1.0 + 1e-9)
    diffs = np.diff(S, axis=-1)
    assert np.all(diffs <= 1e-9), f"survival not monotone (max +dS = {diffs.max():.2e})"

    diag = gam_model.diagnostics
    for key in (
        "reml_iterations",
        "reml_score",
        "edf_total",
        "sigma_residual",
        "n_events",
        "converged",
        "formula",
        "baseline_target",
        "noise_formula",
        "covariate_center",
        "covariate_scale",
    ):
        assert key in diag, f"missing diagnostic {key!r}"
    assert bool(diag["converged"]) is True
    assert any(message.startswith("fit start ") for message in progress_messages)
    assert any(message.startswith("fit complete") for message in progress_messages)


def test_predict_shape():
    """Check (n_uncertainty_slices, n_new, n_t) shape contracts."""
    from causal_pred.gam.survival import SurvivalGAM, _SubmodelFit

    n_slices = 37
    gam_model = SurvivalGAM(
        columns=("BMI", "age"),
        diagnostics={},
        _fit=_SubmodelFit(
            model=_FakeModel(),
            columns=("BMI", "age"),
            n_train=10,
            n_events=4,
            formula="Surv(entry, exit, event) ~ BMI + age",
            train_summary={},
            x_center=np.zeros(2),
            x_scale=np.ones(2),
        ),
        _n_uncertainty_slices=n_slices,
    )

    X_new = np.zeros((4, 2))
    t_grid = np.array([1.0, 2.0, 5.0, 10.0])

    S = gam_model.predict_survival(X_new, t_grid)
    assert S.shape == (n_slices, 4, 4), S.shape

    hz = gam_model.predict_hazard(X_new, t_grid)
    assert hz.shape == (n_slices, 4, 4), hz.shape

    med = gam_model.predict_median_survival(X_new)
    assert med.shape == (n_slices, 4), med.shape
    assert np.all(med > 0)


def test_predict_survival_mean_uses_gamfit_surface():
    from causal_pred.gam.survival import (
        _SubmodelFit,
        _predict_survival_matrix,
        _predict_survival_surfaces,
    )

    n_new = 11
    t_grid = np.arange(7, dtype=float)
    X = np.zeros((n_new, 1), dtype=float)
    surface = 0.9 - np.arange(n_new, dtype=float).reshape(-1, 1) * 0.01
    surface = surface - np.arange(t_grid.size, dtype=float).reshape(1, -1) * 0.02

    class _Prediction:
        def __init__(self, *, with_uncertainty: bool):
            self.with_uncertainty = bool(with_uncertainty)

        def survival_at(self, times):
            np.testing.assert_allclose(times, 1.0 + t_grid)
            return surface

        def survival_se_at(self, times):
            np.testing.assert_allclose(times, 1.0 + t_grid)
            if not self.with_uncertainty:
                return None
            return np.full((n_new, t_grid.size), 0.05)

    class _Model:
        def predict(self, df_new, *, with_uncertainty=False):
            assert df_new.shape[0] == n_new
            np.testing.assert_allclose(
                df_new["exit"].to_numpy(),
                np.full(n_new, 1.0 + np.max(t_grid)),
            )
            return _Prediction(with_uncertainty=with_uncertainty)

    fit = _SubmodelFit(
        model=_Model(),
        columns=("x",),
        n_train=10,
        n_events=3,
        formula="Surv(entry, exit, event) ~ x",
        train_summary={},
        survival_likelihood="location-scale",
        x_center=np.zeros(1),
        x_scale=np.ones(1),
    )
    S = _predict_survival_matrix(fit, X, t_grid)

    assert S.shape == (n_new, t_grid.size)
    np.testing.assert_allclose(S, surface)
    S_mean, S_se = _predict_survival_surfaces(fit, X, t_grid)
    np.testing.assert_allclose(S_mean, surface)
    np.testing.assert_allclose(S_se, 0.05)


def test_bma_weights(small_data, monkeypatch):
    from causal_pred.data.nodes import NODE_INDEX
    import causal_pred.gam.survival as survival_mod

    d = small_data
    n_keep = 300
    X_full = d.X[:n_keep]
    time = d.time[:n_keep]
    event = d.event[:n_keep]

    set_A = (NODE_INDEX["BMI"], NODE_INDEX["HbA1c"])
    set_B = (NODE_INDEX["ancestry_PC1"],)
    parent_sets = [set_A, set_B]
    weights = np.array([0.9, 0.1])
    t_grid = np.array([2.0, 5.0, 10.0])
    X_eval = X_full[:10]

    class _FixedSurvivalFit:
        def __init__(self, offset: float):
            self.offset = float(offset)

        def predict_survival_mean(self, X_new, t_grid):
            row_term = np.arange(X_new.shape[0], dtype=float).reshape(-1, 1) * 0.001
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


def test_survival_se_requires_gamfit_uncertainty():
    from causal_pred.gam.survival import SurvivalGAM, _SubmodelFit

    class _NoSePrediction:
        def __init__(self, n_rows: int):
            self.n_rows = int(n_rows)

        def survival_at(self, times):
            return np.full((self.n_rows, len(times)), 0.8, dtype=float)

        def survival_se_at(self, times):
            return None

    class _NoSeModel:
        def predict(self, df_new, *, with_uncertainty=False):
            return _NoSePrediction(df_new.shape[0])

    gam_model = SurvivalGAM(
        columns=("x",),
        diagnostics={},
        _fit=_SubmodelFit(
            model=_NoSeModel(),
            columns=("x",),
            n_train=4,
            n_events=2,
            formula="Surv(entry, exit, event) ~ x",
            train_summary={},
            survival_likelihood="location-scale",
            x_center=np.zeros(1),
            x_scale=np.ones(1),
        ),
    )

    with pytest.raises(RuntimeError, match="survival_se"):
        gam_model.predict_survival_se(np.zeros((2, 1)), np.array([1.0, 2.0]))
