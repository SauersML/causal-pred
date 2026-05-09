"""Survival GAM wrapper over the gamfit Python library.

The module is intentionally a thin wrapper over ``gamfit`` (PyO3 bindings to
SauersML/gam's Rust engine). gamfit fits the right-censored
Gompertz-Makeham GAMLSS survival model and evaluates survival curves and
response-scale standard errors at requested horizons. Bayesian model averaging
adds structural uncertainty across parent sets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gamfit as gam
import numpy as np
import pandas as pd
from scipy.special import ndtri


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]

_SURVIVAL_ENTRY_ORIGIN = 1.0
_SURVIVAL_LIKELIHOOD = "location-scale"
_SURVIVAL_BASELINE_TARGET = "gompertz-makeham"
_SURVIVAL_NOISE_FORMULA = "1"
_SURVIVAL_MODEL_FAMILY = "gamlss-gompertz-makeham"
_UNCERTAINTY_MODE = "gamfit_gompertz_makeham_gamlss_delta_method_response_se"
_UNCERTAINTY_SOURCE = "gamfit.SurvivalPrediction.survival_se_at"


def _progress_callback(progress: bool | ProgressCallback) -> Optional[ProgressCallback]:
    if callable(progress):
        return progress
    if progress:
        return logger.info
    return None


def _build_survival_formula(columns: Sequence[str]) -> str:
    return " + ".join(columns) if columns else "1"


@dataclass
class _SubmodelFit:
    """Fitted gamfit survival model artefacts."""

    model: Any
    columns: Tuple[str, ...]
    n_train: int
    n_events: int
    formula: str
    train_summary: Dict[str, Any]
    location_formula: str = "1"
    survival_likelihood: str = _SURVIVAL_LIKELIHOOD
    baseline_target: str = _SURVIVAL_BASELINE_TARGET
    noise_formula: str = _SURVIVAL_NOISE_FORMULA
    x_center: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    x_scale: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))


def _fit_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Tuple[str, ...],
    progress: bool | ProgressCallback = False,
    location_formula: Optional[str] = None,
    noise_formula: Optional[str] = None,
) -> _SubmodelFit:
    """Fit a gamfit Gompertz-Makeham GAMLSS survival model."""

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if event.shape != time.shape:
        raise ValueError("event must have the same shape as time")
    if not np.all(np.isfinite(time)) or np.any(time <= 0.0):
        raise ValueError("survival GAM requires finite positive follow-up times")
    if not np.all(np.isfinite(event)) or not set(np.unique(event).tolist()).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("event must contain only finite 0/1 indicators")
    if X.size == 0:
        X = X.reshape(time.shape[0], 0)
    else:
        X = X.reshape(time.shape[0], -1)
    if X.shape[0] != time.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != time length {time.shape[0]}")
    if X.shape[1] != len(columns):
        raise ValueError(f"columns length {len(columns)} != X.shape[1] {X.shape[1]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("survival GAM covariates must be finite")

    location_rhs = (
        _build_survival_formula(columns)
        if location_formula is None
        else str(location_formula)
    )
    sigma_rhs = _SURVIVAL_NOISE_FORMULA if noise_formula is None else str(noise_formula)
    formula = f"Surv(entry, exit, event) ~ {location_rhs}"
    if X.shape[1] > 0:
        x_center = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
        X_model = (X - x_center.reshape(1, -1)) / x_scale.reshape(1, -1)
    else:
        x_center = np.zeros(0, dtype=float)
        x_scale = np.ones(0, dtype=float)
        X_model = X
    df = pd.DataFrame(
        {
            "entry": np.full(time.shape[0], _SURVIVAL_ENTRY_ORIGIN, dtype=float),
            "exit": _SURVIVAL_ENTRY_ORIGIN + time,
            "event": event,
            **{name: X_model[:, i] for i, name in enumerate(columns)},
        }
    )

    emit = _progress_callback(progress)
    if emit is not None:
        emit(
            "fit start "
            f"n={time.shape[0]} events={int(np.sum(event > 0.0))} "
            f"p={len(columns)} likelihood={_SURVIVAL_LIKELIHOOD} "
            f"baseline={_SURVIVAL_BASELINE_TARGET} formula={formula} "
            f"noise_formula={sigma_rhs}"
        )
    model = gam.fit(
        df,
        formula,
        survival_likelihood=_SURVIVAL_LIKELIHOOD,
        baseline_target=_SURVIVAL_BASELINE_TARGET,
        config={"noise_formula": sigma_rhs},
    )
    train_summary: Dict[str, Any] = dict(model.summary().to_dict())
    if emit is not None:
        summary_items = []
        for key in (
            "reml_iterations",
            "iterations",
            "reml_score",
            "edf_total",
            "deviance",
            "sigma_residual",
        ):
            value = train_summary.get(key)
            if value is not None:
                summary_items.append(f"{key}={value}")
        suffix = " " + " ".join(summary_items) if summary_items else ""
        emit(f"fit complete{suffix}")

    return _SubmodelFit(
        model=model,
        columns=tuple(columns),
        n_train=int(time.shape[0]),
        n_events=int(np.sum(event > 0.0)),
        formula=formula,
        location_formula=location_rhs,
        train_summary=train_summary,
        survival_likelihood=_SURVIVAL_LIKELIHOOD,
        baseline_target=_SURVIVAL_BASELINE_TARGET,
        noise_formula=sigma_rhs,
        x_center=x_center.astype(float, copy=True),
        x_scale=x_scale.astype(float, copy=True),
    )


def _shape_prediction_inputs(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    p = len(fit.columns)
    X_new = np.asarray(X_new, dtype=float)
    if p > 0:
        X_new = X_new.reshape(-1, p)
    elif X_new.ndim == 2:
        if X_new.shape[1] != 0:
            raise ValueError(
                f"zero-parent survival model expected X_new with 0 columns, "
                f"got shape {X_new.shape}"
            )
        X_new = X_new.reshape(X_new.shape[0], 0)
    else:
        X_new = np.zeros((X_new.reshape(-1).shape[0], 0), dtype=float)

    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if not np.all(np.isfinite(t_grid)) or np.any(t_grid < 0.0):
        raise ValueError(
            "survival prediction grid must contain finite non-negative times"
        )
    if t_grid.size > 1 and np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("survival prediction grid must be strictly increasing")
    return X_new, t_grid, int(X_new.shape[0]), int(t_grid.shape[0])


def _model_matrix(fit: _SubmodelFit, X: np.ndarray) -> np.ndarray:
    if len(fit.columns) == 0:
        return X.reshape(X.shape[0], 0)
    if fit.x_center.shape[0] != len(fit.columns) or fit.x_scale.shape[0] != len(
        fit.columns
    ):
        raise RuntimeError("gamfit survival fit is missing covariate scaling metadata")
    return (X - fit.x_center.reshape(1, -1)) / fit.x_scale.reshape(1, -1)


def _prediction_frame(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    n_new = int(X_new.shape[0])
    X_model = _model_matrix(fit, X_new)
    max_horizon = float(np.max(t_grid)) if t_grid.size else 0.0
    exit_time = _SURVIVAL_ENTRY_ORIGIN + max(max_horizon, 1e-9)
    return pd.DataFrame(
        {
            "entry": np.full(n_new, _SURVIVAL_ENTRY_ORIGIN, dtype=float),
            "exit": np.full(n_new, exit_time, dtype=float),
            "event": np.zeros(n_new, dtype=float),
            **{name: X_model[:, i] for i, name in enumerate(fit.columns)},
        }
    )


def _validate_gamfit_surface(
    values: Any,
    *,
    n_new: int,
    n_t: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_new, n_t):
        raise RuntimeError(
            f"gamfit returned {label} surface of shape {arr.shape}; "
            f"expected ({n_new}, {n_t})"
        )
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"gamfit returned non-finite {label} values")
    return arr


def _predict_survival_grid(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
    *,
    with_uncertainty: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    X_new, t_grid, n_new, n_t = _shape_prediction_inputs(fit, X_new, t_grid)
    if n_new == 0 or n_t == 0:
        empty = np.zeros((n_new, n_t), dtype=float)
        return empty, empty if with_uncertainty else None
    pred = fit.model.predict(
        _prediction_frame(fit, X_new, t_grid),
        with_uncertainty=with_uncertainty,
    )
    backend_t_grid = _SURVIVAL_ENTRY_ORIGIN + t_grid
    S_mean = _validate_gamfit_surface(
        pred.survival_at(backend_t_grid),
        n_new=n_new,
        n_t=n_t,
        label="survival",
    )
    S_mean = np.minimum.accumulate(np.clip(S_mean, 0.0, 1.0), axis=1)
    if not with_uncertainty:
        return S_mean, None
    values = pred.survival_se_at(backend_t_grid)
    if values is None:
        raise RuntimeError("gamfit did not return survival_se for survival prediction")
    S_se = _validate_gamfit_surface(
        values,
        n_new=n_new,
        n_t=n_t,
        label="survival_se",
    )
    return S_mean, np.clip(S_se, 0.0, None)


def _predict_survival_matrix(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    S_mean, _S_se = _predict_survival_grid(
        fit,
        X_new,
        t_grid,
        with_uncertainty=False,
    )
    return S_mean


def _predict_survival_surfaces(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    S_mean, S_se = _predict_survival_grid(
        fit,
        X_new,
        t_grid,
        with_uncertainty=True,
    )
    if S_se is None:
        raise RuntimeError("gamfit did not return survival_se for survival prediction")
    return S_mean, S_se


@dataclass
class SurvivalGAM:
    """Fitted gamfit Gompertz-Makeham GAMLSS survival model."""

    columns: Tuple[str, ...]
    diagnostics: Dict[str, Any]
    _fit: Optional[_SubmodelFit] = field(repr=False, default=None)
    _n_uncertainty_slices: int = 200
    _mean_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _uncertainty_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )

    def _prediction_cache_key(self, X_new: np.ndarray, t_grid: np.ndarray) -> int:
        return hash((X_new.tobytes(), X_new.shape, t_grid.tobytes(), t_grid.shape))

    def _shape_X_new(self, X_new: Any) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        p = len(self.columns)
        if p > 0:
            return X_new.reshape(-1, p)
        if X_new.ndim == 2:
            if X_new.shape[1] != 0:
                raise ValueError(
                    f"zero-parent survival model expected X_new with 0 columns, "
                    f"got shape {X_new.shape}"
                )
            return X_new.reshape(X_new.shape[0], 0)
        return np.zeros((X_new.reshape(-1).shape[0], 0), dtype=float)

    def _cached_mean(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        if self._fit is None:
            raise RuntimeError("SurvivalGAM has no fit attached")
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = self._prediction_cache_key(X_new, t_grid)
        cached = self._mean_cache.get(key)
        if cached is not None:
            return cached
        mean = _predict_survival_matrix(self._fit, X_new, t_grid)
        self._mean_cache[key] = mean
        return mean

    def _cached_predict(
        self,
        X_new: np.ndarray,
        t_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit is None:
            raise RuntimeError("SurvivalGAM has no fit attached")
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = self._prediction_cache_key(X_new, t_grid)
        cached = self._uncertainty_cache.get(key)
        if cached is not None:
            return cached
        surfaces = _predict_survival_surfaces(self._fit, X_new, t_grid)
        self._uncertainty_cache[key] = surfaces
        self._mean_cache[key] = surfaces[0]
        return surfaces

    def predict_survival(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        n_slices = max(int(self._n_uncertainty_slices), 1)
        S_mean, S_se = self._cached_predict(X_new, t_grid)
        if n_slices == 1:
            return S_mean[None, :, :].copy()
        probs = (np.arange(n_slices, dtype=float) + 0.5) / float(n_slices)
        z = ndtri(probs)
        S = S_mean[None, :, :] + z[:, None, None] * S_se[None, :, :]
        return np.minimum.accumulate(np.clip(S, 0.0, 1.0), axis=2)

    def predict_survival_mean(
        self, X_new: np.ndarray, t_grid: np.ndarray
    ) -> np.ndarray:
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        return self._cached_mean(X_new, t_grid).copy()

    def predict_survival_se(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        _S_mean, S_se = self._cached_predict(X_new, t_grid)
        return S_se.copy()

    def predict_survival_variance(
        self,
        X_new: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        S_se = self.predict_survival_se(X_new, t_grid)
        return S_se * S_se

    def predict_median_survival(self, X_new: np.ndarray) -> np.ndarray:
        X_new = self._shape_X_new(X_new)
        t_grid = np.geomspace(1e-3, 1.0e2, 400)
        S_slices = self.predict_survival(X_new, t_grid)
        out = np.full(S_slices.shape[:2], float("inf"))
        for s in range(S_slices.shape[0]):
            for k in range(S_slices.shape[1]):
                curve = S_slices[s, k]
                below = np.where(curve <= 0.5)[0]
                if below.size == 0:
                    out[s, k] = t_grid[-1]
                elif below[0] == 0:
                    out[s, k] = t_grid[0]
                else:
                    j = int(below[0])
                    s_lo, s_hi = curve[j - 1], curve[j]
                    t_lo_g, t_hi_g = t_grid[j - 1], t_grid[j]
                    if s_hi == s_lo:
                        out[s, k] = t_hi_g
                    else:
                        frac = (s_lo - 0.5) / (s_lo - s_hi)
                        out[s, k] = t_lo_g + frac * (t_hi_g - t_lo_g)
        return out

    def predict_hazard(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        S = self.predict_survival(X_new, t_grid)
        t = np.asarray(t_grid, dtype=float).ravel()
        if t.size == 0:
            return S
        if t.size == 1:
            return np.zeros_like(S)
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("hazard prediction grid must be strictly increasing")
        dt = np.diff(t)
        eps = 1e-12
        logS = np.log(np.clip(S, eps, 1.0))
        dlog = -np.diff(logS, axis=2) / dt[None, None, :]
        pad = dlog[:, :, -1:]
        return np.concatenate([dlog, pad], axis=2)

    def uncertainty_summary(self) -> dict:
        d = dict(self.diagnostics)
        d["n_uncertainty_slices"] = int(self._n_uncertainty_slices)
        d["uncertainty_mode"] = _UNCERTAINTY_MODE
        d["uncertainty_source"] = _UNCERTAINTY_SOURCE
        if self._fit is not None:
            d["survival_likelihood"] = self._fit.survival_likelihood
            d["baseline_target"] = self._fit.baseline_target
            d["noise_formula"] = self._fit.noise_formula
        d["columns"] = list(self.columns)
        return d


def fit_survival_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Optional[Tuple[str, ...]] = None,
    n_uncertainty_slices: int = 1000,
    progress: bool | ProgressCallback = False,
    location_formula: Optional[str] = None,
    noise_formula: Optional[str] = None,
) -> SurvivalGAM:
    """Fit a right-censored Gompertz-Makeham GAMLSS survival model via gamfit."""

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        X = X.reshape(time.shape[0], 0)
    else:
        X = X.reshape(time.shape[0], -1)
    if columns is None:
        columns = tuple(f"x{i}" for i in range(X.shape[1]))
    columns = tuple(columns)

    fit = _fit_gam(
        time,
        event,
        X,
        columns,
        progress=progress,
        location_formula=location_formula,
        noise_formula=noise_formula,
    )
    summ = fit.train_summary or {}

    def _summary_float(key: str, default: float = float("nan")) -> float:
        value = summ.get(key)
        if value is None:
            return default
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"gamfit summary field {key!r} must be numeric, got bool")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raise TypeError(
            f"gamfit summary field {key!r} must be numeric or None, "
            f"got {type(value).__name__}"
        )

    def _summary_int(*keys: str, default: int = 0) -> int:
        for key in keys:
            value = summ.get(key)
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"gamfit summary field {key!r} must be an integer, got bool"
                )
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (float, np.floating)) and float(value).is_integer():
                return int(value)
            raise TypeError(
                f"gamfit summary field {key!r} must be an integer or None, "
                f"got {type(value).__name__}"
            )
        return default

    diagnostics = {
        "backend": "gamfit",
        "model_family": _SURVIVAL_MODEL_FAMILY,
        "library_version": gam.build_info().get("version"),
        "formula": fit.formula,
        "location_formula": fit.location_formula,
        "noise_formula": fit.noise_formula,
        "covariate_center": fit.x_center.tolist(),
        "covariate_scale": fit.x_scale.tolist(),
        "train_summary": summ,
        "converged": True,
        "reml_iterations": _summary_int("reml_iterations", "iterations"),
        "reml_score": _summary_float("reml_score"),
        "edf_total": _summary_float("edf_total"),
        "sigma_residual": _summary_float("sigma_residual"),
        "deviance": _summary_float("deviance"),
        "n_train": fit.n_train,
        "n_events": fit.n_events,
        "survival_likelihood": fit.survival_likelihood,
        "baseline_target": fit.baseline_target,
        "n_uncertainty_slices": int(n_uncertainty_slices),
        "uncertainty_mode": _UNCERTAINTY_MODE,
        "uncertainty_source": _UNCERTAINTY_SOURCE,
    }

    return SurvivalGAM(
        columns=columns,
        diagnostics=diagnostics,
        _fit=fit,
        _n_uncertainty_slices=int(n_uncertainty_slices),
    )


def bma_survival(
    parent_sets: List[Tuple[int, ...]],
    weights: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    data_matrix: np.ndarray,
    column_names: Tuple[str, ...],
    t_grid: np.ndarray,
    X_eval: Optional[np.ndarray] = None,
    **gam_kwargs,
) -> dict:
    """Bayesian model averaging over candidate parent sets."""

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != len(parent_sets):
        raise ValueError("weights must match len(parent_sets)")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("weights sum to zero")
    weights = weights / weights.sum()

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    data_matrix = np.asarray(data_matrix, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    X_eval_arr = data_matrix if X_eval is None else np.asarray(X_eval, dtype=float)
    if X_eval_arr.ndim != 2:
        raise ValueError("X_eval must be 2-D (n_eval, p)")
    n_eval = X_eval_arr.shape[0]

    per_set_mean: List[np.ndarray] = []
    per_set_var: List[np.ndarray] = []
    fits: List[SurvivalGAM] = []
    for cols in parent_sets:
        cols = tuple(int(c) for c in cols)
        sub_X = data_matrix[:, list(cols)] if cols else np.zeros((time.shape[0], 0))
        names = tuple(column_names[c] for c in cols)
        fit = fit_survival_gam(time, event, sub_X, columns=names, **gam_kwargs)
        fits.append(fit)
        sub_eval = X_eval_arr[:, list(cols)] if cols else np.zeros((n_eval, 0))
        per_set_mean.append(fit.predict_survival_mean(sub_eval, t_grid))
        per_set_var.append(fit.predict_survival_variance(sub_eval, t_grid))

    stack_mean = np.stack(per_set_mean, axis=0)
    stack_var = np.stack(per_set_var, axis=0)
    survival_mean = np.einsum("k,kij->ij", weights, stack_mean)
    variance_parametric = np.einsum("k,kij->ij", weights, stack_var)
    variance_structural = np.einsum(
        "k,kij->ij",
        weights,
        (stack_mean - survival_mean[None, :, :]) ** 2,
    )
    variance_total = variance_parametric + variance_structural
    return {
        "survival_mean": survival_mean,
        "per_set_mean": stack_mean,
        "per_set_variance": stack_var,
        "variance_parametric": variance_parametric,
        "variance_structural": variance_structural,
        "variance_total": variance_total,
        "weights": weights,
        "fits": fits,
        "t_grid": t_grid,
    }


__all__ = ["SurvivalGAM", "fit_survival_gam", "bma_survival"]
