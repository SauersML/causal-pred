"""Distributional survival GAM via the gamfit Python library.

This module is a thin wrapper over the ``gamfit`` Python library (PyO3
bindings to SauersML/gam's Rust engine). The library implements the
penalised-spline survival GAM with REML smoothing; we feed it pandas frames
and read back the dense survival surface via
:class:`gamfit.SurvivalPrediction`. There is no IPCW, no EM, no imputation,
no events-only complete-case hack: the library's native survival likelihood
handles censoring exactly.

Public API
----------

``fit_survival_gam(time, event, X, columns=None, ...) -> SurvivalGAM``
``SurvivalGAM.predict_survival(X_new, t_grid) -> (n_slices, n_new, n_t)``
``SurvivalGAM.predict_survival_mean(X_new, t_grid) -> (n_new, n_t)``
``SurvivalGAM.predict_survival_se(X_new, t_grid) -> (n_new, n_t)``
``SurvivalGAM.predict_median_survival(X_new) -> (n_slices, n_new)``
``SurvivalGAM.predict_hazard(X_new, t_grid) -> (n_slices, n_new, n_t)``
``SurvivalGAM.uncertainty_summary() -> dict``
``bma_survival(parent_sets, weights, time, event, data_matrix, columns,
               t_grid, X_eval=None, **gam_kwargs) -> dict``

Mean survival curves and response-scale standard errors are read directly
from gamfit's survival prediction object. This wrapper always uses
gamfit's ``location-scale`` survival likelihood because that is the
survival mode with library-provided delta-method uncertainty.
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


def _progress_callback(progress: bool | ProgressCallback) -> Optional[ProgressCallback]:
    if callable(progress):
        return progress
    if progress:
        return logger.info
    return None


# ---------------------------------------------------------------------------
# Column kind detection + formula
# ---------------------------------------------------------------------------


def _is_binary_column(x: np.ndarray) -> bool:
    vals = np.unique(x)
    return vals.shape[0] <= 2 and set(vals.tolist()).issubset({0.0, 1.0})


# Univariate continuous covariates use a penalised B-spline (P-spline,
# Eilers & Marx 1996) -- ``s(x, type=ps, knots=N)`` -- which is the gam
# engine's canonical univariate smooth in its own benchmark suite.
_PSPLINE_DEFAULT_KNOTS = 10
_SURVIVAL_ENTRY_ORIGIN = 1.0
_DEFAULT_SURVIVAL_LIKELIHOOD = "weibull"
_UNCERTAINTY_SURVIVAL_LIKELIHOODS = frozenset({"location-scale"})


def _build_survival_formula(columns: Sequence[str], kinds: Sequence[str]) -> str:
    """Return ``s(x1, type=ps, knots=10) + x2 + ...``.

    Continuous columns become penalised P-splines; binary columns enter
    as plain linear terms. An empty covariate set collapses to
    intercept-only via the literal ``1`` term.
    """
    terms: List[str] = []
    for name, kind in zip(columns, kinds):
        if kind == "continuous":
            terms.append(f"s({name}, type=ps, knots={_PSPLINE_DEFAULT_KNOTS})")
        else:
            terms.append(name)
    return " + ".join(terms) if terms else "1"


# ---------------------------------------------------------------------------
# Internal fit object
# ---------------------------------------------------------------------------


@dataclass
class _SubmodelFit:
    """Fitted gam survival model artefacts."""

    model: Any
    columns: Tuple[str, ...]
    kinds: Tuple[str, ...]            # "continuous" | "binary"
    n_train: int
    n_events: int
    formula: str
    survival_likelihood: str
    train_summary: Dict[str, Any]


def _fit_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Tuple[str, ...],
    survival_likelihood: str = _DEFAULT_SURVIVAL_LIKELIHOOD,
    progress: bool | ProgressCallback = False,
) -> _SubmodelFit:
    """Fit a survival GAM through ``gamfit.fit`` and return a ``_SubmodelFit``.

    The cohort has no delayed entry. gamfit's survival schema requires a
    strictly positive entry age for prediction, so follow-up durations are
    represented as the interval ``[1, 1 + time]``. This preserves the
    observed durations while avoiding an all-zero entry column being inferred
    as binary by the backend schema. The covariates enter the formula as
    P-spline smooths (continuous) or linear terms (binary).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(time)) or np.any(time <= 0.0):
        raise ValueError("survival GAM requires finite positive follow-up times")
    if X.size == 0:
        X = X.reshape(time.shape[0], 0)
    else:
        X = X.reshape(time.shape[0], -1)
    if X.shape[1] != len(columns):
        raise ValueError(
            f"columns length {len(columns)} != X.shape[1] {X.shape[1]}"
        )

    kinds = tuple(
        "binary" if _is_binary_column(X[:, i]) else "continuous"
        for i in range(X.shape[1])
    )
    rhs = _build_survival_formula(columns, kinds)
    formula = f"Surv(entry, exit, event) ~ {rhs}"

    df = pd.DataFrame(
        {
            "entry": np.full(time.shape[0], _SURVIVAL_ENTRY_ORIGIN, dtype=float),
            "exit": _SURVIVAL_ENTRY_ORIGIN + time,
            "event": event,
            **{name: X[:, i] for i, name in enumerate(columns)},
        }
    )

    emit = _progress_callback(progress)
    if emit is not None:
        emit(
            "fit start "
            f"n={time.shape[0]} events={int(np.sum(event > 0.0))} "
            f"p={len(columns)} likelihood={survival_likelihood} formula={formula}"
        )
    model = gam.fit(df, formula, survival_likelihood=survival_likelihood)

    # ``Summary`` is a frozen dict-wrapper; expose its payload directly so
    # caller-visible diagnostics carry every field the library emitted.
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
        kinds=kinds,
        n_train=int(time.shape[0]),
        n_events=int(np.sum(event > 0.0)),
        formula=formula,
        survival_likelihood=str(survival_likelihood),
        train_summary=train_summary,
    )


# ---------------------------------------------------------------------------
# Predict via gam.Model.predict + SurvivalPrediction.survival_at_chunks
# ---------------------------------------------------------------------------


def _shape_prediction_inputs(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    p = len(fit.columns)
    if p > 0:
        X_new = np.asarray(X_new, dtype=float).reshape(-1, p)
    else:
        X_new = np.asarray(X_new, dtype=float)
        X_new = (
            X_new.reshape(X_new.shape[0], 0)
            if X_new.ndim == 2
            else X_new.reshape(0, 0)
        )
    n_new = X_new.shape[0]
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    n_t = t_grid.shape[0]
    return X_new, t_grid, n_new, n_t


def _prediction_frame(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    n_new = X_new.shape[0]
    return pd.DataFrame(
        {
            "entry": np.full(n_new, _SURVIVAL_ENTRY_ORIGIN, dtype=float),
            "exit": np.full(
                n_new,
                _SURVIVAL_ENTRY_ORIGIN + max(float(t_grid[0]), 0.0),
                dtype=float,
            ),
            "event": np.ones(n_new, dtype=float),
            **{name: X_new[:, i] for i, name in enumerate(fit.columns)},
        }
    )


def _survival_at_matrix(pred: Any, t_grid: np.ndarray, n_new: int, n_t: int) -> np.ndarray:
    try:
        S_mean = np.asarray(pred.survival_at(t_grid), dtype=float)
    except ValueError as exc:
        if "dense survival curves are limited" not in str(exc) or not hasattr(
            pred,
            "survival_at_chunks",
        ):
            raise
        S_mean = np.empty((n_new, n_t), dtype=float)
        for row_slice, time_slice, block in pred.survival_at_chunks(t_grid):
            S_mean[row_slice, time_slice] = np.asarray(block, dtype=float)
    if S_mean.shape != (n_new, n_t):
        raise RuntimeError(
            f"gamfit returned survival surface of shape {S_mean.shape}; "
            f"expected ({n_new}, {n_t})"
        )
    S_mean = np.minimum.accumulate(S_mean, axis=1)
    return np.clip(S_mean, 0.0, 1.0)


def _predict_survival_matrix(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Return gamfit point-estimate survival surface ``(n_new, n_t)``."""
    X_new, t_grid, n_new, n_t = _shape_prediction_inputs(fit, X_new, t_grid)
    if n_new == 0 or n_t == 0:
        return np.zeros((n_new, n_t), dtype=float)
    pred = fit.model.predict(_prediction_frame(fit, X_new, t_grid))
    backend_t = _SURVIVAL_ENTRY_ORIGIN + t_grid
    return _survival_at_matrix(pred, backend_t, n_new, n_t)


def _predict_survival_surfaces(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gamfit survival mean and SE surfaces, both ``(n_new, n_t)``."""
    X_new, t_grid, n_new, n_t = _shape_prediction_inputs(fit, X_new, t_grid)
    if n_new == 0 or n_t == 0:
        empty = np.zeros((n_new, n_t), dtype=float)
        return empty, empty

    pred = fit.model.predict(
        _prediction_frame(fit, X_new, t_grid),
        with_uncertainty=True,
    )
    backend_t = _SURVIVAL_ENTRY_ORIGIN + t_grid
    S_mean = _survival_at_matrix(pred, backend_t, n_new, n_t)
    S_se_raw = pred.survival_se_at(backend_t)
    if S_se_raw is None:
        raise RuntimeError("gamfit did not return survival_se for survival prediction")
    S_se = np.asarray(S_se_raw, dtype=float)
    if S_se.shape != (n_new, n_t):
        raise RuntimeError(
            f"gamfit returned survival SE surface of shape {S_se.shape}; "
            f"expected ({n_new}, {n_t})"
        )
    if not np.all(np.isfinite(S_se)):
        raise RuntimeError("gamfit returned non-finite survival standard errors")
    return S_mean, np.clip(S_se, 0.0, None)


def _gamfit_supports_survival_uncertainty(fit: _SubmodelFit) -> bool:
    return fit.survival_likelihood in _UNCERTAINTY_SURVIVAL_LIKELIHOODS


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SurvivalGAM:
    """Fitted distributional survival GAM."""

    columns: tuple
    diagnostics: dict
    _fit: Optional[_SubmodelFit] = field(repr=False, default=None)
    _n_uncertainty_slices: int = 200
    _mean_cache: Dict[int, np.ndarray] = field(
        default_factory=dict,
        repr=False,
    )

    _uncertainty_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )

    def _cached_predict(
        self,
        X_new: np.ndarray,
        t_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fit is None:
            raise RuntimeError("SurvivalGAM has no fit attached")
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = hash((X_new.tobytes(), X_new.shape, t_grid.tobytes()))
        cached = self._uncertainty_cache.get(key)
        if cached is not None:
            return cached
        surfaces = _predict_survival_surfaces(self._fit, X_new, t_grid)
        self._uncertainty_cache[key] = surfaces
        return surfaces

    def _cached_mean(
        self,
        X_new: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        if self._fit is None:
            raise RuntimeError("SurvivalGAM has no fit attached")
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = hash((X_new.tobytes(), X_new.shape, t_grid.tobytes()))
        cached = self._mean_cache.get(key)
        if cached is not None:
            return cached
        mean = _predict_survival_matrix(self._fit, X_new, t_grid)
        self._mean_cache[key] = mean
        return mean

    def _shape_X_new(self, X_new: Any) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        p = len(self.columns)
        if p == 0:
            return (
                X_new.reshape(X_new.shape[0], 0)
                if X_new.ndim == 2
                else X_new.reshape(0, 0)
            )
        return X_new.reshape(-1, p)

    def predict_survival(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Deterministic survival curve stack ``(n_slices, n_new, n_t)``."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        n_slices = max(int(self._n_uncertainty_slices), 1)
        S_mean = self._cached_mean(X_new, t_grid)
        return np.repeat(S_mean[None, :, :], n_slices, axis=0)

    def predict_survival_mean(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Point-estimate survival surface ``(n_new, n_t)`` without draw expansion."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        S_mean = self._cached_mean(X_new, t_grid)
        return S_mean.copy()

    def predict_survival_se(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Within-model survival standard error.

        The current gamfit backend path is deterministic; BMA intervals
        therefore carry structural parent-set uncertainty only.
        """
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        return np.zeros_like(self._cached_mean(X_new, t_grid))

    def predict_survival_variance(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Within-model survival variance; zero for deterministic gamfit curves."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        return np.zeros_like(self._cached_mean(X_new, t_grid))

    def predict_median_survival(self, X_new: np.ndarray) -> np.ndarray:
        """Median-survival quantile slices per row."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.geomspace(1e-3, 1.0e2, 400)
        S_slices = self.predict_survival(X_new, t_grid)           # (S, n_new, n_t)
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
        """Hazard slices ``h(t|x) = -d/dt log S(t|x)``."""
        S = self.predict_survival(X_new, t_grid)
        t = np.asarray(t_grid, dtype=float).ravel()
        dt = np.diff(t)
        eps = 1e-12
        logS = np.log(np.clip(S, eps, 1.0))
        dlog = -np.diff(logS, axis=2) / dt[None, None, :]
        pad = dlog[:, :, -1:]
        return np.concatenate([dlog, pad], axis=2)

    def uncertainty_summary(self) -> dict:
        """Diagnostics-ready dict for the fitted gamfit backend."""
        d = dict(self.diagnostics)
        d["n_uncertainty_slices"] = int(self._n_uncertainty_slices)
        d["uncertainty_mode"] = "structural_only_point_estimate"
        d["uncertainty_source"] = "parent_set_bma"
        if self._fit is not None:
            d["survival_likelihood"] = self._fit.survival_likelihood
        d["columns"] = list(self.columns)
        return d


# ---------------------------------------------------------------------------
# Public fit entry point
# ---------------------------------------------------------------------------


def fit_survival_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Optional[Tuple[str, ...]] = None,
    n_uncertainty_slices: int = 1000,
    progress: bool | ProgressCallback = False,
) -> SurvivalGAM:
    """Fit a distributional survival GAM via the gamfit Python library.

    Parameters
    ----------
    time, event : arrays of shape (n,)
        Observed follow-up times (positive) and event indicators
        (1 = failure, 0 = right-censored).
    X : (n, p) ndarray
        Covariate matrix. Columns whose unique values are a subset of
        ``{0, 1}`` are treated as binary and enter linearly; other
        columns enter as penalised P-splines.
    columns : tuple, optional
        Names for the ``p`` covariates. Defaults to ``("x0", ...)``.
    n_uncertainty_slices : int
        Number of deterministic slices returned by ``predict_survival`` for
        plotting-style callers.
    progress : bool or callable
        When callable, receives concise gamfit fit/summary messages. When
        True, messages are sent to this module's logger.
    """
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
                raise TypeError(f"gamfit summary field {key!r} must be an integer, got bool")
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (float, np.floating)) and float(value).is_integer():
                return int(value)
            raise TypeError(
                f"gamfit summary field {key!r} must be an integer or None, "
                f"got {type(value).__name__}"
            )
        return default

    gam = _load_gamfit()
    diagnostics = {
        "backend": "gamfit",
        "library_version": gam.build_info().get("version"),
        "formula": fit.formula,
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
        "n_uncertainty_slices": int(n_uncertainty_slices),
        "uncertainty_mode": "structural_only_point_estimate",
        "uncertainty_source": "parent_set_bma",
    }

    return SurvivalGAM(
        columns=columns,
        diagnostics=diagnostics,
        _fit=fit,
        _n_uncertainty_slices=int(n_uncertainty_slices),
    )


# ---------------------------------------------------------------------------
# Bayesian model averaging (Draper 1995)
# ---------------------------------------------------------------------------


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
    """Bayesian model averaging over candidate parent sets (Draper 1995)."""
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

    X_eval_arr = (
        data_matrix if X_eval is None else np.asarray(X_eval, dtype=float)
    )
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

    stack_mean = np.stack(per_set_mean, axis=0)                   # (K, n_eval, n_t)
    stack_var = np.stack(per_set_var, axis=0)

    S_bma = np.einsum("k,kij->ij", weights, stack_mean)
    V_param = np.einsum("k,kij->ij", weights, stack_var)
    V_struct = np.einsum(
        "k,kij->ij", weights, (stack_mean - S_bma[None, :, :]) ** 2,
    )
    V_total = V_param + V_struct
    return {
        "survival_mean": S_bma,
        "per_set_mean": stack_mean,
        "per_set_variance": stack_var,
        "variance_parametric": V_param,
        "variance_structural": V_struct,
        "variance_total": V_total,
        "weights": weights,
        "fits": fits,
        "t_grid": t_grid,
    }


__all__ = [
    "SurvivalGAM",
    "fit_survival_gam",
    "bma_survival",
]
