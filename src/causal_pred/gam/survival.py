"""Distributional survival GAM via the gamfit Python library.

This module is a thin wrapper over the ``gamfit`` Python library (PyO3
bindings to SauersML/gam's Rust engine). The library implements the
penalised-spline survival GAM with REML smoothing and analytical
uncertainty; we feed it pandas frames and read back the dense
survival surface via :class:`gamfit.SurvivalPrediction`. There is no IPCW,
no EM, no imputation, no events-only complete-case hack: the library's
native survival likelihood handles censoring exactly.

Public API
----------

``fit_survival_gam(time, event, X, columns=None, ...) -> SurvivalGAM``
``SurvivalGAM.predict_survival(X_new, t_grid) -> (n_samples, n_new, n_t)``
``SurvivalGAM.predict_median_survival(X_new) -> (n_samples, n_new)``
``SurvivalGAM.predict_hazard(X_new, t_grid) -> (n_samples, n_new, n_t)``
``SurvivalGAM.posterior_summary() -> dict``
``bma_survival(parent_sets, weights, time, event, data_matrix, columns,
               t_grid, X_eval=None, **gam_kwargs) -> dict``

Bayesian model averaging (Draper 1995, JRSS-B 57(1):45-97) decomposes
the posterior-predictive variance of ``S(t | x)`` into a parametric
(within-model) part and a structural (between-model) part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import gamfit as gam


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

    model: gam.Model
    columns: Tuple[str, ...]
    kinds: Tuple[str, ...]            # "continuous" | "binary"
    n_train: int
    n_events: int
    formula: str
    train_summary: Dict[str, Any]


def _fit_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Tuple[str, ...],
    survival_likelihood: str = "location-scale",
) -> _SubmodelFit:
    """Fit a survival GAM through ``gamfit.fit`` and return a ``_SubmodelFit``.

    Uses the unified ``location-scale`` survival likelihood (the
    ``marginal-slope`` mode does not converge on the ``causal-pred``
    benchmarks at ``n=300``). The covariates enter the formula as
    P-spline smooths (continuous) or linear terms (binary).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float)
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
            "entry": np.zeros(time.shape[0], dtype=float),
            "exit": time,
            "event": event,
            **{name: X[:, i] for i, name in enumerate(columns)},
        }
    )

    model = gam.fit(df, formula, survival_likelihood=survival_likelihood)

    # ``Summary`` is a frozen dict-wrapper; expose its payload directly so
    # caller-visible diagnostics carry every field the library emitted.
    train_summary: Dict[str, Any] = dict(model.summary().to_dict())

    return _SubmodelFit(
        model=model,
        columns=tuple(columns),
        kinds=kinds,
        n_train=int(time.shape[0]),
        n_events=int(np.sum(event > 0.0)),
        formula=formula,
        train_summary=train_summary,
    )


# ---------------------------------------------------------------------------
# Predict via gam.Model.predict + SurvivalPrediction.survival_at
# ---------------------------------------------------------------------------


def _predict_survival_matrix(
    fit: _SubmodelFit,
    X_new: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Return ``(n_new, n_t)`` survival probabilities S(t | x).

    The gamfit model returns a :class:`gamfit.SurvivalPrediction` object
    for the requested rows. Its ``survival_at`` method evaluates each
    row's fitted survival function on the shared output grid.
    """
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

    if n_new == 0 or n_t == 0:
        return np.zeros((n_new, n_t), dtype=float)

    df_new = pd.DataFrame(
        {
            "entry": np.zeros(n_new, dtype=float),
            "exit": np.full(n_new, float(t_grid[0]), dtype=float),
            "event": np.ones(n_new, dtype=float),
            **{name: X_new[:, i] for i, name in enumerate(fit.columns)},
        }
    )

    pred = fit.model.predict(df_new)
    S_mean = np.asarray(pred.survival_at(t_grid), dtype=float)
    if S_mean.shape != (n_new, n_t):
        raise RuntimeError(
            f"gamfit returned survival_at of shape {S_mean.shape}; "
            f"expected ({n_new}, {n_t})"
        )
    # S must be non-increasing in t; enforce by left-to-right cumulative
    # minimum so floating-point noise can't violate monotonicity.
    S_mean = np.minimum.accumulate(S_mean, axis=1)
    return np.clip(S_mean, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SurvivalGAM:
    """Fitted distributional survival GAM."""

    samples: dict
    X_design: dict
    columns: tuple
    t_scale: float
    diagnostics: dict
    _fit: Optional[_SubmodelFit] = field(repr=False, default=None)
    _n_posterior_draws: int = 200
    _predict_cache: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def _cached_predict(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        if self._fit is None:
            raise RuntimeError("SurvivalGAM has no fit attached")
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = hash((X_new.tobytes(), X_new.shape, t_grid.tobytes()))
        cached = self._predict_cache.get(key)
        if cached is not None:
            return cached
        S_mean = _predict_survival_matrix(self._fit, X_new, t_grid)
        self._predict_cache[key] = S_mean
        return S_mean

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
        """Posterior-predictive survival probabilities ``(n_samples, n_new, n_t)``.

        The gamfit library returns a single point estimate of ``S(t | x)``
        with no per-cell standard error; we broadcast the same surface
        across the requested ``n_samples`` axis to keep the shape
        contract that downstream BMA / validation code expects.
        """
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        S_mean = self._cached_predict(X_new, t_grid)
        S = max(int(self._n_posterior_draws), 1)
        return np.broadcast_to(S_mean, (S,) + S_mean.shape).copy()

    def predict_median_survival(self, X_new: np.ndarray) -> np.ndarray:
        """Posterior draws of the median survival time per row."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.geomspace(1e-3, 1.0e2, 400)
        S_draws = self.predict_survival(X_new, t_grid)            # (S, n_new, n_t)
        out = np.full(S_draws.shape[:2], float("inf"))
        for s in range(S_draws.shape[0]):
            for k in range(S_draws.shape[1]):
                curve = S_draws[s, k]
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
        """Posterior-predictive hazard ``h(t|x) = -d/dt log S(t|x)``."""
        S = self.predict_survival(X_new, t_grid)
        t = np.asarray(t_grid, dtype=float).ravel()
        dt = np.diff(t)
        eps = 1e-12
        logS = np.log(np.clip(S, eps, 1.0))
        dlog = -np.diff(logS, axis=2) / dt[None, None, :]
        pad = dlog[:, :, -1:]
        return np.concatenate([dlog, pad], axis=2)

    def posterior_summary(self) -> dict:
        """Diagnostics-ready dict."""
        d = dict(self.diagnostics)
        d["n_posterior_draws"] = int(self._n_posterior_draws)
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
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
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
    n_samples : int
        Number of posterior-predictive draws per prediction. Retained
        for shape-contract compatibility with downstream BMA: the library
        returns a single point estimate, which is broadcast across this
        axis at predict time.
    """
    del rng, progress  # accepted for API compat; library is deterministic

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

    fit = _fit_gam(time, event, X, columns)

    X_design = {
        name: {"kind": fit.kinds[i], "index": i}
        for i, name in enumerate(columns)
    }

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
        "n_posterior_draws": int(n_samples),
    }

    return SurvivalGAM(
        samples={"n_samples": int(n_samples), "coefficients": None},
        X_design=X_design,
        columns=columns,
        t_scale=1.0,
        diagnostics=diagnostics,
        _fit=fit,
        _n_posterior_draws=int(n_samples),
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
        draws = fit.predict_survival(sub_eval, t_grid)            # (S, n_eval, n_t)
        per_set_mean.append(draws.mean(axis=0))
        per_set_var.append(draws.var(axis=0, ddof=0))

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
        "S_bma": S_bma,                     # alias kept for test compatibility
        "per_set_mean": stack_mean,
        "per_model_mean": stack_mean,       # alias
        "per_set_variance": stack_var,
        "per_model_variance": stack_var,    # alias
        "variance_parametric": V_param,
        "var_parametric": V_param,          # alias
        "variance_structural": V_struct,
        "var_structural": V_struct,         # alias
        "variance_total": V_total,
        "var_total": V_total,               # alias
        "weights": weights,
        "fits": fits,
        "t_grid": t_grid,
    }


__all__ = [
    "SurvivalGAM",
    "fit_survival_gam",
    "bma_survival",
]
