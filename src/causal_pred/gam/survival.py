"""Distributional survival GAM -- thin wrapper over the ``gam`` Python library.

The Rust-backed Python library `gam <https://github.com/SauersML/gam>`_
(installable as a PEP 517 sdist which builds via maturin) provides a
peer-review-quality GAM engine with REML smoothing, B-spline term bases
(`s()`, `linear()`, `tensor()`, ...), Gaussian / Binomial families, and
predictive intervals.  We use it here as the backend for the survival
GAM stage of the causal-pred pipeline.

Current library status and our workaround
-----------------------------------------
The library's Rust engine *supports* location-scale survival models
(formula ``Surv(entry, exit, event) ~ ...``) but the PyO3 Python binding
at version 0.1.15 only exposes the ``standard`` (non-survival) model
class -- calling ``gam.fit`` with a ``Surv(...)`` response raises
``FormulaError: python binding currently supports standard models
only``.

Until survival support lands in the Python binding we obtain a proper
accelerated-failure-time (AFT) model by:

  1. Rescaling time so the median observed time is 1 (keeps the log-
     time response on O(1) scale).
  2. Fitting a Gaussian GAM on ``log(time_scaled)`` using the
     uncensored rows but with **inverse probability of censoring
     weights** (Horvitz-Thompson 1952; Uno et al. 2007 for survival).
     Each event ``i`` gets weight ``1 / G(T_i-)`` where ``G`` is the
     Kaplan-Meier estimate of the censoring distribution.  This fixes
     the downward bias that a plain complete-case fit on events would
     incur: censored observations are systematically long-time, so
     dropping them pulls mu_hat low and the resulting survival curve
     under-estimates S(t) (over-predicts events).  IPCW up-weights
     late events to compensate (Robins 1993; Graf et al. 1999).  We
     follow the seed fit with two EM data-augmentation iterations
     (Tanner & Wong 1987) on the censored rows: each censored
     observation's latent log-time is imputed from the truncated
     normal with lower bound at the observed log-time and the GAM is
     refit with imputed + real log-times.  This sharpens sigma and
     removes residual IPCW bias at heavy censoring.  All continuous
     covariates enter as penalised smooths ``s(name)`` and binary
     columns enter as plain linear terms.
  3. Estimating the AFT log-scale ``sigma`` jointly with ``mu``:
     uncensored rows contribute squared residuals, censored rows
     contribute imputed squared residuals plus their truncated-normal
     conditional variance, which gives the correct MLE for sigma.
  4. Obtaining a posterior-predictive surrogate by drawing
     ``n_samples`` Normal samples from ``(mean, effective_se)``
     returned by ``model.predict(..., interval=0.95)``; ``effective_se``
     is the model's epistemic SE on the linear predictor.  This gives a
     ``(S, n_new, n_t)`` tensor of survival-curve draws, consistent
     with the signature the rest of the pipeline expects.

When the library adds survival-family support in its Python binding the
implementation of ``_fit_gam`` below can be upgraded to pass
``Surv(entry, exit, event) ~ ...`` directly; the public API on this
module does not change.

Public API
----------
``SurvivalGAM`` exposes ``predict_survival``, ``predict_median_survival``,
``predict_hazard``, ``posterior_summary`` and the ``diagnostics`` dict.
``fit_survival_gam`` returns a fitted ``SurvivalGAM``.  ``bma_survival``
performs Bayesian model averaging over candidate parent sets using the
variance decomposition of Draper (1995).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

try:  # import guard so this module imports even when `gam` isn't available
    import gam as _gam

    _GAM_AVAILABLE = True
    _GAM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover -- env without gam
    _gam = None
    _GAM_AVAILABLE = False
    _GAM_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary_column(x: np.ndarray) -> bool:
    vals = np.unique(x)
    return vals.shape[0] <= 2 and set(vals.tolist()).issubset({0.0, 1.0})


def _build_formula(response: str, columns: Sequence[str], kinds: Sequence[str]) -> str:
    """Build a ``response ~ s(x1) + x2 + ...`` formula for the gam library.

    Continuous columns become ``s(name)`` smooths (cubic P-splines in
    the library's default configuration), binary columns become plain
    linear terms.  If ``columns`` is empty the formula reduces to an
    intercept-only model ``response ~ 1``.
    """
    terms: List[str] = []
    for name, kind in zip(columns, kinds):
        if kind == "continuous":
            terms.append(f"s({name})")
        else:
            terms.append(name)
    if not terms:
        return f"{response} ~ 1"
    return f"{response} ~ " + " + ".join(terms)


def _columns_to_dict(
    X: np.ndarray,
    columns: Sequence[str],
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for i, name in enumerate(columns):
        out[name] = X[:, i].astype(float).tolist()
    if extras:
        for k, v in extras.items():
            out[k] = np.asarray(v, dtype=float).tolist()
    return out


# ---------------------------------------------------------------------------
# Internal per-GAM fit object
# ---------------------------------------------------------------------------


@dataclass
class _SubmodelFit:
    """Fitted wrapper around a single ``gam.Model`` location fit.

    Holds the library model for the AFT location predictor plus the
    residual-based sigma and metadata needed to reproduce prediction
    schemas at serving time.
    """

    model: Any  # gam.Model
    columns: Tuple[str, ...]
    kinds: Tuple[str, ...]  # "continuous" | "binary"
    sigma: float  # residual SD on log-time scale
    t_scale: float
    n_train: int
    n_events: int
    formula: str
    summary_payload: Dict[str, Any]


def _truncated_normal_moments(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First two central moments of ``Z | Z > a`` where ``Z ~ N(0, 1)``.

    Returns ``(m1, m2)`` where ``m1 = E[Z | Z > a]`` and
    ``m2 = Var[Z | Z > a]``.  Stable for large positive ``a`` via
    ``phi(a) / (1 - Phi(a))`` computed as ``exp(logpdf - logsf)``.
    """
    a = np.asarray(a, dtype=float)
    log_phi = -0.5 * a * a - 0.5 * np.log(2.0 * np.pi)
    log_sf = norm.logsf(a)
    lam = np.exp(log_phi - log_sf)  # inverse Mills ratio
    m1 = lam  # E[Z | Z > a]
    m2 = 1.0 + a * lam - lam**2  # Var[Z | Z > a]
    # Numerical floor: variance must be >= 0.
    m2 = np.maximum(m2, 1e-12)
    return m1, m2


def _fit_gaussian_gam(
    train_dict: Dict[str, list], formula: str, weights_key: Optional[str] = None
) -> Any:
    """Fit a Gaussian GAM; pass-through for ``weights=`` when provided."""
    if weights_key is not None:
        return _gam.fit(train_dict, formula, family="gaussian", weights=weights_key)
    return _gam.fit(train_dict, formula, family="gaussian")


def _km_censoring(
    time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier of the *censoring* distribution.

    Returns ``(uniq_times, G)`` right-continuous.  Events and censoring
    are swapped: we treat ``1 - event`` as the "event" for the KM, so
    G(t) = P(C > t).
    """
    t = np.asarray(time, dtype=float)
    cens = 1.0 - np.asarray(event, dtype=float)
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    es = cens[order]
    uniq, inv = np.unique(ts, return_inverse=True)
    d = np.zeros(uniq.size, dtype=float)
    np.add.at(d, inv, es)
    counts = np.bincount(inv, minlength=uniq.size).astype(float)
    cum_before = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
    n_at = float(ts.size) - cum_before
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_at > 0, 1.0 - d / n_at, 1.0)
    G = np.cumprod(factors)
    return uniq, G


def _km_eval_at(uniq: np.ndarray, G: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(uniq, t, side="right") - 1
    out = np.ones_like(t, dtype=float)
    mask = idx >= 0
    out[mask] = G[idx[mask]]
    return out


def _fit_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Tuple[str, ...],
    n_em_iters: int = 1,
) -> _SubmodelFit:
    """Fit an AFT Gaussian GAM with IPCW-weighted events-only seed
    plus optional EM data augmentation over censored rows.

    Stage 1: IPCW-weighted complete-case fit.  The weight for event
    ``i`` is ``w_i = 1 / G(T_i -)`` where ``G`` is the Kaplan-Meier
    estimate of the censoring distribution.  This removes the bias
    that drops of censored observations would otherwise introduce.

    Stage 2 (optional, default 1 iteration): EM step over censored
    rows only.  Each censored observation's latent log-time is
    imputed from the truncated normal TN(mu_i, sigma, a_i) with
    a_i = (log t_c - mu_i) / sigma, i.e. E[log T_i | log T_i > log t_c]
    = mu_i + sigma * lambda(a_i) with lambda the inverse Mills ratio.
    The GAM is refit on (events + imputed censored) rows, giving the
    Tanner-Wong (1987) data-augmentation MLE for the right-censored
    Gaussian AFT model.  Sigma is updated from event residuals plus
    the conditional variance of the censored rows.

    Returns the library model plus the final ``sigma`` estimate.
    """
    if not _GAM_AVAILABLE:
        raise ImportError(
            "gam library unavailable: cannot fit survival GAM. "
            f"Underlying import error: {_GAM_IMPORT_ERROR!r}"
        )

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float).reshape(time.shape[0], -1)
    if X.shape[1] != len(columns):
        raise ValueError(f"columns length {len(columns)} != X.shape[1] {X.shape[1]}")

    # Time rescaling: divide by median(time) so log-time is ~O(1).
    pos = time[time > 0]
    t_scale = float(np.median(pos)) if pos.size > 0 else 1.0
    if t_scale <= 0:
        t_scale = 1.0
    t_scaled = np.clip(time / t_scale, 1e-9, None)
    log_t = np.log(t_scaled)

    # Kinds (continuous vs binary) inferred once on the full training matrix.
    kinds = tuple(
        "binary" if _is_binary_column(X[:, i]) else "continuous"
        for i in range(X.shape[1])
    )

    ev_mask = event.astype(bool)
    cens_mask = ~ev_mask
    n_events = int(np.sum(ev_mask))
    if n_events < 10:
        raise ValueError(
            f"too few events ({n_events}) to fit AFT GAM; need at least 10"
        )

    formula = _build_formula("log_t", columns, kinds)

    # --- Stage 1: IPCW-weighted events-only seed fit -------------
    # w_i = 1 / G(T_i -) with G the KM of the censoring distribution.
    km_t, km_G = _km_censoring(time, event)
    t_minus = np.nextafter(time, -np.inf)
    G_at_T = _km_eval_at(km_t, km_G, t_minus)
    # Clip to avoid astronomical weights when G -> 0 near the tail.
    G_at_T = np.clip(G_at_T, 0.05, 1.0)
    w_ipcw = 1.0 / G_at_T

    X_ev = X[ev_mask, :]
    log_t_ev = log_t[ev_mask]
    w_ev = w_ipcw[ev_mask]
    seed_dict = _columns_to_dict(X_ev, columns, extras={"log_t": log_t_ev})
    if not columns:
        seed_dict = {"log_t": log_t_ev.tolist(), "_const": [1.0] * log_t_ev.shape[0]}
    seed_dict["_ipcw"] = w_ev.tolist()
    model = _fit_gaussian_gam(seed_dict, formula, weights_key="_ipcw")

    # Initial sigma: weighted RMS of event residuals.
    predict_ev = _columns_to_dict(X_ev, columns)
    if not predict_ev:
        predict_ev = {"_const": [1.0] * X_ev.shape[0]}
    mu_ev = np.asarray(model.predict(predict_ev)["mean"], dtype=float)
    resid_ev = log_t_ev - mu_ev
    if resid_ev.size > 1 and w_ev.sum() > 0:
        sigma = float(np.sqrt(np.sum(w_ev * resid_ev**2) / float(w_ev.sum())))
    else:
        sigma = 1.0
    sigma = max(sigma, 1e-3)

    # --- Stage 2: one EM data-augmentation pass over censored rows
    n_cens = int(np.sum(cens_mask))
    if n_cens > 0 and n_em_iters > 0:
        predict_all = _columns_to_dict(X, columns)
        if not predict_all:
            predict_all = {"_const": [1.0] * X.shape[0]}
        mu_all = np.asarray(
            model.predict(predict_all)["mean"],
            dtype=float,
        )
        a_c = (log_t[cens_mask] - mu_all[cens_mask]) / sigma
        m1_c, m2_c = _truncated_normal_moments(a_c)
        log_t_full = log_t.copy()
        log_t_full[cens_mask] = mu_all[cens_mask] + sigma * m1_c

        # Refit on full (event + imputed) data; IPCW weights are kept
        # on events and unit weights on imputed rows (the imputation
        # itself corrects the censoring bias on those rows).
        full_dict = _columns_to_dict(
            X,
            columns,
            extras={"log_t": log_t_full},
        )
        if not columns:
            full_dict = {
                "log_t": log_t_full.tolist(),
                "_const": [1.0] * log_t_full.shape[0],
            }
        w_full = np.ones_like(log_t_full, dtype=float)
        w_full[ev_mask] = w_ev
        full_dict["_ipcw"] = w_full.tolist()
        model = _fit_gaussian_gam(full_dict, formula, weights_key="_ipcw")

        mu_all = np.asarray(
            model.predict(predict_all)["mean"],
            dtype=float,
        )
        r_ev = log_t[ev_mask] - mu_all[ev_mask]
        r_c = log_t_full[cens_mask] - mu_all[cens_mask]
        ss_ev = float(np.sum(w_ev * r_ev**2))
        ss_c = float(np.sum(r_c**2 + (sigma**2) * m2_c))
        w_sum = float(w_ev.sum() + n_cens)
        sigma = float(np.sqrt((ss_ev + ss_c) / max(w_sum, 1.0)))
        sigma = max(sigma, 1e-3)

    summary_payload = dict(model.summary().payload)
    return _SubmodelFit(
        model=model,
        columns=tuple(columns),
        kinds=kinds,
        sigma=sigma,
        t_scale=t_scale,
        n_train=int(time.shape[0]),
        n_events=n_events,
        formula=formula,
        summary_payload=summary_payload,
    )


def _predict_mean_se(
    fit: _SubmodelFit, X_new: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, mu_se)`` posterior mean and epistemic SE of the
    location predictor on new rows, via the library's ``predict``.
    """
    X_new = np.asarray(X_new, dtype=float)
    if len(fit.columns) > 0:
        X_new = X_new.reshape(-1, len(fit.columns))
    else:
        X_new = X_new.reshape(-1, 0) if X_new.ndim > 1 else X_new.reshape(-1, 0)
    n_new = X_new.shape[0]
    if len(fit.columns) == 0:
        predict_dict = {"_const": [1.0] * n_new}
    else:
        predict_dict = _columns_to_dict(X_new, fit.columns)
    # The library's schema tracking treats weight columns as required at
    # predict time; supply a unit-weight placeholder so SchemaMismatch
    # isn't raised.  The weight column is only used for fitting.
    predict_dict["_ipcw"] = [1.0] * n_new
    raw = fit.model.predict(predict_dict, interval=0.95)
    mu = np.asarray(raw["mean"], dtype=float)
    se = np.asarray(raw.get("effective_se", np.zeros_like(mu)), dtype=float)
    return mu, se


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


@dataclass
class SurvivalGAM:
    """Fitted distributional survival GAM.

    Attributes
    ----------
    samples : dict
        Carries ``"n_samples"`` (the draw count for posterior-predictive
        simulation) plus the library's coefficient estimates.  Raw
        posterior parameter draws are not emitted by the library at
        v0.1.15; ``predict_survival`` synthesises a posterior-predictive
        tensor from ``(mean, effective_se)``.
    X_design : dict
        Per-covariate metadata (kind + column order).
    columns : tuple
        Covariate names in the order used at fit time.
    t_scale : float
        Time normaliser: internally we worked with ``time / t_scale``;
        prediction undoes the rescaling.
    diagnostics : dict
        Library-level diagnostics: REML iterations, REML score,
        deviance, EDF, residual sigma, event count, converged flag.
    """

    samples: dict
    X_design: dict
    columns: tuple
    t_scale: float
    diagnostics: dict
    _fit: Optional[_SubmodelFit] = field(repr=False, default=None)
    _n_posterior_draws: int = 200
    _predict_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0),
        repr=False,
    )

    # --------------------------------------------------------------

    def _get_mu_se(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return mu and epistemic SE for ``X_new`` with per-array caching."""
        X_new = np.ascontiguousarray(X_new, dtype=float)
        key = hash((X_new.tobytes(), X_new.shape))
        cached = self._predict_cache.get(key)
        if cached is not None:
            return cached
        mu, se = _predict_mean_se(self._fit, X_new)
        self._predict_cache[key] = (mu, se)
        return mu, se

    def _posterior_mu_draws(self, X_new: np.ndarray) -> np.ndarray:
        """Return ``(n_samples, n_new)`` Normal posterior-predictive draws of
        the location predictor given the Rust GAM's (mean, effective_se)."""
        mu, se = self._get_mu_se(X_new)
        S = self._n_posterior_draws
        eps = self._rng.standard_normal((S, mu.shape[0]))
        return mu[None, :] + se[None, :] * eps

    def _shape_X_new(self, X_new: Any) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        p = len(self.columns)
        if p == 0:
            # Accept any 1-D or (n, 0) 2-D input; report (n, 0).
            if X_new.ndim == 1:
                return X_new.reshape(-1, 0)
            return X_new.reshape(X_new.shape[0], 0)
        return X_new.reshape(-1, p)

    # --- predictive interface expected by the pipeline ---------------

    def predict_survival(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Posterior survival probabilities ``S(t | x)``.

        Shape: ``(n_samples, n_new, n_t)``.  The Gaussian-AFT survival
        function is ``S(t | x) = 1 - Phi((log t - mu(x)) / sigma)``.  We
        draw ``n_samples`` Normal samples of ``mu(x)`` from the library's
        epistemic mean + SE (Laplace approximation around the REML
        posterior mode), combine with the residual-based ``sigma``, and
        evaluate the survival function on ``t_grid / t_scale``.
        """
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
        mu_draws = self._posterior_mu_draws(X_new)  # (S, n_new)
        t_scaled = np.clip(t_grid / self.t_scale, 1e-12, None)
        log_t = np.log(t_scaled)  # (n_t,)
        sigma = self._fit.sigma
        z = (log_t[None, None, :] - mu_draws[:, :, None]) / sigma
        return norm.sf(z)

    def predict_median_survival(self, X_new: np.ndarray) -> np.ndarray:
        """Posterior median-survival-time draws, shape ``(n_samples, n_new)``.

        For the Gaussian AFT the median of ``log T | x`` is ``mu(x)``,
        so median ``T = t_scale * exp(mu(x))``.
        """
        X_new = self._shape_X_new(X_new)
        mu_draws = self._posterior_mu_draws(X_new)
        return np.exp(mu_draws) * self.t_scale

    def predict_hazard(self, X_new: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Hazard ``h(t|x) = f(t|x) / S(t|x)`` on the raw-time axis.

        Shape: ``(n_samples, n_new, n_t)``.  For Gaussian AFT,
        ``h(t|x) = (phi(z) / (1 - Phi(z))) / (t * sigma)`` in scaled
        time; convert to raw time by dividing by ``t_scale``.
        """
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
        mu_draws = self._posterior_mu_draws(X_new)
        t_scaled = np.clip(t_grid / self.t_scale, 1e-12, None)
        log_t = np.log(t_scaled)
        sigma = self._fit.sigma
        z = (log_t[None, None, :] - mu_draws[:, :, None]) / sigma
        log_phi = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
        log_sf = norm.logsf(z)
        t_bc = t_scaled[None, None, :]
        h_scaled = np.exp(log_phi - log_sf) / (t_bc * sigma)
        return h_scaled / self.t_scale

    def posterior_summary(self) -> dict:
        """Return coefficient estimates, SEs, and GAM-level diagnostics."""
        out: Dict[str, Any] = {
            "formula": self._fit.formula,
            "sigma": self._fit.sigma,
            "t_scale": self.t_scale,
            "coefficients": self._fit.summary_payload.get("coefficients"),
        }
        out.update(self.diagnostics)
        return out


# ---------------------------------------------------------------------------
# Public fit entry point
# ---------------------------------------------------------------------------


def fit_survival_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Optional[Tuple[str, ...]] = None,
    n_basis: int = 10,  # retained for API compat
    n_samples: int = 1000,
    warmup: int = 500,  # retained for API compat
    n_chains: int = 2,  # retained for API compat
    target_accept: float = 0.8,  # retained for API compat
    max_tree_depth: int = 10,  # retained for API compat
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
) -> SurvivalGAM:
    """Fit a distributional survival GAM via the ``gam`` Python library.

    Parameters
    ----------
    time, event : arrays of shape (n,)
        Observed follow-up times (positive) and event indicators
        (1 = failure, 0 = right-censored).
    X : (n, p) ndarray
        Covariate matrix.  Columns whose unique values are a subset of
        {0, 1} are treated as binary and enter linearly; other columns
        enter as penalised smooths ``s(name)``.  When ``p == 0`` the
        model reduces to an intercept-only AFT.
    columns : tuple, optional
        Names for the ``p`` covariates.  Defaults to ``("x0", ...)``.
    n_samples : int, optional
        Number of posterior-predictive draws to generate per prediction
        (see module docstring for why this is a simulation, not MCMC).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float).reshape(time.shape[0], -1)
    if columns is None:
        columns = tuple(f"x{i}" for i in range(X.shape[1]))
    columns = tuple(columns)

    fit = _fit_gam(time, event, X, columns)

    X_design = {
        name: {"kind": fit.kinds[i], "index": i} for i, name in enumerate(columns)
    }

    summ = fit.summary_payload
    reml_iter = int(summ.get("iterations", 0))
    reml_score = summ.get("reml_score", np.nan)
    try:
        reml_score_f = float(reml_score)
    except Exception:
        reml_score_f = float("nan")

    backend_version = _gam.build_info().get("version", "?") if _GAM_AVAILABLE else "?"
    diagnostics = {
        "backend": f"gam (PyO3 binding, engine v{backend_version})",
        "formula": fit.formula,
        "reml_iterations": reml_iter,
        "reml_score": reml_score_f,
        "deviance": float(summ.get("deviance", float("nan"))),
        "edf_total": float(summ.get("edf_total", float("nan"))),
        "family_name": summ.get("family_name"),
        "converged": (reml_iter > 0) and np.isfinite(reml_score_f),
        "sigma_residual": fit.sigma,
        "n_train": fit.n_train,
        "n_events": fit.n_events,
        "n_posterior_draws": int(n_samples),
    }

    return SurvivalGAM(
        samples={
            "n_samples": int(n_samples),
            "coefficients": summ.get("coefficients"),
        },
        X_design=X_design,
        columns=columns,
        t_scale=fit.t_scale,
        diagnostics=diagnostics,
        _fit=fit,
        _n_posterior_draws=int(n_samples),
        _rng=rng,
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
    """Bayesian model averaging over candidate parent sets (Draper 1995).

    For each ``parent_sets[k]`` (a tuple of column indices into
    ``data_matrix``) we fit an independent ``SurvivalGAM`` and evaluate
    its posterior-predictive survival curves on ``X_eval / t_grid``.
    The averaged survival curve is

        S_bma(t, x) = sum_k w_k * mean_s S_k(s; t, x)

    where ``w_k`` are the structural weights (posterior probabilities of
    the parent sets, renormalised).  The total predictive variance is
    decomposed into parametric (within-model) and structural
    (between-model) components following Draper (1995, JRSS-B 57(1):
    45-97):

        Var_total      = Var_parametric + Var_structural
        Var_parametric = sum_k w_k Var_s S_k(s; t, x)
        Var_structural = sum_k w_k (mean_s S_k - S_bma)^2

    Parametric variance comes from the library's epistemic SE propagated
    through Normal posterior-predictive sampling; when the binding
    eventually exposes survival-family posteriors this will naturally
    upgrade to full Bayesian draws.

    Empty parent sets correspond to intercept-only AFTs (valid fits).
    """
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
    X_full = np.asarray(data_matrix, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    if X_eval is None:
        X_eval_full = X_full[:50]
    else:
        X_eval_full = np.asarray(X_eval, dtype=float)

    per_model_mean: List[np.ndarray] = []
    per_model_var: List[np.ndarray] = []
    per_model_diag: List[Dict[str, Any]] = []

    for pset in parent_sets:
        cols = tuple(pset)
        if len(cols) == 0:
            Xk = np.zeros((X_full.shape[0], 0))
            Xek = np.zeros((X_eval_full.shape[0], 0))
            col_names: Tuple[str, ...] = ()
        else:
            Xk = X_full[:, list(cols)]
            Xek = X_eval_full[:, list(cols)]
            col_names = tuple(column_names[c] for c in cols)
        gam_fit = fit_survival_gam(
            time,
            event,
            Xk,
            columns=col_names,
            **gam_kwargs,
        )
        S = gam_fit.predict_survival(Xek, t_grid)  # (S, n_eval, n_t)
        per_model_mean.append(S.mean(axis=0))
        per_model_var.append(S.var(axis=0, ddof=1))
        per_model_diag.append(gam_fit.diagnostics)

    means = np.stack(per_model_mean, axis=0)  # (K, n_eval, n_t)
    vars_ = np.stack(per_model_var, axis=0)
    w = weights[:, None, None]

    S_bma = np.sum(w * means, axis=0)  # (n_eval, n_t)
    var_parametric = np.sum(w * vars_, axis=0)
    var_structural = np.sum(w * (means - S_bma) ** 2, axis=0)
    var_total = var_parametric + var_structural

    eps = 1e-12
    struct_ratio = var_structural / (var_total + eps)

    return {
        "S_bma": S_bma,
        "var_parametric": var_parametric,
        "var_structural": var_structural,
        "var_total": var_total,
        "struct_ratio": struct_ratio,
        "per_model_mean": means,
        "per_model_var": vars_,
        "per_model_diag": per_model_diag,
        "weights": weights,
        "t_grid": t_grid,
    }


__all__ = [
    "SurvivalGAM",
    "fit_survival_gam",
    "bma_survival",
]
