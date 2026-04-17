"""Baseline survival + causal-edge models used to benchmark causal-pred.

Each baseline returns a dictionary with a consistent schema so the
benchmark driver (``scripts/benchmark.py``) can produce a uniform JSON
report and bar chart.  The baselines are:

  * ``run_kaplan_meier``: marginal survival with no covariates.
  * ``run_cox_ph``: Cox proportional-hazards regression on all covariates
    except the T2D indicator.
  * ``run_naive_logistic``: sklearn logistic regression predicting
    ``(time <= 10y) & (event == 1)``.
  * ``run_mr_ivw``: causal-edge classifier using the published MR-IVW
    estimates in :mod:`causal_pred.data.real_gwas` with a Bonferroni
    cut-off.
  * ``run_causal_pred``: the full pipeline (imports :mod:`causal_pred.pipeline`).
    Allowed to return ``{"status": "skipped", ...}`` if MCMC/GAM stages
    are unavailable.

Metrics computed (where defined):
  * Nagelkerke R^2 at t=10 y from the predicted survival.
  * Time-dependent AUC at t = 5, 10, 15 y.
  * Integrated Brier score over [0.5, 20] y.
  * For edge baselines: AUROC / AUPRC against ``CANONICAL_EDGES``.

Dependencies are intentionally limited to numpy, scipy, pandas, sklearn
and statsmodels, all of which ship in ``pyproject.toml``.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np

from .data.nodes import NODE_INDEX, NODE_NAMES, CANONICAL_EDGES
from .data.synthetic import SyntheticDataset
from .data.real_gwas import PUBLISHED_MR, LITERATURE_UNAVAILABLE, CIRCULAR_PAIRS
from .validation import nagelkerke_r2, time_dependent_auc, brier_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Default evaluation grid used for Brier / time-AUC on the benchmarks.
DEFAULT_T_GRID = np.linspace(0.5, 20.0, 40)

#: Default time-points for time-dependent AUC reporting.
DEFAULT_AUC_TIMES = (5.0, 10.0, 15.0)


def _covariate_matrix(data: SyntheticDataset) -> Tuple[np.ndarray, list]:
    """Covariates = all columns of ``X`` except the T2D outcome column.

    Returns the (n, p-1) array and the matching column-name list.
    """
    t2d_idx = NODE_INDEX["T2D"]
    cols = [i for i in range(data.X.shape[1]) if i != t2d_idx]
    X = data.X[:, cols].astype(float, copy=False)
    names = [data.columns[i] for i in cols]
    return X, names


def _safe_t_idx(t_grid: np.ndarray, t_star: float) -> int:
    return int(np.argmin(np.abs(t_grid - t_star)))


def _surv_metrics(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    t_grid: np.ndarray,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    eval_t: float = 10.0,
) -> dict:
    """Compute Nagelkerke R^2 at ``eval_t``, td-AUC, and IBS from a survival
    matrix ``surv`` of shape ``(n, len(t_grid))``."""
    t_idx = _safe_t_idx(t_grid, eval_t)
    p_event = 1.0 - surv[:, t_idx]
    y_event = ((time <= eval_t) & (event == 1)).astype(int)
    r2 = nagelkerke_r2(y_event, p_event) if y_event.sum() > 0 else float("nan")

    # Risk score = 1 - S(10y); higher = worse.
    td = time_dependent_auc(
        time=time,
        event=event,
        risk_score=p_event,
        eval_times=np.array(auc_times, dtype=float),
    )
    br = brier_score(time=time, event=event, survival_pred=surv, eval_times=t_grid)

    return {
        "nagelkerke_at_10y": float(r2),
        "time_dep_auc": {
            "times": [float(x) for x in td["times"]],
            "auc": [float(x) for x in td["auc"]],
            "integrated_auc": float(td["integrated_auc"]),
        },
        "ibs": float(br["ibs"]),
        "ibs_km": float(br["ibs_km"]),
        "scaled_brier": float(br["scaled_brier"]),
    }


# ---------------------------------------------------------------------------
# 1. Kaplan-Meier (no covariates)
# ---------------------------------------------------------------------------


def _km(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Plain Kaplan-Meier product-limit estimator.

    Returns ``(unique_event_times, S_hat)`` both of length k.  Duplicates
    the implementation in :mod:`causal_pred.validation.metrics` so the
    benchmark module is self-contained and doesn't import private APIs.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    order = np.argsort(t, kind="mergesort")
    t_s = t[order]
    e_s = e[order]
    uniq, inv = np.unique(t_s, return_inverse=True)
    d = np.zeros(uniq.size, dtype=float)
    np.add.at(d, inv, e_s.astype(float))
    counts = np.bincount(inv, minlength=uniq.size).astype(float)
    cum_before = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
    n_at = t.size - cum_before
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_at > 0, 1.0 - d / n_at, 1.0)
    S = np.cumprod(factors)
    return uniq, S


def _km_eval(times: np.ndarray, S: np.ndarray, t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, t, side="right") - 1
    out = np.ones_like(t, dtype=float)
    mask = idx >= 0
    out[mask] = S[idx[mask]]
    return out


def run_kaplan_meier(
    data: SyntheticDataset,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
) -> dict:
    """Population-level KM prediction.

    Every individual receives the marginal survival curve S_hat(t).  This
    is uninformative -- Nagelkerke R^2 and time-dependent AUC will be
    degenerate (NaN / 0.5) since all predictions are identical; we report
    them anyway for completeness.
    """
    t0 = time.perf_counter()
    uniq, S = _km(data.time, data.event)
    S_grid = _km_eval(uniq, S, t_grid)  # (n_t,)
    surv = np.broadcast_to(S_grid, (data.n, t_grid.size)).copy()
    metrics = _surv_metrics(data.time, data.event, surv, t_grid, auc_times=auc_times)
    metrics["runtime_s"] = float(time.perf_counter() - t0)
    metrics["model"] = "kaplan_meier"
    metrics["note"] = (
        "Population-level marginal survival; identical prediction per "
        "individual, so Nagelkerke R^2 and AUC are not meaningful."
    )
    return metrics


# ---------------------------------------------------------------------------
# 2. Cox proportional-hazards
# ---------------------------------------------------------------------------


def run_cox_ph(
    data: SyntheticDataset,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
) -> dict:
    """Cox PH using all covariates except the T2D outcome column."""
    from statsmodels.duration.hazard_regression import PHReg

    t0 = time.perf_counter()
    X, names = _covariate_matrix(data)

    # Standardise continuous columns for numerical stability (Cox PH is
    # scale-invariant on the hazard ratio but the solver is not).
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PHReg(
            endog=data.time.astype(float),
            exog=Xs,
            status=data.event.astype(float),
            ties="breslow",
        )
        fit = model.fit(method="bfgs", disp=0, maxiter=200)

    # Baseline cumulative hazard H0(t) via Breslow.  statsmodels returns
    # ``baseline_cumulative_hazard_function`` as a list of callables (one
    # per stratum); we have no strata so index 0.
    bch = fit.baseline_cumulative_hazard_function[0]
    H0_grid = np.asarray(bch(t_grid), dtype=float)  # (n_t,)

    lp = Xs @ fit.params  # linear predictor (n,)
    # S(t | X) = exp(-H0(t) * exp(lp))
    exp_lp = np.exp(lp)
    # broadcasting: (n, n_t) = exp(-H0(t)[None, :] * exp_lp[:, None])
    surv = np.exp(-np.outer(exp_lp, H0_grid))
    surv = np.clip(surv, 1e-9, 1.0)

    metrics = _surv_metrics(data.time, data.event, surv, t_grid, auc_times=auc_times)
    metrics["runtime_s"] = float(time.perf_counter() - t0)
    metrics["model"] = "cox_ph"
    metrics["covariates"] = names
    metrics["n_params"] = int(fit.params.size)
    metrics["llf"] = float(fit.llf)
    return metrics


# ---------------------------------------------------------------------------
# 3. Naive logistic regression at t = 10 y
# ---------------------------------------------------------------------------


def run_naive_logistic(
    data: SyntheticDataset,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    t_eval: float = 10.0,
) -> dict:
    """Logistic regression with ``y = I(T <= t_eval, delta = 1)``.

    The classifier is trained to predict the 10-year event; we expand the
    prediction to a "survival curve" by assuming a Weibull-like scaling
    so we can still report td-AUC and IBS.  This is obviously a crude
    extrapolation -- the integrated Brier is meaningful primarily as a
    sanity check that naive logistic is worse than Cox.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.perf_counter()
    X, names = _covariate_matrix(data)

    y = ((data.time <= t_eval) & (data.event == 1)).astype(int)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # ``max_iter=500`` to suppress convergence warnings on harder folds.
    # No regularisation tuning -- we keep the baseline "naive".
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(Xs, y)

    p10 = clf.predict_proba(Xs)[:, 1]
    p10 = np.clip(p10, 1e-6, 1.0 - 1e-6)

    # Construct a survival curve S(t) = (1 - p10)^((t / t_eval)^shape).
    # For t=t_eval we recover S(t_eval) = 1 - p10.  Shape=1 gives
    # exponential scaling.
    shape = 1.0
    with np.errstate(divide="ignore"):
        log_S_ref = np.log(1.0 - p10)  # (n,) at t=t_eval
    scale = (t_grid / t_eval) ** shape  # (n_t,)
    surv = np.exp(np.outer(log_S_ref, scale))
    surv = np.clip(surv, 1e-9, 1.0)

    # Compute Nagelkerke directly on the classifier's 10y probability (it
    # is exactly the model's target).
    r2 = nagelkerke_r2(y, p10)
    t_idx = _safe_t_idx(t_grid, t_eval)
    p_event = 1.0 - surv[:, t_idx]
    # Sanity check: p_event at t_idx should match p10 up to grid snap.
    td = time_dependent_auc(
        time=data.time,
        event=data.event,
        risk_score=p_event,
        eval_times=np.array(auc_times, dtype=float),
    )
    br = brier_score(
        time=data.time, event=data.event, survival_pred=surv, eval_times=t_grid
    )

    out = {
        "nagelkerke_at_10y": float(r2),
        "time_dep_auc": {
            "times": [float(x) for x in td["times"]],
            "auc": [float(x) for x in td["auc"]],
            "integrated_auc": float(td["integrated_auc"]),
        },
        "ibs": float(br["ibs"]),
        "ibs_km": float(br["ibs_km"]),
        "scaled_brier": float(br["scaled_brier"]),
        "runtime_s": float(time.perf_counter() - t0),
        "model": "naive_logistic",
        "covariates": names,
        "n_positives": int(y.sum()),
        "t_eval": t_eval,
    }
    return out


# ---------------------------------------------------------------------------
# 4. Naive MR-IVW edge classifier
# ---------------------------------------------------------------------------


def _build_mr_edge_scores(
    node_names: Sequence[str] = NODE_NAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build three (p, p) matrices from PUBLISHED_MR: ``|z|``, ``beta``
    and ``p-value``.  NaN where no published estimate is available."""
    from scipy.stats import norm

    p = len(node_names)
    idx = {n: i for i, n in enumerate(node_names)}
    z_abs = np.full((p, p), np.nan)
    beta = np.full((p, p), np.nan)
    pvals = np.full((p, p), np.nan)

    for (exp_name, out_name), entry in PUBLISHED_MR.items():
        if entry is LITERATURE_UNAVAILABLE:
            continue
        if exp_name not in idx or out_name not in idx:
            continue
        b, se, _L, _cite = entry
        if se <= 0 or not np.isfinite(b):
            continue
        i, j = idx[exp_name], idx[out_name]
        z_abs[i, j] = abs(b / se)
        beta[i, j] = b
        pvals[i, j] = 2.0 * (1.0 - norm.cdf(abs(b / se)))

    return z_abs, beta, pvals


def run_mr_ivw(
    node_names: Sequence[str] = NODE_NAMES,
    ground_truth_edges: Sequence[Tuple[str, str]] = CANONICAL_EDGES,
    alpha: float = 0.05,
) -> dict:
    """Naive MR-IVW edge classifier.

    For each exposure->outcome pair with a published IVW estimate, we
    declare an edge present if ``p < alpha / n_tests`` (Bonferroni).  The
    continuous score used for AUROC/AUPRC is ``|z|`` (the absolute
    IVW-derived Wald statistic).

    Cells with no published estimate are left NaN and excluded from both
    the predictions and the ground-truth set.  Circular pairs
    (HbA1c -> T2D) are also excluded.
    """
    t0 = time.perf_counter()
    z_abs, beta, pvals = _build_mr_edge_scores(node_names)

    # Exclude circular pairs.
    idx = {n: i for i, n in enumerate(node_names)}
    for a, b in CIRCULAR_PAIRS:
        if a in idx and b in idx:
            z_abs[idx[a], idx[b]] = np.nan
            pvals[idx[a], idx[b]] = np.nan

    p = len(node_names)
    off = ~np.eye(p, dtype=bool)
    usable = off & np.isfinite(z_abs)
    # Bonferroni cut-off across tested cells.
    n_tests = int(usable.sum())
    if n_tests == 0:
        raise RuntimeError("no usable MR-IVW cells; check PUBLISHED_MR")
    thresh = alpha / n_tests

    A = np.zeros((p, p), dtype=bool)
    for parent, child in ground_truth_edges:
        if parent not in idx or child not in idx:
            continue
        A[idx[parent], idx[child]] = True

    # Restrict labels + scores to usable cells only.
    scores = z_abs[usable]
    labels = A[usable]

    # AUROC / AUPRC on usable cells.
    from .validation.known_edges import _auroc, _auprc

    auroc = float(_auroc(scores, labels))
    auprc = float(_auprc(scores, labels))

    y_pred_sig = pvals[usable] < thresh

    # Confusion-matrix summary.
    tp = int(np.sum(y_pred_sig & labels))
    fp = int(np.sum(y_pred_sig & ~labels))
    fn = int(np.sum(~y_pred_sig & labels))
    tn = int(np.sum(~y_pred_sig & ~labels))

    out = {
        "model": "mr_ivw",
        "edge_auprc": auprc,
        "edge_auroc": auroc,
        "bonferroni_alpha": float(thresh),
        "n_tests": n_tests,
        "significant_edges": int(y_pred_sig.sum()),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_ground_truth_in_usable": int(labels.sum()),
        "runtime_s": float(time.perf_counter() - t0),
    }
    return out


# ---------------------------------------------------------------------------
# 5. Full causal-pred pipeline
# ---------------------------------------------------------------------------


def run_causal_pred(
    data: SyntheticDataset,
    mcmc_iter: int = 500,
    mcmc_chains: int = 1,
    gam_samples: int = 100,
    gam_warmup: int = 50,
    use_real_gwas: bool = True,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Run the full causal-pred pipeline and return its benchmark row.

    Gracefully returns ``{"status": "skipped", ...}`` if any stage raises
    ``NotImplementedError`` (e.g. MCMC or GAM not yet wired up).
    """
    t0 = time.perf_counter()
    try:
        from .pipeline import run_pipeline

        if rng is None:
            rng = np.random.default_rng(20260416)

        # We re-use the provided dataset -- run_pipeline regenerates its
        # own.  Here we just wrap it and measure.  For tight integration
        # the caller should already have matched n / seeds.
        result = run_pipeline(
            n=data.n,
            use_real_gwas=use_real_gwas,
            mcmc_iter=mcmc_iter,
            mcmc_chains=mcmc_chains,
            gam_samples=gam_samples,
            gam_warmup=gam_warmup,
            rng=rng,
            verbose=False,
        )

        # Validation already computed on the eval split inside run_pipeline.
        val = result.validation
        td = val.get("time_dependent_auc", {})
        br = val.get("brier", {})

        out = {
            "model": "causal_pred",
            "status": "ok",
            "nagelkerke_at_10y": float(val.get("nagelkerke_r2_at_10y", float("nan"))),
            "time_dep_auc": {
                "times": [float(x) for x in td.get("times", [])],
                "auc": [float(x) for x in td.get("auc", [])],
                "integrated_auc": float(td.get("integrated_auc", float("nan"))),
            },
            "ibs": float(br.get("ibs", float("nan"))),
            "edge_auroc": float(
                val.get("known_edge_recovery", {}).get("auroc", float("nan"))
            ),
            "edge_auprc": float(
                val.get("known_edge_recovery", {}).get("auprc", float("nan"))
            ),
            "runtime_s": float(time.perf_counter() - t0),
        }
        return out
    except NotImplementedError as exc:
        return {
            "model": "causal_pred",
            "status": "skipped",
            "reason": f"NotImplementedError: {exc}",
            "runtime_s": float(time.perf_counter() - t0),
        }
    except Exception as exc:
        return {
            "model": "causal_pred",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": float(time.perf_counter() - t0),
        }


# ---------------------------------------------------------------------------
# Aggregate driver
# ---------------------------------------------------------------------------


def run_all_baselines(
    data: SyntheticDataset,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    run_full_pipeline: bool = True,
    mcmc_iter: int = 500,
    mcmc_chains: int = 1,
    gam_samples: int = 100,
    gam_warmup: int = 50,
    use_real_gwas: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Run every baseline on ``data`` and return a dict keyed by model."""
    out = {
        "kaplan_meier": run_kaplan_meier(data, t_grid=t_grid, auc_times=auc_times),
        "cox_ph": run_cox_ph(data, t_grid=t_grid, auc_times=auc_times),
        "naive_logistic": run_naive_logistic(data, t_grid=t_grid, auc_times=auc_times),
        "mr_ivw": run_mr_ivw(),
    }
    if run_full_pipeline:
        out["causal_pred"] = run_causal_pred(
            data,
            mcmc_iter=mcmc_iter,
            mcmc_chains=mcmc_chains,
            gam_samples=gam_samples,
            gam_warmup=gam_warmup,
            use_real_gwas=use_real_gwas,
            t_grid=t_grid,
            auc_times=auc_times,
            rng=rng,
        )
    else:
        out["causal_pred"] = {
            "model": "causal_pred",
            "status": "skipped",
            "reason": "run_full_pipeline=False",
        }
    return out


__all__ = [
    "DEFAULT_T_GRID",
    "DEFAULT_AUC_TIMES",
    "run_kaplan_meier",
    "run_cox_ph",
    "run_naive_logistic",
    "run_mr_ivw",
    "run_causal_pred",
    "run_all_baselines",
]
