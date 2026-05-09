"""Baseline survival + causal-edge models used to benchmark causal-pred.

Each baseline returns a dictionary with a consistent schema so the
benchmark driver (``scripts/benchmark.py``) can produce a uniform JSON
report and bar chart.  The baselines are:

  * ``run_kaplan_meier``: marginal survival with no covariates.
  * ``run_cox_ph``: Cox proportional-hazards regression on all covariates
    except the T2D indicator.
  * ``run_naive_logistic``: sklearn logistic regression predicting an
    IPCW fixed-horizon 10-year endpoint.
  * ``run_mr_ivw``: causal-edge classifier using the published MR-IVW
    estimates in :mod:`causal_pred.data.real_gwas` with a Bonferroni
    cut-off.
  * ``run_causal_pred``: synthetic-data run of the causal-pred stack:
    MrDAG priors -> DAGSLAM -> structure MCMC -> gamfit survival GAM.

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data.nodes import NODE_INDEX, NODE_NAMES, NODE_TYPES, NODES, CANONICAL_EDGES
from .data.synthetic import SyntheticDataset
from .data.gwas import simulate_gwas
from .data.real_gwas import (
    PUBLISHED_MR,
    LITERATURE_UNAVAILABLE,
    CIRCULAR_PAIRS,
    load_real_gwas,
)
from .dagslam import run_dagslam
from .mcmc import run_structure_mcmc
from .mrdag import run_mrdag
from .validation import nagelkerke_r2, time_dependent_auc, brier_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Default evaluation grid used for Brier / time-AUC on the benchmarks.
DEFAULT_T_GRID = np.linspace(0.5, 20.0, 40)

#: Default time-points for time-dependent AUC reporting.
DEFAULT_AUC_TIMES = (5.0, 10.0, 15.0)


def _train_test_indices(
    event: np.ndarray,
    test_fraction: float = 0.3,
    seed: int = 20260416,
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(event, dtype=int).ravel()
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for value in (0, 1):
        idx = np.flatnonzero(e == value)
        rng.shuffle(idx)
        if idx.size == 0:
            continue
        n_test = max(1, int(round(test_fraction * idx.size))) if idx.size > 1 else 0
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train = np.concatenate(train_parts) if train_parts else np.arange(e.size)
    test = np.concatenate(test_parts) if test_parts else np.zeros(0, dtype=int)
    if train.size == 0 or test.size == 0:
        raise ValueError("benchmark split produced an empty train or test set")
    train.sort()
    test.sort()
    return train, test


def _covariate_matrix(data: SyntheticDataset) -> Tuple[np.ndarray, list]:
    """Covariates = all columns of ``X`` except the T2D outcome column.

    Returns the (n, p-1) array and the matching column-name list.
    """
    t2d_idx = NODE_INDEX["T2D"]
    cols = [i for i in range(data.X.shape[1]) if i != t2d_idx]
    X = data.X[:, cols].astype(float, copy=False)
    names = [data.columns[i] for i in cols]
    return X, names


def _surv_at_times(surv: np.ndarray, t_grid: np.ndarray, times: Sequence[float]) -> np.ndarray:
    S = np.asarray(surv, dtype=float)
    grid = np.asarray(t_grid, dtype=float).ravel()
    eval_times = np.asarray(times, dtype=float).ravel()
    if S.ndim != 2:
        raise ValueError(f"survival matrix must be 2-D, got shape {S.shape}")
    if S.shape[1] != grid.size:
        raise ValueError(f"survival matrix has {S.shape[1]} columns but grid has {grid.size}")
    if grid.size == 0:
        raise ValueError("survival time grid is empty")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("survival time grid must be strictly increasing")
    if not (
        np.all(np.isfinite(S))
        and np.all(np.isfinite(grid))
        and np.all(np.isfinite(eval_times))
    ):
        raise ValueError("survival matrix, grid, and evaluation times must be finite")
    if np.any((S < -1e-12) | (S > 1.0 + 1e-12)):
        raise ValueError("survival matrix must contain probabilities in [0, 1]")
    S = np.clip(S, 0.0, 1.0)
    out = np.empty((S.shape[0], eval_times.size), dtype=float)
    for i in range(S.shape[0]):
        out[i] = np.interp(eval_times, grid, S[i], left=S[i, 0], right=S[i, -1])
    return out


def _determined_horizon_status(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    indeterminate = (t < horizon) & (e == 0)
    determined = ~indeterminate
    y = ((t[determined] <= horizon) & (e[determined] == 1)).astype(int)
    return y, determined


def _horizon_ipcw_training_endpoint(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    case = (t <= horizon) & (e == 1)
    control = t > horizon
    keep = case | control
    if not np.any(keep):
        raise ValueError("no determined rows for fixed-horizon logistic baseline")

    cens = 1 - e
    km_t, km_s = _km(t, cens)
    g_t_minus = _km_eval(km_t, km_s, np.nextafter(t, -np.inf))
    g_horizon = float(_km_eval(km_t, km_s, np.asarray([horizon]))[0])
    eps = 1e-8
    weights = np.zeros_like(t, dtype=float)
    weights[case] = 1.0 / np.clip(g_t_minus[case], eps, 1.0)
    weights[control] = 1.0 / np.clip(g_horizon, eps, 1.0)
    y = case[keep].astype(int)
    return y, weights[keep], keep


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
    p_event = 1.0 - _surv_at_times(surv, t_grid, [eval_t])[:, 0]
    y_event, determined = _determined_horizon_status(time, event, eval_t)
    if y_event.size > 0 and 0 < int(y_event.sum()) < y_event.size:
        r2 = nagelkerke_r2(y_event, p_event[determined])
    else:
        r2 = float("nan")

    auc_surv = _surv_at_times(surv, t_grid, auc_times)
    td = time_dependent_auc(
        time=time,
        event=event,
        risk_score=1.0 - auc_surv,
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
        "nagelkerke_n_used": int(determined.sum()),
        "nagelkerke_n_indeterminate": int((~determined).sum()),
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
    train_idx, test_idx = _train_test_indices(data.event)
    uniq, S = _km(data.time[train_idx], data.event[train_idx])
    S_grid = _km_eval(uniq, S, t_grid)  # (n_t,)
    surv = np.broadcast_to(S_grid, (test_idx.size, t_grid.size)).copy()
    metrics = _surv_metrics(
        data.time[test_idx],
        data.event[test_idx],
        surv,
        t_grid,
        auc_times=auc_times,
    )
    metrics["runtime_s"] = float(time.perf_counter() - t0)
    metrics["model"] = "kaplan_meier"
    metrics["evaluation"] = "held_out"
    metrics["n_train"] = int(train_idx.size)
    metrics["n_test"] = int(test_idx.size)
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
    train_idx, test_idx = _train_test_indices(data.event)
    X_train = X[train_idx]
    X_test = X[test_idx]

    # Standardise continuous columns for numerical stability (Cox PH is
    # scale-invariant on the hazard ratio but the solver is not).
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs_train = (X_train - mu) / sd
    Xs_test = (X_test - mu) / sd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PHReg(
            endog=data.time[train_idx].astype(float),
            exog=Xs_train,
            status=data.event[train_idx].astype(float),
            ties="breslow",
        )
        fit = model.fit(method="bfgs", disp=0, maxiter=200)

    # Baseline cumulative hazard H0(t) via Breslow.  statsmodels returns
    # ``baseline_cumulative_hazard_function`` as a list of callables (one
    # per stratum); we have no strata so index 0.
    bch = fit.baseline_cumulative_hazard_function[0]
    H0_grid = np.asarray(bch(t_grid), dtype=float)  # (n_t,)

    lp = Xs_test @ fit.params  # linear predictor (n_test,)
    # S(t | X) = exp(-H0(t) * exp(lp))
    exp_lp = np.exp(lp)
    # broadcasting: (n, n_t) = exp(-H0(t)[None, :] * exp_lp[:, None])
    surv = np.exp(-np.outer(exp_lp, H0_grid))
    surv = np.clip(surv, 1e-9, 1.0)

    metrics = _surv_metrics(
        data.time[test_idx],
        data.event[test_idx],
        surv,
        t_grid,
        auc_times=auc_times,
    )
    metrics["runtime_s"] = float(time.perf_counter() - t0)
    metrics["model"] = "cox_ph"
    metrics["evaluation"] = "held_out"
    metrics["n_train"] = int(train_idx.size)
    metrics["n_test"] = int(test_idx.size)
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
    t0 = time.perf_counter()
    X, names = _covariate_matrix(data)
    train_idx, test_idx = _train_test_indices(data.event)

    y_train, w_train, keep_train = _horizon_ipcw_training_endpoint(
        data.time[train_idx],
        data.event[train_idx],
        t_eval,
    )
    X_train = X[train_idx][keep_train]
    X_test = X[test_idx]

    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    # ``max_iter=500`` to suppress convergence warnings on harder folds.
    # No regularisation tuning -- we keep the baseline "naive".
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(Xs_train, y_train, sample_weight=w_train)

    p10 = clf.predict_proba(Xs_test)[:, 1]
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
    y_test, determined_test = _determined_horizon_status(
        data.time[test_idx],
        data.event[test_idx],
        t_eval,
    )
    if y_test.size > 0 and 0 < int(y_test.sum()) < y_test.size:
        r2 = nagelkerke_r2(y_test, p10[determined_test])
    else:
        r2 = float("nan")
    auc_surv = _surv_at_times(surv, t_grid, auc_times)
    td = time_dependent_auc(
        time=data.time[test_idx],
        event=data.event[test_idx],
        risk_score=1.0 - auc_surv,
        eval_times=np.array(auc_times, dtype=float),
    )
    br = brier_score(
        time=data.time[test_idx],
        event=data.event[test_idx],
        survival_pred=surv,
        eval_times=t_grid,
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
        "evaluation": "held_out",
        "covariates": names,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_train_determined": int(keep_train.sum()),
        "n_test_determined": int(determined_test.sum()),
        "n_positives": int(y_train.sum()),
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
    use_real_gwas: bool = True,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Run the causal-pred stack on the synthetic benchmark data.

    The benchmark uses the same model families as production while keeping
    the input synthetic-data native: MrDAG supplies edge priors, DAGSLAM
    gives a warm-start DAG, structure MCMC samples parent sets, and gamfit
    fits held-out survival curves for the sampled T2D parent sets.
    """
    t0 = time.perf_counter()
    if rng is None:
        rng = np.random.default_rng(20260416)
    if tuple(data.columns) != tuple(NODE_NAMES):
        raise ValueError("causal-pred benchmark requires synthetic NODE_NAMES order")
    if tuple(data.node_types) != tuple(NODE_TYPES):
        raise ValueError("causal-pred benchmark requires synthetic NODE_TYPES")

    train_idx, test_idx = _train_test_indices(data.event)
    X_train = np.asarray(data.X[train_idx], dtype=float)
    X_test = np.asarray(data.X[test_idx], dtype=float)
    allowed_edges = _benchmark_allowed_edges(data.columns, data.node_types)

    gwas = load_real_gwas() if use_real_gwas else simulate_gwas(rng=rng)
    mrdag_iter = max(80, min(1000, int(mcmc_iter) * 4))
    mrdag_burn = max(20, min(mrdag_iter // 2, mrdag_iter // 5))
    mrdag_thin = max(1, (mrdag_iter - mrdag_burn) // max(20, int(mcmc_iter)))
    mrdag = run_mrdag(
        gwas,
        rng=rng,
        n_iter=mrdag_iter,
        n_chains=max(1, int(mcmc_chains)),
        n_burn=mrdag_burn,
        thin=mrdag_thin,
    )
    pi_prior = np.asarray(mrdag.pi, dtype=float)

    dagslam = run_dagslam(
        data=X_train,
        node_types=data.node_types,
        max_parents=3,
        max_iter=max(25, min(500, int(mcmc_iter))),
        restarts=1,
        rng=rng,
        pi_prior=pi_prior,
        allowed_edges=allowed_edges,
        survival_time=data.time[train_idx],
        survival_event=data.event[train_idx],
        survival_horizon=10.0,
    )

    n_graph_samples = max(1, int(mcmc_iter))
    mcmc = run_structure_mcmc(
        data=X_train,
        node_types=data.node_types,
        start_adj=dagslam.adjacency,
        pi_prior=pi_prior,
        n_samples=n_graph_samples,
        burn_in=max(5, min(100, n_graph_samples // 2)),
        thin=1,
        n_chains=max(1, int(mcmc_chains)),
        rng=rng,
        perturb_flips=2,
        hybrid_prob=0.1,
        edge_resample_prob=0.2,
        block_resample_prob=0.0,
        exact_parent_resample=False,
        max_parents=3,
        allowed_edges=allowed_edges,
        survival_time=data.time[train_idx],
        survival_event=data.event[train_idx],
        survival_horizon=10.0,
    )
    samples = (
        np.stack(mcmc.samples, axis=0).astype(np.int8)
        if mcmc.samples
        else np.zeros((0, data.p, data.p), dtype=np.int8)
    )
    target_idx = NODE_INDEX["T2D"]
    parent_sets, weights, parent_counts = _sampled_parent_sets(
        samples,
        target_idx=target_idx,
        edge_probs=np.asarray(mcmc.edge_probs, dtype=float),
        top_k=3,
    )

    from .gam.survival import fit_survival_gam

    per_model = []
    parent_set_rows = []
    fit_summaries = []
    gam_t0 = time.perf_counter()
    for parent_set, weight, count in zip(parent_sets, weights, parent_counts):
        cols = tuple(data.columns[i] for i in parent_set)
        train_X_ps = (
            X_train[:, list(parent_set)]
            if parent_set
            else np.zeros((train_idx.size, 0), dtype=float)
        )
        test_X_ps = (
            X_test[:, list(parent_set)]
            if parent_set
            else np.zeros((test_idx.size, 0), dtype=float)
        )
        fit = fit_survival_gam(
            data.time[train_idx],
            data.event[train_idx],
            train_X_ps,
            columns=cols,
            n_uncertainty_slices=max(1, int(gam_samples)),
            progress=False,
        )
        per_model.append(fit.predict_survival_mean(test_X_ps, t_grid))
        diag = fit.posterior_summary()
        diag["parent_columns"] = list(cols)
        diag["posterior_parent_set_weight"] = float(weight)
        diag["posterior_parent_set_count"] = int(count)
        fit_summaries.append(diag)
        parent_set_rows.append(
            {
                "columns": list(cols),
                "weight": float(weight),
                "count": int(count),
            }
        )

    stack = np.stack(per_model, axis=0)
    survival = np.einsum("k,knt->nt", weights, stack)
    metrics = _surv_metrics(
        data.time[test_idx],
        data.event[test_idx],
        survival,
        t_grid,
        auc_times=auc_times,
    )
    edge_metrics = _edge_recovery_from_probs(
        np.asarray(mcmc.edge_probs, dtype=float),
        data.ground_truth_adj,
        allowed_edges,
    )
    metrics.update(edge_metrics)
    metrics.update(
        {
            "model": "causal_pred",
            "backend": "gamfit",
            "evaluation": "held_out",
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "runtime_s": float(time.perf_counter() - t0),
            "gam_runtime_s": float(time.perf_counter() - gam_t0),
            "gwas_source": "literature" if use_real_gwas else "simulated",
            "mrdag_n_candidate_edges": int(
                mrdag.diagnostics.get("n_candidate_edges", 0)
            ),
            "dagslam_n_edges": int(dagslam.n_edges),
            "dagslam_log_score": float(dagslam.log_score),
            "mcmc_n_samples": int(samples.shape[0]),
            "mcmc_accept_rate": dict(mcmc.diagnostics.get("accept_rate", {})),
            "mcmc_max_rhat_skeleton": float(
                mcmc.diagnostics.get("max_rhat_skeleton", float("nan"))
            ),
            "parent_sets": parent_set_rows,
            "gam_fit_summaries": fit_summaries,
        }
    )
    return metrics


def _benchmark_allowed_edges(
    columns: Sequence[str],
    node_types: Sequence[str],
) -> np.ndarray:
    p = len(columns)
    allowed = np.ones((p, p), dtype=bool)
    np.fill_diagonal(allowed, False)
    exogenous = {node.name for node in NODES if node.exogenous}
    for i, (name, kind) in enumerate(zip(columns, node_types)):
        if str(kind) == "survival":
            allowed[i, :] = False
        if str(name) in exogenous:
            allowed[:, i] = False
    np.fill_diagonal(allowed, False)
    return allowed


def _sampled_parent_sets(
    samples: np.ndarray,
    target_idx: int,
    edge_probs: np.ndarray,
    top_k: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, list[int]]:
    counts: dict[tuple[int, ...], int] = {}
    for sample in samples:
        parents = tuple(int(i) for i in np.flatnonzero(sample[:, target_idx]))
        parents = tuple(i for i in parents if i != target_idx)
        counts[parents] = counts.get(parents, 0) + 1

    if not counts:
        probs = np.asarray(edge_probs[:, target_idx], dtype=float).copy()
        probs[target_idx] = 0.0
        parents = tuple(int(i) for i in np.argsort(-probs)[:3] if probs[i] > 0.0)
        counts[parents] = 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    parent_sets = [k for k, _v in ranked]
    raw_counts = [int(v) for _k, v in ranked]
    weights = np.asarray(raw_counts, dtype=float)
    weights /= float(weights.sum())
    return parent_sets, weights, raw_counts


def _edge_recovery_from_probs(
    edge_probs: np.ndarray,
    ground_truth_adj: np.ndarray,
    allowed_edges: np.ndarray,
) -> dict:
    from .validation.known_edges import _auprc, _auroc

    p = edge_probs.shape[0]
    off_diag = ~np.eye(p, dtype=bool)
    mask = off_diag & np.asarray(allowed_edges, dtype=bool)
    scores = np.asarray(edge_probs, dtype=float)[mask]
    labels = np.asarray(ground_truth_adj, dtype=bool)[mask]
    if scores.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return {
            "edge_auroc": float("nan"),
            "edge_auprc": float("nan"),
            "edge_n_eval": int(scores.size),
            "edge_n_positive": int(labels.sum()),
        }
    return {
        "edge_auroc": float(_auroc(scores, labels)),
        "edge_auprc": float(_auprc(scores, labels)),
        "edge_n_eval": int(scores.size),
        "edge_n_positive": int(labels.sum()),
    }


# ---------------------------------------------------------------------------
# Aggregate driver
# ---------------------------------------------------------------------------


def run_all_baselines(
    data: SyntheticDataset,
    t_grid: np.ndarray = DEFAULT_T_GRID,
    auc_times: Sequence[float] = DEFAULT_AUC_TIMES,
    mcmc_iter: int = 500,
    mcmc_chains: int = 1,
    gam_samples: int = 100,
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
    out["causal_pred"] = run_causal_pred(
        data,
        mcmc_iter=mcmc_iter,
        mcmc_chains=mcmc_chains,
        gam_samples=gam_samples,
        use_real_gwas=use_real_gwas,
        t_grid=t_grid,
        auc_times=auc_times,
        rng=rng,
    )
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
