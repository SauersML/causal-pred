"""Validation metrics for the causal-prediction pipeline.

This module implements the numerical quantities used to evaluate the GAM
stage's calibration, discrimination, and uncertainty on (censored) survival
data, plus generic bootstrap confidence intervals.

Implemented metrics and their references:

* Brier score: Brier (1950), "Verification of forecasts expressed in terms
  of probability".
* Brier-score decomposition (reliability / resolution / uncertainty):
  Murphy (1973), "A new vector partition of the probability score".
* Nagelkerke R^2: Nagelkerke (1991), "A note on a general definition of the
  coefficient of determination".
* Hosmer-Lemeshow goodness-of-fit: Hosmer & Lemeshow (1980).
* Expected Calibration Error: Naeini, Cooper & Hauskrecht (2015).
* Time-dependent cumulative/dynamic AUC: Heagerty, Lumley & Pepe (2000);
  IPCW form due to Uno et al. (2007).
* Time-dependent (IPCW) Brier score and Integrated Brier Score: Graf,
  Schmoor, Sauerbrei & Schumacher (1999).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import stats

_EPS = 1e-12


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y with respect to x.

    Small shim so the module works under both NumPy 1.x (``np.trapz``)
    and NumPy 2.x (``np.trapezoid``).
    """
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if fn is None:  # pragma: no cover -- NumPy always has one of these.
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
    return float(fn(y, x))


# ---------------------------------------------------------------------------
# Nagelkerke R^2  (Nagelkerke 1991)
# ---------------------------------------------------------------------------


def nagelkerke_r2(y_true, p_pred) -> float:
    """Nagelkerke (1991) generalised R^2 for a Bernoulli model.

    ``R^2_{CS} = 1 - (L0 / L1)^{2/n}`` is the Cox-Snell pseudo-R^2, and

        R^2_N = R^2_{CS} / (1 - L0^{2/n})

    rescales it so that the maximum attainable value is 1.  ``L0`` is the
    intercept-only likelihood (Bernoulli at the sample mean) and ``L1`` is
    the likelihood of the supplied per-sample predictions.

    Parameters
    ----------
    y_true : (n,) array_like of 0/1 outcomes.
    p_pred : (n,) array_like of predicted probabilities for ``y=1``.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y_true and p_pred must have the same shape")
    n = y.size
    if n == 0:
        return float("nan")
    p = np.clip(p, _EPS, 1.0 - _EPS)
    ybar = float(np.mean(y))
    ybar_c = min(max(ybar, _EPS), 1.0 - _EPS)

    # Log-likelihood of supplied model (L1) and null (L0).
    log_L1 = np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    log_L0 = n * (ybar_c * np.log(ybar_c) + (1.0 - ybar_c) * np.log1p(-ybar_c))

    # R^2_CS = 1 - exp( (2/n) * (log L0 - log L1) )
    cs = 1.0 - np.exp((2.0 / n) * (log_L0 - log_L1))
    denom = 1.0 - np.exp((2.0 / n) * log_L0)
    if denom <= 0.0:
        return 0.0
    return float(cs / denom)


# ---------------------------------------------------------------------------
# Calibration metrics (Brier, Murphy decomposition, ECE, MCE, Hosmer-Lemeshow)
# ---------------------------------------------------------------------------


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        # Ensure strict monotonicity (collapse duplicates by nudging upward).
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        return edges
    raise ValueError(f"unknown strategy {strategy!r}")


def _bin_assign(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Assign into [edges[i], edges[i+1]); last bin closed on the right.
    k = np.searchsorted(edges, p, side="right") - 1
    k = np.clip(k, 0, edges.size - 2)
    return k


def calibration_metrics(
    y_true, p_pred, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Calibration diagnostics for binary probabilistic forecasts.

    Returns
    -------
    dict with keys
        ``brier``: Brier (1950) mean squared error.
        ``brier_decomposition``: reliability / resolution / uncertainty a la
            Murphy (1973): ``brier = reliability - resolution + uncertainty``.
        ``ece``: expected calibration error (Naeini et al. 2015), a weighted
            average of per-bin |mean(p) - mean(y)|.
        ``mce``: max over bins of |mean(p) - mean(y)|.
        ``reliability``: dict of arrays describing the reliability diagram.
        ``hl_stat``, ``hl_df``, ``hl_pvalue``: Hosmer-Lemeshow (1980)
            goodness-of-fit chi-square with ``n_bins - 2`` degrees of freedom.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y_true and p_pred must have the same shape")
    n = y.size
    p_clip = np.clip(p, 0.0, 1.0)

    # Brier score (Brier 1950).
    brier = float(np.mean((y - p_clip) ** 2))

    # Murphy (1973) decomposition using the binning induced by ``strategy``.
    edges = _bin_edges(p_clip, n_bins, strategy)
    k = _bin_assign(p_clip, edges)

    bin_n = np.zeros(n_bins, dtype=float)
    bin_sum_p = np.zeros(n_bins, dtype=float)
    bin_sum_y = np.zeros(n_bins, dtype=float)
    np.add.at(bin_n, k, 1.0)
    np.add.at(bin_sum_p, k, p_clip)
    np.add.at(bin_sum_y, k, y)

    nonempty = bin_n > 0
    mean_p = np.zeros(n_bins, dtype=float)
    mean_y = np.zeros(n_bins, dtype=float)
    mean_p[nonempty] = bin_sum_p[nonempty] / bin_n[nonempty]
    mean_y[nonempty] = bin_sum_y[nonempty] / bin_n[nonempty]

    ybar = float(np.mean(y))

    # Murphy (1973) decomposition, generalised (Broecker 2009) so that the
    # identity  BS = REL - RES + UNC  holds exactly for continuous forecasts:
    #     resolution  = (1/n) sum_k n_k (mean_y_k - ybar)^2
    #     uncertainty = ybar (1 - ybar)
    #     reliability = BS - UNC + RES
    # For discrete forecasts (all forecasts within a bin share one value),
    # this is equivalent to the textbook Murphy 1973 formula
    # (1/n) sum_k n_k (mean_p_k - mean_y_k)^2; for continuous forecasts it
    # additionally absorbs the within-bin forecast variance (Broecker 2009
    # "Reliability, sufficiency, and the decomposition of proper scores").
    resolution = float(np.sum(bin_n * (mean_y - ybar) ** 2) / n)
    uncertainty = float(ybar * (1.0 - ybar))
    reliability = float(brier - uncertainty + resolution)

    # ECE / MCE (Naeini et al. 2015).
    gaps = np.abs(mean_p - mean_y)
    ece = float(np.sum((bin_n / n) * gaps))
    mce = float(np.max(gaps[nonempty])) if np.any(nonempty) else 0.0

    # Hosmer-Lemeshow (1980): chi^2 = sum (O - E)^2 / (E (1 - E/n_k)).
    # df = n_bins - 2 (standard for HL with continuous predictors).
    hl = 0.0
    eff_bins = 0
    for j in range(n_bins):
        nj = bin_n[j]
        if nj <= 0:
            continue
        eff_bins += 1
        oj = bin_sum_y[j]
        ej = bin_sum_p[j]
        pj = ej / nj
        denom = nj * pj * (1.0 - pj)
        if denom <= 0.0:
            # Perfect bin (all 0 or all 1 predicted); skip to avoid div/0.
            continue
        hl += (oj - ej) ** 2 / denom
    hl_df = max(eff_bins - 2, 1)
    hl_pvalue = float(1.0 - stats.chi2.cdf(hl, df=hl_df))

    return {
        "brier": brier,
        "brier_decomposition": {
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
        },
        "ece": ece,
        "mce": mce,
        "reliability": {
            "mean_predicted": mean_p,
            "mean_observed": mean_y,
            "bin_counts": bin_n.astype(int),
            "bin_edges": edges,
        },
        "hl_stat": float(hl),
        "hl_df": int(hl_df),
        "hl_pvalue": hl_pvalue,
    }


# ---------------------------------------------------------------------------
# Kaplan-Meier (for IPCW weights)
# ---------------------------------------------------------------------------


def _km_estimator(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier product-limit estimator.

    Returns the unique sorted event times and S(t) evaluated at them
    (right-continuous; caller uses ``_km_eval`` to evaluate at arbitrary t).
    """
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    e = event[order].astype(float)
    uniq, inv = np.unique(t, return_inverse=True)
    d = np.zeros(uniq.size, dtype=float)
    n_at = np.zeros(uniq.size, dtype=float)
    total = t.size
    # At each unique time the number at risk is (total - #events-or-censors
    # strictly before it).  We compute n_at_risk with a running counter.
    # Count events and total observations at each unique time.
    np.add.at(d, inv, e)
    counts = np.bincount(inv, minlength=uniq.size).astype(float)
    # Risk set at time t_i = total minus number of observations strictly
    # before t_i = total - cumulative count up to (but not including) i.
    cum_before = np.concatenate([[0.0], np.cumsum(counts)[:-1]])
    n_at = total - cum_before
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(n_at > 0, 1.0 - d / n_at, 1.0)
    S = np.cumprod(factors)
    return uniq, S


def _km_eval(times: np.ndarray, S: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Right-continuous KM survival S(t) evaluated at ``t``.

    Uses ``searchsorted(..., side='right') - 1``; S(t) = 1 for t < min(times).
    """
    t = np.asarray(t, dtype=float)
    idx = np.searchsorted(times, t, side="right") - 1
    out = np.ones_like(t, dtype=float)
    mask = idx >= 0
    out[mask] = S[idx[mask]]
    return out


# ---------------------------------------------------------------------------
# Time-dependent AUC (Uno et al. 2007 IPCW; cumulative/dynamic form).
# ---------------------------------------------------------------------------


def time_dependent_auc(time, event, risk_score, eval_times) -> dict:
    """Cumulative/dynamic time-dependent AUC with IPCW weights.

    Defines cases at time ``tau`` as ``{T_i <= tau, delta_i = 1}`` and
    controls as ``{T_i > tau}`` (Heagerty, Lumley & Pepe 2000).  The IPCW
    estimator of Uno et al. (2007) weights each case by
    ``1 / G(T_i-)``, where ``G`` is the Kaplan-Meier estimate of the
    *censoring* distribution (obtained by flipping the event indicator).

    Parameters
    ----------
    time, event : (n,) observed time and 0/1 event indicators.
    risk_score : (n,) higher values mean higher risk (earlier event).
    eval_times : sequence of tau at which to evaluate AUC(tau).

    Returns
    -------
    dict with ``times``, ``auc`` array and ``integrated_auc`` — a
    time-weighted mean of AUC(tau) over the eval grid (trapezoidal rule,
    normalised by the grid length).
    """
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    s = np.asarray(risk_score, dtype=float).ravel()
    taus = np.asarray(eval_times, dtype=float).ravel()
    if not (t.shape == e.shape == s.shape):
        raise ValueError("time, event, risk_score must be the same length")

    # KM of the censoring distribution: swap event.
    cens = 1 - e
    km_t, km_S = _km_estimator(t, cens)
    # G(T_i -) ~ G at left limit; use right-continuous evaluator on
    # min(t, t - eps).  For practical purposes, G(T_i-) is G at the
    # largest event-time strictly less than T_i.
    t_minus = np.nextafter(t, -np.inf)
    G_t = _km_eval(km_t, km_S, t_minus)
    # Avoid zero weights.
    G_t = np.clip(G_t, _EPS, 1.0)
    w = 1.0 / G_t

    aucs = np.zeros_like(taus)
    for idx, tau in enumerate(taus):
        is_case = (t <= tau) & (e == 1)
        is_ctrl = t > tau
        if not (np.any(is_case) and np.any(is_ctrl)):
            aucs[idx] = float("nan")
            continue
        w_case = w[is_case]
        s_case = s[is_case]
        s_ctrl = s[is_ctrl]
        w_ctrl = np.ones_like(s_ctrl)  # controls are uncensored at tau.

        # Weighted AUC = P(s_i > s_j) + 0.5 P(s_i == s_j), i case, j ctrl.
        # Compute by sorting all scores and accumulating weight of ctrls
        # beaten/tied by each case.
        num = 0.0
        den = 0.0
        # Sort controls by score; walk cases against it.
        ord_c = np.argsort(s_ctrl, kind="mergesort")
        s_c_sorted = s_ctrl[ord_c]
        w_c_sorted = w_ctrl[ord_c]
        W_tot = float(np.sum(w_c_sorted))
        cw = np.concatenate([[0.0], np.cumsum(w_c_sorted)])
        # For each case score x, weight of ctrls with s < x is
        # cw[searchsorted(s_c_sorted, x, 'left')]; weight with s == x is
        # cw[right] - cw[left].
        left = np.searchsorted(s_c_sorted, s_case, side="left")
        right = np.searchsorted(s_c_sorted, s_case, side="right")
        w_lt = cw[left]
        w_eq = cw[right] - cw[left]
        contrib = w_case * (w_lt + 0.5 * w_eq)
        num = float(np.sum(contrib))
        den = float(np.sum(w_case) * W_tot)
        aucs[idx] = num / den if den > 0 else float("nan")

    # Integrated AUC: trapezoidal average over eval_times (time-weighted).
    if taus.size >= 2 and np.all(np.isfinite(aucs)):
        width = taus[-1] - taus[0]
        if width > 0:
            integrated = _trapz(aucs, taus) / width
        else:
            integrated = float(np.mean(aucs))
    elif taus.size == 1 and np.isfinite(aucs[0]):
        integrated = float(aucs[0])
    else:
        integrated = float(np.nanmean(aucs))

    return {
        "times": taus,
        "auc": aucs,
        "integrated_auc": integrated,
    }


# ---------------------------------------------------------------------------
# IPCW Brier score and Integrated Brier Score (Graf et al. 1999).
# ---------------------------------------------------------------------------


def brier_score(time, event, survival_pred, eval_times) -> dict:
    """Time-dependent IPCW Brier score of Graf et al. (1999).

    For a grid of evaluation times ``t*`` the estimator is

        BS(t*) = (1/n) sum_i  [  I(T_i <= t*, delta_i=1) (0 - S_hat_i(t*))^2
                                   / G(T_i -)
                                + I(T_i > t*) (1 - S_hat_i(t*))^2 / G(t*) ],

    where ``G`` is the Kaplan-Meier estimate of the censoring distribution.
    The Integrated Brier Score (IBS) is the trapezoidal integral of
    BS(t*) over ``[0, max(eval_times)]`` normalised by the interval width.
    The Scaled Brier Score ``IBS / IBS_KM`` compares the model to a
    Kaplan-Meier marginal survival baseline (Graf et al. 1999).

    Parameters
    ----------
    time, event : (n,) arrays.
    survival_pred : (n, n_times) predicted S_hat_i(t*) at ``eval_times``.
        Alternatively a callable ``t -> (n,) survival``; the grid form is
        preferred.
    eval_times : (n_times,) grid.
    """
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    taus = np.asarray(eval_times, dtype=float).ravel()
    n = t.size

    S_pred = np.asarray(survival_pred, dtype=float)
    if S_pred.ndim == 1:
        S_pred = S_pred[:, None]
    if S_pred.shape != (n, taus.size):
        raise ValueError(
            f"survival_pred must have shape (n, n_times) = "
            f"({n}, {taus.size}); got {S_pred.shape}"
        )

    # KM of censoring distribution.
    cens = 1 - e
    km_t, km_S = _km_estimator(t, cens)
    t_minus = np.nextafter(t, -np.inf)
    G_at_T = np.clip(_km_eval(km_t, km_S, t_minus), _EPS, 1.0)
    G_at_tau = np.clip(_km_eval(km_t, km_S, taus), _EPS, 1.0)

    # KM baseline (of the event distribution, for the Scaled Brier Score).
    km_e_t, km_e_S = _km_estimator(t, e)
    S_km_at_tau = _km_eval(km_e_t, km_e_S, taus)  # (n_times,)
    S_km_pred = np.broadcast_to(S_km_at_tau, (n, taus.size))

    def _bs_at_grid(S_hat: np.ndarray) -> np.ndarray:
        bs = np.zeros(taus.size, dtype=float)
        for j, tau in enumerate(taus):
            case = (t <= tau) & (e == 1)
            ctrl = t > tau
            term_case = np.where(case, (0.0 - S_hat[:, j]) ** 2 / G_at_T, 0.0)
            term_ctrl = np.where(ctrl, (1.0 - S_hat[:, j]) ** 2 / G_at_tau[j], 0.0)
            bs[j] = float(np.mean(term_case + term_ctrl))
        return bs

    bs = _bs_at_grid(S_pred)
    bs_km = _bs_at_grid(S_km_pred)

    def _integrate(vals: np.ndarray) -> float:
        if taus.size < 2:
            return float(vals[0]) if vals.size else 0.0
        width = taus[-1] - taus[0]
        if width <= 0:
            return float(np.mean(vals))
        return _trapz(vals, taus) / width

    ibs = _integrate(bs)
    ibs_km = _integrate(bs_km)
    scaled = ibs / ibs_km if ibs_km > 0 else float("nan")

    return {
        "times": taus,
        "brier": bs,
        "brier_km": bs_km,
        "ibs": ibs,
        "ibs_km": ibs_km,
        "scaled_brier": scaled,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval (percentile method).
# ---------------------------------------------------------------------------


def _looks_like_event(a: np.ndarray) -> bool:
    if a.ndim != 1:
        return False
    if a.dtype == bool:
        return True
    try:
        u = np.unique(a[~np.isnan(a)]) if a.dtype.kind == "f" else np.unique(a)
    except TypeError:
        u = np.unique(a)
    return u.size <= 2 and set(np.asarray(u).astype(int).tolist()).issubset({0, 1})


def bootstrap_metric(
    fn: Callable,
    *arrays,
    n_boot: int = 500,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Percentile-method nonparametric bootstrap of a scalar statistic.

    Rows are resampled *jointly* across the input arrays (all arrays must
    share their leading dimension).  If the last positional array looks
    like a 0/1 event indicator (dtype bool, or values in {0,1} matching
    the leading length), resampling is stratified on it to preserve the
    event rate across bootstrap replicates.

    Returns
    -------
    dict with ``point`` (statistic on the original data), ``mean``,
    ``lo``, ``hi`` (percentile CI endpoints) and ``samples`` — the
    bootstrap distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    arrs = [np.asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("bootstrap_metric needs at least one array")
    n = arrs[0].shape[0]
    for a in arrs:
        if a.shape[0] != n:
            raise ValueError("all arrays must share the leading dimension")

    stratify = None
    if len(arrs) >= 1 and _looks_like_event(arrs[-1]) and arrs[-1].shape[0] == n:
        ev = np.asarray(arrs[-1]).astype(int)
        if set(np.unique(ev).tolist()) <= {0, 1}:
            stratify = ev

    # Point estimate on the original data.
    point = float(fn(*arrs))

    samples = np.empty(n_boot, dtype=float)
    if stratify is not None:
        idx_pos = np.where(stratify == 1)[0]
        idx_neg = np.where(stratify == 0)[0]
    for b in range(n_boot):
        if stratify is not None and idx_pos.size > 0 and idx_neg.size > 0:
            pick_pos = rng.integers(0, idx_pos.size, size=idx_pos.size)
            pick_neg = rng.integers(0, idx_neg.size, size=idx_neg.size)
            idx = np.concatenate([idx_pos[pick_pos], idx_neg[pick_neg]])
        else:
            idx = rng.integers(0, n, size=n)
        resampled = [a[idx] for a in arrs]
        samples[b] = float(fn(*resampled))

    alpha = 1.0 - ci
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {
        "point": point,
        "mean": float(np.mean(samples)),
        "lo": lo,
        "hi": hi,
        "samples": samples,
    }


__all__ = [
    "nagelkerke_r2",
    "calibration_metrics",
    "time_dependent_auc",
    "brier_score",
    "bootstrap_metric",
]
