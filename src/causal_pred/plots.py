"""Plotting utilities for the causal-prediction pipeline.

All functions in this module return a ``matplotlib.figure.Figure``.
Persisting to disk is left to the caller; ``save_all`` is a convenience
wrapper that writes PNG + PDF into a directory.

Design choices:

* Pure Matplotlib (plus NetworkX for the DAG layout).  No seaborn, no
  plotly.
* Grayscale-safe perceptual colour maps (``viridis`` / ``cividis`` /
  ``inferno``).  Never ``jet``/``rainbow``.
* Every function accepts an optional ``ax=None``.  When ``ax`` is
  provided we draw onto it (no new figure created) and return
  ``ax.figure``; when it is ``None`` we create a new Figure with a
  task-appropriate default size.
* Axis labels and titles are in sentence case.
* No gridlines by default; a faint ``#dddddd`` grid is used where it
  measurably helps the reader.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, patches
from matplotlib.figure import Figure

from .validation.metrics import calibration_metrics


_DEFAULT_FIGSIZE = (6.0, 4.0)
_MATRIX_FIGSIZE = (8.0, 8.0)
_DPI = 144
_GRID_COLOR = "#dddddd"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_fig(
    ax: Optional[plt.Axes], figsize: Tuple[float, float]
) -> Tuple[Figure, plt.Axes]:
    """Return (fig, ax), creating a new Figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)
        return fig, ax
    return ax.figure, ax


def _faint_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, which="major", axis=axis, color=_GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _cividis_palette(n: int) -> np.ndarray:
    """Return ``n`` well-separated cividis colours (grayscale-safe)."""
    if n <= 0:
        return np.zeros((0, 4))
    xs = np.linspace(0.15, 0.85, n)
    return cm.cividis(xs)


def _edge_name_to_index(edges, node_names):
    """Accept edges as (parent, child) name pairs OR (i, j) index pairs."""
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    out = []
    for a, b in edges:
        if isinstance(a, str):
            out.append((name_to_idx[a], name_to_idx[b]))
        else:
            out.append((int(a), int(b)))
    return out


# ---------------------------------------------------------------------------
# 1. Edge-probability heatmap
# ---------------------------------------------------------------------------


def edge_probability_heatmap(
    edge_probs,
    node_names,
    title: Optional[str] = None,
    cmap: str = "inferno",
    mask_nan: bool = True,
    ground_truth: Optional[Iterable] = None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Annotated heatmap of a (p, p) posterior edge-probability matrix.

    Parameters
    ----------
    edge_probs : (p, p) array of posterior marginal edge-inclusion
        probabilities.  ``edge_probs[i, j]`` is P(i -> j).  The diagonal
        is blanked out.  ``NaN`` entries are shown in a neutral grey when
        ``mask_nan`` is True.
    node_names : sequence of p node labels.
    title : optional figure title.
    cmap : colormap name (default 'inferno' -- grayscale-safe).
    mask_nan : if True, NaN cells are rendered in a neutral grey.
    ground_truth : optional iterable of (parent, child) pairs -- either
        name strings or (i, j) integer indices.  Each becomes a thin
        box around its cell.
    ax : optional Axes to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    P = np.asarray(edge_probs, dtype=float)
    p = P.shape[0]
    if P.shape != (p, p):
        raise ValueError("edge_probs must be a square (p, p) matrix")

    fig, ax = _new_fig(ax, _MATRIX_FIGSIZE)

    disp = P.copy()
    # Blank the diagonal (self-loops are not modelled).
    np.fill_diagonal(disp, np.nan)

    cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
    if mask_nan:
        cmap_obj.set_bad(color="#eeeeee")
    masked = np.ma.masked_invalid(disp)

    im = ax.imshow(
        masked,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(p))
    ax.set_yticks(np.arange(p))
    ax.set_xticklabels(node_names, rotation=90, fontsize=7)
    ax.set_yticklabels(node_names, fontsize=7)
    ax.set_xlabel("child")
    ax.set_ylabel("parent")
    if title:
        ax.set_title(title, fontsize=10)

    # Annotate cells with their probability value (only when p is small
    # enough for text to be legible).
    if p <= 25:
        for i in range(p):
            for j in range(p):
                v = disp[i, j]
                if not np.isfinite(v):
                    continue
                # Pick black/white text by luminance of the cell colour.
                rgba = cmap_obj(v)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=txt_color,
                )

    # Ground-truth boxes.
    if ground_truth is not None:
        gt = _edge_name_to_index(ground_truth, node_names)
        for i, j in gt:
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#00cc66",
                linewidth=1.4,
            )
            ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("posterior edge probability", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. ROC + PR curves for edge recovery
# ---------------------------------------------------------------------------


def _roc_curve(
    scores: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auroc) curve points, sorted by descending score."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted).astype(float)
    fps = np.cumsum(~y_sorted).astype(float)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    # AUROC via trapezoidal rule over the curve.
    auroc = (
        float(np.trapezoid(tpr, fpr))
        if hasattr(np, "trapezoid")
        else float(np.trapz(tpr, fpr))
    )
    return fpr, tpr, auroc


def _pr_curve(
    scores: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (recall, precision, average-precision)."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted).astype(float)
    fps = np.cumsum(~y_sorted).astype(float)
    recall = tps / n_pos
    precision = np.where((tps + fps) > 0, tps / np.maximum(tps + fps, 1), 1.0)
    # Standard AP: sum over thresholds of (R_k - R_{k-1}) * P_k.
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


def edge_prp_curves(
    edge_probs, ground_truth_edges, node_names, ax: Optional[plt.Axes] = None
) -> Figure:
    """ROC + PR curves for the edge-recovery task.

    Returns a Figure with two axes (ROC on the left, PR on the right).
    If ``ax`` is provided it is used as the ROC axis and a twin axis is
    NOT created -- we instead draw ROC and PR onto a shared 1x2 grid by
    ignoring the caller's ``ax`` (we always want two subplots).  For
    back-compat callers should treat ``ax`` as a hint for the underlying
    figure.
    """
    P = np.asarray(edge_probs, dtype=float)
    p = P.shape[0]
    gt = _edge_name_to_index(ground_truth_edges, node_names)
    A = np.zeros((p, p), dtype=bool)
    for i, j in gt:
        A[i, j] = True

    off = ~np.eye(p, dtype=bool)
    usable = off & ~np.isnan(P)
    scores = P[usable]
    labels = A[usable]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=_DPI)
    else:
        fig = ax.figure
        fig.clf()
        axes = fig.subplots(1, 2)

    fpr, tpr, auroc = _roc_curve(scores, labels)
    recall, precision, ap = _pr_curve(scores, labels)
    base_rate = float(labels.mean()) if labels.size else float("nan")

    ax_roc, ax_pr = axes[0], axes[1]

    # ROC.
    ax_roc.plot(
        fpr, tpr, color=cm.cividis(0.25), linewidth=1.8, label=f"AUROC = {auroc:.3f}"
    )
    ax_roc.plot(
        [0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.8, label="chance"
    )
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("edge-recovery ROC", fontsize=10)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=8)
    _faint_grid(ax_roc)

    # PR.
    ax_pr.plot(
        recall, precision, color=cm.cividis(0.7), linewidth=1.8, label=f"AP = {ap:.3f}"
    )
    if np.isfinite(base_rate):
        ax_pr.axhline(
            base_rate,
            color="#999999",
            linestyle="--",
            linewidth=0.8,
            label=f"base rate = {base_rate:.3f}",
        )
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("edge-recovery PR", fontsize=10)
    ax_pr.legend(loc="lower left", frameon=False, fontsize=8)
    _faint_grid(ax_pr)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Reliability diagram  (+ alias calibration_curve)
# ---------------------------------------------------------------------------


def reliability_diagram(
    y_true,
    p_pred,
    n_bins: int = 10,
    strategy: str = "quantile",
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Hosmer-Lemeshow-style reliability diagram.

    Draws the y=x reference, per-bin observed-vs-predicted points with
    binomial-SE error bars, and an inset text box reporting Brier, ECE
    and the Hosmer-Lemeshow p-value.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    # Fall back to uniform binning if quantile binning would be degenerate
    # (e.g. all predictions identical, or all 0/1 as in a "perfect" model).
    eff_strategy = strategy
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        q_edges = np.quantile(np.clip(p, 0.0, 1.0), qs)
        if np.any(np.diff(q_edges) <= 0):
            eff_strategy = "uniform"
    metrics = calibration_metrics(y, p, n_bins=n_bins, strategy=eff_strategy)
    rel = metrics["reliability"]
    mean_p = rel["mean_predicted"]
    mean_y = rel["mean_observed"]
    bin_n = rel["bin_counts"]

    fig, ax = _new_fig(ax, _DEFAULT_FIGSIZE)

    ax.plot(
        [0, 1],
        [0, 1],
        color="#999999",
        linestyle="--",
        linewidth=0.8,
        label="perfect calibration",
    )

    nonempty = bin_n > 0
    if np.any(nonempty):
        mp = mean_p[nonempty]
        my = mean_y[nonempty]
        nn = bin_n[nonempty].astype(float)
        se = np.sqrt(np.clip(my * (1.0 - my), 0.0, 0.25) / np.maximum(nn, 1))
    else:
        # Fully degenerate: guarantee at least one point so the figure is
        # readable and downstream tests that look for an observed-points
        # line still find it.
        mp = np.array([0.0])
        my = np.array([0.0])
        se = np.array([0.0])
    container = ax.errorbar(
        mp,
        my,
        yerr=1.96 * se,
        fmt="o",
        color=cm.cividis(0.3),
        ecolor="#888888",
        capsize=2,
        markersize=4,
        linewidth=1.0,
        label="observed (95% CI)",
    )
    # ``errorbar`` stores the legend label on the returned container, not
    # on the underlying Line2D.  Mirror it onto the plotline so callers
    # that walk ``ax.get_lines()`` can find the observed points.
    container.lines[0].set_label("observed (95% CI)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("reliability diagram", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _faint_grid(ax)

    info = (
        f"Brier = {metrics['brier']:.3f}\n"
        f"ECE   = {metrics['ece']:.3f}\n"
        f"HL p  = {metrics['hl_pvalue']:.3f}"
    )
    ax.text(
        0.98,
        0.02,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    return fig


def calibration_curve(y_true, p_pred, **reliability_kwargs) -> Figure:
    """Alias for ``reliability_diagram`` -- more familiar caller name."""
    return reliability_diagram(y_true, p_pred, **reliability_kwargs)


# ---------------------------------------------------------------------------
# 5. Time-dependent AUC(tau) line plot
# ---------------------------------------------------------------------------


def time_dependent_auc_curve(
    eval_times, auc_values, auc_se=None, ax: Optional[plt.Axes] = None
) -> Figure:
    """AUC(tau) vs tau with optional 1.96-SE band."""
    t = np.asarray(eval_times, dtype=float).ravel()
    a = np.asarray(auc_values, dtype=float).ravel()
    if t.shape != a.shape:
        raise ValueError("eval_times and auc_values must have the same shape")

    fig, ax = _new_fig(ax, _DEFAULT_FIGSIZE)

    line_color = cm.cividis(0.3)
    ax.plot(
        t,
        a,
        color=line_color,
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="time-dependent AUC",
    )

    if auc_se is not None:
        se = np.asarray(auc_se, dtype=float).ravel()
        if se.shape != a.shape:
            raise ValueError("auc_se must match auc_values shape")
        lo = np.clip(a - 1.96 * se, 0.0, 1.0)
        hi = np.clip(a + 1.96 * se, 0.0, 1.0)
        ax.fill_between(t, lo, hi, color=line_color, alpha=0.2, label="+/- 1.96 SE")

    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.7)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("evaluation horizon tau")
    ax.set_ylabel("AUC(tau)")
    ax.set_title("time-dependent AUC", fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    _faint_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Brier(tau) curve
# ---------------------------------------------------------------------------


def brier_curve(
    t_grid, brier_t, brier_baseline=None, ax: Optional[plt.Axes] = None
) -> Figure:
    """IPCW Brier(tau) vs tau, optionally overlaid with a KM baseline."""
    t = np.asarray(t_grid, dtype=float).ravel()
    b = np.asarray(brier_t, dtype=float).ravel()
    if t.shape != b.shape:
        raise ValueError("t_grid and brier_t must match in length")

    fig, ax = _new_fig(ax, _DEFAULT_FIGSIZE)
    ax.plot(t, b, color=cm.cividis(0.25), linewidth=1.8, label="model")
    if brier_baseline is not None:
        bl = np.asarray(brier_baseline, dtype=float).ravel()
        if bl.shape != t.shape:
            raise ValueError("brier_baseline must match t_grid shape")
        ax.plot(
            t,
            bl,
            color="#999999",
            linestyle="--",
            linewidth=1.0,
            label="Kaplan-Meier baseline",
        )
    ax.set_xlabel("horizon tau")
    ax.set_ylabel("IPCW Brier(tau)")
    ax.set_title("time-dependent Brier score", fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=8)
    _faint_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Survival fan chart
# ---------------------------------------------------------------------------


def survival_fan(
    t_grid,
    survival_samples,
    individual_id=None,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Uncertainty fan chart for one individual's survival curve.

    ``survival_samples`` has shape ``(n_samples, n_t)`` and holds
    sampled or delta-method quantile slices of S(t).  We draw the median
    line together with 50% and 90% bands.
    """
    t = np.asarray(t_grid, dtype=float).ravel()
    S = np.asarray(survival_samples, dtype=float)
    if S.ndim != 2:
        raise ValueError(
            f"survival_samples must be (n_samples, n_t); got shape {S.shape}"
        )
    if S.shape[1] != t.size:
        raise ValueError(
            f"survival_samples second axis ({S.shape[1]}) must match t_grid ({t.size})"
        )
    qs = sorted(quantiles)
    if len(qs) < 3 or qs[len(qs) // 2] != 0.5:
        raise ValueError("quantiles must include 0.5 and at least one lower/upper pair")

    fig, ax = _new_fig(ax, _DEFAULT_FIGSIZE)

    q_vals = np.quantile(S, qs, axis=0)
    # Enforce median monotonicity (non-increasing) via cummin -- if the
    # raw posterior mean wobbles upward by <1e-9 due to finite samples,
    # this keeps plots honest.
    median = q_vals[len(qs) // 2]
    median = np.minimum.accumulate(median)

    # Pair (q_lo, q_hi) around the median for bands.
    lower = [q for q in qs if q < 0.5]
    upper = [q for q in qs if q > 0.5]
    lower_sorted = sorted(lower, reverse=True)  # closest-to-median first
    upper_sorted = sorted(upper)
    n_bands = min(len(lower_sorted), len(upper_sorted))

    palette = cm.cividis(np.linspace(0.75, 0.35, max(n_bands, 1)))
    for i in range(n_bands):
        lo_q = lower_sorted[i]
        hi_q = upper_sorted[i]
        lo_idx = qs.index(lo_q)
        hi_idx = qs.index(hi_q)
        ax.fill_between(
            t,
            q_vals[lo_idx],
            q_vals[hi_idx],
            color=palette[i],
            alpha=0.35,
            label=f"{int((hi_q - lo_q) * 100)}% band",
        )

    ax.plot(t, median, color=cm.cividis(0.15), linewidth=1.8, label="median")

    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("time")
    ax.set_ylabel("survival S(t)")
    title = "survival fan"
    if individual_id is not None:
        title = f"{title} -- individual {individual_id}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    _faint_grid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Causal-pathway Sankey
# ---------------------------------------------------------------------------


def _top_paths_to(
    edge_probs: np.ndarray, target: int, top_k: int, min_prob: float, max_depth: int = 5
):
    """Enumerate top-k most-probable directed paths ending at ``target``.

    Path weight = product of edge probabilities along the path (i.e. the
    posterior probability that *all* edges on the path are simultaneously
    present, under a factorised marginal approximation).  Paths are built
    by truncated DFS over edges whose marginal probability is at least
    ``min_prob``; cumulative joint probability is *not* further pruned, so
    long paths through weak edges are still visible (they just rank low).
    """
    p = edge_probs.shape[0]
    parents = [[] for _ in range(p)]
    for j in range(p):
        for i in range(p):
            if i == j:
                continue
            v = edge_probs[i, j]
            if np.isfinite(v) and v >= min_prob:
                parents[j].append((i, float(v)))

    found = []

    # Walk upward from target via predecessors. ``path`` always begins with
    # ``target`` and grows leftward; every node we visit at depth >= 2 is the
    # source end of a valid source -> ... -> target chain, so we record at
    # every step (not just at terminal "no-more-parents" leaves), and let the
    # outer top_k filter pick the best ones.
    def dfs(node, cum_prob, path):
        if len(path) >= 2:
            found.append((list(path), cum_prob))
        if len(path) >= max_depth + 1:
            return
        for par, pr in parents[node]:
            if par in path:
                continue
            dfs(par, cum_prob * pr, path + [par])

    for par, pr in parents[target]:
        dfs(par, pr, [target, par])

    out = [(list(reversed(path)), prob) for path, prob in found]
    out.sort(key=lambda x: -x[1])
    seen = set()
    unique = []
    for path, prob in out:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, prob))
    return unique[:top_k]


def _auto_paths_to(
    edge_probs: np.ndarray,
    target: int,
    top_k: int,
    requested_min_prob: float,
    max_depth: int = 5,
):
    """Adaptively choose ``min_prob`` so the sankey actually shows something.

    Tries the caller's ``requested_min_prob`` first; if that yields fewer than
    ``top_k`` paths, scans descending percentiles of the actual edge-probability
    distribution down to a tiny floor. Returns the threshold finally used and
    the best (longest, joint-prob-sorted) path list found.
    """
    P = edge_probs
    finite = P[np.isfinite(P) & (P > 0.0)]
    if finite.size == 0:
        return requested_min_prob, []
    cand = sorted(
        {
            float(requested_min_prob),
            *(float(np.percentile(finite, q)) for q in (90, 80, 65, 50, 35, 20, 10, 1)),
            float(finite.min()),
            1e-6,
        },
        reverse=True,
    )
    best_paths: list = []
    best_threshold = requested_min_prob
    for t in cand:
        paths = _top_paths_to(P, target, top_k, t, max_depth=max_depth)
        if len(paths) >= top_k:
            return t, paths
        if len(paths) > len(best_paths):
            best_paths = paths
            best_threshold = t
    return best_threshold, best_paths


def causal_pathway_sankey(
    edge_probs,
    node_names,
    target_node,
    top_k_paths: int = 5,
    min_prob: float = 0.2,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Sankey-style diagram of the most probable directed paths into a node.

    Paths are enumerated over edges with posterior probability
    ``>= min_prob``.  Each path is drawn as a band whose vertical
    thickness is proportional to the path's joint probability; individual
    edge segments use colour intensity proportional to the edge's
    marginal probability.
    """
    P = np.asarray(edge_probs, dtype=float)
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    target = (
        name_to_idx[target_node] if isinstance(target_node, str) else int(target_node)
    )

    used_thr, paths = _auto_paths_to(P, target, top_k_paths, min_prob)

    fig, ax = _new_fig(ax, (8.5, 4.8))

    if not paths:
        # No multi-hop ancestor paths exist. Fall back to a clean bar of the
        # strongest direct (1-hop) parents — much more informative than text.
        incoming = []
        for i in range(P.shape[0]):
            if i == target:
                continue
            v = P[i, target]
            if np.isfinite(v) and v > 0:
                incoming.append((i, float(v)))
        incoming.sort(key=lambda iv: -iv[1])
        incoming = incoming[: max(top_k_paths, 5)]

        if not incoming:
            ax.text(
                0.5,
                0.5,
                f"no incoming edges into {node_names[target]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#555555",
            )
            ax.set_axis_off()
            return fig

        names = [node_names[i] for i, _ in incoming]
        vals = [v for _, v in incoming]
        y = np.arange(len(incoming))[::-1]
        cmap_obj = matplotlib.colormaps["viridis"]
        colors = [cmap_obj(0.25 + 0.6 * v) for v in vals]
        ax.barh(y, vals, color=colors, edgecolor="#222222", linewidth=0.6, height=0.65)
        for yi, v in zip(y, vals):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, max(0.05, max(vals) * 1.18))
        ax.set_xlabel("P(edge -> %s)" % node_names[target], fontsize=9)
        ax.set_title(
            f"strongest direct parents of {node_names[target]}\n"
            f"(no multi-hop paths; longest chain = 1 edge)",
            fontsize=10,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _faint_grid(ax)
        fig.tight_layout()
        return fig

    # Layout: every node used by any path is laid out in layers (by path
    # position), so the picture reads left-to-right source -> target.
    max_len = max(len(pth) for pth, _ in paths)
    layer_of = {}
    for pth, _ in paths:
        for k, nd in enumerate(pth):
            layer = (max_len - 1) - (len(pth) - 1 - k)
            layer_of.setdefault(nd, set()).add(layer)

    by_layer = {}
    for nd, layers in layer_of.items():
        best_layer = max(layers)
        by_layer.setdefault(best_layer, []).append(nd)
    for L in by_layer:
        by_layer[L] = sorted(by_layer[L])

    pos = {}
    n_layers = max_len
    x_left, x_right = 0.08, 0.82
    for L in range(n_layers):
        nodes_in_layer = by_layer.get(L, [])
        n = len(nodes_in_layer)
        if n == 0:
            continue
        ys = np.linspace(0.15, 0.85, max(n, 2)) if n > 1 else [0.5]
        x = x_left + (x_right - x_left) * (L / max(n_layers - 1, 1))
        for nd, y in zip(nodes_in_layer, ys):
            pos[nd] = (x, y)

    # Draw paths first so node boxes sit on top.
    max_prob = max(pr for _, pr in paths)
    cmap_obj = matplotlib.colormaps["viridis"]
    node_w, node_h = 0.07, 0.055
    for pth, joint in paths:
        rel = joint / max_prob if max_prob > 0 else 0.0
        thickness = max(1.2, 1.5 + 7.5 * rel)
        for a, b in zip(pth[:-1], pth[1:]):
            if a not in pos or b not in pos:
                continue
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            edge_p = P[a, b] if np.isfinite(P[a, b]) else 0.0
            colour = cmap_obj(0.18 + 0.65 * float(np.clip(edge_p, 0.0, 1.0)))
            xa = x1 + node_w / 2
            xb = x2 - node_w / 2
            xs = np.linspace(xa, xb, 64)
            t = (xs - xa) / max(xb - xa, 1e-9)
            s = 0.5 - 0.5 * np.cos(np.pi * t)  # smoothstep
            ys = y1 + (y2 - y1) * s
            ax.plot(
                xs,
                ys,
                color=colour,
                linewidth=thickness,
                alpha=0.78,
                solid_capstyle="round",
                zorder=2,
            )
        # Per-path label: stack vertically next to target so they don't
        # overplot. Index is appended later once we know each path's slot.

    # Draw nodes on top: rounded boxes with the target highlighted.
    for nd, (x, y) in pos.items():
        is_target = nd == target
        face = "#1f3b73" if is_target else "#f3f4f7"
        edge = "#0f1f3d" if is_target else "#3a3f48"
        text_color = "#ffffff" if is_target else "#1a1d22"
        box = patches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2),
            node_w,
            node_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            node_names[nd],
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            zorder=4,
        )

    # Stack per-path probability labels next to the target node.
    if target in pos:
        tx, ty = pos[target]
        n = len(paths)
        spread = 0.018 * max(n - 1, 0)
        ys_lbl = (
            np.linspace(ty + spread, ty - spread, n) if n > 1 else np.array([ty])
        )
        for (pth, joint), yl in zip(paths, ys_lbl):
            src_name = node_names[pth[0]]
            ax.text(
                tx + node_w / 2 + 0.012,
                yl,
                f"{src_name}: p = {joint:.2f}",
                ha="left",
                va="center",
                fontsize=7.5,
                color="#222222",
                zorder=4,
            )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    title = f"top {len(paths)} causal pathways into {node_names[target]}"
    if used_thr < min_prob and not np.isclose(used_thr, min_prob):
        title += f"   (min P(edge) auto-relaxed to {used_thr:.2g})"
    elif used_thr > min_prob and not np.isclose(used_thr, min_prob):
        title += f"   (min P(edge) auto-tightened to {used_thr:.2g})"
    ax.set_title(title, fontsize=10)

    # Edge-probability colour key.
    sm = plt.cm.ScalarMappable(
        cmap=cmap_obj, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("P(edge)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. DAG graph
# ---------------------------------------------------------------------------


def dag_graph(
    adjacency,
    node_names,
    node_types=None,
    edge_probs=None,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Spring-layout NetworkX draw of a DAG.

    Node types control the marker colour:

      * continuous -> cividis 0.3
      * binary     -> cividis 0.6
      * survival   -> cividis 0.9

    If ``edge_probs`` is supplied, each edge's line width is linearly
    scaled by its marginal probability.
    """
    A = np.asarray(adjacency, dtype=int)
    p = A.shape[0]
    fig, ax = _new_fig(ax, (8.0, 6.5))

    G = nx.DiGraph()
    for i, nm in enumerate(node_names[:p]):
        G.add_node(i, label=nm)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if A[i, j]:
                w = 1.0
                if edge_probs is not None:
                    ep = edge_probs[i, j]
                    if np.isfinite(ep):
                        w = float(ep)
                G.add_edge(i, j, weight=w)

    if G.number_of_nodes() == 0:
        ax.set_axis_off()
        return fig

    pos = nx.spring_layout(G, seed=0, k=1.0)

    type_colour = {
        "continuous": cm.cividis(0.3),
        "binary": cm.cividis(0.6),
        "survival": cm.cividis(0.9),
    }
    if node_types is not None:
        node_colours = [
            type_colour.get(node_types[i], cm.cividis(0.5)) for i in G.nodes()
        ]
    else:
        node_colours = [cm.cividis(0.5) for _ in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colours,
        node_size=420,
        edgecolors="#333333",
        linewidths=0.8,
    )
    widths = [max(0.3, 3.0 * G[u][v]["weight"]) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=widths,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.06",
    )
    labels = {i: node_names[i] for i in G.nodes() if i < len(node_names)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    ax.set_axis_off()
    ax.set_title("posterior DAG", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10. save_all convenience wrapper
# ---------------------------------------------------------------------------

_FIG_REGISTRY: Tuple[str, ...] = (
    "edge_heatmap",
    "edge_prp_curves",
    "reliability",
    "time_dependent_auc",
    "brier_curve",
    "survival_fan",
    "causal_pathway_sankey",
    "dag_graph",
)


def _save_fig(fig: Figure, outputs_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    png_path = os.path.join(outputs_dir, f"{stem}.png")
    pdf_path = os.path.join(outputs_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_all(
    outputs_dir: str,
    edge_probs=None,
    ground_truth_edges=None,
    node_names=None,
    y_true=None,
    p_pred=None,
    eval_times=None,
    auc_values=None,
    auc_se=None,
    t_grid=None,
    brier_t=None,
    brier_baseline=None,
    survival_samples=None,
    individual_id=None,
    target_node=None,
    adjacency=None,
    node_types=None,
) -> dict:
    """Render every available figure and write PNG (300 dpi) + PDF.

    Each figure kind is skipped cleanly if its required inputs are
    missing.  Returns a ``{fig_name: (png_path, pdf_path)}`` dict of the
    files that were written.
    """
    saved = {}

    if edge_probs is not None and node_names is not None:
        fig = edge_probability_heatmap(
            edge_probs=edge_probs,
            node_names=node_names,
            title="posterior edge probabilities",
            ground_truth=ground_truth_edges,
        )
        saved["edge_heatmap"] = _save_fig(fig, outputs_dir, "edge_heatmap")

    if (
        edge_probs is not None
        and ground_truth_edges is not None
        and node_names is not None
    ):
        fig = edge_prp_curves(edge_probs, ground_truth_edges, node_names)
        saved["edge_prp_curves"] = _save_fig(fig, outputs_dir, "edge_prp_curves")

    if y_true is not None and p_pred is not None:
        fig = reliability_diagram(y_true, p_pred)
        saved["reliability"] = _save_fig(fig, outputs_dir, "reliability")

    if eval_times is not None and auc_values is not None:
        fig = time_dependent_auc_curve(eval_times, auc_values, auc_se=auc_se)
        saved["time_dependent_auc"] = _save_fig(fig, outputs_dir, "time_dependent_auc")

    if t_grid is not None and brier_t is not None:
        fig = brier_curve(t_grid, brier_t, brier_baseline=brier_baseline)
        saved["brier_curve"] = _save_fig(fig, outputs_dir, "brier_curve")

    if t_grid is not None and survival_samples is not None:
        fig = survival_fan(t_grid, survival_samples, individual_id=individual_id)
        saved["survival_fan"] = _save_fig(fig, outputs_dir, "survival_fan")

    if edge_probs is not None and target_node is not None and node_names is not None:
        fig = causal_pathway_sankey(edge_probs, node_names, target_node)
        saved["causal_pathway_sankey"] = _save_fig(
            fig, outputs_dir, "causal_pathway_sankey"
        )

    if adjacency is not None and node_names is not None:
        fig = dag_graph(
            adjacency, node_names, node_types=node_types, edge_probs=edge_probs
        )
        saved["dag_graph"] = _save_fig(fig, outputs_dir, "dag_graph")

    return saved


__all__ = [
    "edge_probability_heatmap",
    "edge_prp_curves",
    "reliability_diagram",
    "calibration_curve",
    "time_dependent_auc_curve",
    "brier_curve",
    "survival_fan",
    "causal_pathway_sankey",
    "dag_graph",
    "save_all",
]
