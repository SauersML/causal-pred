"""Designed plots for the TopK crosscoder.

These plots cover three audiences in one module:

* **Mech-interp**: feature-share distribution, decoder-norm geometry, training
  health curves, activation density, co-activation similarity.
* **Bio**: a signed decoder-weight heatmap over the full PRS + EHR vocabulary,
  per-feature "dossier" cards showing the strongest PRS / EHR loadings, and
  per-column reconstruction R^2.
* **Selection**: the cross-modal eligibility band and which features were
  promoted to DAG nodes.

Design choices, applied uniformly:

* Pure Matplotlib. Two streams have a fixed colour identity: the *genome*
  side is always navy, the *EHR* side is always amber, *cross-modal* is
  teal, and *promoted* features are a crimson highlight. Dead features are
  a warm light grey. Diverging weights use ``RdBu_r``.
* Each function returns a ``Figure`` and may also be called for the side
  effect via :func:`save_all_genscore_plots`.
* Inputs that an individual plot does not need are optional. The
  top-level saver simply skips plots whose inputs are missing rather
  than raising.
* Outputs are written as PNG (300 dpi, for the report) and PDF (vector,
  for downstream editing).

Public surface:

* :func:`save_all_genscore_plots` -- one call, writes everything available.
* Individual plot functions, each ``-> matplotlib.figure.Figure``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from .crosscoder import (
    BANK_EHR_PRIVATE,
    BANK_GENOME_PRIVATE,
    BANK_SHARED,
    TopKCrosscoder,
    _ehr_kind_masks,
    _ehr_recon_per_column,
    encode,
    feature_stream_share,
)
from .integrate import AlignedPanels, FeatureSelection


# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------

GENOME_COLOR = "#1F2D5C"      # deep navy-purple -- "genome" stream
EHR_COLOR = "#D6932A"         # warm amber       -- "EHR" stream
CROSSMODAL_COLOR = "#3E8C8E"  # teal             -- shared / cross-modal
PROMOTED_COLOR = "#A4243B"    # crimson          -- promoted highlight
DEAD_COLOR = "#C9C7C2"        # warm light grey  -- inactive features
BAND_COLOR = "#F2E9D8"        # warm cream       -- promotion band shade
BAND_EDGE = "#D9CDB1"
AXIS_COLOR = "#2E2E2E"
TEXT_COLOR = "#1B1B1B"
GRID_COLOR = "#E2E0DA"
DIVERGING_CMAP = "RdBu_r"

# Colours for EHR feature kinds, used to band the EHR side of the decoder
# heatmap. Values not present in this map fall back to EHR_COLOR.
EHR_KIND_COLORS: Dict[str, str] = {
    "condition": "#7B4F94",
    "drug": "#A4243B",
    "lab_mean": "#3E8C8E",
    "lab_min": "#2A6566",
    "lab_max": "#5BAFAF",
    "lab_slope": "#86C0BF",
    "utilisation": "#5C6B73",
}

_DPI = 144
_SAVE_DPI = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_axes(figsize: Tuple[float, float]) -> Tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI, constrained_layout=True)
    _style_axes(ax)
    return fig, ax


def _style_axes(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=AXIS_COLOR, length=3.0, width=0.8, labelsize=9)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def _faint_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(
        True, which="major", axis=axis,
        color=GRID_COLOR, linewidth=0.6, zorder=0,
    )
    ax.set_axisbelow(True)


def _classify_features(
    r_g: np.ndarray,
    rate: np.ndarray,
    band: Tuple[float, float],
    dead_threshold: float = 0.0,
) -> np.ndarray:
    """Return a per-feature class label.

    Classes: ``"dead"``, ``"genome"``, ``"ehr"``, ``"cross"``.
    """
    out = np.full(r_g.shape, "cross", dtype=object)
    out[r_g > band[1]] = "genome"
    out[r_g < band[0]] = "ehr"
    out[rate <= dead_threshold] = "dead"
    return out


def _color_for_class(cls: str) -> str:
    return {
        "genome": GENOME_COLOR,
        "ehr": EHR_COLOR,
        "cross": CROSSMODAL_COLOR,
        "dead": DEAD_COLOR,
    }[cls]


def _short(name: str, n: int = 22) -> str:
    if len(name) <= n:
        return name
    return name[: n - 1] + "…"


def _save(fig: Figure, outputs_dir: str, stem: str) -> Tuple[str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    png = os.path.join(outputs_dir, f"{stem}.png")
    pdf = os.path.join(outputs_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=_SAVE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# 1. Genome-share distribution
# ---------------------------------------------------------------------------


def genome_share_distribution(
    r_g: np.ndarray,
    *,
    activation_rate: Optional[np.ndarray] = None,
    band: Tuple[float, float] = (0.2, 0.8),
    promoted_r_g: Optional[np.ndarray] = None,
    bins: int = 60,
) -> Figure:
    """Histogram of per-feature genome share, coloured by region.

    Region colours:

    * ``r_G > band[1]`` -- **genome-only** features (navy)
    * ``r_G < band[0]`` -- **EHR-only** features (amber)
    * otherwise        -- **cross-modal** features (teal), shaded band

    If ``promoted_r_g`` is provided, promoted features are marked as
    crimson ticks above the histogram. If ``activation_rate`` is given,
    fully-dead features (``rate == 0``) are split off into a small grey
    rug under the x-axis with a count annotation.
    """
    r_g = np.asarray(r_g, dtype=float)
    fig, ax = _new_axes((7.5, 4.4))

    alive = (
        np.ones_like(r_g, dtype=bool)
        if activation_rate is None
        else np.asarray(activation_rate) > 0
    )
    n_dead = int((~alive).sum())

    # ---- shaded promotion band -------------------------------------------
    ax.axvspan(band[0], band[1], color=BAND_COLOR, alpha=0.65, zorder=0)
    ax.axvline(band[0], color=BAND_EDGE, linewidth=0.8, zorder=1)
    ax.axvline(band[1], color=BAND_EDGE, linewidth=0.8, zorder=1)

    # ---- histogram per region (alive features only) ----------------------
    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts_total, _ = np.histogram(r_g[alive], bins=edges)

    cls = _classify_features(
        r_g, np.asarray(activation_rate) if activation_rate is not None
        else np.ones_like(r_g),
        band,
    )
    for label, color in [
        ("genome", GENOME_COLOR),
        ("cross", CROSSMODAL_COLOR),
        ("ehr", EHR_COLOR),
    ]:
        sel = (cls == label) & alive
        if not np.any(sel):
            continue
        cnt, _ = np.histogram(r_g[sel], bins=edges)
        ax.bar(
            centres, cnt,
            width=(1.0 / bins) * 0.95,
            color=color, edgecolor="white", linewidth=0.3,
            label=_class_legend_label(label),
            zorder=2,
        )

    # ---- promoted feature ticks above the bars ---------------------------
    if promoted_r_g is not None and len(promoted_r_g) > 0:
        ymax = max(counts_total.max(), 1)
        tick_y = ymax * 1.08
        ax.scatter(
            promoted_r_g, np.full(len(promoted_r_g), tick_y),
            marker="|", s=90, color=PROMOTED_COLOR, linewidths=1.2,
            label=f"promoted ({len(promoted_r_g)})",
            zorder=4, clip_on=False,
        )
        ax.set_ylim(0, ymax * 1.15)

    # ---- annotations ------------------------------------------------------
    n_genome = int(np.sum((cls == "genome") & alive))
    n_cross = int(np.sum((cls == "cross") & alive))
    n_ehr = int(np.sum((cls == "ehr") & alive))
    summary = (
        f"alive  genome-only {n_genome}   cross-modal {n_cross}   "
        f"EHR-only {n_ehr}"
        + (f"     dead {n_dead}" if n_dead else "")
    )
    ax.text(
        0.5, 1.02, summary,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color=TEXT_COLOR,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("genome share  $r_G[j] = \\|W^G_d[j]\\|^2$")
    ax.set_ylabel("feature count")
    ax.set_title("Crosscoder feature stream share", loc="left",
                 fontsize=12, fontweight="bold", pad=14)
    _faint_grid(ax, axis="y")

    # legend below the title, no border
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=4, frameon=False, fontsize=8.5,
    )
    return fig


def _class_legend_label(cls: str) -> str:
    return {
        "genome": "genome-only",
        "ehr": "EHR-only",
        "cross": "cross-modal",
        "dead": "dead",
    }[cls]


# ---------------------------------------------------------------------------
# 2. Decoder-norm scatter
# ---------------------------------------------------------------------------


def decoder_norm_scatter(
    model: TopKCrosscoder,
    *,
    activation_rate: Optional[np.ndarray] = None,
    promoted_indices: Optional[np.ndarray] = None,
    band: Tuple[float, float] = (0.2, 0.8),
) -> Figure:
    """Per-feature scatter of (||W^G_d[j]||, ||W^E_d[j]||).

    The joint unit-norm constraint puts every feature exactly on the
    quarter-circle of radius 1 in the first quadrant; how a feature sits
    along that arc is its genome share. Promoted features get a crimson
    ring; dead features are unfilled.
    """
    n_g = np.linalg.norm(model.W_d_G, axis=1)
    n_e = np.linalg.norm(model.W_d_E, axis=1)

    if activation_rate is None:
        rate = np.ones_like(n_g)
    else:
        rate = np.asarray(activation_rate)

    r_g = feature_stream_share(model)
    cls = _classify_features(r_g, rate, band)

    fig, ax = _new_axes((6.2, 6.0))

    # unit-norm arc
    theta = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(theta), np.sin(theta),
            linewidth=1.0, color=AXIS_COLOR, alpha=0.45,
            linestyle="--", zorder=2)
    ax.plot([0, 1.0], [0, 1.0],
            linewidth=0.8, color=AXIS_COLOR, alpha=0.18,
            linestyle=":", zorder=1)

    # band as two dashed rays from origin (boundary of cross-modal)
    for thresh, color in [(band[0], EHR_COLOR), (band[1], GENOME_COLOR)]:
        # r_G = thresh  =>  n_g^2 = thresh, n_e^2 = 1 - thresh
        x, y = math.sqrt(thresh), math.sqrt(1.0 - thresh)
        ax.plot([0, x * 1.04], [0, y * 1.04],
                linewidth=0.7, color=color, alpha=0.4, zorder=1)

    # plot per class
    for label in ("genome", "cross", "ehr", "dead"):
        sel = cls == label
        if not np.any(sel):
            continue
        face = _color_for_class(label)
        edge = "white" if label != "dead" else DEAD_COLOR
        alpha = 0.85 if label != "dead" else 0.6
        ax.scatter(
            n_g[sel], n_e[sel],
            s=18, c=face, edgecolors=edge, linewidths=0.4,
            alpha=alpha, label=_class_legend_label(label),
            zorder=3,
        )

    # promoted highlight
    if promoted_indices is not None and len(promoted_indices) > 0:
        idx = np.asarray(promoted_indices, dtype=int)
        ax.scatter(
            n_g[idx], n_e[idx],
            s=90, facecolors="none", edgecolors=PROMOTED_COLOR,
            linewidths=1.6, label="promoted", zorder=4,
        )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("genome decoder norm  $\\|W^G_d[j]\\|$")
    ax.set_ylabel("EHR decoder norm  $\\|W^E_d[j]\\|$")
    ax.set_title("Per-feature decoder geometry", loc="left",
                 fontsize=12, fontweight="bold", pad=12)
    _faint_grid(ax)

    # corner labels
    ax.text(0.97, 0.04, "genome-only", color=GENOME_COLOR,
            ha="right", va="bottom", fontsize=8.5)
    ax.text(0.04, 0.97, "EHR-only", color=EHR_COLOR,
            ha="left", va="top", fontsize=8.5)
    # cross-modal label near 45 degrees on the arc
    cross_x, cross_y = math.cos(math.pi / 4), math.sin(math.pi / 4)
    ax.text(cross_x + 0.02, cross_y + 0.02, "cross-modal",
            color=CROSSMODAL_COLOR, ha="left", va="bottom", fontsize=8.5)

    ax.legend(
        loc="upper right", frameon=False, fontsize=8.5,
        labelcolor=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Training dynamics
# ---------------------------------------------------------------------------


def training_dynamics(
    history: Mapping[str, Sequence[float]],
    *,
    k: Optional[int] = None,
    d: Optional[int] = None,
) -> Figure:
    """Four-panel training health figure from ``model.history``.

    Panels:

    * (A) Reconstruction loss (main, AuxK) on log-y.
    * (B) Fraction of dead features over training.
    * (C) Active features per minibatch (target = ``k / d`` if known).
    * (D) Cumulative ever-active feature count (saturation toward ``d``).
    """
    step = np.asarray(history.get("step", []), dtype=float)
    loss_main = np.asarray(history.get("loss_main", []), dtype=float)
    loss_val = np.asarray(history.get("loss_val", []), dtype=float)
    loss_cross = np.asarray(history.get("loss_cross", []), dtype=float)
    loss_aux = np.asarray(history.get("loss_aux", []), dtype=float)
    frac_dead = np.asarray(history.get("frac_dead", []), dtype=float)
    frac_active = np.asarray(
        history.get("avg_l0_batch", history.get("frac_active_batch", [])),
        dtype=float,
    )
    ever_active = np.asarray(history.get("ever_active_count", []), dtype=float)

    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.4), dpi=_DPI,
        sharex=True, constrained_layout=True,
    )
    for ax in axes.ravel():
        _style_axes(ax)
        _faint_grid(ax)

    # (A) Loss
    ax = axes[0, 0]
    if step.size:
        ax.semilogy(step, loss_main, color=GENOME_COLOR, linewidth=1.6,
                    label="main")
        if loss_val.size == step.size:
            ax.semilogy(step, np.maximum(loss_val, 1e-12),
                        color=CROSSMODAL_COLOR, linewidth=1.2,
                        label="validation")
        if loss_cross.size == step.size:
            ax.semilogy(step, np.maximum(loss_cross, 1e-12),
                        color=PROMOTED_COLOR, linewidth=1.1,
                        linestyle="-.", label="cross")
        if np.any(loss_aux > 0):
            ax.semilogy(step, np.maximum(loss_aux, 1e-12),
                        color=EHR_COLOR, linewidth=1.2,
                        linestyle="--", label="AuxK")
        ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_ylabel("loss")
    ax.set_title("A. Reconstruction loss", loc="left",
                 fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    # (B) Frac dead
    ax = axes[0, 1]
    if step.size:
        ax.plot(step, frac_dead, color=PROMOTED_COLOR, linewidth=1.6)
        ax.fill_between(step, 0, frac_dead, color=PROMOTED_COLOR, alpha=0.12)
    ax.set_ylim(0.0, max(0.05, float(frac_dead.max() if frac_dead.size else 0.05)) * 1.1)
    ax.set_ylabel("dead fraction")
    ax.set_title("B. Dead features", loc="left",
                 fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    # (C) Active features per batch
    ax = axes[1, 0]
    if step.size:
        ax.plot(step, frac_active, color=CROSSMODAL_COLOR, linewidth=1.6)
        ax.fill_between(step, 0, frac_active,
                        color=CROSSMODAL_COLOR, alpha=0.12)
    if k is not None and d is not None and d > 0:
        target = float(k)
        ax.axhline(target, color=AXIS_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.text(
            ax.get_xlim()[1] if step.size else 1.0,
            target, f"  BatchTopK avg k = {target:.0f}",
            ha="left", va="center", fontsize=8.5, color=AXIS_COLOR,
            transform=ax.transData,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("average L0 per row")
    ax.set_title("C. Sparsity", loc="left",
                 fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    # (D) Ever-active count
    ax = axes[1, 1]
    if step.size:
        ax.plot(step, ever_active, color=GENOME_COLOR, linewidth=1.6)
        ax.fill_between(step, 0, ever_active,
                        color=GENOME_COLOR, alpha=0.10)
    if d is not None:
        ax.axhline(d, color=AXIS_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.text(
            ax.get_xlim()[1] if step.size else 1.0,
            d, f"  d = {d}",
            ha="left", va="center", fontsize=8.5, color=AXIS_COLOR,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("ever-active features")
    ax.set_title("D. Cumulative coverage", loc="left",
                 fontsize=10.5, fontweight="bold", color=TEXT_COLOR)

    fig.suptitle(
        "Crosscoder training dynamics",
        x=0.01, y=1.02, ha="left",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Selection scatter
# ---------------------------------------------------------------------------


def selection_scatter(
    r_g: np.ndarray,
    activation_rate: np.ndarray,
    *,
    promoted_indices: Optional[np.ndarray] = None,
    band: Tuple[float, float] = (0.2, 0.8),
    min_activation_rate: float = 0.01,
) -> Figure:
    """Scatter of every feature in ``(r_G, activation_rate)`` space.

    The eligibility region is the rectangle where the band on r_G meets
    the floor on activation rate; promoted features are highlighted.
    Background contours show the selection score
    ``r_G * (1 - r_G) * activation_rate`` so the reader can see why one
    feature is preferred over another.
    """
    r_g = np.asarray(r_g, dtype=float)
    rate = np.asarray(activation_rate, dtype=float)

    fig, ax = _new_axes((7.8, 5.0))

    # background score isolines
    xs = np.linspace(0.0, 1.0, 200)
    ys = np.geomspace(max(rate[rate > 0].min() if np.any(rate > 0) else 1e-4, 1e-4),
                      max(rate.max(), 1e-3), 200)
    XS, YS = np.meshgrid(xs, ys)
    SC = XS * (1.0 - XS) * YS
    cs = ax.contour(
        XS, YS, SC,
        levels=8, colors=AXIS_COLOR, linewidths=0.5, alpha=0.18,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g", colors=AXIS_COLOR)

    # eligibility region
    rate_top = max(rate.max(), 1.0) * 1.2
    ax.add_patch(
        patches.Rectangle(
            (band[0], min_activation_rate),
            band[1] - band[0], rate_top,
            facecolor=BAND_COLOR, edgecolor=BAND_EDGE,
            linewidth=0.8, alpha=0.55, zorder=0,
        )
    )
    ax.axhline(min_activation_rate, color=BAND_EDGE,
               linewidth=0.8, alpha=0.7, zorder=1)

    cls = _classify_features(r_g, rate, band)
    for label in ("genome", "cross", "ehr", "dead"):
        sel = cls == label
        if not np.any(sel):
            continue
        ax.scatter(
            r_g[sel], np.maximum(rate[sel], 1e-6),
            s=14, c=_color_for_class(label),
            edgecolors="white", linewidths=0.3,
            alpha=0.8 if label != "dead" else 0.5,
            label=_class_legend_label(label),
            zorder=3,
        )

    if promoted_indices is not None and len(promoted_indices) > 0:
        idx = np.asarray(promoted_indices, dtype=int)
        ax.scatter(
            r_g[idx], np.maximum(rate[idx], 1e-6),
            s=85, facecolors="none", edgecolors=PROMOTED_COLOR,
            linewidths=1.5, label="promoted", zorder=4,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    if np.any(rate > 0):
        ax.set_ylim(max(rate[rate > 0].min() / 2, 1e-5), rate_top)
    ax.set_xlabel("genome share  $r_G$")
    ax.set_ylabel("activation rate  $P(z_j > 0)$")
    ax.set_title("Promotion candidates and selection score",
                 loc="left", fontsize=12, fontweight="bold", pad=12)
    _faint_grid(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=5, frameon=False, fontsize=8.5,
    )

    # text annotation: eligibility band
    ax.text(
        (band[0] + band[1]) / 2, rate_top * 0.92,
        f"eligibility  r_G \\in [{band[0]:.2f}, {band[1]:.2f}],   "
        f"rate \\geq {min_activation_rate:.2g}",
        ha="center", va="top", fontsize=8.5, color="#5F4A1F",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Decoder weight heatmap
# ---------------------------------------------------------------------------


def decoder_heatmap(
    model: TopKCrosscoder,
    selection: FeatureSelection,
    prs_columns: Sequence[str],
    ehr_columns: Sequence[str],
    *,
    ehr_kinds: Optional[Sequence[str]] = None,
    max_genome_rows: int = 25,
    max_ehr_rows: int = 35,
) -> Figure:
    """Signed decoder-weight heatmap over the most informative PRS + EHR
    columns, *transposed* relative to a conventional feature-vs-variable
    matrix: promoted features run along the x-axis (so all 14-32 of them
    fit comfortably), variables run down the y-axis with full readable
    labels, EHR columns grouped by kind. A signed cell value is drawn
    inside whenever ``|weight| >= 0.05`` so the dominant loadings are
    legible without zooming.
    """
    idx = np.asarray(selection.indices, dtype=int)
    order = np.argsort(-selection.genome_share)
    idx = idx[order]
    feat_names = np.asarray(selection.names)[order]
    r_g_promoted = selection.genome_share[order]

    Wg = model.W_d_G[idx]   # (n_promote, m_G)
    We = model.W_d_E[idx]   # (n_promote, m_E)

    # Pick the columns with the strongest absolute loading on at least
    # one promoted feature. Sort genome side by max-abs descending so the
    # most-influential PRS rises to the top of its block; EHR side groups
    # by kind first, then by max-abs within kind.
    def _topk_by_max_abs(W: np.ndarray, k: int) -> np.ndarray:
        score = np.max(np.abs(W), axis=0)
        n = min(int(k), int(W.shape[1]))
        if n <= 0:
            return np.array([], dtype=int)
        return np.argsort(-score)[:n]

    g_keep = _topk_by_max_abs(Wg, max_genome_rows)
    g_keep = g_keep[np.argsort(-np.max(np.abs(Wg[:, g_keep]), axis=0))]
    g_names = np.asarray(prs_columns)[g_keep] if g_keep.size else np.array([], dtype=object)

    e_keep_pre = _topk_by_max_abs(We, max_ehr_rows)
    if e_keep_pre.size and ehr_kinds is not None:
        e_kinds_arr = np.asarray(ehr_kinds)
        e_kinds_pre = e_kinds_arr[e_keep_pre]
        kind_order = ["condition", "drug", "lab_mean", "lab_min", "lab_max",
                      "lab_slope", "lab_missing", "utilisation"]
        order_keys = []
        for j, idx_keep in enumerate(e_keep_pre):
            kind = str(e_kinds_pre[j])
            kind_rank = kind_order.index(kind) if kind in kind_order else len(kind_order)
            score = float(np.max(np.abs(We[:, idx_keep])))
            order_keys.append((kind_rank, -score, idx_keep))
        order_keys.sort()
        e_keep = np.array([k[2] for k in order_keys], dtype=int)
    else:
        e_keep = e_keep_pre
    e_names = np.asarray(ehr_columns)[e_keep] if e_keep.size else np.array([], dtype=object)
    e_kinds_show = np.asarray(ehr_kinds)[e_keep] if (
        ehr_kinds is not None and e_keep.size
    ) else None

    Wg_show = Wg[:, g_keep] if g_keep.size else np.zeros((Wg.shape[0], 0))
    We_show = We[:, e_keep] if e_keep.size else np.zeros((We.shape[0], 0))

    # Stack rows: PRS block on top, EHR block below; transpose so the
    # heatmap shows variables-as-rows, features-as-columns.
    n_features = Wg.shape[0]
    n_g_rows = Wg_show.shape[1]
    n_e_rows = We_show.shape[1]
    n_rows = n_g_rows + n_e_rows
    if n_rows == 0:
        # Defensive: render an empty placeholder rather than dividing by zero.
        fig, ax = _new_axes((6.0, 3.0))
        ax.text(0.5, 0.5, "no decoder loadings to plot",
                ha="center", va="center", color=TEXT_COLOR)
        ax.set_axis_off()
        return fig
    H = np.zeros((n_rows, n_features), dtype=float)
    if n_g_rows:
        H[:n_g_rows] = Wg_show.T
    if n_e_rows:
        H[n_g_rows:] = We_show.T
    vmax = float(np.max(np.abs(H))) or 1.0

    row_labels = [str(n) for n in g_names] + [str(n) for n in e_names]

    # Layout: main heatmap centre, kind-color side strip on the left, r_G
    # bar across the top of the feature columns, colorbar on the right.
    fig_h = max(6.5, 0.32 * n_rows + 2.5)
    fig_w = max(9.0, 0.55 * n_features + 5.5)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[0.06, 1.0],
        width_ratios=[0.04, 1.0, 0.05],
        hspace=0.02, wspace=0.02,
    )
    ax_rg = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1], sharey=ax_side)
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ---- left side strip: kind-coloured band per row ----------------------
    side_band = np.zeros((n_rows, 1, 3), dtype=float)
    if n_g_rows:
        side_band[:n_g_rows, 0] = matplotlib.colors.to_rgb(GENOME_COLOR)
    for r in range(n_g_rows, n_rows):
        if e_kinds_show is not None:
            kind = str(e_kinds_show[r - n_g_rows])
            side_band[r, 0] = matplotlib.colors.to_rgb(
                EHR_KIND_COLORS.get(kind, EHR_COLOR)
            )
        else:
            side_band[r, 0] = matplotlib.colors.to_rgb(EHR_COLOR)
    ax_side.imshow(side_band, aspect="auto", interpolation="nearest")
    ax_side.set_xticks([])
    ax_side.set_yticks(range(n_rows))
    ax_side.set_yticklabels(row_labels, fontsize=8.5, color=TEXT_COLOR)
    ax_side.tick_params(axis="y", length=0, pad=4)
    for spine in ax_side.spines.values():
        spine.set_visible(False)

    # ---- top: r_G bar per feature column ---------------------------------
    ax_rg.bar(
        range(n_features), r_g_promoted,
        color=[GENOME_COLOR if x >= 0.5 else EHR_COLOR for x in r_g_promoted],
        edgecolor="white", linewidth=0.4, width=0.86,
    )
    ax_rg.axhline(0.5, color=AXIS_COLOR, linewidth=0.5, alpha=0.5, linestyle=":")
    ax_rg.set_xlim(-0.5, n_features - 0.5)
    ax_rg.set_ylim(0, 1.0)
    ax_rg.set_yticks([0.0, 0.5, 1.0])
    ax_rg.set_yticklabels(["0", "0.5", "1"], fontsize=7)
    ax_rg.set_xticks([])
    ax_rg.tick_params(axis="y", length=2, pad=2)
    for spine in ("top", "right", "bottom"):
        ax_rg.spines[spine].set_visible(False)
    ax_rg.spines["left"].set_color(AXIS_COLOR)
    ax_rg.spines["left"].set_linewidth(0.8)
    ax_rg.set_ylabel("$r_G$", fontsize=8, color=TEXT_COLOR)

    # ---- main heatmap ----------------------------------------------------
    im = ax_main.imshow(
        H, aspect="auto", cmap=DIVERGING_CMAP,
        norm=Normalize(vmin=-vmax, vmax=vmax), interpolation="nearest",
    )
    if n_g_rows and n_e_rows:
        ax_main.axhline(n_g_rows - 0.5, color="black", linewidth=1.0)
    ax_main.set_xticks(range(n_features))
    ax_main.set_xticklabels(
        [str(n) for n in feat_names],
        rotation=45, ha="right", fontsize=8.5, color=TEXT_COLOR,
    )
    ax_main.set_yticks([])
    ax_main.tick_params(axis="x", length=2, pad=3)
    for spine in ax_main.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)

    # ---- annotate strong cells ------------------------------------------
    annotate_threshold = 0.05
    for r in range(n_rows):
        for c in range(n_features):
            v = H[r, c]
            if abs(v) >= annotate_threshold:
                # White text on saturated colors, dark on faint cells.
                txt_color = "white" if abs(v) / vmax > 0.55 else TEXT_COLOR
                ax_main.text(
                    c, r, f"{v:+.2f}",
                    ha="center", va="center",
                    fontsize=7.0, color=txt_color,
                )

    # ---- colorbar -------------------------------------------------------
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.set_label("decoder weight", fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=8)

    fig.suptitle(
        "Promoted feature decoder weights",
        x=0.01, y=1.01, ha="left",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Per-feature dossier (top loadings, small multiples)
# ---------------------------------------------------------------------------


def feature_dossier(
    model: TopKCrosscoder,
    selection: FeatureSelection,
    prs_columns: Sequence[str],
    ehr_columns: Sequence[str],
    *,
    n_features: int = 8,
    top_k_per_side: int = 8,
) -> Figure:
    """Small-multiples 'trading cards' for the top promoted features.

    For each feature j we show the largest-magnitude PRS loadings on the
    left (navy bars, signed) and the largest-magnitude EHR loadings on
    the right (amber bars, signed). The card title carries the feature
    name, ``r_G``, and activation rate.
    """
    n_features = max(1, min(int(n_features), len(selection.indices)))
    # rank promoted features by selection score r_G(1-r_G)*rate, descending
    score = (
        selection.genome_share
        * (1.0 - selection.genome_share)
        * selection.activation_rate
    )
    order = np.argsort(-score)[:n_features]
    indices = np.asarray(selection.indices)[order]
    names = np.asarray(selection.names)[order]
    r_g = selection.genome_share[order]
    rate = selection.activation_rate[order]

    n_cols = 2
    n_rows = int(math.ceil(n_features / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(11.0, 2.4 * n_rows + 0.6),
        dpi=_DPI,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for slot in range(n_rows * n_cols):
        if slot >= n_features:
            for sub in (slot * 2, slot * 2 + 1):
                r, c = divmod(sub, n_cols * 2)
                axes[r, c].axis("off")
            continue
        row, col = divmod(slot, n_cols)
        ax_g = axes[row, col * 2]
        ax_e = axes[row, col * 2 + 1]
        feat_idx = int(indices[slot])

        Wg = model.W_d_G[feat_idx]
        We = model.W_d_E[feat_idx]

        def _bars(ax: plt.Axes, weights: np.ndarray, labels: Sequence[str],
                  base_color: str, title: str) -> None:
            top_idx = np.argsort(-np.abs(weights))[:top_k_per_side]
            top_idx = top_idx[np.argsort(weights[top_idx])]
            w = weights[top_idx]
            lbls = [_short(str(labels[i]), 22) for i in top_idx]
            colors = [base_color if v >= 0 else _darken(base_color, 0.45) for v in w]
            y = np.arange(len(w))
            ax.barh(y, w, color=colors, edgecolor="white", linewidth=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(lbls, fontsize=7.5, color=TEXT_COLOR)
            ax.axvline(0, color=AXIS_COLOR, linewidth=0.6)
            _style_axes(ax)
            _faint_grid(ax, axis="x")
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_title(title, loc="left", fontsize=9.5,
                         fontweight="bold", color=TEXT_COLOR, pad=2)

        title = (
            f"{names[slot]}    "
            f"$r_G$={r_g[slot]:.2f}   rate={rate[slot]:.3f}"
        )
        _bars(ax_g, Wg, prs_columns, GENOME_COLOR, "PRS loadings")
        _bars(ax_e, We, ehr_columns, EHR_COLOR, "EHR loadings")

        # card title across both subplots: place above the genome panel
        ax_g.text(
            0.0, 1.18, title,
            transform=ax_g.transAxes, ha="left", va="bottom",
            fontsize=10, color=TEXT_COLOR, fontweight="bold",
        )

    fig.suptitle(
        "Promoted feature dossier  (top loadings per stream)",
        x=0.01, y=1.02, ha="left",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


def _darken(color: str, factor: float) -> str:
    """Return ``color`` blended toward black by ``factor`` in [0, 1]."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return matplotlib.colors.to_hex(rgb * (1.0 - factor))


# ---------------------------------------------------------------------------
# 7. Co-activation Jaccard matrix
# ---------------------------------------------------------------------------


def coactivation_matrix(
    z_promoted: np.ndarray,
    selection: FeatureSelection,
) -> Figure:
    """Pairwise Jaccard similarity of binary activation patterns.

    ``z_promoted`` is the post-TopK activation matrix restricted to the
    promoted feature columns, shape ``(n, n_promote)``. Rows are sorted
    by genome share so the matrix shares an axis with the decoder
    heatmap. The diagonal carries each feature's own activation rate.
    """
    z_promoted = np.asarray(z_promoted)
    binar = (z_promoted > 0).astype(np.int64)
    rate = binar.mean(axis=0)

    order = np.argsort(-selection.genome_share)
    binar = binar[:, order]
    rate = rate[order]
    names = np.asarray(selection.names)[order]
    r_g = selection.genome_share[order]

    n = binar.shape[1]
    inter = binar.T @ binar
    counts = binar.sum(axis=0)
    union = counts[:, None] + counts[None, :] - inter
    union_safe = np.where(union > 0, union, 1)
    jacc = inter / union_safe
    jacc = np.where(union > 0, jacc, 0.0)
    np.fill_diagonal(jacc, rate)  # diagonal == activation rate (in [0, 1])

    fig_w = max(7.0, 0.32 * n + 3.0)
    fig_h = max(6.0, 0.32 * n + 2.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.04, 1.0],
        width_ratios=[1.0, 0.04],
        wspace=0.02, hspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # r_G strips on top and right, sharing axes with the main image
    band_cmap = LinearSegmentedColormap.from_list(
        "rg_band", [EHR_COLOR, "#FFFFFF", GENOME_COLOR],
    )
    norm = Normalize(0, 1)
    ax_top.imshow(r_g[None, :], aspect="auto", cmap=band_cmap, norm=norm)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)
    ax_right.imshow(r_g[:, None], aspect="auto", cmap=band_cmap, norm=norm)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for s in ax_right.spines.values():
        s.set_visible(False)

    im = ax_main.imshow(
        jacc, aspect="auto", cmap="magma_r",
        norm=Normalize(0, max(jacc.max(), 0.05)),
        interpolation="nearest",
    )
    ax_main.set_xticks(range(n))
    ax_main.set_yticks(range(n))
    ax_main.set_xticklabels(
        [_short(str(x), 16) for x in names],
        rotation=70, ha="right", fontsize=7, color=TEXT_COLOR,
    )
    ax_main.set_yticklabels(
        [_short(str(x), 16) for x in names],
        fontsize=7.5, color=TEXT_COLOR,
    )
    for spine in ax_main.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)
    ax_main.tick_params(length=2)

    cbar = fig.colorbar(
        im, ax=ax_main, orientation="vertical",
        pad=0.06, fraction=0.025, shrink=0.7,
    )
    cbar.set_label("Jaccard  (diagonal: activation rate)",
                   fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=8)

    fig.suptitle(
        "Promoted feature co-activation",
        x=0.01, y=1.01, ha="left",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Reconstruction R^2 per panel column
# ---------------------------------------------------------------------------


def reconstruction_quality(
    model: TopKCrosscoder,
    panels: AlignedPanels,
    prs_columns: Sequence[str],
    ehr_columns: Sequence[str],
    *,
    ehr_kinds: Optional[Sequence[str]] = None,
    max_ehr_rows: int = 30,
) -> Figure:
    """Per-column reconstruction R^2 on the standardised inputs.

    Genome side: every PRS column is shown explicitly. EHR side: the
    distribution per kind (when ``ehr_kinds`` is provided) or the top /
    bottom ``max_ehr_rows`` columns by R^2 (otherwise) so the figure
    stays legible even when ``m_E`` is in the hundreds.
    """
    z = encode(model, panels.A, panels.B)
    a_hat = z @ model.W_d_G
    b_hat = z @ model.W_d_E
    a_z = (panels.A - model.mean_G) / model.std_G
    b_z = (panels.B - model.mean_E) / model.std_E

    # Genome side is always Gaussian (PRS values), so plain z-space R^2 is
    # the right metric.
    ss_res_g = np.sum((a_z - a_hat) ** 2, axis=0)
    ss_tot_g = np.sum((a_z - a_z.mean(axis=0, keepdims=True)) ** 2, axis=0)
    ss_tot_g = np.where(ss_tot_g > 1e-12, ss_tot_g, np.nan)
    r2_g = 1.0 - ss_res_g / ss_tot_g

    # EHR side is mixed-likelihood: ``b_hat`` is a z-score / logit /
    # log-rate depending on the column's kind. Comparing it against the
    # z-scored target with a single R^2 is meaningless (see the docstring
    # of ``_ehr_recon_metrics``), so evaluate each kind in its native
    # space and return a comparable pseudo-R^2 per column.
    gaussian_e, binary_e, count_e = _ehr_kind_masks(
        model.ehr_feature_kinds, model.m_E
    )
    r2_e = _ehr_recon_per_column(
        b_z, panels.B, b_hat,
        gaussian=gaussian_e, binary=binary_e, count=count_e,
        mean_E=model.mean_E,
    )
    mean_g = float(np.nanmean(r2_g)) if r2_g.size else float("nan")
    mean_e = float(np.nanmean(r2_e)) if r2_e.size else float("nan")

    fig = plt.figure(figsize=(11.0, 7.5), dpi=_DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.25)
    ax_g = fig.add_subplot(gs[0])
    ax_e = fig.add_subplot(gs[1])
    _style_axes(ax_g)
    _style_axes(ax_e)

    # ---- genome side: all columns ----------------------------------------
    g_order = np.argsort(r2_g)
    ax_g.barh(
        np.arange(r2_g.size),
        r2_g[g_order],
        color=GENOME_COLOR, edgecolor="white", linewidth=0.4,
    )
    ax_g.set_yticks(np.arange(r2_g.size))
    ax_g.set_yticklabels(
        [_short(str(prs_columns[i]), 28) for i in g_order],
        fontsize=7.5, color=TEXT_COLOR,
    )
    ax_g.axvline(mean_g, color=PROMOTED_COLOR, linewidth=1.2,
                 linestyle="--", alpha=0.85)
    ax_g.text(mean_g, r2_g.size - 0.5,
              f"  mean R² = {mean_g:.3f}",
              ha="left", va="top", fontsize=8.5, color=PROMOTED_COLOR)
    ax_g.set_xlabel("R² (standardised target)")
    ax_g.set_title("Genome side: per-PRS reconstruction",
                   loc="left", fontsize=11, fontweight="bold")
    _faint_grid(ax_g, axis="x")

    # ---- EHR side: by kind, distribution; or best/worst N ----------------
    quality_label = "reconstruction quality (R² / Brier lift)"
    if ehr_kinds is not None and len(ehr_kinds) == r2_e.size:
        kinds = np.asarray(ehr_kinds)
        kind_order = sorted(np.unique(kinds).tolist())
        positions: list[float] = []
        labels: list[str] = []
        bp_data: list[np.ndarray] = []
        for i, kind in enumerate(kind_order):
            sel = kinds == kind
            vals = r2_e[sel]
            bp_data.append(vals[np.isfinite(vals)])
            positions.append(float(i))
            labels.append(f"{kind}\n(n={int(sel.sum())})")
        bp = ax_e.boxplot(
            bp_data, positions=positions, widths=0.55,
            patch_artist=True, showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            color = EHR_KIND_COLORS.get(kind_order[i], EHR_COLOR)
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(_darken(color, 0.35))
        for whisker in bp["whiskers"]:
            whisker.set_color(AXIS_COLOR)
            whisker.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_color(AXIS_COLOR)
            cap.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(1.6)
        # overlay individual points lightly
        rng = np.random.default_rng(0)
        for i, vals in enumerate(bp_data):
            jitter = rng.uniform(-0.18, 0.18, size=vals.size)
            ax_e.scatter(
                np.full(vals.size, positions[i]) + jitter, vals,
                s=10, color=_darken(EHR_KIND_COLORS.get(kind_order[i], EHR_COLOR), 0.25),
                alpha=0.5, edgecolors="none", zorder=3,
            )
        ax_e.set_xticks(positions)
        ax_e.set_xticklabels(labels, fontsize=8.5, color=TEXT_COLOR)
        ax_e.axhline(mean_e, color=PROMOTED_COLOR, linewidth=1.2,
                     linestyle="--", alpha=0.85)
        ax_e.text(
            ax_e.get_xlim()[1] * 0.99, mean_e,
            f"  mean = {mean_e:.3f}",
            ha="right", va="bottom", fontsize=8.5, color=PROMOTED_COLOR,
        )
        ax_e.set_ylabel(quality_label)
        ax_e.set_title("EHR side: per-column reconstruction by kind",
                       loc="left", fontsize=11, fontweight="bold")
        _faint_grid(ax_e, axis="y")
    else:
        finite = np.isfinite(r2_e)
        idx_all = np.flatnonzero(finite)
        order_finite = idx_all[np.argsort(r2_e[idx_all])]
        n = order_finite.size
        if n <= 2 * max_ehr_rows:
            keep = order_finite
        else:
            keep = np.concatenate(
                [order_finite[:max_ehr_rows], order_finite[-max_ehr_rows:]]
            )
        r2_show = r2_e[keep]
        ax_e.barh(
            np.arange(r2_show.size),
            r2_show,
            color=EHR_COLOR, edgecolor="white", linewidth=0.4,
        )
        ax_e.set_yticks(np.arange(r2_show.size))
        ax_e.set_yticklabels(
            [_short(str(ehr_columns[i]), 28) for i in keep],
            fontsize=7.5, color=TEXT_COLOR,
        )
        ax_e.axvline(mean_e, color=PROMOTED_COLOR, linewidth=1.2,
                     linestyle="--", alpha=0.85)
        ax_e.text(
            mean_e, r2_show.size - 0.5,
            f"  mean = {mean_e:.3f}",
            ha="left", va="top", fontsize=8.5, color=PROMOTED_COLOR,
        )
        ax_e.set_xlabel(quality_label)
        ax_e.set_title(
            "EHR side: per-column reconstruction "
            f"(top + bottom {max_ehr_rows})",
            loc="left", fontsize=11, fontweight="bold",
        )
        _faint_grid(ax_e, axis="x")

    fig.suptitle(
        "Crosscoder reconstruction quality",
        x=0.01, y=1.02, ha="left",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Activation density ridgeline
# ---------------------------------------------------------------------------


def activation_ridgeline(
    z_promoted: np.ndarray,
    selection: FeatureSelection,
    *,
    log_scale: bool = True,
    overlap: float = 0.6,
) -> Figure:
    """Ridgeline of per-feature activation magnitude distributions.

    Each row is a promoted feature's distribution of *positive* activation
    magnitudes (zeros excluded). Rows are ordered by genome share so the
    figure stacks vertically from genome-heavy at the top to EHR-heavy
    at the bottom; row tint mirrors that gradient.
    """
    z_promoted = np.asarray(z_promoted)
    order = np.argsort(-selection.genome_share)
    z_sorted = z_promoted[:, order]
    names = np.asarray(selection.names)[order]
    r_g = selection.genome_share[order]
    rate = selection.activation_rate[order]

    n = z_sorted.shape[1]
    fig, ax = _new_axes((9.0, max(4.5, 0.42 * n + 1.5)))

    # global range across nonzero values
    all_pos = z_sorted[z_sorted > 0]
    if all_pos.size == 0:
        ax.text(0.5, 0.5, "no nonzero activations", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_COLOR)
        return fig
    if log_scale:
        lo = np.quantile(all_pos, 0.02)
        hi = np.quantile(all_pos, 0.995)
        edges = np.geomspace(max(lo, 1e-6), max(hi, lo * 10), 60)
    else:
        lo = 0.0
        hi = np.quantile(all_pos, 0.995)
        edges = np.linspace(0, max(hi, 1e-6), 60)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # tint scale: navy at top (high r_G), amber at bottom (low r_G)
    cmap = LinearSegmentedColormap.from_list(
        "ridge_tint", [EHR_COLOR, CROSSMODAL_COLOR, GENOME_COLOR],
    )

    yticks: list[float] = []
    for i in range(n):
        col = z_sorted[:, i]
        pos = col[col > 0]
        if pos.size == 0:
            density = np.zeros(centres.size)
        else:
            density, _ = np.histogram(pos, bins=edges, density=True)
        if density.max() > 0:
            density = density / density.max()
        baseline = (n - 1 - i) * (1.0 - overlap)
        yticks.append(baseline + 0.5 * (1.0 - overlap))
        face = cmap(float(np.clip(r_g[i], 0, 1)))
        ax.fill_between(
            centres, baseline, baseline + density,
            color=face, alpha=0.78, linewidth=0.6,
            edgecolor=_darken(matplotlib.colors.to_hex(face), 0.30),
        )
        # rate annotation on the right edge
        ax.text(
            edges[-1] * (1.04 if log_scale else 1.0),
            baseline + 0.5 * (1.0 - overlap),
            f"{names[i]}    rate={rate[i]:.3f}    $r_G$={r_g[i]:.2f}",
            ha="left", va="center", fontsize=8, color=TEXT_COLOR,
            transform=ax.transData,
        )

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-0.1, n * (1.0 - overlap) + 1.1)
    ax.set_yticks([])
    ax.set_xlabel("activation magnitude  $z_j$  (positive only)")
    ax.set_title(
        "Promoted feature activation density",
        loc="left", fontsize=12, fontweight="bold", pad=12,
    )
    _faint_grid(ax, axis="x")
    return fig


# ---------------------------------------------------------------------------
# 10. Headline overview (one figure that summarises the model)
# ---------------------------------------------------------------------------


def participant_umap(
    z: np.ndarray,
    event: Optional[np.ndarray] = None,
    *,
    target_name: str = "event",
    sample_size: int = 12000,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    seed: int = 0,
) -> Figure:
    """UMAP of participant latent embeddings ``z = (n, d)``.

    Coloured by ``event`` (0/1) when provided so the cohort can be
    eyeballed for separation between the outcome positive and negative
    classes in crosscoder space. Subsampled to ``sample_size`` rows by
    default because UMAP on n>50k is wall-clock dominated by the
    nearest-neighbour graph build.
    """
    import umap  # type: ignore[import-not-found]

    n = int(z.shape[0])
    rng = np.random.default_rng(seed)
    if sample_size and n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        z_sub = np.ascontiguousarray(z[idx], dtype=np.float32)
        ev_sub = np.asarray(event)[idx] if event is not None else None
    else:
        z_sub = np.ascontiguousarray(z, dtype=np.float32)
        ev_sub = np.asarray(event) if event is not None else None

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    emb = reducer.fit_transform(z_sub)

    fig, ax = _new_axes((9.5, 7.5))
    if ev_sub is not None:
        non_event = ev_sub.astype(int) == 0
        is_event = ev_sub.astype(int) == 1
        ax.scatter(
            emb[non_event, 0], emb[non_event, 1],
            s=3, c="#9AA5B1", alpha=0.30, linewidths=0,
            label=f"non-{target_name} (n={int(non_event.sum()):,})",
        )
        ax.scatter(
            emb[is_event, 0], emb[is_event, 1],
            s=8, c=GENOME_COLOR, alpha=0.75, linewidths=0,
            label=f"{target_name} case (n={int(is_event.sum()):,})",
        )
        ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=9)
    else:
        ax.scatter(
            emb[:, 0], emb[:, 1], s=3, c=GENOME_COLOR,
            alpha=0.30, linewidths=0,
        )
    _style_axes(ax)
    ax.set_xlabel("UMAP 1", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("UMAP 2", fontsize=10, color=TEXT_COLOR)
    ax.set_title(
        f"Participant latent space (UMAP, n={int(z_sub.shape[0]):,} of {n:,})",
        fontsize=12, color=TEXT_COLOR,
    )
    return fig


def feature_umap(
    model: TopKCrosscoder,
    selection: FeatureSelection,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.08,
    seed: int = 0,
) -> Figure:
    """UMAP of crosscoder features in their decoder-vector space.

    Each point is one of the ``d`` latent features; coordinates come from
    UMAP applied to the concatenated decoder rows ``[W_d_G[j] | W_d_E[j]]``
    (cosine metric). Colour encodes ``r_G`` so genome-decoder-dominant
    features land at one end of the colour bar and EHR-decoder-dominant
    at the other; promoted features are drawn larger with a black star
    marker so the chosen subset is visible against the background of all
    learned features. Dead features (zero decoder norm) are filtered out.
    """
    import umap  # type: ignore[import-not-found]

    W = np.concatenate([model.W_d_G, model.W_d_E], axis=1)
    norm = np.linalg.norm(W, axis=1)
    alive = norm > 1e-8
    W_alive = np.ascontiguousarray(W[alive], dtype=np.float32)
    if W_alive.shape[0] < 4:
        fig, ax = _new_axes((6.0, 4.0))
        ax.text(0.5, 0.5, "too few alive features for UMAP",
                ha="center", va="center", color=TEXT_COLOR)
        ax.set_axis_off()
        return fig

    g_norm_sq = np.sum(model.W_d_G ** 2, axis=1)
    e_norm_sq = np.sum(model.W_d_E ** 2, axis=1)
    total = np.where(g_norm_sq + e_norm_sq > 0, g_norm_sq + e_norm_sq, 1.0)
    r_g_full = g_norm_sq / total
    r_g = r_g_full[alive]

    bank = np.asarray(model.latent_bank)[alive]
    idx_alive = np.flatnonzero(alive)
    promoted_set = set(int(i) for i in selection.indices)
    is_promoted = np.array(
        [int(j) in promoted_set for j in idx_alive], dtype=bool
    )

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, W_alive.shape[0] - 1)),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    emb = reducer.fit_transform(W_alive)

    fig, ax = _new_axes((10.5, 7.5))

    bank_marker = {
        BANK_SHARED: "o",
        BANK_GENOME_PRIVATE: "s",
        BANK_EHR_PRIVATE: "^",
    }
    bank_label = {
        BANK_SHARED: "shared bank",
        BANK_GENOME_PRIVATE: "genome-private",
        BANK_EHR_PRIVATE: "EHR-private",
    }
    sc = None
    for b, marker in bank_marker.items():
        sel = (bank == b) & ~is_promoted
        if not bool(sel.any()):
            continue
        sc = ax.scatter(
            emb[sel, 0], emb[sel, 1],
            c=r_g[sel], cmap=DIVERGING_CMAP,
            norm=Normalize(vmin=0.0, vmax=1.0),
            s=14, alpha=0.75, edgecolor="none", marker=marker,
            label=f"{bank_label[b]} (n={int(sel.sum())})",
        )
    if bool(is_promoted.any()):
        ax.scatter(
            emb[is_promoted, 0], emb[is_promoted, 1],
            c=r_g[is_promoted], cmap=DIVERGING_CMAP,
            norm=Normalize(vmin=0.0, vmax=1.0),
            s=130, alpha=1.0, edgecolor="black", linewidth=1.4,
            marker="*", label=f"promoted (n={int(is_promoted.sum())})",
            zorder=5,
        )
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("genome share $r_G$", fontsize=10, color=TEXT_COLOR)
        cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=8)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=9)
    _style_axes(ax)
    ax.set_xlabel("UMAP 1", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("UMAP 2", fontsize=10, color=TEXT_COLOR)
    ax.set_title(
        f"Crosscoder features in decoder space "
        f"({int(alive.sum())} of {int(model.d)} alive, "
        f"colour = $r_G$, marker = bank, star = promoted)",
        fontsize=12, color=TEXT_COLOR,
    )
    return fig


def overview(
    model: TopKCrosscoder,
    *,
    activation_rate: np.ndarray,
    selection: FeatureSelection,
    panels: Optional[AlignedPanels] = None,
    band: Tuple[float, float] = (0.2, 0.8),
    min_activation_rate: float = 0.01,
) -> Figure:
    """Two-by-two headline figure that summarises the trained crosscoder."""
    r_g = feature_stream_share(model)
    fig = plt.figure(figsize=(12.5, 8.5), dpi=_DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20)

    # (A) genome share histogram
    ax_a = fig.add_subplot(gs[0, 0])
    _style_axes(ax_a)
    ax_a.axvspan(band[0], band[1], color=BAND_COLOR, alpha=0.65, zorder=0)
    edges = np.linspace(0, 1, 41)
    cls = _classify_features(r_g, activation_rate, band)
    for label, color in (("genome", GENOME_COLOR), ("cross", CROSSMODAL_COLOR),
                         ("ehr", EHR_COLOR)):
        sel = (cls == label) & (activation_rate > 0)
        cnt, _ = np.histogram(r_g[sel], bins=edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax_a.bar(centres, cnt, width=0.025, color=color,
                 edgecolor="white", linewidth=0.3)
    ax_a.scatter(
        selection.genome_share,
        np.full(len(selection.genome_share),
                ax_a.get_ylim()[1] * 0.95 if ax_a.get_ylim()[1] else 1),
        marker="|", s=60, color=PROMOTED_COLOR, linewidths=1.0,
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("$r_G$")
    ax_a.set_ylabel("feature count")
    ax_a.set_title("A. Genome share distribution",
                   loc="left", fontsize=11, fontweight="bold")
    _faint_grid(ax_a, axis="y")

    # (B) decoder norm scatter
    ax_b = fig.add_subplot(gs[0, 1])
    _style_axes(ax_b)
    n_g = np.linalg.norm(model.W_d_G, axis=1)
    n_e = np.linalg.norm(model.W_d_E, axis=1)
    theta = np.linspace(0, np.pi / 2, 200)
    ax_b.plot(np.cos(theta), np.sin(theta),
              color=AXIS_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    for label in ("genome", "cross", "ehr", "dead"):
        sel = cls == label
        if not np.any(sel):
            continue
        ax_b.scatter(n_g[sel], n_e[sel], s=10,
                     c=_color_for_class(label),
                     edgecolors="white", linewidths=0.2,
                     alpha=0.8 if label != "dead" else 0.5)
    idx = np.asarray(selection.indices, dtype=int)
    ax_b.scatter(n_g[idx], n_e[idx], s=70, facecolors="none",
                 edgecolors=PROMOTED_COLOR, linewidths=1.4)
    ax_b.set_aspect("equal")
    ax_b.set_xlim(-0.02, 1.05)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_xlabel("$\\|W^G_d[j]\\|$")
    ax_b.set_ylabel("$\\|W^E_d[j]\\|$")
    ax_b.set_title("B. Decoder geometry",
                   loc="left", fontsize=11, fontweight="bold")
    _faint_grid(ax_b)

    # (C) selection scatter
    ax_c = fig.add_subplot(gs[1, 0])
    _style_axes(ax_c)
    rate_top = max(activation_rate.max(), 1.0) * 1.2
    ax_c.add_patch(
        patches.Rectangle(
            (band[0], min_activation_rate),
            band[1] - band[0], rate_top,
            facecolor=BAND_COLOR, edgecolor=BAND_EDGE,
            linewidth=0.8, alpha=0.55, zorder=0,
        )
    )
    for label in ("genome", "cross", "ehr", "dead"):
        sel = cls == label
        if not np.any(sel):
            continue
        ax_c.scatter(
            r_g[sel], np.maximum(activation_rate[sel], 1e-6),
            s=10, c=_color_for_class(label),
            edgecolors="white", linewidths=0.2,
            alpha=0.8 if label != "dead" else 0.5,
        )
    ax_c.scatter(
        selection.genome_share,
        np.maximum(selection.activation_rate, 1e-6),
        s=60, facecolors="none", edgecolors=PROMOTED_COLOR, linewidths=1.4,
    )
    ax_c.set_yscale("log")
    ax_c.set_xlim(0, 1)
    if np.any(activation_rate > 0):
        ax_c.set_ylim(activation_rate[activation_rate > 0].min() / 2, rate_top)
    ax_c.set_xlabel("$r_G$")
    ax_c.set_ylabel("activation rate")
    ax_c.set_title("C. Promotion rule",
                   loc="left", fontsize=11, fontweight="bold")
    _faint_grid(ax_c)

    # (D) reconstruction R^2 summary
    ax_d = fig.add_subplot(gs[1, 1])
    _style_axes(ax_d)
    if panels is not None:
        z = encode(model, panels.A, panels.B)
        a_hat = z @ model.W_d_G
        b_hat = z @ model.W_d_E
        a_z = (panels.A - model.mean_G) / model.std_G
        b_z = (panels.B - model.mean_E) / model.std_E
        ss_res_a = np.sum((a_z - a_hat) ** 2, axis=0)
        ss_tot_a = np.sum((a_z - a_z.mean(axis=0, keepdims=True)) ** 2, axis=0)
        ss_tot_a = np.where(ss_tot_a > 1e-12, ss_tot_a, np.nan)
        r2_g_arr = 1.0 - ss_res_a / ss_tot_a
        # EHR side: mixed-likelihood, so compute per-kind reconstruction
        # quality (R^2 for gaussian/count, Brier-lift for binary) rather
        # than comparing z-scored targets to logits/log-rates.
        gaussian_e, binary_e, count_e = _ehr_kind_masks(
            model.ehr_feature_kinds, model.m_E
        )
        r2_e_arr = _ehr_recon_per_column(
            b_z, panels.B, b_hat,
            gaussian=gaussian_e, binary=binary_e, count=count_e,
            mean_E=model.mean_E,
        )
        bp = ax_d.boxplot(
            [r2_g_arr[np.isfinite(r2_g_arr)], r2_e_arr[np.isfinite(r2_e_arr)]],
            positions=[0, 1], widths=0.55,
            patch_artist=True, showfliers=False,
        )
        for patch, c in zip(bp["boxes"], (GENOME_COLOR, EHR_COLOR)):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
            patch.set_edgecolor(_darken(c, 0.35))
        for whisker in bp["whiskers"]:
            whisker.set_color(AXIS_COLOR)
            whisker.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_color(AXIS_COLOR)
            cap.set_linewidth(0.8)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(1.6)
        rng = np.random.default_rng(1)
        for i, vals in enumerate([r2_g_arr, r2_e_arr]):
            finite = vals[np.isfinite(vals)]
            j = rng.uniform(-0.18, 0.18, size=finite.size)
            ax_d.scatter(np.full(finite.size, i) + j, finite, s=8,
                         color=_darken([GENOME_COLOR, EHR_COLOR][i], 0.25),
                         alpha=0.55, edgecolors="none")
        ax_d.set_xticks([0, 1])
        ax_d.set_xticklabels(
            [f"genome\nm={r2_g_arr.size}", f"EHR\nm={r2_e_arr.size}"],
            fontsize=9.5,
        )
        ax_d.set_ylabel("reconstruction quality (R² / Brier lift)")
        ax_d.set_title("D. Reconstruction quality",
                       loc="left", fontsize=11, fontweight="bold")
        _faint_grid(ax_d, axis="y")
    else:
        ax_d.text(0.5, 0.5, "panels not available",
                  ha="center", va="center", color=TEXT_COLOR,
                  transform=ax_d.transAxes)
        ax_d.set_title("D. Reconstruction quality",
                       loc="left", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"TopK crosscoder — overview     "
        f"d={model.d}   k={model.k}   "
        f"promoted={len(selection.indices)}",
        x=0.01, y=1.02, ha="left",
        fontsize=14, fontweight="bold", color=TEXT_COLOR,
    )
    return fig


# ---------------------------------------------------------------------------
# Top-level saver
# ---------------------------------------------------------------------------


@dataclass
class GenscorePlotInputs:
    """Bundle of inputs accepted by :func:`save_all_genscore_plots`.

    Any field may be ``None``; the saver skips plots whose required
    inputs are missing.
    """

    model: Optional[TopKCrosscoder] = None
    panels: Optional[AlignedPanels] = None
    selection: Optional[FeatureSelection] = None
    prs_columns: Optional[Sequence[str]] = None
    ehr_columns: Optional[Sequence[str]] = None
    ehr_kinds: Optional[Sequence[str]] = None
    history: Optional[Mapping[str, Sequence[float]]] = None
    event: Optional[np.ndarray] = None
    target_name: str = "T2D"
    band: Tuple[float, float] = (0.2, 0.8)
    min_activation_rate: float = 0.01


def save_all_genscore_plots(
    outputs_dir: str,
    inputs: GenscorePlotInputs,
) -> Dict[str, Tuple[str, str]]:
    """Render every available genscore figure into ``outputs_dir``.

    Returns ``{plot_name: (png_path, pdf_path)}``. Plots whose required
    inputs are missing are skipped silently.
    """
    saved: Dict[str, Tuple[str, str]] = {}
    band = inputs.band
    min_rate = inputs.min_activation_rate

    # Pre-compute commonly used quantities once.
    activation_rate: Optional[np.ndarray] = None
    z_promoted: Optional[np.ndarray] = None
    if inputs.model is not None and inputs.panels is not None:
        z = encode(inputs.model, inputs.panels.A, inputs.panels.B)
        activation_rate = (z > 0).mean(axis=0)
        if inputs.selection is not None and len(inputs.selection.indices):
            z_promoted = z[:, np.asarray(inputs.selection.indices, dtype=int)]
    r_g_full: Optional[np.ndarray] = (
        feature_stream_share(inputs.model) if inputs.model is not None else None
    )

    # 1. genome share distribution
    if r_g_full is not None:
        fig = genome_share_distribution(
            r_g_full,
            activation_rate=activation_rate,
            band=band,
            promoted_r_g=(
                inputs.selection.genome_share
                if inputs.selection is not None else None
            ),
        )
        saved["crosscoder_genome_share"] = _save(
            fig, outputs_dir, "crosscoder_01_genome_share")

    # 2. decoder norm scatter
    if inputs.model is not None:
        fig = decoder_norm_scatter(
            inputs.model,
            activation_rate=activation_rate,
            promoted_indices=(
                np.asarray(inputs.selection.indices, dtype=int)
                if inputs.selection is not None else None
            ),
            band=band,
        )
        saved["crosscoder_decoder_geometry"] = _save(
            fig, outputs_dir, "crosscoder_02_decoder_geometry")

    # 3. training dynamics
    if inputs.history is not None:
        fig = training_dynamics(
            inputs.history,
            k=inputs.model.k if inputs.model is not None else None,
            d=inputs.model.d if inputs.model is not None else None,
        )
        saved["crosscoder_training"] = _save(
            fig, outputs_dir, "crosscoder_03_training_dynamics")

    # 4. selection scatter
    if r_g_full is not None and activation_rate is not None:
        fig = selection_scatter(
            r_g_full, activation_rate,
            promoted_indices=(
                np.asarray(inputs.selection.indices, dtype=int)
                if inputs.selection is not None else None
            ),
            band=band, min_activation_rate=min_rate,
        )
        saved["crosscoder_selection"] = _save(
            fig, outputs_dir, "crosscoder_04_selection")

    # 5. decoder heatmap
    if (
        inputs.model is not None
        and inputs.selection is not None
        and inputs.prs_columns is not None
        and inputs.ehr_columns is not None
    ):
        fig = decoder_heatmap(
            inputs.model, inputs.selection,
            inputs.prs_columns, inputs.ehr_columns,
            ehr_kinds=inputs.ehr_kinds,
        )
        saved["crosscoder_decoder_heatmap"] = _save(
            fig, outputs_dir, "crosscoder_05_decoder_heatmap")

    # 6. dossier
    if (
        inputs.model is not None
        and inputs.selection is not None
        and inputs.prs_columns is not None
        and inputs.ehr_columns is not None
    ):
        fig = feature_dossier(
            inputs.model, inputs.selection,
            inputs.prs_columns, inputs.ehr_columns,
        )
        saved["crosscoder_dossier"] = _save(
            fig, outputs_dir, "crosscoder_06_feature_dossier")

    # 7. co-activation
    if (
        z_promoted is not None
        and inputs.selection is not None
        and len(inputs.selection.indices) >= 2
    ):
        fig = coactivation_matrix(z_promoted, inputs.selection)
        saved["crosscoder_coactivation"] = _save(
            fig, outputs_dir, "crosscoder_07_coactivation")

    # 8. reconstruction quality
    if (
        inputs.model is not None
        and inputs.panels is not None
        and inputs.prs_columns is not None
        and inputs.ehr_columns is not None
    ):
        fig = reconstruction_quality(
            inputs.model, inputs.panels,
            inputs.prs_columns, inputs.ehr_columns,
            ehr_kinds=inputs.ehr_kinds,
        )
        saved["crosscoder_reconstruction"] = _save(
            fig, outputs_dir, "crosscoder_08_reconstruction")

    # 9. activation ridgeline
    if z_promoted is not None and inputs.selection is not None:
        fig = activation_ridgeline(z_promoted, inputs.selection)
        saved["crosscoder_ridgeline"] = _save(
            fig, outputs_dir, "crosscoder_09_activation_ridgeline")

    # 10. overview headline
    if (
        inputs.model is not None
        and activation_rate is not None
        and inputs.selection is not None
    ):
        fig = overview(
            inputs.model,
            activation_rate=activation_rate,
            selection=inputs.selection,
            panels=inputs.panels,
            band=band, min_activation_rate=min_rate,
        )
        saved["crosscoder_overview"] = _save(
            fig, outputs_dir, "crosscoder_00_overview")

    # 11. participant UMAP (cohort latent space coloured by event status)
    if z is not None:
        try:
            fig = participant_umap(
                z,
                event=inputs.event,
                target_name=inputs.target_name,
            )
            saved["crosscoder_participant_umap"] = _save(
                fig, outputs_dir, "crosscoder_10_participant_umap")
        except Exception as exc:  # pragma: no cover - umap is optional
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "participant_umap failed: %s", exc, exc_info=True
            )

    # 12. feature UMAP (decoder-vector space, coloured by r_G)
    if inputs.model is not None and inputs.selection is not None:
        try:
            fig = feature_umap(inputs.model, inputs.selection)
            saved["crosscoder_feature_umap"] = _save(
                fig, outputs_dir, "crosscoder_11_feature_umap")
        except Exception as exc:  # pragma: no cover - umap is optional
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "feature_umap failed: %s", exc, exc_info=True
            )

    return saved


__all__ = [
    "GenscorePlotInputs",
    "activation_ridgeline",
    "coactivation_matrix",
    "decoder_heatmap",
    "decoder_norm_scatter",
    "feature_dossier",
    "genome_share_distribution",
    "overview",
    "reconstruction_quality",
    "save_all_genscore_plots",
    "selection_scatter",
    "training_dynamics",
]
