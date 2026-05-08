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

from .crosscoder import TopKCrosscoder, encode, feature_stream_share
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
    loss_aux = np.asarray(history.get("loss_aux", []), dtype=float)
    frac_dead = np.asarray(history.get("frac_dead", []), dtype=float)
    frac_active = np.asarray(history.get("frac_active_batch", []), dtype=float)
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
        target = float(k) / float(d)
        ax.axhline(target, color=AXIS_COLOR, linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.text(
            ax.get_xlim()[1] if step.size else 1.0,
            target, f"  k/d = {target:.3f}",
            ha="left", va="center", fontsize=8.5, color=AXIS_COLOR,
            transform=ax.transData,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("active per batch")
    ax.set_title("C. Sparsity (per-batch coverage)", loc="left",
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
    max_columns_per_side: int = 80,
) -> Figure:
    """Signed decoder-weight heatmap over the full PRS + EHR vocabulary.

    Rows: promoted features, sorted by genome share descending (genome-
    heavy at the top). Columns: PRS columns followed by EHR columns; a
    vertical divider marks the split. If the EHR side is too wide, only
    the columns with the largest absolute promoted-feature loading are
    shown. A coloured kind-bar above the EHR columns indicates which
    feature kind each column belongs to.
    """
    idx = np.asarray(selection.indices, dtype=int)
    order = np.argsort(-selection.genome_share)
    idx = idx[order]
    feat_names = np.asarray(selection.names)[order]
    r_g_promoted = selection.genome_share[order]

    Wg = model.W_d_G[idx]   # (n_promote, m_G)
    We = model.W_d_E[idx]   # (n_promote, m_E)

    # Optionally trim columns by greatest absolute weight across promoted features
    def _pick(W: np.ndarray, names: Sequence[str], cap: int) -> Tuple[np.ndarray, np.ndarray]:
        if W.shape[1] <= cap:
            return np.arange(W.shape[1]), np.asarray(names)
        score = np.max(np.abs(W), axis=0)
        keep = np.argsort(-score)[:cap]
        keep_sorted = np.sort(keep)
        return keep_sorted, np.asarray(names)[keep_sorted]

    g_keep, g_names = _pick(Wg, prs_columns, max_columns_per_side)
    e_keep, e_names = _pick(We, ehr_columns, max_columns_per_side)
    Wg_show = Wg[:, g_keep]
    We_show = We[:, e_keep]
    if ehr_kinds is not None:
        e_kinds_show = np.asarray(ehr_kinds)[e_keep]
    else:
        e_kinds_show = None

    H = np.concatenate([Wg_show, We_show], axis=1)
    vmax = np.max(np.abs(H)) if H.size else 1.0
    n_g_cols = Wg_show.shape[1]

    n_rows = H.shape[0]
    n_cols = H.shape[1]
    fig_w = max(8.0, 0.18 * n_cols + 4.0)
    fig_h = max(4.5, 0.30 * n_rows + 2.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.05, 1.0],
        width_ratios=[1.0, 0.10],
        hspace=0.02, wspace=0.02,
    )
    ax_kind = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_rg = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ---- kind / side bar above heatmap ------------------------------------
    kind_row = np.zeros((1, n_cols, 3), dtype=float)
    # genome side: solid genome color
    kind_row[0, :n_g_cols] = matplotlib.colors.to_rgb(GENOME_COLOR)
    # EHR side: per-column kind color or default
    for j in range(n_g_cols, n_cols):
        if e_kinds_show is not None:
            kind = str(e_kinds_show[j - n_g_cols])
            kind_row[0, j] = matplotlib.colors.to_rgb(
                EHR_KIND_COLORS.get(kind, EHR_COLOR)
            )
        else:
            kind_row[0, j] = matplotlib.colors.to_rgb(EHR_COLOR)
    ax_kind.imshow(kind_row, aspect="auto", interpolation="nearest")
    ax_kind.set_xticks([])
    ax_kind.set_yticks([])
    for spine in ax_kind.spines.values():
        spine.set_visible(False)
    ax_kind.text(n_g_cols / 2 - 0.5, 0, "  genome (PRS)",
                 ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold")
    ax_kind.text((n_g_cols + n_cols) / 2 - 0.5, 0, "EHR",
                 ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold")

    # ---- main heatmap -----------------------------------------------------
    im = ax_main.imshow(
        H, aspect="auto", cmap=DIVERGING_CMAP,
        norm=Normalize(vmin=-vmax, vmax=vmax), interpolation="nearest",
    )
    ax_main.axvline(n_g_cols - 0.5, color="black", linewidth=1.0)
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels(
        [_short(n, 24) for n in feat_names], fontsize=8.5,
        color=TEXT_COLOR,
    )
    col_names = list(g_names) + list(e_names)
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels(
        [_short(c, 18) for c in col_names],
        rotation=70, ha="right", fontsize=7, color=TEXT_COLOR,
    )
    ax_main.tick_params(axis="x", length=2, pad=2)
    ax_main.tick_params(axis="y", length=2)
    for spine in ax_main.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax_main, orientation="vertical",
                        pad=0.10, fraction=0.025, shrink=0.7)
    cbar.set_label("decoder weight", fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=AXIS_COLOR, labelsize=8)

    # ---- right strip: r_G per row ----------------------------------------
    ax_rg.barh(
        range(n_rows), r_g_promoted,
        color=[GENOME_COLOR if x >= 0.5 else EHR_COLOR for x in r_g_promoted],
        edgecolor="white", linewidth=0.4, height=0.85,
    )
    ax_rg.axvline(0.5, color=AXIS_COLOR, linewidth=0.6, alpha=0.6,
                  linestyle=":")
    ax_rg.invert_yaxis()
    ax_rg.set_xlim(0, 1.0)
    ax_rg.set_xticks([0, 0.5, 1.0])
    ax_rg.set_xticklabels(["0", "0.5", "1"], fontsize=8)
    ax_rg.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_rg.spines[spine].set_visible(False)
    ax_rg.spines["bottom"].set_color(AXIS_COLOR)
    ax_rg.spines["bottom"].set_linewidth(0.8)
    ax_rg.set_xlabel("$r_G$", fontsize=9, color=TEXT_COLOR)

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

    def _r2_per_col(target: np.ndarray, recon: np.ndarray) -> np.ndarray:
        ss_res = np.sum((target - recon) ** 2, axis=0)
        ss_tot = np.sum(target ** 2, axis=0)  # mean ~0 because z-scored
        ss_tot = np.where(ss_tot > 1e-12, ss_tot, 1.0)
        return 1.0 - ss_res / ss_tot

    r2_g = _r2_per_col(a_z, a_hat)
    r2_e = _r2_per_col(b_z, b_hat)
    mean_g = float(np.mean(r2_g)) if r2_g.size else float("nan")
    mean_e = float(np.mean(r2_e)) if r2_e.size else float("nan")

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
    if ehr_kinds is not None and len(ehr_kinds) == r2_e.size:
        kinds = np.asarray(ehr_kinds)
        kind_order = sorted(np.unique(kinds).tolist())
        positions: list[float] = []
        labels: list[str] = []
        bp_data: list[np.ndarray] = []
        for i, kind in enumerate(kind_order):
            sel = kinds == kind
            bp_data.append(r2_e[sel])
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
            f"  mean R² = {mean_e:.3f}",
            ha="right", va="bottom", fontsize=8.5, color=PROMOTED_COLOR,
        )
        ax_e.set_ylabel("R²")
        ax_e.set_title("EHR side: per-column reconstruction by kind",
                       loc="left", fontsize=11, fontweight="bold")
        _faint_grid(ax_e, axis="y")
    else:
        order = np.argsort(r2_e)
        n = r2_e.size
        if n <= 2 * max_ehr_rows:
            keep = order
        else:
            keep = np.concatenate([order[:max_ehr_rows], order[-max_ehr_rows:]])
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
            f"  mean R² = {mean_e:.3f}",
            ha="left", va="top", fontsize=8.5, color=PROMOTED_COLOR,
        )
        ax_e.set_xlabel("R²")
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
        ss_tot_a = np.maximum(np.sum(a_z ** 2, axis=0), 1e-12)
        r2_g_arr = 1.0 - ss_res_a / ss_tot_a
        ss_res_b = np.sum((b_z - b_hat) ** 2, axis=0)
        ss_tot_b = np.maximum(np.sum(b_z ** 2, axis=0), 1e-12)
        r2_e_arr = 1.0 - ss_res_b / ss_tot_b
        bp = ax_d.boxplot(
            [r2_g_arr, r2_e_arr],
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
            j = rng.uniform(-0.18, 0.18, size=vals.size)
            ax_d.scatter(np.full(vals.size, i) + j, vals, s=8,
                         color=_darken([GENOME_COLOR, EHR_COLOR][i], 0.25),
                         alpha=0.55, edgecolors="none")
        ax_d.set_xticks([0, 1])
        ax_d.set_xticklabels(
            [f"genome\nm={r2_g_arr.size}", f"EHR\nm={r2_e_arr.size}"],
            fontsize=9.5,
        )
        ax_d.set_ylabel("R² per column")
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
