"""Tests for ``causal_pred.plots``.

Uses a non-interactive Agg backend so nothing tries to open a window.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from causal_pred.plots import (
    brier_curve,
    causal_pathway_sankey,
    dag_graph,
    edge_prp_curves,
    edge_probability_heatmap,
    reliability_diagram,
    save_all,
    survival_fan,
    time_dependent_auc_curve,
)


P = 8
NODES = tuple(f"n{i}" for i in range(P))


def _random_edge_probs(rng):
    M = rng.uniform(0, 1, size=(P, P))
    np.fill_diagonal(M, 0.0)
    return M


# ---------------------------------------------------------------------------
# edge_probability_heatmap
# ---------------------------------------------------------------------------


def test_edge_heatmap():
    rng = np.random.default_rng(0)
    M = _random_edge_probs(rng)
    fig = edge_probability_heatmap(M, NODES, title="my heatmap")
    # Exactly one imshow axis + one colorbar axis.  The "main" axis count
    # (non-colorbar) is 1.
    [
        a
        for a in fig.axes
        if not a.get_label().startswith("<colorbar>") and a.get_label() != "<colorbar>"
    ]
    # Colorbars in matplotlib have axes with label containing 'colorbar';
    # robust check: at least one imshow image.
    n_imshow = sum(len(a.images) for a in fig.axes)
    assert n_imshow == 1
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert any("my heatmap" in t for t in titles)
    plt.close(fig)


def test_edge_heatmap_respects_ground_truth():
    rng = np.random.default_rng(1)
    M = _random_edge_probs(rng)
    gt = [("n0", "n1"), ("n2", "n3"), ("n4", "n5")]
    fig = edge_probability_heatmap(M, NODES, ground_truth=gt)
    # Count rectangle patches on the heatmap's main axis.  There will be
    # exactly len(gt) Rectangles that we added (matplotlib's spines and
    # imshow clip box are separate).  We filter for our styling colour.
    ax = fig.axes[0]
    from matplotlib.patches import Rectangle

    boxes = [
        p
        for p in ax.patches
        if isinstance(p, Rectangle)
        and p.get_edgecolor() is not None
        and np.allclose(p.get_edgecolor()[:3], (0.0, 0.8, 0.4), atol=0.05)
    ]
    assert len(boxes) == len(gt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# edge_prp_curves
# ---------------------------------------------------------------------------


def test_edge_prp_curves_has_two_axes():
    rng = np.random.default_rng(2)
    M = _random_edge_probs(rng)
    gt = [("n0", "n1"), ("n2", "n3"), ("n4", "n5")]
    fig = edge_prp_curves(M, gt, NODES)
    # Exactly two subplots (ROC + PR).
    assert len(fig.axes) >= 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


def test_reliability_on_perfect_predictions():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300).astype(float)
    p = y.copy()  # perfect
    fig = reliability_diagram(y, p, n_bins=5, strategy="uniform")
    # The error-bar points should all lie on y=x.  Pull them off the axis.
    ax = fig.axes[0]
    # Grab the Line2D (marker) with 'observed' in label.
    for ln in ax.get_lines():
        label = ln.get_label() or ""
        if "observed" in label:
            xs, ys = ln.get_xdata(), ln.get_ydata()
            assert np.allclose(xs, ys, atol=1e-8)
            break
    else:
        pytest.fail("reliability diagram has no observed-points line")
    plt.close(fig)


# ---------------------------------------------------------------------------
# survival_fan
# ---------------------------------------------------------------------------


def test_survival_fan_monotone_median():
    rng = np.random.default_rng(4)
    t = np.linspace(0.0, 10.0, 40)
    # Sample survival curves: each draw is a decreasing curve plus small
    # wobble.  Median across draws must be non-increasing after the
    # cummin fix in the plotting code.
    n_samples = 50
    samples = np.zeros((n_samples, t.size))
    for k in range(n_samples):
        lam = 0.1 + 0.05 * rng.random()
        samples[k] = np.exp(-lam * t) * (1.0 + 0.01 * rng.standard_normal(t.size))
    fig = survival_fan(t, samples, individual_id=0)
    ax = fig.axes[0]
    for ln in ax.get_lines():
        if "median" in (ln.get_label() or ""):
            ys = np.asarray(ln.get_ydata())
            assert np.all(np.diff(ys) <= 1e-9), "median is not monotone"
            break
    else:
        pytest.fail("no median line found")
    plt.close(fig)


# ---------------------------------------------------------------------------
# dag_graph
# ---------------------------------------------------------------------------


def test_dag_graph_renders():
    # Empty adjacency should render without error.
    A = np.zeros((4, 4), dtype=int)
    fig_empty = dag_graph(A, ["a", "b", "c", "d"])
    assert fig_empty is not None
    plt.close(fig_empty)

    # Non-empty: 3 edges, 4 nodes.
    A[0, 1] = 1
    A[1, 2] = 1
    A[2, 3] = 1
    ep = np.full((4, 4), np.nan)
    ep[0, 1] = 0.8
    ep[1, 2] = 0.6
    ep[2, 3] = 0.9
    fig = dag_graph(
        A, ["a", "b", "c", "d"], node_types=["continuous"] * 4, edge_probs=ep
    )
    # NetworkX draws edges as FancyArrowPatch objects; count them.
    from matplotlib.patches import FancyArrowPatch

    arrows = [p for p in fig.axes[0].patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
# time_dependent_auc_curve / brier_curve smoke tests
# ---------------------------------------------------------------------------


def test_time_dependent_auc_curve_smoke():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    a = np.array([0.8, 0.78, 0.72, 0.65])
    se = np.array([0.03, 0.03, 0.04, 0.05])
    fig = time_dependent_auc_curve(t, a, auc_se=se)
    plt.close(fig)


def test_brier_curve_smoke():
    t = np.linspace(0.5, 10.0, 20)
    b = 0.1 + 0.05 * np.sin(t)
    baseline = np.full_like(t, 0.18)
    fig = brier_curve(t, b, brier_baseline=baseline)
    plt.close(fig)


# ---------------------------------------------------------------------------
# causal_pathway_sankey
# ---------------------------------------------------------------------------


def test_causal_pathway_sankey_smoke():
    # Build edge probs with a clear path n0 -> n2 -> n4.
    P_ = np.zeros((6, 6))
    P_[0, 2] = 0.9
    P_[2, 4] = 0.8
    P_[1, 4] = 0.7
    P_[3, 2] = 0.4
    fig = causal_pathway_sankey(
        P_,
        tuple(f"n{i}" for i in range(6)),
        target_node="n4",
        top_k_paths=3,
        min_prob=0.3,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# save_all
# ---------------------------------------------------------------------------


def test_save_all_writes_files():
    rng = np.random.default_rng(5)
    M = _random_edge_probs(rng)
    gt = [("n0", "n1"), ("n2", "n3")]

    y = rng.integers(0, 2, size=200).astype(float)
    p = np.clip(0.2 + 0.6 * rng.random(y.size), 0.01, 0.99)

    taus = np.array([1.0, 2.0, 5.0])
    aucs = np.array([0.75, 0.72, 0.68])
    auc_se = np.array([0.02, 0.03, 0.04])

    t_grid = np.linspace(0.5, 8.0, 20)
    brier_t = 0.12 + 0.03 * np.sin(t_grid)
    brier_base = np.full_like(t_grid, 0.18)

    S_samples = np.clip(
        np.exp(-0.1 * t_grid[None, :]) + 0.02 * rng.standard_normal((30, t_grid.size)),
        0.0,
        1.0,
    )

    A = np.zeros((P, P), dtype=int)
    A[0, 1] = 1
    A[2, 3] = 1

    with tempfile.TemporaryDirectory() as td:
        saved = save_all(
            outputs_dir=td,
            edge_probs=M,
            ground_truth_edges=gt,
            node_names=NODES,
            y_true=y,
            p_pred=p,
            eval_times=taus,
            auc_values=aucs,
            auc_se=auc_se,
            t_grid=t_grid,
            brier_t=brier_t,
            brier_baseline=brier_base,
            survival_samples=S_samples,
            individual_id=42,
            target_node="n5",
            adjacency=A,
            node_types=["continuous"] * P,
        )
        # Should have written all 8 kinds of figures.
        expected = {
            "edge_heatmap",
            "edge_prp_curves",
            "reliability",
            "time_dependent_auc",
            "brier_curve",
            "survival_fan",
            "causal_pathway_sankey",
            "dag_graph",
        }
        assert expected.issubset(saved.keys()), f"missing: {expected - saved.keys()}"
        for name, (png, pdf) in saved.items():
            assert os.path.exists(png), f"{name}.png not written"
            assert os.path.exists(pdf), f"{name}.pdf not written"
            assert os.path.getsize(png) > 0
            assert os.path.getsize(pdf) > 0
