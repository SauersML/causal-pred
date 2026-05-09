"""Render PNG/PDF figures from saved pipeline outputs.

Reads the standard files under ``outputs/`` and writes figures to
``outputs/plots/`` as standalone figure files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

from causal_pred import plots  # noqa: E402
from causal_pred.pipeline import _known_edges_for_columns  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as fh:
        return json.load(fh)


def _load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False)


def _save(fig, plots_dir: Path, stem: str) -> tuple[str, str]:
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    png = plots_dir / f"{stem}.png"
    pdf = plots_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)


def _survival_fan_samples(outputs_dir: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    t_grid = _load_npy(outputs_dir / "survival_time_grid.npy")
    mean = _load_npy(outputs_dir / "survival_mean.npy")
    lower = _load_npy(outputs_dir / "survival_lower.npy")
    upper = _load_npy(outputs_dir / "survival_upper.npy")
    if t_grid is None or mean is None or mean.ndim != 2 or mean.shape[0] == 0:
        return None, None
    center = mean[0]
    if lower is None or upper is None or lower.shape != mean.shape or upper.shape != mean.shape:
        samples = np.repeat(center[None, :], 5, axis=0)
    else:
        lo = lower[0]
        hi = upper[0]
        samples = np.vstack(
            [
                lo,
                0.5 * (lo + center),
                center,
                0.5 * (center + hi),
                hi,
            ]
        )
    return t_grid, np.clip(samples, 0.0, 1.0)


def main() -> int:
    outputs_dir = OUTPUTS_DIR
    plots_dir = PLOTS_DIR
    summary = _load_json(outputs_dir / "summary.json")
    columns = [str(c) for c in summary.get("columns", [])]
    node_types = [str(t) for t in summary.get("node_types", [])]
    target_node = str(summary.get("target_node", ""))
    saved: dict[str, tuple[str, str]] = {}

    edge_probs = _load_npy(outputs_dir / "mcmc_edge_probs.npy")
    if edge_probs is not None and columns:
        known_edges = _known_edges_for_columns(columns)
        fig = plots.edge_probability_heatmap(
            edge_probs=edge_probs,
            node_names=columns,
            title="posterior edge probabilities",
            ground_truth=known_edges,
        )
        saved["edge_heatmap"] = _save(fig, plots_dir, "edge_heatmap")
        if known_edges:
            fig = plots.edge_prp_curves(edge_probs, known_edges, columns)
            saved["edge_prp_curves"] = _save(fig, plots_dir, "edge_prp_curves")
        if target_node in columns:
            fig = plots.causal_pathway_sankey(edge_probs, columns, target_node)
            saved["causal_pathway_sankey"] = _save(
                fig,
                plots_dir,
                "causal_pathway_sankey",
            )

    adjacency = _load_npy(outputs_dir / "thresholded_adjacency.npy")
    if adjacency is not None and columns:
        fig = plots.dag_graph(
            adjacency,
            columns,
            node_types=node_types if len(node_types) == len(columns) else None,
            edge_probs=edge_probs,
        )
        saved["dag_graph"] = _save(fig, plots_dir, "dag_graph")

    survival = summary.get("validation", {}).get("survival", {})
    td = survival.get("time_dependent_auc", {}) if isinstance(survival, dict) else {}
    if isinstance(td, dict) and "times" in td and "auc" in td:
        fig = plots.time_dependent_auc_curve(td["times"], td["auc"])
        saved["time_dependent_auc"] = _save(fig, plots_dir, "time_dependent_auc")

    br = survival.get("brier", {}) if isinstance(survival, dict) else {}
    if isinstance(br, dict) and "times" in br and "brier" in br:
        fig = plots.brier_curve(
            br["times"],
            br["brier"],
            brier_baseline=br.get("brier_km"),
        )
        saved["brier_curve"] = _save(fig, plots_dir, "brier_curve")

    t_grid, fan_samples = _survival_fan_samples(outputs_dir)
    if t_grid is not None and fan_samples is not None:
        fig = plots.survival_fan(t_grid, fan_samples, individual_id=0)
        saved["survival_fan"] = _save(fig, plots_dir, "survival_fan")

    print(f"Figures written to {plots_dir}: {len(saved)}")
    for name, (png, pdf) in sorted(saved.items()):
        print(f"{name}: {png} {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
