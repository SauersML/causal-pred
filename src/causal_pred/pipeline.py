"""End-to-end causal-prediction pipeline (single, no-args path).

    cohort wide CSV  ->  DAGSLAM hill-climb  ->  structure MCMC  ->  artefacts

The pipeline takes no arguments. It resolves the cohort CSV through
:func:`causal_pred.data.cohort.resolve_cohort_csv` (local-then-bucket
cache), runs DAGSLAM to get a MAP DAG, runs structure MCMC for posterior
edge probabilities, and writes everything under :data:`DEFAULT_OUTPUT_DIR`.

There are no fallbacks: every stage either succeeds or raises. Errors
propagate to the caller rather than being substituted with placeholder
arrays.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data.cohort import load_cohort_dataset, resolve_cohort_csv
from .data.synthetic import SyntheticDataset
from .dagslam import run_dagslam
from .mcmc import run_structure_mcmc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")

THRESHOLD_DEFAULT = 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    columns: Tuple[str, ...] = field(default_factory=tuple)
    node_types: Tuple[str, ...] = field(default_factory=tuple)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    dagslam_adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    dagslam_log_score: float = 0.0
    dagslam_n_edges: int = 0
    mcmc_edge_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mcmc_diagnostics: Dict[str, Any] = field(default_factory=dict)
    thresholded_adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    threshold: float = THRESHOLD_DEFAULT
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("causal_pred.pipeline")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_pipeline(
    seed: int = 20260416,
    verbose: bool = False,
    max_parents: int = 3,
    max_iter: int = 500,
    restarts: int = 3,
    mcmc_samples: int = 1500,
    mcmc_burn_in: int = 500,
    mcmc_thin: int = 10,
    mcmc_chains: int = 4,
    threshold: float = THRESHOLD_DEFAULT,
    cache_dir: str = ".",
    bucket: Optional[str] = None,
    cohort_name: str = "complete",
) -> PipelineResult:
    """Cohort CSV -> DAGSLAM -> structure MCMC -> result."""
    logger = _setup_logger(verbose)
    rng = np.random.default_rng(seed)
    timings: Dict[str, float] = {}

    # -- data ---------------------------------------------------------
    t0 = time.time()
    csv_path = resolve_cohort_csv(
        name=cohort_name, cache_dir=cache_dir, bucket=bucket
    )
    data: SyntheticDataset = load_cohort_dataset(str(csv_path))
    timings["data"] = time.time() - t0
    logger.info(
        "[data] path=%s n=%d p=%d columns=%s",
        csv_path,
        data.n,
        data.p,
        list(data.columns),
    )

    # -- dagslam ------------------------------------------------------
    t0 = time.time()
    dagslam = run_dagslam(
        data=data.X,
        node_types=data.node_types,
        max_parents=max_parents,
        max_iter=max_iter,
        restarts=restarts,
        rng=rng,
        verbose=verbose,
    )
    timings["dagslam"] = time.time() - t0
    logger.info(
        "[dagslam] log_score=%.3f n_edges=%d",
        float(dagslam.log_score),
        int(dagslam.n_edges),
    )

    # -- structure mcmc ----------------------------------------------
    # Uniform Bernoulli(0.5) edge prior: NaN -> 0.5 inside run_structure_mcmc.
    pi_prior = np.full((data.p, data.p), np.nan)
    t0 = time.time()
    mcmc = run_structure_mcmc(
        data=data.X,
        node_types=data.node_types,
        start_adj=dagslam.adjacency,
        pi_prior=pi_prior,
        n_samples=mcmc_samples,
        burn_in=mcmc_burn_in,
        thin=mcmc_thin,
        n_chains=mcmc_chains,
        rng=rng,
        progress=verbose,
    )
    timings["mcmc"] = time.time() - t0
    edge_probs = np.asarray(mcmc.edge_probs, dtype=float)
    logger.info(
        "[mcmc] accept_overall=%.3f max_rhat_skel=%.3f min_ess=%.1f",
        float(mcmc.diagnostics["accept_rate"]["overall"]),
        float(mcmc.diagnostics.get("max_rhat_skeleton", float("nan"))),
        float(mcmc.diagnostics.get("min_ess", float("nan"))),
    )

    # -- threshold posterior into a DAG ------------------------------
    thresholded = (edge_probs >= threshold).astype(int)
    np.fill_diagonal(thresholded, 0)

    return PipelineResult(
        columns=tuple(data.columns),
        node_types=tuple(data.node_types),
        data_summary={
            "n": int(data.n),
            "p": int(data.p),
            "columns": list(data.columns),
            "node_types": list(data.node_types),
            "csv_path": str(csv_path),
        },
        dagslam_adjacency=np.asarray(dagslam.adjacency, dtype=int),
        dagslam_log_score=float(dagslam.log_score),
        dagslam_n_edges=int(dagslam.n_edges),
        mcmc_edge_probs=edge_probs,
        mcmc_diagnostics=dict(mcmc.diagnostics),
        thresholded_adjacency=thresholded,
        threshold=float(threshold),
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _json_sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size <= 400:
            return obj.tolist()
        return {
            "__omitted_ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def _adj_to_edge_list(adj: np.ndarray, columns: Sequence[str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if int(adj[i, j]) == 1:
                rows.append((parent, child))
    return rows


def _edge_prob_long(
    edge_probs: np.ndarray, columns: Sequence[str]
) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if i == j:
                continue
            rows.append((parent, child, float(edge_probs[i, j])))
    rows.sort(key=lambda r: -r[2])
    return rows


def save_result(
    result: PipelineResult,
    outdir: str = DEFAULT_OUTPUT_DIR,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Serialise the pipeline artefacts into ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    paths: Dict[str, str] = {}
    columns = list(result.columns)

    # --- numpy adjacency / edge-prob arrays ---
    paths["dagslam_adjacency"] = os.path.join(outdir, "dagslam_adjacency.npy")
    np.save(paths["dagslam_adjacency"], result.dagslam_adjacency)
    paths["mcmc_edge_probs"] = os.path.join(outdir, "mcmc_edge_probs.npy")
    np.save(paths["mcmc_edge_probs"], result.mcmc_edge_probs)
    paths["thresholded_adjacency"] = os.path.join(outdir, "thresholded_adjacency.npy")
    np.save(paths["thresholded_adjacency"], result.thresholded_adjacency)

    # --- CSVs (parent/child format) ---
    paths["greedy_edges_csv"] = os.path.join(outdir, "greedy_edges.csv")
    with open(paths["greedy_edges_csv"], "w") as fh:
        fh.write("parent,child\n")
        for parent, child in _adj_to_edge_list(result.dagslam_adjacency, columns):
            fh.write(f"{parent},{child}\n")

    paths["mcmc_thresholded_edges_csv"] = os.path.join(
        outdir, "mcmc_thresholded_edges.csv"
    )
    with open(paths["mcmc_thresholded_edges_csv"], "w") as fh:
        fh.write("parent,child\n")
        for parent, child in _adj_to_edge_list(result.thresholded_adjacency, columns):
            fh.write(f"{parent},{child}\n")

    paths["mcmc_edge_probabilities_long_csv"] = os.path.join(
        outdir, "mcmc_edge_probabilities_long.csv"
    )
    with open(paths["mcmc_edge_probabilities_long_csv"], "w") as fh:
        fh.write("parent,child,posterior_edge_probability\n")
        for parent, child, prob in _edge_prob_long(result.mcmc_edge_probs, columns):
            fh.write(f"{parent},{child},{prob}\n")

    # --- summary.json ---
    diagnostics = {
        k: v
        for k, v in result.mcmc_diagnostics.items()
        if k not in {"rhat_per_edge", "rhat_per_skeleton_edge", "ess_per_edge"}
    }
    summary = {
        "columns": columns,
        "node_types": list(result.node_types),
        "data_summary": result.data_summary,
        "dagslam_log_score": result.dagslam_log_score,
        "dagslam_n_edges": result.dagslam_n_edges,
        "mcmc_diagnostics": diagnostics,
        "threshold": result.threshold,
        "timings": result.timings,
    }
    paths["summary_json"] = os.path.join(outdir, "summary.json")
    with open(paths["summary_json"], "w") as fh:
        json.dump(_json_sanitise(summary), fh, indent=2, sort_keys=True)

    if run_config is None:
        run_config = {}
    paths["run_config_json"] = os.path.join(outdir, "run_config.json")
    with open(paths["run_config_json"], "w") as fh:
        json.dump(_json_sanitise(run_config), fh, indent=2, sort_keys=True)

    return paths


# ---------------------------------------------------------------------------
# Entry point (no arguments)
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Single-path entry point: cohort CSV -> DAGSLAM -> MCMC -> artefacts."""
    del argv  # unused; this entry point takes no arguments
    result = run_pipeline()
    save_result(result, outdir=DEFAULT_OUTPUT_DIR, run_config={})
    print(f"\nArtefacts written to {DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
