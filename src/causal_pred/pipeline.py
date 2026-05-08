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
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data.cohort import load_cohort_dataset_with_person_ids, resolve_cohort_csv
from .data.synthetic import SyntheticDataset
from .dagslam import run_dagslam
from .mcmc import run_structure_mcmc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
DEFAULT_PRS_PATH = os.path.join(DEFAULT_CACHE_DIR, "aou_prs_panel.csv.gz")

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
    genscore_features: Dict[str, Any] = field(default_factory=dict)


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


def _read_prs_panel(path: str | os.PathLike) -> pd.DataFrame:
    """Read the cached AoU PRS panel written by ``prepare_aou_prs.py``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PRS panel not found: {p}")
    df = pd.read_csv(p, dtype={0: "string"})
    index_col = "person_id" if "person_id" in df.columns else df.columns[0]
    out = df.set_index(index_col)
    out.index = out.index.astype(str)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.shape[1] == 0:
        raise ValueError(f"PRS panel {p} has no score columns")
    return out


def _prs_node_name(original: str, used: set[str]) -> str:
    stem = re.sub(r"[^0-9A-Za-z]+", "_", str(original)).strip("_").lower()
    if not stem:
        stem = "score"
    if not stem.startswith("pgs_"):
        stem = f"pgs_{stem}"
    stem = stem[:48].rstrip("_")
    name = stem
    i = 2
    while name in used:
        suffix = f"_{i}"
        name = f"{stem[:48 - len(suffix)]}{suffix}"
        i += 1
    used.add(name)
    return name


def _augment_with_prs_nodes(
    data: SyntheticDataset,
    person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    *,
    n_prs_nodes: int,
    max_missing: float,
) -> tuple[SyntheticDataset, np.ndarray, Dict[str, Any]]:
    """Append selected microarray-derived PRS columns as continuous DAG nodes."""
    if n_prs_nodes <= 0:
        raise ValueError("n_prs_nodes must be positive when a PRS panel is supplied")
    if not 0.0 <= max_missing < 1.0:
        raise ValueError("max_missing must be in [0, 1)")
    if len(person_ids) != data.n:
        raise ValueError(
            f"person_ids has length {len(person_ids)} but dataset has {data.n} rows"
        )
    pid = np.asarray([str(p) for p in person_ids])
    aligned = prs_df.reindex(pid)

    candidates: list[tuple[float, str, np.ndarray]] = []
    for col in aligned.columns:
        vals = pd.to_numeric(aligned[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        present_rate = float(finite.mean())
        if present_rate < 1.0 - max_missing:
            continue
        if int(finite.sum()) < 100:
            continue
        v = vals[finite]
        sd = float(v.std(ddof=0))
        if sd == 0.0:
            continue
        candidates.append((present_rate, str(col), vals))

    if not candidates:
        raise ValueError(
            "no PRS columns have enough non-missing, non-constant data after "
            "alignment with the cohort"
        )

    chosen = candidates[: min(n_prs_nodes, len(candidates))]

    keep = np.ones(data.n, dtype=bool)
    for _present, _col, vals in chosen:
        keep &= np.isfinite(vals)
    if int(keep.sum()) < 100:
        raise ValueError(
            f"only {int(keep.sum())} rows have complete selected PRS values"
        )

    prs_cols = []
    original_names = []
    present_rates = []
    used_names = set(data.columns)
    for present, col, vals in chosen:
        raw = vals[keep].astype(float)
        sd = float(raw.std(ddof=0))
        if sd == 0.0:
            raise ValueError(f"selected PRS column became constant after row filter: {col}")
        prs_cols.append((raw - float(raw.mean())) / sd)
        original_names.append(col)
        present_rates.append(present)

    X_keep = data.X[keep]
    X_prs = np.column_stack(prs_cols).astype(np.float64, copy=False)
    X_aug = np.concatenate([X_keep, X_prs], axis=1)
    prs_names = tuple(_prs_node_name(c, used_names) for c in original_names)
    p_old = data.p
    p_new = X_aug.shape[1]
    gt = np.zeros((p_new, p_new), dtype=int)
    if data.ground_truth_adj.size:
        gt[:p_old, :p_old] = data.ground_truth_adj

    augmented = SyntheticDataset(
        X=X_aug,
        time=data.time[keep].copy(),
        event=data.event[keep].copy(),
        columns=tuple(data.columns) + prs_names,
        node_types=tuple(data.node_types) + ("continuous",) * len(prs_names),
        ground_truth_adj=gt,
    )
    meta = {
        "prs_rows_before_alignment": int(data.n),
        "prs_rows_after_alignment": int(keep.sum()),
        "prs_columns_available": int(prs_df.shape[1]),
        "prs_columns_selected": int(len(prs_names)),
        "prs_original_names": original_names,
        "prs_node_names": list(prs_names),
        "prs_present_rate": present_rates,
        "prs_max_missing": float(max_missing),
        "prs_selection_rule": "first_complete_nonconstant_scores_from_curated_panel",
    }
    return augmented, pid[keep], meta



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
    cache_dir: str = DEFAULT_CACHE_DIR,
    bucket: Optional[str] = None,
    cohort_name: str = "complete",
    prs_path: str = DEFAULT_PRS_PATH,
    n_prs_nodes: int = 8,
    prs_max_missing: float = 0.2,
) -> PipelineResult:
    """Cohort CSV + cached AoU microarray PRS -> DAGSLAM -> structure MCMC.

    ``scripts/prepare_aou_prs.py`` must run first. It scores the AoU
    microarray PLINK files with gnomon and writes ``prs_path``. This function
    appends the first complete, non-constant PRS columns from the curated panel
    as continuous DAG nodes before structure learning.
    """
    logger = _setup_logger(verbose)
    rng = np.random.default_rng(seed)
    timings: Dict[str, float] = {}
    genscore_features: Dict[str, Any] = {}

    # -- data ---------------------------------------------------------
    t0 = time.time()
    csv_path = resolve_cohort_csv(
        name=cohort_name, cache_dir=cache_dir, bucket=bucket
    )
    data, person_ids = load_cohort_dataset_with_person_ids(str(csv_path))
    timings["data"] = time.time() - t0
    logger.info(
        "[data] path=%s n=%d p=%d columns=%s",
        csv_path,
        data.n,
        data.p,
        list(data.columns),
    )

    # -- microarray-derived PRS nodes --------------------------------
    t0 = time.time()
    prs_df = _read_prs_panel(prs_path)
    data, _kept_person_ids, prs_meta = _augment_with_prs_nodes(
        data,
        person_ids,
        prs_df,
        n_prs_nodes=n_prs_nodes,
        max_missing=prs_max_missing,
    )
    timings["prs"] = time.time() - t0
    genscore_features.update({"prs_path": str(prs_path), **prs_meta})
    logger.info(
        "[prs] path=%s selected=%d n=%d p=%d nodes=%s",
        prs_path,
        int(prs_meta["prs_columns_selected"]),
        data.n,
        data.p,
        prs_meta["prs_node_names"],
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
        genscore_features=genscore_features,
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
        "genscore_features": result.genscore_features,
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
