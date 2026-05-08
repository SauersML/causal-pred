"""Single end-to-end causal-prediction pipeline.

Path:

    cohort CSV -> microarray PRS cache -> PRS-augmented matrix
      -> MrDAG priors -> DAGSLAM -> structure MCMC -> artefacts

The public runner takes no configuration arguments. Operational settings live
as constants in this module so there is one production path and one place to
change it. Small reusable intermediates are cached locally under ``data/`` and,
when ``WORKSPACE_BUCKET`` is present, mirrored under
``$WORKSPACE_BUCKET/intermediates/causal-pred``. Cached files are keyed by the
actual matrix/configuration that produced them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .data.cohort import (
    EhrPanel,
    build_ehr_panel,
    discover_genotype_dir,
    fetch_omop_long_frames,
    load_cohort_dataset_with_person_ids,
    resolve_aou_genotypes,
    resolve_baseline_dt,
    resolve_cohort_csv,
)
from .data.nodes import CANONICAL_EDGES, NODE_INDEX
from .data.polygenic import score_panel
from .data.real_gwas import load_real_gwas
from .data.synthetic import SyntheticDataset
from .dagslam import run_dagslam
from .gam.survival import fit_survival_gam
from .genscore.integrate import run_genscore
from .genscore.panels import download_panel
from .mcmc import run_structure_mcmc
from .mrdag import run_mrdag
from .validation.known_edges import known_edge_recovery
from .validation.metrics import (
    brier_score,
    calibration_metrics,
    nagelkerke_r2,
    time_dependent_auc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = str(REPO_ROOT / "data")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "outputs")

WORKSPACE_CACHE_PREFIX = "intermediates/causal-pred"
WORKSPACE_RESULTS_PREFIX = "results/causal-pred/latest"
PRS_PANEL_FILENAME = "aou_prs_panel.csv.gz"
PGS_PANEL_DIRNAME = "pgs_panel"
GNOMON_OUT_DIRNAME = "gnomon_score"
GENOTYPE_CACHE_DIR = str(Path.home() / "causal-pred" / "genomes")

PIPELINE_CONFIG_VERSION = "2026-05-08.single-path.2"
PIPELINE_SEED = 20260416
PIPELINE_VERBOSE = False

COHORT_NAME = "complete"
PRS_NODES = 8
PRS_MAX_MISSING = 0.2
PRS_MIN_COMPLETE_ROWS = 100
GNOMON_TIMEOUT_SECONDS = 24 * 60 * 60

MRDAG_N_ITER = 6000
MRDAG_N_CHAINS = 4
MRDAG_N_BURN = 1000
MRDAG_THIN = 5

DAGSLAM_MAX_PARENTS = 3
DAGSLAM_MAX_ITER = 500
DAGSLAM_RESTARTS = 3

MCMC_SAMPLES = 1500
MCMC_BURN_IN = 500
MCMC_THIN = 10
MCMC_CHAINS = 4
THRESHOLD_DEFAULT = 0.5

VALIDATION_N_PERMUTE = 200

GENSCORE_N_PROMOTE = 32
GENSCORE_GENOME_SHARE_MIN = 0.2
GENSCORE_GENOME_SHARE_MAX = 0.8
GENSCORE_MIN_ACTIVATION_RATE = 0.01
GENSCORE_CROSSCODER_KWARGS = {
    "d": 512,
    "k": 32,
    "n_steps": 4000,
    "batch_size": 1024,
    "lr": 3e-4,
}

GAM_N_SAMPLES = 200
SURVIVAL_TIME_GRID_POINTS = 50
SURVIVAL_EVAL_TIMES = (5.0, 10.0, 15.0)
SURVIVAL_TARGET_COLUMN = "type2_diabetes"

CAUSAL_PATH_TARGET = "type2_diabetes"
CAUSAL_PATH_TOP_K = 20
CAUSAL_PATH_MIN_EDGE_PROB = 0.1
CAUSAL_PATH_MAX_DEPTH = 5

COHORT_TO_MR_NODE = {
    "type2_diabetes": "T2D",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "ldl_cholesterol": "LDL",
    "systolic_bp": "systolic_BP",
}

MR_TO_COHORT_NODE = {v: k for k, v in COHORT_TO_MR_NODE.items()}


@dataclass
class PipelineResult:
    person_ids: tuple[str, ...] = field(default_factory=tuple)
    columns: tuple[str, ...] = field(default_factory=tuple)
    node_types: tuple[str, ...] = field(default_factory=tuple)
    data_summary: dict[str, Any] = field(default_factory=dict)
    mrdag_pi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mrdag_prior: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mrdag_diagnostics: dict[str, Any] = field(default_factory=dict)
    dagslam_adjacency: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
    dagslam_log_score: float = 0.0
    dagslam_n_edges: int = 0
    mcmc_edge_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mcmc_samples: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=int))
    mcmc_diagnostics: dict[str, Any] = field(default_factory=dict)
    thresholded_adjacency: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )
    threshold: float = THRESHOLD_DEFAULT
    survival_time_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    survival_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    survival_diagnostics: dict[str, Any] = field(default_factory=dict)
    survival_parent_columns: tuple[str, ...] = field(default_factory=tuple)
    causal_pathways: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    genscore_features: dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""


@dataclass(frozen=True)
class WorkspaceCache:
    local_root: Path
    bucket: Optional[str]

    @property
    def local_dir(self) -> Path:
        return self.local_root / WORKSPACE_CACHE_PREFIX

    def path(self, filename: str) -> Path:
        return self.local_dir / filename

    def uri(self, filename: str) -> Optional[str]:
        if self.bucket is None:
            return None
        return f"{self.bucket}/{WORKSPACE_CACHE_PREFIX}/{filename}"

    def fetch(
        self,
        filename: str,
        local_path: Optional[Path] = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        dst = local_path if local_path is not None else self.path(filename)
        if dst.is_file() and dst.stat().st_size > 0 and not overwrite:
            return dst
        uri = self.uri(filename)
        if uri is not None and _gsutil_exists(uri):
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            if part.exists():
                part.unlink()
            subprocess.run(["gsutil", "cp", uri, str(part)], check=True)
            os.replace(part, dst)
        return dst

    def store(self, src: Path, filename: Optional[str] = None) -> None:
        uri = self.uri(filename or src.name)
        if uri is not None:
            subprocess.run(["gsutil", "cp", str(src), uri], check=True)


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


def _workspace_bucket() -> Optional[str]:
    bucket = os.environ.get("WORKSPACE_BUCKET", "").rstrip("/")
    return bucket or None


def _gsutil_exists(uri: str) -> bool:
    return subprocess.run(["gsutil", "-q", "stat", uri], check=False).returncode == 0


def _cache() -> WorkspaceCache:
    return WorkspaceCache(Path(DEFAULT_CACHE_DIR), _workspace_bucket())


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    with part.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(part, path)


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


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(
        _json_sanitise(obj), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(_json_bytes(obj)).hexdigest()[:20]


def _array_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def _dataframe_values_hash(df: pd.DataFrame) -> str:
    values = np.ascontiguousarray(df.to_numpy(dtype=np.float64))
    return _array_hash(values)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pipeline_config() -> dict[str, Any]:
    return {
        "version": PIPELINE_CONFIG_VERSION,
        "seed": PIPELINE_SEED,
        "cohort_name": COHORT_NAME,
        "prs_nodes": PRS_NODES,
        "prs_max_missing": PRS_MAX_MISSING,
        "prs_min_complete_rows": PRS_MIN_COMPLETE_ROWS,
        "genscore": {
            "n_promote": GENSCORE_N_PROMOTE,
            "genome_share_min": GENSCORE_GENOME_SHARE_MIN,
            "genome_share_max": GENSCORE_GENOME_SHARE_MAX,
            "min_activation_rate": GENSCORE_MIN_ACTIVATION_RATE,
            "crosscoder_kwargs": GENSCORE_CROSSCODER_KWARGS,
        },
        "mrdag": {
            "n_iter": MRDAG_N_ITER,
            "n_chains": MRDAG_N_CHAINS,
            "n_burn": MRDAG_N_BURN,
            "thin": MRDAG_THIN,
        },
        "dagslam": {
            "max_parents": DAGSLAM_MAX_PARENTS,
            "max_iter": DAGSLAM_MAX_ITER,
            "restarts": DAGSLAM_RESTARTS,
        },
        "mcmc": {
            "samples": MCMC_SAMPLES,
            "burn_in": MCMC_BURN_IN,
            "thin": MCMC_THIN,
            "chains": MCMC_CHAINS,
        },
        "threshold": THRESHOLD_DEFAULT,
        "gam": {
            "n_samples": GAM_N_SAMPLES,
            "time_grid_points": SURVIVAL_TIME_GRID_POINTS,
            "eval_times": list(SURVIVAL_EVAL_TIMES),
            "target_column": SURVIVAL_TARGET_COLUMN,
        },
        "causal_paths": {
            "target": CAUSAL_PATH_TARGET,
            "top_k": CAUSAL_PATH_TOP_K,
            "min_edge_prob": CAUSAL_PATH_MIN_EDGE_PROB,
            "max_depth": CAUSAL_PATH_MAX_DEPTH,
        },
    }


def _run_key(data: SyntheticDataset, mrdag_prior: np.ndarray) -> str:
    return _short_hash(
        {
            "config": _pipeline_config(),
            "columns": list(data.columns),
            "node_types": list(data.node_types),
            "x_sha256": _array_hash(data.X),
            "time_sha256": _array_hash(data.time),
            "event_sha256": _array_hash(data.event),
            "mrdag_prior_sha256": _array_hash(mrdag_prior),
        }
    )


def _genscore_key(
    data: SyntheticDataset,
    person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
) -> str:
    return _short_hash(
        {
            "config": _pipeline_config()["genscore"],
            "columns": list(data.columns),
            "node_types": list(data.node_types),
            "x_sha256": _array_hash(data.X),
            "time_sha256": _array_hash(data.time),
            "event_sha256": _array_hash(data.event),
            "person_ids": [str(p) for p in person_ids],
            "prs_columns": [str(c) for c in prs_df.columns],
            "prs_sha256": _dataframe_values_hash(prs_df),
            "ehr_features": list(ehr_panel.feature_names),
            "ehr_kinds": list(ehr_panel.feature_kinds),
            "ehr_sha256": _array_hash(ehr_panel.matrix),
        }
    )


def _read_prs_panel(path: str | os.PathLike) -> pd.DataFrame:
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


def _prs_cache_usable(path: Path, person_ids: Sequence[str]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    ids = pd.read_csv(path, usecols=[0], dtype={0: "string"}).iloc[:, 0]
    have = ids.astype("string").astype(str).to_numpy()
    want = pd.Series(person_ids, dtype="string").astype(str).to_numpy()
    return have.shape == want.shape and bool(np.array_equal(have, want))


def _cohort_person_ids(cohort_csv: Path) -> pd.Series:
    df = pd.read_csv(cohort_csv, usecols=["person_id"], dtype={"person_id": "string"})
    ids = df["person_id"].astype("string")
    if ids.empty:
        raise RuntimeError(f"cohort CSV has no rows: {cohort_csv}")
    return ids


def _resolve_microarray_bed() -> Path:
    hit = discover_genotype_dir([Path.home(), REPO_ROOT / "genomes"])
    if hit is not None:
        return hit / "arrays.bed"
    return resolve_aou_genotypes(cache_dir=GENOTYPE_CACHE_DIR)


def _build_prs_panel(cohort_csv: Path, path: Path, logger: logging.Logger) -> pd.DataFrame:
    person_ids = _cohort_person_ids(cohort_csv)
    bed = _resolve_microarray_bed()
    panel_dir = Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME
    out_dir = Path(DEFAULT_CACHE_DIR) / GNOMON_OUT_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[prs] downloading PGS scoring panel into %s", panel_dir)
    download_panel(panel_dir)
    logger.info("[prs] scoring AoU microarray genotypes at %s", bed)
    scores = score_panel(
        genotype_path=str(bed),
        score_path=str(panel_dir),
        out_dir=str(out_dir),
        n_threads=os.cpu_count(),
        timeout=GNOMON_TIMEOUT_SECONDS,
    )
    scores.index = scores.index.astype(str)

    overlap = person_ids.astype(str).isin(scores.index)
    min_required = min(PRS_MIN_COMPLETE_ROWS, int(person_ids.size))
    n_overlap = int(overlap.sum())
    if n_overlap < min_required:
        raise RuntimeError(
            f"only {n_overlap} cohort participants overlap the gnomon score output; "
            f"required at least {min_required}"
        )

    cohort_scores = scores.reindex(person_ids.astype(str)).dropna(axis=1, how="all")
    if cohort_scores.shape[1] < PRS_NODES:
        raise RuntimeError(
            f"gnomon produced {cohort_scores.shape[1]} usable PRS columns; "
            f"required at least {PRS_NODES}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    cohort_scores.to_csv(part, index_label="person_id", compression="gzip")
    os.replace(part, path)
    logger.info(
        "[prs] built %s rows=%d overlap=%d cols=%d",
        path,
        cohort_scores.shape[0],
        n_overlap,
        cohort_scores.shape[1],
    )
    return cohort_scores


def _load_or_build_prs_panel(
    cache: WorkspaceCache,
    cohort_csv: Path,
    person_ids: Sequence[str],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str]:
    path = Path(DEFAULT_CACHE_DIR) / PRS_PANEL_FILENAME
    if _prs_cache_usable(path, person_ids):
        logger.info("[prs] using local cache %s", path)
        return _read_prs_panel(path), str(path)

    cache.fetch(PRS_PANEL_FILENAME, path, overwrite=True)
    if _prs_cache_usable(path, person_ids):
        logger.info("[prs] restored workspace cache %s", path)
        return _read_prs_panel(path), str(path)

    panel = _build_prs_panel(cohort_csv, path, logger)
    cache.store(path, PRS_PANEL_FILENAME)
    return panel, str(path)


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
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any]]:
    if PRS_NODES <= 0:
        raise ValueError("PRS_NODES must be positive")
    if not 0.0 <= PRS_MAX_MISSING < 1.0:
        raise ValueError("PRS_MAX_MISSING must be in [0, 1)")
    if len(person_ids) != data.n:
        raise ValueError(
            f"person_ids has length {len(person_ids)} but dataset has {data.n} rows"
        )
    if "type2_diabetes" not in data.columns:
        raise ValueError("PRS selection requires the type2_diabetes outcome column")

    pid = np.asarray([str(p) for p in person_ids])
    aligned = prs_df.reindex(pid)
    y = data.X[:, data.columns.index("type2_diabetes")].astype(float)
    if np.nanstd(y) == 0.0:
        raise ValueError("type2_diabetes has zero variance after cohort loading")

    min_required = min(PRS_MIN_COMPLETE_ROWS, data.n)
    candidates: list[tuple[float, float, str, np.ndarray]] = []
    for col in aligned.columns:
        vals = pd.to_numeric(aligned[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        present_rate = float(finite.mean())
        if present_rate < 1.0 - PRS_MAX_MISSING:
            continue
        if int(finite.sum()) < min_required:
            continue
        v = vals[finite]
        sd = float(v.std(ddof=0))
        if sd == 0.0:
            continue
        yc = y[finite]
        if float(yc.std(ddof=0)) == 0.0:
            continue
        z = (v - float(v.mean())) / sd
        corr = float(abs(np.corrcoef(z, yc)[0, 1]))
        if np.isfinite(corr):
            candidates.append((corr, present_rate, str(col), vals))

    if len(candidates) < PRS_NODES:
        raise ValueError(
            f"only {len(candidates)} PRS columns pass completeness/variance checks; "
            f"required {PRS_NODES}"
        )

    candidates.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    chosen = candidates[:PRS_NODES]

    keep = np.ones(data.n, dtype=bool)
    for _corr, _present, _col, vals in chosen:
        keep &= np.isfinite(vals)
    if int(keep.sum()) < min_required:
        raise ValueError(
            f"only {int(keep.sum())} rows have complete selected PRS values; "
            f"required {min_required}"
        )

    prs_cols = []
    original_names = []
    correlations = []
    present_rates = []
    used_names = set(data.columns)
    for corr, present, col, vals in chosen:
        raw = vals[keep].astype(float)
        sd = float(raw.std(ddof=0))
        if sd == 0.0:
            raise ValueError(f"selected PRS column became constant after row filter: {col}")
        prs_cols.append((raw - float(raw.mean())) / sd)
        original_names.append(col)
        correlations.append(corr)
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
        "prs_abs_corr_with_t2d": correlations,
        "prs_present_rate": present_rates,
        "prs_max_missing": float(PRS_MAX_MISSING),
        "prs_min_complete_rows": int(min_required),
    }
    return augmented, pid[keep], meta


def _mrdag_cache_key() -> str:
    return _short_hash({"version": PIPELINE_CONFIG_VERSION, "mrdag": _pipeline_config()["mrdag"]})


def _load_or_run_mrdag(
    cache: WorkspaceCache, logger: logging.Logger
) -> tuple[np.ndarray, dict[str, Any]]:
    key = _mrdag_cache_key()
    filename = f"mrdag-real-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[mrdag] using cache %s", path)
            return z["pi"], json.loads(str(z["diagnostics_json"].item()))

    logger.info("[mrdag] running literature MR prior sampler")
    t0 = time.time()
    result = run_mrdag(
        load_real_gwas(),
        rng=np.random.default_rng(PIPELINE_SEED + 1),
        n_iter=MRDAG_N_ITER,
        n_chains=MRDAG_N_CHAINS,
        n_burn=MRDAG_N_BURN,
        thin=MRDAG_THIN,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["runtime_s"] = time.time() - t0
    _atomic_npz(
        path,
        pi=result.pi,
        diagnostics_json=np.array(json.dumps(_json_sanitise(diagnostics))),
    )
    cache.store(path)
    return result.pi, diagnostics


def _mrdag_prior_for_data(mrdag_pi: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    p = len(columns)
    prior = np.full((p, p), np.nan, dtype=float)
    for i, parent in enumerate(columns):
        mr_parent = COHORT_TO_MR_NODE.get(parent)
        if mr_parent is None:
            continue
        for j, child in enumerate(columns):
            mr_child = COHORT_TO_MR_NODE.get(child)
            if mr_child is None:
                continue
            pi = mrdag_pi[NODE_INDEX[mr_parent], NODE_INDEX[mr_child]]
            if np.isfinite(pi):
                prior[i, j] = float(pi)
    np.fill_diagonal(prior, np.nan)
    return prior


def _load_or_run_dagslam(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    logger: logging.Logger,
) -> dict[str, Any]:
    filename = f"dagslam-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[dagslam] using cache %s", path)
            return {
                "adjacency": z["adjacency"].astype(int),
                "log_score": float(z["log_score"].item()),
                "n_edges": int(z["n_edges"].item()),
                "runtime_s": float(z["runtime_s"].item()),
            }

    logger.info("[dagslam] running hill-climb")
    t0 = time.time()
    result = run_dagslam(
        data=data.X,
        node_types=data.node_types,
        max_parents=DAGSLAM_MAX_PARENTS,
        max_iter=DAGSLAM_MAX_ITER,
        restarts=DAGSLAM_RESTARTS,
        rng=np.random.default_rng(PIPELINE_SEED + 2),
        verbose=PIPELINE_VERBOSE,
    )
    runtime_s = time.time() - t0
    _atomic_npz(
        path,
        adjacency=np.asarray(result.adjacency, dtype=int),
        log_score=np.array(float(result.log_score)),
        n_edges=np.array(int(result.n_edges)),
        runtime_s=np.array(runtime_s),
    )
    cache.store(path)
    return {
        "adjacency": np.asarray(result.adjacency, dtype=int),
        "log_score": float(result.log_score),
        "n_edges": int(result.n_edges),
        "runtime_s": runtime_s,
    }


def _load_or_run_mcmc(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    start_adj: np.ndarray,
    pi_prior: np.ndarray,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any], float]:
    filename = f"mcmc-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[mcmc] using cache %s", path)
            return (
                z["edge_probs"],
                json.loads(str(z["diagnostics_json"].item())),
                float(z["runtime_s"].item()),
            )

    logger.info("[mcmc] running posterior structure sampler")
    t0 = time.time()
    result = run_structure_mcmc(
        data=data.X,
        node_types=data.node_types,
        start_adj=start_adj,
        pi_prior=pi_prior,
        n_samples=MCMC_SAMPLES,
        burn_in=MCMC_BURN_IN,
        thin=MCMC_THIN,
        n_chains=MCMC_CHAINS,
        rng=np.random.default_rng(PIPELINE_SEED + 3),
        progress=PIPELINE_VERBOSE,
    )
    runtime_s = time.time() - t0
    diagnostics = dict(result.diagnostics)
    _atomic_npz(
        path,
        edge_probs=np.asarray(result.edge_probs, dtype=float),
        diagnostics_json=np.array(json.dumps(_json_sanitise(diagnostics))),
        runtime_s=np.array(runtime_s),
    )
    cache.store(path)
    return np.asarray(result.edge_probs, dtype=float), diagnostics, runtime_s


def _known_edges_for_columns(columns: Sequence[str]) -> tuple[tuple[str, str], ...]:
    available = set(columns)
    edges: list[tuple[str, str]] = []
    for parent, child in CANONICAL_EDGES:
        p = MR_TO_COHORT_NODE.get(parent)
        c = MR_TO_COHORT_NODE.get(child)
        if p in available and c in available:
            edges.append((p, c))
    return tuple(edges)


def _validate_edges(
    edge_probs: np.ndarray,
    columns: Sequence[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    known_edges = _known_edges_for_columns(columns)
    if not known_edges:
        return {"known_edges": []}
    out = known_edge_recovery(
        edge_probs=edge_probs,
        ground_truth_edges=known_edges,
        node_names=columns,
        n_permute=VALIDATION_N_PERMUTE,
        rng=rng,
    )
    out["known_edges"] = [list(edge) for edge in known_edges]
    return out


def run_pipeline() -> PipelineResult:
    """Run the single production path."""
    logger = _setup_logger(PIPELINE_VERBOSE)
    cache = _cache()
    timings: dict[str, float] = {}

    t0 = time.time()
    csv_path = resolve_cohort_csv(name=COHORT_NAME, cache_dir=DEFAULT_CACHE_DIR)
    data, person_ids = load_cohort_dataset_with_person_ids(str(csv_path))
    timings["data"] = time.time() - t0
    logger.info(
        "[data] path=%s n=%d p=%d columns=%s",
        csv_path,
        data.n,
        data.p,
        list(data.columns),
    )

    t0 = time.time()
    prs_df, prs_path = _load_or_build_prs_panel(cache, csv_path, person_ids, logger)
    data, kept_person_ids, prs_meta = _augment_with_prs_nodes(data, person_ids, prs_df)
    timings["prs"] = time.time() - t0
    logger.info(
        "[prs] selected=%d n=%d p=%d nodes=%s",
        prs_meta["prs_columns_selected"],
        data.n,
        data.p,
        prs_meta["prs_node_names"],
    )

    t0 = time.time()
    mrdag_pi, mrdag_diagnostics = _load_or_run_mrdag(cache, logger)
    mrdag_prior = _mrdag_prior_for_data(mrdag_pi, data.columns)
    timings["mrdag"] = time.time() - t0

    key = _run_key(data, mrdag_prior)
    dagslam = _load_or_run_dagslam(cache, key, data, logger)
    timings["dagslam"] = float(dagslam["runtime_s"])
    logger.info(
        "[dagslam] log_score=%.3f n_edges=%d",
        float(dagslam["log_score"]),
        int(dagslam["n_edges"]),
    )

    edge_probs, mcmc_diagnostics, mcmc_runtime = _load_or_run_mcmc(
        cache,
        key,
        data,
        np.asarray(dagslam["adjacency"], dtype=int),
        mrdag_prior,
        logger,
    )
    timings["mcmc"] = mcmc_runtime
    logger.info(
        "[mcmc] accept_overall=%.3f max_rhat_skel=%.3f min_ess=%.1f",
        float(mcmc_diagnostics["accept_rate"]["overall"]),
        float(mcmc_diagnostics.get("max_rhat_skeleton", float("nan"))),
        float(mcmc_diagnostics.get("min_ess", float("nan"))),
    )

    thresholded = (edge_probs >= THRESHOLD_DEFAULT).astype(int)
    np.fill_diagonal(thresholded, 0)

    validation = _validate_edges(
        edge_probs,
        data.columns,
        rng=np.random.default_rng(PIPELINE_SEED + 4),
    )

    return PipelineResult(
        columns=tuple(data.columns),
        node_types=tuple(data.node_types),
        data_summary={
            "n": int(data.n),
            "p": int(data.p),
            "columns": list(data.columns),
            "node_types": list(data.node_types),
            "csv_path": str(csv_path),
            "cohort_sha256": _file_sha256(Path(csv_path)),
            "prs_path": prs_path,
            "prs_sha256": _file_sha256(Path(prs_path)),
            "person_id_rows_after_prs": int(kept_person_ids.size),
        },
        mrdag_pi=mrdag_pi,
        mrdag_prior=mrdag_prior,
        mrdag_diagnostics=mrdag_diagnostics,
        dagslam_adjacency=np.asarray(dagslam["adjacency"], dtype=int),
        dagslam_log_score=float(dagslam["log_score"]),
        dagslam_n_edges=int(dagslam["n_edges"]),
        mcmc_edge_probs=edge_probs,
        mcmc_diagnostics=mcmc_diagnostics,
        thresholded_adjacency=thresholded,
        threshold=float(THRESHOLD_DEFAULT),
        validation=validation,
        timings=timings,
        genscore_features={"prs_path": prs_path, **prs_meta},
        cache_key=key,
    )


def _adj_to_edge_list(
    adj: np.ndarray, columns: Sequence[str]
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if int(adj[i, j]) == 1:
                rows.append((parent, child))
    return rows


def _edge_prob_long(
    edge_probs: np.ndarray, columns: Sequence[str]
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if i == j:
                continue
            rows.append((parent, child, float(edge_probs[i, j])))
    rows.sort(key=lambda r: -r[2])
    return rows


def save_result(
    result: PipelineResult,
    outdir: Optional[str] = None,
    run_config: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Serialise pipeline artefacts."""
    out = Path(outdir) if outdir is not None else Path(DEFAULT_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    columns = list(result.columns)

    paths["mrdag_pi"] = str(out / "mrdag_pi.npy")
    np.save(paths["mrdag_pi"], result.mrdag_pi)
    paths["mrdag_prior"] = str(out / "mrdag_prior.npy")
    np.save(paths["mrdag_prior"], result.mrdag_prior)
    paths["dagslam_adjacency"] = str(out / "dagslam_adjacency.npy")
    np.save(paths["dagslam_adjacency"], result.dagslam_adjacency)
    paths["mcmc_edge_probs"] = str(out / "mcmc_edge_probs.npy")
    np.save(paths["mcmc_edge_probs"], result.mcmc_edge_probs)
    paths["thresholded_adjacency"] = str(out / "thresholded_adjacency.npy")
    np.save(paths["thresholded_adjacency"], result.thresholded_adjacency)

    paths["greedy_edges_csv"] = str(out / "greedy_edges.csv")
    with open(paths["greedy_edges_csv"], "w") as fh:
        fh.write("parent,child\n")
        for parent, child in _adj_to_edge_list(result.dagslam_adjacency, columns):
            fh.write(f"{parent},{child}\n")

    paths["mcmc_thresholded_edges_csv"] = str(out / "mcmc_thresholded_edges.csv")
    with open(paths["mcmc_thresholded_edges_csv"], "w") as fh:
        fh.write("parent,child\n")
        for parent, child in _adj_to_edge_list(result.thresholded_adjacency, columns):
            fh.write(f"{parent},{child}\n")

    paths["mcmc_edge_probabilities_long_csv"] = str(
        out / "mcmc_edge_probabilities_long.csv"
    )
    with open(paths["mcmc_edge_probabilities_long_csv"], "w") as fh:
        fh.write("parent,child,posterior_edge_probability\n")
        for parent, child, prob in _edge_prob_long(result.mcmc_edge_probs, columns):
            fh.write(f"{parent},{child},{prob}\n")

    diagnostics = {
        k: v
        for k, v in result.mcmc_diagnostics.items()
        if k not in {"rhat_per_edge", "rhat_per_skeleton_edge", "ess_per_edge"}
    }
    summary = {
        "cache_key": result.cache_key,
        "columns": columns,
        "node_types": list(result.node_types),
        "data_summary": result.data_summary,
        "mrdag_diagnostics": result.mrdag_diagnostics,
        "dagslam_log_score": result.dagslam_log_score,
        "dagslam_n_edges": result.dagslam_n_edges,
        "mcmc_diagnostics": diagnostics,
        "threshold": result.threshold,
        "validation": result.validation,
        "timings": result.timings,
        "genscore_features": result.genscore_features,
    }
    paths["summary_json"] = str(out / "summary.json")
    with open(paths["summary_json"], "w") as fh:
        json.dump(_json_sanitise(summary), fh, indent=2, sort_keys=True)

    config = _pipeline_config() if run_config is None else run_config
    paths["run_config_json"] = str(out / "run_config.json")
    with open(paths["run_config_json"], "w") as fh:
        json.dump(_json_sanitise(config), fh, indent=2, sort_keys=True)

    return paths


def _sync_outputs_to_workspace(outdir: Path) -> None:
    bucket = _workspace_bucket()
    if bucket is not None:
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(outdir), f"{bucket}/{WORKSPACE_RESULTS_PREFIX}"],
            check=True,
        )


def main() -> int:
    result = run_pipeline()
    save_result(result)
    outdir = Path(DEFAULT_OUTPUT_DIR)
    _sync_outputs_to_workspace(outdir)
    print(f"\nArtefacts written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
