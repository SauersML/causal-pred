"""Single end-to-end causal-prediction pipeline.

Path:

    cohort CSV -> microarray PRS cache -> EHR panel -> TopK crosscoder features
      -> MrDAG priors -> DAGSLAM -> structure MCMC -> survival GAM
      -> per-person risk curves, causal pathways, validation artefacts

The public runner takes no configuration arguments. Operational settings live
as constants in this module so there is one production path and one place to
change it. Reusable intermediates are cached locally under ``data/`` and,
when ``WORKSPACE_BUCKET`` is present, mirrored under
``$WORKSPACE_BUCKET/intermediates/causal-pred``. This includes PRS score
outputs, OMOP parquets, EHR panels, feature matrices, and downstream model
artefacts, but never genotype files. Cached files are keyed by the actual
matrix/configuration that produced them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data.cohort import (
    CURATED_OMOP_CONDITION_CATALOG,
    CURATED_OMOP_CONDITION_IDS,
    EhrPanel,
    SurvivalOutcome,
    build_ehr_panel,
    build_survival_outcome,
    discover_genotype_dir,
    fetch_omop_long_frames,
    load_cohort_dataset_with_person_ids,
    resolve_aou_genotypes,
    resolve_cohort_csv,
)
from .data.nodes import CANONICAL_EDGES, NODE_INDEX, NODE_NAMES
from .data.polygenic import parse_sscore, score_panel
from .data.opengwas import load_live_gwas
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

PIPELINE_CONFIG_VERSION = "2026-05-08.single-path.3"
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
MCMC_BLOCK_RESAMPLE_PROB = 0.0
MCMC_EDGE_RESAMPLE_PROB = 0.7
MCMC_PARENT_RESAMPLE_PROB = 0.2
MCMC_BLOCK_SIZE = 3
MCMC_EXACT_PARENT_RESAMPLE = True
THRESHOLD_DEFAULT = 0.5

VALIDATION_N_PERMUTE = 200

EHR_FETCH_DRUGS = True
EHR_FETCH_MEASUREMENTS = True
# The curated condition catalog lives in `causal_pred.data.cohort` so that
# both the production pipeline and direct callers of `fetch_omop_long_frames`
# share one source of truth. Extend the catalog there.
EHR_CONDITION_CATALOG = CURATED_OMOP_CONDITION_CATALOG
EHR_CONDITION_CONCEPT_IDS: Optional[Tuple[int, ...]] = CURATED_OMOP_CONDITION_IDS
EHR_MIN_PREVALENCE = 50
EHR_MIN_LAB_OBSERVATIONS = 50
EHR_LOOKBACK_DAYS: Optional[int] = 365 * 5

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
    "age": "age",
    "sex": "sex",
    "ancestry_pc1": "ancestry_PC1",
    "family_history_t2d": "family_history_T2D",
    "years_smoking": "years_smoking",
    "physical_activity": "physical_activity",
    "diet_quality": "diet_quality",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "ldl_cholesterol": "LDL",
    "systolic_bp": "systolic_BP",
    "hypertension": "hypertension",
    "cardiovascular_disease": "cardiovascular_disease",
    "pgs_t2d": "PGS_T2D",
    "pgs_bmi": "PGS_BMI",
    "pgs_ldl": "PGS_LDL",
    "pgs_hba1c": "PGS_HbA1c",
}

MR_TO_COHORT_NODE = {v: k for k, v in COHORT_TO_MR_NODE.items()}

PGS_ANCHOR_PRIORS = {
    ("PGS_T2D", "T2D"): 0.95,
    ("PGS_BMI", "BMI"): 0.95,
    ("PGS_LDL", "LDL"): 0.95,
    ("PGS_HbA1c", "HbA1c"): 0.95,
}


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
    survival_lower: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    survival_upper: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
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
        logger = logging.getLogger("causal_pred.pipeline")
        dst = local_path if local_path is not None else self.path(filename)
        if dst.is_file() and dst.stat().st_size > 0 and not overwrite:
            logger.info(
                "[cache] hit %s size=%.1fMiB",
                dst,
                _path_size_mib(dst),
            )
            return dst
        uri = self.uri(filename)
        if uri is not None and _gsutil_exists(uri):
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_name(dst.name + ".part")
            if part.exists():
                part.unlink()
            t0 = time.time()
            logger.info("[cache] downloading %s -> %s", uri, dst)
            subprocess.run(["gsutil", "cp", uri, str(part)], check=True)
            os.replace(part, dst)
            logger.info(
                "[cache] downloaded %s size=%.1fMiB elapsed=%s",
                dst,
                _path_size_mib(dst),
                _format_seconds(time.time() - t0),
            )
        return dst

    def store(self, src: Path, filename: Optional[str] = None) -> None:
        logger = logging.getLogger("causal_pred.pipeline")
        uri = self.uri(filename or src.name)
        if uri is not None:
            t0 = time.time()
            logger.info(
                "[cache] uploading %s size=%.1fMiB -> %s",
                src,
                _path_size_mib(src),
                uri,
            )
            subprocess.run(["gsutil", "cp", str(src), uri], check=True)
            logger.info(
                "[cache] uploaded %s elapsed=%s",
                uri,
                _format_seconds(time.time() - t0),
            )


PIPELINE_LOG_FILENAME = "pipeline.log"


def _setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("causal_pred.pipeline")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    log_dir = Path(DEFAULT_CACHE_DIR)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / PIPELINE_LOG_FILENAME
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.info(
            "[pipeline] log file %s pid=%d python=%s",
            log_path,
            os.getpid(),
            sys.version.split()[0],
        )
    except OSError as exc:
        sys.stderr.write(
            f"[pipeline] WARNING failed to attach file log under {log_dir}: {exc}\n"
        )
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


def _flush_log_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


_SIGNAL_HANDLERS_INSTALLED = False


def _install_signal_handlers(logger: logging.Logger) -> None:
    """Log a banner if the process receives a termination signal.

    SIGKILL cannot be caught from Python; everything catchable
    (SIGTERM/SIGHUP/SIGUSR1/SIGUSR2/SIGXCPU/SIGPIPE) gets logged before the
    default handler runs so silent terminations are easier to diagnose. The
    OS-level OOM killer typically uses SIGKILL — that one stays invisible to
    user code, so the persistent log file is the only forensic trace.
    """
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    _SIGNAL_HANDLERS_INSTALLED = True

    def _make_handler(signum: int):
        def _handler(_signum, _frame):
            try:
                name = signal.Signals(_signum).name
            except ValueError:
                name = f"signal_{_signum}"
            logger.error(
                "[pipeline] received %s — re-raising for default handling",
                name,
            )
            _flush_log_handlers(logger)
            signal.signal(_signum, signal.SIG_DFL)
            os.kill(os.getpid(), _signum)

        return _handler

    for attr in ("SIGTERM", "SIGHUP", "SIGUSR1", "SIGUSR2", "SIGXCPU", "SIGPIPE"):
        sig = getattr(signal, attr, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _make_handler(int(sig)))
        except (ValueError, OSError):
            pass


@contextlib.contextmanager
def _phase(logger: logging.Logger, name: str) -> Iterator[None]:
    """Wrap a pipeline phase so failures log the phase name + elapsed time.

    The exception is re-raised unchanged; this only adds context to the log.
    """
    started_at = time.time()
    try:
        yield
    except BaseException as exc:
        logger.error(
            "[%s] FAILED after elapsed=%s with %s: %s",
            name,
            _format_seconds(time.time() - started_at),
            type(exc).__name__,
            exc,
        )
        _flush_log_handlers(logger)
        raise


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(rem_minutes):02d}m{rem:04.1f}s"


def _path_size_mib(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


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


def _file_stat_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"name": path.name, "size": int(st.st_size)}


def _plink_stat_fingerprint(bed: Path) -> dict[str, Any]:
    prefix = bed.with_suffix("")
    files = []
    for suffix in (".bed", ".bim", ".fam"):
        p = prefix.with_suffix(suffix)
        files.append(_file_stat_fingerprint(p))
    return {"prefix": prefix.name, "files": files}


def _gnomon_score_cache_filename(bed: Path, score_files: Sequence[Path]) -> str:
    return "gnomon-scores-" + _short_hash(
        {
            "version": PIPELINE_CONFIG_VERSION,
            "genotype": _plink_stat_fingerprint(bed),
            "score_files": [
                {
                    "name": p.name,
                    "size": int(p.stat().st_size),
                    "sha256": _file_sha256(p),
                }
                for p in sorted(score_files, key=lambda x: x.name)
            ],
        }
    ) + ".sscore"


def _pipeline_config() -> dict[str, Any]:
    return {
        "version": PIPELINE_CONFIG_VERSION,
        "seed": PIPELINE_SEED,
        "cohort_name": COHORT_NAME,
        "prs_nodes": PRS_NODES,
        "prs_max_missing": PRS_MAX_MISSING,
        "prs_min_complete_rows": PRS_MIN_COMPLETE_ROWS,
        "ehr": {
            "fetch_drugs": EHR_FETCH_DRUGS,
            "fetch_measurements": EHR_FETCH_MEASUREMENTS,
            "condition_concept_ids": (
                None
                if EHR_CONDITION_CONCEPT_IDS is None
                else list(EHR_CONDITION_CONCEPT_IDS)
            ),
            "min_prevalence": EHR_MIN_PREVALENCE,
            "min_lab_observations": EHR_MIN_LAB_OBSERVATIONS,
            "lookback_days": EHR_LOOKBACK_DAYS,
        },
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
            "block_resample_prob": MCMC_BLOCK_RESAMPLE_PROB,
            "block_size": MCMC_BLOCK_SIZE,
            "edge_resample_prob": MCMC_EDGE_RESAMPLE_PROB,
            "parent_resample_prob": MCMC_PARENT_RESAMPLE_PROB,
            "exact_parent_resample": MCMC_EXACT_PARENT_RESAMPLE,
            "max_parents": DAGSLAM_MAX_PARENTS,
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


def _cohort_scores_from_gnomon_scores(
    scores: pd.DataFrame,
    person_ids: pd.Series,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, int]:
    scores.index = scores.index.astype(str)
    overlap = person_ids.astype(str).isin(scores.index)
    min_required = min(PRS_MIN_COMPLETE_ROWS, int(person_ids.size))
    n_overlap = int(overlap.sum())
    if n_overlap < min_required:
        raise RuntimeError(
            f"only {n_overlap} cohort participants overlap the gnomon score output; "
            f"required at least {min_required}"
        )

    reindexed = scores.reindex(person_ids.astype(str))
    all_nan_cols = [c for c in reindexed.columns if reindexed[c].isna().all()]
    cohort_scores = reindexed.dropna(axis=1, how="all")
    if logger is not None:
        logger.info(
            "[prs] gnomon scored cols=%d kept=%d dropped_all_nan=%d",
            int(scores.shape[1]),
            int(cohort_scores.shape[1]),
            len(all_nan_cols),
        )
        if all_nan_cols:
            logger.info(
                "[prs] dropped (all NaN over cohort): %s",
                ", ".join(sorted(all_nan_cols)),
            )
    if cohort_scores.shape[1] < PRS_NODES:
        raise RuntimeError(
            f"gnomon produced {cohort_scores.shape[1]} usable PRS columns; "
            f"required at least {PRS_NODES}"
        )
    return cohort_scores, n_overlap


def _restore_gnomon_scores(
    cache: Optional[WorkspaceCache],
    local_sscore: Path,
    remote_name: str,
    person_ids: pd.Series,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    if cache is None:
        return None
    restored = cache.fetch(remote_name, local_sscore)
    if not restored.is_file() or restored.stat().st_size == 0:
        return None
    logger.info(
        "[prs] parsing cached gnomon scores %s size=%.1fMiB",
        restored,
        _path_size_mib(restored),
    )
    t_parse = time.time()
    scores = parse_sscore(restored, keep_iids=person_ids.astype(str).tolist())
    logger.info(
        "[prs] parsed cached gnomon scores rows=%d cols=%d elapsed=%s",
        scores.shape[0],
        scores.shape[1],
        _format_seconds(time.time() - t_parse),
    )
    return scores


def _latest_sscore(out_dir: Path, started_at: float) -> Optional[Path]:
    candidates = [
        p
        for p in out_dir.glob("*.sscore")
        if p.stat().st_size > 0 and p.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_prs_panel(
    cohort_csv: Path,
    path: Path,
    logger: logging.Logger,
    cache: Optional[WorkspaceCache] = None,
) -> pd.DataFrame:
    person_ids = _cohort_person_ids(cohort_csv)
    bed = _resolve_microarray_bed()
    panel_dir = Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME
    out_dir = Path(DEFAULT_CACHE_DIR) / GNOMON_OUT_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[prs] downloading PGS scoring panel into %s", panel_dir)
    score_files = [Path(p) for p in download_panel(panel_dir)]
    sscore_name = _gnomon_score_cache_filename(bed, score_files)
    local_sscore = out_dir / sscore_name
    remote_sscore = f"{GNOMON_OUT_DIRNAME}/{sscore_name}"
    scores = _restore_gnomon_scores(
        cache,
        local_sscore,
        remote_sscore,
        person_ids,
        logger,
    )
    if scores is None:
        logger.info(
            "[prs] scoring AoU microarray genotypes bed=%s cohort_ids=%d score_files=%d",
            bed,
            int(person_ids.size),
            len(score_files),
        )
        t_score = time.time()
        started_at = time.time()
        scores = score_panel(
            genotype_path=str(bed),
            score_path=[str(p) for p in score_files],
            out_dir=str(out_dir),
            n_threads=os.cpu_count(),
            timeout=GNOMON_TIMEOUT_SECONDS,
            keep_iids=person_ids.astype(str).tolist(),
        )
        logger.info(
            "[prs] parsed gnomon scores rows=%d cols=%d elapsed=%s",
            scores.shape[0],
            scores.shape[1],
            _format_seconds(time.time() - t_score),
        )
        raw_sscore = _latest_sscore(out_dir, started_at)
        if raw_sscore is not None:
            if raw_sscore != local_sscore:
                shutil.copy2(raw_sscore, local_sscore)
            if cache is not None:
                cache.store(local_sscore, remote_sscore)

    cohort_scores, n_overlap = _cohort_scores_from_gnomon_scores(
        scores, person_ids, logger
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    t_write = time.time()
    logger.info(
        "[prs] writing cohort score cache %s rows=%d cols=%d",
        path,
        cohort_scores.shape[0],
        cohort_scores.shape[1],
    )
    cohort_scores.to_csv(
        part,
        index_label="person_id",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    os.replace(part, path)
    logger.info(
        "[prs] built %s rows=%d overlap=%d cols=%d size=%.1fMiB write_elapsed=%s",
        path,
        cohort_scores.shape[0],
        n_overlap,
        cohort_scores.shape[1],
        _path_size_mib(path),
        _format_seconds(time.time() - t_write),
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

    panel = _build_prs_panel(cohort_csv, path, logger, cache=cache)
    cache.store(path, PRS_PANEL_FILENAME)
    return panel, str(path)


def _has_survival_outcome(data: SyntheticDataset) -> bool:
    return bool(np.any(data.time > 0.0) and np.any(data.event == 1))


def _survival_outcome_key(person_ids: Sequence[str]) -> str:
    return _short_hash(
        {
            "version": PIPELINE_CONFIG_VERSION,
            "person_ids": [str(p) for p in person_ids],
        }
    )


def _load_or_build_survival_outcome(
    cache: WorkspaceCache,
    person_ids: Sequence[str],
    logger: logging.Logger,
) -> SurvivalOutcome:
    filename = f"survival-outcome-{_survival_outcome_key(person_ids)}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[survival] using cache %s", path)
            return SurvivalOutcome(
                person_id=z["person_id"].astype(str),
                time=z["time"].astype(float),
                event=z["event"].astype(int),
                keep=z["keep"].astype(bool),
                baseline_dt=z["baseline_dt"],
                end_dt=z["end_dt"],
                t2d_dt=z["t2d_dt"],
                meta=json.loads(str(z["meta_json"].item())),
            )

    logger.info("[survival] fetching OMOP follow-up frames for n=%d", len(person_ids))
    frames = fetch_omop_long_frames(
        person_ids=person_ids,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "omop",
        workspace_bucket=cache.bucket,
        workspace_prefix=f"{WORKSPACE_CACHE_PREFIX}/omop",
        progress=lambda message: logger.info("[survival] %s", message),
    )
    required = ("visit_baseline", "observation_period", "t2d_event")
    missing = [name for name in required if name not in frames]
    if missing:
        raise RuntimeError(f"OMOP survival frames missing required tables: {missing}")
    outcome = build_survival_outcome(
        person_ids,
        frames["visit_baseline"],
        frames["observation_period"],
        frames["t2d_event"],
    )
    _atomic_npz(
        path,
        person_id=outcome.person_id.astype(str),
        time=outcome.time,
        event=outcome.event.astype(int),
        keep=outcome.keep.astype(bool),
        baseline_dt=outcome.baseline_dt,
        end_dt=outcome.end_dt,
        t2d_dt=outcome.t2d_dt,
        meta_json=np.array(json.dumps(_json_sanitise(outcome.meta))),
    )
    cache.store(path)
    logger.info(
        "[survival] built outcome n=%d events=%d dropped=%d",
        int(outcome.keep.sum()),
        int(outcome.event[outcome.keep].sum()),
        int(outcome.keep.size - outcome.keep.sum()),
    )
    return outcome


def _apply_survival_outcome(
    data: SyntheticDataset,
    person_ids: Sequence[str],
    outcome: SurvivalOutcome,
) -> tuple[SyntheticDataset, np.ndarray]:
    pid = np.asarray([str(p) for p in person_ids])
    if pid.shape[0] != data.n:
        raise ValueError(
            f"person_ids has length {pid.shape[0]} but dataset has {data.n} rows"
        )
    if outcome.person_id.shape[0] != data.n:
        raise ValueError(
            f"survival outcome has {outcome.person_id.shape[0]} rows but dataset has {data.n}"
        )
    if not np.array_equal(outcome.person_id.astype(str), pid.astype(str)):
        raise ValueError("survival outcome person_id order does not match the cohort")
    keep = outcome.keep.astype(bool)
    if int(keep.sum()) == 0:
        raise ValueError("survival outcome removed every cohort row")
    event = outcome.event.astype(int)
    if int(event[keep].sum()) == 0:
        raise ValueError("survival outcome has no incident T2D events after filtering")

    X = data.X[keep].copy()
    target_idx = _target_index(data.columns, SURVIVAL_TARGET_COLUMN)
    X[:, target_idx] = event[keep].astype(float)
    p = X.shape[1]
    ground_truth_adj = np.zeros((p, p), dtype=int)
    if data.ground_truth_adj.size:
        ground_truth_adj = data.ground_truth_adj.copy()

    return (
        SyntheticDataset(
            X=X,
            time=outcome.time[keep].astype(float, copy=False),
            event=event[keep],
            columns=data.columns,
            node_types=data.node_types,
            ground_truth_adj=ground_truth_adj,
        ),
        pid[keep],
    )


def _ehr_panel_key(person_ids: Sequence[str]) -> str:
    return _short_hash(
        {
            "version": PIPELINE_CONFIG_VERSION,
            "person_ids": [str(p) for p in person_ids],
            "fetch_drugs": EHR_FETCH_DRUGS,
            "fetch_measurements": EHR_FETCH_MEASUREMENTS,
            "condition_concept_ids": (
                None
                if EHR_CONDITION_CONCEPT_IDS is None
                else list(EHR_CONDITION_CONCEPT_IDS)
            ),
            "min_prevalence": EHR_MIN_PREVALENCE,
            "min_lab_observations": EHR_MIN_LAB_OBSERVATIONS,
            "lookback_days": EHR_LOOKBACK_DAYS,
        }
    )


def _load_or_build_ehr_panel(
    cache: WorkspaceCache,
    person_ids: Sequence[str],
    logger: logging.Logger,
) -> EhrPanel:
    filename = f"ehr-panel-{_ehr_panel_key(person_ids)}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[ehr] using cache %s", path)
            return EhrPanel(
                matrix=z["matrix"],
                person_id=z["person_id"].astype(str),
                feature_names=tuple(json.loads(str(z["feature_names_json"].item()))),
                feature_kinds=tuple(json.loads(str(z["feature_kinds_json"].item()))),
            )

    logger.info(
        "[ehr] fetching OMOP frames for n=%d and building baseline panel",
        len(person_ids),
    )
    frames = fetch_omop_long_frames(
        person_ids=person_ids,
        cache_dir=Path(DEFAULT_CACHE_DIR) / "omop",
        workspace_bucket=cache.bucket,
        workspace_prefix=f"{WORKSPACE_CACHE_PREFIX}/omop",
        condition_concept_ids=EHR_CONDITION_CONCEPT_IDS,
        fetch_drugs=EHR_FETCH_DRUGS,
        fetch_measurements=EHR_FETCH_MEASUREMENTS,
        progress=lambda message: logger.info("[ehr] %s", message),
    )
    visit_baseline = frames.get("visit_baseline")
    if visit_baseline is None or len(visit_baseline) == 0:
        raise RuntimeError(
            "OMOP visit_baseline returned no rows; cannot define EHR baseline"
        )
    baseline_frame = visit_baseline.copy()
    baseline_frame["person_id"] = baseline_frame["person_id"].astype(str)
    baseline_frame["baseline_dt"] = pd.to_datetime(
        baseline_frame["baseline_dt"], errors="coerce"
    )
    baseline_dt = baseline_frame.set_index("person_id")["baseline_dt"]
    baseline_dt = baseline_dt.reindex([str(p) for p in person_ids]).rename(
        "baseline_dt"
    )
    logger.info(
        "[ehr] baseline dates resolved observed=%d missing=%d",
        int(baseline_dt.notna().sum()),
        int(baseline_dt.isna().sum()),
    )
    for name, frame in frames.items():
        logger.info("[ehr] frame %s rows=%d cols=%d", name, len(frame), frame.shape[1])
    t_build = time.time()
    panel = build_ehr_panel(
        person_ids=person_ids,
        baseline_dt=baseline_dt,
        condition_long=frames.get("condition_long"),
        drug_long=frames.get("drug_long"),
        measurement_long=frames.get("measurement_long"),
        min_prevalence=EHR_MIN_PREVALENCE,
        min_lab_observations=EHR_MIN_LAB_OBSERVATIONS,
        lookback_days=EHR_LOOKBACK_DAYS,
    )
    if panel.m == 0:
        raise RuntimeError("EHR panel has zero features; crosscoder cannot run")
    kind_counts: dict[str, int] = {}
    for k in panel.feature_kinds:
        kind_counts[k] = kind_counts.get(k, 0) + 1
    logger.info(
        "[ehr] built matrix n=%d m=%d elapsed=%s kinds=%s",
        panel.n,
        panel.m,
        _format_seconds(time.time() - t_build),
        ",".join(f"{k}={v}" for k, v in sorted(kind_counts.items())),
    )
    _atomic_npz(
        path,
        matrix=panel.matrix,
        person_id=panel.person_id.astype(str),
        feature_names_json=np.array(json.dumps(list(panel.feature_names))),
        feature_kinds_json=np.array(json.dumps(list(panel.feature_kinds))),
    )
    cache.store(path)
    logger.info("[ehr] cached panel %s n=%d m=%d", path, panel.n, panel.m)
    return panel


def _prs_node_name(original: str, used: set[str]) -> str:
    stem = re.sub(r"[^0-9A-Za-z]+", "_", str(original)).strip("_").lower()
    if not stem:
        stem = "score"
    canonical = None
    if "hba1c" in stem or "hemoglobin_a1c" in stem or "glycated" in stem:
        canonical = "pgs_hba1c"
    elif "t2d" in stem or "diabetes" in stem:
        canonical = "pgs_t2d"
    elif "bmi" in stem or "body_mass" in stem:
        canonical = "pgs_bmi"
    elif "ldl" in stem:
        canonical = "pgs_ldl"
    if canonical is not None and canonical not in used:
        used.add(canonical)
        return canonical
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
    logger: Optional[logging.Logger] = None,
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

    # Deterministic, outcome-independent selection. Filter PRS columns by
    # completeness and non-zero variance, then take the first PRS_NODES in
    # alphabetical column order. Ranking by correlation with type2_diabetes
    # would cherry-pick the columns most predictive of the outcome and bias
    # the DAG search toward exactly the PRS->T2D edges we want to discover;
    # that is target leakage in a causal-inference setup.
    min_required = min(PRS_MIN_COMPLETE_ROWS, data.n)
    candidates: list[tuple[str, float, np.ndarray]] = []
    drop_reasons: list[tuple[str, str]] = []
    for col in aligned.columns:
        vals = pd.to_numeric(aligned[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        present_rate = float(finite.mean())
        n_finite = int(finite.sum())
        if present_rate < 1.0 - PRS_MAX_MISSING:
            drop_reasons.append((
                str(col),
                f"missing={1.0 - present_rate:.3f} > max={PRS_MAX_MISSING:.3f}",
            ))
            continue
        if n_finite < min_required:
            drop_reasons.append((
                str(col),
                f"finite_rows={n_finite} < min_required={min_required}",
            ))
            continue
        sd = float(vals[finite].std(ddof=0))
        if sd == 0.0:
            drop_reasons.append((str(col), "constant (sd=0)"))
            continue
        candidates.append((str(col), present_rate, vals))

    if logger is not None:
        logger.info(
            "[prs] panel cols=%d eligible=%d dropped=%d (max_missing=%.3f min_rows=%d)",
            int(aligned.shape[1]),
            len(candidates),
            len(drop_reasons),
            float(PRS_MAX_MISSING),
            int(min_required),
        )
        for col, reason in drop_reasons:
            logger.info("[prs] drop %s: %s", col, reason)

    if len(candidates) < PRS_NODES:
        raise ValueError(
            f"only {len(candidates)} PRS columns pass completeness/variance checks; "
            f"required {PRS_NODES}"
        )

    candidates.sort(key=lambda row: row[0])
    chosen = candidates[:PRS_NODES]

    keep = np.ones(data.n, dtype=bool)
    for _col, _present, vals in chosen:
        keep &= np.isfinite(vals)
    if int(keep.sum()) < min_required:
        raise ValueError(
            f"only {int(keep.sum())} rows have complete selected PRS values; "
            f"required {min_required}"
        )

    prs_cols = []
    original_names = []
    present_rates = []
    used_names = set(data.columns)
    for col, present, vals in chosen:
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
        "prs_max_missing": float(PRS_MAX_MISSING),
        "prs_min_complete_rows": int(min_required),
    }
    return augmented, pid[keep], meta


def _load_or_run_genscore_features(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
    logger: logging.Logger,
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any]]:
    key = _genscore_key(data, person_ids, prs_df, ehr_panel)
    filename = f"genscore-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[genscore] using cache %s", path)
            columns = tuple(json.loads(str(z["columns_json"].item())))
            node_types = tuple(json.loads(str(z["node_types_json"].item())))
            meta = json.loads(str(z["meta_json"].item()))
            dataset = SyntheticDataset(
                X=z["X"],
                time=z["time"],
                event=z["event"].astype(int),
                columns=columns,
                node_types=node_types,
                ground_truth_adj=z["ground_truth_adj"].astype(int),
            )
            return dataset, z["person_id"].astype(str), meta

    logger.info("[genscore] training TopK crosscoder and promoting shared features")
    logger.info(
        "[genscore] config n_promote=%d share_band=[%.2f,%.2f] "
        "min_activation_rate=%.3f crosscoder=%s",
        GENSCORE_N_PROMOTE,
        GENSCORE_GENOME_SHARE_MIN,
        GENSCORE_GENOME_SHARE_MAX,
        GENSCORE_MIN_ACTIVATION_RATE,
        GENSCORE_CROSSCODER_KWARGS,
    )
    t0 = time.time()
    aug_result, model = run_genscore(
        base_dataset=data,
        base_person_ids=person_ids,
        prs_df=prs_df,
        ehr_panel=ehr_panel,
        n_promote=GENSCORE_N_PROMOTE,
        genome_share_min=GENSCORE_GENOME_SHARE_MIN,
        genome_share_max=GENSCORE_GENOME_SHARE_MAX,
        min_activation_rate=GENSCORE_MIN_ACTIVATION_RATE,
        crosscoder_kwargs=GENSCORE_CROSSCODER_KWARGS,
        rng=np.random.default_rng(PIPELINE_SEED + 10),
        progress=lambda message: logger.info("%s", message),
    )
    sel = aug_result.feature_selection
    top_genome_loadings: dict[str, list[dict[str, float]]] = {}
    top_ehr_loadings: dict[str, list[dict[str, float]]] = {}
    for feature_idx, feature_name in zip(sel.indices.tolist(), sel.names):
        g_weights = model.W_d_G[int(feature_idx)]
        e_weights = model.W_d_E[int(feature_idx)]
        g_order = np.argsort(-np.abs(g_weights))[:10]
        e_order = np.argsort(-np.abs(e_weights))[:10]
        top_genome_loadings[str(feature_name)] = [
            {"feature": str(prs_df.columns[j]), "weight": float(g_weights[j])}
            for j in g_order
        ]
        top_ehr_loadings[str(feature_name)] = [
            {"feature": str(ehr_panel.feature_names[j]), "weight": float(e_weights[j])}
            for j in e_order
        ]
    meta = {
        "crosscoder_d": int(model.d),
        "crosscoder_k": int(model.k),
        "promoted_indices": sel.indices.tolist(),
        "promoted_names": list(sel.names),
        "promoted_genome_share": sel.genome_share.tolist(),
        "promoted_activation_rate": sel.activation_rate.tolist(),
        "base_n": int(aug_result.base_n),
        "augmented_n": int(aug_result.augmented_n),
        "ehr_feature_count": int(ehr_panel.m),
        "promoted_feature_top_genome_loadings": top_genome_loadings,
        "promoted_feature_top_ehr_loadings": top_ehr_loadings,
        "loss_history_step": list(model.history["step"]),
        "loss_history_main": list(model.history["loss_main"]),
        "loss_history_aux": list(model.history["loss_aux"]),
        "frac_dead_history": list(model.history["frac_dead"]),
        "runtime_s": time.time() - t0,
    }
    dataset = aug_result.dataset
    _atomic_npz(
        path,
        X=dataset.X,
        time=dataset.time,
        event=dataset.event,
        ground_truth_adj=dataset.ground_truth_adj,
        person_id=aug_result.kept_person_id.astype(str),
        columns_json=np.array(json.dumps(list(dataset.columns))),
        node_types_json=np.array(json.dumps(list(dataset.node_types))),
        meta_json=np.array(json.dumps(_json_sanitise(meta))),
    )
    cache.store(path)

    # ---- post-training summary --------------------------------------------
    from .genscore.crosscoder import encode, feature_stream_share
    from .genscore.integrate import align_panels_by_iid

    r_G = feature_stream_share(model)
    panels_aligned = align_panels_by_iid(person_ids, prs_df, ehr_panel)
    z_full = encode(model, panels_aligned.A, panels_aligned.B)
    activation_rate_all = (z_full > 0).mean(axis=0)
    dead_mask_all = activation_rate_all == 0.0
    genome_only = (~dead_mask_all) & (r_G > GENSCORE_GENOME_SHARE_MAX)
    ehr_only = (~dead_mask_all) & (r_G < GENSCORE_GENOME_SHARE_MIN)
    cross_modal = (
        (~dead_mask_all)
        & (r_G >= GENSCORE_GENOME_SHARE_MIN)
        & (r_G <= GENSCORE_GENOME_SHARE_MAX)
    )

    a_hat = z_full @ model.W_d_G
    b_hat = z_full @ model.W_d_E
    a_z = (panels_aligned.A - model.mean_G) / model.std_G
    b_z = (panels_aligned.B - model.mean_E) / model.std_E
    ss_res_a = float(np.sum((a_z - a_hat) ** 2))
    ss_tot_a = float(np.sum(a_z ** 2))
    ss_res_b = float(np.sum((b_z - b_hat) ** 2))
    ss_tot_b = float(np.sum(b_z ** 2))
    r2_g = 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 0 else float("nan")
    r2_e = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 0 else float("nan")

    rg_q = np.quantile(r_G, [0.0, 0.25, 0.5, 0.75, 1.0])
    rate_alive = activation_rate_all[~dead_mask_all]
    if rate_alive.size > 0:
        rate_q = np.quantile(rate_alive, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        rate_q = np.zeros(5)

    logger.info(
        "[genscore] reconstruction R^2 genome=%.4f ehr=%.4f",
        r2_g,
        r2_e,
    )
    logger.info(
        "[genscore] feature counts d=%d alive=%d dead=%d "
        "genome_only(rG>%.2f)=%d ehr_only(rG<%.2f)=%d cross_modal=%d",
        int(model.d),
        int((~dead_mask_all).sum()),
        int(dead_mask_all.sum()),
        GENSCORE_GENOME_SHARE_MAX,
        int(genome_only.sum()),
        GENSCORE_GENOME_SHARE_MIN,
        int(ehr_only.sum()),
        int(cross_modal.sum()),
    )
    logger.info(
        "[genscore] genome_share quantiles min=%.3f q25=%.3f med=%.3f q75=%.3f max=%.3f",
        *(float(x) for x in rg_q),
    )
    logger.info(
        "[genscore] activation_rate quantiles (alive only) "
        "min=%.4f q25=%.4f med=%.4f q75=%.4f max=%.4f",
        *(float(x) for x in rate_q),
    )
    eligible_pool = int(
        (cross_modal & (activation_rate_all >= GENSCORE_MIN_ACTIVATION_RATE)).sum()
    )
    logger.info(
        "[genscore] selected (n_promote=%d) eligible pool=%d",
        GENSCORE_N_PROMOTE,
        eligible_pool,
    )
    for name, idx, share, rate in zip(
        sel.names,
        sel.indices.tolist(),
        sel.genome_share.tolist(),
        sel.activation_rate.tolist(),
    ):
        top_g = top_genome_loadings.get(str(name), [])[:3]
        top_e = top_ehr_loadings.get(str(name), [])[:3]
        g_summary = ",".join(
            f"{entry['feature']}={entry['weight']:+.2f}" for entry in top_g
        )
        e_summary = ",".join(
            f"{entry['feature']}={entry['weight']:+.2f}" for entry in top_e
        )
        logger.info(
            "[genscore]   %s idx=%d r_G=%.3f rate=%.4f top_genome=[%s] top_ehr=[%s]",
            name,
            int(idx),
            float(share),
            float(rate),
            g_summary,
            e_summary,
        )

    logger.info(
        "[genscore] promoted=%d n=%d p=%d",
        len(sel.names),
        dataset.n,
        dataset.p,
    )
    return dataset, aug_result.kept_person_id.astype(str), meta


def _mr_gwas_source() -> str:
    """Pick the MR-prior source.

    ``MR_GWAS_SOURCE=live`` (default when ``OPENGWAS_JWT`` is set) runs
    live two-sample IVW via ``load_live_gwas``.  ``MR_GWAS_SOURCE=literature``
    forces the hand-coded ``load_real_gwas`` fallback.
    """
    explicit = os.environ.get("MR_GWAS_SOURCE", "").strip().lower()
    if explicit in {"live", "literature"}:
        return explicit
    return "live" if os.environ.get("OPENGWAS_JWT", "").strip() else "literature"


def _mrdag_cache_key() -> str:
    return _short_hash({
        "version": PIPELINE_CONFIG_VERSION,
        "mrdag": _pipeline_config()["mrdag"],
        "mr_gwas_source": _mr_gwas_source(),
    })


def _log_gwas_per_pair(
    logger: logging.Logger, summary: Any, source: str
) -> None:
    rows = getattr(summary, "per_pair", None)
    if not rows:
        finite = int(np.sum(np.isfinite(summary.betas)))
        total = int(summary.betas.size - len(summary.exposures))  # off-diagonal
        logger.info(
            "[mrdag] gwas_source=%s usable_cells=%d/%d",
            source,
            finite,
            total,
        )
        return
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    logger.info(
        "[mrdag] gwas_source=%s pair_status=%s",
        source,
        ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())),
    )
    fetched_or_cached = [
        r for r in rows if r.source in {"fetched", "cache"}
    ]
    fetched_or_cached.sort(key=lambda r: -abs(r.beta) if np.isfinite(r.beta) else 0.0)
    for r in fetched_or_cached[:20]:
        logger.info(
            "[mrdag] %s -> %s : beta=%.4f se=%.4f n_snps=%d (%s)",
            r.exposure,
            r.outcome,
            r.beta,
            r.se,
            r.n_snps,
            r.source,
        )


def _load_gwas_for_mrdag(logger: logging.Logger) -> tuple[Any, str]:
    """Load the MR summary statistics for the MrDAG prior.

    Tries ``live`` (OpenGWAS two-sample IVW) when configured; falls
    back to the curated literature table when no JWT is set or when
    the fetch yields zero usable cells.  Logs which source was used
    and a per-pair (beta, SE, n_snps) breakdown.
    """
    source = _mr_gwas_source()
    if source == "live":
        try:
            cache_dir = Path(DEFAULT_CACHE_DIR) / "mr_cache"
            summary = load_live_gwas(cache_dir=cache_dir)
            usable = int(np.sum(np.isfinite(summary.betas) & np.isfinite(summary.ses)))
            if usable > 0:
                _log_gwas_per_pair(logger, summary, "live")
                return summary, "live"
            logger.warning(
                "[mrdag] live OpenGWAS fetch produced 0 usable cells; "
                "falling back to literature table"
            )
        except Exception as exc:
            logger.warning(
                "[mrdag] live OpenGWAS fetch failed (%s); falling back to literature",
                exc,
            )
    summary = load_real_gwas()
    _log_gwas_per_pair(logger, summary, "literature")
    return summary, "literature"


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

    gwas, source_used = _load_gwas_for_mrdag(logger)
    logger.info("[mrdag] running MR prior sampler (source=%s)", source_used)
    t0 = time.time()
    result = run_mrdag(
        gwas,
        rng=np.random.default_rng(PIPELINE_SEED + 1),
        n_iter=MRDAG_N_ITER,
        n_chains=MRDAG_N_CHAINS,
        n_burn=MRDAG_N_BURN,
        thin=MRDAG_THIN,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["runtime_s"] = time.time() - t0
    diagnostics["gwas_source"] = source_used
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
            anchor = PGS_ANCHOR_PRIORS.get((mr_parent, mr_child))
            if anchor is not None:
                prior[i, j] = float(anchor)
            reverse_anchor = PGS_ANCHOR_PRIORS.get((mr_child, mr_parent))
            if reverse_anchor is not None:
                prior[i, j] = 1.0 - float(reverse_anchor)
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    filename = f"mcmc-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[mcmc] using cache %s", path)
            return (
                z["edge_probs"],
                z["samples"].astype(int),
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
        block_resample_prob=MCMC_BLOCK_RESAMPLE_PROB,
        block_size=MCMC_BLOCK_SIZE,
        edge_resample_prob=MCMC_EDGE_RESAMPLE_PROB,
        hybrid_prob=MCMC_PARENT_RESAMPLE_PROB,
        exact_parent_resample=MCMC_EXACT_PARENT_RESAMPLE,
        max_parents=DAGSLAM_MAX_PARENTS,
    )
    runtime_s = time.time() - t0
    diagnostics = dict(result.diagnostics)
    samples = (
        np.stack(result.samples, axis=0).astype(np.int8)
        if result.samples
        else np.zeros((0, data.p, data.p), dtype=np.int8)
    )
    _atomic_npz(
        path,
        edge_probs=np.asarray(result.edge_probs, dtype=float),
        samples=samples,
        diagnostics_json=np.array(json.dumps(_json_sanitise(diagnostics))),
        runtime_s=np.array(runtime_s),
    )
    cache.store(path)
    return np.asarray(result.edge_probs, dtype=float), samples, diagnostics, runtime_s


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


def _target_index(columns: Sequence[str], target: str) -> int:
    if target not in columns:
        raise ValueError(f"target column {target!r} is absent from {tuple(columns)!r}")
    return int(tuple(columns).index(target))


def _posterior_parent_sets(
    samples: np.ndarray,
    target_idx: int,
    *,
    top_k: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, list[int]]:
    if samples.ndim != 3:
        raise ValueError(f"MCMC samples must be (s, p, p), got {samples.shape}")
    if samples.shape[0] == 0:
        raise RuntimeError("posterior parent sets require MCMC graph samples")
    counts: dict[tuple[int, ...], int] = {}
    for sample in samples:
        parents = tuple(int(i) for i in np.flatnonzero(sample[:, target_idx]))
        parents = tuple(i for i in parents if i != target_idx)
        counts[parents] = counts.get(parents, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    parent_sets = [k for k, _v in ranked]
    raw_counts = [int(v) for _k, v in ranked]
    weights = np.asarray(raw_counts, dtype=float)
    weights /= float(weights.sum())
    return parent_sets, weights, raw_counts


def _survival_time_grid(time_arr: np.ndarray) -> np.ndarray:
    t = np.asarray(time_arr, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("survival GAM requires finite positive follow-up times")
    t_min = max(float(np.quantile(t, 0.02)), 1e-3)
    t_max = float(np.max(t))
    if t_max <= t_min:
        raise ValueError("survival follow-up times have no usable range")
    return np.linspace(t_min, t_max, SURVIVAL_TIME_GRID_POINTS)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_vals = np.take_along_axis(values, order, axis=0)
    sorted_w = np.take_along_axis(
        np.broadcast_to(weights[:, None, None], values.shape), order, axis=0
    )
    cdf = np.cumsum(sorted_w, axis=0)
    pick = np.argmax(cdf >= q, axis=0)
    return np.take_along_axis(sorted_vals, pick[None, :, :], axis=0)[0]


def _survival_metrics(
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    survival_mean: np.ndarray,
    t_grid: np.ndarray,
) -> dict[str, Any]:
    eval_times = np.asarray(SURVIVAL_EVAL_TIMES, dtype=float)
    t_idx = int(np.argmin(np.abs(t_grid - 10.0)))
    p_event = 1.0 - survival_mean[:, t_idx]
    y_event = ((time_arr <= 10.0) & (event_arr == 1)).astype(int)
    calibration = calibration_metrics(y_event, p_event, n_bins=10, strategy="quantile")
    td = time_dependent_auc(
        time=time_arr,
        event=event_arr,
        risk_score=p_event,
        eval_times=eval_times,
    )
    br = brier_score(
        time=time_arr,
        event=event_arr,
        survival_pred=survival_mean,
        eval_times=t_grid,
    )
    return {
        "nagelkerke_r2_at_10y": float(nagelkerke_r2(y_event, p_event)),
        "calibration_at_10y": calibration,
        "time_dependent_auc": {
            "times": td["times"],
            "auc": td["auc"],
            "integrated_auc": float(td["integrated_auc"]),
        },
        "brier": br,
    }


def _load_or_run_survival_gam(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    samples: np.ndarray,
    logger: logging.Logger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    dict[str, Any],
    float,
]:
    filename = f"survival-gam-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[gam] using cache %s", path)
            return (
                z["t_grid"],
                z["survival_mean"],
                z["survival_lower"],
                z["survival_upper"],
                tuple(json.loads(str(z["parent_columns_json"].item()))),
                json.loads(str(z["diagnostics_json"].item())),
                float(z["runtime_s"].item()),
            )

    if not np.any(data.time > 0.0):
        raise RuntimeError("survival GAM requires time/event columns in the cohort CSV")
    if not np.any(data.event == 1):
        raise RuntimeError("survival GAM requires at least one observed disease event")

    target_idx = _target_index(data.columns, SURVIVAL_TARGET_COLUMN)
    parent_sets, weights, parent_set_counts = _posterior_parent_sets(
        samples,
        target_idx,
        top_k=min(CAUSAL_PATH_TOP_K, 8),
    )
    t_grid = _survival_time_grid(data.time)
    per_model = []
    fit_summaries = []
    t0 = time.time()
    logger.info("[gam] fitting %d gamfit survival parent-set models", len(parent_sets))
    for parent_set, weight, count in zip(parent_sets, weights, parent_set_counts):
        cols = tuple(data.columns[i] for i in parent_set)
        X = data.X[:, list(parent_set)] if parent_set else np.zeros((data.n, 0))
        fit = fit_survival_gam(
            data.time,
            data.event,
            X,
            columns=cols,
            n_samples=GAM_N_SAMPLES,
            rng=np.random.default_rng(PIPELINE_SEED + 20),
        )
        draws = fit.predict_survival(X, t_grid)
        mean = draws.mean(axis=0)
        per_model.append(mean)
        diag = fit.posterior_summary()
        diag["posterior_parent_set_weight"] = float(weight)
        diag["posterior_parent_set_count"] = int(count)
        diag["parent_columns"] = list(cols)
        fit_summaries.append(diag)

    stack = np.stack(per_model, axis=0)
    survival_mean = np.einsum("k,knt->nt", weights, stack)
    survival_lower = _weighted_quantile(stack, weights, 0.05)
    survival_upper = _weighted_quantile(stack, weights, 0.95)
    variance_structural = np.einsum(
        "k,knt->nt", weights, (stack - survival_mean[None, :, :]) ** 2
    )
    runtime_s = time.time() - t0
    diagnostics = {
        "backend": "gamfit",
        "n_parent_sets": int(len(parent_sets)),
        "parent_sets": [
            {
                "columns": [data.columns[i] for i in ps],
                "weight": float(w),
                "count": int(c),
            }
            for ps, w, c in zip(parent_sets, weights, parent_set_counts)
        ],
        "fit_summaries": fit_summaries,
        "variance_structural_mean": float(np.mean(variance_structural)),
        "metrics": _survival_metrics(data.time, data.event, survival_mean, t_grid),
    }
    parent_columns = tuple(
        dict.fromkeys(col for ps in parent_sets for col in (data.columns[i] for i in ps))
    )
    _atomic_npz(
        path,
        t_grid=t_grid,
        survival_mean=survival_mean,
        survival_lower=survival_lower,
        survival_upper=survival_upper,
        variance_structural=variance_structural,
        parent_columns_json=np.array(json.dumps(list(parent_columns))),
        diagnostics_json=np.array(json.dumps(_json_sanitise(diagnostics))),
        runtime_s=np.array(runtime_s),
    )
    cache.store(path)
    return (
        t_grid,
        survival_mean,
        survival_lower,
        survival_upper,
        parent_columns,
        diagnostics,
        runtime_s,
    )


def _top_paths_to_target(
    edge_probs: np.ndarray,
    target_idx: int,
) -> list[tuple[tuple[int, ...], float]]:
    parents = [[] for _ in range(edge_probs.shape[0])]
    for j in range(edge_probs.shape[0]):
        for i in range(edge_probs.shape[0]):
            if i == j:
                continue
            prob = float(edge_probs[i, j])
            if np.isfinite(prob) and prob >= CAUSAL_PATH_MIN_EDGE_PROB:
                parents[j].append((i, prob))
    found: list[tuple[tuple[int, ...], float]] = []

    def dfs(node: int, path: tuple[int, ...], prob: float) -> None:
        if len(path) > CAUSAL_PATH_MAX_DEPTH:
            return
        if len(path) >= 2:
            found.append((tuple(reversed(path)), prob))
        for parent, edge_prob in parents[node]:
            if parent in path:
                continue
            dfs(parent, path + (parent,), prob * edge_prob)

    dfs(target_idx, (target_idx,), 1.0)
    best: dict[tuple[int, ...], float] = {}
    for path, prob in found:
        best[path] = max(best.get(path, 0.0), float(prob))
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:CAUSAL_PATH_TOP_K]


def _causal_pathway_probabilities(
    edge_probs: np.ndarray,
    samples: np.ndarray,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    target_idx = _target_index(columns, CAUSAL_PATH_TARGET)
    if samples.ndim != 3 or samples.shape[0] == 0:
        raise RuntimeError("causal pathway probabilities require MCMC graph samples")
    paths = _top_paths_to_target(edge_probs, target_idx)
    out = []
    for path, marginal_product in paths:
        present = np.ones(samples.shape[0], dtype=bool)
        for parent, child in zip(path[:-1], path[1:]):
            present &= samples[:, parent, child].astype(bool)
        posterior_prob = float(present.mean())
        out.append(
            {
                "path": [columns[i] for i in path],
                "posterior_probability": posterior_prob,
                "marginal_edge_product": float(marginal_product),
            }
        )
    return out


def run_pipeline() -> PipelineResult:
    """Run the single production path."""
    logger = logging.getLogger("causal_pred.pipeline")
    if not logger.handlers:
        logger = _setup_logger(PIPELINE_VERBOSE)
    _install_signal_handlers(logger)
    cache = _cache()
    timings: dict[str, float] = {}
    pipeline_started_at = time.time()

    with _phase(logger, "data"):
        t0 = time.time()
        logger.info(
            "[data] resolving cohort CSV name=%s cache_dir=%s",
            COHORT_NAME,
            DEFAULT_CACHE_DIR,
        )
        csv_path = resolve_cohort_csv(name=COHORT_NAME, cache_dir=DEFAULT_CACHE_DIR)
        data, person_ids = load_cohort_dataset_with_person_ids(str(csv_path))
        timings["data"] = time.time() - t0
        logger.info(
            "[data] path=%s n=%d p=%d elapsed=%s columns=%s",
            csv_path,
            data.n,
            data.p,
            _format_seconds(timings["data"]),
            list(data.columns),
        )

    with _phase(logger, "prs"):
        t0 = time.time()
        logger.info("[prs] loading or building cohort PRS panel")
        prs_df, prs_path = _load_or_build_prs_panel(cache, csv_path, person_ids, logger)
        data, kept_person_ids, prs_meta = _augment_with_prs_nodes(
            data, person_ids, prs_df, logger
        )
        person_id_rows_after_prs = int(kept_person_ids.size)
        timings["prs"] = time.time() - t0
        logger.info(
            "[prs] selected=%d n=%d p=%d dropped_rows=%d elapsed=%s nodes=%s",
            prs_meta["prs_columns_selected"],
            data.n,
            data.p,
            int(len(person_ids) - data.n),
            _format_seconds(timings["prs"]),
            prs_meta["prs_node_names"],
        )

    survival_outcome_meta: dict[str, Any] = {
        "source": "cohort_csv",
        "n_input": int(data.n),
        "n_kept": int(data.n),
        "n_events": int(data.event.sum()),
    }
    if not _has_survival_outcome(data):
        with _phase(logger, "survival-outcome"):
            t0 = time.time()
            outcome = _load_or_build_survival_outcome(cache, kept_person_ids, logger)
            data, kept_person_ids = _apply_survival_outcome(
                data,
                kept_person_ids,
                outcome,
            )
            survival_outcome_meta = outcome.meta
            timings["survival_outcome"] = time.time() - t0
            logger.info(
                "[survival] outcome ready n=%d events=%d elapsed=%s",
                data.n,
                int(data.event.sum()),
                _format_seconds(timings["survival_outcome"]),
            )
    else:
        timings["survival_outcome"] = 0.0
        logger.info(
            "[survival] using cohort CSV time/event columns n=%d events=%d",
            data.n,
            int(data.event.sum()),
        )

    with _phase(logger, "ehr"):
        t0 = time.time()
        ehr_panel = _load_or_build_ehr_panel(cache, kept_person_ids, logger)
        timings["ehr"] = time.time() - t0
        logger.info("[ehr] complete elapsed=%s", _format_seconds(timings["ehr"]))

    with _phase(logger, "genscore"):
        t0 = time.time()
        data, kept_person_ids, genscore_meta = _load_or_run_genscore_features(
            cache,
            data,
            kept_person_ids,
            prs_df,
            ehr_panel,
            logger,
        )
        timings["genscore"] = time.time() - t0
        logger.info(
            "[genscore] complete n=%d p=%d elapsed=%s",
            data.n,
            data.p,
            _format_seconds(timings["genscore"]),
        )

    with _phase(logger, "mrdag"):
        t0 = time.time()
        mrdag_pi, mrdag_diagnostics = _load_or_run_mrdag(cache, logger)
        mrdag_prior = _mrdag_prior_for_data(mrdag_pi, data.columns)
        timings["mrdag"] = time.time() - t0
        _log_mrdag_diagnostics(logger, mrdag_pi, mrdag_diagnostics)
        logger.info("[mrdag] complete elapsed=%s", _format_seconds(timings["mrdag"]))

    with _phase(logger, "dagslam"):
        key = _run_key(data, mrdag_prior)
        dagslam = _load_or_run_dagslam(cache, key, data, logger)
        timings["dagslam"] = float(dagslam["runtime_s"])
        logger.info(
            "[dagslam] log_score=%.3f n_edges=%d",
            float(dagslam["log_score"]),
            int(dagslam["n_edges"]),
        )
        _log_dagslam_top_edges(logger, dagslam["adjacency"], data.columns)

    with _phase(logger, "mcmc"):
        edge_probs, mcmc_samples, mcmc_diagnostics, mcmc_runtime = _load_or_run_mcmc(
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
        _log_mcmc_diagnostics(
            logger,
            edge_probs,
            mcmc_diagnostics,
            data.columns,
            float(THRESHOLD_DEFAULT),
        )

    thresholded = (edge_probs >= THRESHOLD_DEFAULT).astype(int)
    np.fill_diagonal(thresholded, 0)

    with _phase(logger, "gam"):
        (
            survival_time_grid,
            survival_mean,
            survival_lower,
            survival_upper,
            survival_parent_columns,
            survival_diagnostics,
            survival_runtime,
        ) = _load_or_run_survival_gam(cache, key, data, mcmc_samples, logger)
        timings["gam"] = survival_runtime
        logger.info(
            "[gam] parent_columns=%s n_parent_sets=%d",
            list(survival_parent_columns),
            int(survival_diagnostics.get("n_parent_sets", 0)),
        )
        _log_survival_metrics(logger, survival_diagnostics)

    causal_pathways = _causal_pathway_probabilities(
        edge_probs,
        mcmc_samples,
        data.columns,
    )
    _log_causal_pathways(logger, causal_pathways)

    validation = _validate_edges(
        edge_probs,
        data.columns,
        rng=np.random.default_rng(PIPELINE_SEED + 4),
    )
    validation["survival"] = survival_diagnostics.get("metrics", {})
    _log_validation_metrics(logger, validation)

    logger.info("[summary] === pipeline metrics ===")
    logger.info(
        "[summary] cohort: n=%d p=%d events=%d event_rate=%.4f",
        data.n,
        data.p,
        int(data.event.sum()),
        float(np.mean(data.event)),
    )
    logger.info(
        "[summary] timings: %s",
        ", ".join(
            f"{k}={_format_seconds(float(v))}" for k, v in timings.items()
        ),
    )
    logger.info(
        "[summary] mrdag: max_rhat=%.3f candidate_edges=%d",
        float(mrdag_diagnostics.get("max_rhat_on_allowed", float("nan"))),
        int(mrdag_diagnostics.get("n_candidate_edges", 0)),
    )
    logger.info(
        "[summary] dagslam: log_score=%.3f n_edges=%d",
        float(dagslam["log_score"]),
        int(dagslam["n_edges"]),
    )
    n_off = data.p * (data.p - 1)
    logger.info(
        "[summary] mcmc: accept=%.3f max_rhat_skel=%.3f min_ess=%.1f "
        "edges_above_%.2f=%d/%d",
        float(mcmc_diagnostics["accept_rate"]["overall"]),
        float(mcmc_diagnostics.get("max_rhat_skeleton", float("nan"))),
        float(mcmc_diagnostics.get("min_ess", float("nan"))),
        float(THRESHOLD_DEFAULT),
        int(np.sum(thresholded)),
        int(n_off),
    )
    if "auroc" in validation:
        logger.info(
            "[summary] validation: AUROC=%.3f AUPRC=%.3f",
            float(validation["auroc"]),
            float(validation["auprc"]),
        )
    surv_metrics = survival_diagnostics.get("metrics") or {}
    if surv_metrics:
        td = surv_metrics.get("time_dependent_auc") or {}
        br = surv_metrics.get("brier") or {}
        cal = surv_metrics.get("calibration_at_10y") or {}
        logger.info(
            "[summary] survival: integrated_AUC=%.3f integrated_Brier=%.4f "
            "ECE_10y=%.4f Nagelkerke_R2_10y=%.3f",
            float(td.get("integrated_auc", float("nan"))),
            float(br.get("integrated_brier", float("nan"))),
            float(cal.get("ece", float("nan"))),
            float(surv_metrics.get("nagelkerke_r2_at_10y", float("nan"))),
        )

    logger.info(
        "[pipeline] complete elapsed=%s n=%d p=%d",
        _format_seconds(time.time() - pipeline_started_at),
        data.n,
        data.p,
    )

    return PipelineResult(
        person_ids=tuple(str(p) for p in kept_person_ids),
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
            "person_id_rows_after_prs": person_id_rows_after_prs,
            "person_id_rows_final": int(kept_person_ids.size),
            "event_rate": float(np.mean(data.event)),
            "survival_outcome": survival_outcome_meta,
        },
        mrdag_pi=mrdag_pi,
        mrdag_prior=mrdag_prior,
        mrdag_diagnostics=mrdag_diagnostics,
        dagslam_adjacency=np.asarray(dagslam["adjacency"], dtype=int),
        dagslam_log_score=float(dagslam["log_score"]),
        dagslam_n_edges=int(dagslam["n_edges"]),
        mcmc_edge_probs=edge_probs,
        mcmc_samples=mcmc_samples,
        mcmc_diagnostics=mcmc_diagnostics,
        thresholded_adjacency=thresholded,
        threshold=float(THRESHOLD_DEFAULT),
        survival_time_grid=survival_time_grid,
        survival_mean=survival_mean,
        survival_lower=survival_lower,
        survival_upper=survival_upper,
        survival_diagnostics=survival_diagnostics,
        survival_parent_columns=survival_parent_columns,
        causal_pathways=causal_pathways,
        validation=validation,
        timings=timings,
        genscore_features={"prs_path": prs_path, **prs_meta, **genscore_meta},
        cache_key=key,
    )


def _format_top_edges(
    matrix: np.ndarray,
    columns: Sequence[str],
    top_k: int = 10,
    min_value: float = 0.0,
) -> str:
    """Return a one-line ``"parent->child=0.93, ..."`` summary of the top edges."""
    rows: list[tuple[str, str, float]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if i == j:
                continue
            v = float(matrix[i, j])
            if not np.isfinite(v) or v <= min_value:
                continue
            rows.append((parent, child, v))
    rows.sort(key=lambda r: -r[2])
    if not rows:
        return "(none)"
    return ", ".join(f"{p}->{c}={v:.3f}" for p, c, v in rows[:top_k])


def _log_mrdag_diagnostics(
    logger: logging.Logger,
    pi: np.ndarray,
    diagnostics: dict[str, Any],
) -> None:
    accept_rates = diagnostics.get("accept_rates") or []
    accept_str = "[" + ", ".join(f"{float(a):.3f}" for a in accept_rates) + "]"
    logger.info(
        "[mrdag] candidate_edges=%d chains=%d samples_per_chain=%s "
        "max_rhat=%.3f between_chain_max_abs_diff=%.3f accept_per_chain=%s",
        int(diagnostics.get("n_candidate_edges", 0)),
        int(diagnostics.get("n_chains", len(accept_rates)))
        if "n_chains" in diagnostics
        else len(accept_rates),
        list(diagnostics.get("n_samples_per_chain", [])),
        float(diagnostics.get("max_rhat_on_allowed", float("nan"))),
        float(diagnostics.get("between_chain_max_abs_diff", float("nan"))),
        accept_str,
    )
    logger.info(
        "[mrdag] mr_traits=%s prior_incl=%.3f prior_effect_var=%.4f "
        "min_effect_scale=%.3f",
        list(diagnostics.get("mr_traits", [])),
        float(diagnostics.get("prior_incl", float("nan"))),
        float(diagnostics.get("prior_effect_var", float("nan"))),
        float(diagnostics.get("min_effect_scale", float("nan"))),
    )
    logger.info("[mrdag] top_pi: %s", _format_top_edges(pi, NODE_NAMES, top_k=10))


def _log_dagslam_top_edges(
    logger: logging.Logger,
    adjacency: np.ndarray,
    columns: Sequence[str],
) -> None:
    edges = _adj_to_edge_list(np.asarray(adjacency, dtype=int), columns)
    if not edges:
        logger.info("[dagslam] top_edges: (none)")
        return
    preview = ", ".join(f"{p}->{c}" for p, c in edges[:15])
    suffix = "" if len(edges) <= 15 else f" (+{len(edges) - 15} more)"
    logger.info("[dagslam] top_edges: %s%s", preview, suffix)


def _log_mcmc_diagnostics(
    logger: logging.Logger,
    edge_probs: np.ndarray,
    diagnostics: dict[str, Any],
    columns: Sequence[str],
    threshold: float,
) -> None:
    accept_rate = diagnostics.get("accept_rate", {}) or {}
    by_type = ", ".join(
        f"{k}={float(v):.3f}" for k, v in accept_rate.items() if k != "overall"
    )
    logger.info(
        "[mcmc] accept_per_move: %s",
        by_type or "(no proposals)",
    )
    logger.info(
        "[mcmc] max_rhat_directed=%.3f max_rhat_skeleton=%.3f min_ess=%.1f "
        "n_samples_per_chain=%s mean_logpost_per_chain=%s",
        float(diagnostics.get("max_rhat_directed", float("nan"))),
        float(diagnostics.get("max_rhat_skeleton", float("nan"))),
        float(diagnostics.get("min_ess", float("nan"))),
        list(diagnostics.get("n_samples_per_chain", [])),
        [
            f"{float(x):.3f}"
            for x in diagnostics.get("mean_log_posterior_per_chain", [])
        ],
    )
    n = edge_probs.shape[0]
    off = ~np.eye(n, dtype=bool)
    flat = edge_probs[off]
    n_above = int(np.sum(flat >= threshold))
    n_certain = int(np.sum(flat >= 0.95))
    n_zero = int(np.sum(flat <= 0.05))
    logger.info(
        "[mcmc] edges_above_%.2f=%d / %d  (>=0.95: %d, <=0.05: %d)",
        float(threshold),
        n_above,
        int(off.sum()),
        n_certain,
        n_zero,
    )
    logger.info(
        "[mcmc] top_edges: %s",
        _format_top_edges(edge_probs, columns, top_k=15),
    )


def _log_validation_metrics(
    logger: logging.Logger, validation: dict[str, Any]
) -> None:
    if not validation or "auroc" not in validation:
        logger.info("[validation] no known-edge ground truth available")
        return
    logger.info(
        "[validation] AUROC=%.3f (null %.3f)  AUPRC=%.3f (null %.3f)  "
        "ground_truth_edges=%d/%d",
        float(validation.get("auroc", float("nan"))),
        float(validation.get("auroc_null_mean", float("nan"))),
        float(validation.get("auprc", float("nan"))),
        float(validation.get("auprc_null_mean", float("nan"))),
        int(validation.get("n_valid_ground_truth_edges", 0)),
        int(validation.get("n_ground_truth_edges", 0)),
    )
    thresholds = list(validation.get("thresholds") or [])
    recovery = list(validation.get("observed_recovery") or [])
    rec_p = list(validation.get("recovery_pvalue") or [])
    mcc = list(validation.get("mcc") or [])
    mcc_p = list(validation.get("mcc_pvalue") or [])
    for idx, thr in enumerate(thresholds):
        rec = recovery[idx] if idx < len(recovery) else float("nan")
        rp = rec_p[idx] if idx < len(rec_p) else float("nan")
        mc = mcc[idx] if idx < len(mcc) else float("nan")
        mp = mcc_p[idx] if idx < len(mcc_p) else float("nan")
        logger.info(
            "[validation] thr=%.2f recovery=%.3f (p=%.3f)  MCC=%.3f (p=%.3f)",
            float(thr),
            float(rec),
            float(rp),
            float(mc),
            float(mp),
        )
    per_edge = validation.get("per_edge") or {}
    if per_edge:
        rows = sorted(
            per_edge.items(),
            key=lambda kv: -float(kv[1].get("probability", 0.0))
            if not kv[1].get("masked", False)
            else 1.0,
        )
        preview = ", ".join(
            f"{p}->{c}={float(d['probability']):.2f}"
            + ("" if not d.get("masked", False) else "(masked)")
            for (p, c), d in rows
        )
        logger.info("[validation] known_edges: %s", preview)


def _log_survival_metrics(
    logger: logging.Logger, diagnostics: dict[str, Any]
) -> None:
    metrics = diagnostics.get("metrics") or {}
    if not metrics:
        logger.info("[gam] no survival metrics computed")
        return
    td = metrics.get("time_dependent_auc") or {}
    times = list(td.get("times") or [])
    aucs = list(td.get("auc") or [])
    pairs = ", ".join(
        f"t={float(t):.1f}:AUC={float(a):.3f}" for t, a in zip(times, aucs)
    )
    logger.info(
        "[gam] integrated_AUC=%.3f  per-time: %s",
        float(td.get("integrated_auc", float("nan"))),
        pairs or "(none)",
    )
    br = metrics.get("brier") or {}
    logger.info(
        "[gam] integrated_Brier=%.4f  Brier_at_eval=%s",
        float(br.get("integrated_brier", float("nan"))),
        [f"{float(x):.4f}" for x in (br.get("brier") or [])],
    )
    cal = metrics.get("calibration_at_10y") or {}
    decomp = cal.get("brier_decomposition") or {}
    logger.info(
        "[gam] calib_at_10y: Brier=%.4f reliability=%.4f resolution=%.4f "
        "uncertainty=%.4f ECE=%.4f MCE=%.4f HL_p=%.3f Nagelkerke_R2=%.3f",
        float(cal.get("brier", float("nan"))),
        float(decomp.get("reliability", float("nan"))),
        float(decomp.get("resolution", float("nan"))),
        float(decomp.get("uncertainty", float("nan"))),
        float(cal.get("ece", float("nan"))),
        float(cal.get("mce", float("nan"))),
        float(cal.get("hl_pvalue", float("nan"))),
        float(metrics.get("nagelkerke_r2_at_10y", float("nan"))),
    )


def _log_causal_pathways(
    logger: logging.Logger, pathways: Sequence[dict[str, Any]]
) -> None:
    if not pathways:
        logger.info("[pathways] (none above threshold)")
        return
    for rank, row in enumerate(pathways[:10], start=1):
        path_txt = " -> ".join(row["path"])
        logger.info(
            "[pathways] %2d. %s  posterior=%.3f  marginal=%.3f",
            rank,
            path_txt,
            float(row["posterior_probability"]),
            float(row["marginal_edge_product"]),
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
    paths["mcmc_samples"] = str(out / "mcmc_samples.npy")
    np.save(paths["mcmc_samples"], result.mcmc_samples)
    paths["thresholded_adjacency"] = str(out / "thresholded_adjacency.npy")
    np.save(paths["thresholded_adjacency"], result.thresholded_adjacency)
    paths["survival_time_grid"] = str(out / "survival_time_grid.npy")
    np.save(paths["survival_time_grid"], result.survival_time_grid)
    paths["survival_mean"] = str(out / "survival_mean.npy")
    np.save(paths["survival_mean"], result.survival_mean)
    paths["survival_lower"] = str(out / "survival_lower.npy")
    np.save(paths["survival_lower"], result.survival_lower)
    paths["survival_upper"] = str(out / "survival_upper.npy")
    np.save(paths["survival_upper"], result.survival_upper)
    paths["disease_risk_mean"] = str(out / "disease_risk_mean.npy")
    np.save(paths["disease_risk_mean"], 1.0 - result.survival_mean)

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

    paths["survival_curves_long_csv"] = str(out / "survival_curves_long.csv")
    with open(paths["survival_curves_long_csv"], "w") as fh:
        fh.write(
            "person_id,time,survival_mean,survival_lower,survival_upper,"
            "disease_risk_mean\n"
        )
        for row_idx, person_id in enumerate(result.person_ids):
            for time_idx, t in enumerate(result.survival_time_grid):
                s_mean = float(result.survival_mean[row_idx, time_idx])
                s_lower = float(result.survival_lower[row_idx, time_idx])
                s_upper = float(result.survival_upper[row_idx, time_idx])
                fh.write(
                    f"{person_id},{float(t)},{s_mean},{s_lower},{s_upper},"
                    f"{1.0 - s_mean}\n"
                )

    paths["causal_pathways_csv"] = str(out / "causal_pathway_probabilities.csv")
    with open(paths["causal_pathways_csv"], "w") as fh:
        fh.write("rank,path,posterior_probability,marginal_edge_product\n")
        for rank, row in enumerate(result.causal_pathways, start=1):
            path_txt = " -> ".join(row["path"])
            fh.write(
                f"{rank},{path_txt},{row['posterior_probability']},"
                f"{row['marginal_edge_product']}\n"
            )

    paths["crosscoder_features_json"] = str(out / "crosscoder_features.json")
    with open(paths["crosscoder_features_json"], "w") as fh:
        json.dump(_json_sanitise(result.genscore_features), fh, indent=2, sort_keys=True)

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
        "survival_parent_columns": list(result.survival_parent_columns),
        "survival_diagnostics": result.survival_diagnostics,
        "causal_pathways": result.causal_pathways,
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
    if bucket is None:
        return
    dst = f"{bucket}/{WORKSPACE_RESULTS_PREFIX}"
    proc = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", str(outdir), dst],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "gsutil rsync of outputs failed exit_code=%d src=%s dst=%s\n"
            "stderr (last 4KiB):\n%s"
            % (proc.returncode, outdir, dst, (proc.stderr or "")[-4096:])
        )


def main() -> int:
    logger = _setup_logger(PIPELINE_VERBOSE)
    _install_signal_handlers(logger)
    started_at = time.time()
    try:
        result = run_pipeline()
        save_result(result)
        outdir = Path(DEFAULT_OUTPUT_DIR)
        _sync_outputs_to_workspace(outdir)
        logger.info(
            "[pipeline] artefacts written to %s total_elapsed=%s",
            outdir,
            _format_seconds(time.time() - started_at),
        )
        print(f"\nArtefacts written to {outdir}")
        return 0
    except KeyboardInterrupt:
        logger.error(
            "[pipeline] interrupted by user after total_elapsed=%s",
            _format_seconds(time.time() - started_at),
        )
        _flush_log_handlers(logger)
        return 130
    except SystemExit:
        _flush_log_handlers(logger)
        raise
    except BaseException as exc:
        logger.error(
            "[pipeline] FAILED after total_elapsed=%s with %s: %s\n%s",
            _format_seconds(time.time() - started_at),
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        _flush_log_handlers(logger)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
