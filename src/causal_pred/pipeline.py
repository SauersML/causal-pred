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
import math
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
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
    CURATED_OMOP_MEASUREMENT_CATALOG,
    EhrPanel,
    SurvivalOutcome,
    T2D_EHR_CONDITION_BLACKLIST_IDS,
    T2D_TREATMENT_DRUG_PREFIXES,
    build_ehr_panel,
    build_survival_outcome,
    discover_genotype_dir,
    fetch_omop_long_frames,
    load_cohort_dataset_with_person_ids,
    resolve_aou_genotypes,
    resolve_cohort_csv,
)
from .data.nodes import CANONICAL_EDGES, NODES, NODE_INDEX, NODE_NAMES
from .data.polygenic import parse_sscore, score_panel
from .data.opengwas import load_live_gwas
from .data.synthetic import SyntheticDataset
from ._parallel import cpu_count, parallel_call
from .dagslam import run_dagslam
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
PGS_PANEL_DIRNAME = "pgs_panel"
PRS_PANEL_CACHE_DIRNAME = "prs_panel_cache"
GNOMON_OUT_DIRNAME = "gnomon_score"
GENOTYPE_CACHE_DIR = str(Path.home() / "causal-pred" / "genomes")

PIPELINE_CONFIG_VERSION = "2026-05-11.refactor-spine-graph-mcmc-names.1"
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

MCMC_SAMPLES = 500
MCMC_BURN_IN = 250
MCMC_THIN = 2
MCMC_CHAINS = 2
MCMC_EDGE_RESAMPLE_PROB = 0.9
MCMC_PARENT_RESAMPLE_PROB = 0.05
MCMC_PROGRESS_INTERVAL = 100
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
    "d": 2048,
    "k": 32,
    "n_steps": 10000,
    "batch_size": 1024,
    "lr": 3e-4,
    "train_dtype": "float32",
    "device": "auto",
    "activation_kind": "batch_topk",
    "row_cap_multiplier": 4.0,
    "shared_fraction": 0.5,
    "cross_reconstruction_coef": 0.35,
    "shared_alignment_coef": 0.05,
    "contrastive_coef": 0.02,
    "validation_fraction": 0.1,
    "mixed_likelihood": True,
}
GENSCORE_CROSSCODER_CHECKPOINT_EVERY = 1000

GAM_N_SAMPLES = 200
SURVIVAL_TIME_GRID_POINTS = 50
SURVIVAL_EVAL_TIMES = (5.0, 10.0, 15.0)
SURVIVAL_TARGET_COLUMN = "type2_diabetes"
SURVIVAL_GAM_MAX_PARENTS = 6
SURVIVAL_GAM_MIN_EDGE_PROB = 0.5
SURVIVAL_INTERVAL_Z_90 = 1.6448536269514722

CAUSAL_PATH_TARGET = "type2_diabetes"
CAUSAL_PATH_TOP_K = 20
CAUSAL_PATH_MIN_EDGE_PROB = 0.1
CAUSAL_PATH_MAX_DEPTH = 5

PGS_ANCHOR_PRIORS = {
    ("pgs_t2d", "type2_diabetes"): 0.95,
    ("pgs_bmi", "bmi"): 0.95,
    ("pgs_ldl", "ldl_cholesterol"): 0.95,
    ("pgs_hba1c", "hba1c"): 0.95,
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
    mcmc_samples: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=int)
    )
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
            uri = self.uri(filename)
            if uri is not None and not _gsutil_exists(uri):
                logger.info("[cache] workspace miss for local hit; mirroring %s", uri)
                self.store(dst, filename)
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


def _log_rss(logger: logging.Logger, label: str) -> None:
    """Log process RSS (Linux only, silently no-op elsewhere) at a checkpoint
    so we can spot memory growth before the kernel OOM-killer fires."""
    try:
        with open("/proc/self/status") as fh:
            vm_rss = vm_peak = vm_size = ""
            for line in fh:
                if line.startswith("VmRSS:"):
                    vm_rss = line.split(":", 1)[1].strip()
                elif line.startswith("VmPeak:"):
                    vm_peak = line.split(":", 1)[1].strip()
                elif line.startswith("VmSize:"):
                    vm_size = line.split(":", 1)[1].strip()
        if vm_rss:
            logger.info(
                "[mem] %s rss=%s peak=%s vsz=%s", label, vm_rss, vm_peak, vm_size
            )
    except OSError:
        pass


def _post_training_reconstruction_stats(
    model: Any,
    panels_aligned: Any,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode the full panel through the trained crosscoder and return
    (activation_rate_per_feature, ss_res_a, ss_tot_a, ss_res_b, ss_tot_b).

    Runs on the GPU when available - the full z_full = (n, d) float64 tensor
    fits comfortably in T4 VRAM (~1.7 GiB at 102 k x 2048) and using the
    full panel preserves batch_topk's global top-k semantics; chunking on
    CPU would change which features get activated.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_full = int(panels_aligned.A.shape[0])
    d_full = int(model.d)
    k = int(model.k)
    row_cap_multiplier = float(
        GENSCORE_CROSSCODER_KWARGS.get("row_cap_multiplier", 4.0)
    )
    logger.info(
        "[genscore] encoding full panel on %s n=%d d=%d k=%d",
        device,
        n_full,
        d_full,
        k,
    )

    def _t(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64)).to(device)

    A = _t(panels_aligned.A)
    B = _t(panels_aligned.B)
    mean_G = _t(model.mean_G)
    std_G = _t(model.std_G)
    mean_E = _t(model.mean_E)
    std_E = _t(model.std_E)
    W_e = _t(model.W_e)
    b_enc = _t(model.b_enc)
    W_d_G = _t(model.W_d_G)
    W_d_E = _t(model.W_d_E)

    a_z = (A - mean_G) / std_G
    b_z = (B - mean_E) / std_E
    del A, B
    pre = torch.cat([a_z, b_z], dim=1) @ W_e + b_enc
    del W_e, b_enc

    activation_kind = str(model.activation_kind)
    if activation_kind == "topk":
        if k >= d_full:
            mask = pre > 0
        else:
            topk_vals, topk_idx = torch.topk(pre, k, dim=1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(1, topk_idx, topk_vals > 0)
    elif activation_kind == "batch_topk":
        scores = torch.relu(pre)
        take = min(scores.numel(), max(1, n_full * k))
        flat = scores.flatten()
        if take >= flat.numel():
            mask = scores > 0
        else:
            topk_vals, topk_idx = torch.topk(flat, take, sorted=False)
            keep = topk_idx[topk_vals > 0]
            mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            mask_flat[keep] = True
            mask = mask_flat.view(n_full, d_full)
        if row_cap_multiplier > 0:
            cap = min(d_full, max(k, int(math.ceil(k * row_cap_multiplier))))
            if cap < d_full and bool(mask.any()):
                row_scores = torch.where(mask, scores, torch.zeros_like(scores))
                _, cap_idx = torch.topk(row_scores, cap, dim=1)
                cap_mask = torch.zeros_like(mask)
                cap_mask.scatter_(1, cap_idx, True)
                mask &= cap_mask
        del scores, flat
    else:
        raise ValueError(f"unknown activation_kind {activation_kind!r}")

    z = torch.where(mask, pre, torch.zeros_like(pre))
    del pre, mask

    a_hat = z @ W_d_G
    b_hat = z @ W_d_E
    activation_count = (z > 0).sum(dim=0)
    del z, W_d_G, W_d_E

    # Genome side: a_hat is in z-score space; standard R^2 is well-defined.
    ss_res_a = float(((a_z - a_hat) ** 2).sum().item())
    a_z_mean = a_z.mean(dim=0, keepdim=True)
    ss_tot_a = float(((a_z - a_z_mean) ** 2).sum().item())
    r2_g = 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 0 else float("nan")

    # EHR side: b_hat is in mixed natural-parameter space (z-score for
    # gaussian columns, logit for binary, log-rate for count). Compute
    # one metric per kind in its native space; aggregating into a single
    # R^2 is meaningless because most columns are binary.
    metrics: dict[str, float] = {"r2_genome": r2_g}
    B_raw_np = panels_aligned.B
    b_hat_np = b_hat.cpu().numpy()
    b_z_np = b_z.cpu().numpy()
    from .genscore.crosscoder import _ehr_kind_masks, _ehr_recon_metrics

    gaussian_e, binary_e, count_e = _ehr_kind_masks(
        tuple(model.ehr_feature_kinds), int(model.m_E)
    )
    metrics.update(
        _ehr_recon_metrics(
            b_z_np, B_raw_np, b_hat_np,
            gaussian=gaussian_e, binary=binary_e, count=count_e,
            mean_E=model.mean_E,
        )
    )
    activation_rate = (
        activation_count.to(torch.float64) / float(n_full)
    ).cpu().numpy()
    return activation_rate, metrics


class _AsyncUploader:
    """Coalescing background uploader for repeatedly-overwritten artefacts
    (e.g. crosscoder checkpoints). ``store`` returns immediately; if a
    previous upload is still running, the request is coalesced and the
    worker picks up the latest snapshot when it next polls. ``flush``
    blocks until any queued upload completes so callers can ensure the
    final state hits the bucket before exiting.
    """

    def __init__(self, cache: WorkspaceCache, label: str = "upload") -> None:
        self._cache = cache
        self._label = label
        self._lock = threading.Lock()
        self._next: Optional[tuple[Path, str]] = None
        self._thread: Optional[threading.Thread] = None

    def store(self, local_path: Path, remote_name: str) -> None:
        with self._lock:
            self._next = (Path(local_path), str(remote_name))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name=f"{self._label}-uploader",
                )
                self._thread.start()

    def _run(self) -> None:
        logger = logging.getLogger("causal_pred.pipeline")
        while True:
            with self._lock:
                target = self._next
                self._next = None
            if target is None:
                return
            local, remote = target
            snap = local.with_suffix(local.suffix + ".upload-tmp")
            try:
                shutil.copy2(local, snap)
                self._cache.store(snap, remote)
            except Exception as exc:
                logger.warning(
                    "[%s] async upload failed: %r", self._label, exc
                )
            finally:
                try:
                    snap.unlink()
                except FileNotFoundError:
                    pass

    def flush(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()


PIPELINE_LOG_FILENAME = "pipeline.log"


def _setup_logger(verbose: bool) -> logging.Logger:
    # Attach handlers to the package parent so submodules (e.g.
    # ``causal_pred.data.polygenic``, ``causal_pred.genscore.panels``)
    # passthrough into the same stdout stream and pipeline.log file.
    parent = logging.getLogger("causal_pred")
    parent.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    )
    parent.addHandler(stream_handler)
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
        parent.addHandler(file_handler)
    except OSError as exc:
        sys.stderr.write(
            f"[pipeline] WARNING failed to attach file log under {log_dir}: {exc}\n"
        )
        log_path = None
    parent.setLevel(logging.DEBUG if verbose else logging.INFO)
    parent.propagate = False

    logger = logging.getLogger("causal_pred.pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = True
    if log_path is not None:
        logger.info(
            "[pipeline] log file %s pid=%d python=%s",
            log_path,
            os.getpid(),
            sys.version.split()[0],
        )
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


def _workspace_cdr() -> Optional[str]:
    cdr = os.environ.get("WORKSPACE_CDR", "").strip()
    return cdr or None


def _gsutil_exists(uri: str) -> bool:
    if shutil.which("gsutil") is None:
        return False
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
            return _json_sanitise(obj.tolist())
        return {
            "__omitted_ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        if np.isfinite(obj):
            return obj
        if np.isnan(obj):
            return "NaN"
        return "Infinity" if obj > 0 else "-Infinity"
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    return str(obj)


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(
        _json_sanitise(obj), sort_keys=True, separators=(",", ":"), allow_nan=False
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


_SAMPLE_CHUNK_BYTES = 4 * (1 << 20)  # 4 MiB per probe
_SAMPLE_INTERIOR_CHUNKS = 6  # head + 6 interior probes + tail
_DIGEST_TAG = "content-v1"


def _sha256_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _read_sha256_sidecar(path: Path, size: int, mtime_ns: int) -> Optional[str]:
    sidecar = _sha256_sidecar_path(path)
    try:
        text = sidecar.read_text()
    except OSError:
        return None
    try:
        record = json.loads(text)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    if int(record.get("size", -1)) != int(size):
        return None
    if int(record.get("mtime_ns", -1)) != int(mtime_ns):
        return None
    if str(record.get("tag", "")) != _DIGEST_TAG:
        return None
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    return digest


def _write_sha256_sidecar(
    path: Path, size: int, mtime_ns: int, digest: str
) -> None:
    sidecar = _sha256_sidecar_path(path)
    payload = json.dumps(
        {
            "size": int(size),
            "mtime_ns": int(mtime_ns),
            "tag": _DIGEST_TAG,
            "sha256": digest,
        }
    )
    try:
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(payload)
        tmp.replace(sidecar)
    except OSError:
        pass


def _file_sha256(
    path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    label: Optional[str] = None,
) -> str:
    st = path.stat()
    size = int(st.st_size)
    mtime_ns = int(st.st_mtime_ns)
    cached = _read_sha256_sidecar(path, size, mtime_ns)
    if cached is not None:
        if logger is not None:
            logger.info(
                "[prs]   digest %s reused from sidecar (%.1f MiB)",
                label or path.name,
                size / (1 << 20),
            )
        return cached
    chunk = _SAMPLE_CHUNK_BYTES
    n_interior = _SAMPLE_INTERIOR_CHUNKS
    if logger is not None:
        logger.info(
            "[prs]   digest %s start size=%.1f MiB "
            "(probe=%.1f MiB, %d interior + head + tail)",
            label or path.name,
            size / (1 << 20),
            chunk / (1 << 20),
            n_interior,
        )
    started = time.time()
    h = hashlib.sha256()
    h.update(_DIGEST_TAG.encode("utf-8"))
    h.update(str(size).encode("utf-8"))
    h.update(b"|")
    raw_offsets = [0]
    for i in range(1, n_interior + 1):
        raw_offsets.append((size * i) // (n_interior + 1))
    raw_offsets.append(max(0, size - chunk))
    seen: set[tuple[int, int]] = set()
    with path.open("rb") as fh:
        for raw in raw_offsets:
            off = max(0, min(raw, max(0, size - chunk)))
            length = max(0, min(chunk, size - off))
            key = (off, length)
            if key in seen:
                continue
            seen.add(key)
            fh.seek(off)
            data = fh.read(length)
            h.update(off.to_bytes(8, "big"))
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)
    digest = h.hexdigest()
    if logger is not None:
        logger.info(
            "[prs]   digest %s done in %s",
            label or path.name,
            _format_seconds(time.time() - started),
        )
    _write_sha256_sidecar(path, size, mtime_ns, digest)
    return digest


def _file_stat_fingerprint(
    path: Path,
    *,
    logger: Optional[logging.Logger] = None,
    label: Optional[str] = None,
) -> dict[str, Any]:
    st = path.stat()
    return {
        "name": path.name,
        "size": int(st.st_size),
        "sha256": _file_sha256(path, logger=logger, label=label),
    }


def _plink_stat_fingerprint(
    bed: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    prefix = bed.with_suffix("")
    files = []
    for suffix in (".bed", ".bim", ".fam"):
        p = prefix.with_suffix(suffix)
        if logger is not None:
            logger.info(
                "[prs]  fingerprinting %s (%.1f MiB)",
                p,
                p.stat().st_size / (1 << 20),
            )
        t0 = time.time()
        files.append(_file_stat_fingerprint(p, logger=logger, label=p.name))
        if logger is not None:
            logger.info(
                "[prs]  fingerprinted %s in %s",
                p.name,
                _format_seconds(time.time() - t0),
            )
    return {"prefix": prefix.name, "files": files}


def _gnomon_score_fingerprint(
    bed: Path,
    score_files: Sequence[Path],
    *,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    if logger is not None:
        logger.info(
            "[prs] fingerprinting genotype PLINK fileset prefix=%s",
            bed.with_suffix(""),
        )
    t0 = time.time()
    genotype_fp = _plink_stat_fingerprint(bed, logger=logger)
    if logger is not None:
        logger.info(
            "[prs] genotype fingerprint done in %s; fingerprinting scorer",
            _format_seconds(time.time() - t0),
        )
    t_scorer = time.time()
    scorer_fp = _gnomon_scorer_fingerprint(logger=logger)
    if logger is not None:
        logger.info(
            "[prs] scorer fingerprint done in %s; fingerprinting %d score files",
            _format_seconds(time.time() - t_scorer),
            len(score_files),
        )
    score_fp = _score_files_fingerprint(score_files, logger=logger)
    # Raw gnomon scoring is a deterministic function of (genotype PLINK
    # fileset, gnomon scorer binary, PGS score files).  None of those
    # change when downstream MrDAG/MCMC/GAM config is bumped, so we do
    # NOT include PIPELINE_CONFIG_VERSION here -- doing so would force a
    # ~3-minute fresh score rebuild on every release.
    return {
        "genotype": genotype_fp,
        "scorer": scorer_fp,
        "score_files": score_fp,
    }


def _gnomon_score_cache_filename(
    bed: Path,
    score_files: Sequence[Path],
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    fp = _gnomon_score_fingerprint(bed, score_files, logger=logger)
    return "gnomon-scores-" + _short_hash(fp) + ".sscore"


def _gnomon_scorer_fingerprint(
    *, logger: Optional[logging.Logger] = None
) -> dict[str, Any]:
    binary = shutil.which("gnomon")
    if binary is None:
        raise RuntimeError(
            "gnomon binary not found on PATH; install gnomon before running the pipeline"
        )
    path = Path(binary).resolve()
    if logger is not None:
        logger.info(
            "[prs]  fingerprinting gnomon binary %s (%.1f MiB)",
            path,
            path.stat().st_size / (1 << 20),
        )
    st = path.stat()
    sha = _file_sha256(path, logger=logger, label=f"gnomon:{path.name}")
    if logger is not None:
        logger.info("[prs]  gnomon sha256=%s", sha[:12])
    return {"size": int(st.st_size), "sha256": sha}


def _score_files_fingerprint(
    score_files: Sequence[Path],
    *,
    logger: Optional[logging.Logger] = None,
) -> list[dict[str, Any]]:
    sorted_files = [
        Path(p) for p in sorted((Path(p) for p in score_files), key=lambda x: x.name)
    ]
    out: list[dict[str, Any]] = []
    total = len(sorted_files)
    started = time.time()
    log_every = max(1, total // 10) if logger is not None else total + 1
    for idx, p in enumerate(sorted_files, start=1):
        out.append(_file_stat_fingerprint(p, logger=logger, label=p.name))
        if logger is not None and (idx % log_every == 0 or idx == total):
            logger.info(
                "[prs]  fingerprinted %d/%d score files (elapsed=%s)",
                idx,
                total,
                _format_seconds(time.time() - started),
            )
    return out


def _prs_panel_cache_filename(
    bed: Path,
    score_files: Sequence[Path],
    person_ids: Sequence[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    if logger is not None:
        logger.info("[prs] fingerprinting genotype PLINK fileset prefix=%s", bed.with_suffix(""))
    t0 = time.time()
    genotype_fp = _plink_stat_fingerprint(bed, logger=logger)
    if logger is not None:
        logger.info(
            "[prs] genotype fingerprint done in %s",
            _format_seconds(time.time() - t0),
        )
        logger.info("[prs] fingerprinting %d PGS score files", len(score_files))
    t_scores = time.time()
    score_fp = _score_files_fingerprint(score_files, logger=logger)
    if logger is not None:
        logger.info(
            "[prs] score-file fingerprints done in %s",
            _format_seconds(time.time() - t_scores),
        )
        logger.info("[prs] fingerprinting gnomon scorer binary")
    t_scorer = time.time()
    scorer_fp = _gnomon_scorer_fingerprint(logger=logger)
    if logger is not None:
        logger.info(
            "[prs] scorer fingerprint done in %s",
            _format_seconds(time.time() - t_scorer),
        )
        logger.info("[prs] hashing %d person IDs into cache key", len(person_ids))
    return "aou-prs-panel-" + _short_hash(
        _prs_panel_fingerprint_from_parts(
            genotype_fp, score_fp, scorer_fp, person_ids
        )
    ) + ".csv.gz"


def _prs_panel_fingerprint_from_parts(
    genotype_fp: dict[str, Any],
    score_fp: list[dict[str, Any]],
    scorer_fp: dict[str, Any],
    person_ids: Sequence[str],
) -> dict[str, Any]:
    # The PRS panel is a pure function of (genotype, score files, gnomon
    # scorer binary, cohort person ids).  Pipeline config bumps that touch
    # only MrDAG / DAG-SLAM / MCMC / GAM must NOT invalidate this cache,
    # so PIPELINE_CONFIG_VERSION is intentionally omitted.
    return {
        "genotype": genotype_fp,
        "score_files": score_fp,
        "scorer": scorer_fp,
        "person_ids": [str(p) for p in person_ids],
    }


def _prs_panel_fingerprint(
    bed: Path,
    score_files: Sequence[Path],
    person_ids: Sequence[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    return _prs_panel_fingerprint_from_parts(
        _plink_stat_fingerprint(bed, logger=logger),
        _score_files_fingerprint(score_files, logger=logger),
        _gnomon_scorer_fingerprint(logger=logger),
        person_ids,
    )


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
            "t2d_condition_blacklist_ids": list(T2D_EHR_CONDITION_BLACKLIST_IDS),
            "t2d_treatment_drug_prefixes": list(T2D_TREATMENT_DRUG_PREFIXES),
            "measurement_catalog": [
                {"lab": lab, "loinc_codes": list(codes)}
                for lab, codes in CURATED_OMOP_MEASUREMENT_CATALOG
            ],
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
            "edge_resample_prob": MCMC_EDGE_RESAMPLE_PROB,
            "parent_resample_prob": MCMC_PARENT_RESAMPLE_PROB,
            "progress_interval": MCMC_PROGRESS_INTERVAL,
            "max_parents": DAGSLAM_MAX_PARENTS,
        },
        "structural_constraints": {
            "target_sink": SURVIVAL_TARGET_COLUMN,
            "exogenous_rule": "node-metadata-or-pgs-prefix",
            "promoted_crosscoder_features": "non_root_with_forbidden_target_to_feature_and_feature_to_pgs",
        },
        "threshold": THRESHOLD_DEFAULT,
        "gam": {
            "n_samples": GAM_N_SAMPLES,
            "time_grid_points": SURVIVAL_TIME_GRID_POINTS,
            "eval_times": list(SURVIVAL_EVAL_TIMES),
            "target_column": SURVIVAL_TARGET_COLUMN,
            "max_parents": SURVIVAL_GAM_MAX_PARENTS,
            "min_edge_prob": SURVIVAL_GAM_MIN_EDGE_PROB,
        },
        "causal_paths": {
            "target": CAUSAL_PATH_TARGET,
            "top_k": CAUSAL_PATH_TOP_K,
            "min_edge_prob": CAUSAL_PATH_MIN_EDGE_PROB,
            "max_depth": CAUSAL_PATH_MAX_DEPTH,
        },
    }


def _run_key(
    data: SyntheticDataset,
    mrdag_prior: np.ndarray,
    allowed_edges: np.ndarray,
) -> str:
    return _short_hash(
        {
            "config": _pipeline_config(),
            "columns": list(data.columns),
            "node_types": list(data.node_types),
            "x_sha256": _array_hash(data.X),
            "time_sha256": _array_hash(data.time),
            "event_sha256": _array_hash(data.event),
            "mrdag_prior_sha256": _array_hash(mrdag_prior),
            "allowed_edges_sha256": _array_hash(
                np.asarray(allowed_edges, dtype=np.bool_)
            ),
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


def _read_prs_panel(
    path: str | os.PathLike,
    *,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PRS panel not found: {p}")
    if logger is not None:
        logger.info(
            "[prs] reading PRS panel CSV %s (%.1f MiB)",
            p,
            p.stat().st_size / (1 << 20),
        )
    t0 = time.time()
    df = pd.read_csv(p, dtype={0: "string"})
    index_col = "person_id" if "person_id" in df.columns else df.columns[0]
    out = df.set_index(index_col)
    out.index = out.index.astype(str)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out.shape[1] == 0:
        raise ValueError(f"PRS panel {p} has no score columns")
    if logger is not None:
        logger.info(
            "[prs] PRS panel loaded rows=%d cols=%d in %s",
            out.shape[0],
            out.shape[1],
            _format_seconds(time.time() - t0),
        )
    return out


def _prs_cache_usable(
    path: Path,
    person_ids: Sequence[str],
    *,
    logger: Optional[logging.Logger] = None,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        if logger is not None:
            logger.info("[prs] cache check: %s missing or empty", path)
        return False
    if logger is not None:
        logger.info(
            "[prs] cache check: scanning ID column of %s (%.1f MiB)",
            path,
            path.stat().st_size / (1 << 20),
        )
    t0 = time.time()
    ids = pd.read_csv(path, usecols=[0], dtype={0: "string"}).iloc[:, 0]
    have = ids.astype("string").astype(str).to_numpy()
    want = pd.Series(person_ids, dtype="string").astype(str).to_numpy()
    ok = have.shape == want.shape and bool(np.array_equal(have, want))
    if logger is not None:
        logger.info(
            "[prs] cache check: %s have=%d want=%d match=%s elapsed=%s",
            path.name,
            int(have.shape[0]),
            int(want.shape[0]),
            ok,
            _format_seconds(time.time() - t0),
        )
    return ok


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


def _list_cache_basenames(
    *,
    cache: WorkspaceCache,
    local_dir: Path,
    remote_dir: str,
    pattern: str,
) -> list[str]:
    """Return basenames of cache files matching ``pattern`` in ``local_dir``
    (on disk) and ``remote_dir`` (in the bucket if configured), merged."""
    seen: set[str] = set()
    if local_dir.is_dir():
        for p in local_dir.glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                seen.add(p.name)
    if cache.bucket is not None:
        uri = f"{cache.bucket}/{WORKSPACE_CACHE_PREFIX}/{remote_dir}/{pattern}"
        try:
            out = subprocess.run(
                ["gsutil", "ls", uri],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except Exception:
            out = None
        if out is not None and out.returncode == 0:
            for line in out.stdout.splitlines():
                line = line.strip()
                if line:
                    seen.add(line.rsplit("/", 1)[-1])
    return sorted(seen)


_SIDECAR_SUFFIX = ".key.json"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _read_workspace_sidecar(
    cache: WorkspaceCache, local_path: Path, remote_name: str
) -> Optional[dict[str, Any]]:
    sidecar_remote = remote_name + _SIDECAR_SUFFIX
    sidecar_local = local_path.with_name(local_path.name + _SIDECAR_SUFFIX)
    fetched = cache.fetch(sidecar_remote, sidecar_local)
    if not fetched.is_file() or fetched.stat().st_size == 0:
        return None
    try:
        record = json.loads(fetched.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _write_workspace_sidecar(
    cache: WorkspaceCache,
    local_path: Path,
    remote_name: str,
    fingerprint: dict[str, Any],
) -> None:
    sidecar_remote = remote_name + _SIDECAR_SUFFIX
    sidecar_local = local_path.with_name(local_path.name + _SIDECAR_SUFFIX)
    sidecar_local.parent.mkdir(parents=True, exist_ok=True)
    sidecar_local.write_text(_canonical_json(fingerprint))
    cache.store(sidecar_local, sidecar_remote)


def _validate_cached_sscore(
    path: Path, person_ids: pd.Series, logger: logging.Logger
) -> Optional[pd.DataFrame]:
    try:
        scores = parse_sscore(path, keep_iids=person_ids.astype(str).tolist())
    except Exception as exc:
        logger.info("[prs] sscore %s failed to parse: %s", path, exc)
        return None
    min_rows = min(PRS_MIN_COMPLETE_ROWS, int(person_ids.size))
    overlap = int(person_ids.astype(str).isin(scores.index.astype(str)).sum())
    if overlap < min_rows or scores.shape[1] < PRS_NODES:
        logger.info(
            "[prs] sscore %s rejected (overlap=%d/%d cols=%d/min=%d)",
            path,
            overlap,
            min_rows,
            scores.shape[1],
            PRS_NODES,
        )
        return None
    return scores


def _restore_prs_panel(
    cache: Optional[WorkspaceCache],
    local_dir: Path,
    fingerprint: dict[str, Any],
    person_ids: Sequence[str],
    logger: logging.Logger,
) -> Optional[tuple[pd.DataFrame, Path]]:
    if cache is None:
        return None
    target = _canonical_json(fingerprint)
    candidates = _list_cache_basenames(
        cache=cache,
        local_dir=local_dir,
        remote_dir=PRS_PANEL_CACHE_DIRNAME,
        pattern="aou-prs-panel-*.csv.gz",
    )
    if not candidates:
        return None
    logger.info(
        "[prs] scanning %d cached PRS panel candidate(s)", len(candidates)
    )
    for basename in candidates:
        local = local_dir / basename
        remote = f"{PRS_PANEL_CACHE_DIRNAME}/{basename}"
        fetched = cache.fetch(remote, local, overwrite=True)
        if not fetched.is_file() or fetched.stat().st_size == 0:
            continue
        sidecar = _read_workspace_sidecar(cache, fetched, remote)
        if sidecar is not None:
            if _canonical_json(sidecar) != target:
                logger.info(
                    "[prs] panel sidecar mismatch for %s; skipping", remote
                )
                continue
            logger.info("[prs] panel sidecar match %s", remote)
            return _read_prs_panel(fetched, logger=logger), fetched
        if not _prs_cache_usable(fetched, person_ids, logger=logger):
            continue
        logger.info(
            "[prs] adopted legacy panel %s; writing sidecar", remote
        )
        _write_workspace_sidecar(cache, fetched, remote, fingerprint)
        return _read_prs_panel(fetched, logger=logger), fetched
    return None


def _restore_gnomon_scores(
    cache: Optional[WorkspaceCache],
    local_dir: Path,
    fingerprint: dict[str, Any],
    person_ids: pd.Series,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Single-path lookup. Scan local + bucket candidates, prefer sidecar
    match (O(1)), fall back to parse-validation for legacy entries and
    write a sidecar on first reuse so subsequent runs are O(1)."""
    if cache is None:
        return None
    target = _canonical_json(fingerprint)
    candidates = _list_cache_basenames(
        cache=cache,
        local_dir=local_dir,
        remote_dir=GNOMON_OUT_DIRNAME,
        pattern="gnomon-scores-*.sscore",
    )
    if not candidates:
        return None
    logger.info(
        "[prs] scanning %d cached gnomon score candidate(s)", len(candidates)
    )
    for basename in candidates:
        local = local_dir / basename
        remote = f"{GNOMON_OUT_DIRNAME}/{basename}"
        fetched = cache.fetch(remote, local)
        if not fetched.is_file() or fetched.stat().st_size == 0:
            continue
        sidecar = _read_workspace_sidecar(cache, fetched, remote)
        if sidecar is not None:
            if _canonical_json(sidecar) != target:
                logger.info("[prs] sidecar mismatch for %s; skipping", remote)
                continue
            logger.info(
                "[prs] sidecar match %s size=%.1fMiB",
                remote,
                _path_size_mib(fetched),
            )
            t_parse = time.time()
            scores = parse_sscore(
                fetched, keep_iids=person_ids.astype(str).tolist()
            )
            logger.info(
                "[prs] parsed cached gnomon scores rows=%d cols=%d elapsed=%s",
                scores.shape[0],
                scores.shape[1],
                _format_seconds(time.time() - t_parse),
            )
            return scores
        logger.info("[prs] no sidecar for %s; validating by parse", remote)
        t_parse = time.time()
        scores = _validate_cached_sscore(fetched, person_ids, logger)
        if scores is None:
            continue
        logger.info(
            "[prs] adopted legacy %s rows=%d cols=%d parse=%s; writing sidecar",
            remote,
            scores.shape[0],
            scores.shape[1],
            _format_seconds(time.time() - t_parse),
        )
        _write_workspace_sidecar(cache, fetched, remote, fingerprint)
        return scores
    return None


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
    bed: Optional[Path] = None,
    score_files: Optional[Sequence[Path]] = None,
) -> pd.DataFrame:
    person_ids = _cohort_person_ids(cohort_csv)
    bed = _resolve_microarray_bed() if bed is None else Path(bed)
    panel_dir = Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME
    out_dir = Path(DEFAULT_CACHE_DIR) / GNOMON_OUT_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if score_files is None:
        logger.info("[prs] resolving PGS scoring panel into %s", panel_dir)
        t_panel = time.time()
        score_files = [
            Path(p)
            for p in download_panel(
                panel_dir,
                progress=lambda msg: logger.info("[prs] %s", msg),
            )
        ]
        logger.info(
            "[prs] PGS scoring panel ready: %d files in %s (dir=%s)",
            len(score_files),
            _format_seconds(time.time() - t_panel),
            panel_dir,
        )
    else:
        score_files = [Path(p) for p in score_files]
        logger.info(
            "[prs] using %d caller-provided PGS score files", len(score_files)
        )
    logger.info(
        "[prs] computing gnomon score fingerprint for bed=%s with %d score files",
        bed,
        len(score_files),
    )
    t_fp = time.time()
    fingerprint = _gnomon_score_fingerprint(bed, score_files, logger=logger)
    sscore_name = "gnomon-scores-" + _short_hash(fingerprint) + ".sscore"
    local_sscore = out_dir / sscore_name
    remote_sscore = f"{GNOMON_OUT_DIRNAME}/{sscore_name}"
    logger.info(
        "[prs] fingerprint ready in %s -> %s (target=%s)",
        _format_seconds(time.time() - t_fp),
        _short_hash(fingerprint),
        remote_sscore,
    )
    scores = _restore_gnomon_scores(
        cache, out_dir, fingerprint, person_ids, logger
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
                _write_workspace_sidecar(
                    cache, local_sscore, remote_sscore, fingerprint
                )

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
    bed = _resolve_microarray_bed()
    panel_dir = Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[prs] resolving PGS scoring panel into %s", panel_dir)
    t_panel = time.time()
    score_files = [
        Path(p)
        for p in download_panel(
            panel_dir,
            progress=lambda msg: logger.info("[prs] %s", msg),
        )
    ]
    logger.info(
        "[prs] PGS scoring panel ready: %d files in %s (dir=%s)",
        len(score_files),
        _format_seconds(time.time() - t_panel),
        panel_dir,
    )
    logger.info(
        "[prs] computing PRS panel fingerprint (bed=%s, score_files=%d, n_person=%d)",
        bed,
        len(score_files),
        len(person_ids),
    )
    t_fp = time.time()
    fingerprint = _prs_panel_fingerprint(
        bed, score_files, person_ids, logger=logger
    )
    filename = "aou-prs-panel-" + _short_hash(fingerprint) + ".csv.gz"
    path = Path(DEFAULT_CACHE_DIR) / PRS_PANEL_CACHE_DIRNAME / filename
    remote_name = f"{PRS_PANEL_CACHE_DIRNAME}/{filename}"
    logger.info(
        "[prs] PRS panel target: %s (fingerprint elapsed=%s)",
        remote_name,
        _format_seconds(time.time() - t_fp),
    )
    panel_local_dir = Path(DEFAULT_CACHE_DIR) / PRS_PANEL_CACHE_DIRNAME
    restored = _restore_prs_panel(
        cache, panel_local_dir, fingerprint, person_ids, logger
    )
    if restored is not None:
        panel, panel_path = restored
        return panel, str(panel_path)
    logger.info(
        "[prs] no usable cache; building PRS panel from gnomon scoring -> %s",
        path,
    )
    _build_prs_panel(
        cohort_csv,
        path,
        logger,
        cache=cache,
        bed=bed,
        score_files=score_files,
    )
    logger.info(
        "[prs] uploading freshly built PRS panel %s -> %s",
        path,
        remote_name,
    )
    t_store = time.time()
    cache.store(path, remote_name)
    _write_workspace_sidecar(cache, path, remote_name, fingerprint)
    logger.info(
        "[prs] workspace store finished in %s",
        _format_seconds(time.time() - t_store),
    )
    # Re-read from disk so the in-memory frame matches what _restore_prs_panel
    # would return on a later run. CSV float roundtrip flips a few low bits,
    # which would otherwise propagate into _genscore_key and invalidate the
    # crosscoder fit checkpoint for any subsequent run.
    panel = _read_prs_panel(path, logger=logger)
    return panel, str(path)


def _has_survival_outcome(data: SyntheticDataset) -> bool:
    return bool(np.any(data.time > 0.0) and np.any(data.event == 1))


def _survival_outcome_key(person_ids: Sequence[str]) -> str:
    # Raw OMOP survival-event extraction depends only on the AoU CDR
    # snapshot and the cohort person ids.  Downstream config bumps must
    # not invalidate this cache, so PIPELINE_CONFIG_VERSION is omitted.
    return _short_hash(
        {
            "cdr": _workspace_cdr(),
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
        cdr=_workspace_cdr(),
        cache_dir=Path(DEFAULT_CACHE_DIR) / "omop",
        workspace_bucket=cache.bucket,
        workspace_prefix=f"{WORKSPACE_CACHE_PREFIX}/omop",
        fetch_conditions=False,
        fetch_drugs=False,
        fetch_measurements=False,
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
        meta_json=np.array(json.dumps(_json_sanitise(outcome.meta), allow_nan=False)),
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
    node_types = list(data.node_types)
    node_types[target_idx] = "survival"
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
            node_types=tuple(node_types),
            ground_truth_adj=ground_truth_adj,
        ),
        pid[keep],
    )


def _ehr_panel_key(person_ids: Sequence[str]) -> str:
    # Raw EHR ETL depends only on the AoU CDR snapshot, cohort person
    # ids, and the EHR-specific catalog/threshold constants below.
    # Pipeline-version bumps that touch downstream stages must not
    # invalidate this cache, so PIPELINE_CONFIG_VERSION is omitted.
    return _short_hash(
        {
            "cdr": _workspace_cdr(),
            "person_ids": [str(p) for p in person_ids],
            "fetch_drugs": EHR_FETCH_DRUGS,
            "fetch_measurements": EHR_FETCH_MEASUREMENTS,
            "condition_concept_ids": (
                None
                if EHR_CONDITION_CONCEPT_IDS is None
                else list(EHR_CONDITION_CONCEPT_IDS)
            ),
            "t2d_condition_blacklist_ids": list(T2D_EHR_CONDITION_BLACKLIST_IDS),
            "t2d_treatment_drug_prefixes": list(T2D_TREATMENT_DRUG_PREFIXES),
            "measurement_catalog": [
                {"lab": lab, "loinc_codes": list(codes)}
                for lab, codes in CURATED_OMOP_MEASUREMENT_CATALOG
            ],
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
        cdr=_workspace_cdr(),
        cache_dir=Path(DEFAULT_CACHE_DIR) / "omop",
        workspace_bucket=cache.bucket,
        workspace_prefix=f"{WORKSPACE_CACHE_PREFIX}/omop",
        condition_concept_ids=EHR_CONDITION_CONCEPT_IDS,
        fetch_drugs=EHR_FETCH_DRUGS,
        fetch_measurements=EHR_FETCH_MEASUREMENTS,
        lookback_days=EHR_LOOKBACK_DAYS,
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
        measurement_summary=frames.get("measurement_summary"),
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
        name = f"{stem[: 48 - len(suffix)]}{suffix}"
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
            drop_reasons.append(
                (
                    str(col),
                    f"missing={1.0 - present_rate:.3f} > max={PRS_MAX_MISSING:.3f}",
                )
            )
            continue
        if n_finite < min_required:
            drop_reasons.append(
                (
                    str(col),
                    f"finite_rows={n_finite} < min_required={min_required}",
                )
            )
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
            raise ValueError(
                f"selected PRS column became constant after row filter: {col}"
            )
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


def _render_genscore_plots_async(
    model_bundle: dict[str, Any] | None,
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
    person_ids: np.ndarray,
    genscore_meta: dict[str, Any],
    logger: logging.Logger,
    event: Optional[np.ndarray] = None,
    target_name: str = "T2D",
) -> None:
    """Render crosscoder figures in a background thread.

    Spawns immediately after the genscore phase so the plots land on
    disk while the rest of the pipeline (MrDAG, DAGSLAM, MCMC, GAM) is
    still running. Failures are logged and swallowed: plotting must not
    take down the pipeline.
    """
    if model_bundle is None:
        logger.info(
            "[genscore-plots] no cached crosscoder weights; "
            "skipping plots this run (rerun after deleting genscore-*.npz "
            "to populate model cache)"
        )
        return

    import threading

    def _worker() -> None:
        try:
            from .genscore.crosscoder import TopKCrosscoder
            from .genscore.integrate import (
                FeatureSelection,
                align_panels_by_iid,
            )
            from .genscore.plots import (
                GenscorePlotInputs,
                save_all_genscore_plots,
            )

            logger.info("[genscore-plots] worker started")
            _log_rss(logger, "genscore-plots worker start")
            t0 = time.time()
            history = {
                "step": list(genscore_meta.get("loss_history_step", [])),
                "loss_main": list(genscore_meta.get("loss_history_main", [])),
                "loss_val": list(genscore_meta.get("loss_history_val", [])),
                "loss_aux": list(genscore_meta.get("loss_history_aux", [])),
                "loss_cross": list(genscore_meta.get("loss_history_cross", [])),
                "loss_align": list(genscore_meta.get("loss_history_align", [])),
                "loss_contrastive": list(
                    genscore_meta.get("loss_history_contrastive", [])
                ),
                "frac_dead": list(genscore_meta.get("frac_dead_history", [])),
                "avg_l0_batch": list(genscore_meta.get("avg_l0_batch_history", [])),
                "frac_active_batch": list(
                    genscore_meta.get("frac_active_batch_history", [])
                ),
                "ever_active_count": list(
                    genscore_meta.get("ever_active_count_history", [])
                ),
                "r2_genome_val": list(genscore_meta.get("r2_genome_val_history", [])),
                "r2_ehr_gaussian_val": list(
                    genscore_meta.get("r2_ehr_gaussian_val_history", [])
                ),
                "brier_ehr_binary_val": list(
                    genscore_meta.get("brier_ehr_binary_val_history", [])
                ),
                "brier_lift_vs_prior_val": list(
                    genscore_meta.get("brier_lift_vs_prior_val_history", [])
                ),
                "r2_ehr_count_logspace_val": list(
                    genscore_meta.get("r2_ehr_count_logspace_val_history", [])
                ),
                "cross_r2_ehr_gaussian_from_genome_val": list(
                    genscore_meta.get("cross_r2_ehr_gaussian_from_genome_val_history", [])
                ),
                "cross_brier_ehr_binary_from_genome_val": list(
                    genscore_meta.get("cross_brier_ehr_binary_from_genome_val_history", [])
                ),
                "cross_brier_lift_vs_prior_from_genome_val": list(
                    genscore_meta.get("cross_brier_lift_vs_prior_from_genome_val_history", [])
                ),
                "cross_r2_genome_from_ehr_val": list(
                    genscore_meta.get("cross_r2_genome_from_ehr_val_history", [])
                ),
                "negative_control_margin_val": list(
                    genscore_meta.get("negative_control_margin_val_history", [])
                ),
                "frac_shared_decoder": list(
                    genscore_meta.get("frac_shared_decoder_history", [])
                ),
            }

            model = TopKCrosscoder(
                W_e=np.asarray(model_bundle["W_e"], dtype=np.float64),
                b_enc=np.asarray(model_bundle["b_enc"], dtype=np.float64),
                W_d_G=np.asarray(model_bundle["W_d_G"], dtype=np.float64),
                W_d_E=np.asarray(model_bundle["W_d_E"], dtype=np.float64),
                mean_G=np.asarray(model_bundle["mean_G"], dtype=np.float64),
                std_G=np.asarray(model_bundle["std_G"], dtype=np.float64),
                mean_E=np.asarray(model_bundle["mean_E"], dtype=np.float64),
                std_E=np.asarray(model_bundle["std_E"], dtype=np.float64),
                k=int(model_bundle["k"]),
                latent_bank=np.asarray(model_bundle["latent_bank"], dtype=np.int8),
                activation_kind=str(model_bundle["activation_kind"]),
                history=history,
                ehr_feature_kinds=tuple(model_bundle.get("ehr_feature_kinds", ())),
                device=str(model_bundle.get("device", "cached")),
            )

            panels = align_panels_by_iid(person_ids, prs_df, ehr_panel)

            promoted_indices = np.asarray(model_bundle["promoted_indices"], dtype=int)
            promoted_names = tuple(
                str(n) for n in genscore_meta.get("promoted_names", [])
            )
            promoted_genome_share = np.asarray(
                genscore_meta.get("promoted_genome_share", []), dtype=float
            )
            promoted_activation_rate = np.asarray(
                genscore_meta.get("promoted_activation_rate", []), dtype=float
            )
            if (
                promoted_genome_share.size != promoted_indices.size
                or promoted_activation_rate.size != promoted_indices.size
                or len(promoted_names) != promoted_indices.size
            ):
                # Recompute on the fly from the model + panels if metadata is
                # inconsistent (e.g. a hand-edited cache).
                from .genscore.crosscoder import encode, feature_stream_share

                z_full = encode(model, panels.A, panels.B)
                rate_full = (z_full > 0).mean(axis=0)
                rg_full = feature_stream_share(model)
                promoted_activation_rate = rate_full[promoted_indices]
                promoted_genome_share = rg_full[promoted_indices]
                promoted_names = tuple(f"feat_{int(j):04d}" for j in promoted_indices)

            selection = FeatureSelection(
                indices=promoted_indices,
                names=promoted_names,
                genome_share=promoted_genome_share,
                activation_rate=promoted_activation_rate,
                score=np.asarray(genscore_meta.get("promoted_score", []), dtype=float),
                cross_reconstruction_gain=np.asarray(
                    genscore_meta.get("promoted_cross_reconstruction_gain", []),
                    dtype=float,
                ),
                bootstrap_stability=np.asarray(
                    genscore_meta.get("promoted_bootstrap_stability", []),
                    dtype=float,
                ),
                negative_control_margin=np.asarray(
                    genscore_meta.get("promoted_negative_control_margin", []),
                    dtype=float,
                ),
                redundancy_penalty=np.asarray(
                    genscore_meta.get("promoted_redundancy_penalty", []),
                    dtype=float,
                ),
            )

            plots_dir = str(Path(DEFAULT_OUTPUT_DIR) / "plots")
            logger.info(
                "[genscore-plots] rendering into %s "
                "(panels n=%d m_G=%d m_E=%d, model d=%d k=%d)",
                plots_dir,
                int(panels.A.shape[0]),
                int(panels.A.shape[1]),
                int(panels.B.shape[1]),
                int(model.d),
                int(model.k),
            )
            _log_rss(logger, "genscore-plots before save_all")

            from .genscore.labels import (
                collect_omop_concept_ids,
                collect_pgs_ids,
                label_ehr_column,
                label_genome_column,
                resolve_omop_concepts,
                resolve_pgs_metadata,
            )

            raw_prs_cols = tuple(str(c) for c in prs_df.columns)
            raw_ehr_cols = tuple(str(c) for c in ehr_panel.feature_names)
            pgs_meta = resolve_pgs_metadata(
                collect_pgs_ids(raw_prs_cols),
                Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME / "_pgs_metadata.json",
            )
            omop_meta = resolve_omop_concepts(
                collect_omop_concept_ids(raw_ehr_cols),
                Path(DEFAULT_CACHE_DIR) / "omop" / "concept_names.json",
                cdr=_workspace_cdr(),
            )
            prs_columns_labelled = tuple(
                label_genome_column(c, pgs_meta) for c in raw_prs_cols
            )
            ehr_columns_labelled = tuple(
                label_ehr_column(c, omop_meta) for c in raw_ehr_cols
            )

            saved = save_all_genscore_plots(
                plots_dir,
                GenscorePlotInputs(
                    model=model,
                    panels=panels,
                    selection=selection,
                    prs_columns=prs_columns_labelled,
                    ehr_columns=ehr_columns_labelled,
                    ehr_kinds=tuple(str(k) for k in ehr_panel.feature_kinds),
                    history=history,
                    event=event,
                    target_name=target_name,
                ),
            )
            _log_rss(logger, "genscore-plots after save_all")
            logger.info(
                "[genscore-plots] wrote %d figures to %s elapsed=%.1fs",
                len(saved),
                plots_dir,
                time.time() - t0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[genscore-plots] rendering failed: %s", exc, exc_info=True)

    t = threading.Thread(
        target=_worker,
        name="genscore-plots",
        daemon=True,
    )
    t.start()
    _BACKGROUND_PLOT_THREADS.append(t)
    logger.info("[genscore-plots] dispatched in background while downstream phases run")


_BACKGROUND_PLOT_THREADS: list = []


def _join_background_plot_threads(
    logger: logging.Logger, timeout: float = 120.0
) -> None:
    """Block briefly so background plot threads finish before sync/exit."""
    for t in list(_BACKGROUND_PLOT_THREADS):
        if t.is_alive():
            logger.info(
                "[genscore-plots] waiting up to %.0fs for background plot thread %s",
                timeout,
                t.name,
            )
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    "[genscore-plots] thread %s still running after "
                    "timeout; continuing",
                    t.name,
                )
    _BACKGROUND_PLOT_THREADS.clear()


def _load_or_run_genscore_features(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
    logger: logging.Logger,
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any], dict[str, Any] | None]:
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
            model_bundle: dict[str, Any] | None = None
            if "cc_W_e" in z.files:
                model_bundle = {
                    "W_e": np.asarray(z["cc_W_e"]),
                    "b_enc": np.asarray(z["cc_b_enc"]),
                    "W_d_G": np.asarray(z["cc_W_d_G"]),
                    "W_d_E": np.asarray(z["cc_W_d_E"]),
                    "mean_G": np.asarray(z["cc_mean_G"]),
                    "std_G": np.asarray(z["cc_std_G"]),
                    "mean_E": np.asarray(z["cc_mean_E"]),
                    "std_E": np.asarray(z["cc_std_E"]),
                    "k": int(np.asarray(z["cc_k"]).item()),
                    "latent_bank": np.asarray(z["cc_latent_bank"], dtype=np.int8),
                    "activation_kind": str(z["cc_activation_kind"].item()),
                    "ehr_feature_kinds": tuple(
                        json.loads(str(z["cc_ehr_feature_kinds_json"].item()))
                    ),
                    "device": str(z["cc_device"].item()),
                    "promoted_indices": np.asarray(z["cc_promoted_indices"], dtype=int),
                }
            return dataset, z["person_id"].astype(str), meta, model_bundle

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
    checkpoint_filename = f"crosscoder-fit-{key}.npz"
    checkpoint_path = cache.fetch(checkpoint_filename)
    if checkpoint_path.is_file():
        logger.info(
            "[crosscoder] using fit checkpoint %s size=%.1fMiB",
            checkpoint_path,
            _path_size_mib(checkpoint_path),
        )

    checkpoint_uploader = _AsyncUploader(cache, label="crosscoder")

    def _store_crosscoder_checkpoint(local_path: Path, step: int) -> None:
        logger.info(
            "[crosscoder] checkpoint step=%d path=%s size=%.1fMiB (queued)",
            int(step),
            local_path,
            _path_size_mib(local_path),
        )
        checkpoint_uploader.store(local_path, checkpoint_filename)

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
        crosscoder_checkpoint_path=checkpoint_path,
        crosscoder_checkpoint_every=GENSCORE_CROSSCODER_CHECKPOINT_EVERY,
        crosscoder_checkpoint_callback=_store_crosscoder_checkpoint,
    )
    logger.info("[genscore] crosscoder training done; flushing final checkpoint upload")
    _log_rss(logger, "after train_crosscoder")
    checkpoint_uploader.flush()
    _log_rss(logger, "after checkpoint flush")
    logger.info(
        "[genscore] post-training: computing top genome/EHR loadings for %d promoted features",
        len(aug_result.feature_selection.indices),
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
        "crosscoder_device": str(model.device),
        "crosscoder_activation_kind": str(model.activation_kind),
        "crosscoder_shared_bank_size": int(np.sum(model.latent_bank == 0)),
        "crosscoder_genome_private_bank_size": int(np.sum(model.latent_bank == 1)),
        "crosscoder_ehr_private_bank_size": int(np.sum(model.latent_bank == 2)),
        "promoted_indices": sel.indices.tolist(),
        "promoted_names": list(sel.names),
        "promoted_genome_share": sel.genome_share.tolist(),
        "promoted_activation_rate": sel.activation_rate.tolist(),
        "promoted_score": sel.score.tolist(),
        "promoted_cross_reconstruction_gain": sel.cross_reconstruction_gain.tolist(),
        "promoted_bootstrap_stability": sel.bootstrap_stability.tolist(),
        "promoted_negative_control_margin": sel.negative_control_margin.tolist(),
        "promoted_redundancy_penalty": sel.redundancy_penalty.tolist(),
        "base_n": int(aug_result.base_n),
        "augmented_n": int(aug_result.augmented_n),
        "ehr_feature_count": int(ehr_panel.m),
        "promoted_feature_top_genome_loadings": top_genome_loadings,
        "promoted_feature_top_ehr_loadings": top_ehr_loadings,
        "loss_history_step": list(model.history["step"]),
        "loss_history_main": list(model.history["loss_main"]),
        "loss_history_val": list(model.history.get("loss_val", [])),
        "loss_history_aux": list(model.history["loss_aux"]),
        "loss_history_cross": list(model.history.get("loss_cross", [])),
        "loss_history_align": list(model.history.get("loss_align", [])),
        "loss_history_contrastive": list(model.history.get("loss_contrastive", [])),
        "frac_dead_history": list(model.history["frac_dead"]),
        "avg_l0_batch_history": list(model.history.get("avg_l0_batch", [])),
        "frac_active_batch_history": list(model.history.get("frac_active_batch", [])),
        "ever_active_count_history": list(model.history.get("ever_active_count", [])),
        "r2_genome_val_history": list(model.history.get("r2_genome_val", [])),
        "r2_ehr_gaussian_val_history": list(
            model.history.get("r2_ehr_gaussian_val", [])
        ),
        "brier_ehr_binary_val_history": list(
            model.history.get("brier_ehr_binary_val", [])
        ),
        "brier_lift_vs_prior_val_history": list(
            model.history.get("brier_lift_vs_prior_val", [])
        ),
        "r2_ehr_count_logspace_val_history": list(
            model.history.get("r2_ehr_count_logspace_val", [])
        ),
        "cross_r2_ehr_gaussian_from_genome_val_history": list(
            model.history.get("cross_r2_ehr_gaussian_from_genome_val", [])
        ),
        "cross_brier_ehr_binary_from_genome_val_history": list(
            model.history.get("cross_brier_ehr_binary_from_genome_val", [])
        ),
        "cross_brier_lift_vs_prior_from_genome_val_history": list(
            model.history.get("cross_brier_lift_vs_prior_from_genome_val", [])
        ),
        "cross_r2_genome_from_ehr_val_history": list(
            model.history.get("cross_r2_genome_from_ehr_val", [])
        ),
        "negative_control_margin_val_history": list(
            model.history.get("negative_control_margin_val", [])
        ),
        "frac_shared_decoder_history": list(
            model.history.get("frac_shared_decoder", [])
        ),
        "runtime_s": time.time() - t0,
    }
    dataset = aug_result.dataset
    logger.info(
        "[genscore] writing genscore bundle to %s (n=%d, d=%d, ehr_m=%d)",
        path,
        int(dataset.X.shape[0]),
        int(model.d),
        int(ehr_panel.m),
    )
    _log_rss(logger, "before bundle write")
    _atomic_npz(
        path,
        X=dataset.X,
        time=dataset.time,
        event=dataset.event,
        ground_truth_adj=dataset.ground_truth_adj,
        person_id=aug_result.kept_person_id.astype(str),
        columns_json=np.array(json.dumps(list(dataset.columns))),
        node_types_json=np.array(json.dumps(list(dataset.node_types))),
        meta_json=np.array(json.dumps(_json_sanitise(meta), allow_nan=False)),
        cc_W_e=model.W_e.astype(np.float32, copy=False),
        cc_b_enc=model.b_enc.astype(np.float32, copy=False),
        cc_W_d_G=model.W_d_G.astype(np.float32, copy=False),
        cc_W_d_E=model.W_d_E.astype(np.float32, copy=False),
        cc_mean_G=model.mean_G.astype(np.float64, copy=False),
        cc_std_G=model.std_G.astype(np.float64, copy=False),
        cc_mean_E=model.mean_E.astype(np.float64, copy=False),
        cc_std_E=model.std_E.astype(np.float64, copy=False),
        cc_k=np.asarray(int(model.k)),
        cc_latent_bank=model.latent_bank.astype(np.int8, copy=False),
        cc_activation_kind=np.array(str(model.activation_kind)),
        cc_ehr_feature_kinds_json=np.array(json.dumps(list(model.ehr_feature_kinds))),
        cc_device=np.array(str(model.device)),
        cc_promoted_indices=sel.indices.astype(np.int64, copy=False),
    )
    logger.info(
        "[genscore] bundle written %s size=%.1fMiB; uploading to bucket",
        path,
        _path_size_mib(path),
    )
    cache.store(path)
    logger.info("[genscore] bundle uploaded; computing post-training summary metrics")
    _log_rss(logger, "before post-training reconstruction")
    model_bundle: dict[str, Any] | None = {
        "W_e": np.asarray(model.W_e),
        "b_enc": np.asarray(model.b_enc),
        "W_d_G": np.asarray(model.W_d_G),
        "W_d_E": np.asarray(model.W_d_E),
        "mean_G": np.asarray(model.mean_G),
        "std_G": np.asarray(model.std_G),
        "mean_E": np.asarray(model.mean_E),
        "std_E": np.asarray(model.std_E),
        "k": int(model.k),
        "latent_bank": np.asarray(model.latent_bank, dtype=np.int8),
        "activation_kind": str(model.activation_kind),
        "ehr_feature_kinds": tuple(model.ehr_feature_kinds),
        "device": str(model.device),
        "promoted_indices": np.asarray(sel.indices, dtype=int),
    }

    # ---- post-training summary --------------------------------------------
    from .genscore.crosscoder import encode, feature_stream_share
    from .genscore.integrate import align_panels_by_iid

    r_G = feature_stream_share(model)
    logger.info("[genscore] aligning full panels for post-training encode")
    panels_aligned = align_panels_by_iid(person_ids, prs_df, ehr_panel)
    n_full = int(panels_aligned.A.shape[0])
    d_full = int(model.d)
    activation_rate_all, recon_metrics = _post_training_reconstruction_stats(
        model, panels_aligned, logger
    )
    logger.info("[genscore] encode complete; computing reconstruction metrics")
    _log_rss(logger, "after post-training reconstruction")
    dead_mask_all = activation_rate_all == 0.0
    genome_only = (~dead_mask_all) & (r_G > GENSCORE_GENOME_SHARE_MAX)
    ehr_only = (~dead_mask_all) & (r_G < GENSCORE_GENOME_SHARE_MIN)
    cross_modal = (
        (~dead_mask_all)
        & (r_G >= GENSCORE_GENOME_SHARE_MIN)
        & (r_G <= GENSCORE_GENOME_SHARE_MAX)
    )
    r2_g = float(recon_metrics["r2_genome"])
    r2_e_gauss = float(recon_metrics["r2_ehr_gaussian"])
    brier_e_bin = float(recon_metrics["brier_ehr_binary"])
    brier_lift = float(recon_metrics["brier_lift_vs_prior"])
    r2_e_count = float(recon_metrics["r2_ehr_count_logspace"])

    rg_q = np.quantile(r_G, [0.0, 0.25, 0.5, 0.75, 1.0])
    rate_alive = activation_rate_all[~dead_mask_all]
    if rate_alive.size > 0:
        rate_q = np.quantile(rate_alive, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        rate_q = np.zeros(5)

    cross_g_hist = model.history.get("cross_r2_genome_from_ehr_val", [])
    cross_brier_hist = model.history.get(
        "cross_brier_ehr_binary_from_genome_val", []
    )
    cross_brier_lift_hist = model.history.get(
        "cross_brier_lift_vs_prior_from_genome_val", []
    )
    logger.info(
        "[genscore] reconstruction R^2 genome=%.4f | EHR per-kind: "
        "gauss_R^2=%.4f binary_brier=%.4f (lift_vs_prior=%+.4f) "
        "count_R^2_logspace=%.4f",
        r2_g, r2_e_gauss, brier_e_bin, brier_lift, r2_e_count,
    )
    logger.info(
        "[genscore] cross-modal: genome_from_ehr_R^2=%.4f "
        "ehr_binary_from_genome_brier=%.4f (lift=%+.4f)",
        float(cross_g_hist[-1]) if cross_g_hist else float("nan"),
        float(cross_brier_hist[-1]) if cross_brier_hist else float("nan"),
        float(cross_brier_lift_hist[-1]) if cross_brier_lift_hist else float("nan"),
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

    # Resolve human-readable labels for all top loadings before formatting.
    # PGS Catalog metadata + OMOP concept names land in their own JSON caches
    # so we only pay the network/BQ cost once per unique ID across the lifetime
    # of the local checkout.
    from .genscore.labels import (
        extract_pgs_id,
        label_ehr_entry,
        label_genome_entry,
        resolve_omop_concepts,
        resolve_pgs_metadata,
    )

    pgs_ids: set[str] = set()
    omop_ids: set[str] = set()
    for entries in top_genome_loadings.values():
        for entry in entries[:3]:
            pid = extract_pgs_id(str(entry["feature"]))
            if pid is not None:
                pgs_ids.add(pid)
    for entries in top_ehr_loadings.values():
        for entry in entries[:3]:
            feat = str(entry["feature"])
            if ":" in feat:
                prefix, ident = feat.split(":", 1)
                if prefix in ("cond", "drug") and ident.isdigit():
                    omop_ids.add(ident)

    pgs_meta = resolve_pgs_metadata(
        pgs_ids,
        Path(DEFAULT_CACHE_DIR) / PGS_PANEL_DIRNAME / "_pgs_metadata.json",
    )
    omop_meta = resolve_omop_concepts(
        omop_ids,
        Path(DEFAULT_CACHE_DIR) / "omop" / "concept_names.json",
        cdr=_workspace_cdr(),
    )

    for name, idx, share, rate, score, neg_margin in zip(
        sel.names,
        sel.indices.tolist(),
        sel.genome_share.tolist(),
        sel.activation_rate.tolist(),
        sel.score.tolist(),
        sel.negative_control_margin.tolist(),
    ):
        top_g = top_genome_loadings.get(str(name), [])[:3]
        top_e = top_ehr_loadings.get(str(name), [])[:3]
        g_summary = ", ".join(
            label_genome_entry(
                str(entry["feature"]),
                float(entry["weight"]),
                pgs_meta,
            )
            for entry in top_g
        )
        e_summary = ", ".join(
            label_ehr_entry(str(entry["feature"]), float(entry["weight"]), omop_meta)
            for entry in top_e
        )
        logger.info(
            "[genscore]   %s idx=%d score=%.6g r_G=%.3f rate=%.4f "
            "neg_margin=%.4f top_genome=[%s] top_ehr=[%s]",
            name,
            int(idx),
            float(score),
            float(share),
            float(rate),
            float(neg_margin),
            g_summary,
            e_summary,
        )

    logger.info(
        "[genscore] promoted=%d n=%d p=%d",
        len(sel.names),
        dataset.n,
        dataset.p,
    )
    return dataset, aug_result.kept_person_id.astype(str), meta, model_bundle


def _gwas_fingerprint(gwas: Any, source: str) -> dict[str, Any]:
    per_pair = []
    for row in getattr(gwas, "per_pair", []) or []:
        per_pair.append(
            {
                "exposure": str(getattr(row, "exposure", "")),
                "outcome": str(getattr(row, "outcome", "")),
                "exposure_id": str(getattr(row, "exposure_id", "")),
                "outcome_id": str(getattr(row, "outcome_id", "")),
                "beta": float(getattr(row, "beta", float("nan"))),
                "se": float(getattr(row, "se", float("nan"))),
                "n_snps": int(getattr(row, "n_snps", 0)),
                "source": str(getattr(row, "source", "")),
                "note": str(getattr(row, "note", "")),
            }
        )
    per_pair.sort(key=lambda r: (r["exposure"], r["outcome"], r["source"]))
    citations = getattr(gwas, "citations", {}) or {}
    source_metadata = getattr(gwas, "source_metadata", {}) or {}
    return {
        "source": source,
        "exposures": [str(x) for x in getattr(gwas, "exposures")],
        "outcomes": [str(x) for x in getattr(gwas, "outcomes")],
        "betas_sha256": _array_hash(np.asarray(getattr(gwas, "betas"), dtype=float)),
        "ses_sha256": _array_hash(np.asarray(getattr(gwas, "ses"), dtype=float)),
        "ivw_pvals_sha256": _array_hash(
            np.asarray(getattr(gwas, "ivw_pvals"), dtype=float)
        ),
        "n_snps_sha256": _array_hash(
            np.asarray(getattr(gwas, "n_snps"), dtype=np.int64)
        ),
        "circular_pairs": [
            [str(a), str(b)] for a, b in getattr(gwas, "circular_pairs", ())
        ],
        "citations": [
            {"exposure": str(k[0]), "outcome": str(k[1]), "citation": str(v)}
            for k, v in sorted(citations.items())
        ],
        "source_metadata": _json_sanitise(source_metadata),
        "per_pair": per_pair,
    }


def _mrdag_cache_key(gwas: Any, source: str) -> str:
    return _short_hash(
        {
            "version": PIPELINE_CONFIG_VERSION,
            "mrdag": _pipeline_config()["mrdag"],
            "gwas": _gwas_fingerprint(gwas, source),
        }
    )


def _log_gwas_per_pair(logger: logging.Logger, summary: Any, source: str) -> None:
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
    fetched_or_cached = [r for r in rows if r.source in {"fetched", "cache"}]
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


def _load_gwas_for_mrdag(logger: logging.Logger) -> Any:
    """Load real two-sample MR summary statistics for the MrDAG prior.

    Runs the OpenGWAS IVW path against the on-disk ``data/mr_cache/`` and
    only hits the network when (a) a cell is uncached AND (b)
    ``OPENGWAS_JWT`` is set.  Raises if no usable cells are produced --
    there is no fabricated fallback.
    """
    cache_dir = Path(DEFAULT_CACHE_DIR) / "mr_cache"
    summary = load_live_gwas(cache_dir=cache_dir)
    usable = int(np.sum(np.isfinite(summary.betas) & np.isfinite(summary.ses)))
    if usable == 0:
        raise RuntimeError(
            "MrDAG: no usable OpenGWAS IVW cells. Populate the cache under "
            f"{cache_dir} or set OPENGWAS_JWT to refresh from the API."
        )
    _log_gwas_per_pair(logger, summary, "opengwas")
    return summary


def _load_or_run_mrdag(
    cache: WorkspaceCache, logger: logging.Logger
) -> tuple[np.ndarray, dict[str, Any]]:
    gwas = _load_gwas_for_mrdag(logger)
    key = _mrdag_cache_key(gwas, "opengwas")
    filename = f"mrdag-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            logger.info("[mrdag] using cache %s", path)
            return z["pi"], json.loads(str(z["diagnostics_json"].item()))

    logger.info("[mrdag] running MR prior sampler (source=opengwas)")
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
    diagnostics["gwas_source"] = "opengwas"
    diagnostics["gwas_cache_key"] = key
    _atomic_npz(
        path,
        pi=result.pi,
        diagnostics_json=np.array(
            json.dumps(_json_sanitise(diagnostics), allow_nan=False)
        ),
    )
    cache.store(path)
    return result.pi, diagnostics


def _mrdag_prior_for_data(mrdag_pi: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    p = len(columns)
    prior = np.full((p, p), np.nan, dtype=float)
    for i, parent in enumerate(columns):
        if parent not in NODE_INDEX:
            continue
        for j, child in enumerate(columns):
            if child not in NODE_INDEX:
                continue
            pi = mrdag_pi[NODE_INDEX[parent], NODE_INDEX[child]]
            if np.isfinite(pi):
                prior[i, j] = float(pi)
            anchor = PGS_ANCHOR_PRIORS.get((parent, child))
            if anchor is not None:
                prior[i, j] = float(anchor)
            reverse_anchor = PGS_ANCHOR_PRIORS.get((child, parent))
            if reverse_anchor is not None:
                prior[i, j] = 1.0 - float(reverse_anchor)
    np.fill_diagonal(prior, np.nan)
    return prior


def _exogenous_node_names() -> set[str]:
    return {node.name for node in NODES if node.exogenous}


def _is_root_covariate(name: str) -> bool:
    lowered = str(name).lower()
    if str(name) in _exogenous_node_names():
        return True
    return lowered.startswith("pgs_")


def _is_promoted_crosscoder_feature(name: str) -> bool:
    return str(name).lower().startswith("feat_")


def _structural_allowed_edges(
    columns: Sequence[str],
    node_types: Sequence[str],
) -> np.ndarray:
    """Production structural mask for temporally impossible directions."""
    p = len(columns)
    if len(node_types) != p:
        raise ValueError(
            f"node_types has length {len(node_types)} but columns has length {p}"
        )
    allowed = np.ones((p, p), dtype=bool)
    np.fill_diagonal(allowed, False)

    target_like = {
        i
        for i, (name, kind) in enumerate(zip(columns, node_types))
        if str(name) == SURVIVAL_TARGET_COLUMN or str(kind) == "survival"
    }
    for idx in target_like:
        allowed[idx, :] = False
        allowed[idx, idx] = False

    for idx, name in enumerate(columns):
        if _is_root_covariate(str(name)):
            allowed[:, idx] = False
            allowed[idx, idx] = False

    pgs_idx = [
        i for i, name in enumerate(columns) if str(name).lower().startswith("pgs_")
    ]
    feature_idx = [
        i
        for i, name in enumerate(columns)
        if _is_promoted_crosscoder_feature(str(name))
    ]
    for feat in feature_idx:
        for pgs in pgs_idx:
            allowed[feat, pgs] = False
        for target in target_like:
            allowed[target, feat] = False

    # Forbid edges *within* the crosscoder-feature subset.
    # Each feat_k = TopK(W_e[:,k] @ panel + b_enc[k]) is a deterministic
    # projection of the same panel input, so two features j, k are
    # statistically dependent through their shared input columns and
    # through TopK co-activation -- not through any causal mechanism
    # between them. With n=100k+ the BGe likelihood detects that
    # dependence and locks pairwise feat->feat edges into the posterior
    # at probability 1.0, crowding the clinically meaningful edges out
    # of the top of the marginal-edge ranking. The correct generative
    # parents of feat_k are the panel inputs (PRS columns + EHR
    # features), which are already encoded by base->feat being allowed
    # but feat->feat being forbidden. This rule converts that into a
    # hard structural prior.
    for j in feature_idx:
        for k in feature_idx:
            if j != k:
                allowed[j, k] = False

    return allowed


def _log_structural_constraints(
    logger: logging.Logger,
    allowed_edges: np.ndarray,
    columns: Sequence[str],
) -> None:
    p = len(columns)
    forbidden = int(p * (p - 1) - allowed_edges.sum())
    root_cols = [str(c) for c in columns if _is_root_covariate(str(c))]
    sink_cols = [str(c) for c in columns if str(c) == SURVIVAL_TARGET_COLUMN]
    logger.info(
        "[graph] structural constraints allowed_edges=%d/%d forbidden=%d "
        "roots=%s sinks=%s",
        int(allowed_edges.sum()),
        int(p * (p - 1)),
        forbidden,
        root_cols,
        sink_cols,
    )


def _cached_allowed_edges_match(
    z: Any,
    allowed_edges: Optional[np.ndarray],
) -> bool:
    if allowed_edges is None:
        return "allowed_edges" not in z.files or z["allowed_edges"].shape == (0, 0)
    if "allowed_edges" not in z.files:
        return False
    return bool(
        np.array_equal(
            z["allowed_edges"].astype(bool),
            np.asarray(allowed_edges, dtype=bool),
        )
    )


def _adjacency_respects_allowed(
    adjacency: np.ndarray,
    allowed_edges: Optional[np.ndarray],
) -> bool:
    if allowed_edges is None:
        return True
    allowed = np.asarray(allowed_edges, dtype=bool)
    adj = np.asarray(adjacency, dtype=bool)
    return adj.shape == allowed.shape and not bool(np.any(adj & ~allowed))


def _load_or_run_dagslam(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    pi_prior: np.ndarray,
    logger: logging.Logger,
    allowed_edges: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    filename = f"dagslam-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            adjacency = z["adjacency"].astype(int)
            if _cached_allowed_edges_match(
                z, allowed_edges
            ) and _adjacency_respects_allowed(
                adjacency,
                allowed_edges,
            ):
                logger.info("[dagslam] using cache %s", path)
                return {
                    "adjacency": adjacency,
                    "log_score": float(z["log_score"].item()),
                    "n_edges": int(z["n_edges"].item()),
                    "runtime_s": float(z["runtime_s"].item()),
                }
            logger.warning(
                "[dagslam] ignoring cache with stale structural mask %s", path
            )

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
        pi_prior=pi_prior,
        allowed_edges=allowed_edges,
        survival_time=data.time,
        survival_event=data.event,
        survival_horizon=10.0,
    )
    runtime_s = time.time() - t0
    _atomic_npz(
        path,
        adjacency=np.asarray(result.adjacency, dtype=int),
        log_score=np.array(float(result.log_score)),
        n_edges=np.array(int(result.n_edges)),
        runtime_s=np.array(runtime_s),
        pi_prior=np.asarray(pi_prior, dtype=float),
        allowed_edges=(
            np.asarray(allowed_edges, dtype=bool)
            if allowed_edges is not None
            else np.zeros((0, 0), dtype=bool)
        ),
    )
    cache.store(path)
    return {
        "adjacency": np.asarray(result.adjacency, dtype=int),
        "log_score": float(result.log_score),
        "n_edges": int(result.n_edges),
        "runtime_s": runtime_s,
    }


def _log_mcmc_progress_event(
    logger: logging.Logger,
    payload: dict[str, Any],
) -> None:
    event = str(payload.get("event", "progress"))
    chain = int(payload.get("chain", 0))
    n_chains = int(payload.get("n_chains", 0))
    iteration = int(payload.get("iter", 0))
    total_iters = int(payload.get("total_iters", 0))
    if event.startswith("exact_parent_sets"):
        scored = payload.get("parent_sets_scored")
        total = payload.get("parent_sets_total")
        scored_text = (
            f" scored={int(scored)}/{int(total)}"
            if scored is not None and total is not None
            else f" total={int(total)}"
            if total is not None
            else ""
        )
        logger.info(
            "[mcmc] %s chain=%d/%d iter=%d/%d target_node=%s "
            "candidates=%s cap=%s%s parent_elapsed=%s cache_entries=%d",
            event,
            chain,
            n_chains,
            iteration,
            total_iters,
            payload.get("target_node", "?"),
            payload.get("candidate_parents", "?"),
            payload.get("parent_cap", "?"),
            scored_text,
            _format_seconds(float(payload.get("parent_elapsed_s", 0.0))),
            int(payload.get("score_cache_entries", 0)),
        )
        return

    accept_rate = payload.get("accept_rate", {}) or {}
    proposals = payload.get("proposals", {}) or {}
    logger.info(
        "[mcmc] %s chain=%d/%d iter=%d/%d phase=%s kept=%d/%d "
        "edges=%d logpost=%.2f score=%.2f prior=%.2f "
        "moves(add/rem/rev=%d/%d/%d total=%d) props(edge=%d parent=%d block=%d) "
        "acc_overall=%.3f acc_mh=%.3f rate=%.2fiter/s elapsed=%s eta=%s "
        "cache_entries=%d move=%s accepted=%s",
        event,
        chain,
        n_chains,
        iteration,
        total_iters,
        str(payload.get("phase", "?")),
        int(payload.get("kept", 0)),
        int(payload.get("target_samples", 0)),
        int(payload.get("edges", 0)),
        float(payload.get("log_post", float("nan"))),
        float(payload.get("score", float("nan"))),
        float(payload.get("prior", float("nan"))),
        int(payload.get("n_add", 0)),
        int(payload.get("n_remove", 0)),
        int(payload.get("n_reverse", 0)),
        int(payload.get("n_moves", 0)),
        int(proposals.get("edge_gibbs", 0)),
        int(proposals.get("hybrid", 0)),
        int(proposals.get("block_gibbs", 0)),
        float(accept_rate.get("overall", 0.0)),
        float(accept_rate.get("metropolis_hastings", 0.0)),
        float(payload.get("iter_per_s", 0.0)),
        _format_seconds(float(payload.get("elapsed_s", 0.0))),
        _format_seconds(float(payload.get("eta_s", 0.0)))
        if np.isfinite(float(payload.get("eta_s", float("nan"))))
        else "?",
        int(payload.get("score_cache_entries", 0)),
        payload.get("move_type", "-"),
        payload.get("accepted", "-"),
    )


def _load_or_run_mcmc(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    start_adj: np.ndarray,
    pi_prior: np.ndarray,
    logger: logging.Logger,
    allowed_edges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    filename = f"mcmc-{key}.npz"
    path = cache.fetch(filename)
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            samples = z["samples"].astype(int)
            edge_probs = z["edge_probs"]
            if _cached_allowed_edges_match(z, allowed_edges) and (
                samples.shape[0] == 0
                or all(
                    _adjacency_respects_allowed(sample, allowed_edges)
                    for sample in samples
                )
            ):
                logger.info("[mcmc] using cache %s", path)
                return (
                    edge_probs,
                    samples,
                    json.loads(str(z["diagnostics_json"].item())),
                    float(z["runtime_s"].item()),
                )
            logger.warning("[mcmc] ignoring cache with stale structural mask %s", path)

    allowed_count = (
        int(np.asarray(allowed_edges, dtype=bool).sum())
        if allowed_edges is not None
        else int(data.p * (data.p - 1))
    )
    total_iters_per_chain = int(MCMC_BURN_IN + MCMC_SAMPLES * MCMC_THIN)
    logger.info(
        "[mcmc] running posterior structure sampler n=%d p=%d "
        "start_edges=%d allowed_edges=%d samples_per_chain=%d burn_in=%d "
        "thin=%d chains=%d total_iters=%d move_probs(edge_gibbs=%.2f "
        "parent=%.2f single_edge=%.2f) max_parents=%d progress_interval=%d",
        data.n,
        data.p,
        int(np.asarray(start_adj, dtype=int).sum()),
        allowed_count,
        MCMC_SAMPLES,
        MCMC_BURN_IN,
        MCMC_THIN,
        MCMC_CHAINS,
        total_iters_per_chain * MCMC_CHAINS,
        float(MCMC_EDGE_RESAMPLE_PROB),
        float(MCMC_PARENT_RESAMPLE_PROB),
        float(1.0 - MCMC_EDGE_RESAMPLE_PROB - MCMC_PARENT_RESAMPLE_PROB),
        int(DAGSLAM_MAX_PARENTS),
        int(MCMC_PROGRESS_INTERVAL),
    )
    t0 = time.time()
    result = run_structure_mcmc(
        data=data.X,
        node_types=data.node_types,
        start_adj=start_adj,
        pi_prior=pi_prior,
        max_parents=DAGSLAM_MAX_PARENTS,
        n_samples=MCMC_SAMPLES,
        burn_in=MCMC_BURN_IN,
        thin=MCMC_THIN,
        n_chains=MCMC_CHAINS,
        rng=np.random.default_rng(PIPELINE_SEED + 3),
        progress=lambda payload: _log_mcmc_progress_event(logger, payload),
        progress_interval=MCMC_PROGRESS_INTERVAL,
        edge_resample_prob=MCMC_EDGE_RESAMPLE_PROB,
        hybrid_prob=MCMC_PARENT_RESAMPLE_PROB,
        allowed_edges=allowed_edges,
        survival_time=data.time,
        survival_event=data.event,
        survival_horizon=10.0,
    )
    runtime_s = time.time() - t0
    diagnostics = dict(result.diagnostics)
    samples = (
        np.stack(result.samples, axis=0).astype(np.int8)
        if result.samples
        else np.zeros((0, data.p, data.p), dtype=np.int8)
    )
    logger.info(
        "[mcmc] sampler returned samples=%d edge_probs_shape=%s runtime=%s "
        "mean_logpost=%s",
        int(samples.shape[0]),
        tuple(np.asarray(result.edge_probs).shape),
        _format_seconds(runtime_s),
        [
            f"{float(x):.3f}"
            for x in diagnostics.get("mean_log_posterior_per_chain", [])
        ],
    )
    cache_write_started_at = time.time()
    logger.info(
        "[mcmc] writing cache path=%s samples_shape=%s",
        path,
        tuple(samples.shape),
    )
    _atomic_npz(
        path,
        edge_probs=np.asarray(result.edge_probs, dtype=float),
        samples=samples,
        diagnostics_json=np.array(
            json.dumps(_json_sanitise(diagnostics), allow_nan=False)
        ),
        runtime_s=np.array(runtime_s),
        allowed_edges=(
            np.asarray(allowed_edges, dtype=bool)
            if allowed_edges is not None
            else np.zeros((0, 0), dtype=bool)
        ),
    )
    logger.info(
        "[mcmc] cache file written path=%s size=%.1fMiB elapsed=%s",
        path,
        _path_size_mib(path),
        _format_seconds(time.time() - cache_write_started_at),
    )
    cache_store_started_at = time.time()
    logger.info("[mcmc] storing cache artefact %s", path.name)
    cache.store(path)
    logger.info(
        "[mcmc] cache artefact stored elapsed=%s total_elapsed=%s",
        _format_seconds(time.time() - cache_store_started_at),
        _format_seconds(time.time() - t0),
    )
    return np.asarray(result.edge_probs, dtype=float), samples, diagnostics, runtime_s


def _known_edges_for_columns(columns: Sequence[str]) -> tuple[tuple[str, str], ...]:
    available = set(columns)
    edges: list[tuple[str, str]] = []
    for parent, child in CANONICAL_EDGES:
        if parent in available and child in available:
            edges.append((parent, child))
    return tuple(edges)


def _validate_edges(
    edge_probs: np.ndarray,
    columns: Sequence[str],
    rng: np.random.Generator,
    allowed_edges: Optional[np.ndarray] = None,
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
        eligible_edges=allowed_edges,
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


def _median_probability_parent_set(
    samples: np.ndarray,
    target_idx: int,
) -> tuple[int, ...]:
    max_parents = int(globals().get("SURVIVAL_GAM_MAX_PARENTS", 6))
    min_edge_prob = float(globals().get("SURVIVAL_GAM_MIN_EDGE_PROB", 0.5))
    edge_probs = samples.astype(float).mean(axis=0)
    probs = edge_probs[:, target_idx].copy()
    probs[target_idx] = 0.0
    order = np.argsort(-probs)
    chosen = [
        int(i)
        for i in order
        if np.isfinite(probs[i]) and float(probs[i]) >= min_edge_prob
    ]
    return tuple(sorted(chosen[:max_parents]))


def _survival_parent_sets(
    samples: np.ndarray,
    target_idx: int,
    *,
    top_k: int,
) -> tuple[list[tuple[int, ...]], np.ndarray, list[int], list[str]]:
    parent_sets, weights, raw_counts = _posterior_parent_sets(
        samples,
        target_idx,
        top_k=top_k,
    )
    sources = ["sampled"] * len(parent_sets)

    median_set = _median_probability_parent_set(samples, target_idx)
    if median_set:
        if median_set in parent_sets:
            sources[parent_sets.index(median_set)] = "sampled+median_probability"
        else:
            pseudo_count = max(raw_counts) if raw_counts else 1
            parent_sets = [median_set] + parent_sets
            raw_counts = [int(pseudo_count)] + raw_counts
            sources = ["median_probability"] + sources
            if len(parent_sets) > top_k:
                parent_sets = parent_sets[:top_k]
                raw_counts = raw_counts[:top_k]
                sources = sources[:top_k]
            weights = np.asarray(raw_counts, dtype=float)
            weights /= float(weights.sum())

    return parent_sets, weights, raw_counts, sources


def _survival_time_grid(time_arr: np.ndarray) -> np.ndarray:
    t = np.asarray(time_arr, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("survival GAM requires finite positive follow-up times")
    t_min = max(float(np.quantile(t, 0.02)), 1e-3)
    # Stay strictly inside the spline support: gamfit's I-spline basis
    # collapses to identically zero on rows whose log-time hits the
    # rightmost knot (knotvec[-1] == log(max(observed_event_time))),
    # which makes calibration metrics catastrophically bad even when the
    # PIRLS fit converges. Cap at the 99th percentile of follow-up so
    # eval rows land strictly inside the support.
    t_max_raw = float(np.max(t))
    t_max = float(np.quantile(t, 0.99))
    if t_max >= t_max_raw:
        # Tiny cohorts can have q99 == max; nudge below max by a hair.
        t_max = max(t_min, t_max_raw * (1.0 - 1e-3))
    if t_max <= t_min:
        raise ValueError("survival follow-up times have no usable range")
    return np.linspace(t_min, t_max, SURVIVAL_TIME_GRID_POINTS)


def _survival_at_times(
    survival: np.ndarray,
    t_grid: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Interpolate a survival surface to exact evaluation times."""
    S = np.asarray(survival, dtype=float)
    grid = np.asarray(t_grid, dtype=float).ravel()
    eval_times = np.asarray(times, dtype=float).ravel()
    if S.ndim != 2:
        raise ValueError(f"survival surface must be 2-D, got shape {S.shape}")
    if S.shape[1] != grid.size:
        raise ValueError(
            f"survival surface has {S.shape[1]} time columns but grid has {grid.size}"
        )
    if grid.size == 0:
        raise ValueError("survival time grid is empty")
    if not (
        np.all(np.isfinite(S))
        and np.all(np.isfinite(grid))
        and np.all(np.isfinite(eval_times))
    ):
        raise ValueError("survival surface, grid, and eval_times must be finite")
    if np.any((S < -1e-12) | (S > 1.0 + 1e-12)):
        raise ValueError("survival surface must contain probabilities in [0, 1]")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("survival time grid must be strictly increasing")
    S = np.clip(S, 0.0, 1.0)
    out = np.empty((S.shape[0], eval_times.size), dtype=float)
    for i in range(S.shape[0]):
        out[i] = np.interp(eval_times, grid, S[i], left=S[i, 0], right=S[i, -1])
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_vals = np.take_along_axis(values, order, axis=0)
    sorted_w = np.take_along_axis(
        np.broadcast_to(weights[:, None, None], values.shape), order, axis=0
    )
    cdf = np.cumsum(sorted_w, axis=0)
    pick = np.argmax(cdf >= q, axis=0)
    return np.take_along_axis(sorted_vals, pick[None, :, :], axis=0)[0]


def _survival_validation_split(
    event_arr: np.ndarray,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(event_arr, dtype=int).ravel()
    rng = np.random.default_rng(PIPELINE_SEED + 21)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for value in (0, 1):
        idx = np.flatnonzero(e == value)
        rng.shuffle(idx)
        if idx.size == 0:
            continue
        n_test = max(1, int(round(test_fraction * idx.size))) if idx.size > 1 else 0
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train = np.concatenate(train_parts) if train_parts else np.arange(e.size)
    test = np.concatenate(test_parts) if test_parts else np.zeros(0, dtype=int)
    if train.size == 0 or test.size == 0:
        raise ValueError(
            "survival validation split produced an empty train or test set"
        )
    train.sort()
    test.sort()
    return train, test


def _survival_metrics(
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    survival_mean: np.ndarray,
    t_grid: np.ndarray,
) -> dict[str, Any]:
    eval_times = np.asarray(SURVIVAL_EVAL_TIMES, dtype=float)
    S_eval = _survival_at_times(survival_mean, t_grid, eval_times)
    p_event_eval = 1.0 - S_eval
    p_event_10y = (
        1.0
        - _survival_at_times(
            survival_mean,
            t_grid,
            np.asarray([10.0], dtype=float),
        )[:, 0]
    )

    # 10y status is *indeterminate* for subjects censored before 10y: we
    # do not know whether they would have had an event by 10y.  The model
    # surface (gamfit MLE) is censoring-aware, but a binary outcome that
    # buckets censored<10y subjects as ``y=0`` would compare a calibrated
    # prediction against a biased label and inflate ECE / Brier and pull
    # Nagelkerke R^2 negative.  Restrict calibration to subjects whose
    # 10y status is determined: events by 10y, or alive at 10y (T >= 10).
    indeterminate = (time_arr < 10.0) & (event_arr == 0)
    determined = ~indeterminate
    p_det = p_event_10y[determined]
    y_det = ((time_arr[determined] <= 10.0) & (event_arr[determined] == 1)).astype(int)
    if p_det.size == 0:
        raise ValueError("no subjects have determined 10-year status for calibration")
    calibration = calibration_metrics(y_det, p_det, n_bins=10, strategy="quantile")

    td = time_dependent_auc(
        time=time_arr,
        event=event_arr,
        risk_score=p_event_eval,
        eval_times=eval_times,
    )

    # IPCW Brier collapses where ``T_i > tau`` is empty (no controls left
    # at the upper tail of follow-up): the ctrl-term mass is zero and the
    # case-term divides by S_hat -> 0, giving a degenerate near-zero
    # value that contaminates the IBS integral.  Trim to taus with at
    # least 1% of the cohort still at risk (and >= 10 subjects).
    n_total = int(time_arr.size)
    min_at_risk = max(10, int(0.01 * n_total))
    n_at_risk = np.array([int(np.sum(time_arr > tau)) for tau in t_grid], dtype=int)
    keep = n_at_risk >= min_at_risk
    if not np.any(keep):
        keep = np.zeros_like(n_at_risk, dtype=bool)
        keep[0] = True
    brier_grid = t_grid[keep]
    brier_pred = survival_mean[:, keep]
    br = brier_score(
        time=time_arr,
        event=event_arr,
        survival_pred=brier_pred,
        eval_times=brier_grid,
    )

    return {
        "nagelkerke_r2_at_10y": float(nagelkerke_r2(y_det, p_det)),
        "calibration_at_10y": calibration,
        "calibration_at_10y_n_used": int(determined.sum()),
        "calibration_at_10y_n_indeterminate": int(indeterminate.sum()),
        "time_dependent_auc": {
            "times": td["times"],
            "auc": td["auc"],
            "integrated_auc": float(td["integrated_auc"]),
        },
        "brier": br,
        "brier_grid_n_at_risk_min": int(min_at_risk),
    }


def _survival_gam_fit_filename(key: str, parent_set: tuple[int, ...]) -> str:
    return (
        "survival-gam-fit-"
        + _short_hash(
            {
                "key": key,
                "parent_set": list(parent_set),
                "gam": _pipeline_config()["gam"],
            }
        )
        + ".npz"
    )


def _fit_survival_parent_set_worker(
    time_arr: np.ndarray,
    event_arr: np.ndarray,
    X: np.ndarray,
    cols: tuple[str, ...],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    """Worker entry point: fit one survival GAM and return predictions.

    No logger / cache plumbing — those live in the parent process.  gamfit's
    Rust core still prints its REML / PIRLS diagnostics directly to stdout,
    which is inherited from the parent.
    """
    from .gam.survival import fit_survival_gam

    t0 = time.time()
    fit = fit_survival_gam(
        time_arr,
        event_arr,
        X,
        columns=cols,
        n_uncertainty_slices=GAM_N_SAMPLES,
        progress=False,
    )
    mean = fit.predict_survival_mean(X, t_grid)
    variance = fit.predict_survival_variance(X, t_grid)
    runtime_s = time.time() - t0
    diag = fit.uncertainty_summary()
    diag["parent_columns"] = list(cols)
    return mean, variance, diag, runtime_s


def _fit_survival_holdout_worker(
    train_time: np.ndarray,
    train_event: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    cols: tuple[str, ...],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from .gam.survival import fit_survival_gam

    fit = fit_survival_gam(
        train_time,
        train_event,
        X_train,
        columns=cols,
        n_uncertainty_slices=GAM_N_SAMPLES,
        progress=False,
    )
    diag = fit.uncertainty_summary()
    diag["parent_columns"] = list(cols)
    return fit.predict_survival_mean(X_test, t_grid), diag


def _bma_threads_per_worker(n_workers: int) -> int:
    return max(1, cpu_count() // max(1, n_workers))


def _run_survival_parent_fits(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    parent_sets: Sequence[tuple[int, ...]],
    t_grid: np.ndarray,
    logger: logging.Logger,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any], float]]:
    """Cached + parallel BMA training fits across parent_sets.

    Cache hits short-circuit before dispatching to a worker.  The remaining
    uncached fits run in parallel processes via joblib's loky backend; each
    worker pins BLAS threads to a slice of the box so K parallel gamfit
    Rust cores do not all try to grab every core.
    """
    results: list[Optional[tuple[np.ndarray, np.ndarray, dict[str, Any], float]]] = [
        None
    ] * len(parent_sets)

    uncached: list[tuple[int, tuple[int, ...], Path]] = []
    for idx, parent_set in enumerate(parent_sets):
        filename = _survival_gam_fit_filename(key, parent_set)
        path = cache.fetch(filename)
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                logger.info("[gamfit] using fit cache %s", path)
                results[idx] = (
                    z["survival_mean"],
                    z["survival_variance"],
                    json.loads(str(z["diagnostics_json"].item())),
                    float(z["runtime_s"].item()),
                )
        else:
            uncached.append((idx, parent_set, path))

    if uncached:
        n_workers = min(cpu_count(), len(uncached))
        threads_per_worker = _bma_threads_per_worker(n_workers)
        logger.info(
            "[gamfit] dispatching %d parent-set fit(s) across %d worker(s) "
            "with %d BLAS thread(s) each",
            len(uncached),
            n_workers,
            threads_per_worker,
        )
        arg_tuples = []
        for _idx, parent_set, _path in uncached:
            cols = tuple(data.columns[i] for i in parent_set)
            X = (
                data.X[:, list(parent_set)]
                if parent_set
                else np.zeros((data.n, 0))
            )
            arg_tuples.append((data.time, data.event, X, cols, t_grid))
        worker_results = parallel_call(
            _fit_survival_parent_set_worker,
            arg_tuples,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
        )
        for (idx, parent_set, path), worker_result in zip(uncached, worker_results):
            mean, variance, diag, runtime_s = worker_result
            _atomic_npz(
                path,
                survival_mean=mean,
                survival_variance=variance,
                diagnostics_json=np.array(
                    json.dumps(_json_sanitise(diag), allow_nan=False)
                ),
                runtime_s=np.array(runtime_s),
            )
            filename = _survival_gam_fit_filename(key, parent_set)
            cache.store(path, filename)
            logger.info(
                "[gamfit] cached parent fit %s elapsed=%s",
                path,
                _format_seconds(runtime_s),
            )
            results[idx] = worker_result

    out: list[tuple[np.ndarray, np.ndarray, dict[str, Any], float]] = []
    for idx, value in enumerate(results):
        if value is None:
            raise RuntimeError(
                f"survival GAM fit slot {idx} not populated (cache miss without worker result)"
            )
        out.append(value)
    return out


def _run_survival_parent_fits_holdout(
    data: SyntheticDataset,
    parent_sets: Sequence[tuple[int, ...]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    t_grid: np.ndarray,
    logger: logging.Logger,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Parallel held-out validation BMA fits.

    No cache layer (validation fits are recomputed every run).
    """
    if not parent_sets:
        return []
    n_workers = min(cpu_count(), len(parent_sets))
    threads_per_worker = _bma_threads_per_worker(n_workers)
    logger.info(
        "[gamfit-validation] dispatching %d parent-set fit(s) across %d worker(s) "
        "with %d BLAS thread(s) each",
        len(parent_sets),
        n_workers,
        threads_per_worker,
    )
    train_time = data.time[train_idx]
    train_event = data.event[train_idx]
    arg_tuples = []
    for parent_set in parent_sets:
        cols = tuple(data.columns[i] for i in parent_set)
        X_all = (
            data.X[:, list(parent_set)] if parent_set else np.zeros((data.n, 0))
        )
        arg_tuples.append(
            (train_time, train_event, X_all[train_idx], X_all[test_idx], cols, t_grid)
        )
    return parallel_call(
        _fit_survival_holdout_worker,
        arg_tuples,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
    )


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
    parent_sets, weights, parent_set_counts, parent_set_sources = _survival_parent_sets(
        samples,
        target_idx,
        top_k=min(CAUSAL_PATH_TOP_K, 8),
    )
    t_grid = _survival_time_grid(data.time)
    per_model = []
    per_model_variance = []
    fit_summaries = []
    t0 = time.time()
    logger.info("[gam] fitting %d gamfit survival parent-set models", len(parent_sets))
    fit_results = _run_survival_parent_fits(
        cache,
        key,
        data,
        [tuple(ps) for ps in parent_sets],
        t_grid,
        logger,
    )
    for (parent_set, weight, count, source), (mean, variance, diag, _runtime) in zip(
        zip(parent_sets, weights, parent_set_counts, parent_set_sources),
        fit_results,
    ):
        cols = tuple(data.columns[i] for i in parent_set)
        per_model.append(mean)
        per_model_variance.append(variance)
        diag["posterior_parent_set_weight"] = float(weight)
        diag["posterior_parent_set_count"] = int(count)
        diag["parent_set_source"] = str(source)
        diag["parent_columns"] = list(cols)
        fit_summaries.append(diag)

    stack = np.stack(per_model, axis=0)
    stack_variance = np.stack(per_model_variance, axis=0)
    survival_mean = np.einsum("k,knt->nt", weights, stack)
    variance_parametric = np.einsum("k,knt->nt", weights, stack_variance)
    variance_structural = np.einsum(
        "k,knt->nt", weights, (stack - survival_mean[None, :, :]) ** 2
    )
    variance_total = variance_parametric + variance_structural
    survival_sd = np.sqrt(np.clip(variance_total, 0.0, None))
    survival_lower = np.clip(
        survival_mean - SURVIVAL_INTERVAL_Z_90 * survival_sd, 0.0, 1.0
    )
    survival_upper = np.clip(
        survival_mean + SURVIVAL_INTERVAL_Z_90 * survival_sd, 0.0, 1.0
    )
    survival_lower = np.minimum.accumulate(survival_lower, axis=1)
    survival_upper = np.minimum.accumulate(survival_upper, axis=1)
    survival_lower = np.minimum(survival_lower, survival_mean)
    survival_upper = np.maximum(survival_upper, survival_mean)
    train_idx, test_idx = _survival_validation_split(data.event)
    validation_models = []
    validation_summaries = []
    logger.info(
        "[gam] fitting %d held-out validation parent-set models n_train=%d n_test=%d",
        len(parent_sets),
        int(train_idx.size),
        int(test_idx.size),
    )
    holdout_results = _run_survival_parent_fits_holdout(
        data,
        [tuple(ps) for ps in parent_sets],
        train_idx,
        test_idx,
        t_grid,
        logger,
    )
    for (parent_set, weight, count, source), (pred, diag) in zip(
        zip(parent_sets, weights, parent_set_counts, parent_set_sources),
        holdout_results,
    ):
        validation_models.append(pred)
        diag["posterior_parent_set_weight"] = float(weight)
        diag["posterior_parent_set_count"] = int(count)
        diag["parent_set_source"] = str(source)
        validation_summaries.append(diag)
    validation_stack = np.stack(validation_models, axis=0)
    validation_mean = np.einsum("k,knt->nt", weights, validation_stack)
    validation_metrics = _survival_metrics(
        data.time[test_idx],
        data.event[test_idx],
        validation_mean,
        t_grid,
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
                "source": str(source),
            }
            for ps, w, c, source in zip(
                parent_sets,
                weights,
                parent_set_counts,
                parent_set_sources,
            )
        ],
        "fit_summaries": fit_summaries,
        "validation_fit_summaries": validation_summaries,
        "metrics_evaluation": {
            "method": "stratified_holdout_refit",
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "test_fraction": float(test_idx.size / data.n),
        },
        "interval_uncertainty": "structural_parent_set_only",
        "interval_level": 0.90,
        "variance_parametric_mean": float(np.mean(variance_parametric)),
        "variance_structural_mean": float(np.mean(variance_structural)),
        "variance_total_mean": float(np.mean(variance_total)),
        "metrics": validation_metrics,
    }
    parent_columns = tuple(
        dict.fromkeys(
            col for ps in parent_sets for col in (data.columns[i] for i in ps)
        )
    )
    _atomic_npz(
        path,
        t_grid=t_grid,
        survival_mean=survival_mean,
        survival_lower=survival_lower,
        survival_upper=survival_upper,
        variance_structural=variance_structural,
        parent_columns_json=np.array(json.dumps(list(parent_columns))),
        diagnostics_json=np.array(
            json.dumps(_json_sanitise(diagnostics), allow_nan=False)
        ),
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


def _load_data_stage(
    cache: WorkspaceCache,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[Path, SyntheticDataset, np.ndarray]:
    """Resolve the cohort CSV and load it as a SyntheticDataset."""
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
    return csv_path, data, person_ids


def _build_prs_panel_stage(
    cache: WorkspaceCache,
    csv_path: Path,
    data: SyntheticDataset,
    person_ids: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any], pd.DataFrame, str, int]:
    """Load the cohort PRS panel and augment the dataset with PRS columns."""
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
    return data, kept_person_ids, prs_meta, prs_df, prs_path, person_id_rows_after_prs


def _ensure_survival_outcome_stage(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    kept_person_ids: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any]]:
    """Apply OMOP survival outcome on top of the cohort if absent in the CSV."""
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
    return data, kept_person_ids, survival_outcome_meta


def _build_ehr_panel_stage(
    cache: WorkspaceCache,
    kept_person_ids: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> EhrPanel:
    with _phase(logger, "ehr"):
        t0 = time.time()
        ehr_panel = _load_or_build_ehr_panel(cache, kept_person_ids, logger)
        timings["ehr"] = time.time() - t0
        logger.info("[ehr] complete elapsed=%s", _format_seconds(timings["ehr"]))
    return ehr_panel


def _run_genscore_stage(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    kept_person_ids: np.ndarray,
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[SyntheticDataset, np.ndarray, dict[str, Any], Any]:
    """Augment the dataset with promoted crosscoder features."""
    with _phase(logger, "genscore"):
        t0 = time.time()
        (
            data,
            kept_person_ids,
            genscore_meta,
            genscore_model_bundle,
        ) = _load_or_run_genscore_features(
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
    # Render genscore figures immediately, before any downstream phase
    # runs. The plots land in <DEFAULT_OUTPUT_DIR>/plots/ so the user
    # can inspect them while MrDAG / DAGSLAM / MCMC / GAM are still
    # running.
    _render_genscore_plots_async(
        genscore_model_bundle,
        prs_df,
        ehr_panel,
        kept_person_ids,
        genscore_meta,
        logger,
        event=np.asarray(data.event, dtype=int) if data.event is not None else None,
        target_name=NODES[NODE_INDEX[SURVIVAL_TARGET_COLUMN]].label,
    )
    return data, kept_person_ids, genscore_meta, genscore_model_bundle


def _build_mrdag_prior_stage(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run MrDAG and reshape the (p, p) prior to data-column order."""
    _log_rss(logger, "before mrdag")
    with _phase(logger, "mrdag"):
        t0 = time.time()
        mrdag_pi, mrdag_diagnostics = _load_or_run_mrdag(cache, logger)
        mrdag_prior = _mrdag_prior_for_data(mrdag_pi, data.columns)
        timings["mrdag"] = time.time() - t0
        _log_mrdag_diagnostics(logger, mrdag_pi, mrdag_diagnostics)
        logger.info("[mrdag] complete elapsed=%s", _format_seconds(timings["mrdag"]))
    return mrdag_pi, mrdag_prior, mrdag_diagnostics


def _run_dagslam_stage(
    cache: WorkspaceCache,
    data: SyntheticDataset,
    mrdag_prior: np.ndarray,
    allowed_edges: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[str, dict[str, Any]]:
    """Run the DAGSLAM hill-climber to produce a warm-start adjacency."""
    _log_rss(logger, "before dagslam")
    with _phase(logger, "dagslam"):
        t0 = time.time()
        key = _run_key(data, mrdag_prior, allowed_edges)
        dagslam = _load_or_run_dagslam(
            cache,
            key,
            data,
            mrdag_prior,
            logger,
            allowed_edges=allowed_edges,
        )
        timings["dagslam"] = time.time() - t0
        logger.info(
            "[dagslam] log_score=%.3f n_edges=%d",
            float(dagslam["log_score"]),
            int(dagslam["n_edges"]),
        )
        _log_dagslam_top_edges(logger, dagslam["adjacency"], data.columns)
    return key, dagslam


def _run_structure_mcmc_stage(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    dagslam: dict[str, Any],
    mrdag_prior: np.ndarray,
    allowed_edges: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the structure MCMC sampler over DAGs."""
    _log_rss(logger, "before mcmc")
    with _phase(logger, "mcmc"):
        t0 = time.time()
        edge_probs, mcmc_samples, mcmc_diagnostics, mcmc_runtime = _load_or_run_mcmc(
            cache,
            key,
            data,
            np.asarray(dagslam["adjacency"], dtype=int),
            mrdag_prior,
            logger,
            allowed_edges=allowed_edges,
        )
        del mcmc_runtime
        timings["mcmc"] = time.time() - t0
        logger.info(
            "[mcmc] accept_overall=%.3f accept_mh=%.3f max_rhat_directed=%.3f min_ess=%.1f",
            float(mcmc_diagnostics["accept_rate"].get("overall", float("nan"))),
            float(
                mcmc_diagnostics["accept_rate"].get("metropolis_hastings", float("nan"))
            ),
            float(mcmc_diagnostics.get("max_rhat_directed", float("nan"))),
            float(mcmc_diagnostics.get("min_ess", float("nan"))),
        )
        _log_mcmc_diagnostics(
            logger,
            edge_probs,
            mcmc_diagnostics,
            data.columns,
            float(THRESHOLD_DEFAULT),
            allowed_edges=allowed_edges,
        )
    return edge_probs, mcmc_samples, mcmc_diagnostics


def _fit_survival_gam_stage(
    cache: WorkspaceCache,
    key: str,
    data: SyntheticDataset,
    mcmc_samples: np.ndarray,
    timings: dict[str, float],
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Fit per-parent-set survival GAMs and BMA-blend them."""
    with _phase(logger, "gam"):
        t0 = time.time()
        (
            survival_time_grid,
            survival_mean,
            survival_lower,
            survival_upper,
            survival_parent_columns,
            survival_diagnostics,
            survival_runtime,
        ) = _load_or_run_survival_gam(cache, key, data, mcmc_samples, logger)
        del survival_runtime
        timings["gam"] = time.time() - t0
        logger.info(
            "[gam] parent_columns=%s n_parent_sets=%d",
            list(survival_parent_columns),
            int(survival_diagnostics.get("n_parent_sets", 0)),
        )
        _log_survival_metrics(logger, survival_diagnostics)
    return (
        survival_time_grid,
        survival_mean,
        survival_lower,
        survival_upper,
        survival_parent_columns,
        survival_diagnostics,
    )


def _log_pipeline_summary(
    data: SyntheticDataset,
    timings: dict[str, float],
    genscore_meta: dict[str, Any],
    mrdag_diagnostics: dict[str, Any],
    dagslam: dict[str, Any],
    mcmc_diagnostics: dict[str, Any],
    edge_probs: np.ndarray,
    mcmc_samples: np.ndarray,
    thresholded: np.ndarray,
    allowed_edges: np.ndarray,
    edge_validation: dict[str, Any],
    survival_diagnostics: dict[str, Any],
    pipeline_started_at: float,
    logger: logging.Logger,
) -> None:
    """Render the multi-line summary block at the end of run_pipeline."""
    surv_metrics = survival_diagnostics.get("metrics") or {}
    td = surv_metrics.get("time_dependent_auc") or {}
    br = surv_metrics.get("brier") or {}
    cal = surv_metrics.get("calibration_at_10y") or {}

    def _flag(condition: bool, label: str) -> str:
        return f"  [WARN: {label}]" if condition else ""

    bar = "=" * 78
    logger.info("[summary] " + bar)
    logger.info("[summary] === pipeline metrics ===")
    logger.info("[summary] " + bar)

    logger.info(
        "[summary] cohort:    n=%d   p=%d   events=%d   event_rate=%.4f",
        data.n,
        data.p,
        int(data.event.sum()),
        float(np.mean(data.event)),
    )
    promoted_names = list(genscore_meta.get("promoted_names", []))
    logger.info(
        "[summary] genscore:  base_n=%d  augmented_n=%d  promoted=%d %s",
        int(genscore_meta.get("base_n", 0)),
        int(genscore_meta.get("augmented_n", 0)),
        len(promoted_names),
        f"({', '.join(promoted_names[:8])}{'...' if len(promoted_names) > 8 else ''})"
        if promoted_names else "",
    )

    sorted_timings = sorted(timings.items(), key=lambda kv: -float(kv[1]))
    logger.info(
        "[summary] timings:   %s",
        "  ".join(f"{k}={_format_seconds(float(v))}" for k, v in sorted_timings),
    )

    mrdag_max_rhat = float(mrdag_diagnostics.get("max_rhat_on_allowed", float("nan")))
    logger.info(
        "[summary] mrdag:     max_rhat=%.3f  candidate_edges=%d%s",
        mrdag_max_rhat,
        int(mrdag_diagnostics.get("n_candidate_edges", 0)),
        _flag(np.isfinite(mrdag_max_rhat) and mrdag_max_rhat > 1.10,
              f"max_rhat={mrdag_max_rhat:.2f} > 1.10 (poor mixing)"),
    )

    logger.info(
        "[summary] dagslam:   log_score=%.3f  n_edges=%d",
        float(dagslam["log_score"]),
        int(dagslam["n_edges"]),
    )

    n_scored = int(allowed_edges.sum())
    mcmc_acc = float(mcmc_diagnostics["accept_rate"].get("overall", float("nan")))
    mcmc_acc_mh = float(
        mcmc_diagnostics["accept_rate"].get("metropolis_hastings", float("nan"))
    )
    mcmc_max_rhat = float(mcmc_diagnostics.get("max_rhat_directed", float("nan")))
    mcmc_min_ess = float(mcmc_diagnostics.get("min_ess", float("nan")))
    logger.info(
        "[summary] mcmc:      accept=%.3f  accept_mh=%.3f  max_rhat=%s  min_ess=%.1f  "
        "edges_above_%.2f=%d/%d%s%s%s",
        mcmc_acc,
        mcmc_acc_mh,
        ("inf" if not np.isfinite(mcmc_max_rhat) else f"{mcmc_max_rhat:.3f}"),
        mcmc_min_ess,
        float(THRESHOLD_DEFAULT),
        int(np.sum(thresholded)),
        int(n_scored),
        _flag(np.isfinite(mcmc_acc_mh) and mcmc_acc_mh < 0.10,
              f"accept_mh={mcmc_acc_mh:.3f} < 0.10 (proposals largely rejected)"),
        _flag(not np.isfinite(mcmc_max_rhat),
              "max_rhat=inf (chain divergence on at least one edge)"),
        _flag(np.isfinite(mcmc_min_ess) and mcmc_min_ess < 100.0,
              f"min_ess={mcmc_min_ess:.0f} < 100 (effective-sample-size deficit)"),
    )

    if mcmc_samples.shape[0] > 0:
        ep = np.asarray(edge_probs, dtype=float)
        cols = list(data.columns)
        flat = []
        for i in range(ep.shape[0]):
            for j in range(ep.shape[1]):
                if i != j and np.isfinite(ep[i, j]) and ep[i, j] > 0:
                    flat.append((ep[i, j], i, j))
        flat.sort(reverse=True)
        if flat:
            top_edges = ", ".join(
                f"{cols[i]}->{cols[j]}={p:.2f}" for p, i, j in flat[:10]
            )
            logger.info("[summary] top_edges: %s", top_edges)

    if "auroc" in edge_validation:
        auroc = float(edge_validation["auroc"])
        auprc = float(edge_validation["auprc"])
        logger.info(
            "[summary] validation: AUROC=%.3f  AUPRC=%.3f%s",
            auroc, auprc,
            _flag(np.isfinite(auroc) and auroc < 0.55,
                  f"AUROC={auroc:.3f} ~ chance (no discrimination of known edges)"),
        )
    known = edge_validation.get("known_edges") if isinstance(edge_validation, dict) else None
    if isinstance(known, dict) and known:
        items = sorted(known.items(), key=lambda kv: -float(kv[1]))
        logger.info(
            "[summary] known_edges: %s",
            ", ".join(f"{k}={float(v):.2f}" for k, v in items[:8]),
        )

    if surv_metrics:
        ibs = float(br.get("ibs", float("nan")))
        ece = float(cal.get("ece", float("nan")))
        nag = float(surv_metrics.get("nagelkerke_r2_at_10y", float("nan")))
        iauc = float(td.get("integrated_auc", float("nan")))
        ev_rate = float(np.mean(data.event))
        null_brier = ev_rate * (1.0 - ev_rate)
        logger.info(
            "[summary] survival:  integrated_AUC=%.3f  integrated_Brier=%.4f  "
            "ECE_10y=%.4f  Nagelkerke_R2_10y=%.3g%s%s%s%s",
            iauc, ibs, ece, nag,
            _flag(np.isfinite(ibs) and ibs > null_brier,
                  f"IBS={ibs:.3f} > null Brier {null_brier:.3f} "
                  "(model worse than baseline rate)"),
            _flag(np.isfinite(ece) and ece > 0.10,
                  f"ECE_10y={ece:.3f} > 0.10 (poor calibration)"),
            _flag(np.isfinite(nag) and (nag < -1.0 or nag > 1.0),
                  f"Nagelkerke_R2={nag:.3g} out of [-1, 1] (degenerate fit)"),
            _flag(np.isfinite(iauc) and iauc < 0.55,
                  f"integrated_AUC={iauc:.3f} ~ chance"),
        )
        per_time = td.get("per_time")
        if isinstance(per_time, dict) and per_time:
            entries = sorted(
                ((float(t), float(v)) for t, v in per_time.items() if v is not None),
                key=lambda tv: tv[0],
            )
            if entries:
                logger.info(
                    "[summary] survival_per_time_AUC: %s",
                    ", ".join(f"t={t:g}: {v:.3f}" for t, v in entries),
                )

    out_dir = Path(DEFAULT_OUTPUT_DIR)
    if out_dir.is_dir():
        plots_dir = out_dir / "plots"
        n_plots = sum(1 for _ in plots_dir.glob("*.png")) if plots_dir.is_dir() else 0
        logger.info(
            "[summary] artefacts: %s  (plots=%d)",
            str(out_dir.resolve()),
            n_plots,
        )

    logger.info("[summary] " + bar)
    logger.info(
        "[pipeline] complete elapsed=%s n=%d p=%d",
        _format_seconds(time.time() - pipeline_started_at),
        data.n,
        data.p,
    )


def run_pipeline() -> PipelineResult:
    """Run the single production path."""
    logger = logging.getLogger("causal_pred.pipeline")
    if not logger.handlers:
        logger = _setup_logger(PIPELINE_VERBOSE)
    _install_signal_handlers(logger)
    cache = _cache()
    timings: dict[str, float] = {}
    pipeline_started_at = time.time()

    csv_path, data, person_ids = _load_data_stage(cache, timings, logger)
    (
        data,
        kept_person_ids,
        prs_meta,
        prs_df,
        prs_path,
        person_id_rows_after_prs,
    ) = _build_prs_panel_stage(cache, csv_path, data, person_ids, timings, logger)
    data, kept_person_ids, survival_outcome_meta = _ensure_survival_outcome_stage(
        cache, data, kept_person_ids, timings, logger
    )
    ehr_panel = _build_ehr_panel_stage(cache, kept_person_ids, timings, logger)
    data, kept_person_ids, genscore_meta, _genscore_model_bundle = (
        _run_genscore_stage(
            cache, data, kept_person_ids, prs_df, ehr_panel, timings, logger
        )
    )
    mrdag_pi, mrdag_prior, mrdag_diagnostics = _build_mrdag_prior_stage(
        cache, data, timings, logger
    )

    allowed_edges = _structural_allowed_edges(data.columns, data.node_types)
    _log_structural_constraints(logger, allowed_edges, data.columns)

    key, dagslam = _run_dagslam_stage(
        cache, data, mrdag_prior, allowed_edges, timings, logger
    )
    edge_probs, mcmc_samples, mcmc_diagnostics = _run_structure_mcmc_stage(
        cache, key, data, dagslam, mrdag_prior, allowed_edges, timings, logger
    )

    thresholded = _acyclic_threshold_from_edge_probs(
        edge_probs,
        THRESHOLD_DEFAULT,
        allowed_edges=allowed_edges,
    )

    (
        survival_time_grid,
        survival_mean,
        survival_lower,
        survival_upper,
        survival_parent_columns,
        survival_diagnostics,
    ) = _fit_survival_gam_stage(cache, key, data, mcmc_samples, timings, logger)

    if CAUSAL_PATH_TARGET in data.columns and mcmc_samples.shape[0] > 0:
        causal_pathways = _causal_pathway_probabilities(
            edge_probs,
            mcmc_samples,
            data.columns,
        )
    else:
        causal_pathways = []
    _log_causal_pathways(logger, causal_pathways)

    edge_validation = _validate_edges(
        edge_probs,
        data.columns,
        rng=np.random.default_rng(PIPELINE_SEED + 4),
        allowed_edges=allowed_edges,
    )
    validation = {
        "known_edge_recovery": edge_validation,
        "survival": survival_diagnostics.get("metrics", {}),
    }
    _log_validation_metrics(logger, edge_validation)

    _log_pipeline_summary(
        data,
        timings,
        genscore_meta,
        mrdag_diagnostics,
        dagslam,
        mcmc_diagnostics,
        edge_probs,
        mcmc_samples,
        thresholded,
        allowed_edges,
        edge_validation,
        survival_diagnostics,
        pipeline_started_at,
        logger,
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
    allowed_edges: Optional[np.ndarray] = None,
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
    if allowed_edges is None:
        scored = ~np.eye(n, dtype=bool)
    else:
        scored = np.asarray(allowed_edges, dtype=bool).copy()
        if scored.shape != (n, n):
            raise ValueError(
                f"allowed_edges must have shape {(n, n)}, got {scored.shape}"
            )
        np.fill_diagonal(scored, False)
    flat = edge_probs[scored]
    n_above = int(np.sum(flat >= threshold))
    n_certain = int(np.sum(flat >= 0.95))
    n_zero = int(np.sum(flat <= 0.05))
    logger.info(
        "[mcmc] edges_above_%.2f=%d / %d  (>=0.95: %d, <=0.05: %d)",
        float(threshold),
        n_above,
        int(scored.sum()),
        n_certain,
        n_zero,
    )
    logger.info(
        "[mcmc] top_edges: %s",
        _format_top_edges(edge_probs, columns, top_k=15),
    )


def _log_validation_metrics(logger: logging.Logger, validation: dict[str, Any]) -> None:
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
    # ``known_edge_recovery`` returns these as dicts keyed by the float
    # threshold value (not lists), so we look up by key here.
    recovery = validation.get("observed_recovery") or {}
    rec_p = validation.get("recovery_pvalue") or {}
    mcc = validation.get("mcc") or {}
    mcc_p = validation.get("mcc_pvalue") or {}

    def _threshold_value(table: Any, threshold: float) -> float:
        if not isinstance(table, dict):
            return float("nan")
        for key in (threshold, str(threshold), f"{threshold:g}"):
            if key in table:
                return float(table[key])
        return float("nan")

    for thr in thresholds:
        key = float(thr)
        rec = _threshold_value(recovery, key)
        rp = _threshold_value(rec_p, key)
        mc = _threshold_value(mcc, key)
        mp = _threshold_value(mcc_p, key)
        logger.info(
            "[validation] thr=%.2f recovery=%.3f (p=%.3f)  MCC=%.3f (p=%.3f)",
            key,
            rec,
            rp,
            mc,
            mp,
        )
    per_edge = validation.get("per_edge") or {}
    if per_edge:
        rows = sorted(
            per_edge.items(),
            key=lambda kv: (
                -float(kv[1].get("probability", 0.0))
                if not kv[1].get("masked", False)
                else 1.0
            ),
        )
        preview = ", ".join(
            f"{p}->{c}={float(d['probability']):.2f}"
            + ("" if not d.get("masked", False) else "(masked)")
            for (p, c), d in rows
        )
        logger.info("[validation] known_edges: %s", preview)


def _log_survival_metrics(logger: logging.Logger, diagnostics: dict[str, Any]) -> None:
    def _values(obj: Any) -> list[Any]:
        if obj is None:
            return []
        if isinstance(obj, np.ndarray):
            return obj.ravel().tolist()
        if isinstance(obj, (list, tuple)):
            return list(obj)
        return list(obj)

    metrics = diagnostics.get("metrics") or {}
    if not metrics:
        logger.info("[gam] no survival metrics computed")
        return
    td = metrics.get("time_dependent_auc") or {}
    times = _values(td.get("times"))
    aucs = _values(td.get("auc"))
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
        float(br.get("ibs", float("nan"))),
        [f"{float(x):.4f}" for x in _values(br.get("brier"))],
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


def _adj_to_edge_list(adj: np.ndarray, columns: Sequence[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for i, parent in enumerate(columns):
        for j, child in enumerate(columns):
            if int(adj[i, j]) == 1:
                rows.append((parent, child))
    return rows


def _is_reachable_local(adj: np.ndarray, src: int, dst: int) -> bool:
    if src == dst:
        return True
    p = adj.shape[0]
    seen = np.zeros(p, dtype=bool)
    seen[src] = True
    stack = [int(src)]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            v = int(v)
            if v == dst:
                return True
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return False


def _acyclic_threshold_from_edge_probs(
    edge_probs: np.ndarray,
    threshold: float,
    allowed_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a thresholded DAG by greedily adding high-probability edges."""
    probs = np.asarray(edge_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
        raise ValueError(f"edge_probs must be square, got shape {probs.shape}")
    p = probs.shape[0]
    if allowed_edges is None:
        allowed = np.ones((p, p), dtype=bool)
    else:
        allowed = np.asarray(allowed_edges, dtype=bool).copy()
        if allowed.shape != (p, p):
            raise ValueError(
                f"allowed_edges must have shape {(p, p)}, got {allowed.shape}"
            )
    np.fill_diagonal(allowed, False)
    adj = np.zeros((p, p), dtype=int)
    candidates: list[tuple[float, int, int]] = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if not bool(allowed[i, j]):
                continue
            prob = float(probs[i, j])
            if np.isfinite(prob) and prob >= threshold:
                candidates.append((prob, i, j))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    for _prob, i, j in candidates:
        if not _is_reachable_local(adj, j, i):
            adj[i, j] = 1
    return adj


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
    if result.mrdag_pi.shape != (len(NODE_NAMES), len(NODE_NAMES)):
        raise ValueError(
            "mrdag_pi shape must match NODE_NAMES: "
            f"{result.mrdag_pi.shape} != {(len(NODE_NAMES), len(NODE_NAMES))}"
        )
    paths["mrdag_pi_long_csv"] = str(out / "mrdag_pi_long.csv")
    with open(paths["mrdag_pi_long_csv"], "w") as fh:
        fh.write("parent,child,mrdag_pi\n")
        for i, parent in enumerate(NODE_NAMES):
            for j, child in enumerate(NODE_NAMES):
                if i == j:
                    continue
                fh.write(f"{parent},{child},{float(result.mrdag_pi[i, j])}\n")

    paths["mrdag_prior_long_csv"] = str(out / "mrdag_prior_long.csv")
    with open(paths["mrdag_prior_long_csv"], "w") as fh:
        fh.write("parent,child,mrdag_prior\n")
        for parent, child, prob in _edge_prob_long(result.mrdag_prior, columns):
            fh.write(f"{parent},{child},{prob}\n")
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
        json.dump(
            _json_sanitise(result.genscore_features),
            fh,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

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
        "target_node": SURVIVAL_TARGET_COLUMN,
        "survival_parent_columns": list(result.survival_parent_columns),
        "survival_diagnostics": result.survival_diagnostics,
        "causal_pathways": result.causal_pathways,
        "validation": result.validation,
        "timings": result.timings,
        "genscore_features": result.genscore_features,
    }
    paths["summary_json"] = str(out / "summary.json")
    with open(paths["summary_json"], "w") as fh:
        json.dump(
            _json_sanitise(summary), fh, indent=2, sort_keys=True, allow_nan=False
        )

    config = _pipeline_config() if run_config is None else run_config
    paths["run_config_json"] = str(out / "run_config.json")
    with open(paths["run_config_json"], "w") as fh:
        json.dump(_json_sanitise(config), fh, indent=2, sort_keys=True, allow_nan=False)

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
        _join_background_plot_threads(logger)
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
