"""End-to-end causal-prediction pipeline.

    GWAS summary statistics  ->  MrDAG edge priors  pi
                                                    |
    Individual-level data    ->  DAGSLAM start DAG  |
                                                    v
                                        Structure MCMC
                                                    |
                                        posterior parent sets
                                                    |
                                                    v
                                     Distributional survival GAM
                                       (AFT, NUTS / library backend, BMA)
                                                    |
                                                    v
                                        survival curves +
                                      causal-pathway probabilities

The public entry point is :func:`run_pipeline`.  Running
``python -m causal_pred.pipeline`` (or ``uv run python -m causal_pred.pipeline``)
executes an end-to-end demo on synthetic data and writes artefacts into
``outputs/`` at the repo root.

Partial-availability policy
---------------------------
Each downstream stage is wrapped in a guard that catches
``NotImplementedError`` (and, conservatively, any ``Exception``) and
substitutes a documented placeholder so upstream stages can still be
smoke-tested when a teammate hasn't finished their module yet.  The
placeholder values are logged clearly and propagated into ``summary.json``
so the downstream consumer can see what was skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data.nodes import (
    CANONICAL_EDGES,
    NODE_INDEX,
    NODE_NAMES,
    NODE_TYPES,
)
from .data.synthetic import SyntheticDataset, simulate

# Downstream modules may raise NotImplementedError or fail outright;
# we import them lazily inside run_pipeline so an import-time failure
# in one module doesn't break the whole pipeline loader.


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Container for the artefacts produced by :func:`run_pipeline`."""

    data_summary: Dict[str, Any] = field(default_factory=dict)
    mrdag_pi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mrdag_diagnostics: Dict[str, Any] = field(default_factory=dict)
    dagslam_adjacency: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    dagslam_diagnostics: Dict[str, Any] = field(default_factory=dict)
    mcmc_edge_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mcmc_diagnostics: Dict[str, Any] = field(default_factory=dict)
    gam: Dict[str, Any] = field(default_factory=dict)
    survival_mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    t_grid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eval_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    validation: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    stage_status: Dict[str, str] = field(default_factory=dict)
    parent_sets: List[Tuple[Tuple[int, ...], float]] = field(default_factory=list)
    target_node: str = "T2D"


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


def _stage(
    logger: logging.Logger, name: str, timings: Dict[str, float], status: Dict[str, str]
):
    """Context-manager-ish helper to time a stage and record status."""

    class _Ctx:
        def __enter__(self_inner):
            logger.info("[%s] starting", name)
            self_inner.t0 = time.time()
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            dt = time.time() - self_inner.t0
            timings[name] = dt
            if exc is None:
                status.setdefault(name, "ok")
                logger.info("[%s] done (%.2fs)", name, dt)
            else:
                status[name] = f"error:{exc_type.__name__}"
                logger.error("[%s] FAILED after %.2fs: %s", name, dt, exc)
                logger.debug(traceback.format_exc())
            return False  # never swallow inside the CM; caller handles it

    return _Ctx()


def _top_parent_sets(
    edge_probs: np.ndarray, child: int, top_k: int = 3, max_parents: int = 6
) -> List[Tuple[Tuple[int, ...], float]]:
    """Return up to ``top_k`` most-probable parent sets for ``child``.

    Uses the "threshold the marginal probabilities and enumerate the
    resulting unique subsets" heuristic: for each k in 1..max_parents we
    form the set of the k highest-marginal parents; we then score each
    subset by the product of marginal in/out probabilities (treating the
    marginals as an independent-edges Bernoulli surrogate), and return
    the top_k renormalised.
    """
    p = edge_probs.shape[0]
    if p == 0:
        return [(tuple(), 1.0)]
    candidates = [
        i
        for i in range(p)
        if i != child
        and np.isfinite(edge_probs[i, child])
        and edge_probs[i, child] > 0.15
    ]
    if len(candidates) == 0:
        return [(tuple(), 1.0)]
    candidates.sort(key=lambda i: edge_probs[i, child], reverse=True)

    seen: set = set()
    sets: List[Tuple[int, ...]] = []
    for k in range(1, min(max_parents, len(candidates)) + 1):
        s = tuple(sorted(candidates[:k]))
        if s not in seen:
            seen.add(s)
            sets.append(s)

    scored: List[Tuple[Tuple[int, ...], float]] = []
    for s in sets:
        lp = 0.0
        for i in range(p):
            if i == child or not np.isfinite(edge_probs[i, child]):
                continue
            q = float(edge_probs[i, child])
            q = min(max(q, 1e-6), 1.0 - 1e-6)
            lp += np.log(q) if i in s else np.log(1.0 - q)
        scored.append((s, lp))

    lps = np.array([s[1] for s in scored])
    w = np.exp(lps - lps.max())
    w = w / w.sum()
    out = [(scored[i][0], float(w[i])) for i in range(len(scored))]
    out.sort(key=lambda x: -x[1])
    return out[:top_k]


def _to_2d(survival: np.ndarray) -> np.ndarray:
    """Reduce a survival tensor to a deterministic ``(n_new, n_t)`` array.

    Accepts either ``(n_new, n_t)`` (point estimate) or
    ``(n_samples, n_new, n_t)`` (posterior draws) and averages along the
    leading sample axis in the latter case.
    """
    a = np.asarray(survival, dtype=float)
    if a.ndim == 3:
        return a.mean(axis=0)
    if a.ndim == 2:
        return a
    raise ValueError(
        f"survival_mean array has unsupported ndim={a.ndim}; expected 2 or 3"
    )


# ---------------------------------------------------------------------------
# Stage runners (each returns a dict {'ok': bool, ...payload...})
# ---------------------------------------------------------------------------


def _run_data(n: int, seed: int, logger: logging.Logger) -> SyntheticDataset:
    data = simulate(n=n, rng=np.random.default_rng(seed))
    logger.info(
        "[data] n=%d p=%d event_rate=%.3f", data.n, data.p, float(data.event.mean())
    )
    return data


def _run_mrdag(
    data: SyntheticDataset, use_real_gwas: bool, seed: int, logger: logging.Logger
) -> Dict[str, Any]:
    from .mrdag import run_mrdag

    if use_real_gwas:
        from .data.real_gwas import load_real_gwas

        gwas = load_real_gwas()
        logger.info("[mrdag] using literature-based GWAS summary")
    else:
        from .data.gwas import simulate_gwas

        gwas = simulate_gwas(rng=np.random.default_rng(seed + 1))
        logger.info("[mrdag] using simulated GWAS summary")
    result = run_mrdag(gwas, rng=np.random.default_rng(seed + 2))
    pi = np.asarray(result.pi, dtype=float)
    logger.info(
        "[mrdag] pi nonzero-defined cells=%d (of %d)",
        int(np.sum(np.isfinite(pi))),
        pi.size,
    )
    return {"pi": pi, "diagnostics": dict(result.diagnostics)}


def _run_dagslam(
    data: SyntheticDataset, seed: int, logger: logging.Logger
) -> Dict[str, Any]:
    from .dagslam import run_dagslam

    result = run_dagslam(
        data=data.X,
        node_types=data.node_types,
        max_parents=5,
        restarts=3,
        rng=np.random.default_rng(seed + 3),
    )
    logger.info(
        "[dagslam] start log_score=%.2f n_edges=%d",
        float(result.log_score),
        int(result.n_edges),
    )
    return {
        "adjacency": np.asarray(result.adjacency, dtype=np.int64),
        "log_score": float(result.log_score),
        "n_edges": int(result.n_edges),
        "trace": list(result.trace),
    }


def _run_mcmc(
    data: SyntheticDataset,
    start_adj: np.ndarray,
    pi: np.ndarray,
    mcmc_iter: int,
    n_chains: int,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    from .mcmc import run_structure_mcmc

    # We split mcmc_iter evenly between burn-in and post-burn-in.
    burn_in = max(100, mcmc_iter // 2)
    n_samples = max(50, mcmc_iter - burn_in)
    thin = 5
    result = run_structure_mcmc(
        data=data.X,
        node_types=data.node_types,
        start_adj=start_adj,
        pi_prior=pi,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        n_chains=n_chains,
        rng=np.random.default_rng(seed + 4),
    )
    edge_probs = np.asarray(result.edge_probs, dtype=float)
    accept = result.diagnostics.get("accept_rate", {})
    logger.info(
        "[mcmc] n_chains=%d edge_probs.shape=%s accept_overall=%.3f",
        n_chains,
        edge_probs.shape,
        float(accept.get("overall", 0.0)),
    )
    return {
        "edge_probs": edge_probs,
        "diagnostics": dict(result.diagnostics),
    }


def _run_gam(
    data: SyntheticDataset,
    edge_probs: np.ndarray,
    target_node: str,
    t_grid: np.ndarray,
    eval_idx: np.ndarray,
    gam_samples: int,
    gam_warmup: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], List[Tuple[Tuple[int, ...], float]]]:
    from .gam import bma_survival

    target = NODE_INDEX[target_node]
    parent_sets = _top_parent_sets(edge_probs, child=target, top_k=3)
    # Fallback: if edge_probs is empty / degenerate, fall back to the
    # canonical parents of the target node from nodes.CANONICAL_EDGES so
    # the GAM has something meaningful to fit.
    if len(parent_sets) == 1 and parent_sets[0][0] == tuple():
        canonical_parents = tuple(
            sorted(
                {
                    NODE_INDEX[p]
                    for (p, c) in CANONICAL_EDGES
                    if c == target_node and p in NODE_INDEX
                }
            )
        )
        if canonical_parents:
            parent_sets = [(canonical_parents, 1.0)]
            logger.info(
                "[gam] edge_probs uninformative -> using canonical parents %s",
                [NODE_NAMES[i] for i in canonical_parents],
            )

    set_indices = [s for s, _w in parent_sets]
    set_weights = np.array([w for _s, w in parent_sets], dtype=float)
    if set_weights.sum() <= 0:
        set_weights = np.ones_like(set_weights) / len(set_weights)
    else:
        set_weights = set_weights / set_weights.sum()

    gam_bma = bma_survival(
        parent_sets=set_indices,
        weights=set_weights,
        time=data.time,
        event=data.event,
        data_matrix=data.X,
        column_names=NODE_NAMES,
        t_grid=t_grid,
        X_eval=data.X[eval_idx],
        n_samples=gam_samples,
        warmup=gam_warmup,
        rng=np.random.default_rng(seed + 5),
    )
    logger.info(
        "[gam] parent_sets=%s", [tuple(NODE_NAMES[i] for i in s) for s in set_indices]
    )
    return gam_bma, parent_sets


def _run_validation(
    data: SyntheticDataset,
    edge_probs: np.ndarray,
    gam_bma: Dict[str, Any],
    t_grid: np.ndarray,
    eval_idx: np.ndarray,
    target_node: str,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    from .validation import (
        known_edge_recovery,
        nagelkerke_r2,
        calibration_metrics,
        time_dependent_auc,
        brier_score,
    )

    # 1) Known-edge recovery.
    recovery: Dict[str, Any]
    if edge_probs.size == 0 or np.all(edge_probs == 0):
        recovery = {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "note": "edge_probs empty -- recovery skipped",
        }
    else:
        recovery = known_edge_recovery(
            edge_probs=edge_probs,
            ground_truth_edges=CANONICAL_EDGES,
            node_names=NODE_NAMES,
            n_permute=500,
            rng=np.random.default_rng(seed + 6),
        )

    # 2) Point-in-time binary validation at t=10 years.
    t_eval = 10.0
    t_idx = int(np.argmin(np.abs(t_grid - t_eval)))

    # The gam BMA output key is S_bma (per gam/survival.py) -- fall back
    # to 'survival_mean' if a future GAM revision exposes that.
    if "S_bma" in gam_bma:
        surv_mean_2d = np.asarray(gam_bma["S_bma"], dtype=float)
    elif "survival_mean" in gam_bma:
        surv_mean_2d = _to_2d(gam_bma["survival_mean"])
    else:
        raise KeyError(
            "gam_bma output has neither 'S_bma' nor 'survival_mean'; "
            f"keys={list(gam_bma.keys())}"
        )
    # Clip to avoid log(0) in Nagelkerke.
    p_event_by_t = np.clip(1.0 - surv_mean_2d[:, t_idx], 1e-6, 1.0 - 1e-6)
    y_event_by_t = (
        (data.time[eval_idx] <= t_eval) & (data.event[eval_idx] == 1)
    ).astype(int)

    if y_event_by_t.sum() == 0 or y_event_by_t.sum() == y_event_by_t.size:
        r2 = float("nan")
        logger.warning(
            "[validation] y is constant at t=%.1f -- Nagelkerke undefined", t_eval
        )
        calib = {
            "brier": float(np.mean((y_event_by_t - p_event_by_t) ** 2)),
            "ece": float("nan"),
            "mce": float("nan"),
        }
    else:
        r2 = nagelkerke_r2(y_event_by_t, p_event_by_t)
        calib = calibration_metrics(y_event_by_t, p_event_by_t, n_bins=10)

    td_auc = time_dependent_auc(
        time=data.time[eval_idx],
        event=data.event[eval_idx],
        risk_score=p_event_by_t,
        eval_times=np.array([5.0, 10.0, 15.0]),
    )
    brier = brier_score(
        time=data.time[eval_idx],
        event=data.event[eval_idx],
        survival_pred=surv_mean_2d,
        eval_times=t_grid,
    )

    logger.info(
        "[validation] Nagelkerke R^2 (t=10y)=%.3f  AUROC_edges=%s",
        r2,
        recovery.get("auroc", "n/a"),
    )

    return {
        "known_edge_recovery": recovery,
        "nagelkerke_r2_at_10y": float(r2) if np.isfinite(r2) else float("nan"),
        "calibration_at_10y": calib,
        "time_dependent_auc": td_auc,
        "brier": brier,
        "t_eval_years": float(t_eval),
        "survival_mean_shape": list(surv_mean_2d.shape),
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_pipeline(
    n: int = 1500,
    use_real_gwas: bool = True,
    mcmc_iter: int = 2000,
    mcmc_chains: int = 2,
    gam_samples: int = 300,
    gam_warmup: int = 200,
    target_node: str = "T2D",
    seed: int = 20260416,
    verbose: bool = False,
) -> PipelineResult:
    """Run MrDAG -> DAGSLAM -> MCMC -> GAM -> validation end-to-end.

    Each stage tolerates ``NotImplementedError`` (or any other exception)
    raised by its underlying module: a warning is logged, a placeholder
    value is recorded in ``stage_status``, and the pipeline continues so
    earlier / later stages can still be smoke-tested.
    """
    logger = _setup_logger(verbose)
    timings: Dict[str, float] = {}
    status: Dict[str, str] = {}

    result = PipelineResult(target_node=target_node)

    # -- data ---------------------------------------------------------
    with _stage(logger, "data", timings, status):
        data = _run_data(n=n, seed=seed, logger=logger)
    result.data_summary = {
        "n": int(data.n),
        "p": int(data.p),
        "event_rate": float(data.event.mean()),
        "followup_median": float(np.median(data.time)),
    }

    # Downstream defaults / placeholders if stages fail.
    p = data.p
    pi = np.full((p, p), np.nan, dtype=float)
    np.fill_diagonal(pi, 0.0)
    start_adj = np.zeros((p, p), dtype=np.int64)
    edge_probs = np.zeros((p, p), dtype=float)

    # -- mrdag --------------------------------------------------------
    try:
        with _stage(logger, "mrdag", timings, status):
            mr = _run_mrdag(data, use_real_gwas=use_real_gwas, seed=seed, logger=logger)
        pi = mr["pi"]
        result.mrdag_pi = pi
        result.mrdag_diagnostics = mr["diagnostics"]
    except NotImplementedError as e:
        logger.warning("[mrdag] NotImplementedError: %s -- skipping", e)
        status["mrdag"] = "placeholder:NotImplementedError"
        result.mrdag_pi = pi
    except Exception as e:
        logger.warning(
            "[mrdag] failed (%s: %s) -- using uniform pi prior", type(e).__name__, e
        )
        status["mrdag"] = f"placeholder:{type(e).__name__}"
        result.mrdag_pi = pi

    # -- dagslam ------------------------------------------------------
    try:
        with _stage(logger, "dagslam", timings, status):
            ds = _run_dagslam(data, seed=seed, logger=logger)
        start_adj = ds["adjacency"]
        result.dagslam_adjacency = start_adj
        result.dagslam_diagnostics = {
            "log_score": ds["log_score"],
            "n_edges": ds["n_edges"],
            "n_restarts": len(ds["trace"]),
        }
    except NotImplementedError as e:
        logger.warning("[dagslam] NotImplementedError: %s -- using empty DAG", e)
        status["dagslam"] = "placeholder:NotImplementedError"
        result.dagslam_adjacency = start_adj
    except Exception as e:
        logger.warning(
            "[dagslam] failed (%s: %s) -- using empty DAG", type(e).__name__, e
        )
        status["dagslam"] = f"placeholder:{type(e).__name__}"
        result.dagslam_adjacency = start_adj

    # -- mcmc ---------------------------------------------------------
    try:
        with _stage(logger, "mcmc", timings, status):
            mc = _run_mcmc(
                data,
                start_adj=start_adj,
                pi=pi,
                mcmc_iter=mcmc_iter,
                n_chains=mcmc_chains,
                seed=seed,
                logger=logger,
            )
        edge_probs = mc["edge_probs"]
        result.mcmc_edge_probs = edge_probs
        result.mcmc_diagnostics = mc["diagnostics"]
    except NotImplementedError as e:
        logger.warning("[mcmc] NotImplementedError: %s -- edge_probs=0", e)
        status["mcmc"] = "placeholder:NotImplementedError"
        result.mcmc_edge_probs = edge_probs
    except Exception as e:
        logger.warning("[mcmc] failed (%s: %s) -- edge_probs=0", type(e).__name__, e)
        status["mcmc"] = f"placeholder:{type(e).__name__}"
        result.mcmc_edge_probs = edge_probs

    # -- gam ----------------------------------------------------------
    t_grid = np.linspace(0.5, 20.0, 40)
    result.t_grid = t_grid
    eval_idx = np.random.default_rng(seed + 999).choice(
        data.n,
        size=min(200, data.n),
        replace=False,
    )
    result.eval_idx = eval_idx

    gam_bma: Dict[str, Any] = {}
    parent_sets: List[Tuple[Tuple[int, ...], float]] = []
    try:
        with _stage(logger, "gam", timings, status):
            gam_bma, parent_sets = _run_gam(
                data,
                edge_probs=edge_probs,
                target_node=target_node,
                t_grid=t_grid,
                eval_idx=eval_idx,
                gam_samples=gam_samples,
                gam_warmup=gam_warmup,
                seed=seed,
                logger=logger,
            )
        result.gam = gam_bma
        result.parent_sets = parent_sets
        # S_bma is the (n_eval, n_t) averaged survival; normalise to a
        # 2D array for downstream validation and serialisation.
        if "S_bma" in gam_bma:
            result.survival_mean = np.asarray(gam_bma["S_bma"], dtype=float)
        elif "survival_mean" in gam_bma:
            result.survival_mean = _to_2d(gam_bma["survival_mean"])
        else:
            result.survival_mean = np.zeros((eval_idx.size, t_grid.size))
    except NotImplementedError as e:
        logger.warning("[gam] NotImplementedError: %s -- survival=KM fallback", e)
        status["gam"] = "placeholder:NotImplementedError"
        result.survival_mean = _km_fallback(data, t_grid, eval_idx)
    except Exception as e:
        logger.warning(
            "[gam] failed (%s: %s) -- survival=KM fallback", type(e).__name__, e
        )
        status["gam"] = f"placeholder:{type(e).__name__}"
        result.survival_mean = _km_fallback(data, t_grid, eval_idx)
        gam_bma = {"S_bma": result.survival_mean, "t_grid": t_grid}
        result.gam = gam_bma

    # -- validation ---------------------------------------------------
    try:
        with _stage(logger, "validation", timings, status):
            validation = _run_validation(
                data,
                edge_probs=edge_probs,
                gam_bma=gam_bma or {"S_bma": result.survival_mean},
                t_grid=t_grid,
                eval_idx=eval_idx,
                target_node=target_node,
                seed=seed,
                logger=logger,
            )
        result.validation = validation
    except NotImplementedError as e:
        logger.warning("[validation] NotImplementedError: %s", e)
        status["validation"] = "placeholder:NotImplementedError"
        result.validation = {}
    except Exception as e:
        logger.warning("[validation] failed (%s: %s)", type(e).__name__, e)
        status["validation"] = f"placeholder:{type(e).__name__}"
        result.validation = {"error": str(e)}

    result.timings = timings
    result.stage_status = status
    logger.info("[pipeline] done in %.2fs (stages: %s)", sum(timings.values()), status)
    return result


def _km_fallback(
    data: SyntheticDataset, t_grid: np.ndarray, eval_idx: np.ndarray
) -> np.ndarray:
    """Kaplan-Meier marginal survival replicated across eval rows.

    Used when the GAM stage fails / is unimplemented so that downstream
    validation can still run end-to-end.
    """
    from .validation.metrics import _km_estimator, _km_eval  # type: ignore

    km_t, km_S = _km_estimator(data.time, data.event)
    S_marg = _km_eval(km_t, km_S, np.asarray(t_grid, dtype=float))
    return np.broadcast_to(S_marg, (eval_idx.size, t_grid.size)).copy()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _json_sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        # Only emit small arrays into JSON.  Large ones are written as .npy.
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
    # Fallback: stringify unknown types.
    return str(obj)


def save_result(
    result: PipelineResult,
    outdir: str = DEFAULT_OUTPUT_DIR,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Serialise the pipeline artefacts into ``outdir``.

    Returns a dict of the file paths that were written.
    """
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # --- numpy artefacts ---
    paths["mrdag_pi"] = os.path.join(outdir, "mrdag_pi.npy")
    np.save(paths["mrdag_pi"], result.mrdag_pi)
    paths["mcmc_edge_probs"] = os.path.join(outdir, "mcmc_edge_probs.npy")
    np.save(paths["mcmc_edge_probs"], result.mcmc_edge_probs)
    paths["survival_mean"] = os.path.join(outdir, "survival_mean.npy")
    np.save(paths["survival_mean"], result.survival_mean)
    paths["t_grid"] = os.path.join(outdir, "t_grid.npy")
    np.save(paths["t_grid"], result.t_grid)
    paths["dagslam_adjacency"] = os.path.join(outdir, "dagslam_adjacency.npy")
    np.save(paths["dagslam_adjacency"], result.dagslam_adjacency)

    # --- summary.json ---
    parent_sets_json = [
        {
            "parents": list(s),
            "parent_names": [NODE_NAMES[i] for i in s],
            "weight": float(w),
        }
        for (s, w) in result.parent_sets
    ]
    summary = {
        "target_node": result.target_node,
        "node_names": list(NODE_NAMES),
        "node_types": list(NODE_TYPES),
        "data_summary": result.data_summary,
        "timings": result.timings,
        "stage_status": result.stage_status,
        "mrdag_diagnostics": result.mrdag_diagnostics,
        "dagslam_diagnostics": result.dagslam_diagnostics,
        "mcmc_diagnostics": result.mcmc_diagnostics,
        "validation": result.validation,
        "parent_sets": parent_sets_json,
    }
    paths["summary_json"] = os.path.join(outdir, "summary.json")
    with open(paths["summary_json"], "w") as fh:
        json.dump(_json_sanitise(summary), fh, indent=2, sort_keys=True)

    # --- run_config.json ---
    if run_config is None:
        run_config = {}
    paths["run_config_json"] = os.path.join(outdir, "run_config.json")
    with open(paths["run_config_json"], "w") as fh:
        json.dump(_json_sanitise(run_config), fh, indent=2, sort_keys=True)

    # --- run.log (one-line summary) ---
    paths["run_log"] = os.path.join(outdir, "run.log")
    with open(paths["run_log"], "w") as fh:
        total = sum(result.timings.values())
        fh.write(
            f"stages={result.stage_status} total_s={total:.2f} "
            f"n={result.data_summary.get('n', '?')} "
            f"target={result.target_node} "
            f"r2_10y={result.validation.get('nagelkerke_r2_at_10y', 'nan')}\n"
        )

    # --- plots ---
    try:
        plot_paths = _write_plots(result, plots_dir)
        paths.update(plot_paths)
    except Exception as e:
        # Plots are nice-to-have; don't break the run if matplotlib misfires.
        logging.getLogger("causal_pred.pipeline").warning(
            "plot writing failed: %s: %s",
            type(e).__name__,
            e,
        )

    return paths


# ---------------------------------------------------------------------------
# Plots (inline matplotlib; upgradable to causal_pred.plots when task #8 lands)
# ---------------------------------------------------------------------------


def _save_plots_via_library(
    result: PipelineResult, plots_dir: str, plots_module
) -> Dict[str, str]:
    """Render every applicable plot via :func:`causal_pred.plots.save_all`.

    Builds the call-site arguments from ``result`` (heatmap inputs,
    reliability data, time-dependent AUC, Brier curve, a per-individual
    survival fan, the DAG adjacency, and the causal-pathway Sankey),
    then invokes ``save_all`` to write both PNG and PDF variants.  We
    then flatten its ``{name: (png, pdf)}`` mapping into ``{name: png}``
    plus ``"<name>_pdf"`` aliases so callers get simple string paths.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # --- inputs for save_all --------------------------------------
    edge_probs = result.mcmc_edge_probs if result.mcmc_edge_probs.size else None
    mrdag_pi = result.mrdag_pi if result.mrdag_pi.size else None
    adjacency = result.dagslam_adjacency if result.dagslam_adjacency.size else None

    validation = result.validation or {}
    calib = validation.get("calibration_at_10y") or {}
    rel = calib.get("reliability") if isinstance(calib, dict) else None

    y_true = p_pred = None
    if isinstance(rel, dict):
        # reliability_diagram wants per-sample y/p; we only have binned
        # means.  Expand each bin into (count,) synthetic points so the
        # plot still looks reasonable.  The caller can always swap this
        # out for the raw y/p if the validation stage exposes them.
        mean_p = np.asarray(rel.get("mean_predicted", []), dtype=float)
        mean_y = np.asarray(rel.get("mean_observed", []), dtype=float)
        counts = np.asarray(rel.get("bin_counts", []), dtype=int)
        if mean_p.size and counts.size and counts.sum() > 0:
            y_true = np.repeat(mean_y, counts)
            p_pred = np.repeat(mean_p, counts)

    tda = validation.get("time_dependent_auc") or {}
    eval_times = np.asarray(
        tda.get("times", tda.get("eval_times", [])),
        dtype=float,
    )
    auc_values = np.asarray(tda.get("auc", []), dtype=float)
    auc_se = tda.get("auc_se")
    if auc_se is not None:
        auc_se = np.asarray(auc_se, dtype=float)

    brier = validation.get("brier") or {}
    brier_t = np.asarray(
        brier.get("brier", brier.get("brier_t", [])),
        dtype=float,
    )
    brier_baseline_vals = brier.get("brier_km", brier.get("brier_baseline"))
    brier_baseline = (
        np.asarray(brier_baseline_vals, dtype=float)
        if brier_baseline_vals is not None
        else None
    )
    brier_times = np.asarray(
        brier.get("times", result.t_grid),
        dtype=float,
    )
    t_grid_for_brier = brier_times if brier_t.size == brier_times.size else None

    # survival_samples for fan chart: K per-model means stacked for
    # one individual yield our available 'posterior' draws.
    per_model = (
        result.gam.get("per_model_mean") if isinstance(result.gam, dict) else None
    )
    survival_samples = None
    individual_id = None
    if (
        isinstance(per_model, np.ndarray)
        and per_model.ndim == 3
        and per_model.shape[0] >= 2
        and per_model.shape[1] >= 1
    ):
        survival_samples = per_model[:, 0, :]
        if result.eval_idx.size:
            individual_id = f"sample {int(result.eval_idx[0])}"

    node_types = list(NODE_TYPES)

    # --- primary heatmap via save_all ----------------------------
    primary_edge = edge_probs if edge_probs is not None else mrdag_pi

    kwargs: Dict[str, Any] = dict(
        outputs_dir=plots_dir,
        edge_probs=primary_edge,
        ground_truth_edges=list(CANONICAL_EDGES),
        node_names=list(NODE_NAMES),
        y_true=y_true,
        p_pred=p_pred,
        eval_times=eval_times if eval_times.size else None,
        auc_values=auc_values if auc_values.size else None,
        auc_se=auc_se,
        t_grid=t_grid_for_brier,
        brier_t=brier_t if brier_t.size else None,
        brier_baseline=brier_baseline,
        survival_samples=survival_samples,
        individual_id=individual_id,
        target_node=result.target_node,
        adjacency=adjacency,
        node_types=node_types,
    )
    saved: Dict[str, Tuple[str, str]] = plots_module.save_all(**kwargs)

    paths: Dict[str, str] = {}
    for name, (png, pdf) in saved.items():
        paths[name] = png
        paths[f"{name}_pdf"] = pdf

    # save_all writes the primary heatmap as "edge_heatmap"; also
    # write a dedicated MrDAG pi heatmap when both are available so
    # the report can show them side by side.
    if edge_probs is not None and mrdag_pi is not None:
        try:
            fig = plots_module.edge_probability_heatmap(
                edge_probs=mrdag_pi,
                node_names=list(NODE_NAMES),
                title="MrDAG edge-inclusion prior (pi)",
            )
            mrdag_path = os.path.join(plots_dir, "mrdag_heatmap.png")
            fig.savefig(mrdag_path, dpi=144)
            import matplotlib.pyplot as _plt

            _plt.close(fig)
            paths["mrdag_heatmap"] = mrdag_path
        except Exception:
            pass

    return paths


def _write_plots(result: PipelineResult, plots_dir: str) -> Dict[str, str]:
    """Write the standard artefact plots.

    Primary path: delegate to :func:`causal_pred.plots.save_all`, which
    owns the full suite of figures (heatmap, PR/ROC, reliability, AUC,
    Brier, survival fan, DAG graph, Sankey).  If that import fails for
    any reason we fall back to a minimal inline matplotlib layout so
    the pipeline still produces artefacts.
    """
    paths: Dict[str, str] = {}
    os.makedirs(plots_dir, exist_ok=True)

    try:
        from . import plots as _plots  # type: ignore[attr-defined]
    except Exception:
        _plots = None

    if _plots is not None:
        paths.update(_save_plots_via_library(result, plots_dir, _plots))
        # Ensure ``edge_heatmap.png`` exists as an alias even if
        # save_all produced it under its own name already.
        edge_hm = os.path.join(plots_dir, "edge_heatmap.png")
        if os.path.exists(edge_hm):
            paths["edge_heatmap"] = edge_hm
        if paths:
            return paths

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 1) MrDAG pi heatmap ---
    paths["mrdag_heatmap"] = os.path.join(plots_dir, "mrdag_heatmap.png")
    pi = result.mrdag_pi
    if pi.size > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        # NaN -> gray via masked array.
        data = np.where(np.isnan(pi), 0.0, pi)
        im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title("MrDAG edge-inclusion prior (pi)")
        ax.set_xlabel("child")
        ax.set_ylabel("parent")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(pi.shape[1]))
        ax.set_yticks(range(pi.shape[0]))
        ax.set_xticklabels(NODE_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(NODE_NAMES, fontsize=6)
        fig.tight_layout()
        fig.savefig(paths["mrdag_heatmap"], dpi=120)
        plt.close(fig)

    # --- 2) MCMC edge-inclusion heatmap ---
    paths["mcmc_heatmap"] = os.path.join(plots_dir, "mcmc_heatmap.png")
    P = result.mcmc_edge_probs
    if P.size > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(P, vmin=0.0, vmax=1.0, cmap="Oranges")
        ax.set_title("MCMC posterior edge-inclusion probability")
        ax.set_xlabel("child")
        ax.set_ylabel("parent")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(P.shape[1]))
        ax.set_yticks(range(P.shape[0]))
        ax.set_xticklabels(NODE_NAMES, rotation=90, fontsize=6)
        ax.set_yticklabels(NODE_NAMES, fontsize=6)
        fig.tight_layout()
        fig.savefig(paths["mcmc_heatmap"], dpi=120)
        plt.close(fig)

    # --- 3) Calibration curve ---
    calib = (result.validation or {}).get("calibration_at_10y", {}) or {}
    rel = calib.get("reliability") if isinstance(calib, dict) else None
    paths["calibration"] = os.path.join(plots_dir, "calibration.png")
    if rel is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        mean_p = np.asarray(rel.get("mean_predicted", []))
        mean_y = np.asarray(rel.get("mean_observed", []))
        counts = np.asarray(rel.get("bin_counts", []))
        mask = counts > 0
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
        ax.plot(mean_p[mask], mean_y[mask], "o-", label="observed", color="C1")
        ax.set_xlabel("mean predicted P(event by 10y)")
        ax.set_ylabel("observed rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Calibration at t=10y  (ECE={calib.get('ece', float('nan')):.3f})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths["calibration"], dpi=120)
        plt.close(fig)

    # --- 4) Per-individual survival fans (3 samples) ---
    paths["survival_fans"] = os.path.join(plots_dir, "survival_fans.png")
    S = result.survival_mean
    t = result.t_grid
    if S.size > 0 and t.size > 0:
        n_show = min(3, S.shape[0])
        fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), sharey=True)
        axes = np.atleast_1d(axes)
        for k in range(n_show):
            ax = axes[k]
            ax.plot(t, S[k], color="C0", linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("time (years)")
            ax.set_title(f"sample {int(result.eval_idx[k])}")
        axes[0].set_ylabel("S(t | x)")
        fig.tight_layout()
        fig.savefig(paths["survival_fans"], dpi=120)
        plt.close(fig)

    # --- 5) Back-compat alias: edge_heatmap.png (first real heatmap) ---
    edge_hm = os.path.join(plots_dir, "edge_heatmap.png")
    src = paths.get("mcmc_heatmap") or paths.get("mrdag_heatmap")
    if src and os.path.exists(src):
        import shutil

        shutil.copyfile(src, edge_hm)
        paths["edge_heatmap"] = edge_hm

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal_pred.pipeline",
        description="End-to-end causal-prediction pipeline.",
    )
    parser.add_argument(
        "--n", type=int, default=1500, help="number of simulated individuals"
    )
    parser.add_argument(
        "--mcmc-iter",
        type=int,
        default=2000,
        help="total MCMC iterations (burn + samples*thin)",
    )
    parser.add_argument(
        "--mcmc-chains", type=int, default=2, help="number of independent MCMC chains"
    )
    parser.add_argument(
        "--gam-samples",
        type=int,
        default=300,
        help="posterior-predictive draws per GAM submodel",
    )
    parser.add_argument(
        "--gam-warmup",
        type=int,
        default=200,
        help="GAM warmup iterations (API-compat; currently "
        "a no-op for the library backend)",
    )
    parser.add_argument(
        "--target", default="T2D", help="target survival node for GAM stage"
    )
    parser.add_argument(
        "--use-real-gwas",
        action="store_true",
        default=True,
        help="use literature-based MR summary (default)",
    )
    parser.add_argument(
        "--no-real-gwas",
        dest="use_real_gwas",
        action="store_false",
        help="use simulated GWAS summary instead",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for artefacts"
    )
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="enable DEBUG-level logging",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    run_config = vars(args).copy()
    result = run_pipeline(
        n=args.n,
        use_real_gwas=args.use_real_gwas,
        mcmc_iter=args.mcmc_iter,
        mcmc_chains=args.mcmc_chains,
        gam_samples=args.gam_samples,
        gam_warmup=args.gam_warmup,
        target_node=args.target,
        seed=args.seed,
        verbose=args.verbose,
    )
    save_result(result, outdir=args.output_dir, run_config=run_config)
    print(f"\nArtefacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
