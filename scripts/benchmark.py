"""Benchmark the causal-pred pipeline against standard survival baselines.

Run with

    uv run python scripts/benchmark.py

Writes ``outputs/benchmarks.json`` (the full metrics table) and
``outputs/benchmarks.png`` (a bar chart comparing the numeric metrics
across baselines).

Baselines included:
  * Kaplan-Meier (no covariates)
  * Cox proportional hazards (all covariates)
  * Naive logistic at t = 10 y
  * Naive MR-IVW edge recovery (Bonferroni on cached OpenGWAS IVW)
  * The causal-pred stack (MrDAG -> DAGSLAM -> MCMC -> gamfit survival GAM)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from causal_pred.data.synthetic import simulate  # noqa: E402
from causal_pred.benchmarks import (  # noqa: E402
    DEFAULT_T_GRID,
    DEFAULT_AUC_TIMES,
    run_all_baselines,
)

N = 1000
SEED = 20260416
MCMC_ITER = 500
MCMC_CHAINS = 1
GAM_SAMPLES = 100
OUTPUT_DIR = os.path.join(ROOT, "outputs")


def _json_sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitise(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        if np.isfinite(obj):
            return obj
        return None
    return obj


def _git_sha(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _bar_chart(baselines: Dict[str, dict], out_path: str) -> None:
    """Emit a bar chart comparing the numeric metrics across baselines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[bench] skipping plot: matplotlib unavailable ({exc})")
        return

    rows = []
    for name, m in baselines.items():
        if m.get("status") in ("skipped", "failed"):
            continue
        r2 = m.get("nagelkerke_at_10y", float("nan"))
        td = m.get("time_dep_auc", {})
        ia = td.get("integrated_auc", float("nan"))
        ibs = m.get("ibs", float("nan"))
        rows.append((name, r2, ia, ibs))

    if not rows:
        print("[bench] no rows to plot")
        return

    names = [r[0] for r in rows]
    data = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Nagelkerke R^2 @ 10y", "Integrated td-AUC", "IBS (lower=better)"]
    for k, ax in enumerate(axes):
        ax.bar(range(len(names)), data[:, k])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(titles[k])
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    data = simulate(n=N, rng=rng)

    print(
        f"[bench] dataset n={data.n} p={data.p} event_rate={data.event.mean():.3f}"
    )

    baselines = run_all_baselines(
        data,
        t_grid=DEFAULT_T_GRID,
        auc_times=DEFAULT_AUC_TIMES,
        mcmc_iter=MCMC_ITER,
        mcmc_chains=MCMC_CHAINS,
        gam_samples=GAM_SAMPLES,
        rng=np.random.default_rng(SEED + 1),
    )

    for name, m in baselines.items():
        if m.get("status") in ("skipped", "failed"):
            print(
                f"[bench] {name:<16} {m.get('status'):<8} "
                f"({m.get('reason') or m.get('error', '')})"
            )
            continue
        if "edge_auprc" in m and "nagelkerke_at_10y" not in m:
            print(
                f"[bench] {name:<16} edge_AUROC={m.get('edge_auroc'):.3f} "
                f"edge_AUPRC={m.get('edge_auprc'):.3f}"
            )
        else:
            r2 = m.get("nagelkerke_at_10y", float("nan"))
            ia = m.get("time_dep_auc", {}).get("integrated_auc", float("nan"))
            ibs = m.get("ibs", float("nan"))
            print(
                f"[bench] {name:<16} R2@10y={r2:.3f} td-AUC={ia:.3f} IBS={ibs:.3f}"
            )

    report = {
        "dataset": {
            "n": int(data.n),
            "p": int(data.p),
            "event_rate": float(data.event.mean()),
        },
        "baselines": baselines,
        "run_config": {
            "n": N,
            "seed": SEED,
            "mcmc_iter": MCMC_ITER,
            "mcmc_chains": MCMC_CHAINS,
            "gam_samples": GAM_SAMPLES,
        },
        "git_sha": _git_sha(ROOT),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runtime_s": float(time.perf_counter() - t0),
    }

    json_path = os.path.join(OUTPUT_DIR, "benchmarks.json")
    with open(json_path, "w") as fh:
        json.dump(_json_sanitise(report), fh, indent=2, allow_nan=False)

    _bar_chart(baselines, os.path.join(OUTPUT_DIR, "benchmarks.png"))

    print(f"[bench] wrote {json_path} (total {report['total_runtime_s']:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
