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

    surv_rows = []
    edge_rows = []
    for name, m in baselines.items():
        if m.get("status") in ("skipped", "failed"):
            continue
        r2 = m.get("nagelkerke_at_10y")
        td = m.get("time_dep_auc", {}) or {}
        ia = td.get("integrated_auc")
        ibs = m.get("ibs")
        if any(v is not None and np.isfinite(v) for v in (r2, ia, ibs)):
            surv_rows.append(
                (
                    name,
                    float(r2) if r2 is not None else float("nan"),
                    float(ia) if ia is not None else float("nan"),
                    float(ibs) if ibs is not None else float("nan"),
                )
            )
        auroc = m.get("edge_auroc")
        auprc = m.get("edge_auprc")
        if auroc is not None or auprc is not None:
            edge_rows.append(
                (
                    name,
                    float(auroc) if auroc is not None else float("nan"),
                    float(auprc) if auprc is not None else float("nan"),
                )
            )

    if not surv_rows and not edge_rows:
        print("[bench] no rows to plot")
        return

    n_surv = 3 if surv_rows else 0
    n_edge = 2 if edge_rows else 0
    n_panels = n_surv + n_edge
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4 * n_panels, 4.2), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]
    panel_idx = 0

    if surv_rows:
        s_names = [r[0] for r in surv_rows]
        s_data = np.array([[r[1], r[2], r[3]] for r in surv_rows], dtype=float)
        s_titles = ["Nagelkerke $R^2$ @ 10y", "Integrated td-AUC", "IBS (lower = better)"]
        for k in range(3):
            ax = axes[panel_idx + k]
            ax.bar(range(len(s_names)), s_data[:, k], color="#4C72B0")
            ax.set_xticks(range(len(s_names)))
            ax.set_xticklabels(s_names, rotation=30, ha="right")
            ax.set_title(s_titles[k])
            ax.grid(True, axis="y", alpha=0.3)
        panel_idx += 3

    if edge_rows:
        e_names = [r[0] for r in edge_rows]
        e_data = np.array([[r[1], r[2]] for r in edge_rows], dtype=float)
        e_titles = ["Edge AUROC", "Edge AUPRC"]
        for k in range(2):
            ax = axes[panel_idx + k]
            ax.bar(range(len(e_names)), e_data[:, k], color="#55A868")
            ax.set_xticks(range(len(e_names)))
            ax.set_xticklabels(e_names, rotation=30, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(e_titles[k])
            ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=140)
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
