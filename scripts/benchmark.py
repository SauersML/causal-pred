"""Benchmark the causal-pred pipeline against standard survival baselines.

Run with

    uv run python scripts/benchmark.py \
        --n 1000 --mcmc-iter 500 --gam-samples 100 --gam-warmup 50

Writes ``outputs/benchmarks.json`` (the full metrics table) and, unless
``--no-plots`` is passed, ``outputs/benchmarks.png`` (a bar chart
comparing the numeric metrics across baselines).

Baselines included:
  * Kaplan-Meier (no covariates)
  * Cox proportional hazards (all covariates)
  * Naive logistic at t = 10 y
  * Naive MR-IVW edge recovery (Bonferroni on published PUBLISHED_MR)
  * The full causal-pred pipeline (MrDAG -> DAGSLAM -> MCMC -> GAM)

``--no-gam`` / ``--skip-causal-pred`` skip the full-pipeline entry so a
quick sanity benchmark can run without fitting the GAM.
"""

from __future__ import annotations

import argparse
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


def _json_sanitise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
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


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n", type=int, default=1000, help="number of synthetic individuals"
    )
    parser.add_argument(
        "--seed", type=int, default=20260416, help="numpy RNG seed for the simulator"
    )
    parser.add_argument("--mcmc-iter", type=int, default=500)
    parser.add_argument("--mcmc-chains", type=int, default=1)
    parser.add_argument("--gam-samples", type=int, default=100)
    parser.add_argument("--gam-warmup", type=int, default=50)
    parser.add_argument(
        "--no-gam", action="store_true", help="skip the full causal-pred pipeline entry"
    )
    parser.add_argument(
        "--skip-causal-pred", action="store_true", help="alias for --no-gam"
    )
    parser.add_argument(
        "--no-real-gwas",
        action="store_true",
        help="use simulated GWAS summaries instead of published",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    skip_full = bool(args.no_gam or args.skip_causal_pred)

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    data = simulate(n=args.n, rng=rng)

    if not args.quiet:
        print(
            f"[bench] dataset n={data.n} p={data.p} event_rate={data.event.mean():.3f}"
        )

    baselines = run_all_baselines(
        data,
        t_grid=DEFAULT_T_GRID,
        auc_times=DEFAULT_AUC_TIMES,
        run_full_pipeline=not skip_full,
        mcmc_iter=args.mcmc_iter,
        mcmc_chains=args.mcmc_chains,
        gam_samples=args.gam_samples,
        gam_warmup=args.gam_warmup,
        use_real_gwas=not args.no_real_gwas,
        rng=np.random.default_rng(args.seed + 1),
    )

    if not args.quiet:
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
            "n": args.n,
            "seed": args.seed,
            "mcmc_iter": args.mcmc_iter,
            "mcmc_chains": args.mcmc_chains,
            "gam_samples": args.gam_samples,
            "gam_warmup": args.gam_warmup,
            "skip_full_pipeline": skip_full,
            "use_real_gwas": not args.no_real_gwas,
        },
        "git_sha": _git_sha(ROOT),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runtime_s": float(time.perf_counter() - t0),
    }

    json_path = os.path.join(args.output_dir, "benchmarks.json")
    with open(json_path, "w") as fh:
        json.dump(_json_sanitise(report), fh, indent=2)

    if not args.no_plots:
        _bar_chart(baselines, os.path.join(args.output_dir, "benchmarks.png"))

    if not args.quiet:
        print(f"[bench] wrote {json_path} (total {report['total_runtime_s']:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
