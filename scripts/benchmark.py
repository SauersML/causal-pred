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


_BENCHMARK_DISPLAY = {
    "kaplan_meier":   ("KM",          "#bdbdbd"),
    "naive_logistic": ("Logistic",    "#9aa0a6"),
    "cox_ph":         ("Cox PH",      "#1a73e8"),
    "mr_ivw":         ("MR-IVW",      "#9aa0a6"),
    "causal_pred":    ("causal-pred", "#d93025"),
}
_SURV_ORDER = ("kaplan_meier", "naive_logistic", "cox_ph", "causal_pred")
_EDGE_ORDER = ("mr_ivw", "causal_pred")


def _bar_chart(baselines: Dict[str, dict], out_path: str) -> None:
    """Emit a 2x2 figure of interesting metrics across baselines:
    integrated time-dependent AUC, Nagelkerke R^2 at 10y, scaled
    integrated Brier (IBS / IBS_KM, lower = better), and causal-edge
    AUROC.  Methods are colored consistently across panels.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[bench] skipping plot: matplotlib unavailable ({exc})")
        return

    def _ok(m):
        return m is not None and m.get("status") not in ("skipped", "failed")

    def _get(name, *keys, default=float("nan")):
        cur = baselines.get(name)
        if not _ok(cur):
            return default
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return float(cur) if cur is not None else default

    ibs_km = _get("kaplan_meier", "ibs")

    def _row(order):
        rows = []
        for key in order:
            disp, color = _BENCHMARK_DISPLAY.get(key, (key, "#4C72B0"))
            rows.append((key, disp, color))
        return rows

    surv_rows = [r for r in _row(_SURV_ORDER) if _ok(baselines.get(r[0]))]
    edge_rows = [r for r in _row(_EDGE_ORDER) if _ok(baselines.get(r[0]))]

    if not surv_rows and not edge_rows:
        print("[bench] no rows to plot")
        return

    def _annotate(ax, bars, vals, fmt="{:.3f}"):
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.015
        for bar, v in zip(bars, vals):
            if v != v:  # NaN
                continue
            y = v + pad if v >= 0 else v - pad
            ax.text(
                bar.get_x() + bar.get_width() / 2, y,
                fmt.format(v),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=11, fontweight="bold",
            )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Panel 1: integrated AUC
    ax = axes[0, 0]
    vals = [_get(k, "time_dep_auc", "integrated_auc") for k, _, _ in surv_rows]
    bars = ax.bar(
        [d for _, d, _ in surv_rows], vals,
        color=[c for _, _, c in surv_rows],
        edgecolor="black", linewidth=0.7, width=0.65,
    )
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Integrated AUC")
    ax.set_title("Discrimination  (higher = better)",
                 fontsize=12, fontweight="bold", loc="left")
    _annotate(ax, bars, vals)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 2: Nagelkerke R^2
    ax = axes[0, 1]
    vals = [_get(k, "nagelkerke_at_10y") for k, _, _ in surv_rows]
    bars = ax.bar(
        [d for _, d, _ in surv_rows], vals,
        color=[c for _, _, c in surv_rows],
        edgecolor="black", linewidth=0.7, width=0.65,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    vmax = max((v for v in vals if v == v), default=0.3)
    vmin = min((v for v in vals if v == v), default=0.0)
    ax.set_ylim(min(vmin - 0.03, -0.05), max(vmax + 0.05, 0.30))
    ax.set_ylabel("Nagelkerke $R^2$ at 10 y")
    ax.set_title("Explained variation  (higher = better)",
                 fontsize=12, fontweight="bold", loc="left")
    _annotate(ax, bars, vals)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 3: scaled Brier (lower better)
    ax = axes[1, 0]
    scaled = [
        (_get(k, "ibs") / ibs_km) if (ibs_km and ibs_km == ibs_km) else float("nan")
        for k, _, _ in surv_rows
    ]
    bars = ax.bar(
        [d for _, d, _ in surv_rows], scaled,
        color=[c for _, _, c in surv_rows],
        edgecolor="black", linewidth=0.7, width=0.65,
    )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    finite = [v for v in scaled if v == v]
    if finite:
        ax.set_ylim(max(0.0, min(finite) - 0.05), max(1.05, max(finite) + 0.05))
    ax.set_ylabel("Scaled Brier (IBS / IBS$_{\\mathrm{KM}}$)")
    ax.set_title("Calibration  (lower = better)",
                 fontsize=12, fontweight="bold", loc="left")
    _annotate(ax, bars, scaled)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 4: causal-edge AUROC
    ax = axes[1, 1]
    if edge_rows:
        vals = [_get(k, "edge_auroc") for k, _, _ in edge_rows]
        bars = ax.bar(
            [d for _, d, _ in edge_rows], vals,
            color=[c for _, _, c in edge_rows],
            edgecolor="black", linewidth=0.7, width=0.5,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Edge AUROC")
        ax.set_title("Causal-edge discovery  (higher = better)",
                     fontsize=12, fontweight="bold", loc="left")
        _annotate(ax, bars, vals)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.axis("off")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
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
