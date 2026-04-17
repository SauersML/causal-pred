"""Tests for :mod:`causal_pred.benchmarks` and ``scripts/benchmark.py``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from causal_pred.data.synthetic import simulate  # noqa: E402
from causal_pred.benchmarks import (  # noqa: E402
    run_kaplan_meier,
    run_cox_ph,
    run_naive_logistic,
    run_mr_ivw,
)


# ---- per-baseline smoke tests -----------------------------------------------

_REQUIRED_SURVIVAL_KEYS = {
    "nagelkerke_at_10y",
    "time_dep_auc",
    "ibs",
    "ibs_km",
    "scaled_brier",
    "runtime_s",
    "model",
}


@pytest.fixture(scope="module")
def n500_data():
    return simulate(n=500, rng=np.random.default_rng(7))


def test_km_runs(n500_data):
    t0 = time.perf_counter()
    out = run_kaplan_meier(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    td = out["time_dep_auc"]
    assert set(td) == {"times", "auc", "integrated_auc"}
    # KM is marginal: td-AUC must be 0.5 (no covariates), IBS == IBS_KM.
    assert abs(out["ibs"] - out["ibs_km"]) < 1e-9
    for a in td["auc"]:
        assert abs(a - 0.5) < 1e-6


def test_cox_runs(n500_data):
    t0 = time.perf_counter()
    out = run_cox_ph(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    # Cox should do materially better than KM on this structured data.
    assert out["scaled_brier"] < 0.95
    # IBS must be finite and in a sensible range.
    assert 0.0 < out["ibs"] < 0.5
    # Time-dependent AUC at t=10y should be > 0.5.
    td = out["time_dep_auc"]
    idx10 = td["times"].index(10.0)
    assert td["auc"][idx10] > 0.55


def test_naive_logistic_runs(n500_data):
    t0 = time.perf_counter()
    out = run_naive_logistic(n500_data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert _REQUIRED_SURVIVAL_KEYS.issubset(out)
    # Nagelkerke R^2 must be nontrivially > 0 on this structured data.
    assert out["nagelkerke_at_10y"] > 0.05
    td = out["time_dep_auc"]
    idx10 = td["times"].index(10.0)
    assert td["auc"][idx10] > 0.55


def test_mr_ivw_runs():
    out = run_mr_ivw()
    assert out["model"] == "mr_ivw"
    # PUBLISHED_MR is literally the ground-truth source for the MR-eligible
    # edges, so the classifier should easily clear AUPRC > 0.5.
    assert out["edge_auprc"] > 0.5
    # AUROC should also be above chance.
    assert out["edge_auroc"] > 0.55
    assert out["n_tests"] > 0
    # The Bonferroni set should be reasonable (not everything, not nothing).
    assert 0 < out["significant_edges"] < out["n_tests"]


# ---- end-to-end CLI smoke ---------------------------------------------------


def test_benchmark_script_smoke(tmp_path):
    outdir = tmp_path / "outputs"
    outdir.mkdir()

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/benchmark.py",
        "--n",
        "300",
        "--mcmc-iter",
        "100",
        "--no-gam",
        "--no-plots",
        "--output-dir",
        str(outdir),
        "--quiet",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - t0

    assert result.returncode == 0, (
        f"benchmark.py failed (exit={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert elapsed < 120.0

    json_path = outdir / "benchmarks.json"
    assert json_path.exists()
    with json_path.open() as fh:
        report = json.load(fh)

    assert set(report) >= {"dataset", "baselines", "run_config", "git_sha", "timestamp"}
    # All five baselines present, even if causal_pred is skipped.
    for name in ("kaplan_meier", "cox_ph", "naive_logistic", "mr_ivw", "causal_pred"):
        assert name in report["baselines"], f"missing baseline {name}"

    # causal_pred was skipped via --no-gam.
    assert report["baselines"]["causal_pred"].get("status") == "skipped"

    # Sanity on survival baselines.
    for name in ("kaplan_meier", "cox_ph", "naive_logistic"):
        row = report["baselines"][name]
        # NaN was converted to None by _json_sanitise; both are OK.
        assert "ibs" in row
