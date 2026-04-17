"""Tests for the end-to-end pipeline (``causal_pred.pipeline``).

These tests exercise the full pipeline integration: wiring between
stages, CLI, partial-availability (one stage failing), and
determinism.  The smoke-test stays small (n=500, mcmc_iter=300,
gam_samples=100) so the suite stays under a few minutes on a laptop.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pytest

try:

    _HAS_GAM = True
except Exception:  # pragma: no cover - env without gam
    _HAS_GAM = False

needs_gam = pytest.mark.skipif(not _HAS_GAM, reason="gam library not installed")


# ---------------------------------------------------------------------------
# Required keys that summary.json must carry (contract surface).
# ---------------------------------------------------------------------------

_REQUIRED_SUMMARY_KEYS = {
    "target_node",
    "node_names",
    "node_types",
    "data_summary",
    "timings",
    "stage_status",
    "mrdag_diagnostics",
    "dagslam_diagnostics",
    "mcmc_diagnostics",
    "validation",
    "parent_sets",
}

_REQUIRED_NPY_FILES = [
    "mrdag_pi.npy",
    "mcmc_edge_probs.npy",
    "survival_mean.npy",
    "t_grid.npy",
    "dagslam_adjacency.npy",
]


# ---------------------------------------------------------------------------
# 1. Smoke test: end-to-end run under the laptop budget.
# ---------------------------------------------------------------------------


@needs_gam
def test_smoke_small(tmp_path):
    """``run_pipeline`` finishes under 5 min and writes every artefact."""
    from causal_pred.pipeline import run_pipeline, save_result, PipelineResult

    t0 = time.time()
    result = run_pipeline(
        n=500,
        mcmc_iter=300,
        mcmc_chains=2,
        gam_samples=100,
        gam_warmup=50,
        seed=20260416,
    )
    elapsed = time.time() - t0

    assert isinstance(result, PipelineResult)
    assert elapsed < 300, f"smoke run too slow: {elapsed:.1f}s"

    # Every stage should have a recorded status.
    for stage in ("data", "mrdag", "dagslam", "mcmc", "gam", "validation"):
        assert stage in result.stage_status, stage

    # Artefacts on disk.
    save_result(
        result, outdir=str(tmp_path), run_config={"n": 500, "seed": 20260416}
    )

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    with open(summary_path) as fh:
        summary = json.load(fh)
    missing = _REQUIRED_SUMMARY_KEYS - set(summary.keys())
    assert not missing, f"summary.json missing keys: {missing}"

    for fname in _REQUIRED_NPY_FILES:
        p = tmp_path / fname
        assert p.exists(), f"missing {fname}"
        arr = np.load(p)
        assert arr.size > 0, f"{fname} is empty"

    # Basic shape sanity.
    pi = np.load(tmp_path / "mrdag_pi.npy")
    edge_probs = np.load(tmp_path / "mcmc_edge_probs.npy")
    assert pi.shape == edge_probs.shape
    assert pi.shape[0] == pi.shape[1]

    # Run-config + run.log are written.
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "run.log").exists()


# ---------------------------------------------------------------------------
# 2. CLI --help exits 0.
# ---------------------------------------------------------------------------


def test_cli_help():
    """``python -m causal_pred.pipeline --help`` exits 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "causal_pred.pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "pipeline" in proc.stdout.lower()
    assert "--n" in proc.stdout


# ---------------------------------------------------------------------------
# 3. Partial availability: one stage raises, the rest still run.
# ---------------------------------------------------------------------------


@needs_gam
def test_partial_availability(monkeypatch, tmp_path):
    """If the GAM stage raises ``NotImplementedError`` the pipeline
    continues and ``stage_status['gam']`` reflects the skip.
    """
    import causal_pred.gam as gam_pkg
    from causal_pred.pipeline import run_pipeline, save_result

    def _bad_bma(*_args, **_kwargs):
        raise NotImplementedError("forced skip for testing")

    monkeypatch.setattr(gam_pkg, "bma_survival", _bad_bma)

    result = run_pipeline(
        n=400,
        mcmc_iter=200,
        mcmc_chains=2,
        gam_samples=50,
        gam_warmup=25,
        seed=1,
    )
    assert result.stage_status.get("gam", "").startswith("placeholder"), (
        result.stage_status
    )
    # survival_mean should still be populated (KM fallback, broadcast).
    assert result.survival_mean.shape[0] > 0
    assert result.survival_mean.shape[1] == result.t_grid.shape[0]
    # Upstream stages should still be OK.
    assert result.stage_status.get("mrdag") == "ok"
    assert result.stage_status.get("mcmc") == "ok"

    # summary.json encodes the placeholder.
    save_result(result, outdir=str(tmp_path))
    with open(tmp_path / "summary.json") as fh:
        summary = json.load(fh)
    assert summary["stage_status"]["gam"].startswith("placeholder")


# ---------------------------------------------------------------------------
# 4. Determinism: two runs with the same seed match on at least one scalar.
# ---------------------------------------------------------------------------


@needs_gam
def test_determinism():
    """Two runs with identical seed produce identical Nagelkerke R^2 to 1e-6."""
    from causal_pred.pipeline import run_pipeline

    kwargs = dict(
        n=400,
        mcmc_iter=200,
        mcmc_chains=2,
        gam_samples=50,
        gam_warmup=25,
        seed=42,
    )
    r1 = run_pipeline(**kwargs)
    r2 = run_pipeline(**kwargs)

    v1 = r1.validation.get("nagelkerke_r2_at_10y")
    v2 = r2.validation.get("nagelkerke_r2_at_10y")
    assert v1 is not None and v2 is not None
    # Both NaN or both finite and equal: either counts as deterministic.
    if np.isnan(v1):
        assert np.isnan(v2), "determinism broken: one NaN, one finite"
    else:
        assert abs(v1 - v2) < 1e-6, f"determinism broken: r2_a={v1}, r2_b={v2}"

    # Edge-probability matrices should agree too.
    np.testing.assert_allclose(r1.mcmc_edge_probs, r2.mcmc_edge_probs, atol=1e-10)
