"""Tests for the single-path pipeline (``causal_pred.pipeline``).

The pipeline runs cohort CSV -> DAGSLAM -> structure MCMC -> save. These
tests build a tiny cohort CSV in a temp directory, point the resolver at
it via ``cache_dir``, and verify the artefacts are produced.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _make_tiny_cohort_csv(path):
    rng = np.random.default_rng(0)
    n = 80
    bmi = rng.normal(28.0, 5.0, size=n)
    hba1c = rng.normal(6.0, 1.0, size=n)
    hdl = rng.normal(50.0, 10.0, size=n)
    ldl = rng.normal(100.0, 25.0, size=n)
    trig = rng.normal(120.0, 50.0, size=n)
    # T2D follows a logistic of BMI + HbA1c so MCMC has a real signal.
    logits = -3.0 + 0.06 * (bmi - 25.0) + 0.8 * (hba1c - 5.5)
    p = 1.0 / (1.0 + np.exp(-logits))
    t2d = (rng.random(n) < p).astype(int)
    pd.DataFrame(
        {
            "person_id": [str(1000 + i) for i in range(n)],
            "type2_diabetes": t2d,
            "bmi": bmi,
            "hba1c": hba1c,
            "hdl_cholesterol": hdl,
            "ldl_cholesterol": ldl,
            "triglycerides": np.clip(trig, 30.0, 1000.0),
        }
    ).to_csv(path, index=False)


def _make_tiny_prs_csv(path):
    rng = np.random.default_rng(1)
    n = 80
    pd.DataFrame(
        {
            "person_id": [str(1000 + i) for i in range(n)],
            "PGS_T2D": rng.normal(size=n),
            "PGS_BMI": rng.normal(size=n),
            "PGS_LDL": rng.normal(size=n),
        }
    ).to_csv(path, index=False)


def test_pipeline_runs_end_to_end(tmp_path):
    """``run_pipeline`` finishes and ``save_result`` writes every artefact."""
    _make_tiny_cohort_csv(tmp_path / "t2d_initial_nodes_complete.csv")
    _make_tiny_prs_csv(tmp_path / "aou_prs_panel.csv.gz")

    from causal_pred.pipeline import (
        PipelineResult,
        run_pipeline,
        save_result,
    )

    result = run_pipeline(
        seed=0,
        max_parents=3,
        max_iter=100,
        restarts=1,
        mcmc_samples=80,
        mcmc_burn_in=40,
        mcmc_thin=5,
        mcmc_chains=2,
        cache_dir=str(tmp_path),
        prs_path=str(tmp_path / "aou_prs_panel.csv.gz"),
        n_prs_nodes=2,
    )

    assert isinstance(result, PipelineResult)

    expected_columns = (
        "type2_diabetes",
        "bmi",
        "hba1c",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
        "pgs_t2d",
        "pgs_bmi",
    )
    assert tuple(result.columns) == expected_columns
    p = len(result.columns)
    assert result.dagslam_adjacency.shape == (p, p)
    assert result.mcmc_edge_probs.shape == (p, p)
    assert result.thresholded_adjacency.shape == (p, p)

    out_dir = tmp_path / "outputs"
    save_result(result, outdir=str(out_dir), run_config={"seed": 0})

    for fname in (
        "dagslam_adjacency.npy",
        "mcmc_edge_probs.npy",
        "thresholded_adjacency.npy",
        "greedy_edges.csv",
        "mcmc_thresholded_edges.csv",
        "mcmc_edge_probabilities_long.csv",
        "summary.json",
        "run_config.json",
    ):
        assert (out_dir / fname).exists(), fname

    with open(out_dir / "summary.json") as fh:
        summary = json.load(fh)
    assert summary["columns"] == list(expected_columns)
    assert summary["dagslam_n_edges"] == result.dagslam_n_edges
    assert summary["threshold"] == 0.5
    assert "mcmc_diagnostics" in summary
    assert "accept_rate" in summary["mcmc_diagnostics"]


def test_pipeline_determinism(tmp_path):
    """Two runs with the same seed produce identical edge_probs."""
    _make_tiny_cohort_csv(tmp_path / "t2d_initial_nodes_complete.csv")
    _make_tiny_prs_csv(tmp_path / "aou_prs_panel.csv.gz")

    from causal_pred.pipeline import run_pipeline

    r1 = run_pipeline(
        seed=7,
        max_parents=3,
        max_iter=80,
        restarts=1,
        mcmc_samples=60,
        mcmc_burn_in=30,
        mcmc_thin=5,
        mcmc_chains=2,
        cache_dir=str(tmp_path),
        prs_path=str(tmp_path / "aou_prs_panel.csv.gz"),
        n_prs_nodes=2,
    )
    r2 = run_pipeline(
        seed=7,
        max_parents=3,
        max_iter=80,
        restarts=1,
        mcmc_samples=60,
        mcmc_burn_in=30,
        mcmc_thin=5,
        mcmc_chains=2,
        cache_dir=str(tmp_path),
        prs_path=str(tmp_path / "aou_prs_panel.csv.gz"),
        n_prs_nodes=2,
    )
    np.testing.assert_allclose(r1.mcmc_edge_probs, r2.mcmc_edge_probs, atol=1e-12)
    np.testing.assert_array_equal(r1.dagslam_adjacency, r2.dagslam_adjacency)


def test_pipeline_raises_when_no_csv(tmp_path):
    """No fallback: missing CSV raises FileNotFoundError up to the caller."""
    from causal_pred.pipeline import run_pipeline

    try:
        run_pipeline(seed=0, cache_dir=str(tmp_path), bucket=None)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError when no cohort CSV is available")
