"""Tests for the single no-argument production pipeline."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _make_tiny_cohort_csv(path):
    rng = np.random.default_rng(0)
    n = 80
    bmi = rng.normal(28.0, 5.0, size=n)
    hba1c = rng.normal(6.0, 1.0, size=n)
    hdl = rng.normal(50.0, 10.0, size=n)
    ldl = rng.normal(100.0, 25.0, size=n)
    trig = rng.normal(120.0, 50.0, size=n)
    logits = -3.0 + 0.06 * (bmi - 25.0) + 0.8 * (hba1c - 5.5)
    p = 1.0 / (1.0 + np.exp(-logits))
    t2d = (rng.random(n) < p).astype(int)
    followup = rng.uniform(2.0, 20.0, size=n)
    followup[t2d == 1] = rng.uniform(1.0, 12.0, size=int(t2d.sum()))
    pd.DataFrame(
        {
            "person_id": [str(1000 + i) for i in range(n)],
            "type2_diabetes": t2d,
            "followup_years": followup,
            "event": t2d,
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
    ).to_csv(path, index=False, compression="gzip")


def _make_tiny_static_cohort_csv(path):
    _make_tiny_cohort_csv(path)
    df = pd.read_csv(path)
    df = df.drop(columns=["followup_years", "event"])
    df.to_csv(path, index=False)


def _configure_tiny_pipeline(monkeypatch, tmp_path):
    from causal_pred import pipeline
    from causal_pred.data.cohort import EhrPanel
    from causal_pred.data.nodes import NODE_INDEX
    from causal_pred.data.synthetic import SyntheticDataset

    monkeypatch.delenv("WORKSPACE_BUCKET", raising=False)
    monkeypatch.setenv("WORKSPACE_CDR", "project.dataset")
    monkeypatch.setattr(pipeline, "DEFAULT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "DEFAULT_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(pipeline, "PIPELINE_CONFIG_VERSION", "test-single-path")
    monkeypatch.setattr(pipeline, "PRS_NODES", 2)
    monkeypatch.setattr(pipeline, "PRS_MIN_COMPLETE_ROWS", 20)
    monkeypatch.setattr(pipeline, "DAGSLAM_MAX_ITER", 60)
    monkeypatch.setattr(pipeline, "DAGSLAM_RESTARTS", 1)
    monkeypatch.setattr(pipeline, "MCMC_SAMPLES", 50)
    monkeypatch.setattr(pipeline, "MCMC_BURN_IN", 20)
    monkeypatch.setattr(pipeline, "MCMC_THIN", 5)
    monkeypatch.setattr(pipeline, "MCMC_CHAINS", 2)
    monkeypatch.setattr(pipeline, "VALIDATION_N_PERMUTE", 10)
    monkeypatch.setattr(pipeline, "SURVIVAL_TIME_GRID_POINTS", 6)

    def _fake_prs(_cache, _cohort_csv, person_ids, _logger):
        prs_path = tmp_path / "aou_prs_panel.csv.gz"
        prs = pipeline._read_prs_panel(prs_path)
        return prs.reindex(pd.Series(person_ids, dtype="string").astype(str)), str(prs_path)

    def _fake_mrdag(_cache, _logger):
        p = len(NODE_INDEX)
        pi = np.full((p, p), 0.5, dtype=float)
        np.fill_diagonal(pi, np.nan)
        return pi, {"source": "test"}

    def _fake_ehr(_cache, person_ids, _logger):
        pid = np.asarray([str(p) for p in person_ids])
        x = np.linspace(-1.0, 1.0, pid.size).reshape(-1, 1)
        return EhrPanel(
            matrix=x,
            person_id=pid,
            feature_names=("ehr_shared_signal",),
            feature_kinds=("condition",),
        )

    def _fake_genscore(_cache, data, person_ids, _prs_df, _ehr_panel, _logger):
        feat = np.mean(data.X[:, 1:3], axis=1, keepdims=True)
        X = np.concatenate([data.X, feat], axis=1)
        p_old = data.p
        p_new = X.shape[1]
        gt = np.zeros((p_new, p_new), dtype=int)
        gt[:p_old, :p_old] = data.ground_truth_adj
        dataset = SyntheticDataset(
            X=X,
            time=data.time.copy(),
            event=data.event.copy(),
            columns=tuple(data.columns) + ("feat_0000",),
            node_types=tuple(data.node_types) + ("continuous",),
            ground_truth_adj=gt,
        )
        return dataset, np.asarray(person_ids).astype(str), {
            "crosscoder_d": 4,
            "crosscoder_k": 1,
            "promoted_names": ["feat_0000"],
            "promoted_indices": [0],
            "promoted_genome_share": [0.5],
            "promoted_activation_rate": [1.0],
            "base_n": int(data.n),
            "augmented_n": int(data.n),
            "ehr_feature_count": int(_ehr_panel.m),
        }, None

    def _fake_survival(_cache, _key, data, _samples, _logger):
        t_grid = np.linspace(1.0, 12.0, 6)
        baseline = np.exp(-t_grid / 20.0)
        risk_shift = 0.05 * data.X[:, data.columns.index("bmi")]
        survival = np.clip(baseline[None, :] - risk_shift[:, None], 0.01, 0.99)
        diag = {
            "backend": "gamfit",
            "n_parent_sets": 1,
            "metrics": {"nagelkerke_r2_at_10y": 0.1},
        }
        return (
            t_grid,
            survival,
            np.clip(survival - 0.02, 0.0, 1.0),
            np.clip(survival + 0.02, 0.0, 1.0),
            ("bmi",),
            diag,
            0.01,
        )

    monkeypatch.setattr(pipeline, "_load_or_build_prs_panel", _fake_prs)
    monkeypatch.setattr(pipeline, "_load_or_run_mrdag", _fake_mrdag)
    monkeypatch.setattr(pipeline, "_load_or_build_ehr_panel", _fake_ehr)
    monkeypatch.setattr(pipeline, "_load_or_run_genscore_features", _fake_genscore)
    monkeypatch.setattr(pipeline, "_load_or_run_survival_gam", _fake_survival)
    return pipeline


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    _make_tiny_cohort_csv(tmp_path / "t2d_initial_nodes_complete.csv")
    _make_tiny_prs_csv(tmp_path / "aou_prs_panel.csv.gz")

    pipeline = _configure_tiny_pipeline(monkeypatch, tmp_path)
    result = pipeline.run_pipeline()

    assert isinstance(result, pipeline.PipelineResult)
    assert tuple(result.columns[:6]) == (
        "type2_diabetes",
        "bmi",
        "hba1c",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    )
    assert len(result.columns) == 9
    assert all(col.startswith("pgs_") for col in result.columns[6:8])
    assert result.columns[-1] == "feat_0000"
    p = len(result.columns)
    assert result.dagslam_adjacency.shape == (p, p)
    assert result.mcmc_edge_probs.shape == (p, p)
    assert result.thresholded_adjacency.shape == (p, p)
    assert result.survival_mean.shape == (result.data_summary["n"], 6)
    assert result.survival_diagnostics["backend"] == "gamfit"

    out_dir = tmp_path / "outputs"
    pipeline.save_result(result, outdir=str(out_dir), run_config={"seed": 0})

    for fname in (
        "dagslam_adjacency.npy",
        "mcmc_edge_probs.npy",
        "mcmc_samples.npy",
        "thresholded_adjacency.npy",
        "survival_time_grid.npy",
        "survival_mean.npy",
        "survival_lower.npy",
        "survival_upper.npy",
        "disease_risk_mean.npy",
        "mrdag_pi_long.csv",
        "mrdag_prior_long.csv",
        "greedy_edges.csv",
        "mcmc_thresholded_edges.csv",
        "mcmc_edge_probabilities_long.csv",
        "survival_curves_long.csv",
        "causal_pathway_probabilities.csv",
        "crosscoder_features.json",
        "summary.json",
        "run_config.json",
    ):
        assert (out_dir / fname).exists(), fname

    with open(out_dir / "summary.json") as fh:
        summary = json.load(fh)
    assert summary["columns"] == list(result.columns)
    assert summary["dagslam_n_edges"] == result.dagslam_n_edges
    assert summary["threshold"] == 0.5
    assert "accept_rate" in summary["mcmc_diagnostics"]
    assert summary["genscore_features"]["prs_columns_selected"] == 2
    assert summary["genscore_features"]["promoted_names"] == ["feat_0000"]
    assert summary["survival_diagnostics"]["backend"] == "gamfit"
    assert "survival" in summary["validation"]


def test_acyclic_threshold_skips_cycle_closing_edges():
    from causal_pred import pipeline

    edge_probs = np.array(
        [
            [0.0, 0.90, 0.10],
            [0.20, 0.0, 0.80],
            [0.70, 0.30, 0.0],
        ]
    )

    adj = pipeline._acyclic_threshold_from_edge_probs(edge_probs, threshold=0.5)

    np.testing.assert_array_equal(
        adj,
        np.array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ]
        ),
    )


def test_acyclic_threshold_respects_allowed_edges():
    from causal_pred import pipeline

    edge_probs = np.array(
        [
            [0.0, 0.9, 0.9],
            [0.1, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ]
    )
    allowed = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(allowed, False)
    allowed[0, 1] = False

    adj = pipeline._acyclic_threshold_from_edge_probs(
        edge_probs,
        threshold=0.5,
        allowed_edges=allowed,
    )

    np.testing.assert_array_equal(
        adj,
        np.array(
            [
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )


def test_log_validation_metrics_uses_threshold_values(caplog):
    from causal_pred import pipeline

    logger = logging.getLogger("test-validation-log")
    validation = {
        "auroc": 0.8,
        "auroc_null_mean": 0.5,
        "auprc": 0.7,
        "auprc_null_mean": 0.2,
        "n_valid_ground_truth_edges": 2,
        "n_ground_truth_edges": 3,
        "thresholds": [0.5],
        "observed_recovery": {0.5: 0.25},
        "recovery_pvalue": {0.5: 0.125},
        "mcc": {0.5: 0.375},
        "mcc_pvalue": {0.5: 0.625},
    }

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline._log_validation_metrics(logger, validation)

    assert (
        "thr=0.50 recovery=0.250 (p=0.125)  MCC=0.375 (p=0.625)"
        in caplog.text
    )


def test_log_validation_metrics_accepts_json_string_threshold_keys(caplog):
    from causal_pred import pipeline

    logger = logging.getLogger("test-validation-log-json")
    validation = {
        "auroc": 0.8,
        "auroc_null_mean": 0.5,
        "auprc": 0.7,
        "auprc_null_mean": 0.2,
        "n_valid_ground_truth_edges": 2,
        "n_ground_truth_edges": 3,
        "thresholds": [0.5],
        "observed_recovery": {"0.5": 0.25},
        "recovery_pvalue": {"0.5": 0.125},
        "mcc": {"0.5": 0.375},
        "mcc_pvalue": {"0.5": 0.625},
    }

    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline._log_validation_metrics(logger, validation)

    assert "recovery=0.250 (p=0.125)" in caplog.text


def test_structural_allowed_edges_make_target_sink_and_roots():
    from causal_pred import pipeline

    columns = (
        "type2_diabetes",
        "bmi",
        "pgs_pgs000013",
        "feat_0001",
        "age",
    )
    node_types = ("binary", "continuous", "continuous", "continuous", "continuous")

    allowed = pipeline._structural_allowed_edges(columns, node_types)

    target = columns.index("type2_diabetes")
    pgs = columns.index("pgs_pgs000013")
    feat = columns.index("feat_0001")
    age = columns.index("age")
    bmi = columns.index("bmi")

    assert not allowed[target].any()
    assert not allowed[:, pgs].any()
    assert not allowed[:, age].any()
    assert allowed[bmi, target]
    assert allowed[pgs, target]
    assert allowed[pgs, feat]
    assert allowed[bmi, feat]
    assert allowed[feat, target]
    assert not allowed[target, feat]
    assert not allowed[feat, pgs]


def test_run_key_hashes_structural_allowed_edges():
    from causal_pred import pipeline
    from causal_pred.data.synthetic import SyntheticDataset

    data = SyntheticDataset(
        X=np.ones((4, 2), dtype=float),
        time=np.linspace(1.0, 4.0, 4),
        event=np.array([0, 1, 0, 1], dtype=int),
        columns=("x0", "x1"),
        node_types=("continuous", "continuous"),
        ground_truth_adj=np.zeros((2, 2), dtype=int),
    )
    prior = np.full((2, 2), np.nan, dtype=float)
    allowed_a = np.ones((2, 2), dtype=bool)
    np.fill_diagonal(allowed_a, False)
    allowed_b = allowed_a.copy()
    allowed_b[0, 1] = False

    assert pipeline._run_key(data, prior, allowed_a) != pipeline._run_key(
        data,
        prior,
        allowed_b,
    )


def test_mrdag_cache_key_hashes_gwas_evidence_and_metadata():
    from causal_pred import pipeline

    def make_gwas(beta: float, pval: float = 5e-8):
        return SimpleNamespace(
            exposures=("BMI",),
            outcomes=("T2D",),
            betas=np.array([[beta]], dtype=float),
            ses=np.array([[0.1]], dtype=float),
            ivw_pvals=np.array([[0.01]], dtype=float),
            n_snps=np.array([12], dtype=int),
            citations={("BMI", "T2D"): "test citation"},
            circular_pairs=(),
            source_metadata={"params": {"pval": pval, "r2": 0.001, "kb": 10000}},
            per_pair=[
                SimpleNamespace(
                    exposure="BMI",
                    outcome="T2D",
                    exposure_id="exp",
                    outcome_id="out",
                    beta=beta,
                    se=0.1,
                    n_snps=12,
                    source="cache",
                    note="",
                )
            ],
        )

    key_a = pipeline._mrdag_cache_key(make_gwas(0.2), "live")
    key_b = pipeline._mrdag_cache_key(make_gwas(0.3), "live")
    key_c = pipeline._mrdag_cache_key(make_gwas(0.2, pval=1e-6), "live")

    assert key_a != key_b
    assert key_a != key_c


def test_prs_cache_names_hash_genotype_score_and_scorer(tmp_path, monkeypatch):
    from causal_pred import pipeline

    bed = tmp_path / "arrays.bed"
    bed.write_bytes(b"aa")
    bed.with_suffix(".bim").write_text("bim-a\n")
    bed.with_suffix(".fam").write_text("fam-a\n")
    score = tmp_path / "PGS000001_hmPOS_GRCh38.txt"
    score.write_text("score-a\n")
    people = ["1", "2"]

    scorer_a = {"available": True, "path": "/bin/gnomon-a", "sha256": "a"}
    scorer_b = {"available": True, "path": "/bin/gnomon-b", "sha256": "b"}
    monkeypatch.setattr(pipeline, "_gnomon_scorer_fingerprint", lambda: scorer_a)
    base_panel = pipeline._prs_panel_cache_filename(bed, [score], people)
    base_sscore = pipeline._gnomon_score_cache_filename(bed, [score])

    bed.write_bytes(b"bb")
    genotype_changed = pipeline._prs_panel_cache_filename(bed, [score], people)
    genotype_sscore_changed = pipeline._gnomon_score_cache_filename(bed, [score])
    assert genotype_changed != base_panel
    assert genotype_sscore_changed != base_sscore

    bed.write_bytes(b"aa")
    score.write_text("score-b\n")
    score_changed = pipeline._prs_panel_cache_filename(bed, [score], people)
    assert score_changed != base_panel

    score.write_text("score-a\n")
    monkeypatch.setattr(pipeline, "_gnomon_scorer_fingerprint", lambda: scorer_b)
    scorer_changed = pipeline._prs_panel_cache_filename(bed, [score], people)
    scorer_sscore_changed = pipeline._gnomon_score_cache_filename(bed, [score])
    assert scorer_changed != base_panel
    assert scorer_sscore_changed != base_sscore


def test_dagslam_cache_rejects_stale_structural_mask(tmp_path, monkeypatch):
    from causal_pred import pipeline
    from causal_pred.data.synthetic import SyntheticDataset

    data = SyntheticDataset(
        X=np.ones((5, 2), dtype=float),
        time=np.linspace(1.0, 5.0, 5),
        event=np.array([0, 1, 0, 1, 0], dtype=int),
        columns=("x0", "x1"),
        node_types=("continuous", "continuous"),
        ground_truth_adj=np.zeros((2, 2), dtype=int),
    )
    prior = np.full((2, 2), np.nan, dtype=float)
    stale_allowed = np.ones((2, 2), dtype=bool)
    np.fill_diagonal(stale_allowed, False)
    allowed = stale_allowed.copy()
    allowed[0, 1] = False

    cache = pipeline.WorkspaceCache(tmp_path, None)
    key = "stale-mask"
    path = cache.path(f"dagslam-{key}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        adjacency=np.array([[0, 1], [0, 0]], dtype=int),
        log_score=np.array(123.0),
        n_edges=np.array(1),
        runtime_s=np.array(0.1),
        pi_prior=prior,
        allowed_edges=stale_allowed,
    )

    calls = []

    class FakeDagResult:
        adjacency = np.zeros((2, 2), dtype=int)
        log_score = 0.0
        n_edges = 0

    def fake_run_dagslam(**kwargs):
        calls.append(kwargs)
        return FakeDagResult()

    monkeypatch.setattr(pipeline, "run_dagslam", fake_run_dagslam)

    out = pipeline._load_or_run_dagslam(
        cache,
        key,
        data,
        prior,
        logging.getLogger("test-dagslam-cache"),
        allowed_edges=allowed,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(out["adjacency"], np.zeros((2, 2), dtype=int))
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["allowed_edges"], allowed)


def test_mcmc_cache_rejects_stale_structural_mask(tmp_path, monkeypatch):
    from causal_pred import pipeline
    from causal_pred.data.synthetic import SyntheticDataset

    data = SyntheticDataset(
        X=np.ones((5, 2), dtype=float),
        time=np.linspace(1.0, 5.0, 5),
        event=np.array([0, 1, 0, 1, 0], dtype=int),
        columns=("x0", "x1"),
        node_types=("continuous", "continuous"),
        ground_truth_adj=np.zeros((2, 2), dtype=int),
    )
    prior = np.full((2, 2), np.nan, dtype=float)
    stale_allowed = np.ones((2, 2), dtype=bool)
    np.fill_diagonal(stale_allowed, False)
    allowed = stale_allowed.copy()
    allowed[0, 1] = False

    cache = pipeline.WorkspaceCache(tmp_path, None)
    key = "stale-mask"
    path = cache.path(f"mcmc-{key}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    stale_sample = np.array([[[0, 1], [0, 0]]], dtype=np.int8)
    np.savez(
        path,
        edge_probs=stale_sample[0].astype(float),
        samples=stale_sample,
        diagnostics_json=np.array(json.dumps({"cached": True})),
        runtime_s=np.array(0.1),
        allowed_edges=stale_allowed,
    )

    calls = []

    class FakeMCMCResult:
        edge_probs = np.zeros((2, 2), dtype=float)
        samples = [np.zeros((2, 2), dtype=np.int8)]
        diagnostics = {"cached": False}

    def fake_run_structure_mcmc(**kwargs):
        calls.append(kwargs)
        return FakeMCMCResult()

    monkeypatch.setattr(pipeline, "run_structure_mcmc", fake_run_structure_mcmc)

    edge_probs, samples, diagnostics, _runtime = pipeline._load_or_run_mcmc(
        cache,
        key,
        data,
        np.zeros((2, 2), dtype=int),
        prior,
        logging.getLogger("test-mcmc-cache"),
        allowed_edges=allowed,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(edge_probs, np.zeros((2, 2), dtype=float))
    np.testing.assert_array_equal(samples, np.zeros((1, 2, 2), dtype=np.int8))
    assert diagnostics == {"cached": False}
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["allowed_edges"], allowed)


def test_posterior_parent_sets_use_empirical_sample_weights():
    from causal_pred import pipeline

    samples = np.zeros((10, 4, 4), dtype=np.int8)
    target = 3
    samples[:5, 0, target] = 1
    samples[:5, 1, target] = 1
    samples[5:, 0, target] = 1
    samples[5:, 2, target] = 1

    parent_sets, weights, counts = pipeline._posterior_parent_sets(
        samples,
        target,
        top_k=4,
    )

    assert parent_sets == [(0, 1), (0, 2)]
    assert counts == [5, 5]
    np.testing.assert_allclose(weights, [0.5, 0.5])
    assert weights.sum() == pytest.approx(1.0)


def test_survival_parent_sets_include_median_probability_model():
    from causal_pred import pipeline

    samples = np.zeros((10, 4, 4), dtype=np.int8)
    target = 3
    samples[:5, 0, target] = 1
    samples[:5, 1, target] = 1
    samples[5:, 0, target] = 1
    samples[5:, 2, target] = 1

    parent_sets, weights, counts, sources = pipeline._survival_parent_sets(
        samples,
        target,
        top_k=4,
    )

    assert (0, 1, 2) in parent_sets
    idx = parent_sets.index((0, 1, 2))
    assert sources[idx] == "median_probability"
    assert counts[idx] == 5
    assert weights.sum() == pytest.approx(1.0)


def test_survival_gam_metrics_use_held_out_refit(tmp_path, monkeypatch):
    from causal_pred import pipeline
    from causal_pred.data.synthetic import SyntheticDataset

    data = SyntheticDataset(
        X=np.column_stack(
            [
                np.array([0, 1] * 20, dtype=float),
                np.linspace(-1.0, 1.0, 40),
            ]
        ),
        time=np.linspace(1.0, 16.0, 40),
        event=np.array([0, 1] * 20, dtype=int),
        columns=("type2_diabetes", "bmi"),
        node_types=("binary", "continuous"),
        ground_truth_adj=np.zeros((2, 2), dtype=int),
    )
    monkeypatch.setattr(pipeline, "SURVIVAL_TIME_GRID_POINTS", 5)
    monkeypatch.setattr(
        pipeline,
        "_survival_parent_sets",
        lambda *_args, **_kwargs: ([(1,)], np.array([1.0]), [10], ["test"]),
    )

    def fake_parent_fit(_cache, _key, fit_data, _parent_set, t_grid, _logger):
        return (
            np.full((fit_data.n, t_grid.size), 0.8, dtype=float),
            np.zeros((fit_data.n, t_grid.size), dtype=float),
            {"parent_columns": ["bmi"]},
            0.01,
        )

    seen = {}

    def fake_holdout(fit_data, parent_set, train_idx, test_idx, t_grid, _logger):
        seen["train_idx"] = train_idx.copy()
        seen["test_idx"] = test_idx.copy()
        seen["parent_set"] = tuple(parent_set)
        return (
            np.full((test_idx.size, t_grid.size), 0.7, dtype=float),
            {"parent_columns": ["bmi"]},
        )

    def fake_metrics(time_arr, event_arr, survival_mean, _t_grid):
        seen["metrics_time"] = np.asarray(time_arr).copy()
        seen["metrics_event"] = np.asarray(event_arr).copy()
        seen["metrics_shape"] = survival_mean.shape
        return {"sentinel": "held_out"}

    monkeypatch.setattr(pipeline, "_load_or_run_survival_parent_fit", fake_parent_fit)
    monkeypatch.setattr(pipeline, "_fit_survival_parent_set_holdout", fake_holdout)
    monkeypatch.setattr(pipeline, "_survival_metrics", fake_metrics)

    cache = pipeline.WorkspaceCache(tmp_path, None)
    (
        _t_grid,
        survival_mean,
        _survival_lower,
        _survival_upper,
        parent_columns,
        diagnostics,
        _runtime_s,
    ) = pipeline._load_or_run_survival_gam(
        cache,
        "heldout-validation",
        data,
        np.zeros((1, 2, 2), dtype=np.int8),
        logging.getLogger("test-survival-heldout"),
    )

    assert survival_mean.shape[0] == data.n
    assert parent_columns == ("bmi",)
    assert diagnostics["metrics"] == {"sentinel": "held_out"}
    assert diagnostics["metrics_evaluation"]["method"] == "stratified_holdout_refit"
    assert diagnostics["metrics_evaluation"]["n_test"] == seen["test_idx"].size
    assert seen["parent_set"] == (1,)
    assert set(seen["train_idx"]).isdisjoint(set(seen["test_idx"]))
    np.testing.assert_array_equal(seen["metrics_time"], data.time[seen["test_idx"]])
    np.testing.assert_array_equal(seen["metrics_event"], data.event[seen["test_idx"]])
    assert seen["metrics_shape"][0] == seen["test_idx"].size


def test_pipeline_determinism(tmp_path, monkeypatch):
    _make_tiny_cohort_csv(tmp_path / "t2d_initial_nodes_complete.csv")
    _make_tiny_prs_csv(tmp_path / "aou_prs_panel.csv.gz")

    pipeline = _configure_tiny_pipeline(monkeypatch, tmp_path)
    r1 = pipeline.run_pipeline()
    r2 = pipeline.run_pipeline()
    np.testing.assert_allclose(r1.mcmc_edge_probs, r2.mcmc_edge_probs, atol=1e-12)
    np.testing.assert_array_equal(r1.dagslam_adjacency, r2.dagslam_adjacency)


def test_dagslam_prior_changes_real_search(tmp_path, monkeypatch):
    from causal_pred import pipeline
    from causal_pred.data.synthetic import SyntheticDataset

    rng = np.random.default_rng(0)
    data = SyntheticDataset(
        X=rng.normal(size=(80, 2)),
        time=np.linspace(1.0, 10.0, 80),
        event=np.zeros(80, dtype=int),
        columns=("x0", "x1"),
        node_types=("continuous", "continuous"),
        ground_truth_adj=np.zeros((2, 2), dtype=int),
    )
    neutral_prior = np.full((2, 2), np.nan, dtype=float)
    strong_prior = np.array([[np.nan, 1.0 - 1e-12], [1e-12, np.nan]], dtype=float)
    monkeypatch.setattr(pipeline, "DAGSLAM_MAX_PARENTS", 1)
    monkeypatch.setattr(pipeline, "DAGSLAM_MAX_ITER", 10)
    monkeypatch.setattr(pipeline, "DAGSLAM_RESTARTS", 1)
    cache = pipeline.WorkspaceCache(tmp_path, None)

    neutral = pipeline._load_or_run_dagslam(
        cache,
        "prior-neutral",
        data,
        neutral_prior,
        logging.getLogger("test-dagslam-neutral"),
    )
    biased = pipeline._load_or_run_dagslam(
        cache,
        "prior-biased",
        data,
        strong_prior,
        logging.getLogger("test-dagslam-biased"),
    )

    np.testing.assert_array_equal(neutral["adjacency"], np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(
        biased["adjacency"],
        np.array([[0, 1], [0, 0]], dtype=int),
    )
    with np.load(cache.path("dagslam-prior-biased.npz"), allow_pickle=False) as z:
        np.testing.assert_allclose(z["pi_prior"], strong_prior, equal_nan=True)


def test_pipeline_builds_survival_outcome_when_csv_lacks_time_event(tmp_path, monkeypatch):
    _make_tiny_static_cohort_csv(tmp_path / "t2d_initial_nodes_complete.csv")
    _make_tiny_prs_csv(tmp_path / "aou_prs_panel.csv.gz")

    pipeline = _configure_tiny_pipeline(monkeypatch, tmp_path)

    def _fake_survival_outcome(_cache, person_ids, _logger):
        pid = np.asarray([str(p) for p in person_ids])
        n = pid.size
        event = np.zeros(n, dtype=int)
        event[::5] = 1
        return pipeline.SurvivalOutcome(
            person_id=pid,
            time=np.linspace(2.0, 12.0, n),
            event=event,
            keep=np.ones(n, dtype=bool),
            baseline_dt=np.array(["2020-01-01"] * n, dtype="datetime64[ns]"),
            end_dt=np.array(["2030-01-01"] * n, dtype="datetime64[ns]"),
            t2d_dt=np.array(["NaT"] * n, dtype="datetime64[ns]"),
            meta={
                "source": "test_omop",
                "n_input": int(n),
                "n_kept": int(n),
                "n_events": int(event.sum()),
            },
        )

    monkeypatch.setattr(
        pipeline,
        "_load_or_build_survival_outcome",
        _fake_survival_outcome,
    )

    result = pipeline.run_pipeline()

    assert result.data_summary["survival_outcome"]["source"] == "test_omop"
    assert result.data_summary["event_rate"] == pytest.approx(0.2)
    assert result.survival_mean.shape == (result.data_summary["n"], 6)
    assert result.survival_diagnostics["backend"] == "gamfit"


def test_pipeline_raises_when_no_csv(tmp_path, monkeypatch):
    pipeline = _configure_tiny_pipeline(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline()


def test_build_prs_panel_scores_downloaded_text_files(tmp_path, monkeypatch):
    from causal_pred import pipeline

    cohort_csv = tmp_path / "t2d_initial_nodes_complete.csv"
    _make_tiny_cohort_csv(cohort_csv)
    person_ids = pd.read_csv(cohort_csv, usecols=["person_id"], dtype=str)[
        "person_id"
    ].tolist()
    score_files = [tmp_path / "PGS000001_hmPOS_GRCh38.txt"]
    score_files[0].write_text("score\n")
    bed = tmp_path / "arrays.bed"
    bed.write_bytes(b"")
    bed.with_suffix(".bim").write_text("")
    bed.with_suffix(".fam").write_text("")

    captured = {}

    def fake_score_panel(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "PGS_A": np.linspace(-1.0, 1.0, len(person_ids)),
                "PGS_B": np.linspace(1.0, -1.0, len(person_ids)),
            },
            index=person_ids,
        )

    monkeypatch.setattr(pipeline, "DEFAULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(pipeline, "PRS_NODES", 2)
    monkeypatch.setattr(pipeline, "PRS_MIN_COMPLETE_ROWS", 20)
    monkeypatch.setattr(pipeline, "_resolve_microarray_bed", lambda: bed)
    monkeypatch.setattr(pipeline, "download_panel", lambda _panel_dir: score_files)
    monkeypatch.setattr(pipeline, "score_panel", fake_score_panel)

    out = pipeline._build_prs_panel(
        cohort_csv,
        tmp_path / "cache" / "aou_prs_panel.csv.gz",
        logging.getLogger("test"),
    )

    assert captured["score_path"] == [str(p) for p in score_files]
    assert captured["keep_iids"] == [str(pid) for pid in person_ids]
    assert out.shape == (len(person_ids), 2)


def test_build_prs_panel_reuses_cached_gnomon_sscore(tmp_path, monkeypatch):
    from causal_pred import pipeline

    cohort_csv = tmp_path / "t2d_initial_nodes_complete.csv"
    _make_tiny_cohort_csv(cohort_csv)
    person_ids = pd.read_csv(cohort_csv, usecols=["person_id"], dtype=str)[
        "person_id"
    ].tolist()
    score_files = [tmp_path / "PGS000001_hmPOS_GRCh38.txt"]
    score_files[0].write_text("score\n")
    bed = tmp_path / "arrays.bed"
    bed.write_bytes(b"")
    bed.with_suffix(".bim").write_text("")
    bed.with_suffix(".fam").write_text("")

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(pipeline, "DEFAULT_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(pipeline, "PRS_NODES", 2)
    monkeypatch.setattr(pipeline, "PRS_MIN_COMPLETE_ROWS", 20)
    monkeypatch.setattr(pipeline, "_resolve_microarray_bed", lambda: bed)
    monkeypatch.setattr(pipeline, "download_panel", lambda _panel_dir: score_files)

    def fail_score_panel(**_kwargs):
        raise AssertionError("cached .sscore should avoid gnomon scoring")

    monkeypatch.setattr(pipeline, "score_panel", fail_score_panel)

    sscore_name = pipeline._gnomon_score_cache_filename(bed, score_files)
    sscore = cache_root / pipeline.GNOMON_OUT_DIRNAME / sscore_name
    sscore.parent.mkdir(parents=True, exist_ok=True)
    rows = ["#IID\tPGS_A_AVG\tPGS_A_MISSING_PCT\tPGS_B_AVG\tPGS_B_MISSING_PCT"]
    for i, pid in enumerate(person_ids):
        rows.append(f"{pid}\t{i * 0.1}\t0\t{1.0 - i * 0.01}\t0")
    sscore.write_text("\n".join(rows) + "\n")

    out = pipeline._build_prs_panel(
        cohort_csv,
        cache_root / "aou_prs_panel.csv.gz",
        logging.getLogger("test-cache"),
        cache=pipeline.WorkspaceCache(cache_root, None),
    )

    assert out.shape == (len(person_ids), 2)
    assert list(out.columns) == ["PGS_A", "PGS_B"]


def test_ehr_panel_uses_visit_baseline_summary(tmp_path, monkeypatch):
    from causal_pred import pipeline

    person_ids = [str(10_000 + i) for i in range(60)]
    baseline = pd.DataFrame(
        {
            "person_id": person_ids,
            "baseline_dt": pd.to_datetime(["2025-01-01"] * len(person_ids)),
        }
    )
    condition = pd.DataFrame(
        {
            "person_id": [int(pid) for pid in person_ids],
            "phecode": ["T2D"] * len(person_ids),
            "datetime": pd.to_datetime(["2024-06-01"] * len(person_ids)),
        }
    )
    progress_messages = []

    def fake_fetch_omop_long_frames(*, person_ids, progress, **_kwargs):
        progress_messages.append(tuple(person_ids))
        progress("visit_baseline cache hit rows=60 cols=2")
        return {"visit_baseline": baseline, "condition_long": condition}

    monkeypatch.setattr(pipeline, "PIPELINE_CONFIG_VERSION", "ehr-baseline-test")
    monkeypatch.setattr(pipeline, "fetch_omop_long_frames", fake_fetch_omop_long_frames)

    cache = pipeline.WorkspaceCache(tmp_path, None)
    panel = pipeline._load_or_build_ehr_panel(
        cache,
        person_ids,
        logging.getLogger("test-ehr"),
    )

    assert panel.n == len(person_ids)
    assert "cond:T2D" in panel.feature_names
    assert "utilisation:n_encounters" not in panel.feature_names
    assert progress_messages == [tuple(person_ids)]


def test_genscore_plot_thread_is_daemon(monkeypatch):
    from causal_pred import pipeline
    from causal_pred.data.cohort import EhrPanel

    created = {}

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            created["target"] = target
            created["name"] = name
            created["daemon"] = daemon
            self.name = name

        def start(self):
            created["started"] = True

        def is_alive(self):
            return False

    import threading

    monkeypatch.setattr(threading, "Thread", FakeThread)
    pipeline._BACKGROUND_PLOT_THREADS.clear()

    pipeline._render_genscore_plots_async(
        model_bundle={"unused": np.array([0])},
        prs_df=pd.DataFrame(index=["1"]),
        ehr_panel=EhrPanel(
            matrix=np.zeros((1, 0)),
            person_id=np.array(["1"]),
            feature_names=(),
            feature_kinds=(),
        ),
        person_ids=np.array(["1"]),
        genscore_meta={},
        logger=logging.getLogger("test-genscore-thread"),
    )

    assert created["name"] == "genscore-plots"
    assert created["daemon"] is True
    assert created["started"] is True
