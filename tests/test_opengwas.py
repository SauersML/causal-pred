"""Unit tests for the OpenGWAS two-sample MR fetcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from causal_pred.data import opengwas


def _hit(rsid: str, beta: float, se: float, ea: str = "A", nea: str = "G",
         eaf: float = 0.3) -> dict:
    return {
        "rsid": rsid,
        "beta": beta,
        "se": se,
        "ea": ea,
        "nea": nea,
        "eaf": eaf,
        "p": 1e-12,
        "n": 100000,
    }


def test_ivw_recovers_planted_effect():
    rng = np.random.default_rng(123)
    true_beta = 0.42
    pairs = []
    for k in range(50):
        bx = float(rng.normal(0.05, 0.005))
        sx = 0.005
        true_by = true_beta * bx
        sy = 0.003
        by = float(rng.normal(true_by, sy))
        pairs.append((bx, sx, by, sy, f"rs{k}"))
    beta, se, n = opengwas.ivw(pairs)
    assert n == 50
    assert abs(beta - true_beta) < 0.05
    assert se > 0


def test_harmonise_flips_outcome_when_alleles_disagree():
    hits = [_hit("rs1", 0.10, 0.01, ea="A", nea="G")]
    assocs = [
        {"rsid": "rs1", "beta": -0.05, "se": 0.01, "ea": "G", "nea": "A"},
    ]
    pairs = opengwas.harmonise_pairs(hits, assocs)
    assert len(pairs) == 1
    bx, sx, by, sy, rsid = pairs[0]
    assert rsid == "rs1"
    # outcome beta should have been sign-flipped from -0.05 -> +0.05
    assert by == 0.05


def test_harmonise_drops_palindromic_with_intermediate_maf():
    hits = [_hit("rs2", 0.10, 0.01, ea="A", nea="T", eaf=0.5)]
    assocs = [
        {"rsid": "rs2", "beta": 0.05, "se": 0.01, "ea": "A", "nea": "T", "eaf": 0.5},
    ]
    pairs = opengwas.harmonise_pairs(hits, assocs)
    assert pairs == []


def test_harmonise_keeps_palindromic_with_extreme_maf():
    hits = [_hit("rs3", 0.10, 0.01, ea="A", nea="T", eaf=0.05)]
    assocs = [
        {"rsid": "rs3", "beta": 0.05, "se": 0.01, "ea": "A", "nea": "T", "eaf": 0.05},
    ]
    pairs = opengwas.harmonise_pairs(hits, assocs)
    assert len(pairs) == 1


def test_harmonise_drops_incompatible_alleles():
    hits = [_hit("rs4", 0.10, 0.01, ea="A", nea="G")]
    assocs = [
        {"rsid": "rs4", "beta": 0.05, "se": 0.01, "ea": "C", "nea": "T"},
    ]
    pairs = opengwas.harmonise_pairs(hits, assocs)
    assert pairs == []


def test_load_live_gwas_real_cache_produces_ivw_cells():
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "mr_cache"
    assert cache_dir.is_dir(), "real OpenGWAS cache must be present"

    summary = opengwas.load_live_gwas(cache_dir=cache_dir)

    usable = np.isfinite(summary.betas) & np.isfinite(summary.ses)
    assert usable.any(), "expected at least one usable IVW cell"
    assert int(usable.sum()) >= 40

    bmi_idx = summary.exposures.index("BMI")
    ldl_idx = summary.outcomes.index("LDL")
    assert np.isfinite(summary.betas[bmi_idx, ldl_idx])
    assert np.isfinite(summary.ses[bmi_idx, ldl_idx])
    assert summary.ses[bmi_idx, ldl_idx] > 0.0
    assert ("BMI", "LDL") in summary.citations

    # diet_quality has no curated study id -> entire row should be NaN.
    diet_idx = summary.exposures.index("diet_quality")
    assert not np.any(np.isfinite(summary.betas[diet_idx])), (
        "diet_quality row should remain NaN since no study ID is curated"
    )

    sources = {row.source for row in summary.per_pair}
    assert sources <= {"cache", "circular", "no_id", "no_overlap"}


def test_no_jwt_yields_all_nan_summary(tmp_path: Path):
    summary = opengwas.load_live_gwas(
        client=opengwas.OpenGWASClient(token=None),
        cache_dir=tmp_path,
    )
    assert not np.any(np.isfinite(summary.betas))


def test_curated_ids_match_documented_set():
    # Lock in the ID picks so accidental edits surface in code review.
    expected = {
        "BMI": "ieu-b-40",
        "LDL": "ieu-b-110",
        "HbA1c": "ebi-a-GCST90002244",
        "systolic_BP": "ieu-b-38",
        "years_smoking": "ieu-b-25",
        "physical_activity": "ukb-b-4710",
        "hypertension": "ukb-b-12493",
        "T2D": "ebi-a-GCST006867",
        "cardiovascular_disease": "ebi-a-GCST005195",
    }
    assert opengwas.OPENGWAS_STUDY_IDS == expected
    for sid in expected.values():
        assert sid in opengwas.STUDY_CITATIONS, f"missing citation for {sid}"
