"""Unit tests for the OpenGWAS two-sample MR fetcher.

These tests do not hit the network: ``load_live_gwas`` receives fake
OpenGWAS clients that return canned payloads so the harmonisation + IVW
math + caching layer can be exercised deterministically.
"""

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


def test_load_live_gwas_uses_cache_and_computes_ivw(tmp_path: Path):
    rng = np.random.default_rng(42)

    def make_hits(study_id: str) -> list[dict]:
        # Plant a known effect: tophit betas are all positive.
        return [_hit(f"{study_id}_rs{k}", float(rng.normal(0.05, 0.005)), 0.005)
                for k in range(20)]

    def make_assocs(out_id: str, rsids: list[str], scale: float) -> list[dict]:
        out = []
        for rs in rsids:
            # outcome beta = scale * exposure-like beta (we synthesise a number
            # consistent with a true causal effect = scale).
            out.append({
                "rsid": rs,
                "beta": float(scale * 0.05 + rng.normal(0.0, 0.001)),
                "se": 0.001,
                "ea": "A",
                "nea": "G",
                "eaf": 0.3,
            })
        return out

    captured_calls: list[tuple[str, object]] = []

    class FakeClient:
        @property
        def authenticated(self) -> bool:
            return True

        def fetch_tophits(self, study_id, *, pval, r2, kb, pop):
            del pval, r2, kb, pop
            captured_calls.append(("tophits", study_id))
            return make_hits(study_id)

        def fetch_associations_by_study(
            self,
            study_ids,
            rsids,
            *,
            max_workers=4,
        ):
            del max_workers
            captured_calls.append(("assocs", tuple(sorted(study_ids))))
            return {
                sid: make_assocs(sid, list(rsids), scale_per_outcome[sid])
                for sid in study_ids
            }

    # We map outcome_id -> scale so we know what IVW should return.
    scale_per_outcome = {opengwas.OPENGWAS_STUDY_IDS[t]: float(i + 1)
                         for i, t in enumerate(opengwas.OPENGWAS_STUDY_IDS)}
    client = FakeClient()

    summary = opengwas.load_live_gwas(client=client, cache_dir=tmp_path)

    # We should have at least one usable cell (BMI -> T2D, etc.).
    usable = np.isfinite(summary.betas) & np.isfinite(summary.ses)
    assert usable.any(), "expected at least one usable IVW cell"

    # IVW point estimates should match the planted scales (within MC noise).
    bmi_idx = summary.exposures.index("BMI")
    t2d_idx = summary.outcomes.index("T2D")
    expected = scale_per_outcome[opengwas.OPENGWAS_STUDY_IDS["T2D"]]
    assert abs(float(summary.betas[bmi_idx, t2d_idx]) - expected) < 0.05

    # diet_quality has no curated study id -> entire row should be NaN.
    diet_idx = summary.exposures.index("diet_quality")
    assert not np.any(np.isfinite(summary.betas[diet_idx])), (
        "diet_quality row should remain NaN since no study ID is curated"
    )

    # Second call should be served entirely from cache.
    captured_calls.clear()
    summary2 = opengwas.load_live_gwas(client=client, cache_dir=tmp_path)
    assert captured_calls == [], (
        f"second call should be all-cache, but hit network: {captured_calls}"
    )
    np.testing.assert_allclose(summary.betas, summary2.betas, equal_nan=True)


def test_no_jwt_yields_all_nan_summary(tmp_path: Path):
    class NoTokenClient:
        @property
        def authenticated(self) -> bool:
            return False

        def fetch_tophits(self, *args, **kwargs):
            raise AssertionError("network should not be hit when client has no token")

        def fetch_associations_by_study(self, *args, **kwargs):
            raise AssertionError("network should not be hit when client has no token")

    summary = opengwas.load_live_gwas(client=NoTokenClient(), cache_dir=tmp_path)
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
