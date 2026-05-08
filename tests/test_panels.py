"""Unit tests for the curated PGS Catalog panel.

These tests do not hit the network. They verify panel-set invariants
(disjointness, expected size, ID format) and the URL helper's structure
against a fixed snapshot date (2026-05-07).
"""

from __future__ import annotations

import re

from causal_pred.genscore.panels import (
    CARDIOMETABOLIC_PANEL_V1,
    PANEL_PROVENANCE,
    all_panel_ids,
    discover_local_panel,
    panel_area_for,
    pgs_catalog_url,
)


_PGS_RE = re.compile(r"^PGS\d{6}$")


def test_panel_has_expected_total_117_unique_ids():
    flat = all_panel_ids()
    assert len(flat) == 117
    assert len(set(flat)) == 117


def test_panel_ids_are_well_formed():
    for area, ids in CARDIOMETABOLIC_PANEL_V1.items():
        for pgs_id in ids:
            assert _PGS_RE.match(pgs_id), (
                f"{area}: {pgs_id!r} does not match PGS\\d{{6}}"
            )


def test_panel_areas_are_disjoint():
    seen: dict[str, str] = {}
    for area, ids in CARDIOMETABOLIC_PANEL_V1.items():
        for pgs_id in ids:
            prev = seen.get(pgs_id)
            assert prev is None, (
                f"{pgs_id} appears in both {prev!r} and {area!r}"
            )
            seen[pgs_id] = area


def test_excluded_aou_t2d_scores_are_absent():
    """The five strong-but-AoU-tainted T2D scores must not be in the panel."""
    excluded = {"PGS000014", "PGS000330", "PGS000729", "PGS001781", "PGS002243"}
    flat = set(all_panel_ids())
    assert excluded.isdisjoint(flat), (
        f"AoU-tainted scores leaked into panel: {excluded & flat}"
    )


def test_panel_area_for_resolves_correctly():
    assert panel_area_for("PGS003725") == "cad"
    assert panel_area_for("PGS004870") == "t2d"
    assert panel_area_for("PGS000888") == "ldl"
    assert panel_area_for("PGS999999") is None


def test_provenance_records_required_keys():
    for key in (
        "name",
        "snapshot_date",
        "source",
        "exclusion_rule",
        "scope",
        "excluded_strong_older_t2d_scores",
    ):
        assert key in PANEL_PROVENANCE
    assert PANEL_PROVENANCE["snapshot_date"] == "2026-05-07"


def test_pgs_catalog_url_structure():
    url = pgs_catalog_url("PGS000014")
    assert url == (
        "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000014/"
        "ScoringFiles/Harmonized/PGS000014_hmPOS_GRCh38.txt.gz"
    )
    url37 = pgs_catalog_url("PGS000014", build="GRCh37")
    assert "_hmPOS_GRCh37.txt.gz" in url37
    raw = pgs_catalog_url("PGS000014", harmonised=False)
    assert raw.endswith("/ScoringFiles/PGS000014.txt.gz")
    mirror = pgs_catalog_url(
        "PGS000014", base="gs://my-mirror/pgs/scores"
    )
    assert mirror.startswith("gs://my-mirror/pgs/scores/PGS000014/")


def test_pgs_catalog_url_validates_inputs():
    import pytest
    with pytest.raises(ValueError):
        pgs_catalog_url("not-a-pgs-id")
    with pytest.raises(ValueError):
        pgs_catalog_url("PGS000014", build="hg19")


def test_discover_local_panel_finds_present_and_missing(tmp_path):
    # Touch a couple of expected score files (with the harmonised suffix
    # discover_local_panel matches by prefix, not extension).
    for pid in ("PGS003725", "PGS000018"):
        (tmp_path / f"{pid}_hmPOS_GRCh38.txt.gz").write_text("")
    found, missing = discover_local_panel(tmp_path)
    found_names = sorted(p.name for p in found)
    assert "PGS003725_hmPOS_GRCh38.txt.gz" in found_names
    assert "PGS000018_hmPOS_GRCh38.txt.gz" in found_names
    assert "PGS003725" not in missing
    assert len(missing) == 117 - 2
