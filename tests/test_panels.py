"""Unit tests for the curated PGS Catalog panel.

Network-free: verifies panel-set invariants (count, ID format, uniqueness),
URL helper structure, and local-discovery against a tmp dir.
"""

from __future__ import annotations

import re

from causal_pred.genscore.panels import (
    PGS_PANEL,
    discover_local_panel,
    pgs_catalog_url,
)


_PGS_RE = re.compile(r"^PGS\d{6}$")


def test_panel_has_117_unique_well_formed_ids():
    assert len(PGS_PANEL) == 117
    assert len(set(PGS_PANEL)) == 117
    for pgs_id in PGS_PANEL:
        assert _PGS_RE.match(pgs_id), f"{pgs_id!r} does not match PGS\\d{{6}}"


def test_pgs_catalog_url_structure():
    url = pgs_catalog_url("PGS000014")
    assert url == (
        "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000014/"
        "ScoringFiles/Harmonized/PGS000014_hmPOS_GRCh38.txt.gz"
    )
    assert "_hmPOS_GRCh37.txt.gz" in pgs_catalog_url("PGS000014", build="GRCh37")


def test_discover_local_panel_finds_present_and_missing(tmp_path):
    for pid in ("PGS003725", "PGS000018"):
        (tmp_path / f"{pid}_hmPOS_GRCh38.txt.gz").write_text("")
    found, missing = discover_local_panel(tmp_path)
    found_names = sorted(p.name for p in found)
    assert "PGS003725_hmPOS_GRCh38.txt.gz" in found_names
    assert "PGS000018_hmPOS_GRCh38.txt.gz" in found_names
    assert "PGS003725" not in missing
    assert len(missing) == 117 - 2
