"""Unit tests for the curated PGS Catalog panel.

Network-free: verifies panel-set invariants (count, ID format, uniqueness),
URL helper structure, and local-discovery against a tmp dir.
"""

from __future__ import annotations

import gzip
import io
import re

import pytest

from causal_pred.genscore.panels import (
    PGS_PANEL,
    discover_local_panel,
    pgs_catalog_url,
    download_panel,
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


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


def test_download_panel_reuses_valid_cached_file(monkeypatch, tmp_path):
    cached = tmp_path / "PGS000892_hmPOS_GRCh38.txt.gz"
    payload = gzip.compress(
        b"###PGS CATALOG SCORING FILE\n"
        b"chr_name\tchr_position\teffect_allele\teffect_weight\n"
    )
    cached.write_bytes(payload)

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("valid cached score file should not be downloaded")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    paths = download_panel(tmp_path, ids=("PGS000892",), n_workers=1)

    assert paths == [cached]
    assert cached.read_bytes() == payload


def test_download_panel_replaces_invalid_cached_file(monkeypatch, tmp_path):
    cached = tmp_path / "PGS000892_hmPOS_GRCh38.txt.gz"
    cached.write_bytes(gzip.compress(b"ok\xffbroken\n"))
    replacement = gzip.compress(
        b"###PGS CATALOG SCORING FILE\n"
        b"chr_name\tchr_position\teffect_allele\teffect_weight\n"
    )

    def fake_urlopen(_url, timeout):
        assert timeout == 600
        return _FakeResponse(replacement)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    paths = download_panel(tmp_path, ids=("PGS000892",), n_workers=1)

    assert paths == [cached]
    assert gzip.decompress(cached.read_bytes()).decode("utf-8").startswith(
        "###PGS CATALOG"
    )


def test_download_panel_rejects_invalid_fresh_download(monkeypatch, tmp_path):
    invalid_payload = gzip.compress(b"bad\xffpayload\n")

    def fake_urlopen(_url, timeout):
        assert timeout == 600
        return _FakeResponse(invalid_payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="PGS downloads failed"):
        download_panel(tmp_path, ids=("PGS000892",), n_workers=1)
