"""Tests for the gnomon-CLI wrapper (``causal_pred.data.polygenic``).

Tests that would actually invoke the gnomon binary are skipped when the
binary is not installed; the rest exercise the TSV / binary parsers using
hand-constructed fixtures in ``tmp_path``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_pred.data import polygenic as pg
from causal_pred.data.synthetic import simulate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

HAS_GNOMON = pg.gnomon_available()


# ---------------------------------------------------------------------------
# binary presence / absence
# ---------------------------------------------------------------------------


def test_binary_present_or_skipped(monkeypatch):
    """If gnomon is installed, ``gnomon --version`` must run cleanly; if it
    isn't, every public function raises :class:`PolygenicToolMissing`."""
    if HAS_GNOMON:
        import subprocess

        result = subprocess.run(
            [pg._locate_gnomon(), "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "gnomon" in (result.stdout + result.stderr).lower()
        return

    # Force the "missing" code path even on a machine that has gnomon.
    monkeypatch.setattr(pg.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(pg.os.path, "isfile", lambda *_a, **_k: False)

    with pytest.raises(pg.PolygenicToolMissing):
        pg.score_cohort("/nonexistent", ["/nonexistent.score"])
    with pytest.raises(pg.PolygenicToolMissing):
        pg.fit_pca("/nonexistent")
    with pytest.raises(pg.PolygenicToolMissing):
        pg.project_pca("/nonexistent", "/nonexistent.json")
    with pytest.raises(pg.PolygenicToolMissing):
        pg.infer_terms("/nonexistent")


def test_missing_binary_raises_even_when_installed(monkeypatch, tmp_path):
    """Regardless of the real environment, shadowing ``shutil.which`` must
    make the wrappers raise ``PolygenicToolMissing`` before any subprocess."""
    monkeypatch.setattr(pg.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(pg.os.path, "isfile", lambda *_a, **_k: False)

    fake_geno = tmp_path / "fake.vcf"
    fake_geno.write_text("dummy")
    fake_score = tmp_path / "fake.score"
    fake_score.write_text("dummy")

    with pytest.raises(pg.PolygenicToolMissing):
        pg.score_cohort(str(fake_geno), [str(fake_score)])


# ---------------------------------------------------------------------------
# TSV / binary parsing (no subprocess)
# ---------------------------------------------------------------------------


def _write_mock_sscore(path: Path, iids, avg_cols, values, missing_pct=None):
    """Write a hand-crafted ``.sscore`` TSV mimicking gnomon's output."""
    with path.open("w") as fh:
        header = ["#IID"]
        for c in avg_cols:
            header.append(f"{c}_AVG")
            header.append(f"{c}_MISSING_PCT")
        fh.write("\t".join(header) + "\n")
        for i, iid in enumerate(iids):
            row = [iid]
            for j in range(len(avg_cols)):
                row.append(f"{values[i][j]:.6f}")
                row.append(f"{(missing_pct or 0.0):.4f}")
            fh.write("\t".join(row) + "\n")


def test_tsv_parsing_sscore(tmp_path):
    sscore = tmp_path / "cohort.sscore"
    _write_mock_sscore(
        sscore,
        iids=["S1", "S2", "S3"],
        avg_cols=["T2D", "BMI"],
        values=[[0.12, -0.34], [0.56, 0.78], [-1.0, 2.5]],
    )
    df = pg.parse_sscore(sscore)
    assert list(df.columns) == ["T2D", "BMI"]
    assert df.index.name == "IID"
    assert df.shape == (3, 2)
    assert df.loc["S1", "T2D"] == pytest.approx(0.12)
    assert df.loc["S3", "BMI"] == pytest.approx(2.5)
    # dtypes
    assert df["T2D"].dtype == np.float64
    assert str(df.index.dtype) in ("string", "object", "string[python]")


def test_tsv_parsing_sscore_with_region_header(tmp_path):
    """Leading ``#REGION`` rows must be skipped."""
    sscore = tmp_path / "cohort.sscore"
    lines = [
        "#REGION\tSCORE\tINTERVAL",
        "#REGION\tT2D\tchr22:16e6-51e6",
        "#IID\tT2D_AVG\tT2D_MISSING_PCT",
        "S1\t0.25\t0.01",
        "S2\t-0.50\t0.02",
    ]
    sscore.write_text("\n".join(lines) + "\n")
    df = pg.parse_sscore(sscore)
    assert df.shape == (2, 1)
    assert df.loc["S2", "T2D"] == pytest.approx(-0.5)


def test_tsv_parsing_sex(tmp_path):
    """``sex.tsv`` is parsed with numeric diagnostic columns as float64."""
    tsv = tmp_path / "sex.tsv"
    header = (
        "IID\tBuild\tSex\tY_Density\tX_AutoHet_Ratio\tComposite_Index\t"
        "Auto_Valid\tAuto_Het\tX_NonPAR_Valid\tX_NonPAR_Het\tY_NonPAR_Valid\tY_PAR_Valid"
    )
    rows = [
        "F1\tBuild38\tfemale\t0.001\t0.42\t-0.8\t100\t40\t20\t8\t0\t0",
        "M1\tBuild38\tmale\t0.88\t0.03\t1.5\t100\t38\t22\t0\t5\t0",
        "U1\tBuild38\tunknown\tNA\tNA\tNA\t10\t3\t5\t1\t0\t0",
    ]
    tsv.write_text(header + "\n" + "\n".join(rows) + "\n")

    df = pg.parse_sex_tsv(tsv)
    assert df.index.name == "IID"
    assert set(df["Sex"]) == {"female", "male", "unknown"}
    assert df["Y_Density"].dtype == np.float64
    assert np.isnan(df.loc["U1", "Composite_Index"])
    assert df.loc["M1", "Y_Density"] == pytest.approx(0.88)


def _write_mock_projection_bin(path: Path, iids, matrix):
    """Write a projection_scores.bin file matching gnomon's on-disk layout."""
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = matrix.shape
    with path.open("wb") as fh:
        fh.write(b"GNPRJ001")
        fh.write(struct.pack("<I", 3))  # version
        fh.write(struct.pack("<Q", rows))
        fh.write(struct.pack("<Q", cols))
        fh.write(struct.pack("<I", 0))  # reserved
        # column-major payload
        fh.write(np.asfortranarray(matrix).tobytes(order="F"))
        # row-id section
        fh.write(b"GNPSID01")
        fh.write(struct.pack("<I", 1))  # version
        fh.write(struct.pack("<I", 0))  # padding up to offset 16
        fh.write(struct.pack("<Q", len(iids)))  # count
        # build offsets + string table
        encoded = [s.encode("utf-8") for s in iids]
        offsets = [0]
        for e in encoded:
            offsets.append(offsets[-1] + len(e))
        str_bytes = offsets[-1]
        fh.write(struct.pack("<Q", str_bytes))  # string_bytes
        for off in offsets:
            fh.write(struct.pack("<Q", off))
        for e in encoded:
            fh.write(e)


def test_binary_parsing_projection(tmp_path):
    bin_path = tmp_path / "projection_scores.bin"
    iids = ["NA12878", "NA12879", "HG00096"]
    matrix = np.array(
        [
            [0.10, -0.20, 0.30, 0.40],
            [1.10, 1.20, 1.30, 1.40],
            [-2.0, -3.0, -4.0, -5.0],
        ],
        dtype=np.float64,
    )
    _write_mock_projection_bin(bin_path, iids, matrix)

    df = pg.parse_projection_bin(bin_path)
    assert df.shape == (3, 4)
    assert list(df.columns) == ["PC1", "PC2", "PC3", "PC4"]
    assert list(df.index) == iids
    for c in df.columns:
        assert df[c].dtype == np.float64
    np.testing.assert_allclose(df.to_numpy(), matrix)


def test_binary_parsing_projection_bad_magic(tmp_path):
    bin_path = tmp_path / "projection_scores.bin"
    bin_path.write_bytes(b"NOT_GNOMON" + b"\x00" * 64)
    with pytest.raises(pg.PolygenicRunError):
        pg.parse_projection_bin(bin_path)


# ---------------------------------------------------------------------------
# synthetic-dataset augmentation
# ---------------------------------------------------------------------------


def test_augment_synthetic_replaces_and_preserves():
    ds = simulate(n=256, rng=np.random.default_rng(0))
    # Build a pgs_df with wildly different scales so we can confirm z-scoring.
    rng = np.random.default_rng(1)
    pgs_df = pd.DataFrame(
        {
            "t2d_raw": rng.normal(100.0, 25.0, size=ds.n),
            "bmi_raw": rng.normal(-50.0, 5.0, size=ds.n),
            "ignored": rng.normal(0.0, 1.0, size=ds.n),
        }
    )

    new_ds = pg.augment_synthetic_with_real_pgs(
        ds,
        pgs_df,
        pgs_map={"PGS_T2D": "t2d_raw", "PGS_BMI": "bmi_raw"},
    )

    # shape / metadata preserved
    assert new_ds.X.shape == ds.X.shape
    assert new_ds.columns == ds.columns
    assert new_ds.node_types == ds.node_types
    np.testing.assert_array_equal(new_ds.ground_truth_adj, ds.ground_truth_adj)
    np.testing.assert_array_equal(new_ds.time, ds.time)
    np.testing.assert_array_equal(new_ds.event, ds.event)

    # replaced columns are standardised
    t2d_idx = ds.columns.index("PGS_T2D")
    bmi_idx = ds.columns.index("PGS_BMI")
    assert abs(new_ds.X[:, t2d_idx].mean()) < 1e-10
    assert abs(new_ds.X[:, t2d_idx].std() - 1.0) < 1e-6
    assert abs(new_ds.X[:, bmi_idx].mean()) < 1e-10

    # unrelated PGS columns are untouched
    ldl_idx = ds.columns.index("PGS_LDL")
    hba_idx = ds.columns.index("PGS_HbA1c")
    np.testing.assert_array_equal(new_ds.X[:, ldl_idx], ds.X[:, ldl_idx])
    np.testing.assert_array_equal(new_ds.X[:, hba_idx], ds.X[:, hba_idx])

    # a non-PGS phenotype column is untouched
    bmi_phe_idx = ds.columns.index("BMI")
    np.testing.assert_array_equal(new_ds.X[:, bmi_phe_idx], ds.X[:, bmi_phe_idx])

    # input dataset was not mutated
    assert new_ds.X is not ds.X


def test_augment_synthetic_validates_arguments():
    ds = simulate(n=64, rng=np.random.default_rng(0))
    df = pd.DataFrame({"real": np.zeros(ds.n)})

    with pytest.raises(KeyError):
        pg.augment_synthetic_with_real_pgs(ds, df, {"not_a_col": "real"})

    with pytest.raises(KeyError):
        pg.augment_synthetic_with_real_pgs(ds, df, {"PGS_T2D": "missing"})

    short = pd.DataFrame({"real": np.zeros(ds.n - 1)})
    with pytest.raises(ValueError):
        pg.augment_synthetic_with_real_pgs(ds, short, {"PGS_T2D": "real"})


# ---------------------------------------------------------------------------
# tests that actually invoke gnomon (skipped if not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GNOMON, reason="gnomon binary not installed")
def test_gnomon_version_runs_cleanly():
    """The smoke test from the teammate spec: a real invocation succeeds."""
    import subprocess

    result = subprocess.run(
        [pg._locate_gnomon(), "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_check_runs(capsys, monkeypatch):
    """The ``check`` CLI sub-command reports whether gnomon is installed."""
    rc = pg._main(["check"])
    out = capsys.readouterr().out
    if HAS_GNOMON:
        assert rc == 0
        assert "gnomon" in out
    else:
        assert rc == 1
        assert "NOT FOUND" in out
