"""Tests for the cohort cleaning + load pipeline in ``data/cohort.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pred.data.cohort import (
    COHORT_NODES,
    COHORT_NODE_TYPES,
    PLAUSIBILITY_BOUNDS,
    attach_node_labels,
    build_cohort_dataset,
    clean_measurements,
    collapse_to_wide_median,
    label_measurement_node,
    load_cohort_csv,
    remove_extreme_outliers_iqr,
    resolve_cohort_csv,
    write_cohort_cache,
)


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_label_measurement_node_canonical_terms():
    assert label_measurement_node("Body mass index (BMI) ratio") == "bmi"
    assert label_measurement_node("Hemoglobin A1c") == "hba1c"
    assert label_measurement_node("Glycated hemoglobin total") == "hba1c"
    assert label_measurement_node("Cholesterol in HDL [Mass/volume]") == "hdl_cholesterol"
    assert label_measurement_node("Cholesterol in LDL [Mass/volume]") == "ldl_cholesterol"
    assert label_measurement_node("Triglyceride [Mass/volume]") == "triglycerides"
    assert label_measurement_node("Fasting glucose mg/dL") == "fasting_glucose"
    assert label_measurement_node("Systolic blood pressure") == "systolic_bp"
    assert label_measurement_node("Random unrelated concept") is None


def test_label_treats_glucose_without_fasting_as_unlabelled():
    assert label_measurement_node("Glucose mass") is None


# ---------------------------------------------------------------------------
# Unit conversion + plausibility bounds
# ---------------------------------------------------------------------------


def _measurement_row(node_text, value, unit_text=""):
    """Construct a one-row OMOP-shaped measurement frame for tests."""
    return {
        "person_id": 1,
        "standard_concept_name": node_text,
        "source_concept_name": "",
        "measurement_source_value": "",
        "value_as_number": value,
        "unit_concept_name": unit_text,
        "unit_source_value": "",
        "measurement_datetime": pd.Timestamp("2020-01-01"),
    }


def test_clean_measurements_converts_mmol_per_mol_for_hba1c():
    df = pd.DataFrame(
        [_measurement_row("Hemoglobin A1c", 53.0, unit_text="mmol/mol")]
    )
    labelled = attach_node_labels(df)
    cleaned = clean_measurements(labelled)
    assert len(cleaned) == 1
    expected = 0.09148 * 53.0 + 2.152
    assert cleaned["value_clean"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_clean_measurements_converts_mmol_per_l_for_lipids():
    rows = [
        _measurement_row("Cholesterol in HDL", 1.5, unit_text="mmol/L"),
        _measurement_row("Cholesterol in LDL", 3.0, unit_text="mmol/L"),
        _measurement_row("Triglyceride [Mass/volume]", 1.2, unit_text="mmol/L"),
    ]
    cleaned = clean_measurements(attach_node_labels(pd.DataFrame(rows)))
    by_node = dict(zip(cleaned["node"], cleaned["value_clean"]))
    assert by_node["hdl_cholesterol"] == pytest.approx(1.5 * 38.67, rel=1e-9)
    assert by_node["ldl_cholesterol"] == pytest.approx(3.0 * 38.67, rel=1e-9)
    assert by_node["triglycerides"] == pytest.approx(1.2 * 88.57, rel=1e-9)


def test_clean_measurements_drops_implausible_values():
    # A BMI of 1000 is implausible; should be dropped.
    df = pd.DataFrame(
        [
            _measurement_row("Body mass index BMI", 28.0),
            _measurement_row("Body mass index BMI", 1000.0),
        ]
    )
    cleaned = clean_measurements(attach_node_labels(df))
    assert list(cleaned["value_clean"]) == [28.0]


def test_clean_measurements_keeps_mg_per_dl_unchanged():
    # If the unit is already mg/dL, no conversion should be applied.
    df = pd.DataFrame(
        [_measurement_row("Cholesterol in HDL", 50.0, unit_text="milligram per deciliter")]
    )
    cleaned = clean_measurements(attach_node_labels(df))
    assert cleaned["value_clean"].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# IQR outlier removal
# ---------------------------------------------------------------------------


def test_remove_extreme_outliers_iqr_drops_only_extreme():
    rng = np.random.default_rng(0)
    body = rng.normal(loc=100.0, scale=20.0, size=400)
    # Append two extreme values whose deviation is way beyond Q3 + 4*IQR.
    extreme = np.array([1e6, -1e6])
    values = np.concatenate([body, extreme])
    df = pd.DataFrame({"node": "ldl_cholesterol", "value_clean": values})
    cleaned, summary = remove_extreme_outliers_iqr(df)
    assert summary.loc[0, "n_removed"] == 2
    assert len(cleaned) == 400


def test_iqr_summary_columns():
    df = pd.DataFrame({"node": "bmi", "value_clean": [20.0, 25.0, 30.0, 35.0]})
    _, summary = remove_extreme_outliers_iqr(df)
    for col in ("node", "n_before", "n_removed", "n_after", "q1", "q3", "iqr"):
        assert col in summary.columns


# ---------------------------------------------------------------------------
# Wide pivot
# ---------------------------------------------------------------------------


def test_collapse_to_wide_median_uses_median():
    df = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2],
            "node": ["bmi", "bmi", "bmi", "hba1c"],
            "value_clean": [25.0, 27.0, 29.0, 6.0],
        }
    )
    wide = collapse_to_wide_median(df)
    row1 = wide.loc[wide["person_id"] == 1].iloc[0]
    assert row1["bmi"] == pytest.approx(27.0)
    row2 = wide.loc[wide["person_id"] == 2].iloc[0]
    assert row2["hba1c"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def _make_synthetic_omop_frames(n_t2d=30, n_control=70, rng=None):
    """Build small OMOP-shaped condition + measurement frames for tests."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = n_t2d + n_control
    pids = np.arange(1000, 1000 + n)
    cond = pd.DataFrame(
        {
            "person_id": pids[:n_t2d],
            "condition_concept_id": 201826,
            "condition_start_datetime": pd.Timestamp("2018-01-01"),
        }
    )
    rows = []
    for pid in pids:
        rows += [
            {
                "person_id": pid,
                "standard_concept_name": "Body mass index (BMI) ratio",
                "source_concept_name": "",
                "measurement_source_value": "",
                "value_as_number": float(rng.normal(28.0, 5.0)),
                "unit_concept_name": "kg/m2",
                "unit_source_value": "kg/m2",
                "measurement_datetime": pd.Timestamp("2020-01-01"),
            },
            {
                "person_id": pid,
                "standard_concept_name": "Hemoglobin A1c",
                "source_concept_name": "",
                "measurement_source_value": "",
                "value_as_number": float(rng.normal(6.0, 1.0)),
                "unit_concept_name": "%",
                "unit_source_value": "%",
                "measurement_datetime": pd.Timestamp("2020-01-01"),
            },
            {
                "person_id": pid,
                "standard_concept_name": "Cholesterol in HDL",
                "source_concept_name": "",
                "measurement_source_value": "",
                "value_as_number": float(rng.normal(50.0, 10.0)),
                "unit_concept_name": "mg/dL",
                "unit_source_value": "mg/dL",
                "measurement_datetime": pd.Timestamp("2020-01-01"),
            },
            {
                "person_id": pid,
                "standard_concept_name": "Cholesterol in LDL",
                "source_concept_name": "",
                "measurement_source_value": "",
                "value_as_number": float(rng.normal(100.0, 25.0)),
                "unit_concept_name": "mg/dL",
                "unit_source_value": "mg/dL",
                "measurement_datetime": pd.Timestamp("2020-01-01"),
            },
            {
                "person_id": pid,
                "standard_concept_name": "Triglyceride [Mass/volume]",
                "source_concept_name": "",
                "measurement_source_value": "",
                "value_as_number": float(abs(rng.normal(120.0, 50.0))),
                "unit_concept_name": "mg/dL",
                "unit_source_value": "mg/dL",
                "measurement_datetime": pd.Timestamp("2020-01-01"),
            },
        ]
    meas = pd.DataFrame(rows)
    return cond, meas


def test_build_cohort_dataset_end_to_end():
    cond, meas = _make_synthetic_omop_frames()
    result = build_cohort_dataset(meas, cond)
    expected_cols = (
        "type2_diabetes",
        "bmi",
        "hba1c",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    )
    assert tuple(result.nodes) == expected_cols
    assert "type2_diabetes" in result.wide.columns
    # Every row has a 0/1 T2D label.
    assert result.wide["type2_diabetes"].isin({0, 1}).all()
    # T2D rate matches the input (30 of 100 in the cohort).
    assert int(result.wide["type2_diabetes"].sum()) == 30
    # Complete-case has no NaNs in cohort columns.
    assert not result.complete[list(result.nodes)].isna().any().any()
    # Node-types tuple aligns with the wide-frame order.
    for col, kind in zip(result.nodes, result.node_types):
        assert kind == COHORT_NODE_TYPES[col]


# ---------------------------------------------------------------------------
# CSV load
# ---------------------------------------------------------------------------


def test_load_cohort_csv_standardises_continuous(tmp_path):
    df = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type2_diabetes": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "bmi": [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
            "hba1c": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            "hdl_cholesterol": [40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0],
            "ldl_cholesterol": [
                80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0,
            ],
            "triglycerides": [
                100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0,
            ],
        }
    )
    p = tmp_path / "cohort.csv"
    df.to_csv(p, index=False)

    X, columns, node_types = load_cohort_csv(str(p))
    assert columns == (
        "type2_diabetes",
        "bmi",
        "hba1c",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    )
    assert node_types == ("binary", "continuous", "continuous", "continuous", "continuous", "continuous")
    assert X.shape == (10, 6)
    # Binary column stays {0, 1}.
    bin_idx = columns.index("type2_diabetes")
    assert set(np.unique(X[:, bin_idx]).tolist()) == {0.0, 1.0}
    # Continuous columns are z-scored: mean ~ 0, std ~ 1.
    for j, kind in enumerate(node_types):
        if kind == "continuous":
            assert X[:, j].mean() == pytest.approx(0.0, abs=1e-9)
            assert X[:, j].std() == pytest.approx(1.0, abs=1e-9)


def test_load_cohort_csv_drops_incomplete_rows(tmp_path):
    df = pd.DataFrame(
        {
            "type2_diabetes": [0, 1, 0],
            "bmi": [22.0, np.nan, 30.0],
            "hba1c": [5.0, 6.0, 7.0],
        }
    )
    p = tmp_path / "with_nan.csv"
    df.to_csv(p, index=False)
    X, _, _ = load_cohort_csv(str(p))
    assert X.shape[0] == 2  # the NaN row is dropped


def test_load_cohort_csv_rejects_unrelated_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = tmp_path / "unrelated.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="cohort columns"):
        load_cohort_csv(str(p))


# ---------------------------------------------------------------------------
# Caching: local-first lookup, build-on-miss, notebook filename variants
# ---------------------------------------------------------------------------


def test_resolve_cohort_csv_local_hit(tmp_path):
    p = tmp_path / "t2d_initial_nodes_complete.csv"
    p.write_text("type2_diabetes,bmi\n0,22\n")
    resolved = resolve_cohort_csv(name="complete", cache_dir=tmp_path, bucket=None)
    assert resolved == p


def test_resolve_cohort_csv_accepts_complete_case_notebook_name(tmp_path):
    p = tmp_path / "t2d_initial_nodes_complete_case.csv"
    p.write_text("type2_diabetes,bmi\n1,30\n")
    resolved = resolve_cohort_csv(name="complete", cache_dir=tmp_path, bucket=None)
    assert resolved == p


def test_resolve_cohort_csv_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_cohort_csv(name="complete", cache_dir=tmp_path, bucket=None)


def test_write_cohort_cache_round_trip(tmp_path):
    cond, meas = _make_synthetic_omop_frames()
    result = build_cohort_dataset(meas, cond)
    pl, cc = write_cohort_cache(result, cache_dir=tmp_path, upload=False)
    assert pl.exists()
    assert cc.exists()
    # Resolver finds the freshly-cached file without touching the bucket.
    found = resolve_cohort_csv(name="complete", cache_dir=tmp_path, bucket=None)
    assert found == cc


# ---------------------------------------------------------------------------
# Cohort module is a strict subset of the full node taxonomy
# ---------------------------------------------------------------------------


def test_plausibility_bounds_cover_every_continuous_cohort_node():
    for node, kind in COHORT_NODE_TYPES.items():
        if kind == "continuous":
            assert node in PLAUSIBILITY_BOUNDS, node


def test_cohort_nodes_have_known_types():
    for node in COHORT_NODES:
        assert node in COHORT_NODE_TYPES
