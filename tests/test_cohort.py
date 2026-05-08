"""Tests for the cohort cleaning + load pipeline in ``data/cohort.py``."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_pred.data.cohort import (
    COHORT_NODES,
    COHORT_NODE_TYPES,
    PLAUSIBILITY_BOUNDS,
    attach_node_labels,
    build_cohort_dataset,
    build_survival_outcome,
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


def test_build_survival_outcome_uses_incident_t2d_and_censoring():
    person_ids = ["101", "102", "103", "104"]
    visit_baseline = pd.DataFrame(
        {
            "person_id": person_ids,
            "baseline_dt": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"]
            ),
        }
    )
    observation_period = pd.DataFrame(
        {
            "person_id": person_ids,
            "observation_start_dt": pd.to_datetime(
                ["2019-01-01", "2019-01-01", "2019-01-01", "2019-01-01"]
            ),
            "observation_end_dt": pd.to_datetime(
                ["2024-01-01", "2023-01-01", "2022-01-01", "2025-01-01"]
            ),
        }
    )
    t2d_event = pd.DataFrame(
        {
            "person_id": ["101", "103", "104"],
            "t2d_dt": pd.to_datetime(["2021-01-01", "2019-06-01", "2026-01-01"]),
        }
    )

    outcome = build_survival_outcome(
        person_ids,
        visit_baseline,
        observation_period,
        t2d_event,
    )

    assert outcome.keep.tolist() == [True, True, False, True]
    assert outcome.event.tolist() == [1, 0, 0, 0]
    assert outcome.time[0] == pytest.approx(366 / 365.25)
    assert outcome.time[1] == pytest.approx(1096 / 365.25)
    assert outcome.time[3] == pytest.approx(1827 / 365.25)
    assert outcome.meta["n_prevalent_t2d"] == 1
    assert outcome.meta["n_events"] == 1


def test_fetch_omop_long_frames_restores_workspace_parquets(tmp_path, monkeypatch):
    from causal_pred.data import cohort as cohort_mod

    class FakeClient:
        def query(self, *_args, **_kwargs):
            raise AssertionError("workspace cache should avoid BigQuery")

    fake_bigquery = types.SimpleNamespace(
        ArrayQueryParameter=lambda *args: args,
        QueryJobConfig=lambda **kwargs: kwargs,
        Client=FakeClient,
    )
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.bigquery = fake_bigquery
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_cloud
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bigquery)

    cdr = "project.dataset"
    person_ids = ["101", "102"]
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()

    def write_remote(name, payload, df):
        key = cohort_mod._omop_cache_key(
            {"cdr": cdr, "person_ids": person_ids, **payload}
        )
        path = remote_dir / f"{name}-{key}.parquet"
        cohort_mod._write_omop_cache(path, key, df)

    write_remote(
        "visit_baseline",
        {"table": "visit_occurrence", "aggregation": "baseline"},
        pd.DataFrame(
            {
                "person_id": person_ids,
                "baseline_dt": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            }
        ),
    )
    write_remote(
        "observation_period",
        {"table": "observation_period", "aggregation": "person_followup_bounds"},
        pd.DataFrame(
            {
                "person_id": person_ids,
                "observation_start_dt": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "observation_end_dt": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            }
        ),
    )
    write_remote(
        "t2d_event",
        {
            "table": "condition_occurrence",
            "condition_ids": cohort_mod.T2D_CONDITION_CONCEPT_IDS,
            "aggregation": "first_t2d_datetime_by_person",
        },
        pd.DataFrame(
            {
                "person_id": ["101"],
                "t2d_dt": pd.to_datetime(["2025-01-01"]),
            }
        ),
    )
    write_remote(
        "condition_long",
        {
            "table": "condition_occurrence",
            "condition_ids": tuple(),
            "aggregation": "min_datetime_by_person_condition",
        },
        pd.DataFrame(
            {
                "person_id": person_ids,
                "phecode": ["1", "2"],
                "datetime": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            }
        ),
    )

    restored_uris = []

    def fake_restore(uri, dst):
        restored_uris.append(uri)
        src = remote_dir / Path(uri).name
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    monkeypatch.setattr(cohort_mod, "_restore_bucket_file", fake_restore)

    frames = cohort_mod.fetch_omop_long_frames(
        person_ids,
        cdr=cdr,
        cache_dir=tmp_path / "omop",
        workspace_bucket="gs://workspace",
        workspace_prefix="intermediates/causal-pred/omop",
        condition_concept_ids=[],
    )

    assert set(frames) == {
        "visit_baseline",
        "observation_period",
        "t2d_event",
        "condition_long",
    }
    assert frames["visit_baseline"]["person_id"].tolist() == person_ids
    assert frames["observation_period"]["observation_end_dt"].dt.year.tolist() == [2026, 2026]
    assert frames["t2d_event"]["person_id"].tolist() == ["101"]
    assert frames["condition_long"]["phecode"].tolist() == ["1", "2"]
    assert all("/intermediates/causal-pred/omop/" in uri for uri in restored_uris)


def test_fetch_omop_long_frames_uploads_workspace_parquets(tmp_path, monkeypatch):
    from causal_pred.data import cohort as cohort_mod

    queries = []

    class FakeJob:
        def __init__(self, df):
            self.df = df
            self.job_id = "job-test"

        def result(self):
            return self

        def to_dataframe(self, **kwargs):
            assert kwargs == {"create_bqstorage_client": True}
            return self.df.copy()

    class FakeClient:
        def query(self, sql, **_kwargs):
            queries.append(sql)
            if "visit_occurrence" in sql:
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101", "102"],
                            "baseline_dt": pd.to_datetime(
                                ["2024-01-01", "2024-02-01"]
                            ),
                        }
                    )
                )
            if "observation_period" in sql:
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101", "102"],
                            "observation_start_dt": pd.to_datetime(
                                ["2023-01-01", "2023-01-01"]
                            ),
                            "observation_end_dt": pd.to_datetime(
                                ["2026-01-01", "2026-02-01"]
                            ),
                        }
                    )
                )
            if "condition_occurrence" in sql and "t2d_dt" in sql:
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101"],
                            "t2d_dt": pd.to_datetime(["2025-01-01"]),
                        }
                    )
                )
            raise AssertionError(sql)

    fake_bigquery = types.SimpleNamespace(
        ArrayQueryParameter=lambda *args: args,
        QueryJobConfig=lambda **kwargs: kwargs,
        Client=FakeClient,
    )
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.bigquery = fake_bigquery
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_cloud
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bigquery)

    copied = []
    monkeypatch.setattr(cohort_mod, "_gsutil_exists", lambda _uri: False)

    def fake_copy(src, dst):
        assert Path(src).is_file()
        copied.append((Path(src).name, dst))

    monkeypatch.setattr(cohort_mod, "_gsutil_copy", fake_copy)

    frames = cohort_mod.fetch_omop_long_frames(
        ["101", "102"],
        cdr="project.dataset",
        cache_dir=tmp_path / "omop",
        workspace_bucket="gs://workspace",
        workspace_prefix="intermediates/causal-pred/omop",
        fetch_conditions=False,
    )

    assert set(frames) == {"visit_baseline", "observation_period", "t2d_event"}
    assert not any("CAST(condition_concept_id AS STRING)" in q for q in queries)
    assert len(copied) == 6
    copied_names = {name for name, _dst in copied}
    assert sum(name.endswith(".parquet") for name in copied_names) == 3
    assert sum(name.endswith(".parquet.key") for name in copied_names) == 3
    assert all(
        dst.startswith("gs://workspace/intermediates/causal-pred/omop/")
        for _name, dst in copied
    )


def test_fetch_omop_long_frames_aggregates_conditions_before_download(
    tmp_path, monkeypatch
):
    from causal_pred.data import cohort as cohort_mod

    queries = []

    class FakeJob:
        def __init__(self, df):
            self.df = df

        def result(self):
            return self

        def to_dataframe(self, **kwargs):
            assert kwargs == {"create_bqstorage_client": True}
            return self.df.copy()

    class FakeClient:
        def query(self, sql, **_kwargs):
            queries.append(sql)
            if "visit_occurrence" in sql:
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101"],
                            "baseline_dt": pd.to_datetime(["2024-01-01"]),
                        }
                    )
                )
            if "observation_period" in sql:
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101"],
                            "observation_start_dt": pd.to_datetime(["2023-01-01"]),
                            "observation_end_dt": pd.to_datetime(["2025-01-01"]),
                        }
                    )
                )
            if "condition_occurrence" in sql:
                if "t2d_dt" in sql:
                    return FakeJob(
                        pd.DataFrame(
                            {
                                "person_id": ["101"],
                                "t2d_dt": pd.to_datetime(["2024-06-01"]),
                            }
                        )
                    )
                return FakeJob(
                    pd.DataFrame(
                        {
                            "person_id": ["101"],
                            "phecode": ["201820"],
                            "datetime": pd.to_datetime(["2023-01-01"]),
                        }
                    )
                )
            raise AssertionError(sql)

    fake_bigquery = types.SimpleNamespace(
        ArrayQueryParameter=lambda *args: args,
        QueryJobConfig=lambda **kwargs: kwargs,
        Client=FakeClient,
    )
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.bigquery = fake_bigquery
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_cloud
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bigquery)

    frames = cohort_mod.fetch_omop_long_frames(
        ["101"],
        cdr="project.dataset",
        cache_dir=tmp_path / "omop",
        workspace_bucket=None,
        condition_concept_ids=[201820],
    )

    condition_sql = next(
        q for q in queries
        if "condition_occurrence" in q and "CAST(condition_concept_id AS STRING)" in q
    )
    assert "MIN(COALESCE" in condition_sql
    assert "GROUP BY person_id, condition_concept_id" in condition_sql
    assert frames["condition_long"]["phecode"].tolist() == ["201820"]


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
