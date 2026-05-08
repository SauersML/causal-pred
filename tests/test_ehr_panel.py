"""Tests for the EHR panel builder.

These tests use synthetic OMOP-shaped frames (``person_id``, group label,
datetime; plus BigQuery-style lab summary rows) to verify:

* baseline-strict censoring (events on or after baseline never appear),
* prevalence filtering (rare groups are dropped),
* lab summary statistics (mean / min / max / slope) with sign of slope,
* utilisation count from a visit frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_pred.data.cohort import EhrPanel, build_ehr_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _baseline_series(person_ids, baseline_dates):
    return pd.Series(
        pd.to_datetime(baseline_dates),
        index=[str(p) for p in person_ids],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_baseline_strict_censoring_drops_post_baseline_events():
    """Events on or after baseline must never appear in the indicator panel."""
    person_ids = [f"p{i:03d}" for i in range(100)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * 100)
    # Each person has one pre-baseline condition and one post-baseline.
    rows = []
    for i, p in enumerate(person_ids):
        rows.append({"person_id": p, "phecode": "PHE_A", "datetime": "2024-01-01"})
        rows.append({"person_id": p, "phecode": "PHE_B", "datetime": "2025-06-01"})
    cond = pd.DataFrame(rows)
    panel = build_ehr_panel(
        person_ids,
        baseline,
        condition_long=cond,
        min_prevalence=10,
        lookback_days=None,
    )
    assert isinstance(panel, EhrPanel)
    assert "cond:PHE_A" in panel.feature_names
    assert "cond:PHE_B" not in panel.feature_names
    j = panel.feature_names.index("cond:PHE_A")
    assert (panel.matrix[:, j] == 1.0).all()


def test_min_prevalence_filters_rare_groups():
    """Drug groups touched by fewer than min_prevalence persons are dropped."""
    person_ids = [f"p{i:03d}" for i in range(100)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * 100)
    rows = []
    # Common: 60 persons receive metformin
    for p in person_ids[:60]:
        rows.append({"person_id": p, "atc_class": "A10BA", "datetime": "2024-06-01"})
    # Rare: 5 persons receive a niche drug
    for p in person_ids[:5]:
        rows.append({"person_id": p, "atc_class": "X99XX", "datetime": "2024-06-01"})
    drug = pd.DataFrame(rows)
    panel = build_ehr_panel(
        person_ids,
        baseline,
        drug_long=drug,
        min_prevalence=10,
    )
    assert "drug:A10BA" in panel.feature_names
    assert "drug:X99XX" not in panel.feature_names


def test_lab_summary_slope_sign_matches_truth():
    """Per-(person, lab) aggregate slopes should flow into the feature matrix."""
    rng = np.random.default_rng(0)
    n = 80
    person_ids = [f"p{i:03d}" for i in range(n)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * n)
    rows = []
    # Half the people: rising HbA1c. Other half: falling.
    for i, p in enumerate(person_ids):
        slope_yr = +0.5 if i % 2 == 0 else -0.5
        measurement_values = []
        for years_ago in (4, 3, 2, 1):
            value = 5.5 - slope_yr * years_ago + rng.normal(scale=0.05)
            measurement_values.append(value)
        values = np.asarray(measurement_values, dtype=float)
        times = -np.asarray([4.0, 3.0, 2.0, 1.0], dtype=float)
        tc = times - times.mean()
        slope = float(np.sum(tc * (values - values.mean())) / np.sum(tc * tc))
        rows.append(
            {
                "person_id": p,
                "lab": "hba1c",
                "value_mean": float(values.mean()),
                "value_min": float(values.min()),
                "value_max": float(values.max()),
                "value_slope": slope,
                "n_measurements": int(values.size),
            }
        )
    summary = pd.DataFrame(rows)
    panel = build_ehr_panel(
        person_ids,
        baseline,
        measurement_summary=summary,
        min_lab_observations=5,
    )
    j_slope = panel.feature_names.index("lab_slope:hba1c")
    j_min = panel.feature_names.index("lab_min:hba1c")
    j_max = panel.feature_names.index("lab_max:hba1c")
    # Even-indexed people have positive slope (rising), odd negative.
    even = np.arange(n) % 2 == 0
    assert (panel.matrix[even, j_slope] > 0).mean() > 0.9
    assert (panel.matrix[~even, j_slope] < 0).mean() > 0.9
    # min < max columnwise, sanity.
    assert np.all(panel.matrix[:, j_min] <= panel.matrix[:, j_max] + 1e-9)


def test_lab_summary_accepts_bigquery_aggregates():
    """AoU production lab features arrive as per-person summary rows."""
    person_ids = [f"p{i:03d}" for i in range(60)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * len(person_ids))
    summary = pd.DataFrame(
        {
            "person_id": person_ids[:55] + person_ids[:5],
            "lab": ["hba1c"] * 55 + ["rare_lab"] * 5,
            "value_mean": [6.0 + i * 0.01 for i in range(55)] + [1.0] * 5,
            "value_min": [5.5 + i * 0.01 for i in range(55)] + [0.5] * 5,
            "value_max": [6.5 + i * 0.01 for i in range(55)] + [1.5] * 5,
            "value_slope": [0.1] * 10 + [np.nan] * 45 + [0.0] * 5,
            "n_measurements": [4] * 60,
        }
    )

    panel = build_ehr_panel(
        person_ids,
        baseline,
        measurement_summary=summary,
        min_lab_observations=50,
    )

    assert "lab_mean:hba1c" in panel.feature_names
    assert "lab_mean:rare_lab" not in panel.feature_names
    j_mean = panel.feature_names.index("lab_mean:hba1c")
    j_slope = panel.feature_names.index("lab_slope:hba1c")
    assert panel.matrix[0, j_mean] == pytest.approx(6.0)
    assert np.isfinite(panel.matrix[:, j_slope]).all()
    assert panel.matrix[20, j_slope] == pytest.approx(0.1)


def test_utilisation_counts_pre_baseline_encounters_only():
    """visit_long should yield a single utilisation column with strict counts."""
    person_ids = [f"p{i:03d}" for i in range(40)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * 40)
    rows = []
    for i, p in enumerate(person_ids):
        # i pre-baseline encounters, plus a few post-baseline (must be ignored).
        for _ in range(i):
            rows.append({"person_id": p, "datetime": "2024-06-01"})
        for _ in range(3):
            rows.append({"person_id": p, "datetime": "2025-07-01"})
    visits = pd.DataFrame(rows)
    panel = build_ehr_panel(
        person_ids,
        baseline,
        visit_long=visits,
    )
    j = panel.feature_names.index("utilisation:n_encounters")
    expected = np.arange(40, dtype=float)
    np.testing.assert_array_equal(panel.matrix[:, j], expected)


def test_lookback_days_window_filters_old_events():
    """Events older than baseline - lookback_days should be dropped."""
    person_ids = [f"p{i:03d}" for i in range(20)]
    baseline = _baseline_series(person_ids, ["2025-01-01"] * 20)
    rows = []
    for p in person_ids:
        # Old (10 years before baseline): should be dropped with lookback=2y
        rows.append({"person_id": p, "phecode": "OLD", "datetime": "2015-01-01"})
        # Recent (1 year before baseline): kept
        rows.append({"person_id": p, "phecode": "NEW", "datetime": "2024-01-01"})
    cond = pd.DataFrame(rows)
    panel = build_ehr_panel(
        person_ids,
        baseline,
        condition_long=cond,
        min_prevalence=5,
        lookback_days=365 * 2,
    )
    assert "cond:NEW" in panel.feature_names
    assert "cond:OLD" not in panel.feature_names
