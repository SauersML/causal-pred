"""Cohort-data loader and cleaner for the unified causal-prediction pipeline.

This module is the cohort-data branch of :func:`causal_pred.pipeline.run_pipeline`'s
single data-loading entry point. Given OMOP-shaped condition + measurement
frames it produces a wide participant-level table; given the resulting
wide CSV it produces a :class:`causal_pred.data.synthetic.SyntheticDataset`
ready to flow through MrDAG -> DAGSLAM -> structure MCMC -> GAM unchanged.

Cleaning steps (mirroring the upstream notebook):

  1. label measurement rows with the DAG node they belong to,
  2. normalise units to a single canonical unit per node,
  3. drop physiologically impossible values (broad plausibility bounds),
  4. drop extreme outliers via a per-node IQR rule (multiplier 4.0),
  5. collapse repeated measurements to one value per participant via the
     median,
  6. derive the binary T2D node from the condition table (presence
     anywhere in the cohort -> 1, otherwise 0),
  7. merge into a single wide frame.

The 7-node cohort schema (``type2_diabetes``, ``bmi``, ``hba1c``,
``hdl_cholesterol``, ``ldl_cholesterol``, ``triglycerides``,
``systolic_bp``) is intentionally a subset of the 18-node synthetic
schema in :mod:`causal_pred.data.nodes` -- it captures what is
typically extractable from biobank lab and condition tables. Both
schemas share the same downstream contract: a
:class:`SyntheticDataset` with ``X``, ``columns``, ``node_types``,
``time``, ``event`` and ``ground_truth_adj`` fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CACHE_FILENAMES: dict = {
    "participant_level": ("t2d_initial_nodes_participant_level.csv",),
    "complete": (
        "t2d_initial_nodes_complete.csv",
        "t2d_initial_nodes_complete_case.csv",
    ),
}


COHORT_NODES: Tuple[str, ...] = (
    "type2_diabetes",
    "age",
    "sex",
    "ancestry_pc1",
    "family_history_t2d",
    "years_smoking",
    "current_smoker",
    "physical_activity",
    "diet_quality",
    "healthcare_access",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "systolic_bp",
    "hypertension",
    "cardiovascular_disease",
)

COHORT_NODE_TYPES: dict = {
    "type2_diabetes": "binary",
    "age": "continuous",
    "sex": "binary",
    "ancestry_pc1": "continuous",
    "family_history_t2d": "binary",
    "years_smoking": "continuous",
    "current_smoker": "binary",
    "physical_activity": "continuous",
    "diet_quality": "continuous",
    "healthcare_access": "continuous",
    "bmi": "continuous",
    "hba1c": "continuous",
    "hdl_cholesterol": "continuous",
    "ldl_cholesterol": "continuous",
    "triglycerides": "continuous",
    "systolic_bp": "continuous",
    "fasting_glucose": "continuous",
    "hypertension": "binary",
    "cardiovascular_disease": "binary",
}

SURVIVAL_TIME_COLUMNS: Tuple[str, ...] = (
    "time",
    "followup_years",
    "follow_up_years",
    "followup_time",
    "t2d_time",
    "time_to_t2d",
    "survival_time",
)

SURVIVAL_EVENT_COLUMNS: Tuple[str, ...] = (
    "event",
    "t2d_event",
    "type2_diabetes_event",
)

T2D_CONDITION_CONCEPT_IDS: Tuple[int, ...] = (
    201826,
    201820,
    443767,
    443729,
    442793,
    443592,
    435216,
    376112,
)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)

# Canonical units after :func:`clean_measurements`:
#   bmi              kg/m^2
#   hba1c            percent (NGSP)
#   fasting_glucose  mg/dL
#   hdl_cholesterol  mg/dL
#   ldl_cholesterol  mg/dL
#   triglycerides    mg/dL
#   systolic_bp      mmHg
PLAUSIBILITY_BOUNDS: dict = {
    "age": (18.0, 120.0),
    "ancestry_pc1": (-20.0, 20.0),
    "years_smoking": (0.0, 90.0),
    "physical_activity": (0.0, 250.0),
    "diet_quality": (-10.0, 10.0),
    "healthcare_access": (-10.0, 10.0),
    "bmi": (10.0, 100.0),
    "hba1c": (3.0, 20.0),
    "fasting_glucose": (30.0, 600.0),
    "hdl_cholesterol": (5.0, 200.0),
    "ldl_cholesterol": (5.0, 400.0),
    "triglycerides": (10.0, 2000.0),
    "systolic_bp": (60.0, 260.0),
}


# ---------------------------------------------------------------------------
# Step 1: label measurement rows by DAG node
# ---------------------------------------------------------------------------


def label_measurement_node(text: str) -> Optional[str]:
    """Return the DAG node a measurement row belongs to, or ``None``.

    ``text`` is a free-text concatenation of the measurement's standard
    concept name, source concept name, and source value. Matching is
    case-insensitive and order-independent within a given concept.
    """
    s = str(text).lower()
    if "body mass index" in s or s.strip() == "bmi" or " bmi" in s:
        return "bmi"
    if (
        "a1c" in s
        or "hba1c" in s
        or "hemoglobin a1c" in s
        or "glycated hemoglobin" in s
    ):
        return "hba1c"
    if "glucose" in s and ("fasting" in s or "fasted" in s):
        return "fasting_glucose"
    if "hdl" in s or "high density lipoprotein" in s:
        return "hdl_cholesterol"
    if "ldl" in s or "low density lipoprotein" in s:
        return "ldl_cholesterol"
    if "triglyceride" in s:
        return "triglycerides"
    if "systolic" in s:
        return "systolic_bp"
    return None


def attach_node_labels(measurement_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``concept_text`` and ``node`` columns to a measurement frame.

    Expects standard OMOP measurement columns
    (``standard_concept_name``, ``source_concept_name``,
    ``measurement_source_value``, ``value_as_number``, ``unit_concept_name``,
    ``unit_source_value``). Missing string columns are tolerated.
    """
    out = measurement_df.copy()
    out["measurement_datetime"] = pd.to_datetime(
        out.get("measurement_datetime"), errors="coerce"
    )
    out["value_as_number"] = pd.to_numeric(out["value_as_number"], errors="coerce")

    for col in ("standard_concept_name", "source_concept_name", "measurement_source_value"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)

    out["concept_text"] = (
        out["standard_concept_name"].fillna("")
        + " "
        + out["source_concept_name"].fillna("")
        + " "
        + out["measurement_source_value"].fillna("")
    ).str.lower()

    out["node"] = out["concept_text"].apply(label_measurement_node)
    return out


# ---------------------------------------------------------------------------
# Step 2: unit normalisation + plausibility bounds
# ---------------------------------------------------------------------------


def _unit_text(df: pd.DataFrame) -> pd.Series:
    a = df.get("unit_concept_name", pd.Series(index=df.index, dtype=object)).fillna("")
    b = df.get("unit_source_value", pd.Series(index=df.index, dtype=object)).fillna("")
    return (a.astype(str) + " " + b.astype(str)).str.lower()


def clean_measurements(measurement_df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-canonical units, then drop physiologically impossible rows.

    Input must already have a ``node`` column (use :func:`attach_node_labels`)
    and a numeric ``value_as_number``. Returns a new frame with a
    ``value_clean`` column in canonical units, restricted to rows whose
    cleaned value lies inside :data:`PLAUSIBILITY_BOUNDS`.
    """
    df = measurement_df.loc[
        measurement_df["node"].notna() & measurement_df["value_as_number"].notna()
    ].copy()

    df["unit_text"] = _unit_text(df)
    df["value_clean"] = df["value_as_number"].astype(float)

    is_mmol_per_mol = df["unit_text"].str.contains("mmol/mol", na=False)
    mask = (df["node"] == "hba1c") & is_mmol_per_mol
    df.loc[mask, "value_clean"] = 0.09148 * df.loc[mask, "value_as_number"] + 2.152

    is_mmol_per_l = df["unit_text"].str.contains("mmol/l|millimole", regex=True, na=False)
    mask = (df["node"] == "fasting_glucose") & is_mmol_per_l
    df.loc[mask, "value_clean"] = df.loc[mask, "value_as_number"] * 18.0182

    for node in ("hdl_cholesterol", "ldl_cholesterol"):
        mask = (df["node"] == node) & is_mmol_per_l
        df.loc[mask, "value_clean"] = df.loc[mask, "value_as_number"] * 38.67

    mask = (df["node"] == "triglycerides") & is_mmol_per_l
    df.loc[mask, "value_clean"] = df.loc[mask, "value_as_number"] * 88.57

    def _within(row):
        bounds = PLAUSIBILITY_BOUNDS.get(row["node"])
        if bounds is None:
            return True
        lo, hi = bounds
        return lo <= row["value_clean"] <= hi

    return df.loc[df.apply(_within, axis=1)].copy()


# ---------------------------------------------------------------------------
# Step 3: per-node IQR outlier removal
# ---------------------------------------------------------------------------


def remove_extreme_outliers_iqr(
    df: pd.DataFrame,
    node_col: str = "node",
    value_col: str = "value_clean",
    iqr_multiplier: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows whose value lies outside ``[Q1 - k*IQR, Q3 + k*IQR]`` per node.

    ``iqr_multiplier=4.0`` is intentionally conservative (vs. the textbook
    1.5): it only trims values implausible enough to be data-entry errors
    while preserving genuine clinical tail variation.

    Returns ``(cleaned_df, summary_df)``.
    """
    cleaned_parts = []
    summary = []
    for node, g in df.groupby(node_col):
        g = g.copy()
        q1 = g[value_col].quantile(0.25)
        q3 = g[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        keep = g[value_col].between(lower, upper, inclusive="both")

        summary.append(
            {
                "node": node,
                "n_before": len(g),
                "n_removed": int((~keep).sum()),
                "n_after": int(keep.sum()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_cutoff": lower,
                "upper_cutoff": upper,
            }
        )
        cleaned_parts.append(g.loc[keep])

    cleaned = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else df.iloc[:0]
    return cleaned, pd.DataFrame(summary).sort_values("node").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: collapse to one row per participant
# ---------------------------------------------------------------------------


def collapse_to_wide_median(
    cleaned_df: pd.DataFrame,
    person_col: str = "person_id",
    node_col: str = "node",
    value_col: str = "value_clean",
) -> pd.DataFrame:
    """Pivot to one row per participant; repeated measures collapsed by median."""
    return (
        cleaned_df.groupby([person_col, node_col])[value_col]
        .median()
        .unstack(node_col)
        .reset_index()
        .rename_axis(None, axis=1)
    )


# ---------------------------------------------------------------------------
# Step 5: T2D node from condition frame
# ---------------------------------------------------------------------------


def build_t2d_node(condition_df: pd.DataFrame, person_col: str = "person_id") -> pd.DataFrame:
    """Derive the binary T2D node from a condition-occurrence frame.

    Every person appearing in ``condition_df`` is assumed to have T2D
    (the upstream cohort query already restricts to T2D codes). Returns a
    frame with columns ``[person_id, type2_diabetes, first_t2d_date,
    n_t2d_records]`` -- only the first two are used by the DAG modelling.
    """
    df = condition_df.copy()
    df["condition_start_datetime"] = pd.to_datetime(
        df.get("condition_start_datetime"), errors="coerce"
    )
    return (
        df.groupby(person_col)
        .agg(
            type2_diabetes=(person_col, lambda _grp: 1),
            first_t2d_date=("condition_start_datetime", "min"),
            n_t2d_records=("condition_concept_id", "size"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class CohortBuildResult:
    wide: pd.DataFrame  # one row per participant; NaN where unmeasured
    complete: pd.DataFrame  # complete-case subset of ``wide``
    outlier_summary: pd.DataFrame
    nodes: Tuple[str, ...]
    node_types: Tuple[str, ...]


def build_cohort_dataset(
    measurement_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    iqr_multiplier: float = 4.0,
) -> CohortBuildResult:
    """Run the full cleaning pipeline and return the wide + complete-case frames.

    The output ``nodes`` / ``node_types`` only include columns that
    actually have observations in ``measurement_df`` -- e.g. if the input
    contains no systolic-BP rows, ``systolic_bp`` is omitted.
    """
    labelled = attach_node_labels(measurement_df)
    cleaned = clean_measurements(labelled)
    cleaned, outlier_summary = remove_extreme_outliers_iqr(
        cleaned, iqr_multiplier=iqr_multiplier
    )
    wide_meas = collapse_to_wide_median(cleaned)

    t2d = build_t2d_node(condition_df)[["person_id", "type2_diabetes"]]

    all_people = pd.DataFrame(
        {
            "person_id": pd.concat(
                [condition_df["person_id"], measurement_df["person_id"]]
            ).drop_duplicates()
        }
    )

    wide = (
        all_people.merge(t2d, on="person_id", how="left")
        .merge(wide_meas, on="person_id", how="left")
    )
    wide["type2_diabetes"] = wide["type2_diabetes"].fillna(0).astype(int)

    present = [c for c in COHORT_NODES if c in wide.columns]
    wide = wide[["person_id"] + present].copy()

    complete = wide.dropna().copy()
    node_types = tuple(COHORT_NODE_TYPES[c] for c in present)
    return CohortBuildResult(
        wide=wide,
        complete=complete,
        outlier_summary=outlier_summary,
        nodes=tuple(present),
        node_types=node_types,
    )


# ---------------------------------------------------------------------------
# CSV -> matrix loader for the DAG search engines
# ---------------------------------------------------------------------------


def load_cohort_csv(
    path: str,
    standardise_continuous: bool = True,
    drop_incomplete: bool = True,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """Load a wide cohort CSV and return ``(X, columns, node_types)``.

    The CSV must contain a subset of :data:`COHORT_NODES`. ``person_id``
    if present is dropped. Continuous columns are z-scored when
    ``standardise_continuous=True`` so the BGe scorer's prior is well-
    conditioned and the Laplace logistic ridge has a comparable scale on
    every covariate. Binary columns are coerced to ``{0, 1}``.
    """
    df = pd.read_csv(path)
    if "person_id" in df.columns:
        df = df.drop(columns=["person_id"])

    present = [c for c in COHORT_NODES if c in df.columns]
    if not present:
        raise ValueError(
            f"CSV {path!r} contains none of the cohort columns "
            f"{COHORT_NODES!r}; got {list(df.columns)!r}"
        )
    df = df[present].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_incomplete:
        df = df.dropna().copy()

    types = tuple(COHORT_NODE_TYPES[c] for c in present)

    for col, kind in zip(present, types):
        if kind == "binary":
            df[col] = df[col].round().astype(int)

    if standardise_continuous:
        for col, kind in zip(present, types):
            if kind == "continuous":
                v = df[col].astype(float).to_numpy()
                mu = v.mean()
                sd = v.std(ddof=0)
                if sd == 0.0:
                    sd = 1.0
                df[col] = (v - mu) / sd

    X = df.to_numpy(dtype=np.float64)
    return X, tuple(present), types


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    present = set(columns)
    for c in candidates:
        if c in present:
            return c
    return None


def _extract_survival_columns(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract time/event arrays from a cohort CSV when present.

    The cohort table may be a static binary endpoint table or a survival
    table. Static tables return zero-filled arrays and mark
    ``has_survival=false``. Survival tables must carry one recognised time
    column and either an explicit event column or ``type2_diabetes``.
    """
    n = int(raw.shape[0])
    time_col = _first_present(raw.columns, SURVIVAL_TIME_COLUMNS)
    if time_col is None:
        return (
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=int),
            {"has_survival": False, "time_col": None, "event_col": None},
        )

    event_col = _first_present(raw.columns, SURVIVAL_EVENT_COLUMNS)
    if event_col is None and "type2_diabetes" in raw.columns:
        event_col = "type2_diabetes"
    if event_col is None:
        raise ValueError(
            f"cohort CSV has survival time column {time_col!r} but no event "
            f"column from {SURVIVAL_EVENT_COLUMNS!r} or 'type2_diabetes'"
        )

    time = pd.to_numeric(raw[time_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(raw[event_col], errors="coerce").to_numpy(dtype=float)
    event = np.rint(event).astype(float)
    meta = {
        "has_survival": True,
        "time_col": time_col,
        "event_col": event_col,
    }
    return time, event, meta


def load_cohort_dataset_with_person_ids(
    path: str,
    standardise_continuous: bool = True,
    drop_incomplete: bool = True,
):
    """Load a cohort CSV and keep the per-row ``person_id`` values.

    This is the real-data entry point used when genomic scores must be
    aligned with the AoU cohort. It applies the same numeric coercion,
    complete-case filtering, binary coercion, and continuous standardisation
    as :func:`load_cohort_csv`, then returns ``(dataset, person_id)`` where
    ``person_id`` is ordered exactly like ``dataset.X``.
    """
    from .synthetic import SyntheticDataset

    raw = pd.read_csv(path, dtype={"person_id": "string"})
    if "person_id" not in raw.columns:
        raise ValueError(f"cohort CSV {path!r} must contain person_id for genomic alignment")

    person_id = raw["person_id"].astype("string").copy()
    time, event, survival_meta = _extract_survival_columns(raw)
    df = raw.drop(columns=["person_id"])

    present = [c for c in COHORT_NODES if c in df.columns]
    if not present:
        raise ValueError(
            f"CSV {path!r} contains none of the cohort columns "
            f"{COHORT_NODES!r}; got {list(df.columns)!r}"
        )
    df = df[present].copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_incomplete:
        keep = ~df.isna().any(axis=1)
        if survival_meta["has_survival"]:
            keep &= np.isfinite(time)
            keep &= np.isfinite(event)
            keep &= time > 0.0
        df = df.loc[keep].copy()
        person_id = person_id.loc[keep].copy()
        time = time[keep.to_numpy()]
        event = event[keep.to_numpy()]
    elif survival_meta["has_survival"] and (
        (not np.all(np.isfinite(time))) or (not np.all(np.isfinite(event)))
    ):
        raise ValueError("survival time/event columns contain non-finite values")

    types = tuple(COHORT_NODE_TYPES[c] for c in present)

    for col, kind in zip(present, types):
        if kind == "binary":
            df[col] = df[col].round().astype(int)

    if standardise_continuous:
        for col, kind in zip(present, types):
            if kind == "continuous":
                v = df[col].astype(float).to_numpy()
                mu = v.mean()
                sd = v.std(ddof=0)
                if sd == 0.0:
                    raise ValueError(f"continuous cohort column has zero variance: {col}")
                df[col] = (v - mu) / sd

    X = df.to_numpy(dtype=np.float64)
    n, p = X.shape
    dataset = SyntheticDataset(
        X=X,
        time=time.astype(float, copy=False) if survival_meta["has_survival"] else np.zeros(n, dtype=float),
        event=event.astype(int, copy=False) if survival_meta["has_survival"] else np.zeros(n, dtype=int),
        columns=tuple(present),
        node_types=types,
        ground_truth_adj=np.zeros((p, p), dtype=int),
    )
    return dataset, person_id.astype(str).to_numpy()


# ---------------------------------------------------------------------------
# Adapter: cohort CSV -> SyntheticDataset (the pipeline's universal type)
# ---------------------------------------------------------------------------


def load_cohort_dataset(
    path: str,
    standardise_continuous: bool = True,
    drop_incomplete: bool = True,
):
    """Load a cohort CSV and return a :class:`SyntheticDataset`.

    Static cohort tables can omit time-to-event columns; in that case
    ``time`` and ``event`` are filled with zeros here and the production
    pipeline must attach an OMOP-derived survival outcome before GAM. The
    structure-search stages (DAGSLAM, MCMC) treat the T2D node as binary.
    """
    from .synthetic import SyntheticDataset

    raw = pd.read_csv(path)
    time, event, survival_meta = _extract_survival_columns(raw)
    X, columns, node_types = load_cohort_csv(
        path,
        standardise_continuous=standardise_continuous,
        drop_incomplete=drop_incomplete,
    )
    n, p = X.shape
    if survival_meta["has_survival"]:
        df = raw.copy()
        if "person_id" in df.columns:
            df = df.drop(columns=["person_id"])
        present = [c for c in COHORT_NODES if c in df.columns]
        for col in present:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        keep = np.ones(raw.shape[0], dtype=bool)
        if drop_incomplete:
            keep &= (~df[present].isna().any(axis=1)).to_numpy()
        keep &= np.isfinite(time)
        keep &= np.isfinite(event)
        keep &= time > 0.0
        time = time[keep]
        event = event[keep]
        if time.shape[0] != n:
            raise RuntimeError("internal survival row alignment mismatch")
    else:
        time = np.zeros(n, dtype=float)
        event = np.zeros(n, dtype=int)
    return SyntheticDataset(
        X=X,
        time=time.astype(float, copy=False),
        event=event.astype(int, copy=False),
        columns=columns,
        node_types=node_types,
        ground_truth_adj=np.zeros((p, p), dtype=int),
    )


@dataclass
class SurvivalOutcome:
    """Incident T2D survival outcome aligned to a person-id order."""

    person_id: np.ndarray
    time: np.ndarray
    event: np.ndarray
    keep: np.ndarray
    baseline_dt: np.ndarray
    end_dt: np.ndarray
    t2d_dt: np.ndarray
    meta: dict


def _require_columns(df: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_survival_outcome(
    person_ids: Sequence[str],
    visit_baseline: pd.DataFrame,
    observation_period: pd.DataFrame,
    t2d_event: pd.DataFrame,
    *,
    min_followup_days: int = 1,
) -> SurvivalOutcome:
    """Build incident T2D time/event arrays from OMOP follow-up frames.

    ``time`` starts at each participant's earliest observed visit and ends at
    first post-baseline T2D diagnosis or observation-period end. Participants
    with T2D on or before baseline are prevalent cases and are excluded from the
    incident survival cohort.
    """
    if min_followup_days <= 0:
        raise ValueError("min_followup_days must be positive")
    _require_columns(visit_baseline, "visit_baseline", ("person_id", "baseline_dt"))
    _require_columns(
        observation_period,
        "observation_period",
        ("person_id", "observation_end_dt"),
    )
    _require_columns(t2d_event, "t2d_event", ("person_id", "t2d_dt"))

    order = pd.Index([str(pid) for pid in person_ids], name="person_id")
    if order.empty:
        raise ValueError("person_ids must be non-empty")

    baseline_df = visit_baseline.copy()
    baseline_df["person_id"] = baseline_df["person_id"].astype(str)
    baseline_df["baseline_dt"] = _coerce_datetime(baseline_df["baseline_dt"])
    baseline = baseline_df.groupby("person_id")["baseline_dt"].min().reindex(order)

    obs_df = observation_period.copy()
    obs_df["person_id"] = obs_df["person_id"].astype(str)
    obs_df["observation_end_dt"] = _coerce_datetime(obs_df["observation_end_dt"])
    obs_end = obs_df.groupby("person_id")["observation_end_dt"].max().reindex(order)

    t2d_df = t2d_event.copy()
    t2d_df["person_id"] = t2d_df["person_id"].astype(str)
    t2d_df["t2d_dt"] = _coerce_datetime(t2d_df["t2d_dt"])
    first_t2d = t2d_df.dropna(subset=["t2d_dt"]).groupby("person_id")["t2d_dt"].min()
    t2d_dt = first_t2d.reindex(order)

    has_dates = baseline.notna() & obs_end.notna()
    prevalent = has_dates & t2d_dt.notna() & (t2d_dt <= baseline)
    incident = has_dates & t2d_dt.notna() & (t2d_dt > baseline) & (t2d_dt <= obs_end)
    end_dt = obs_end.where(~incident, t2d_dt)
    followup_days = (end_dt - baseline).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    keep = (
        has_dates.to_numpy(dtype=bool)
        & ~prevalent.to_numpy(dtype=bool)
        & np.isfinite(followup_days)
        & (followup_days >= float(min_followup_days))
    )
    event = incident.to_numpy(dtype=int)
    time_years = followup_days / 365.25
    time_years = np.where(keep, time_years, 0.0)
    event = np.where(keep, event, 0).astype(int)

    meta = {
        "source": "omop",
        "n_input": int(len(order)),
        "n_kept": int(keep.sum()),
        "n_events": int(event[keep].sum()),
        "n_missing_baseline": int(baseline.isna().sum()),
        "n_missing_observation_end": int(obs_end.isna().sum()),
        "n_prevalent_t2d": int(prevalent.sum()),
        "n_nonpositive_followup": int(
            (has_dates.to_numpy(dtype=bool) & ~np.isfinite(followup_days)).sum()
            + (
                has_dates.to_numpy(dtype=bool)
                & np.isfinite(followup_days)
                & (followup_days < float(min_followup_days))
            ).sum()
        ),
        "min_followup_days": int(min_followup_days),
    }
    if meta["n_kept"] == 0:
        raise ValueError("OMOP survival outcome has no rows with usable follow-up")
    if meta["n_events"] == 0:
        raise ValueError("OMOP survival outcome has no incident T2D events after baseline")

    return SurvivalOutcome(
        person_id=order.to_numpy(dtype=str),
        time=time_years.astype(float, copy=False),
        event=event,
        keep=keep,
        baseline_dt=baseline.astype("datetime64[ns]").to_numpy(),
        end_dt=end_dt.astype("datetime64[ns]").to_numpy(),
        t2d_dt=t2d_dt.astype("datetime64[ns]").to_numpy(),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Caching: local-then-bucket lookup, build only on miss
# ---------------------------------------------------------------------------


def _gsutil() -> Optional[str]:
    return shutil.which("gsutil")


def _gcloud() -> Optional[str]:
    return shutil.which("gcloud")


def _bucket_path(bucket: str, filename: str) -> str:
    bucket = bucket.rstrip("/")
    return f"{bucket}/data/{filename}"


def _bucket_prefixed_path(bucket: str, prefix: str, filename: str) -> str:
    bucket = bucket.rstrip("/")
    prefix = prefix.strip("/")
    return f"{bucket}/{prefix}/{filename}"


def _gsutil_exists(uri: str) -> bool:
    gs = _gsutil()
    if gs is None:
        return False
    r = subprocess.run([gs, "-q", "stat", uri], capture_output=True, check=False)
    return r.returncode == 0


def _gsutil_copy(src: str, dst: str) -> None:
    gs = _gsutil()
    if gs is None:
        raise RuntimeError("gsutil is not on PATH; cannot fetch from bucket")
    proc = subprocess.run(
        [gs, "cp", src, dst],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "gsutil cp failed exit_code={code} src={src!r} dst={dst!r}\n"
            "stderr (last 4KiB):\n{stderr}".format(
                code=proc.returncode,
                src=src,
                dst=dst,
                stderr=(proc.stderr or "")[-4096:],
            )
        )


def _restore_bucket_file(uri: str, dst: Path) -> bool:
    if not _gsutil_exists(uri):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()
    _gsutil_copy(uri, str(part))
    os.replace(part, dst)
    return True


def resolve_cohort_csv(
    name: str = "complete",
    cache_dir: str | os.PathLike = ".",
    bucket: Optional[str] = None,
) -> Path:
    """Return a local path to the cached cohort CSV, fetching from bucket on miss.

    ``name`` is one of :data:`CACHE_FILENAMES` (``"complete"`` or
    ``"participant_level"``). The lookup order is:

      1. ``cache_dir/<canonical>.csv`` or accepted notebook filename;
      2. ``$WORKSPACE_BUCKET/data/<filename>.csv`` -- copied into
         ``cache_dir`` as the canonical filename via ``gsutil cp``;
      3. ``bucket/data/<canonical>.csv`` if ``bucket`` is given explicitly.

    Raises ``FileNotFoundError`` if no source resolves.
    """
    if name not in CACHE_FILENAMES:
        raise KeyError(f"unknown cohort name {name!r}; expected one of {list(CACHE_FILENAMES)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidates = CACHE_FILENAMES[name]
    canonical = candidates[0]

    for fname in candidates:
        local = cache_dir / fname
        if local.exists():
            return local

    bucket_uri = bucket or os.environ.get("WORKSPACE_BUCKET")
    if bucket_uri:
        for fname in candidates:
            uri = _bucket_path(bucket_uri, fname)
            if _gsutil_exists(uri):
                dst = cache_dir / canonical
                _gsutil_copy(uri, str(dst))
                return dst

    locations = [str(cache_dir / f) for f in candidates]
    if bucket_uri:
        locations += [_bucket_path(bucket_uri, f) for f in candidates]
    raise FileNotFoundError(
        f"cohort CSV {name!r} not found; looked in: {locations}"
    )


def write_cohort_cache(
    result: CohortBuildResult,
    cache_dir: str | os.PathLike = ".",
    bucket: Optional[str] = None,
    upload: bool = False,
) -> Tuple[Path, Path]:
    """Write the wide and complete-case CSVs to ``cache_dir`` (and optionally bucket).

    Returns ``(participant_level_path, complete_path)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pl = cache_dir / CACHE_FILENAMES["participant_level"][0]
    cc = cache_dir / CACHE_FILENAMES["complete"][0]
    result.wide.to_csv(pl, index=False)
    result.complete.to_csv(cc, index=False)
    if upload:
        bucket_uri = bucket or os.environ.get("WORKSPACE_BUCKET")
        if not bucket_uri:
            raise RuntimeError("upload=True but no bucket available")
        _gsutil_copy(str(pl), _bucket_path(bucket_uri, pl.name))
        _gsutil_copy(str(cc), _bucket_path(bucket_uri, cc.name))
    return pl, cc


AOU_GENOTYPE_BUCKET: str = "gs://fc-aou-datasets-controlled/v8/microarray/plink"
AOU_GENOTYPE_FILES: Tuple[str, ...] = ("arrays.bed", "arrays.bim", "arrays.fam")


CURATED_OMOP_CONDITION_IDS: Tuple[int, ...] = (
    # Type 2 diabetes and close metabolic comorbidities.
    *T2D_CONDITION_CONCEPT_IDS,
    # Hypertension / cardiovascular disease.
    320128,
    316866,
    319835,
    321588,
    312327,
    314666,
    317009,
    4329847,
    # Dyslipidaemia and obesity-related codes.
    432867,
    443392,
    433736,
    437663,
    434376,
    # Kidney and liver disease, common T2D correlates.
    46271022,
    4030518,
    197320,
)


def _remote_gcloud_size(uri: str, billing_project: str) -> int:
    gc = _gcloud()
    if gc is None:
        raise RuntimeError("gcloud is not on PATH; cannot inspect the AoU microarray bucket")
    r = subprocess.run(
        [gc, "storage", "ls", "-l", uri, f"--billing-project={billing_project}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == uri:
            return int(parts[0])
    raise RuntimeError(f"could not read remote size for {uri!r} from gcloud output")


def resolve_aou_genotypes(
    cache_dir: str | os.PathLike = ".",
    bucket: str = AOU_GENOTYPE_BUCKET,
    billing_project: Optional[str] = None,
) -> Path:
    """Return local AoU v8 microarray PLINK ``arrays.bed``.

    Each member of ``arrays.{bed,bim,fam}`` is downloaded from the requester
    pays AoU controlled bucket when absent or size-mismatched. Downloads go
    to ``*.part`` and are atomically moved into place after the copied file
    matches the remote byte count, so interrupted downloads are never treated
    as usable genotype inputs.
    """
    project = billing_project or os.environ.get("GOOGLE_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_PROJECT must be set to download AoU microarray data")

    gc = _gcloud()
    if gc is None:
        raise RuntimeError("gcloud is not on PATH; cannot fetch AoU microarray data")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    bucket = bucket.rstrip("/")

    for fname in AOU_GENOTYPE_FILES:
        src = f"{bucket}/{fname}"
        dst = cache / fname
        expected_size = _remote_gcloud_size(src, project)
        if dst.is_file() and dst.stat().st_size == expected_size:
            continue

        part = dst.with_suffix(dst.suffix + ".part")
        if part.exists():
            part.unlink()
        subprocess.run(
            [gc, "storage", "cp", src, str(part), f"--billing-project={project}"],
            check=True,
        )
        got_size = part.stat().st_size
        if got_size != expected_size:
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"downloaded {src} has {got_size} bytes; expected {expected_size}"
            )
        os.replace(part, dst)

    bed = cache / "arrays.bed"
    bim = cache / "arrays.bim"
    fam = cache / "arrays.fam"
    if not (bed.is_file() and bim.is_file() and fam.is_file()):
        raise FileNotFoundError(f"incomplete AoU microarray PLINK triple in {cache}")
    return bed


# ---------------------------------------------------------------------------
# Genotype discovery (cheap pre-check before resolve_aou_genotypes downloads)
# ---------------------------------------------------------------------------


def _has_genotype_triple(d: Path) -> bool:
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in AOU_GENOTYPE_FILES)


def discover_genotype_dir(extra: Sequence[str | os.PathLike] = ()) -> Optional[Path]:
    """Return the first directory containing a complete arrays.{bed,bim,fam} triple.

    Search order (first hit wins): each path in ``extra`` (caller-supplied
    overrides), then ``$PWD``, ``$HOME``, ``$HOME/causal-pred``,
    ``$HOME/causal-pred/genomes``. Returns ``None`` if none have all three
    non-zero files; the caller is responsible for downloading via
    :func:`resolve_aou_genotypes`.
    """
    candidates: list[Path] = [Path(p) for p in extra]
    candidates += [
        Path.cwd(),
        Path.home(),
        Path.home() / "causal-pred",
        Path.home() / "causal-pred" / "genomes",
    ]
    seen: set[Path] = set()
    for c in candidates:
        c = c.expanduser()
        if c in seen:
            continue
        seen.add(c)
        if c.is_dir() and _has_genotype_triple(c):
            return c
    return None


# ---------------------------------------------------------------------------
# OMOP long-frame fetchers for the EHR / crosscoder stream
# ---------------------------------------------------------------------------


def _normalise_person_ids(person_ids: Sequence[str]) -> tuple[list[str], list[int]]:
    ids = [str(pid).strip() for pid in person_ids]
    if not ids:
        raise ValueError("person_ids must be non-empty")
    if any(pid == "" or not pid.lstrip("+-").isdigit() for pid in ids):
        raise ValueError("AoU OMOP person_id values must be integer-like")
    return ids, [int(pid) for pid in ids]


def _omop_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _read_omop_cache(path: Path, key: str) -> Optional[pd.DataFrame]:
    key_path = path.with_suffix(path.suffix + ".key")
    if path.is_file() and key_path.is_file() and key_path.read_text().strip() == key:
        return pd.read_parquet(path)
    return None


def _write_omop_cache(path: Path, key: str, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    df.to_parquet(part, index=False)
    os.replace(part, path)
    path.with_suffix(path.suffix + ".key").write_text(key)


def fetch_omop_long_frames(
    person_ids: Sequence[str],
    *,
    cdr: Optional[str] = None,
    cache_dir: str | os.PathLike = "data/omop",
    workspace_bucket: Optional[str] = None,
    workspace_prefix: str = "intermediates/causal-pred/omop",
    condition_concept_ids: Optional[Sequence[int]] = CURATED_OMOP_CONDITION_IDS,
    drug_concept_ids: Optional[Sequence[int]] = None,
    measurement_concept_ids: Optional[Sequence[int]] = None,
    fetch_drugs: bool = False,
    fetch_measurements: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch cached AoU OMOP long frames for the EHR feature stream.

    The production pipeline currently consumes the prepared wide cohort CSV.
    This helper is kept wired for the EHR/crosscoder branch: it fetches
    baseline-censorable OMOP frames from the AoU CDR, caches only parquet
    intermediates under ``cache_dir``, and mirrors those parquets to the
    workspace bucket when available. It never uploads or copies genotype
    files. Visit data is aggregated to each participant's earliest visit
    before it leaves BigQuery; the pipeline only needs that baseline date, and
    downloading every visit row is the slow path at AoU scale.
    """
    from google.cloud import bigquery

    cdr = cdr or os.environ.get("WORKSPACE_CDR")
    if not cdr:
        raise RuntimeError("WORKSPACE_CDR must be set or passed as cdr")

    pid_str, pid_int = _normalise_person_ids(person_ids)
    cache_root = Path(cache_dir)
    workspace_bucket = (
        workspace_bucket or os.environ.get("WORKSPACE_BUCKET") or ""
    ).rstrip("/")
    workspace_bucket = workspace_bucket or None
    client: Optional[bigquery.Client] = None

    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    def _array_param(name: str, values: Sequence[int]) -> bigquery.ArrayQueryParameter:
        return bigquery.ArrayQueryParameter(name, "INT64", [int(v) for v in values])

    def _query(
        name: str,
        sql: str,
        params: Sequence[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter],
        payload: dict,
    ) -> pd.DataFrame:
        nonlocal client
        key = _omop_cache_key({"cdr": cdr, "person_ids": pid_str, **payload})
        path = cache_root / f"{name}-{key}.parquet"
        cached = _read_omop_cache(path, key)
        if cached is not None:
            _emit(f"{name} cache hit rows={len(cached)} cols={cached.shape[1]}")
            return cached
        key_path = path.with_suffix(path.suffix + ".key")
        if workspace_bucket is not None:
            data_uri = _bucket_prefixed_path(
                workspace_bucket, workspace_prefix, path.name
            )
            key_uri = _bucket_prefixed_path(
                workspace_bucket, workspace_prefix, key_path.name
            )
            if _restore_bucket_file(data_uri, path) and _restore_bucket_file(
                key_uri, key_path
            ):
                cached = _read_omop_cache(path, key)
                if cached is not None:
                    _emit(
                        f"{name} workspace cache hit rows={len(cached)} "
                        f"cols={cached.shape[1]}"
                    )
                    return cached
        _emit(f"{name} BigQuery start cdr={cdr} n_persons={len(pid_str)}")
        started_at = time.time()
        job_config = bigquery.QueryJobConfig(query_parameters=list(params))
        if client is None:
            client = bigquery.Client()
        try:
            job = client.query(sql, job_config=job_config)
            _emit(
                f"{name} BigQuery job submitted job_id={getattr(job, 'job_id', '?')}"
            )
            df = job.to_dataframe(create_bqstorage_client=True)
        except Exception as exc:
            raise RuntimeError(
                "BigQuery {name} failed cdr={cdr} n_persons={n} "
                "elapsed={elapsed:.1f}s exc_type={etype} exc={exc}".format(
                    name=name,
                    cdr=cdr,
                    n=len(pid_str),
                    elapsed=time.time() - started_at,
                    etype=type(exc).__name__,
                    exc=exc,
                )
            ) from exc
        _write_omop_cache(path, key, df)
        if workspace_bucket is not None:
            _gsutil_copy(
                str(path),
                _bucket_prefixed_path(workspace_bucket, workspace_prefix, path.name),
            )
            _gsutil_copy(
                str(key_path),
                _bucket_prefixed_path(workspace_bucket, workspace_prefix, key_path.name),
            )
        _emit(
            f"{name} BigQuery done rows={len(df)} cols={df.shape[1]} "
            f"elapsed={time.time() - started_at:.1f}s"
        )
        return df

    person_param = _array_param("person_ids", pid_int)
    frames: dict[str, pd.DataFrame] = {}

    frames["visit_baseline"] = _query(
        "visit_baseline",
        f"""
        SELECT
          CAST(person_id AS STRING) AS person_id,
          MIN(COALESCE(CAST(visit_start_datetime AS TIMESTAMP), TIMESTAMP(visit_start_date))) AS baseline_dt
        FROM `{cdr}.visit_occurrence`
        WHERE person_id IN UNNEST(@person_ids)
        GROUP BY person_id
        """,
        [person_param],
        {"table": "visit_occurrence", "aggregation": "baseline"},
    )

    frames["observation_period"] = _query(
        "observation_period",
        f"""
        SELECT
          CAST(person_id AS STRING) AS person_id,
          TIMESTAMP(MIN(observation_period_start_date)) AS observation_start_dt,
          TIMESTAMP(MAX(observation_period_end_date)) AS observation_end_dt
        FROM `{cdr}.observation_period`
        WHERE person_id IN UNNEST(@person_ids)
        GROUP BY person_id
        """,
        [person_param],
        {"table": "observation_period", "aggregation": "person_followup_bounds"},
    )

    t2d_params: list = [
        person_param,
        _array_param("t2d_condition_concept_ids", T2D_CONDITION_CONCEPT_IDS),
    ]
    frames["t2d_event"] = _query(
        "t2d_event",
        f"""
        SELECT
          CAST(person_id AS STRING) AS person_id,
          MIN(COALESCE(CAST(condition_start_datetime AS TIMESTAMP), TIMESTAMP(condition_start_date))) AS t2d_dt
        FROM `{cdr}.condition_occurrence`
        WHERE person_id IN UNNEST(@person_ids)
          AND condition_concept_id IN UNNEST(@t2d_condition_concept_ids)
        GROUP BY person_id
        """,
        t2d_params,
        {
            "table": "condition_occurrence",
            "condition_ids": T2D_CONDITION_CONCEPT_IDS,
            "aggregation": "first_t2d_datetime_by_person",
        },
    )

    condition_ids = (
        tuple(int(x) for x in condition_concept_ids)
        if condition_concept_ids is not None
        else tuple()
    )
    condition_filter = (
        "AND condition_concept_id IN UNNEST(@condition_concept_ids)"
        if condition_ids
        else ""
    )
    condition_params: list = [person_param]
    if condition_ids:
        condition_params.append(_array_param("condition_concept_ids", condition_ids))
    frames["condition_long"] = _query(
        "condition_long",
        f"""
        SELECT
          CAST(person_id AS STRING) AS person_id,
          CAST(condition_concept_id AS STRING) AS phecode,
          MIN(COALESCE(CAST(condition_start_datetime AS TIMESTAMP), TIMESTAMP(condition_start_date))) AS datetime
        FROM `{cdr}.condition_occurrence`
        WHERE person_id IN UNNEST(@person_ids)
          {condition_filter}
        GROUP BY person_id, condition_concept_id
        """,
        condition_params,
        {
            "table": "condition_occurrence",
            "condition_ids": condition_ids,
            "aggregation": "min_datetime_by_person_condition",
        },
    )

    if fetch_drugs:
        drug_ids = (
            tuple(int(x) for x in drug_concept_ids)
            if drug_concept_ids is not None
            else tuple()
        )
        drug_filter = "AND drug_concept_id IN UNNEST(@drug_concept_ids)" if drug_ids else ""
        drug_params: list = [person_param]
        if drug_ids:
            drug_params.append(_array_param("drug_concept_ids", drug_ids))
        frames["drug_long"] = _query(
            "drug_long",
            f"""
            SELECT
              CAST(person_id AS STRING) AS person_id,
              CAST(drug_concept_id AS STRING) AS atc_class,
              MIN(COALESCE(CAST(drug_exposure_start_datetime AS TIMESTAMP), TIMESTAMP(drug_exposure_start_date))) AS datetime
            FROM `{cdr}.drug_exposure`
            WHERE person_id IN UNNEST(@person_ids)
              {drug_filter}
            GROUP BY person_id, drug_concept_id
            """,
            drug_params,
            {
                "table": "drug_exposure",
                "drug_ids": drug_ids,
                "aggregation": "min_datetime_by_person_drug",
            },
        )

    if fetch_measurements:
        measurement_ids = (
            tuple(int(x) for x in measurement_concept_ids)
            if measurement_concept_ids is not None
            else tuple()
        )
        measurement_filter = (
            "AND measurement_concept_id IN UNNEST(@measurement_concept_ids)"
            if measurement_ids
            else ""
        )
        measurement_params: list = [person_param]
        if measurement_ids:
            measurement_params.append(
                _array_param("measurement_concept_ids", measurement_ids)
            )
        frames["measurement_long"] = _query(
            "measurement_long",
            f"""
            SELECT
              CAST(person_id AS STRING) AS person_id,
              CAST(measurement_concept_id AS STRING) AS lab,
              value_as_number AS value,
              COALESCE(CAST(measurement_datetime AS TIMESTAMP), TIMESTAMP(measurement_date)) AS datetime
            FROM `{cdr}.measurement`
            WHERE person_id IN UNNEST(@person_ids)
              AND value_as_number IS NOT NULL
              {measurement_filter}
            """,
            measurement_params,
            {"table": "measurement", "measurement_ids": measurement_ids},
        )

    return frames


def resolve_baseline_dt(person_ids: Sequence[str], visit_long: pd.DataFrame) -> pd.Series:
    """Return each participant's earliest observed visit datetime.

    Missing participants are retained with ``NaT`` so downstream
    baseline-censoring drops their EHR events instead of inventing a date.
    """
    order = pd.Index([str(pid) for pid in person_ids], name="person_id")
    if visit_long.empty:
        return pd.Series(pd.NaT, index=order, name="baseline_dt")
    df = visit_long.copy()
    df["person_id"] = df["person_id"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    baseline = df.dropna(subset=["datetime"]).groupby("person_id")["datetime"].min()
    return baseline.reindex(order).rename("baseline_dt")


# ---------------------------------------------------------------------------
# EHR panel: high-dim baseline-censored matrix for the crosscoder's EHR stream
# ---------------------------------------------------------------------------


@dataclass
class EhrPanel:
    """Wide, baseline-censored EHR feature matrix.

    Attributes
    ----------
    matrix : (n, m_E) float64
        One row per participant, in the order of ``person_id``.
    person_id : (n,) array of str
        Participant identifiers (string).
    feature_names : tuple of str, length m_E
        Column names of ``matrix``.
    feature_kinds : tuple of str, length m_E
        Per-column kind tag, one of ``"condition"``, ``"drug"``, ``"lab_mean"``,
        ``"lab_min"``, ``"lab_max"``, ``"lab_slope"``, ``"utilisation"``.
        Useful for downstream reweighting / residualisation.
    """

    matrix: np.ndarray
    person_id: np.ndarray
    feature_names: Tuple[str, ...]
    feature_kinds: Tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def m(self) -> int:
        return int(self.matrix.shape[1])


def _filter_pre_baseline(
    long_df: pd.DataFrame,
    datetime_col: str,
    baseline_dt: pd.Series,
    lookback_days: Optional[int],
    person_col: str,
) -> pd.DataFrame:
    """Keep only events strictly before each person's baseline datetime.

    ``baseline_dt`` is a Series indexed by ``person_id`` carrying the baseline
    datetime (the GAM ``time = 0``). If ``lookback_days`` is given, events
    older than ``baseline - lookback_days`` are dropped as well.
    """
    df = long_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col, person_col])
    bd = baseline_dt.reindex(df[person_col]).to_numpy()
    keep = df[datetime_col].to_numpy() < bd
    if lookback_days is not None:
        lookback_floor = pd.to_datetime(bd) - pd.Timedelta(days=int(lookback_days))
        keep &= df[datetime_col].to_numpy() >= lookback_floor.to_numpy()
    return df.loc[keep].copy()


def _wide_indicator(
    long_df: pd.DataFrame,
    person_col: str,
    group_col: str,
    person_order: pd.Series,
    min_prevalence: int,
    prefix: str,
) -> Tuple[np.ndarray, list[str]]:
    """Return a binary (n, m) matrix and column names for a long event frame.

    Each row is a participant in the order of ``person_order``; each column is
    a group whose count of distinct participants meets ``min_prevalence``.
    Cells are 1 if the (person, group) pair has at least one row, else 0.
    """
    if long_df.empty:
        return np.zeros((len(person_order), 0), dtype=np.float64), []
    counts = (
        long_df.drop_duplicates([person_col, group_col])
        .groupby(group_col)[person_col]
        .nunique()
    )
    keep_groups = counts[counts >= min_prevalence].index.tolist()
    if not keep_groups:
        return np.zeros((len(person_order), 0), dtype=np.float64), []
    n = len(person_order)
    m = len(keep_groups)
    mat = np.zeros((n, m), dtype=np.float64)
    sub = long_df.loc[
        long_df[group_col].isin(set(keep_groups)), [person_col, group_col]
    ].drop_duplicates()
    row_idx = pd.Categorical(
        sub[person_col].astype(str),
        categories=person_order.astype(str).tolist(),
    ).codes
    col_idx = pd.Categorical(sub[group_col], categories=keep_groups).codes
    valid = (row_idx >= 0) & (col_idx >= 0)
    mat[row_idx[valid], col_idx[valid]] = 1.0
    cols = [f"{prefix}:{g}" for g in keep_groups]
    return mat, cols


def _lab_summary(
    long_df: pd.DataFrame,
    person_col: str,
    lab_col: str,
    value_col: str,
    datetime_col: str,
    baseline_dt: pd.Series,
    person_order: pd.Series,
    min_observations: int,
) -> Tuple[np.ndarray, list[str], list[str]]:
    """Per-(person, lab) summary stats: mean / min / max / slope (yr^-1).

    The slope is OLS of ``value`` on ``years_until_baseline`` (so a *positive*
    slope means the lab has been rising as the person approaches baseline).
    Persons / labs with fewer than ``min_observations`` measurements get NaN
    in the slope; the per-lab column mean fills NaN at the end. Labs whose
    *number of distinct participants* with any measurement is below
    ``min_observations`` are dropped.
    """
    if long_df.empty:
        return np.zeros((len(person_order), 0), dtype=np.float64), [], []
    df = long_df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[person_col, lab_col, value_col, datetime_col])
    bd = baseline_dt.reindex(df[person_col]).to_numpy()
    df["years_until_baseline"] = (
        bd - df[datetime_col].to_numpy()
    ) / np.timedelta64(365, "D")
    df["years_until_baseline"] = -df["years_until_baseline"].astype(float)
    # i.e. negative for events in the past.

    lab_counts = df.groupby(lab_col)[person_col].nunique()
    keep_labs = sorted(lab_counts[lab_counts >= min_observations].index.tolist())
    if not keep_labs:
        return np.zeros((len(person_order), 0), dtype=np.float64), [], []
    df = df[df[lab_col].isin(set(keep_labs))]

    idx_by_pid = {pid: i for i, pid in enumerate(person_order.tolist())}
    col_by_lab = {lab: j for j, lab in enumerate(keep_labs)}
    n = len(person_order)
    m = len(keep_labs)

    means = np.full((n, m), np.nan, dtype=np.float64)
    mins = np.full((n, m), np.nan, dtype=np.float64)
    maxs = np.full((n, m), np.nan, dtype=np.float64)
    slopes = np.full((n, m), np.nan, dtype=np.float64)

    for (pid, lab), g in df.groupby([person_col, lab_col]):
        i = idx_by_pid.get(pid)
        if i is None:
            continue
        j = col_by_lab[lab]
        v = g[value_col].to_numpy(dtype=float)
        means[i, j] = float(np.mean(v))
        mins[i, j] = float(np.min(v))
        maxs[i, j] = float(np.max(v))
        if len(v) >= 2:
            t = g["years_until_baseline"].to_numpy(dtype=float)
            tc = t - t.mean()
            denom = float(np.sum(tc * tc))
            if denom > 1e-12:
                slopes[i, j] = float(np.sum(tc * (v - v.mean())) / denom)

    def _impute_col_means(mat: np.ndarray) -> np.ndarray:
        col_means = np.nanmean(mat, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        out = np.where(np.isnan(mat), col_means[None, :], mat)
        return out

    means = _impute_col_means(means)
    mins = _impute_col_means(mins)
    maxs = _impute_col_means(maxs)
    slopes = _impute_col_means(slopes)

    block = np.concatenate([means, mins, maxs, slopes], axis=1)
    names = (
        [f"lab_mean:{lab}" for lab in keep_labs]
        + [f"lab_min:{lab}" for lab in keep_labs]
        + [f"lab_max:{lab}" for lab in keep_labs]
        + [f"lab_slope:{lab}" for lab in keep_labs]
    )
    kinds = (
        ["lab_mean"] * m + ["lab_min"] * m + ["lab_max"] * m + ["lab_slope"] * m
    )
    return block, names, kinds


def build_ehr_panel(
    person_ids: Sequence[str],
    baseline_dt: pd.Series,
    *,
    condition_long: Optional[pd.DataFrame] = None,
    drug_long: Optional[pd.DataFrame] = None,
    measurement_long: Optional[pd.DataFrame] = None,
    visit_long: Optional[pd.DataFrame] = None,
    person_col: str = "person_id",
    condition_group_col: str = "phecode",
    drug_group_col: str = "atc_class",
    measurement_lab_col: str = "lab",
    measurement_value_col: str = "value",
    datetime_col: str = "datetime",
    min_prevalence: int = 50,
    min_lab_observations: int = 50,
    lookback_days: Optional[int] = 365 * 5,
) -> EhrPanel:
    """Assemble the EHR-stream panel from baseline-censored OMOP long-frames.

    All event frames are filtered to *strictly before* each participant's
    baseline datetime (``time = 0`` for the survival GAM). This is the hard
    no-lookahead invariant -- features must not be informed by anything that
    happens after the prediction horizon starts.

    Parameters
    ----------
    person_ids : sequence of str
        Cohort participants. The output ``matrix`` rows are in this order.
    baseline_dt : pandas.Series
        Series indexed by ``person_id`` with each participant's baseline
        datetime. Persons missing here have all events dropped.
    condition_long : DataFrame, optional
        Long frame of conditions with columns
        ``[person_col, condition_group_col, datetime_col]``. Each row is one
        condition occurrence; ``condition_group_col`` is the rolled-up group
        (PheCode is the canonical choice; ``condition_concept_id`` is a
        usable fallback that yields a more granular but noisier panel).
    drug_long : DataFrame, optional
        Long frame of drug exposures with columns
        ``[person_col, drug_group_col, datetime_col]``. ATC level-3 is the
        canonical group; ``drug_concept_id`` works as a fallback.
    measurement_long : DataFrame, optional
        Long frame of lab measurements with columns
        ``[person_col, measurement_lab_col, measurement_value_col,
        datetime_col]``. Values must be in canonical units already (use
        :func:`clean_measurements` upstream).
    visit_long : DataFrame, optional
        Long frame of healthcare encounters with columns
        ``[person_col, datetime_col]``. Used to add a single ``utilisation``
        covariate (count of pre-baseline encounters).
    min_prevalence : int
        Minimum number of distinct participants a condition / drug group
        must touch to be retained.
    min_lab_observations : int
        Minimum number of distinct participants a lab must touch to be kept.
    lookback_days : int, optional
        If set, drop events older than ``baseline - lookback_days``.
        ``None`` keeps the participant's full pre-baseline history.

    Returns
    -------
    EhrPanel
    """
    pid_array = np.asarray([str(p) for p in person_ids])
    person_order = pd.Series(pid_array, name=person_col)
    bd = pd.to_datetime(baseline_dt, errors="coerce")
    bd.index = bd.index.astype(str)

    blocks: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []

    def _push(mat: np.ndarray, cols: list[str], kind: str) -> None:
        if mat.shape[1] == 0:
            return
        blocks.append(mat)
        names.extend(cols)
        kinds.extend([kind] * mat.shape[1])

    if condition_long is not None and len(condition_long):
        cf = _filter_pre_baseline(
            condition_long, datetime_col, bd, lookback_days, person_col
        )
        cf[person_col] = cf[person_col].astype(str)
        cf[condition_group_col] = cf[condition_group_col].astype(str)
        mat, cols = _wide_indicator(
            cf, person_col, condition_group_col, person_order,
            min_prevalence, prefix="cond",
        )
        _push(mat, cols, "condition")

    if drug_long is not None and len(drug_long):
        df = _filter_pre_baseline(
            drug_long, datetime_col, bd, lookback_days, person_col
        )
        df[person_col] = df[person_col].astype(str)
        df[drug_group_col] = df[drug_group_col].astype(str)
        mat, cols = _wide_indicator(
            df, person_col, drug_group_col, person_order,
            min_prevalence, prefix="drug",
        )
        _push(mat, cols, "drug")

    if measurement_long is not None and len(measurement_long):
        mf = _filter_pre_baseline(
            measurement_long, datetime_col, bd, lookback_days, person_col
        )
        mf[person_col] = mf[person_col].astype(str)
        mf[measurement_lab_col] = mf[measurement_lab_col].astype(str)
        block, lab_names, lab_kinds = _lab_summary(
            mf,
            person_col,
            measurement_lab_col,
            measurement_value_col,
            datetime_col,
            bd,
            person_order,
            min_observations=min_lab_observations,
        )
        if block.shape[1] > 0:
            blocks.append(block)
            names.extend(lab_names)
            kinds.extend(lab_kinds)

    if visit_long is not None and len(visit_long):
        vf = _filter_pre_baseline(
            visit_long, datetime_col, bd, lookback_days, person_col
        )
        vf[person_col] = vf[person_col].astype(str)
        counts = vf.groupby(person_col).size()
        util = counts.reindex(pid_array, fill_value=0).to_numpy(dtype=float)
        _push(util.reshape(-1, 1), ["utilisation:n_encounters"], "utilisation")

    if not blocks:
        matrix = np.zeros((len(pid_array), 0), dtype=np.float64)
    else:
        matrix = np.concatenate(blocks, axis=1)

    return EhrPanel(
        matrix=matrix.astype(np.float64, copy=False),
        person_id=pid_array,
        feature_names=tuple(names),
        feature_kinds=tuple(kinds),
    )


__all__ = [
    "COHORT_NODES",
    "COHORT_NODE_TYPES",
    "PLAUSIBILITY_BOUNDS",
    "CACHE_FILENAMES",
    "T2D_CONDITION_CONCEPT_IDS",
    "label_measurement_node",
    "attach_node_labels",
    "clean_measurements",
    "remove_extreme_outliers_iqr",
    "collapse_to_wide_median",
    "build_t2d_node",
    "build_cohort_dataset",
    "load_cohort_csv",
    "load_cohort_dataset",
    "load_cohort_dataset_with_person_ids",
    "build_survival_outcome",
    "resolve_cohort_csv",
    "write_cohort_cache",
    "discover_genotype_dir",
    "resolve_aou_genotypes",
    "AOU_GENOTYPE_BUCKET",
    "AOU_GENOTYPE_FILES",
    "CURATED_OMOP_CONDITION_IDS",
    "fetch_omop_long_frames",
    "resolve_baseline_dt",
    "build_ehr_panel",
    "CohortBuildResult",
    "SurvivalOutcome",
    "EhrPanel",
]
