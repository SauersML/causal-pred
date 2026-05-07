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

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd


CACHE_FILENAMES: dict = {
    # Logical name -> ordered list of acceptable on-disk filenames. The first
    # entry is the canonical name used when *writing* a fresh cache; the
    # remaining entries are accepted on read so we tolerate the trailing
    # ``_case`` variant produced by some upstream notebooks.
    "participant_level": ("t2d_initial_nodes_participant_level.csv",),
    "complete": (
        "t2d_initial_nodes_complete.csv",
        "t2d_initial_nodes_complete_case.csv",
    ),
}


COHORT_NODES: Tuple[str, ...] = (
    "type2_diabetes",
    "bmi",
    "hba1c",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "systolic_bp",
)

COHORT_NODE_TYPES: dict = {
    "type2_diabetes": "binary",
    "bmi": "continuous",
    "hba1c": "continuous",
    "hdl_cholesterol": "continuous",
    "ldl_cholesterol": "continuous",
    "triglycerides": "continuous",
    "systolic_bp": "continuous",
    "fasting_glucose": "continuous",
}

# Canonical units after :func:`clean_measurements`:
#   bmi              kg/m^2
#   hba1c            percent (NGSP)
#   fasting_glucose  mg/dL
#   hdl_cholesterol  mg/dL
#   ldl_cholesterol  mg/dL
#   triglycerides    mg/dL
#   systolic_bp      mmHg
PLAUSIBILITY_BOUNDS: dict = {
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


# ---------------------------------------------------------------------------
# Adapter: cohort CSV -> SyntheticDataset (the pipeline's universal type)
# ---------------------------------------------------------------------------


def load_cohort_dataset(
    path: str,
    standardise_continuous: bool = True,
    drop_incomplete: bool = True,
):
    """Load a cohort CSV and return a :class:`SyntheticDataset`.

    The cohort table has no time-to-event outcome (T2D appears as a
    binary endpoint), so ``time`` and ``event`` are filled with zeros
    and ``ground_truth_adj`` is the empty matrix. The structure-search
    stages (DAGSLAM, MCMC) treat the binary T2D node as binary; the GAM
    stage skips itself when no ``"survival"`` node is present.
    """
    from .synthetic import SyntheticDataset

    X, columns, node_types = load_cohort_csv(
        path,
        standardise_continuous=standardise_continuous,
        drop_incomplete=drop_incomplete,
    )
    n, p = X.shape
    return SyntheticDataset(
        X=X,
        time=np.zeros(n, dtype=float),
        event=np.zeros(n, dtype=int),
        columns=columns,
        node_types=node_types,
        ground_truth_adj=np.zeros((p, p), dtype=int),
    )


# ---------------------------------------------------------------------------
# Caching: local-then-bucket lookup, build only on miss
# ---------------------------------------------------------------------------


def _gsutil() -> Optional[str]:
    return shutil.which("gsutil")


def _bucket_path(bucket: str, filename: str) -> str:
    bucket = bucket.rstrip("/")
    return f"{bucket}/data/{filename}"


def _gsutil_exists(uri: str) -> bool:
    gs = _gsutil()
    if gs is None:
        return False
    try:
        r = subprocess.run([gs, "-q", "stat", uri], capture_output=True, check=False)
    except OSError:
        return False
    return r.returncode == 0


def _gsutil_copy(src: str, dst: str) -> None:
    gs = _gsutil()
    if gs is None:
        raise RuntimeError("gsutil is not on PATH; cannot fetch from bucket")
    subprocess.run([gs, "cp", src, dst], check=True)


def resolve_cohort_csv(
    name: str = "complete",
    cache_dir: str | os.PathLike = ".",
    bucket: Optional[str] = None,
) -> Path:
    """Return a local path to the cached cohort CSV, fetching from bucket on miss.

    ``name`` is one of :data:`CACHE_FILENAMES` (``"complete"`` or
    ``"participant_level"``). The lookup order is:

      1. ``cache_dir/<canonical>.csv`` (or any accepted alias);
      2. ``$WORKSPACE_BUCKET/data/<canonical>.csv`` (or alias) -- copied
         into ``cache_dir`` via ``gsutil cp``;
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


def load_or_build_cohort(
    name: str = "complete",
    cache_dir: str | os.PathLike = ".",
    bucket: Optional[str] = None,
    builder: Optional[Callable[[], CohortBuildResult]] = None,
    upload_after_build: bool = False,
) -> Path:
    """Resolve a cached cohort CSV; build it on cache miss.

    Tries :func:`resolve_cohort_csv` first. If that misses and ``builder``
    is given, calls ``builder()`` (which must return a
    :class:`CohortBuildResult`), writes both CSVs to ``cache_dir``, and
    returns the path matching ``name``.
    """
    try:
        return resolve_cohort_csv(name=name, cache_dir=cache_dir, bucket=bucket)
    except FileNotFoundError:
        if builder is None:
            raise
        result = builder()
        pl, cc = write_cohort_cache(
            result, cache_dir=cache_dir, bucket=bucket, upload=upload_after_build
        )
        return cc if name == "complete" else pl


__all__ = [
    "COHORT_NODES",
    "COHORT_NODE_TYPES",
    "PLAUSIBILITY_BOUNDS",
    "CACHE_FILENAMES",
    "label_measurement_node",
    "attach_node_labels",
    "clean_measurements",
    "remove_extreme_outliers_iqr",
    "collapse_to_wide_median",
    "build_t2d_node",
    "build_cohort_dataset",
    "load_cohort_csv",
    "load_cohort_dataset",
    "resolve_cohort_csv",
    "write_cohort_cache",
    "load_or_build_cohort",
    "CohortBuildResult",
]
