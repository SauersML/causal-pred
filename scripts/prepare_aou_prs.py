"""Prepare the AoU microarray PRS matrix for the real pipeline.

This script is intentionally AoU-specific and has no command-line flags. It
uses the workspace bucket only for small/reusable intermediates:

* reads the cohort CSV from ``$WORKSPACE_BUCKET/data`` via ``resolve_cohort_csv``;
* downloads/verifies ``arrays.{bed,bim,fam}`` locally only, never uploading it;
* downloads the public PGS Catalog panel locally;
* runs ``gnomon score`` on the AoU microarray data;
* filters the score matrix to the cohort participants;
* caches ``data/aou_prs_panel.csv.gz`` in
  ``$WORKSPACE_BUCKET/intermediates/causal-pred/``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from causal_pred.data.cohort import (
    discover_genotype_dir,
    resolve_aou_genotypes,
    resolve_cohort_csv,
)
from causal_pred.data.polygenic import score_panel
from causal_pred.genscore.panels import download_panel


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PRS_PATH = DATA_DIR / "aou_prs_panel.csv.gz"
PGS_PANEL_DIR = DATA_DIR / "pgs_panel"
GNOMON_OUT_DIR = DATA_DIR / "gnomon_score"
BUCKET_CACHE_PATH = "intermediates/causal-pred/aou_prs_panel.csv.gz"


def _workspace_bucket() -> str:
    bucket = os.environ.get("WORKSPACE_BUCKET", "").rstrip("/")
    if not bucket:
        raise RuntimeError("WORKSPACE_BUCKET is not set")
    return bucket


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True)


def _gsutil_exists(uri: str) -> bool:
    return subprocess.run(["gsutil", "-q", "stat", uri], check=False).returncode == 0


def _copy_from_bucket(uri: str, dst: Path) -> bool:
    if not _gsutil_exists(uri):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    if tmp.exists():
        tmp.unlink()
    _run(["gsutil", "cp", uri, str(tmp)])
    os.replace(tmp, dst)
    return True


def _upload_to_bucket(src: Path, uri: str) -> None:
    _run(["gsutil", "cp", str(src), uri])


def _cohort_person_ids(cohort_csv: Path) -> pd.Series:
    df = pd.read_csv(cohort_csv, usecols=["person_id"], dtype={"person_id": "string"})
    ids = df["person_id"].astype("string")
    if ids.empty:
        raise RuntimeError(f"cohort CSV has no rows: {cohort_csv}")
    return ids


def _cached_prs_usable(path: Path, person_ids: pd.Series) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    ids = pd.read_csv(path, usecols=[0], dtype={0: "string"}).iloc[:, 0].astype("string")
    return int(person_ids.isin(set(ids.astype(str))).sum()) >= 100


def _resolve_microarray_bed() -> Path:
    hit = discover_genotype_dir([Path.home(), REPO_ROOT / "genomes"])
    geno_dir = hit if hit is not None else Path.home()
    bed = resolve_aou_genotypes(cache_dir=geno_dir)
    print(f"microarray ready: {bed}")
    return bed


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bucket = _workspace_bucket()
    bucket_prs_uri = f"{bucket}/{BUCKET_CACHE_PATH}"
    cohort_csv = resolve_cohort_csv(name="complete", cache_dir=DATA_DIR)
    person_ids = _cohort_person_ids(cohort_csv)

    if _cached_prs_usable(PRS_PATH, person_ids):
        print(f"PRS cache ready: {PRS_PATH}")
        return 0

    if _copy_from_bucket(bucket_prs_uri, PRS_PATH) and _cached_prs_usable(PRS_PATH, person_ids):
        print(f"downloaded cached PRS matrix: {bucket_prs_uri} -> {PRS_PATH}")
        return 0

    bed = _resolve_microarray_bed()

    print(f"downloading PGS panel into {PGS_PANEL_DIR}")
    download_panel(PGS_PANEL_DIR)

    print("running gnomon score on AoU microarray data")
    scores = score_panel(
        genotype_path=str(bed),
        score_path=str(PGS_PANEL_DIR),
        out_dir=str(GNOMON_OUT_DIR),
        n_threads=os.cpu_count(),
        timeout=24 * 60 * 60,
    )
    scores.index = scores.index.astype(str)

    overlap = person_ids.isin(scores.index)
    n_overlap = int(overlap.sum())
    if n_overlap < 100:
        raise RuntimeError(
            f"only {n_overlap} cohort participants overlap the gnomon score output"
        )

    cohort_scores = scores.reindex(person_ids.astype(str))
    cohort_scores = cohort_scores.dropna(axis=1, how="all")
    if cohort_scores.shape[1] == 0:
        raise RuntimeError("gnomon produced no usable PRS columns after cohort alignment")

    tmp = PRS_PATH.with_name(PRS_PATH.name + ".part")
    if tmp.exists():
        tmp.unlink()
    cohort_scores.to_csv(tmp, index_label="person_id", compression="gzip")
    os.replace(tmp, PRS_PATH)
    print(
        f"PRS matrix ready: {PRS_PATH} "
        f"rows={cohort_scores.shape[0]} overlap={n_overlap} cols={cohort_scores.shape[1]}"
    )

    _upload_to_bucket(PRS_PATH, bucket_prs_uri)
    print(f"cached PRS matrix in workspace bucket: {bucket_prs_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
