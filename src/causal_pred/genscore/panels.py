"""Curated PGS Catalog panel for the genome-side crosscoder stream.

Flat tuple of PGS IDs (no AoU contamination, screened from the 2026-05-07
PGS Catalog snapshot). Resolve to harmonised scoring files on the EBI FTP
via :func:`pgs_catalog_url`; :func:`download_panel` fetches them in parallel
into a directory consumable by :func:`causal_pred.data.polygenic.score_panel`.
"""

from __future__ import annotations

import concurrent.futures
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Sequence


PGS_PANEL: tuple[str, ...] = (
    "PGS003725", "PGS000018", "PGS000013", "PGS004879", "PGS000116",
    "PGS002244", "PGS000329", "PGS000337",
    "PGS004870", "PGS004152", "PGS002308", "PGS000804", "PGS004602",
    "PGS004868", "PGS005341", "PGS005342", "PGS005343", "PGS005344",
    "PGS000888", "PGS000889", "PGS000892", "PGS000115", "PGS000814",
    "PGS003978", "PGS000887", "PGS000890", "PGS000891", "PGS000895",
    "PGS000897",
    "PGS002781", "PGS002782", "PGS004156", "PGS000686", "PGS004914",
    "PGS000671",
    "PGS002784", "PGS000699", "PGS004342", "PGS003401", "PGS000066",
    "PGS002783", "PGS000677", "PGS004333", "PGS002352", "PGS002718",
    "PGS000667", "PGS000689", "PGS000752", "PGS004205", "PGS000672",
    "PGS002102",
    "PGS000913", "PGS000912", "PGS004231", "PGS004232", "PGS005120",
    "PGS005350", "PGS005351", "PGS000706",
    "PGS005168", "PGS004878", "PGS000016", "PGS000035", "PGS005313",
    "PGS004613",
    "PGS000039", "PGS002724", "PGS002725", "PGS000911", "PGS005230",
    "PGS004154",
    "PGS001790", "PGS005097", "PGS012544", "PGS004861", "PGS004862",
    "PGS004948", "PGS004949", "PGS004910", "PGS004911", "PGS000739",
    "PGS000027", "PGS005199", "PGS004150", "PGS003897", "PGS003893",
    "PGS003400", "PGS002356", "PGS005337", "PGS005338", "PGS005339",
    "PGS005340",
    "PGS001350", "PGS001351", "PGS001352", "PGS004157", "PGS000684",
    "PGS000685", "PGS000877", "PGS002953",
    "PGS000043", "PGS003332", "PGS001796", "PGS002235", "PGS002794",
    "PGS012546", "PGS012563", "PGS003429", "PGS000753", "PGS002236",
    "PGS002267", "PGS002055",
    "PGS004928", "PGS002282", "PGS002245", "PGS000655", "PGS012549",
)


def pgs_catalog_url(pgs_id: str, build: str = "GRCh38") -> str:
    """URL of the harmonised scoring file for ``pgs_id`` on EBI FTP."""
    return (
        f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs_id}/"
        f"ScoringFiles/Harmonized/{pgs_id}_hmPOS_{build}.txt.gz"
    )


def _download_one(url: str, dst: Path, timeout: int) -> Path:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    os.replace(tmp, dst)
    return dst


def download_panel(
    out_dir: str | os.PathLike,
    ids: Sequence[str] = PGS_PANEL,
    build: str = "GRCh38",
    n_workers: int = 8,
    timeout: int = 600,
) -> List[Path]:
    """Download every PGS scoring file in ``ids`` into ``out_dir`` in parallel.

    Files are named ``<pgs_id>_hmPOS_<build>.txt.gz`` so the directory is
    consumable by :func:`causal_pred.data.polygenic.score_panel`. Files
    already present with non-zero size are skipped.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_hmPOS_{build}.txt.gz"

    def _job(pgs_id: str) -> Path:
        return _download_one(
            pgs_catalog_url(pgs_id, build=build),
            out / f"{pgs_id}{suffix}",
            timeout=timeout,
        )

    paths: List[Path] = []
    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for fut in concurrent.futures.as_completed(
            {ex.submit(_job, pid): pid for pid in ids}
        ):
            try:
                paths.append(fut.result())
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                errors.append(exc)

    if errors:
        raise RuntimeError(
            f"{len(errors)}/{len(ids)} PGS downloads failed: {errors[0]!r} (and {len(errors) - 1} more)"
        )
    return paths


def discover_local_panel(
    score_dir: str | os.PathLike,
    ids: Sequence[str] = PGS_PANEL,
) -> tuple[List[Path], List[str]]:
    """Return ``(found_paths, missing_ids)`` for ``ids`` against ``score_dir``."""
    d = Path(score_dir)
    on_disk = list(d.iterdir()) if d.is_dir() else []
    found: List[Path] = []
    missing: List[str] = []
    for pid in ids:
        hit = next(
            (p for p in on_disk if p.is_file() and p.name.startswith(pid)),
            None,
        )
        if hit is None:
            missing.append(pid)
        else:
            found.append(hit)
    return found, missing


__all__ = [
    "PGS_PANEL",
    "discover_local_panel",
    "download_panel",
    "pgs_catalog_url",
]
