"""Curated PGS Catalog panels for the genome-side crosscoder stream.

The default panel here -- ``CARDIOMETABOLIC_PANEL_V1`` -- is a snapshot from
the **2026-05-07** PGS Catalog bulk-metadata release, restricted to scores
whose development *and* evaluation metadata contain no ``AllofUs`` cohort
record (strict no-AoU rule). It is grouped by phenotypic area so that
downstream analysis can reweight, drop areas, or do per-area robustness
checks. Every ID resolves to harmonised scoring files on the EBI FTP via
:func:`pgs_catalog_url`; :func:`download_panel` fetches them in parallel
into a cache directory whose contents are directly consumable by
:func:`causal_pred.data.polygenic.score_panel`.

Provenance
----------
* Source: PGS Catalog bulk metadata, https://ftp.ebi.ac.uk/pub/databases/spot/pgs/metadata/
* Snapshot date: 2026-05-07.
* Inclusion scope: cardiometabolic (CAD/CHD/MI, T2D, lipids, BP, AF, stroke,
  HF/cardiomyopathy, obesity/adiposity, glycemic/insulin, VTE, PAD/AAA,
  metabolic syndrome / MASLD / NAFLD).
* Exclusion: any PGS ID whose development or evaluation metadata mentions
  cohort ``AllofUs``, or whose publication is explicitly titled around
  All of Us. Some otherwise-strong older T2D scores (PGS000014, PGS000330,
  PGS000729, PGS001781, PGS002243) fail this rule and are *not* in the panel.

If you have a private PGS catalog mirror on a biobank workspace bucket,
point :func:`download_panel` at it via the ``mirror_base`` argument; the
URL resolver substitutes that base and otherwise reuses the catalog's path
conventions.
"""

from __future__ import annotations

import concurrent.futures
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


CARDIOMETABOLIC_PANEL_V1: Dict[str, Tuple[str, ...]] = {
    "cad": (
        "PGS003725", "PGS000018", "PGS000013", "PGS004879", "PGS000116",
        "PGS002244", "PGS000329", "PGS000337",
    ),
    "t2d": (
        "PGS004870", "PGS004152", "PGS002308", "PGS000804", "PGS004602",
        "PGS004868", "PGS005341", "PGS005342", "PGS005343", "PGS005344",
    ),
    "ldl": (
        "PGS000888", "PGS000889", "PGS000892", "PGS000115", "PGS000814",
        "PGS003978", "PGS000887", "PGS000890", "PGS000891", "PGS000895",
        "PGS000897",
    ),
    "hdl": (
        "PGS002781", "PGS002782", "PGS004156", "PGS000686", "PGS004914",
        "PGS000671",
    ),
    "triglycerides": (
        "PGS002784", "PGS000699", "PGS004342", "PGS003401", "PGS000066",
    ),
    "total_cholesterol": (
        "PGS002783", "PGS000677", "PGS004333", "PGS002352", "PGS002718",
    ),
    "lpa_apob": (
        "PGS000667", "PGS000689", "PGS000752", "PGS004205", "PGS000672",
        "PGS002102",
    ),
    "blood_pressure": (
        "PGS000913", "PGS000912", "PGS004231", "PGS004232", "PGS005120",
        "PGS005350", "PGS005351", "PGS000706",
    ),
    "atrial_fibrillation": (
        "PGS005168", "PGS004878", "PGS000016", "PGS000035", "PGS005313",
        "PGS004613",
    ),
    "stroke": (
        "PGS000039", "PGS002724", "PGS002725", "PGS000911", "PGS005230",
        "PGS004154",
    ),
    "heart_failure": (
        "PGS001790", "PGS005097", "PGS012544", "PGS004861", "PGS004862",
        "PGS004948", "PGS004949", "PGS004910", "PGS004911", "PGS000739",
    ),
    "bmi_obesity_whr": (
        "PGS000027", "PGS005199", "PGS004150", "PGS003897", "PGS003893",
        "PGS003400", "PGS002356", "PGS005337", "PGS005338", "PGS005339",
        "PGS005340",
    ),
    "glycemic_insulin": (
        "PGS001350", "PGS001351", "PGS001352", "PGS004157", "PGS000684",
        "PGS000685", "PGS000877", "PGS002953",
    ),
    "vte": (
        "PGS000043", "PGS003332", "PGS001796", "PGS002235", "PGS002794",
    ),
    "pad_aaa_aorta": (
        "PGS012546", "PGS012563", "PGS003429", "PGS000753", "PGS002236",
        "PGS002267", "PGS002055",
    ),
    "metsyn_masld_nafld": (
        "PGS004928", "PGS002282", "PGS002245", "PGS000655", "PGS012549",
    ),
}


PANEL_PROVENANCE: Dict[str, str] = {
    "name": "cardiometabolic_v1",
    "snapshot_date": "2026-05-07",
    "source": "PGS Catalog bulk metadata",
    "exclusion_rule": (
        "strict no-AllofUs: any PGS whose development or evaluation cohort "
        "metadata mentions 'AllofUs' is excluded; AoU-titled publications "
        "are excluded"
    ),
    "scope": (
        "cardiometabolic: CAD/CHD/MI, T2D, lipids (LDL/HDL/TG/TC/Lp(a)/ApoB), "
        "BP/hypertension, AF, stroke, HF/cardiomyopathy, "
        "obesity/adiposity/WHR, glycemic/insulin traits, VTE, PAD/AAA, "
        "metabolic syndrome / MASLD / NAFLD"
    ),
    "excluded_strong_older_t2d_scores": (
        "PGS000014, PGS000330, PGS000729, PGS001781, PGS002243"
    ),
}


def all_panel_ids(panel: Optional[Dict[str, Tuple[str, ...]]] = None) -> Tuple[str, ...]:
    """Flat tuple of unique PGS IDs in panel, preserving area order then internal order."""
    p = panel if panel is not None else CARDIOMETABOLIC_PANEL_V1
    seen: set[str] = set()
    out: List[str] = []
    for ids in p.values():
        for pgs_id in ids:
            if pgs_id in seen:
                continue
            seen.add(pgs_id)
            out.append(pgs_id)
    return tuple(out)


def panel_area_for(pgs_id: str, panel: Optional[Dict[str, Tuple[str, ...]]] = None) -> Optional[str]:
    """Return the panel-area key whose list contains ``pgs_id``, or ``None``."""
    p = panel if panel is not None else CARDIOMETABOLIC_PANEL_V1
    for area, ids in p.items():
        if pgs_id in ids:
            return area
    return None


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


_PGS_CATALOG_FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"


def pgs_catalog_url(
    pgs_id: str,
    *,
    build: str = "GRCh38",
    harmonised: bool = True,
    base: Optional[str] = None,
) -> str:
    """Return the FTP URL of a PGS Catalog scoring file.

    Parameters
    ----------
    pgs_id : str
        PGS Catalog ID, e.g. ``"PGS000014"``.
    build : str
        ``"GRCh38"`` (default) or ``"GRCh37"``. Only used when
        ``harmonised=True``.
    harmonised : bool
        If True (default) returns the harmonised scoring file path; if False
        returns the original-build raw scoring file.
    base : str, optional
        Override the FTP base. Useful for biobank-workspace mirrors that
        replicate the catalog directory layout under a different prefix
        (``gs://`` URIs work with gsutil-aware downloaders).

    Returns
    -------
    str
        Direct URL to ``PGS<id>.txt.gz`` (raw) or
        ``PGS<id>_hm<build>.txt.gz`` (harmonised).
    """
    if not pgs_id.startswith("PGS"):
        raise ValueError(f"expected PGS Catalog ID like 'PGS000014', got {pgs_id!r}")
    if build not in {"GRCh37", "GRCh38"}:
        raise ValueError(f"build must be GRCh37 or GRCh38, got {build!r}")
    root = (base or _PGS_CATALOG_FTP_BASE).rstrip("/")
    if harmonised:
        return (
            f"{root}/{pgs_id}/ScoringFiles/Harmonized/"
            f"{pgs_id}_hmPOS_{build}.txt.gz"
        )
    return f"{root}/{pgs_id}/ScoringFiles/{pgs_id}.txt.gz"


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class PanelDownloadError(RuntimeError):
    """Raised when one or more PGS scoring files cannot be fetched."""


def _download_one(url: str, dst: Path, timeout: int) -> Path:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
        os.replace(tmp, dst)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise PanelDownloadError(f"failed to download {url}: {exc!r}") from exc
    return dst


def download_panel(
    out_dir: str | os.PathLike,
    *,
    ids: Optional[Sequence[str]] = None,
    panel: Optional[Dict[str, Tuple[str, ...]]] = None,
    build: str = "GRCh38",
    harmonised: bool = True,
    mirror_base: Optional[str] = None,
    n_workers: int = 8,
    timeout: int = 600,
    skip_existing: bool = True,
) -> List[Path]:
    """Download every PGS scoring file in the panel into ``out_dir``.

    Files are placed flat under ``out_dir`` named
    ``<pgs_id>_hmPOS_<build>.txt.gz`` (harmonised) or
    ``<pgs_id>.txt.gz`` (raw); this layout is directly consumable by
    :func:`causal_pred.data.polygenic.score_panel`.

    Parameters
    ----------
    out_dir : path
        Destination directory; created if missing.
    ids : sequence of str, optional
        Subset of PGS IDs to fetch. Defaults to :func:`all_panel_ids`
        of ``panel``.
    panel : dict, optional
        Panel definition. Defaults to :data:`CARDIOMETABOLIC_PANEL_V1`.
    build : str
        ``GRCh38`` (default) or ``GRCh37``.
    harmonised : bool
        Whether to fetch harmonised scoring files. Default True.
    mirror_base : str, optional
        Override of the FTP base, e.g. an internal mirror.
    n_workers : int
        Concurrent download threads.
    timeout : int
        Per-file network timeout (seconds).
    skip_existing : bool
        Skip files that already exist with non-zero size. Default True.

    Returns
    -------
    list of Path
        Local paths to every downloaded score file, in input ID order.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_ids = list(ids) if ids is not None else list(
        all_panel_ids(panel)
    )
    if not panel_ids:
        return []

    suffix = f"_hmPOS_{build}.txt.gz" if harmonised else ".txt.gz"

    def _job(pgs_id: str) -> Path:
        url = pgs_catalog_url(
            pgs_id, build=build, harmonised=harmonised, base=mirror_base
        )
        dst = out / f"{pgs_id}{suffix}"
        if skip_existing and dst.is_file() and dst.stat().st_size > 0:
            return dst
        return _download_one(url, dst, timeout=timeout)

    paths: List[Path] = [out / f"{pid}{suffix}" for pid in panel_ids]
    errors: list[BaseException] = []
    if n_workers <= 1 or len(panel_ids) == 1:
        for pid in panel_ids:
            try:
                _job(pid)
            except PanelDownloadError as exc:
                errors.append(exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers
        ) as ex:
            futs = {ex.submit(_job, pid): pid for pid in panel_ids}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except PanelDownloadError as exc:
                    errors.append(exc)

    if errors:
        joined = "\n  - ".join(repr(e) for e in errors)
        raise PanelDownloadError(
            f"{len(errors)}/{len(panel_ids)} downloads failed:\n  - {joined}"
        )
    return paths


def discover_local_panel(
    score_dir: str | os.PathLike,
    *,
    ids: Optional[Sequence[str]] = None,
    panel: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> Tuple[List[Path], List[str]]:
    """Inspect an on-disk score directory and report which panel IDs are present.

    Returns
    -------
    found : list of Path
        Local paths to score files matching panel IDs (any extension that
        starts with ``<PGS_ID>``).
    missing : list of str
        Panel IDs with no matching file in ``score_dir``.
    """
    d = Path(score_dir)
    panel_ids = list(ids) if ids is not None else list(
        all_panel_ids(panel)
    )
    on_disk = list(d.iterdir()) if d.is_dir() else []
    found: List[Path] = []
    missing: List[str] = []
    for pid in panel_ids:
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
    "CARDIOMETABOLIC_PANEL_V1",
    "PANEL_PROVENANCE",
    "PanelDownloadError",
    "all_panel_ids",
    "discover_local_panel",
    "download_panel",
    "panel_area_for",
    "pgs_catalog_url",
]
