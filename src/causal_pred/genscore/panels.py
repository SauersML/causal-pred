"""Curated PGS Catalog panel for the genome-side crosscoder stream.

Flat tuple of PGS IDs (no AoU contamination, screened from the 2026-05-07
PGS Catalog snapshot). Resolve to harmonised scoring files on the EBI FTP
via :func:`pgs_catalog_url`; :func:`download_panel` fetches them in parallel
into a directory consumable by :func:`causal_pred.data.polygenic.score_panel`.
"""

from __future__ import annotations

import concurrent.futures
import codecs
import gzip
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence


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


class _InvalidScoreFile(RuntimeError):
    """Raised when a downloaded PGS score file cannot be read by gnomon."""


def pgs_catalog_url(pgs_id: str, build: str = "GRCh38") -> str:
    """URL of the harmonised scoring file for ``pgs_id`` on EBI FTP."""
    return (
        f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs_id}/"
        f"ScoringFiles/Harmonized/{pgs_id}_hmPOS_{build}.txt.gz"
    )


def _validate_score_file(path: Path) -> None:
    """Validate a decompressed PGS score file as strict UTF-8 text."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise _InvalidScoreFile(
            f"{path} is not a valid UTF-8 score file: {exc}"
        ) from exc


def _decompress_score_file(src_gz: Path, dst_txt: Path) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with gzip.open(src_gz, "rb") as src, dst_txt.open("wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                decoder.decode(chunk)
                dst.write(chunk)
            decoder.decode(b"", final=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise _InvalidScoreFile(
            f"{src_gz} does not contain a valid gzip UTF-8 score file: {exc}"
        ) from exc


def _download_one(
    url: str,
    dst: Path,
    timeout: int,
    stale_paths: Sequence[Path] = (),
) -> tuple[Path, str, int, float]:
    """Return ``(path, status, bytes_written, elapsed_seconds)``.

    ``status`` is ``"cached"`` if the existing file validated, otherwise
    ``"downloaded"``.
    """
    t0 = time.time()
    for stale in stale_paths:
        if stale != dst and stale.exists():
            stale.unlink()
    if dst.is_file() and dst.stat().st_size > 0:
        try:
            _validate_score_file(dst)
            return dst, "cached", dst.stat().st_size, time.time() - t0
        except _InvalidScoreFile:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    compressed_tmp = dst.with_suffix(dst.suffix + ".gz.part")
    text_tmp = dst.with_suffix(dst.suffix + ".part")
    for tmp in (compressed_tmp, text_tmp):
        if tmp.exists():
            tmp.unlink()
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with open(compressed_tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                bytes_written += len(chunk)
    try:
        _decompress_score_file(compressed_tmp, text_tmp)
    finally:
        if compressed_tmp.exists():
            compressed_tmp.unlink()
    os.replace(text_tmp, dst)
    return dst, "downloaded", bytes_written, time.time() - t0


def download_panel(
    out_dir: str | os.PathLike,
    ids: Sequence[str] = PGS_PANEL,
    build: str = "GRCh38",
    n_workers: int = 8,
    timeout: int = 600,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Download every PGS scoring file in ``ids`` into ``out_dir`` in parallel.

    Files are downloaded from the PGS Catalog ``*.txt.gz`` endpoints but are
    cached as decompressed ``<pgs_id>_hmPOS_<build>.txt`` files, because
    gnomon 0.1.2 rejects the catalog gzip streams as non-UTF-8. Existing text
    files are reused only after strict UTF-8 validation.

    If ``progress`` is given, it is called with human-readable status strings
    as each scoring file is resolved (cached or downloaded), plus a final
    summary line.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text_suffix = f"_hmPOS_{build}.txt"
    gzip_suffix = f"{text_suffix}.gz"
    total = len(ids)

    def _emit(msg: str) -> None:
        if progress is not None:
            progress(msg)

    _emit(
        f"download_panel: {total} PGS scoring files into {out} "
        f"(workers={n_workers}, timeout={timeout}s, build={build})"
    )

    def _job(pgs_id: str) -> tuple[str, Path, str, int, float]:
        path, status, nbytes, elapsed = _download_one(
            pgs_catalog_url(pgs_id, build=build),
            out / f"{pgs_id}{text_suffix}",
            timeout=timeout,
            stale_paths=(out / f"{pgs_id}{gzip_suffix}",),
        )
        return pgs_id, path, status, nbytes, elapsed

    paths: List[Path] = []
    errors: list[BaseException] = []
    cached_n = 0
    downloaded_n = 0
    bytes_downloaded = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_id = {ex.submit(_job, pid): pid for pid in ids}
        for completed, fut in enumerate(
            concurrent.futures.as_completed(future_to_id), start=1
        ):
            pgs_id = future_to_id[fut]
            try:
                pid, path, status, nbytes, elapsed = fut.result()
            except (
                urllib.error.URLError,
                OSError,
                TimeoutError,
                _InvalidScoreFile,
            ) as exc:
                errors.append(exc)
                _emit(
                    f"[{completed}/{total}] FAILED {pgs_id}: {exc!r}"
                )
                continue
            paths.append(path)
            if status == "cached":
                cached_n += 1
            else:
                downloaded_n += 1
                bytes_downloaded += nbytes
            _emit(
                f"[{completed}/{total}] {status} {pid} "
                f"({nbytes / (1 << 20):.2f} MiB in {elapsed:.1f}s) "
                f"[cached={cached_n} downloaded={downloaded_n} failed={len(errors)}]"
            )

    _emit(
        f"download_panel: done {len(paths)}/{total} files "
        f"(cached={cached_n} downloaded={downloaded_n} failed={len(errors)} "
        f"bytes_downloaded={bytes_downloaded / (1 << 20):.1f} MiB "
        f"elapsed={time.time() - started:.1f}s)"
    )

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
            (
                p
                for p in on_disk
                if p.is_file() and p.name.startswith(pid) and p.suffix == ".txt"
            ),
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
