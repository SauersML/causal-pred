"""Wrapper around the ``gnomon`` CLI for polygenic scoring and ancestry PCA.

This module shells out to the ``gnomon`` binary (``/Users/user/.local/bin/gnomon``
in the development environment; resolved at call-time via ``shutil.which``)
and parses the on-disk artefacts it produces.  We intentionally do NOT
reimplement any of gnomon's numerical work in Python — gnomon already does
natively-parallel scoring and block-streamed PCA across variants and samples.

gnomon sub-commands used here
-----------------------------
* ``gnomon score <SCORE_PATH> <GENOTYPE_PATH>``
      Writes ``<genotype_prefix>.sscore`` with a header of the form
      ``#IID  <name>_AVG  <name>_MISSING_PCT ...`` (tab-separated).  Optional
      ``#REGION`` comment lines may precede the column header.
* ``gnomon fit --components <N> <GENOTYPE_PATH>``
      Writes ``hwe.json`` (+ ``samples.tsv``, ``hwe_summary.tsv``) next to the
      genotype data.
* ``gnomon project <GENOTYPE_PATH>``
      Reads ``hwe.json`` next to the genotype data and writes
      ``projection_scores.bin`` + ``projection_scores.metadata.json``.
      The binary starts with a 32-byte header ("GNPRJ001" magic, u32 version=3,
      u64 rows, u64 cols, u32 reserved), followed by ``rows*cols`` little-endian
      ``f64`` values in **column-major** order, followed by an embedded
      row-id (IID) section with magic ``GNPSID01``.
* ``gnomon terms --sex <GENOTYPE_PATH>``
      Writes ``sex.tsv`` next to the genotype data.  Columns:
      ``IID Build Sex Y_Density X_AutoHet_Ratio Composite_Index ...``

Because gnomon always drops its outputs *next to the genotype data*, we copy
or symlink the input into a fresh directory when callers pass ``out_dir=None``
so that the user's source directory is never polluted.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Collection, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# public errors
# ---------------------------------------------------------------------------


class PolygenicToolMissing(RuntimeError):
    """Raised when the ``gnomon`` binary cannot be located on ``$PATH``."""


class PolygenicRunError(RuntimeError):
    """Raised when a gnomon invocation exits non-zero or produces no output."""


# Standard PGS column names understood by the synthetic dataset.
_SYNTHETIC_PGS_COLUMNS: tuple[str, ...] = (
    "PGS_T2D",
    "PGS_BMI",
    "PGS_LDL",
    "PGS_HbA1c",
)


# ---------------------------------------------------------------------------
# binary discovery / invocation helpers
# ---------------------------------------------------------------------------


def _locate_gnomon() -> str:
    """Return the absolute path to the ``gnomon`` binary or raise."""
    path = shutil.which("gnomon")
    if path:
        return path
    fallback = "/Users/user/.local/bin/gnomon"
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    raise PolygenicToolMissing(
        "Could not find the `gnomon` CLI on $PATH. Install it from "
        "https://github.com/SauersML/gnomon or set PATH to include the "
        "binary before calling causal_pred.data.polygenic.*"
    )


def gnomon_available() -> bool:
    """Lightweight check used by tests/CLI to decide whether to skip."""
    try:
        _locate_gnomon()
    except PolygenicToolMissing:
        return False
    return True


def _format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, rem_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(rem_minutes):02d}m{rem:04.1f}s"


def _file_size_mib(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def _fam_iids(genotype_path: Path) -> set[str] | None:
    fam_path = genotype_path.with_suffix(".fam")
    if not fam_path.is_file():
        return None
    iids: set[str] = set()
    with fam_path.open("r") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) >= 2:
                iids.add(fields[1])
    return iids


def _write_keep_file(
    path: Path,
    keep_iids: Collection[str],
    genotype_path: Path,
) -> tuple[int, int]:
    requested = {str(iid) for iid in keep_iids}
    fam_iids = _fam_iids(genotype_path)
    matched = requested if fam_iids is None else requested & fam_iids
    if not matched:
        raise PolygenicRunError(
            "no requested keep_iids were found in the genotype .fam file"
        )
    with path.open("w") as fh:
        for iid in sorted(matched):
            fh.write(f"{iid}\n")
    return len(matched), len(requested)


def _run(
    cmd: Sequence[str],
    timeout: int,
    *,
    env: Mapping[str, str] | None = None,
    label: str = "gnomon invocation",
) -> subprocess.CompletedProcess:
    """Run gnomon with stdout/stderr inherited by the caller's terminal."""
    print(f"[gnomon] start {' '.join(cmd)}", flush=True)
    started_at = time.time()
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise PolygenicRunError(
            f"{label} timed out after {timeout}s "
            f"(elapsed={_format_seconds(time.time() - started_at)}): "
            f"{' '.join(cmd)!s}"
        ) from exc
    except FileNotFoundError as exc:
        raise PolygenicRunError(
            f"{label} could not start (binary not found on PATH): {' '.join(cmd)!s}"
        ) from exc
    except OSError as exc:
        raise PolygenicRunError(
            f"{label} failed to spawn ({type(exc).__name__}: {exc}): "
            f"{' '.join(cmd)!s}"
        ) from exc
    if proc.returncode != 0:
        raise PolygenicRunError(
            f"{label} failed exit_code={proc.returncode} "
            f"elapsed={_format_seconds(time.time() - started_at)}: "
            f"{' '.join(cmd)!s}"
        )
    print(
        f"[gnomon] done {label} elapsed={_format_seconds(time.time() - started_at)}",
        flush=True,
    )
    return proc


def _materialise_genotype(src: str, dst_dir: Path) -> Path:
    """Copy (or symlink) the genotype dataset to ``dst_dir`` and return the
    canonical path that gnomon should be invoked with.

    gnomon writes its outputs next to the genotype data, so we isolate inputs
    in a scratch directory whenever the caller did not request a specific
    ``out_dir``.  For PLINK triples we also copy ``.bim`` + ``.fam`` siblings.
    """
    src_path = Path(src)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        # Might be a PLINK prefix without the extension.
        candidate = src_path.with_suffix(".bed")
        if candidate.exists():
            src_path = candidate
        else:
            raise FileNotFoundError(f"genotype path does not exist: {src}")

    suffix = src_path.suffix.lower()
    if suffix == ".bed":
        prefix = src_path.with_suffix("")
        for ext in (".bed", ".bim", ".fam"):
            sib = prefix.with_suffix(ext)
            if not sib.is_file():
                raise FileNotFoundError(f"incomplete PLINK fileset: missing {sib}")
            link = dst_dir / sib.name
            if not link.exists():
                try:
                    os.symlink(sib, link)
                except OSError:
                    shutil.copy2(sib, link)
        return dst_dir / src_path.name

    if suffix in (".vcf", ".bcf") or src_path.name.endswith(".vcf.gz"):
        link = dst_dir / src_path.name
        if not link.exists():
            try:
                os.symlink(src_path, link)
            except OSError:
                shutil.copy2(src_path, link)
        # index sidecar if present
        for idx_ext in (".tbi", ".csi"):
            idx = src_path.with_suffix(src_path.suffix + idx_ext)
            if idx.is_file():
                idx_link = dst_dir / idx.name
                if not idx_link.exists():
                    try:
                        os.symlink(idx, idx_link)
                    except OSError:
                        shutil.copy2(idx, idx_link)
        return link

    # Unknown / text DTC-style input: just symlink the single file.
    link = dst_dir / src_path.name
    if not link.exists():
        try:
            os.symlink(src_path, link)
        except OSError:
            shutil.copy2(src_path, link)
    return link


# ---------------------------------------------------------------------------
# output-parsing helpers (exported for testing)
# ---------------------------------------------------------------------------


def parse_sscore(
    sscore_path: str | os.PathLike,
    *,
    keep_iids: Collection[str] | None = None,
    chunksize: int = 200_000,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Parse a ``*.sscore`` TSV emitted by ``gnomon score``.

    The file may begin with ``#REGION ...`` comment lines.  The actual column
    header is the first line starting with ``#IID``.  The returned frame is
    indexed by string ``IID`` with one ``float64`` column per score (the
    ``*_AVG`` column; any ``*_MISSING_PCT`` columns are not read). When
    ``keep_iids`` is provided, rows are filtered while streaming so callers do
    not materialise the whole scored biobank before cohort alignment.
    """
    path = Path(sscore_path)
    header_line = None
    skip = 0
    with path.open("r") as fh:
        for i, line in enumerate(fh):
            if line.startswith("#IID"):
                header_line = line.rstrip("\n")
                skip = i
                break
    if header_line is None:
        raise PolygenicRunError(f"no #IID header row in {path}")

    columns = header_line.lstrip("#").split("\t")
    avg_cols = [c for c in columns if c.endswith("_AVG")]
    read_cols = ["IID", *avg_cols]
    dtype = {"IID": "string", **{c: "float64" for c in avg_cols}}
    keep_set = {str(iid) for iid in keep_iids} if keep_iids is not None else None

    # Stream rows in chunks so cohort-filtered production runs do not keep the
    # full scored biobank in memory. Only AVG score columns are read; gnomon's
    # per-score missingness columns are intentionally skipped.
    frames: list[pd.DataFrame] = []
    rows_seen = 0
    rows_kept = 0
    last_progress_at = time.time()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        skiprows=skip + 1,
        comment=None,
        usecols=read_cols,
        dtype=dtype,
        chunksize=int(chunksize),
    ):
        rows_seen += int(chunk.shape[0])
        if keep_set is not None:
            chunk = chunk[chunk["IID"].isin(keep_set)]
            if chunk.empty:
                now = time.time()
                if progress is not None and now - last_progress_at >= 10.0:
                    progress(
                        f"parse progress rows_seen={rows_seen} rows_kept={rows_kept}"
                    )
                    last_progress_at = now
                continue
        rows_kept += int(chunk.shape[0])
        frames.append(chunk)
        now = time.time()
        if progress is not None and now - last_progress_at >= 10.0:
            progress(f"parse progress rows_seen={rows_seen} rows_kept={rows_kept}")
            last_progress_at = now
    df = (
        pd.concat(frames, axis=0, ignore_index=True)
        if frames
        else pd.DataFrame(columns=read_cols).astype(dtype)
    )

    df["IID"] = df["IID"].astype("string")
    out = df[["IID"] + avg_cols].copy()
    # Strip the ``_AVG`` suffix so the frame is indexed by the user's score
    # label (matches the file stem of the input score file by default).
    out.columns = ["IID"] + [c[: -len("_AVG")] for c in avg_cols]
    out = out.set_index("IID")
    return out


def parse_projection_bin(bin_path: str | os.PathLike) -> pd.DataFrame:
    """Parse ``projection_scores.bin`` emitted by ``gnomon project``.

    Returns a DataFrame indexed by ``IID`` (string) with columns
    ``PC1 .. PCn`` (``float64``), in the input sample order.
    """
    path = Path(bin_path)
    data = path.read_bytes()
    if len(data) < 32:
        raise PolygenicRunError(f"projection file too short: {path}")
    magic = data[:8]
    if magic != b"GNPRJ001":
        raise PolygenicRunError(
            f"bad projection magic in {path}: expected b'GNPRJ001', got {magic!r}"
        )
    version = struct.unpack_from("<I", data, 8)[0]
    rows = struct.unpack_from("<Q", data, 12)[0]
    cols = struct.unpack_from("<Q", data, 20)[0]
    # reserved u32 at offset 28
    if version != 3:
        # Forward-compat: warn but still try.  Raise if the header layout is
        # likely incompatible.
        raise PolygenicRunError(
            f"unsupported projection matrix version {version} in {path}"
        )

    header_len = 32
    score_bytes = rows * cols * 8
    if header_len + score_bytes > len(data):
        raise PolygenicRunError(f"projection payload truncated in {path}")
    mat = (
        np.frombuffer(data, dtype="<f8", count=rows * cols, offset=header_len)
        .reshape((cols, rows))
        .T
    )  # column-major on disk -> transpose
    mat = np.ascontiguousarray(mat, dtype=np.float64)

    # Row-id section follows immediately after the matrix payload.
    rid_off = header_len + score_bytes
    if rid_off + 32 > len(data):
        raise PolygenicRunError(f"missing row-id header in {path}")
    rid_magic = data[rid_off : rid_off + 8]
    if rid_magic != b"GNPSID01":
        raise PolygenicRunError(
            f"bad row-id magic in {path}: expected b'GNPSID01', got {rid_magic!r}"
        )
    count = struct.unpack_from("<Q", data, rid_off + 16)[0]
    str_bytes = struct.unpack_from("<Q", data, rid_off + 24)[0]
    offsets_off = rid_off + 32
    n_offsets = count + 1
    offsets = np.frombuffer(data, dtype="<u8", count=n_offsets, offset=offsets_off)
    str_tbl_off = offsets_off + n_offsets * 8
    str_tbl = data[str_tbl_off : str_tbl_off + str_bytes]
    iids: list[str] = []
    for i in range(count):
        s, e = int(offsets[i]), int(offsets[i + 1])
        iids.append(str_tbl[s:e].decode("utf-8"))
    if len(iids) != rows:
        raise PolygenicRunError(
            f"projection row-id count ({len(iids)}) != matrix rows ({rows}) in {path}"
        )

    cols_names = [f"PC{i + 1}" for i in range(cols)]
    df = pd.DataFrame(mat, columns=cols_names)
    df.insert(0, "IID", pd.array(iids, dtype="string"))
    df = df.set_index("IID")
    return df


def parse_sex_tsv(sex_tsv_path: str | os.PathLike) -> pd.DataFrame:
    """Parse ``sex.tsv`` produced by ``gnomon terms --sex``.

    Returns a DataFrame indexed by string ``IID`` with at minimum a ``Sex``
    column (values: ``male``/``female``/``unknown``) and the supporting numeric
    diagnostics preserved as ``float64`` (``Y_Density``, ``X_AutoHet_Ratio``,
    ``Composite_Index``).  Missing diagnostic values (written as ``NA`` by
    gnomon) are represented as ``NaN``.
    """
    df = pd.read_csv(sex_tsv_path, sep="\t", dtype={"IID": "string"})
    df["IID"] = df["IID"].astype("string")
    for c in ("Y_Density", "X_AutoHet_Ratio", "Composite_Index"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.set_index("IID")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def score_cohort(
    genotype_path: str,
    score_files: list[str],
    out_dir: str | None = None,
    n_threads: int | None = None,
    timeout: int = 600,
) -> pd.DataFrame:
    """Run ``gnomon score`` on ``genotype_path`` against every score file in
    ``score_files`` and return a single ``DataFrame`` indexed by sample IID
    with one ``float64`` column per input score file (the column name is the
    score file's stem).

    Parameters
    ----------
    genotype_path: path to a PLINK prefix / ``.bed`` / VCF / BCF input.
    score_files:   one or more PGS Catalog-formatted score files.
    out_dir:       directory to hold gnomon artefacts.  If ``None`` a
                   ``TemporaryDirectory`` is used and files are read before
                   cleanup.
    n_threads:     forwarded via the ``RAYON_NUM_THREADS`` env-var if set
                   (gnomon uses rayon internally and currently has no
                   dedicated CLI flag).
    timeout:       per-invocation subprocess timeout (seconds).
    """
    binary = _locate_gnomon()
    if not score_files:
        raise ValueError("score_files must contain at least one path")

    env = os.environ.copy()
    if n_threads is not None:
        env["RAYON_NUM_THREADS"] = str(int(n_threads))

    def _score_one(score_file: str, work_dir: Path) -> pd.DataFrame:
        genotype_in = _materialise_genotype(genotype_path, work_dir)
        cmd = [binary, "score", str(score_file), str(genotype_in)]
        _run(cmd, timeout, env=env, label=f"gnomon score for {score_file}")

        # gnomon writes `<genotype_stem>.sscore` in work_dir.
        stem = Path(genotype_in).stem
        sscore = work_dir / f"{stem}.sscore"
        if not sscore.exists():
            # VCF inputs sometimes keep the compound suffix ".vcf.gz" in the
            # sscore name; scan the directory as a fallback.
            candidates = list(work_dir.glob("*.sscore"))
            if not candidates:
                raise PolygenicRunError(
                    f"gnomon score produced no .sscore in {work_dir}"
                )
            sscore = candidates[0]
        frame = parse_sscore(sscore)
        # Use the score file stem as the column label (rather than gnomon's
        # internal score-name), so callers can predict the output schema.
        if frame.shape[1] == 1:
            frame.columns = [Path(score_file).stem]
        return frame

    frames: list[pd.DataFrame] = []
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="gnomon_score_") as td:
            base = Path(td)
            for sf in score_files:
                sub = base / Path(sf).stem
                frames.append(_score_one(sf, sub))
    else:
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        for sf in score_files:
            sub = base / Path(sf).stem
            frames.append(_score_one(sf, sub))

    merged = pd.concat(frames, axis=1, join="outer")
    merged.index = merged.index.astype("string")
    for c in merged.columns:
        merged[c] = merged[c].astype("float64")
    return merged


def score_panel(
    genotype_path: str,
    score_path: str | Sequence[str],
    out_dir: str | None = None,
    n_threads: int | None = None,
    timeout: int = 1800,
    keep_iids: Collection[str] | None = None,
) -> pd.DataFrame:
    """Score a cohort against a *panel* of PGS files in a single gnomon call.

    ``gnomon score`` accepts a path to either a single PGS file or a
    directory of PGS files (per ``cli/main.rs`` ``ScoreArgs::score`` doc).
    Given a directory it natively parallelises across both samples and score
    files via rayon, writes one merged ``<cohort>_<dirname>.sscore`` to the
    genotype's directory, and the score column names come from the **headers
    of each PGS file** (one ``<name>_AVG`` column per score-name in the file
    header). Many PGS files in one directory thus produce one ``.sscore``
    with one column per score-name, alphabetically ordered (gnomon collects
    the names through a ``BTreeSet`` in ``score/prepare.rs``).

    This wrapper accepts either a directory or a list of files:

    * If ``score_path`` is a string and points to an existing directory,
      gnomon is invoked directly against it.
    * If it is a string pointing to a single file, this is just a single-PGS
      score run (still uses one process).
    * If it is a sequence of file paths, the wrapper stages them into a
      private scratch directory and invokes gnomon once against that
      directory.

    Parameters
    ----------
    genotype_path : str
        Path to a PLINK prefix / ``.bed`` / VCF / BCF input.
    score_path : str or sequence of str
        Either a directory of PGS files, a single PGS file, or a list of
        PGS files to be staged into one directory.
    out_dir : str, optional
        Directory to hold gnomon artefacts. If ``None`` a scratch directory
        is used and the parsed frame is returned before cleanup.
    n_threads : int, optional
        Forwarded as ``RAYON_NUM_THREADS`` to gnomon.
    timeout : int
        Subprocess timeout (seconds). Defaults to 30 min for biobank-scale
        panels.
    keep_iids : collection of str, optional
        If provided, parse only these sample IDs from the produced ``.sscore``.
        gnomon still scores every sample, but Python parsing, memory use, and
        downstream CSV writing are limited to the cohort.

    Returns
    -------
    pandas.DataFrame
        Indexed by sample IID (string), one ``float64`` column per
        score-name read from the PGS-file headers. Column order matches
        gnomon's internal ordering (alphabetical via ``BTreeSet``); use
        ``df.columns`` to discover what came back.
    """
    binary = _locate_gnomon()

    env = os.environ.copy()
    if n_threads is not None:
        env["RAYON_NUM_THREADS"] = str(int(n_threads))

    # Resolve score_path into either an existing directory or a list of files
    # we have to stage.
    files_to_stage: Optional[list[str]] = None
    direct_score_path: Optional[Path] = None
    if isinstance(score_path, str):
        sp = Path(score_path)
        if not sp.exists():
            raise FileNotFoundError(f"score path does not exist: {score_path}")
        direct_score_path = sp
    else:
        files_to_stage = list(score_path)
        if not files_to_stage:
            raise ValueError("score_path sequence must be non-empty")

    def _run_panel(work_dir: Path) -> pd.DataFrame:
        score_tmp: tempfile.TemporaryDirectory[str] | None = None
        try:
            if files_to_stage is not None:
                score_tmp = tempfile.TemporaryDirectory(prefix="gnomon_scores_")
                scores_dir = Path(score_tmp.name)
                for sf in files_to_stage:
                    src = Path(sf)
                    if not src.is_file():
                        raise FileNotFoundError(f"score file does not exist: {sf}")
                    link = scores_dir / src.name
                    try:
                        os.symlink(src.resolve(), link)
                    except OSError:
                        shutil.copy2(src, link)
                score_arg: Path = scores_dir
            else:
                assert direct_score_path is not None
                score_arg = direct_score_path

            genotype_in = _materialise_genotype(genotype_path, work_dir)
            keep_arg: Path | None = None
            keep_label: str = "all"
            if keep_iids is not None:
                keep_arg = work_dir / "cohort.keep"
                matched, requested = _write_keep_file(keep_arg, keep_iids, genotype_in)
                keep_label = f"{matched}/{requested}"
            cmd = [binary, "score", str(score_arg), str(genotype_in)]
            if keep_arg is not None:
                cmd = [binary, "score", "--keep", str(keep_arg), str(score_arg), str(genotype_in)]
            started_at = time.time()
            print(
                "[gnomon] parse plan "
                f"score_path={score_arg} genotype={genotype_in} "
                f"keep_iids={keep_label}",
                flush=True,
            )
            _run(cmd, timeout, env=env, label="gnomon score panel")

            # gnomon writes <genotype_stem>_<score_basename>.sscore in the dir
            # that holds the staged genotype (that's our work_dir).
            candidates = [
                p
                for p in work_dir.glob("*.sscore")
                if p.stat().st_mtime >= started_at - 1.0
            ]
            if not candidates:
                raise PolygenicRunError(
                    f"gnomon score produced no .sscore in {work_dir}"
                )
            # Pick the candidate whose stem starts with the genotype stem.
            gstem = Path(genotype_in).stem
            preferred = [c for c in candidates if c.stem.startswith(gstem)]
            sscore = preferred[0] if preferred else candidates[0]

            parse_started_at = time.time()
            frame = parse_sscore(
                sscore,
                keep_iids=keep_iids,
                progress=lambda message: print(f"[gnomon] {message}", flush=True),
            )
            print(
                "[gnomon] parsed "
                f"{sscore} size={_file_size_mib(sscore):.1f}MiB "
                f"rows={frame.shape[0]} cols={frame.shape[1]} "
                f"elapsed={_format_seconds(time.time() - parse_started_at)}",
                flush=True,
            )
            return frame
        finally:
            if score_tmp is not None:
                score_tmp.cleanup()

    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="gnomon_panel_") as td:
            return _run_panel(Path(td))
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    return _run_panel(base)


def fit_pca(
    genotype_path: str,
    n_pcs: int = 10,
    out_dir: str | None = None,
    timeout: int = 300,
) -> str:
    """Run ``gnomon fit --components <n_pcs>`` and return the path to
    ``hwe.json``.  If ``out_dir`` is ``None`` the model is written to a
    persistent directory next to a *managed copy* of the genotype inputs.
    """
    binary = _locate_gnomon()

    def _fit_in(work_dir: Path) -> Path:
        geno = _materialise_genotype(genotype_path, work_dir)
        cmd = [binary, "fit", "--components", str(int(n_pcs)), str(geno)]
        _run(cmd, timeout)
        model = work_dir / "hwe.json"
        if not model.exists():
            # Some gnomon versions place it next to the .bed prefix directly.
            alt = Path(geno).with_name("hwe.json")
            if alt.exists():
                model = alt
            else:
                raise PolygenicRunError(
                    f"gnomon fit did not produce hwe.json in {work_dir}"
                )
        return model

    if out_dir is None:
        # We must NOT use a TemporaryDirectory here because the caller needs
        # the model to persist for later ``project``-ion.  Use mkdtemp instead.
        base = Path(tempfile.mkdtemp(prefix="gnomon_fit_"))
        model = _fit_in(base)
        return str(model)

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    model = _fit_in(base)
    return str(model)


def project_pca(
    genotype_path: str,
    model_path: str,
    n_pcs: int = 10,
    out_dir: str | None = None,
    timeout: int = 300,
) -> pd.DataFrame:
    """Run ``gnomon project`` using ``model_path`` (an ``hwe.json`` from
    :func:`fit_pca`) and return a DataFrame indexed by IID with columns
    ``PC1 .. PCn_pcs`` (trailing components beyond what the model retains
    are simply omitted).
    """
    binary = _locate_gnomon()

    def _project_in(work_dir: Path) -> pd.DataFrame:
        geno = _materialise_genotype(genotype_path, work_dir)
        # gnomon expects hwe.json *next to* the genotype data.
        local_model = work_dir / "hwe.json"
        if not local_model.exists():
            try:
                os.symlink(model_path, local_model)
            except OSError:
                shutil.copy2(model_path, local_model)
        cmd = [binary, "project", str(geno)]
        _run(cmd, timeout)
        bin_path = work_dir / "projection_scores.bin"
        if not bin_path.exists():
            raise PolygenicRunError(
                f"gnomon project did not produce projection_scores.bin in {work_dir}"
            )
        frame = parse_projection_bin(bin_path)
        if n_pcs is not None and frame.shape[1] > n_pcs:
            frame = frame.iloc[:, :n_pcs]
        return frame

    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="gnomon_project_") as td:
            return _project_in(Path(td))
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    return _project_in(base)


def infer_terms(
    genotype_path: str,
    out_dir: str | None = None,
    timeout: int = 300,
) -> pd.DataFrame:
    """Run ``gnomon terms --sex`` and return a DataFrame with the per-sample
    inferred metadata (sex + diagnostics)."""
    binary = _locate_gnomon()

    def _terms_in(work_dir: Path) -> pd.DataFrame:
        geno = _materialise_genotype(genotype_path, work_dir)
        cmd = [binary, "terms", "--sex", str(geno)]
        _run(cmd, timeout)
        tsv = work_dir / "sex.tsv"
        if not tsv.exists():
            raise PolygenicRunError(
                f"gnomon terms did not produce sex.tsv in {work_dir}"
            )
        return parse_sex_tsv(tsv)

    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="gnomon_terms_") as td:
            return _terms_in(Path(td))
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    return _terms_in(base)


# ---------------------------------------------------------------------------
# integration with the synthetic dataset
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 0:
        return x - mu
    return (x - mu) / sd


def augment_synthetic_with_real_pgs(
    dataset,
    pgs_df: pd.DataFrame,
    pgs_map: Mapping[str, str],
):
    """Replace the synthetic PGS columns in ``dataset`` with the real scores
    in ``pgs_df``.

    Parameters
    ----------
    dataset:  a :class:`causal_pred.data.synthetic.SyntheticDataset`.
    pgs_df:   DataFrame whose rows correspond 1-to-1 (by position) to rows of
              ``dataset.X``.  It does not need to be indexed by sample ID —
              only its column-name mapping matters.
    pgs_map:  mapping from synthetic column name (``PGS_T2D`` etc.) to a
              column of ``pgs_df``.  Any synthetic column not mentioned in
              ``pgs_map`` is left untouched.

    Returns
    -------
    A NEW SyntheticDataset with the relevant columns substituted in.  All
    other columns (phenotypes, survival outcome) are copied verbatim.  Real
    scores are z-scored so that their scale matches the synthetic ones.
    """
    from .synthetic import SyntheticDataset  # local import to avoid cycles

    if not isinstance(dataset.X, np.ndarray):
        raise TypeError("dataset.X must be a numpy array")

    for syn_col, real_col in pgs_map.items():
        if syn_col not in dataset.columns:
            raise KeyError(f"{syn_col!r} is not a column of the dataset")
        if real_col not in pgs_df.columns:
            raise KeyError(f"{real_col!r} is not in pgs_df columns")

    if len(pgs_df) != dataset.n:
        raise ValueError(f"pgs_df has {len(pgs_df)} rows but dataset has {dataset.n}")

    X = dataset.X.astype(float, copy=True)
    for syn_col, real_col in pgs_map.items():
        idx = dataset.columns.index(syn_col)
        vals = np.asarray(pgs_df[real_col].to_numpy(), dtype=float)
        X[:, idx] = _zscore(vals)

    return SyntheticDataset(
        X=X,
        time=dataset.time.copy(),
        event=dataset.event.copy(),
        columns=dataset.columns,
        node_types=dataset.node_types,
        ground_truth_adj=dataset.ground_truth_adj.copy(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEMO_PGS_URL = (
    # PGS Catalog — Mahajan et al. T2D PGS, hg19; ~700KB after harmonisation.
    "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000014/ScoringFiles/PGS000014.txt.gz"
)
_DEMO_VCF_URL = (
    # 1000 Genomes phase-3 chr22, GRCh37 — ~11MB compressed.
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
)
_DEMO_CACHE = Path.home() / ".cache" / "causal_pred" / "polygenic_demo"


def _demo(timeout: int = 900) -> int:
    """Minimal end-to-end smoke test against publicly-available 1KG data."""
    import urllib.request

    _DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    vcf_path = _DEMO_CACHE / Path(_DEMO_VCF_URL).name
    pgs_path = _DEMO_CACHE / Path(_DEMO_PGS_URL).name

    try:
        for url, dst in [(_DEMO_VCF_URL, vcf_path), (_DEMO_PGS_URL, pgs_path)]:
            if not dst.is_file() or dst.stat().st_size == 0:
                print(f"> downloading {url} -> {dst}")
                urllib.request.urlretrieve(url, dst)
    except (OSError, Exception) as exc:  # broad: covers urllib network errors
        print(f"demo requires internet + 1KG subset download; skipped ({exc!r})")
        return 0

    print("> running gnomon score on the downloaded cohort ...")
    try:
        scores = score_cohort(
            str(vcf_path),
            [str(pgs_path)],
            out_dir=str(_DEMO_CACHE / "out"),
            timeout=timeout,
        )
    except PolygenicToolMissing as exc:
        print(f"gnomon not available: {exc}")
        return 1
    print(scores.describe().to_string())
    print("> first 5 rows:")
    print(scores.head().to_string())
    return 0


def _main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="causal_pred.data.polygenic",
        description=(
            "Wrappers around the gnomon CLI for polygenic scoring, HWE-PCA "
            "fitting/projection, and sex-term inference."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser(
        "score", help="Score a cohort against one or more PGS files"
    )
    p_score.add_argument("genotype")
    p_score.add_argument("score_files", nargs="+")
    p_score.add_argument("--out-dir", default=None)
    p_score.add_argument("--threads", type=int, default=None)
    p_score.add_argument("--timeout", type=int, default=600)

    p_fit = sub.add_parser("fit", help="Fit an HWE-PCA model (writes hwe.json)")
    p_fit.add_argument("genotype")
    p_fit.add_argument("--n-pcs", type=int, default=10)
    p_fit.add_argument("--out-dir", default=None)
    p_fit.add_argument("--timeout", type=int, default=300)

    p_prj = sub.add_parser("project", help="Project samples onto an existing HWE-PCA")
    p_prj.add_argument("genotype")
    p_prj.add_argument("model")
    p_prj.add_argument("--n-pcs", type=int, default=10)
    p_prj.add_argument("--out-dir", default=None)
    p_prj.add_argument("--timeout", type=int, default=300)

    p_terms = sub.add_parser("terms", help="Infer per-sample metadata (sex)")
    p_terms.add_argument("genotype")
    p_terms.add_argument("--out-dir", default=None)
    p_terms.add_argument("--timeout", type=int, default=300)

    sub.add_parser("demo", help="Run the 1KG + PGS Catalog smoke test")
    sub.add_parser("check", help="Print whether gnomon is installed")

    args = parser.parse_args(argv)

    if args.command == "check":
        if gnomon_available():
            print(f"gnomon: {_locate_gnomon()}")
            return 0
        print("gnomon: NOT FOUND on $PATH")
        return 1

    if args.command == "score":
        df = score_cohort(
            args.genotype,
            args.score_files,
            out_dir=args.out_dir,
            n_threads=args.threads,
            timeout=args.timeout,
        )
        print(df.to_csv(sep="\t"))
        return 0

    if args.command == "fit":
        path = fit_pca(
            args.genotype,
            n_pcs=args.n_pcs,
            out_dir=args.out_dir,
            timeout=args.timeout,
        )
        print(path)
        return 0

    if args.command == "project":
        df = project_pca(
            args.genotype,
            args.model,
            n_pcs=args.n_pcs,
            out_dir=args.out_dir,
            timeout=args.timeout,
        )
        print(df.to_csv(sep="\t"))
        return 0

    if args.command == "terms":
        df = infer_terms(args.genotype, out_dir=args.out_dir, timeout=args.timeout)
        print(df.to_csv(sep="\t"))
        return 0

    if args.command == "demo":
        return _demo()

    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover


__all__ = [
    "PolygenicToolMissing",
    "PolygenicRunError",
    "gnomon_available",
    "score_cohort",
    "score_panel",
    "fit_pca",
    "project_pca",
    "infer_terms",
    "augment_synthetic_with_real_pgs",
    "parse_sscore",
    "parse_projection_bin",
    "parse_sex_tsv",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
