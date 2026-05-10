"""Resolve human-readable labels for PGS Catalog IDs and OMOP concept IDs.

Both lookups are cached to JSON on disk so the per-feature log lines that
print top-genome and top-EHR loadings can show concept names alongside
the numeric IDs without paying the network cost on every run.

Genome side
-----------
PGS Catalog REST API:

    GET https://www.pgscatalog.org/rest/score/{pgs_id}/

returns ``trait_reported``, ``trait_efo``, and a ``publication`` block we
read for first-author / year. Failures fall back to the bare PGS ID so
the pipeline never blocks on the public API.

EHR side
--------
The cohort builder names columns ``cond:{phecode}`` (where ``phecode``
is actually the OMOP ``condition_concept_id`` despite the column alias)
and ``drug:{atc_class}`` (the OMOP ``concept_id`` of an ATC-class node).
Both are integer concept IDs in the OMOP vocabulary, so a single
BigQuery against ``{cdr}.concept`` resolves names for both.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PGS_CATALOG_URL = "https://www.pgscatalog.org/rest/score/{pgs_id}/"
_PGS_ID_RE = re.compile(r"PGS\d{6}", re.IGNORECASE)


def extract_pgs_id(name: str) -> Optional[str]:
    """Pull the canonical ``PGS00XXXX`` id out of a column name. Tolerates
    ``PGS004910``, ``PGS004910_hmPOS_GRCh38``, ``pgs_pgs004910`` etc."""
    match = _PGS_ID_RE.search(str(name))
    return match.group(0).upper() if match else None
_DEFAULT_TIMEOUT = 10.0
_TRAIT_TRUNCATE = 40


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _truncate(text: str, n: int = _TRAIT_TRUNCATE) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def resolve_pgs_metadata(
    pgs_ids: Iterable[str],
    cache_path: Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, dict]:
    """Return ``{PGS00xxxx: {'trait', 'first_author', 'pub_year'}}``.

    Reads the cache file first, fetches missing IDs from the PGS Catalog
    REST API one at a time, persists the cache after every batch of 25
    so an interrupted run still keeps progress.
    """
    wanted = sorted({str(p).upper() for p in pgs_ids if p})
    cache = _load_json(cache_path)
    missing = [p for p in wanted if p not in cache]
    if missing:
        logger.info(
            "[labels] fetching %d PGS metadata entries from PGS Catalog",
            len(missing),
        )
    for idx, pgs in enumerate(missing, 1):
        url = PGS_CATALOG_URL.format(pgs_id=pgs)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                rec = json.loads(resp.read().decode("utf-8"))
            trait = str(rec.get("trait_reported") or "").strip()
            if not trait:
                efo = rec.get("trait_efo") or []
                if efo and isinstance(efo[0], dict):
                    trait = str(efo[0].get("label") or "").strip()
            pub = rec.get("publication") or {}
            cache[pgs] = {
                "trait": trait,
                "first_author": str(pub.get("firstauthor") or "").strip(),
                "pub_year": int(pub["pub_year"]) if pub.get("pub_year") else None,
            }
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("[labels] PGS %s lookup failed: %s", pgs, exc)
            cache[pgs] = {"trait": "", "first_author": "", "pub_year": None}
        if idx % 25 == 0:
            _save_json(cache_path, cache)
    if missing:
        _save_json(cache_path, cache)
    return {p: cache.get(p, {"trait": "", "first_author": "", "pub_year": None}) for p in wanted}


def resolve_omop_concepts(
    concept_ids: Iterable[str | int],
    cache_path: Path,
    *,
    cdr: Optional[str] = None,
) -> dict[str, dict]:
    """Return ``{concept_id: {'name', 'class', 'vocabulary', 'domain'}}``.

    Reads cached entries first, queries ``{cdr}.concept`` for any missing
    IDs in one batched ``WHERE concept_id IN (...)``. ``cdr`` defaults to
    ``$WORKSPACE_CDR`` so the AoU notebook environment "just works".

    Failures (missing CDR env var, BQ permissions, etc.) are logged but
    not fatal: missing entries are returned with empty fields so the
    caller can fall back to the bare concept ID.
    """
    cleaned = sorted({str(c).strip() for c in concept_ids if str(c).strip().isdigit()})
    cache = _load_json(cache_path)
    missing = [c for c in cleaned if c not in cache]
    if not missing:
        return {c: cache[c] for c in cleaned}

    if cdr is None:
        cdr = os.environ.get("WORKSPACE_CDR", "").strip() or None
    if cdr is None:
        logger.info(
            "[labels] OMOP concept lookup skipped: WORKSPACE_CDR not set "
            "(%d ids will render bare)",
            len(missing),
        )
        for c in missing:
            cache[c] = {"name": "", "class": "", "vocabulary": "", "domain": ""}
        _save_json(cache_path, cache)
        return {c: cache[c] for c in cleaned}

    try:
        from google.cloud import bigquery  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.warning("[labels] google-cloud-bigquery unavailable: %s", exc)
        for c in missing:
            cache[c] = {"name": "", "class": "", "vocabulary": "", "domain": ""}
        _save_json(cache_path, cache)
        return {c: cache[c] for c in cleaned}

    try:
        client = bigquery.Client()
        ids_csv = ", ".join(missing)
        query = (
            "SELECT CAST(concept_id AS STRING) AS concept_id, "
            "concept_name, concept_class_id, vocabulary_id, domain_id "
            f"FROM `{cdr}.concept` "
            f"WHERE concept_id IN ({ids_csv})"
        )
        logger.info(
            "[labels] fetching %d OMOP concept names from %s.concept",
            len(missing),
            cdr,
        )
        rows = list(client.query(query).result())
        for row in rows:
            cache[str(row["concept_id"])] = {
                "name": str(row["concept_name"] or "").strip(),
                "class": str(row["concept_class_id"] or "").strip(),
                "vocabulary": str(row["vocabulary_id"] or "").strip(),
                "domain": str(row["domain_id"] or "").strip(),
            }
    except Exception as exc:  # pragma: no cover - depends on BQ env
        logger.warning("[labels] OMOP concept BQ lookup failed: %s", exc)

    for c in missing:
        if c not in cache:
            cache[c] = {"name": "", "class": "", "vocabulary": "", "domain": ""}
    _save_json(cache_path, cache)
    return {c: cache[c] for c in cleaned}


def label_genome_entry(
    feature_name: str,
    weight: float,
    pgs_meta: dict[str, dict],
) -> str:
    """Render one genome top-loading entry with its PGS Catalog trait label.
    ``feature_name`` is the original DataFrame column name; the PGS id is
    extracted via ``extract_pgs_id`` so name suffixes don't matter."""
    pgs_id = extract_pgs_id(feature_name)
    if pgs_id is None:
        return f"{feature_name}={weight:+.2f}"
    rec = pgs_meta.get(pgs_id, {})
    trait = rec.get("trait", "")
    if trait:
        return f"{pgs_id}={weight:+.2f} ({_truncate(trait)})"
    return f"{pgs_id}={weight:+.2f}"


def label_ehr_entry(
    feature_name: str,
    weight: float,
    omop_meta: dict[str, dict],
) -> str:
    """Render one EHR top-loading entry, decorating ``cond:`` and ``drug:``
    columns with their OMOP concept names. ``lab_*`` columns are already
    human readable and pass through unchanged.
    """
    if ":" not in feature_name:
        return f"{feature_name}={weight:+.2f}"
    prefix, ident = feature_name.split(":", 1)
    if prefix in ("cond", "drug") and ident.isdigit():
        rec = omop_meta.get(ident, {})
        name = rec.get("name", "")
        if name:
            return f"{feature_name}={weight:+.2f} ({_truncate(name)})"
    return f"{feature_name}={weight:+.2f}"


def label_genome_column(name: str, pgs_meta: dict[str, dict]) -> str:
    """Human-readable label for a PGS panel column (used as plot tick label)."""
    pgs_id = extract_pgs_id(name)
    if pgs_id is None:
        return str(name)
    trait = pgs_meta.get(pgs_id, {}).get("trait", "")
    if trait:
        return f"{pgs_id} {_truncate(trait, 30)}"
    return pgs_id


def label_ehr_column(name: str, omop_meta: dict[str, dict]) -> str:
    """Human-readable label for an EHR panel column (used as plot tick label)."""
    name = str(name)
    if ":" not in name:
        return name
    prefix, ident = name.split(":", 1)
    if prefix in ("cond", "drug") and ident.isdigit():
        concept = omop_meta.get(ident, {}).get("name", "")
        if concept:
            return f"{prefix}:{ident} {_truncate(concept, 30)}"
    return name


def collect_pgs_ids(names: Iterable[str]) -> set[str]:
    """Extract canonical PGS IDs from a column-name iterable."""
    out: set[str] = set()
    for n in names:
        pid = extract_pgs_id(str(n))
        if pid is not None:
            out.add(pid)
    return out


def collect_omop_concept_ids(names: Iterable[str]) -> set[str]:
    """Extract OMOP concept IDs from ``cond:`` / ``drug:`` column names."""
    out: set[str] = set()
    for n in names:
        n = str(n)
        if ":" not in n:
            continue
        prefix, ident = n.split(":", 1)
        if prefix in ("cond", "drug") and ident.isdigit():
            out.add(ident)
    return out
