"""Live two-sample MR via the OpenGWAS (MR-Base) REST API.

Provides the programmatic fetch + IVW pipeline used by MrDAG to obtain
per-pair causal-effect estimates with full per-SNP provenance:

1. For each exposure trait, fetch genome-wide-significant, LD-clumped
   tophits via OpenGWAS ``/tophits``.
2. For each outcome trait, look up those rsids in the outcome GWAS via
   ``/associations``.
3. Harmonise alleles (flip outcome beta when effect alleles disagree;
   drop palindromic SNPs with intermediate MAF).
4. Compute the inverse-variance-weighted (IVW) estimator
   ``beta_IVW = sum(w_j * (b_y/b_x)) / sum(w_j)`` with
   ``w_j = b_x^2 / se_y^2`` and ``SE = 1 / sqrt(sum(w_j))``.

Results are cached to disk (one JSON per exposure+outcome pair) so
repeat runs do not re-hit the API.  Authentication is carried by an
``OpenGWASClient``; the default client reads ``OPENGWAS_JWT``.  Without
a token the fetcher returns a result with NaN cells (the MrDAG pipeline
already treats NaN as "no MR information").

Curated study IDs
-----------------
One OpenGWAS / EBI GWAS Catalog study per trait, picked for largest
sample size and -- where available -- multi-ancestry coverage (the
AoU cohort is ~50% non-European):

* BMI               -- ``ieu-b-40``           Yengo 2018 UKB+GIANT, n=681,275
                                              (canonical MR instrument set;
                                              multi-ancestry BMI is not yet
                                              published in OpenGWAS).
* LDL               -- ``ieu-b-110``          Willer 2013 GLGC, n=188,577.
* HbA1c             -- ``ebi-a-GCST90002244`` Chen 2021 MAGIC trans-ancestry,
                                              n=159,940 (EUR/AFR/EAS/SAS).
* systolic_BP       -- ``ieu-b-38``           Evangelou 2018 ICBP+UKB, n=757,601.
* years_smoking     -- ``ieu-b-25``           Wootton 2020 lifetime smoking
                                              index, UKB n=462,690 (the
                                              canonical MR exposure for
                                              "lifetime smoking burden").
* physical_activity -- ``ukb-b-4710``         IEU UKB MET-min/wk vigorous PA.
* hypertension      -- ``ukb-b-12493``        IEU UKB self-reported HTN,
                                              n=463,010.
* T2D               -- ``ebi-a-GCST006867``   Xue et al. 2018 T2D, n=655,666
                                              EUR (61,714 cases; PMID 30054458).
* cardiovascular_disease (CAD)
                    -- ``ebi-a-GCST005195``   van der Harst & Verweij 2018
                                              CARDIoGRAMplusC4D+UKB,
                                              n_cases=122,733.

Diet quality is intentionally left out -- no consensus single-score
GWAS exists; Cole 2020 published 85 dietary patterns instead.

Alternative source: EBI GWAS Catalog hosts the harmonised summary
stats behind every ``ebi-a-GCSTxxx`` ID at
``https://www.ebi.ac.uk/gwas/summary-statistics/api/`` if OpenGWAS
itself is unreachable.

Notes
-----
* OpenGWAS requires JWT auth for most endpoints since 2024-05.  Get a
  token at https://opengwas.io/profile/ and set ``OPENGWAS_JWT``.
* The cache is keyed by exposure_id + outcome_id + clumping params so
  changing any of those invalidates the cache automatically.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple

import numpy as np

from .gwas import (
    CIRCULAR_PAIRS,
    GWASSummary,
    MR_EXPOSURES,
    MR_OUTCOMES,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated study IDs.  See module docstring for citations.
# ---------------------------------------------------------------------------

OPENGWAS_STUDY_IDS: Dict[str, str] = {
    "BMI": "ieu-b-40",
    "LDL": "ieu-b-110",
    "HbA1c": "ebi-a-GCST90002244",
    "systolic_BP": "ieu-b-38",
    "years_smoking": "ieu-b-25",
    "physical_activity": "ukb-b-4710",
    "hypertension": "ukb-b-12493",
    "T2D": "ebi-a-GCST006867",
    "cardiovascular_disease": "ebi-a-GCST005195",
    # diet_quality intentionally omitted -- no consensus single-score GWAS.
}

STUDY_CITATIONS: Dict[str, str] = {
    "ieu-b-40": "Yengo et al. 2018 (UKB+GIANT, n=681,275, EUR)",
    "ieu-b-110": "Willer et al. 2013 GLGC (n=188,577, EUR)",
    "ebi-a-GCST90002244": "Chen et al. 2021 MAGIC (n=159,940, trans-ancestry)",
    "ieu-b-38": "Evangelou et al. 2018 ICBP+UKB SBP (n=757,601, EUR)",
    "ieu-b-25": "Wootton et al. 2020 lifetime smoking index (UKB, n=462,690)",
    "ukb-b-4710": "IEU UKB MET-min/wk vigorous activity (n=377,234)",
    "ukb-b-12493": "UKB self-reported hypertension (n=463,010)",
    "ebi-a-GCST006867": "Xue et al. 2018 T2D (n=655,666; 61,714 cases EUR; "
                        "PMID 30054458; nsnp=5M)",
    "ebi-a-GCST005195": "van der Harst & Verweij 2018 CARDIoGRAMplusC4D+UKB CAD "
                        "(122,733 cases)",
}


# ---------------------------------------------------------------------------
# OpenGWAS REST client (no third-party deps -- urllib only).
# ---------------------------------------------------------------------------

OPENGWAS_BASE = "https://api.opengwas.io/api"
DEFAULT_PVAL = 5e-8
DEFAULT_R2 = 0.001
DEFAULT_KB = 10000
PALINDROMIC = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
PALINDROME_MAF_BAND = (0.42, 0.58)


ASSOCS_BATCH_SIZE = 50


class OpenGWASClientLike(Protocol):
    @property
    def authenticated(self) -> bool:
        ...

    def fetch_tophits(
        self,
        study_id: str,
        *,
        pval: float = DEFAULT_PVAL,
        r2: float = DEFAULT_R2,
        kb: int = DEFAULT_KB,
    ) -> list[dict]:
        ...

    def fetch_associations_by_study(
        self,
        study_ids: Sequence[str],
        rsids: Sequence[str],
        *,
        max_workers: int = 4,
    ) -> dict[str, list[dict]]:
        ...


@dataclass(frozen=True)
class OpenGWASClient:
    """Small OpenGWAS REST client used by ``load_live_gwas``."""

    token: Optional[str]
    base_url: str = OPENGWAS_BASE
    timeout: float = 60.0
    retries: int = 3

    @classmethod
    def from_environment(cls) -> "OpenGWASClient":
        token = os.environ.get("OPENGWAS_JWT", "").strip() or None
        return cls(token=token)

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def _post(self, path: str, body: dict) -> Any:
        """POST JSON to the OpenGWAS API; retry transient 5xx and 429."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}{path}"

        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 502, 503, 504) and attempt + 1 < self.retries:
                    time.sleep(2.0 ** attempt)
                    last_err = exc
                    continue
                if exc.code == 401:
                    raise RuntimeError(
                        "OpenGWAS request rejected (401). Set OPENGWAS_JWT to a valid "
                        "token from https://opengwas.io/profile/."
                    ) from exc
                raise
            except urllib.error.URLError as exc:
                if attempt + 1 < self.retries:
                    time.sleep(2.0 ** attempt)
                    last_err = exc
                    continue
                raise
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"OpenGWAS POST {path} failed without an exception")

    def fetch_tophits(
        self,
        study_id: str,
        *,
        pval: float = DEFAULT_PVAL,
        r2: float = DEFAULT_R2,
        kb: int = DEFAULT_KB,
    ) -> list[dict]:
        """Server-side LD-clumped tophits for one exposure GWAS.

        We do not pass ``pop`` -- OpenGWAS picks the LD reference
        server-side and we deliberately avoid baking a single ancestry
        into the request.
        """
        body = {
            "id": study_id,
            "pval": pval,
            "r2": r2,
            "kb": kb,
            "preclumped": 0,
            "clump": 1,
            "force_server": 0,
        }
        out = self._post("/tophits", body)
        if not isinstance(out, list):
            raise RuntimeError(
                f"OpenGWAS /tophits returned non-list for {study_id}: {out!r}"
            )
        return out

    def fetch_associations(
        self,
        study_id: str,
        rsids: Sequence[str],
    ) -> list[dict]:
        """Per-SNP outcome associations for one outcome GWAS."""
        rsids = list(rsids)
        if not rsids:
            return []
        out: list[dict] = []
        for start in range(0, len(rsids), ASSOCS_BATCH_SIZE):
            chunk = rsids[start:start + ASSOCS_BATCH_SIZE]
            body = {"id": [study_id], "variant": chunk}
            resp = self._post("/associations", body)
            if not isinstance(resp, list):
                raise RuntimeError(
                    f"OpenGWAS /associations returned non-list for {study_id} "
                    f"chunk {start}: {resp!r}"
                )
            out.extend(resp)
        return out

    def fetch_associations_by_study(
        self,
        study_ids: Sequence[str],
        rsids: Sequence[str],
        *,
        max_workers: int = 4,
    ) -> dict[str, list[dict]]:
        """Per-SNP associations for many outcome studies."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        rsids = list(rsids)
        study_ids = list(dict.fromkeys(study_ids))
        out: dict[str, list[dict]] = {sid: [] for sid in study_ids}
        if not rsids or not study_ids:
            return out

        if max_workers <= 1 or len(study_ids) == 1:
            for sid in study_ids:
                out[sid] = self.fetch_associations(sid, rsids)
            return out

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.fetch_associations, sid, rsids): sid
                for sid in study_ids
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                out[sid] = fut.result()
        return out


# ---------------------------------------------------------------------------
# Harmonise + IVW
# ---------------------------------------------------------------------------


def _is_palindromic(ea: str, nea: str) -> bool:
    return (ea.upper(), nea.upper()) in PALINDROMIC


def harmonise_pairs(
    exposure_hits: Iterable[dict],
    outcome_assocs: Iterable[dict],
) -> list[Tuple[float, float, float, float, str]]:
    """Match exposure tophits to outcome associations by rsid.

    Returns a list of ``(beta_x, se_x, beta_y, se_y, rsid)`` tuples after:
      * sign-flipping outcome beta when effect alleles disagree,
      * dropping palindromic SNPs with ambiguous (intermediate) MAF,
      * dropping any SNP with non-finite beta/se.
    """
    by_rsid_out: Dict[str, dict] = {}
    for a in outcome_assocs:
        a_rsid = a.get("rsid")
        if isinstance(a_rsid, str) and a_rsid:
            by_rsid_out[a_rsid] = a
    pairs: list[Tuple[float, float, float, float, str]] = []
    for hit in exposure_hits:
        rsid = hit.get("rsid")
        if not isinstance(rsid, str) or not rsid or rsid not in by_rsid_out:
            continue
        out = by_rsid_out[rsid]
        try:
            bx = float(hit["beta"])
            sx = float(hit["se"])
            by = float(out["beta"])
            sy = float(out["se"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            np.isfinite(bx) and np.isfinite(sx) and sx > 0.0
            and np.isfinite(by) and np.isfinite(sy) and sy > 0.0
        ):
            continue
        ea_x = str(hit.get("ea", "")).upper()
        nea_x = str(hit.get("nea", "")).upper()
        ea_y = str(out.get("ea", "")).upper()
        nea_y = str(out.get("nea", "")).upper()
        if not (ea_x and nea_x and ea_y and nea_y):
            continue
        if _is_palindromic(ea_x, nea_x):
            eaf_x = hit.get("eaf")
            try:
                eaf_x = float(eaf_x) if eaf_x is not None else None
            except (TypeError, ValueError):
                eaf_x = None
            if eaf_x is None or PALINDROME_MAF_BAND[0] <= eaf_x <= PALINDROME_MAF_BAND[1]:
                continue
        if (ea_x, nea_x) == (ea_y, nea_y):
            pass
        elif (ea_x, nea_x) == (nea_y, ea_y):
            by = -by
        else:
            continue
        pairs.append((bx, sx, by, sy, rsid))
    return pairs


def ivw(pairs: Sequence[Tuple[float, float, float, float, str]]) -> Tuple[float, float, int]:
    """Inverse-variance-weighted estimator on harmonised SNPs."""
    if not pairs:
        return float("nan"), float("nan"), 0
    bx = np.asarray([p[0] for p in pairs], dtype=float)
    by = np.asarray([p[2] for p in pairs], dtype=float)
    sy = np.asarray([p[3] for p in pairs], dtype=float)
    w = (bx ** 2) / (sy ** 2)
    ratio = by / bx
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan"), float("nan"), len(pairs)
    beta = float(np.sum(w * ratio) / denom)
    se = float(1.0 / np.sqrt(denom))
    return beta, se, len(pairs)


# ---------------------------------------------------------------------------
# On-disk cache (one JSON per exposure+outcome pair).
# ---------------------------------------------------------------------------

CACHE_VERSION = "v1"


def _cache_key(
    exposure_id: str,
    outcome_id: str,
    pval: float,
    r2: float,
    kb: int,
) -> str:
    return (
        f"{CACHE_VERSION}__{exposure_id}__{outcome_id}__"
        f"p{pval:.0e}_r{r2}_kb{kb}.json"
    )


def _read_cache(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


@dataclass
class _PerPair:
    exposure: str
    outcome: str
    exposure_id: str
    outcome_id: str
    beta: float
    se: float
    n_snps: int
    source: str  # "cache" | "fetched" | "circular" | "no_id" | "no_overlap" | "error"
    note: str = ""


def _emit(per_pair: list[_PerPair], record: _PerPair) -> None:
    per_pair.append(record)


def _per_exposure_tophits_cache_key(
    study_id: str,
    pval: float,
    r2: float,
    kb: int,
) -> str:
    return f"{CACHE_VERSION}__tophits__{study_id}__p{pval:.0e}_r{r2}_kb{kb}.json"


def _load_or_fetch_tophits(
    cache_dir: Path,
    client: OpenGWASClientLike,
    study_id: str,
    pval: float,
    r2: float,
    kb: int,
) -> list[dict]:
    path = cache_dir / _per_exposure_tophits_cache_key(study_id, pval, r2, kb)
    cached = _read_cache(path)
    if cached is not None and isinstance(cached.get("tophits"), list):
        return cached["tophits"]
    hits = client.fetch_tophits(study_id, pval=pval, r2=r2, kb=kb)
    _write_cache(path, {"study_id": study_id, "tophits": hits, "fetched_at": time.time()})
    return hits


def load_live_gwas(
    exposures: Sequence[str] = MR_EXPOSURES,
    outcomes: Sequence[str] = MR_OUTCOMES,
    *,
    client: Optional[OpenGWASClientLike] = None,
    cache_dir: Optional[Path] = None,
    pval: float = DEFAULT_PVAL,
    r2: float = DEFAULT_R2,
    kb: int = DEFAULT_KB,
    drop_circular: bool = True,
    raise_on_error: bool = False,
) -> GWASSummary:
    """Run two-sample IVW for every (exposure, outcome) pair in the trait set.

    Returns a ``GWASSummary`` (NaN for unavailable cells).  When the client
    has no token, uncached cells end up NaN and the function logs warnings
    rather than raising.
    """
    from scipy.stats import norm

    exposures = tuple(exposures)
    outcomes = tuple(outcomes)
    n_exp, n_out = len(exposures), len(outcomes)
    client = OpenGWASClient.from_environment() if client is None else client

    cache_root = Path(cache_dir) if cache_dir is not None else (
        Path.home() / "causal-pred" / "data" / "mr_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    betas = np.full((n_exp, n_out), np.nan, dtype=float)
    ses = np.full((n_exp, n_out), np.nan, dtype=float)
    n_snps = np.zeros(n_exp, dtype=int)
    citations: Dict[Tuple[str, str], str] = {}
    per_pair: list[_PerPair] = []
    circular = set(CIRCULAR_PAIRS) if drop_circular else set()

    # Tophits: always try the on-disk cache first.  Live fetch only
    # fires when both (a) the cache is missing AND (b) a JWT is set.
    have_token = bool(client.authenticated)
    tophits_by_exposure: Dict[str, list[dict]] = {}
    for exp in exposures:
        exp_id = OPENGWAS_STUDY_IDS.get(exp)
        if exp_id is None:
            continue
        cache_path = cache_root / _per_exposure_tophits_cache_key(
            exp_id, pval, r2, kb
        )
        cached = _read_cache(cache_path)
        if cached is not None and isinstance(cached.get("tophits"), list):
            tophits_by_exposure[exp] = cached["tophits"]
            logger.info(
                "[opengwas] tophits cache hit %s (%s): n=%d",
                exp, exp_id, len(tophits_by_exposure[exp]),
            )
            continue
        if not have_token:
            logger.warning(
                "[opengwas] no tophits cache for %s (%s) and no OPENGWAS_JWT; "
                "skipping (set token at https://opengwas.io/profile/ to refresh)",
                exp, exp_id,
            )
            continue
        try:
            tophits_by_exposure[exp] = _load_or_fetch_tophits(
                cache_root, client, exp_id, pval, r2, kb
            )
            logger.info(
                "[opengwas] tophits fetched %s (%s): n=%d",
                exp, exp_id, len(tophits_by_exposure[exp]),
            )
        except Exception as exc:
            if raise_on_error:
                raise
            logger.warning(
                "[opengwas] tophits FAILED for %s (%s): %s", exp, exp_id, exc
            )

    for i, exp in enumerate(exposures):
        exp_id = OPENGWAS_STUDY_IDS.get(exp)
        hits = tophits_by_exposure.get(exp, []) if exp_id else []
        rsids: list[str] = (
            [h["rsid"] for h in hits if isinstance(h.get("rsid"), str) and h["rsid"]]
            if hits
            else []
        )
        # Pass 1: enumerate every (exposure, outcome) target, classify
        # each as cached / circular / no_id / needs_fetch.
        per_pair_results: dict[str, dict] = {}
        outcomes_to_fetch: list[str] = []
        for out in outcomes:
            if exp == out:
                continue
            out_id = OPENGWAS_STUDY_IDS.get(out)
            if (exp, out) in circular:
                per_pair_results[out] = {
                    "exp_id": exp_id or "", "out_id": out_id or "",
                    "beta": float("nan"), "se": float("nan"), "n_snps": 0,
                    "source": "circular", "note": "",
                }
                continue
            if not exp_id or not out_id:
                per_pair_results[out] = {
                    "exp_id": exp_id or "", "out_id": out_id or "",
                    "beta": float("nan"), "se": float("nan"), "n_snps": 0,
                    "source": "no_id",
                    "note": "missing exposure or outcome study id",
                }
                continue
            if not hits:
                per_pair_results[out] = {
                    "exp_id": exp_id, "out_id": out_id,
                    "beta": float("nan"), "se": float("nan"), "n_snps": 0,
                    "source": "no_id", "note": "no tophits for exposure",
                }
                continue
            cache_path = cache_root / _cache_key(exp_id, out_id, pval, r2, kb)
            cached = _read_cache(cache_path)
            if cached is not None and "beta" in cached and "se" in cached:
                beta = float(cached["beta"]) if cached["beta"] is not None else float("nan")
                se = float(cached["se"]) if cached["se"] is not None else float("nan")
                per_pair_results[out] = {
                    "exp_id": exp_id, "out_id": out_id,
                    "beta": beta, "se": se,
                    "n_snps": int(cached.get("n_snps", 0)),
                    "source": "cache", "note": cached.get("note", ""),
                }
                continue
            # Needs fetch.
            outcomes_to_fetch.append(out)

        # Pass 2: fetch any uncached outcomes via fan-out single-id calls
        # (one thread per outcome study).  Skipped silently when no JWT
        # is available so the loader can still serve cached pairs.
        if outcomes_to_fetch and not have_token:
            for out in outcomes_to_fetch:
                out_id = OPENGWAS_STUDY_IDS[out]
                per_pair_results[out] = {
                    "exp_id": exp_id, "out_id": out_id,
                    "beta": float("nan"), "se": float("nan"), "n_snps": 0,
                    "source": "no_id", "note": "uncached and OPENGWAS_JWT unset",
                }
            outcomes_to_fetch = []
        if outcomes_to_fetch and rsids and exp_id:
            study_ids_to_fetch = [OPENGWAS_STUDY_IDS[o] for o in outcomes_to_fetch]
            try:
                assocs_by_id = client.fetch_associations_by_study(
                    study_ids_to_fetch,
                    rsids,
                )
            except Exception as exc:
                if raise_on_error:
                    raise
                logger.warning(
                    "[opengwas] multi-assocs FAILED for %s: %s", exp, exc
                )
                assocs_by_id = {sid: [] for sid in study_ids_to_fetch}
                err = str(exc)
                for out in outcomes_to_fetch:
                    out_id = OPENGWAS_STUDY_IDS[out]
                    per_pair_results[out] = {
                        "exp_id": exp_id, "out_id": out_id,
                        "beta": float("nan"), "se": float("nan"), "n_snps": 0,
                        "source": "error", "note": err,
                    }
                outcomes_to_fetch = []

            for out in outcomes_to_fetch:
                out_id = OPENGWAS_STUDY_IDS[out]
                assocs = assocs_by_id.get(out_id, [])
                pairs = harmonise_pairs(hits, assocs)
                beta, se, k = ivw(pairs)
                cache_path = cache_root / _cache_key(exp_id, out_id, pval, r2, kb)
                _write_cache(
                    cache_path,
                    {
                        "exposure": exp,
                        "outcome": out,
                        "exposure_id": exp_id,
                        "outcome_id": out_id,
                        "beta": None if not np.isfinite(beta) else beta,
                        "se": None if not np.isfinite(se) else se,
                        "n_snps": int(k),
                        "params": {"pval": pval, "r2": r2, "kb": kb},
                        "fetched_at": time.time(),
                    },
                )
                per_pair_results[out] = {
                    "exp_id": exp_id, "out_id": out_id,
                    "beta": beta, "se": se, "n_snps": int(k),
                    "source": "fetched", "note": "",
                }

        # Pass 3: stamp results into the (β, SE) matrices.
        row_max_n = 0
        for j, out in enumerate(outcomes):
            if exp == out:
                continue
            r = per_pair_results.get(out)
            if r is None:
                continue
            beta = float(r["beta"])
            se = float(r["se"])
            k = int(r["n_snps"])
            if k == 0 or not np.isfinite(beta) or not np.isfinite(se) or se <= 0.0:
                _emit(per_pair, _PerPair(
                    exp, out, str(r["exp_id"]), str(r["out_id"]),
                    float("nan"), float("nan"), k,
                    source=("no_overlap" if k == 0 and r["source"] == "fetched"
                            else r["source"]),
                    note=str(r.get("note", "")),
                ))
                continue
            betas[i, j] = beta
            ses[i, j] = se
            row_max_n = max(row_max_n, k)
            citations[(exp, out)] = (
                f"OpenGWAS IVW: {STUDY_CITATIONS.get(str(r['exp_id']), str(r['exp_id']))} -> "
                f"{STUDY_CITATIONS.get(str(r['out_id']), str(r['out_id']))}"
            )
            _emit(per_pair, _PerPair(
                exp, out, str(r["exp_id"]), str(r["out_id"]),
                beta, se, k, source=str(r["source"]),
            ))
        n_snps[i] = row_max_n

    with np.errstate(invalid="ignore"):
        z = betas / ses
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    summary = GWASSummary(
        exposures=exposures,
        outcomes=outcomes,
        betas=betas,
        ses=ses,
        ivw_pvals=pvals,
        n_snps=n_snps,
        citations=citations,
        circular_pairs=tuple(circular),
    )
    summary.per_pair = per_pair  # type: ignore[attr-defined]
    summary.source_metadata = {  # type: ignore[attr-defined]
        "source": "opengwas",
        "cache_version": CACHE_VERSION,
        "params": {"pval": float(pval), "r2": float(r2), "kb": int(kb)},
        "drop_circular": bool(drop_circular),
        "study_ids": {
            trait: OPENGWAS_STUDY_IDS[trait]
            for trait in sorted(set(exposures) | set(outcomes))
            if trait in OPENGWAS_STUDY_IDS
        },
        "authenticated": bool(client.authenticated),
        "cache_dir": str(cache_root),
    }
    return summary


__all__ = [
    "OPENGWAS_STUDY_IDS",
    "STUDY_CITATIONS",
    "OpenGWASClient",
    "OpenGWASClientLike",
    "harmonise_pairs",
    "ivw",
    "load_live_gwas",
]
