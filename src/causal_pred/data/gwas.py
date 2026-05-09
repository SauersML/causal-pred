"""GWAS summary-statistics container and trait set used by MrDAG.

The MrDAG stage of the pipeline consumes per-pair IVW point estimates
and standard errors on a fixed set of exposure/outcome traits.  The
estimates themselves are never hard-coded here -- they are produced by
:func:`causal_pred.data.opengwas.load_live_gwas`, which performs a real
two-sample MR (LD-clumped tophits + outcome harmonisation + IVW) against
the OpenGWAS REST API and caches the per-pair results to disk.

This module only declares:

* :data:`MR_EXPOSURES` / :data:`MR_OUTCOMES` -- the trait set MrDAG operates on,
* :data:`CIRCULAR_PAIRS` -- exposure/outcome pairs that are definitionally
  confounded (e.g. HbA1c is part of the T2D case definition; SBP is part
  of the hypertension definition).  These are masked to NaN by the live
  loader so MrDAG does not learn a spurious causal prior from a
  measurement artefact.
* :class:`GWASSummary` -- the in-memory container used by ``run_mrdag``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


MR_EXPOSURES: Tuple[str, ...] = (
    "BMI",
    "LDL",
    "HbA1c",
    "systolic_BP",
    "years_smoking",
    "physical_activity",
    "diet_quality",
    "hypertension",
)

MR_OUTCOMES: Tuple[str, ...] = (
    "BMI",
    "LDL",
    "HbA1c",
    "systolic_BP",
    "hypertension",
    "cardiovascular_disease",
    "T2D",
)


# Circular / definitional exposure-outcome pairs that we exclude from the
# MR prior because a "causal" interpretation is confounded with how the
# outcome is defined.
#
#   HbA1c -> T2D                  : HbA1c >= 6.5% is part of the T2D
#                                    case definition.
#   systolic_BP <-> hypertension  : SBP is used to define HTN, and HTN is
#                                    an indicator of the same blood-pressure
#                                    process rather than an ordinary cause.
CIRCULAR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("HbA1c", "T2D"),
    ("systolic_BP", "hypertension"),
    ("hypertension", "systolic_BP"),
)


@dataclass
class GWASSummary:
    """Per-(exposure, outcome) IVW summary statistics.

    Attributes
    ----------
    exposures, outcomes : tuple[str]
        Trait names in row / column order.
    betas, ses : (n_exp, n_out) arrays
        IVW point estimates and standard errors of the causal effect
        ``exposure -> outcome``.  NaN where no usable cell is available
        (e.g. circular pair, no overlapping instruments, or the cache /
        OpenGWAS fetch returned nothing).
    ivw_pvals : (n_exp, n_out) array
        Two-sided p-values of ``betas / ses`` under a standard normal.
    n_snps : (n_exp,) array
        Maximum number of harmonised instrument SNPs across the cells of
        each exposure row.
    citations : dict
        Optional per-pair provenance string keyed by ``(exposure, outcome)``.
    circular_pairs : tuple
        Pairs that were dropped as definitional/circular.
    """

    exposures: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    betas: np.ndarray
    ses: np.ndarray
    ivw_pvals: np.ndarray
    n_snps: np.ndarray
    citations: Dict[Tuple[str, str], str] = field(default_factory=dict)
    circular_pairs: Tuple[Tuple[str, str], ...] = ()

    def exposure_index(self, name: str) -> int:
        return self.exposures.index(name)

    def outcome_index(self, name: str) -> int:
        return self.outcomes.index(name)


def _repo_mr_cache_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "mr_cache"


@cache
def _cached_open_gwas_summary(
    exposures: Tuple[str, ...],
    outcomes: Tuple[str, ...],
) -> GWASSummary:
    from .opengwas import OpenGWASClient, load_live_gwas

    summary = load_live_gwas(
        exposures=exposures,
        outcomes=outcomes,
        client=OpenGWASClient(token=None),
        cache_dir=_repo_mr_cache_dir(),
        drop_circular=True,
    )
    usable = np.isfinite(summary.betas) & np.isfinite(summary.ses) & (summary.ses > 0.0)
    if not np.any(usable):
        raise RuntimeError(
            "cached OpenGWAS MR summaries are unavailable; expected usable IVW cells "
            f"under {_repo_mr_cache_dir()}"
        )
    return summary


def simulate_gwas(
    exposures: Sequence[str] = MR_EXPOSURES,
    outcomes: Sequence[str] = MR_OUTCOMES,
) -> GWASSummary:
    """Return cached OpenGWAS IVW total-effect summaries in GWASSummary form."""
    exposures = tuple(exposures)
    outcomes = tuple(outcomes)
    summary = _cached_open_gwas_summary(exposures, outcomes)
    return GWASSummary(
        exposures=exposures,
        outcomes=outcomes,
        betas=np.array(summary.betas, dtype=float, copy=True),
        ses=np.array(summary.ses, dtype=float, copy=True),
        ivw_pvals=np.array(summary.ivw_pvals, dtype=float, copy=True),
        n_snps=np.array(summary.n_snps, dtype=int, copy=True),
        citations=dict(summary.citations),
        circular_pairs=tuple(summary.circular_pairs),
    )


__all__ = [
    "GWASSummary",
    "MR_EXPOSURES",
    "MR_OUTCOMES",
    "CIRCULAR_PAIRS",
    "simulate_gwas",
]
