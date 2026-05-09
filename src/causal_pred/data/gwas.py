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
from typing import Dict, Tuple

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


__all__ = [
    "GWASSummary",
    "MR_EXPOSURES",
    "MR_OUTCOMES",
    "CIRCULAR_PAIRS",
]
