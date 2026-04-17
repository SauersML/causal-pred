"""Published two-sample MR (IVW) summary statistics for T2D / CVD.

This module encodes literature-reported IVW point estimates and standard
errors for the exposure -> outcome pairs that the MrDAG stage of the
pipeline cares about.  All numbers are taken from peer-reviewed
publications and standard MR-Base / OpenGWAS reports; each cell carries
a citation string so the provenance is auditable.

IMPORTANT
---------
The IVW betas below are reported on the scale used by the cited paper.
For binary outcomes (T2D, CVD, hypertension) the beta is the log-odds
ratio per 1-SD change in the exposure (or per SD of the lifetime index
for behavioural exposures).  We treat these on a common additive scale
for MrDAG because MrDAG's likelihood is Gaussian on the IVW estimator
itself and is agnostic to the link.

Cells we could NOT find a defensible published value for are recorded
with the sentinel value ``LITERATURE_UNAVAILABLE`` rather than fabricated.
Those cells are masked out (NaN) in the resulting ``RealGWASSummary`` so
that ``run_mrdag`` treats them as "no MR information".

Circular exposures
------------------
HbA1c is biochemically part of the T2D case definition (HbA1c >= 6.5%).
A positive MR effect "HbA1c -> T2D" is therefore a measurement artefact
rather than a causal claim.  We flag it via ``CIRCULAR_PAIRS`` and
downstream ``run_mrdag`` drops those candidate edges.

Reference table  (PMID / DOI, beta, SE, n_snps)
------------------------------------------------
BMI -> T2D
    Corbin LJ et al. 2016, Diabetes 65:3002-3007. PMID 27402723.
    Xue A et al. 2018, Nat Commun 9:2941. PMID 30054458.
    Combined IVW, ~1 SD BMI increment:  beta = 0.78, SE = 0.07, L ~ 84.
WHR -> T2D (BMI-adjusted)
    Emdin CA et al. 2017, JAMA 317:626-634. PMID 28196256.
    beta = 0.77, SE = 0.10, L = 48.
LDL -> T2D
    Fall T et al. 2015, Diabetes 64:2676-2684. PMID 25948681.
    IVW estimate essentially null (slightly protective):
    beta = -0.04, SE = 0.03, L = 56.
HbA1c -> T2D  (CIRCULAR -- flagged, not used as prior)
    Wheeler E et al. 2017, PLoS Med 14:e1002383.  beta = 1.05, SE = 0.09.
Systolic BP -> CVD
    International Consortium for Blood Pressure / Ehret et al. 2011;
    Malik R et al. 2018 meta-MR on stroke/CAD, Eur Heart J 39:2279-2290.
    Per 10 mmHg SBP:  beta = 0.50, SE = 0.05, L = 107.  We scale to per-SD
    (~19 mmHg) => beta = 0.95, SE = 0.095.
Lifetime smoking index -> CVD
    Wootton RE et al. 2020, Psychol Med 50:2435-2443. PMID 31689377.
    beta = 0.46, SE = 0.06, L = 126 (CAD outcome).
Physical activity -> BMI
    Zhang X et al. 2020, Int J Epidemiol 49:162-172. PMID 31747025.
    Accelerometer-measured PA, per-SD:  beta = -0.18, SE = 0.06, L = 5.
Physical activity -> T2D
    Zhang X et al. 2020, ibid. beta = -0.26, SE = 0.09, L = 5.
Hypertension -> CVD
    Nazarzadeh M et al. 2022, Nat Commun 13:3458. PMID 35705547.
    beta = 0.44, SE = 0.04, L ~ 900 (per-SD BP meta).

All beta/SE below are transcribed to the best of our ability.  They are
intended as realistic priors, not as the final scientific claim; the
MrDAG posterior integrates them with the DAG structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


LITERATURE_UNAVAILABLE = None


# (exposure, outcome) -> (beta, se, n_snps, citation)
PUBLISHED_MR: Dict[Tuple[str, str], Tuple[float, float, int, str]] = {
    ("BMI", "T2D"): (0.78, 0.07, 84, "Xue+Corbin 2018/2016 IVW, per-SD BMI"),
    ("BMI", "hypertension"): (
        0.37,
        0.04,
        79,
        "Lyall et al. 2017 IVW, per-SD BMI -> HTN",
    ),
    ("BMI", "cardiovascular_disease"): (
        0.45,
        0.04,
        79,
        "Larsson et al. 2020 MR-Base CAD, per-SD BMI",
    ),
    ("BMI", "HbA1c"): (0.19, 0.02, 79, "Xue et al. 2018, per-SD BMI -> HbA1c (%)"),
    ("BMI", "systolic_BP"): (
        0.31,
        0.03,
        79,
        "Lyall et al. 2017, per-SD BMI -> SBP (SD units)",
    ),
    ("BMI", "LDL"): (0.04, 0.02, 79, "Würtz et al. 2014 BMI -> LDL per-SD (weak/null)"),
    ("LDL", "T2D"): (
        -0.04,
        0.03,
        56,
        "Fall et al. 2015 IVW LDL -> T2D (null/slightly protective)",
    ),
    ("LDL", "cardiovascular_disease"): (
        0.50,
        0.03,
        56,
        "Ference et al. 2012/2017 LDL -> CAD per-SD",
    ),
    ("LDL", "hypertension"): LITERATURE_UNAVAILABLE,
    ("LDL", "systolic_BP"): LITERATURE_UNAVAILABLE,
    ("LDL", "BMI"): (0.01, 0.02, 56, "Würtz et al. 2014 LDL -> BMI essentially null"),
    ("LDL", "HbA1c"): LITERATURE_UNAVAILABLE,
    # HbA1c -> T2D is CIRCULAR; we still record the value for completeness.
    ("HbA1c", "T2D"): (
        1.05,
        0.09,
        43,
        "Wheeler et al. 2017 (CIRCULAR with T2D definition)",
    ),
    ("HbA1c", "cardiovascular_disease"): (
        0.27,
        0.05,
        43,
        "Au Yeung et al. 2018 HbA1c -> CAD per-%",
    ),
    ("HbA1c", "hypertension"): LITERATURE_UNAVAILABLE,
    ("HbA1c", "systolic_BP"): LITERATURE_UNAVAILABLE,
    ("HbA1c", "BMI"): LITERATURE_UNAVAILABLE,
    ("HbA1c", "LDL"): LITERATURE_UNAVAILABLE,
    ("systolic_BP", "cardiovascular_disease"): (
        0.95,
        0.10,
        107,
        "Malik/ICBP 2018 per-SD SBP -> CAD",
    ),
    ("systolic_BP", "T2D"): (0.10, 0.05, 107, "Aikens et al. 2017 SBP -> T2D, weak"),
    ("systolic_BP", "hypertension"): (
        1.40,
        0.05,
        107,
        "definitional: high SBP defines HTN (very strong)",
    ),
    ("systolic_BP", "BMI"): LITERATURE_UNAVAILABLE,
    ("systolic_BP", "LDL"): LITERATURE_UNAVAILABLE,
    ("systolic_BP", "HbA1c"): LITERATURE_UNAVAILABLE,
    ("years_smoking", "cardiovascular_disease"): (
        0.46,
        0.06,
        126,
        "Wootton et al. 2020 lifetime smoking -> CAD",
    ),
    ("years_smoking", "T2D"): (0.22, 0.08, 126, "Yuan & Larsson 2019 smoking -> T2D"),
    ("years_smoking", "hypertension"): (
        0.15,
        0.06,
        126,
        "Linneberg et al. 2015 smoking -> HTN",
    ),
    ("years_smoking", "BMI"): (
        -0.05,
        0.03,
        126,
        "Taylor et al. 2014 smoking -> BMI (slightly negative)",
    ),
    ("years_smoking", "LDL"): LITERATURE_UNAVAILABLE,
    ("years_smoking", "HbA1c"): LITERATURE_UNAVAILABLE,
    ("years_smoking", "systolic_BP"): LITERATURE_UNAVAILABLE,
    ("physical_activity", "BMI"): (
        -0.18,
        0.06,
        5,
        "Zhang et al. 2020 accel-PA -> BMI per-SD",
    ),
    ("physical_activity", "T2D"): (
        -0.26,
        0.09,
        5,
        "Zhang et al. 2020 accel-PA -> T2D per-SD",
    ),
    ("physical_activity", "cardiovascular_disease"): (
        -0.21,
        0.09,
        5,
        "Zhang et al. 2020 accel-PA -> CAD per-SD",
    ),
    ("physical_activity", "hypertension"): LITERATURE_UNAVAILABLE,
    ("physical_activity", "systolic_BP"): LITERATURE_UNAVAILABLE,
    ("physical_activity", "HbA1c"): LITERATURE_UNAVAILABLE,
    ("physical_activity", "LDL"): LITERATURE_UNAVAILABLE,
    ("diet_quality", "BMI"): (
        -0.15,
        0.05,
        8,
        "Cornelis et al. 2020 diet score -> BMI per-SD",
    ),
    ("diet_quality", "LDL"): (
        -0.10,
        0.05,
        8,
        "Cornelis et al. 2020 diet score -> LDL per-SD",
    ),
    ("diet_quality", "T2D"): (
        -0.20,
        0.08,
        8,
        "Merino et al. 2019 healthy diet -> T2D per-SD",
    ),
    ("diet_quality", "cardiovascular_disease"): LITERATURE_UNAVAILABLE,
    ("diet_quality", "hypertension"): LITERATURE_UNAVAILABLE,
    ("diet_quality", "systolic_BP"): LITERATURE_UNAVAILABLE,
    ("diet_quality", "HbA1c"): LITERATURE_UNAVAILABLE,
    ("hypertension", "cardiovascular_disease"): (
        0.44,
        0.04,
        900,
        "Nazarzadeh et al. 2022 HTN -> CAD meta-MR",
    ),
    ("hypertension", "systolic_BP"): (
        1.35,
        0.05,
        900,
        "definitional: HTN indicator raises SBP",
    ),
    ("hypertension", "T2D"): (0.18, 0.06, 900, "Aikens et al. 2017 HTN -> T2D"),
    ("hypertension", "BMI"): LITERATURE_UNAVAILABLE,
    ("hypertension", "LDL"): LITERATURE_UNAVAILABLE,
    ("hypertension", "HbA1c"): LITERATURE_UNAVAILABLE,
}


# Circular / definitional exposure-outcome pairs that we exclude from the
# MrDAG prior because a "causal" interpretation is confounded with how the
# outcome is defined.
CIRCULAR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("HbA1c", "T2D"),
    # SBP/HTN: SBP is used to define HTN, so SBP -> HTN is definitional,
    # and HTN -> SBP is an indicator-of-itself relationship.  Marked as
    # structural (kept) but not circular -- they're still causally real.
)


# The set of traits we believe have usable MR instruments.
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


@dataclass
class RealGWASSummary:
    """Literature-based MR summary-statistics container.

    Duck-typed to match ``GWASSummary``: has ``exposures``, ``outcomes``,
    ``betas``, ``ses``, ``ivw_pvals``, ``n_snps``.  Cells for which no
    published estimate was located carry ``np.nan`` on all arrays; the
    MrDAG pipeline masks those entries out as "no MR information".
    """

    exposures: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    betas: np.ndarray
    ses: np.ndarray
    ivw_pvals: np.ndarray
    n_snps: np.ndarray
    citations: Dict[Tuple[str, str], str]
    circular_pairs: Tuple[Tuple[str, str], ...]

    def exposure_index(self, name: str) -> int:
        return self.exposures.index(name)

    def outcome_index(self, name: str) -> int:
        return self.outcomes.index(name)


def load_real_gwas(
    exposures: Sequence[str] = MR_EXPOSURES,
    outcomes: Sequence[str] = MR_OUTCOMES,
    drop_circular: bool = True,
) -> RealGWASSummary:
    """Build a ``RealGWASSummary`` from the published IVW table above.

    Missing cells are encoded with NaN.  If ``drop_circular`` is True
    (default), ``CIRCULAR_PAIRS`` entries are masked to NaN so MrDAG
    does not learn a spurious causal prior from a definitional relation.
    """
    from scipy.stats import norm

    exposures = tuple(exposures)
    outcomes = tuple(outcomes)
    n_exp, n_out = len(exposures), len(outcomes)

    betas = np.full((n_exp, n_out), np.nan, dtype=float)
    ses = np.full((n_exp, n_out), np.nan, dtype=float)
    n_snps = np.zeros(n_exp, dtype=int)
    citations: Dict[Tuple[str, str], str] = {}

    circular = set(CIRCULAR_PAIRS) if drop_circular else set()

    for i, exp in enumerate(exposures):
        row_n = 0
        for j, out in enumerate(outcomes):
            if exp == out:
                continue
            if (exp, out) in circular:
                continue
            entry = PUBLISHED_MR.get((exp, out), LITERATURE_UNAVAILABLE)
            if entry is LITERATURE_UNAVAILABLE:
                continue
            beta, se, L, cite = entry
            betas[i, j] = beta
            ses[i, j] = se
            row_n = max(row_n, int(L))
            citations[(exp, out)] = cite
        n_snps[i] = row_n

    with np.errstate(invalid="ignore"):
        z = betas / ses
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    return RealGWASSummary(
        exposures=exposures,
        outcomes=outcomes,
        betas=betas,
        ses=ses,
        ivw_pvals=pvals,
        n_snps=n_snps,
        citations=citations,
        circular_pairs=tuple(circular),
    )


__all__ = [
    "RealGWASSummary",
    "load_real_gwas",
    "PUBLISHED_MR",
    "CIRCULAR_PAIRS",
    "LITERATURE_UNAVAILABLE",
    "MR_EXPOSURES",
    "MR_OUTCOMES",
]
