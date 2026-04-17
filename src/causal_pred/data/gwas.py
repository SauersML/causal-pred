"""Simulated GWAS summary statistics for the MrDAG stage.

MrDAG only needs *per-variant* effect estimates and standard errors on
each "exposure" trait and on each "outcome" trait.  For the proposal's
MR set-up this is:

    * L  independent SNPs (instruments, approximately LD-free)
    * a set of exposure traits (continuous clinical / lifestyle
      mediators for which genetic instruments exist)
    * a set of outcome traits (diseases / continuous phenotypes)

This module produces a deterministic, realistic-looking GWASSummary
object from the ground-truth DAG in ``nodes.py`` so that:

    * exposures that truly causally affect an outcome show a non-zero
      MR effect estimate with moderate standard errors,
    * exposures with no causal link to an outcome show estimates
      scattered around zero,
    * a small amount of horizontal pleiotropy is added so MrDAG has to
      do real work (not the noise-free oracle).

The result mimics what you would get by running IVW regression on
real two-sample MR summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np



# Which nodes have strong enough genetic instruments to serve as MR
# exposures in this project?  In real practice exposures are continuous
# traits with >= a handful of genome-wide-significant hits.
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

# Candidate outcomes (downstream nodes) that MR targets.
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
class GWASSummary:
    """Per-exposure / per-outcome MR summary statistics.

    Attributes
    ----------
    exposures, outcomes : tuple[str]
        Trait names in row / column order.
    betas, ses : (n_exp, n_out) arrays
        IVW point estimates and standard errors of the causal effect
        exposure -> outcome.
    ivw_pvals : (n_exp, n_out) array
        Two-sided p-values of ``betas / ses`` under a standard normal.
    n_snps : (n_exp,) array
        Number of instrument SNPs per exposure.
    """

    exposures: Tuple[str, ...]
    outcomes: Tuple[str, ...]
    betas: np.ndarray
    ses: np.ndarray
    ivw_pvals: np.ndarray
    n_snps: np.ndarray

    def exposure_index(self, name: str) -> int:
        return self.exposures.index(name)

    def outcome_index(self, name: str) -> int:
        return self.outcomes.index(name)


def _true_mr_effect(parent: str, child: str) -> float:
    """Return the population-level MR effect size for a (parent, child)
    edge in the ground-truth DAG.  Non-edges return 0."""
    effects = {
        ("BMI", "T2D"): 0.55,
        ("BMI", "hypertension"): 0.40,
        ("BMI", "HbA1c"): 0.25,
        ("HbA1c", "T2D"): 0.45,
        ("LDL", "T2D"): 0.00,  # LDL is NOT causal for T2D
        ("LDL", "cardiovascular_disease"): 0.30,
        ("systolic_BP", "cardiovascular_disease"): 0.38,
        ("years_smoking", "cardiovascular_disease"): 0.28,
        ("physical_activity", "BMI"): -0.22,
        ("diet_quality", "BMI"): -0.20,
        ("diet_quality", "LDL"): -0.15,
        ("hypertension", "cardiovascular_disease"): 0.35,
        ("hypertension", "systolic_BP"): 0.50,
    }
    return effects.get((parent, child), 0.0)


def simulate_gwas(
    rng: Optional[np.random.Generator] = None,
    exposures: Sequence[str] = MR_EXPOSURES,
    outcomes: Sequence[str] = MR_OUTCOMES,
    pleiotropy: float = 0.05,
    base_se: float = 0.04,
    snps_per_exposure_range: Tuple[int, int] = (20, 80),
) -> GWASSummary:
    """Generate a GWASSummary that reflects the ground-truth DAG."""
    if rng is None:
        rng = np.random.default_rng(1)

    exposures = tuple(exposures)
    outcomes = tuple(outcomes)
    n_exp, n_out = len(exposures), len(outcomes)

    betas = np.zeros((n_exp, n_out))
    ses = np.zeros((n_exp, n_out))
    n_snps = rng.integers(
        snps_per_exposure_range[0], snps_per_exposure_range[1] + 1, size=n_exp
    )

    # Intercept: a small amount of pleiotropy per (exposure, outcome)
    pleio = rng.normal(0.0, pleiotropy, size=(n_exp, n_out))

    for i, exp in enumerate(exposures):
        for j, out in enumerate(outcomes):
            if exp == out:
                betas[i, j] = np.nan
                ses[i, j] = np.nan
                continue
            mu = _true_mr_effect(exp, out) + pleio[i, j]
            se = base_se * np.sqrt(40.0 / max(n_snps[i], 1))
            betas[i, j] = mu + rng.normal(0.0, se)
            ses[i, j] = se

    # Z-tests -> p-values (two-sided).
    with np.errstate(invalid="ignore"):
        z = betas / ses
    # Use scipy if available, else vectorised erfc.
    from scipy.stats import norm

    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    return GWASSummary(
        exposures=exposures,
        outcomes=outcomes,
        betas=betas,
        ses=ses,
        ivw_pvals=pvals,
        n_snps=n_snps,
    )


__all__ = [
    "GWASSummary",
    "simulate_gwas",
    "MR_EXPOSURES",
    "MR_OUTCOMES",
]
