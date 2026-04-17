"""Sanity checks for simulated GWAS summary statistics."""

from causal_pred.data.gwas import MR_EXPOSURES, MR_OUTCOMES


def test_shapes(small_gwas):
    g = small_gwas
    assert g.betas.shape == (len(g.exposures), len(g.outcomes))
    assert g.betas.shape == g.ses.shape == g.ivw_pvals.shape
    assert g.exposures == MR_EXPOSURES
    assert g.outcomes == MR_OUTCOMES


def test_bmi_has_effect_on_t2d(small_gwas):
    i = small_gwas.exposure_index("BMI")
    j = small_gwas.outcome_index("T2D")
    assert small_gwas.betas[i, j] > 0.3
    assert small_gwas.ivw_pvals[i, j] < 0.05


def test_ldl_has_no_effect_on_t2d(small_gwas):
    i = small_gwas.exposure_index("LDL")
    j = small_gwas.outcome_index("T2D")
    # LDL -> T2D is set to 0 in the ground truth; |beta|/se should be small.
    z = small_gwas.betas[i, j] / small_gwas.ses[i, j]
    assert abs(z) < 3.0
