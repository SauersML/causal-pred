"""Distributional survival GAM.

Public interface
----------------

``fit_survival_gam(time, event, X, n_basis=10, n_samples=1000, warmup=500,
                   rng=None, **hyper) -> SurvivalGAM``

``SurvivalGAM``
    ``predict_survival(X_new, t_grid) -> (n_samples, n_new, n_t)``
        Posterior survival function draws.
    ``predict_median_survival(X_new) -> (n_samples, n_new)``
    ``posterior_summary() -> dict``  effective sample sizes, divergences.

Bayesian model averaging over several parent-sets:

``bma_survival(parent_sets, weights, time, event, data, columns_per_set,
              t_grid, X_new_per_set, **gam_kwargs)``

Returns averaged survival predictions with uncertainty from both
structural (weights) and parametric (GAM posterior) sources.
"""

from .survival import SurvivalGAM, fit_survival_gam, bma_survival  # noqa: F401
