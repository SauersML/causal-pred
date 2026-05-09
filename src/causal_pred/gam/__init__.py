"""Distributional survival GAM.

Public interface
----------------

``fit_survival_gam(time, event, X, columns=None, n_uncertainty_slices=1000, ...) -> SurvivalGAM``

``SurvivalGAM``
    ``predict_survival(X_new, t_grid) -> (n_slices, n_new, n_t)``
        Deterministic delta-method normal-quantile slices from gamfit
        survival standard errors.
    ``predict_survival_mean(X_new, t_grid) -> (n_new, n_t)``
    ``predict_survival_se(X_new, t_grid) -> (n_new, n_t)``
    ``predict_median_survival(X_new) -> (n_slices, n_new)``
    ``posterior_summary() -> dict``  gamfit fit diagnostics.

Bayesian model averaging over several parent-sets:

``bma_survival(parent_sets, weights, time, event, data, columns_per_set,
              t_grid, X_new_per_set, **gam_kwargs)``

Returns averaged survival predictions with backend within-model variance
where available plus structural uncertainty from parent-set weights.
"""

from .survival import SurvivalGAM, fit_survival_gam, bma_survival  # noqa: F401
