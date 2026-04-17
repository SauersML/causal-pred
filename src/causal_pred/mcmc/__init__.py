"""Structure MCMC over DAGs with MrDAG prior.

Public interface
----------------

``run_structure_mcmc(data, node_types, start_adj, pi_prior,
                     n_samples=2000, burn_in=1000, thin=5,
                     n_chains=1, rng=None, **hyper) -> MCMCResult``

MCMCResult
----------
``samples``      list of (p, p) adjacency matrices (post burn-in, thinned).
``log_post``     log posterior at each sample.
``edge_probs``   (p, p) posterior marginal edge probabilities.
``diagnostics``  dict with acceptance rate, ESS, R-hat (if >1 chain).
"""

from .structure_mcmc import MCMCResult, run_structure_mcmc  # noqa: F401
