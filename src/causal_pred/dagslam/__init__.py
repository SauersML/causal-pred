"""DAGSLAM-style hill-climbing search over mixed-variable nodes.

Public interface
----------------

``run_dagslam(data, node_types, max_parents=5, restarts=5, rng=None,
             **hyper) -> DAGSLAMResult``

Returns an object with ``adjacency`` (a (p, p) 0/1 matrix, acyclic),
``log_score``, and ``trace`` fields.  The adjacency is consumed by the
MCMC sampler as its initial state.
"""

from .search import DAGSLAMResult, run_dagslam  # noqa: F401
