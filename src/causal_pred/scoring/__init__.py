"""Mixed-variable DAG scoring.

Public interface
----------------

``score_node(j, parents, data, node_types, cache=None, **hyper) -> float``
    Returns the log marginal likelihood of node ``j`` conditioned on its
    parents.  Log domain, additive over nodes.  ``cache`` may be a
    ``dict``-like mapping ``(j, frozenset(parents)) -> float`` that the
    scorer will update; reusing the cache is the critical optimisation
    for MCMC.

``score_dag(adj, data, node_types, cache=None, **hyper) -> float``
    Sum of ``score_node`` over all nodes.

``score_delta_add_edge(i, j, adj, data, node_types, cache=None, **hyper)``
``score_delta_remove_edge(i, j, adj, data, node_types, cache=None, **hyper)``
``score_delta_reverse_edge(i, j, adj, data, node_types, cache=None, **hyper)``
    Return the log score difference after applying the proposed move.
    These are what MCMC actually calls.
"""

from .mixed import (  # noqa: F401
    score_node,
    score_dag,
    score_delta_add_edge,
    score_delta_remove_edge,
    score_delta_reverse_edge,
)
