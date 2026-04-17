"""MrDAG pipeline: Mendelian-randomisation edge priors.

Public interface (agents implementing this module must provide):

    run_mrdag(gwas, nodes=NODE_NAMES, rng=None, **kwargs) -> MrDAGResult

where ``MrDAGResult`` exposes at least ``pi`` (a (p, p) matrix of
edge-inclusion probabilities with zeros on the diagonal and on entries
whose parent is not an MR-eligible exposure).
"""

from .pipeline import MrDAGResult, run_mrdag  # noqa: F401
