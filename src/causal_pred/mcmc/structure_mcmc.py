"""Structure MCMC over DAGs with an MrDAG edge-inclusion prior.

Overview
--------
Metropolis-Hastings sampler over the space of directed acyclic graphs on
``p`` nodes.  The target log posterior decomposes as

    log P(G | D, pi) = log S(G, D) + log P(G | pi) + const,

where ``S(G, D)`` is the marginal-likelihood score provided by
:mod:`causal_pred.scoring.mixed` (exact BGe for continuous nodes,
Laplace-approximated logistic marginal for binary / survival nodes) and
``P(G | pi)`` is an independent Bernoulli prior per potential edge.  For
each ordered pair ``(i, j)`` with ``i != j`` we set

    P(edge i -> j in G) = pi_ij if pi_prior[i, j] is finite,
                         = 0.5  if pi_prior[i, j] is NaN (no MR evidence).

Self-loops are forbidden.  Acyclicity is enforced at every proposal.

Moves
-----
Single-edge moves, proposed uniformly over the legal move set ``N(G)``:

  * add      (i, j):   adj[i, j] = 0 -> 1,    requires acyclicity.
  * remove   (i, j):   adj[i, j] = 1 -> 0.
  * reverse  (i, j):   adj[i, j] = 1, adj[j, i] = 0 -> adj[i, j] = 0,
                        adj[j, i] = 1, requires acyclicity.

Acceptance ratio (Madigan & York 1995; Giudici & Castelo 2003)
--------------------------------------------------------------
For proposal G -> G' drawn uniformly from the legal moves at G,

    q(G' | G) = 1 / |N(G)|,    q(G | G') = 1 / |N(G')|,

so the Metropolis-Hastings acceptance probability is

    alpha = min{1, [S(G', D) / S(G, D)]
                   * [P(G' | pi) / P(G | pi)]
                   * [|N(G)| / |N(G')|]}.

In log-space the code uses delta quantities from the scoring module and
closed-form log prior ratios, which are O(1) per move thanks to the
cache in :mod:`causal_pred.scoring.mixed`.

Notes on |N(G)|
---------------
|N(G)| = #{addable (i, j): adj is 0 and adding keeps acyclic}
      + #{removable (i, j): adj[i, j] = 1}
      + #{reversible (i, j): adj[i, j] = 1 and reversing stays acyclic}.

Computed exactly by DFS-based reachability at each iteration.  The cost
is O(p^4) worst-case for the neighbourhood count on p ~ 20 but in
practice dominated by the much smaller sparse graphs the sampler
spends most of its time in.

In addition to the single-edge moves, a **parent-set resample** hybrid
move (Madigan & York 1995 Section 4) is proposed on a fraction
``hybrid_prob`` of iterations.  The production path uses an exact Gibbs
variant: one step picks a target node j uniformly, removes its incoming
edges, enumerates the legal parent subsets under the configured parent
limit, and samples the new parent set from its conditional posterior.
This move is always accepted and is the main mixing path when the data
posterior is too sharp for random structural proposals.

The legacy random-flip variant remains available for unbounded toy runs:
one step picks a target node j uniformly, flips each candidate edge
(i, j) into/out of the parent set independently with probability
``resample_flip``, rejects any proposal that would create a cycle, and
otherwise accepts with

    log alpha = log S(G'; j) - log S(G; j)
              + sum_i [ I'(i,j) (log pi_ij - log(1-pi_ij))
                       - I(i,j)  (log pi_ij - log(1-pi_ij)) ]

where only S(G; j) (the local score at j) changes and the proposal
density ratio is 1: each edge is flipped independently with the same
probability in both directions, so the forward and reverse proposal
densities coincide on the symmetric difference between old and new
parent columns.  The move is a pure Metropolis-Hastings step and
preserves detailed balance; with ``hybrid_prob = 0`` the sampler
reduces to the single-edge chain.

Chains and diagnostics
----------------------
``n_chains`` independent sequential chains (no multiprocessing), each
starting from a lightly perturbed copy of ``start_adj`` with its own
RNG stream spawned from the user-supplied generator.  Diagnostics:

  * per-move-type acceptance rate,
  * mean log posterior per chain post-burn-in,
  * Gelman-Rubin R-hat on edge-inclusion indicators (sqrt of variance-
    of-means over mean-of-variances).  We report R-hat for both
    directed-edge indicators (``rhat_per_edge``, ``max_rhat_directed``)
    and the undirected skeleton ``adj | adj.T``
    (``rhat_per_skeleton_edge``, ``max_rhat_skeleton``).  The primary
    ``max_rhat`` statistic is the directed-edge R-hat because this pipeline
    reports directed causal edge and path probabilities and uses directed
    MR priors.  Skeleton R-hat is still reported as a secondary diagnostic.
  * effective sample size per edge via Geyer's initial-positive-
    sequence estimator (min ESS across edges is reported).

References
----------
* Madigan D., York J.  "Bayesian graphical models for discrete data."
  *International Statistical Review* 63(2), 215-232 (1995).
* Giudici P., Castelo R.  "Improving Markov chain Monte Carlo model
  search for data mining."  *Machine Learning* 50(1-2), 127-158 (2003).
* Kuipers J., Suter P., Moffa G.  "Efficient sampling and structure
  learning of Bayesian networks."  *J. Comput. Graph. Stat.* 31(3),
  639-650 (2022); see the BiDAG reference implementation for the
  neighbourhood-correction trick applied here.
* Geyer C. J.  "Practical Markov chain Monte Carlo."  *Statistical
  Science* 7(4), 473-483 (1992) -- initial-positive-sequence ESS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import comb
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..graph import is_dag
from ..scoring.mixed import (
    score_dag,
    score_delta_add_edge,
    score_delta_remove_edge,
    score_delta_reverse_edge,
    score_node,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult:
    samples: List[np.ndarray] = field(default_factory=list)
    log_post: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edge_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    diagnostics: dict = field(default_factory=dict)


ProgressCallback = Callable[[dict[str, Any]], None]


def _progress_callback(progress: bool | ProgressCallback) -> Optional[ProgressCallback]:
    if callable(progress):
        return progress
    if not progress:
        return None

    def _print_progress(payload: dict[str, Any]) -> None:
        chain = int(payload.get("chain", 0))
        n_chains = int(payload.get("n_chains", 0))
        iteration = int(payload.get("iter", 0))
        total_iters = int(payload.get("total_iters", 0))
        kept = int(payload.get("kept", 0))
        target = int(payload.get("target_samples", 0))
        elapsed = float(payload.get("elapsed_s", 0.0))
        eta = float(payload.get("eta_s", 0.0))
        event = str(payload.get("event", "progress"))
        phase = str(payload.get("phase", "unknown"))
        print(
            "[mcmc] "
            f"event={event} chain={chain}/{n_chains} "
            f"iter={iteration}/{total_iters} phase={phase} "
            f"kept={kept}/{target} edges={int(payload.get('edges', 0))} "
            f"logpost={float(payload.get('log_post', 0.0)):.2f} "
            f"iter_per_s={float(payload.get('iter_per_s', 0.0)):.2f} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )

    return _print_progress


# ---------------------------------------------------------------------------
# Log prior on a DAG given a (possibly NaN) edge-inclusion probability matrix.
# ---------------------------------------------------------------------------


def _prepare_prior(pi_prior: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log_pi, log_1m_pi) matrices with NaN entries mapped to 0.5.

    Diagonal entries are mapped to log(1) = 0 on both sides so they never
    contribute to the log prior (no self-loops are ever present).  All
    values are clipped away from {0, 1} by 1e-12 to avoid log(0).
    """
    Pi = np.asarray(pi_prior, dtype=float)
    if Pi.shape != (p, p):
        raise ValueError(f"pi_prior has shape {Pi.shape}, expected ({p}, {p})")
    # NaN -> uniform Bernoulli(0.5).
    Pi = np.where(np.isnan(Pi), 0.5, Pi)
    # Zero out diagonal contributions.
    np.fill_diagonal(Pi, 0.5)
    Pi = np.clip(Pi, 1e-12, 1.0 - 1e-12)
    log_pi = np.log(Pi)
    log_1m = np.log1p(-Pi)
    # Diagonal -> 0 on both sides so the diag contributes nothing to the sum.
    np.fill_diagonal(log_pi, 0.0)
    np.fill_diagonal(log_1m, 0.0)
    return log_pi, log_1m


def _log_prior(adj: np.ndarray, log_pi: np.ndarray, log_1m: np.ndarray) -> float:
    """log P(G | pi) under independent Bernoulli edge prior."""
    mask = adj.astype(bool)
    # Every off-diagonal entry contributes log_pi or log_1m; diag terms are 0.
    return float(np.where(mask, log_pi, log_1m).sum())


def _prepare_allowed_edges(allowed_edges: Optional[np.ndarray], p: int) -> np.ndarray:
    """Return the boolean structural mask used by every proposal kernel."""
    if allowed_edges is None:
        allowed = np.ones((p, p), dtype=bool)
    else:
        allowed = np.asarray(allowed_edges, dtype=bool).copy()
        if allowed.shape != (p, p):
            raise ValueError(
                f"allowed_edges has shape {allowed.shape}, expected ({p}, {p})"
            )
    np.fill_diagonal(allowed, False)
    return allowed


# ---------------------------------------------------------------------------
# Neighbourhood size |N(G)|.
# ---------------------------------------------------------------------------


def _ancestors_matrix(adj: np.ndarray) -> np.ndarray:
    """Boolean (p, p) matrix R with R[i, j] = True iff i reaches j (i -> .. -> j).

    Computed by Warshall's transitive-closure algorithm in O(p^3) via
    numpy boolean outer products.  Sets R[i, i] = False so the matrix
    does not include the trivial self-loop.
    """
    R = adj.astype(bool).copy()
    p = R.shape[0]
    # Warshall: for each intermediate k, propagate reachability.
    for k in range(p):
        R |= np.outer(R[:, k], R[k, :])
    np.fill_diagonal(R, False)
    return R


def _addable_mask(
    adj: np.ndarray,
    reach: np.ndarray,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean (p, p) mask of legal ADD moves.

    addable[i, j] iff:
      i != j
      adj[i, j] == 0
      adj[j, i] == 0            (reversed edge absent, so it's a pure add)
      not reach[j, i]           (j cannot reach i => no cycle after add)
    Note: if adj[j, i] == 1 then reach[j, i] is True via the direct edge,
    so the adj.T==0 check is implied by ~reach.T; we keep both for clarity.
    """
    p = adj.shape[0]
    off_diag = ~np.eye(p, dtype=bool)
    no_edge = (adj == 0) & (adj.T == 0) & off_diag
    addable = no_edge & (~reach.T)
    if allowed_edges is not None:
        addable &= allowed_edges
    if max_parents is not None:
        parent_counts = adj.sum(axis=0)
        addable &= parent_counts[np.newaxis, :] < int(max_parents)
    return addable


def _reversible_mask(
    adj: np.ndarray,
    reach: np.ndarray,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean (p, p) mask of edges (i, j) whose reversal keeps the DAG.

    For a present edge (i, j), reversing creates a cycle iff there is a
    directed path i -> ... -> j of length >= 2 (i.e. some route other than
    the direct edge).  Equivalently:

        (adj @ reach)[i, j] > 0

    because ``(adj @ reach)[i, j] = sum_k adj[i, k] * reach[k, j]`` and
    any contributing k != j gives a length >= 2 path i -> k -> .. -> j
    (the k == j term vanishes since reach has zero diagonal, and k == i
    cannot contribute since adj[i, i] = 0).  So an edge is reversible iff
    ``(adj @ reach)[i, j] == 0``.
    """
    A = adj.astype(bool)
    other_path = (A @ reach) > 0
    reversible = A & (~other_path)
    if allowed_edges is not None:
        reversible &= allowed_edges.T
    if max_parents is not None:
        parent_counts = adj.sum(axis=0)
        reversible &= parent_counts[:, np.newaxis] < int(max_parents)
    return reversible


def _count_neighbourhood(
    adj: np.ndarray,
    reach: Optional[np.ndarray] = None,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, np.ndarray]:
    """Count legal single-edge moves at ``adj``.

    Returns ``(n_add, n_remove, n_reverse, reach)`` where ``reach`` is the
    reachability matrix used to compute the counts (and may be re-used by
    the caller for move sampling).

    * n_add: ordered pairs (i, j) with i != j, adj[i, j] = 0, adj[j, i] = 0,
      and j does NOT reach i.
    * n_remove: count of present edges = adj.sum().
    * n_reverse: present edges (i, j) whose reversal keeps the DAG.
    """
    if reach is None:
        reach = _ancestors_matrix(adj)
    n_add = int(
        _addable_mask(
            adj,
            reach,
            max_parents=max_parents,
            allowed_edges=allowed_edges,
        ).sum()
    )
    n_remove = int(adj.sum())
    n_reverse = int(
        _reversible_mask(
            adj,
            reach,
            max_parents=max_parents,
            allowed_edges=allowed_edges,
        ).sum()
    )
    return n_add, n_remove, n_reverse, reach


def _sample_add_target(
    adj: np.ndarray,
    reach: np.ndarray,
    rng: np.random.Generator,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Sample (i, j) uniformly from the addable set."""
    idxs = np.argwhere(
        _addable_mask(
            adj,
            reach,
            max_parents=max_parents,
            allowed_edges=allowed_edges,
        )
    )
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


def _sample_remove_target(adj: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    idxs = np.argwhere(adj == 1)
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


def _sample_reverse_target(
    adj: np.ndarray,
    reach: np.ndarray,
    rng: np.random.Generator,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int]]:
    """Uniform sample from reversible edges (or None if none)."""
    idxs = np.argwhere(
        _reversible_mask(
            adj,
            reach,
            max_parents=max_parents,
            allowed_edges=allowed_edges,
        )
    )
    if idxs.shape[0] == 0:
        return None
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


# ---------------------------------------------------------------------------
# Initial-graph perturbation for multi-chain starts.
# ---------------------------------------------------------------------------


def _perturb_dag(
    start_adj: np.ndarray,
    n_flips: int,
    rng: np.random.Generator,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a perturbed acyclic copy of ``start_adj``.

    We attempt up to ``4 * n_flips`` random single-edge moves; each move is
    kept only if the resulting graph is still acyclic.  Keeps the flipped
    count close to ``n_flips`` while never returning a non-DAG.
    """
    adj = np.asarray(start_adj, dtype=np.int64).copy()
    p = adj.shape[0]
    if allowed_edges is None:
        allowed_edges = np.ones((p, p), dtype=bool)
        np.fill_diagonal(allowed_edges, False)
    budget = max(1, 4 * n_flips)
    applied = 0
    tries = 0
    while applied < n_flips and tries < budget:
        tries += 1
        kind = rng.integers(0, 3)  # 0 add, 1 remove, 2 reverse
        if kind == 0:
            i = int(rng.integers(0, p))
            j = int(rng.integers(0, p))
            if i == j or adj[i, j] or adj[j, i]:
                continue
            if not allowed_edges[i, j]:
                continue
            if max_parents is not None and adj[:, j].sum() >= int(max_parents):
                continue
            adj[i, j] = 1
            if is_dag(adj):
                applied += 1
            else:
                adj[i, j] = 0
        elif kind == 1:
            present = np.argwhere(adj == 1)
            if present.size == 0:
                continue
            k = int(rng.integers(0, present.shape[0]))
            i = int(present[k, 0])
            j = int(present[k, 1])
            adj[i, j] = 0
            applied += 1
        else:
            present = np.argwhere(adj == 1)
            if present.size == 0:
                continue
            k = int(rng.integers(0, present.shape[0]))
            i = int(present[k, 0])
            j = int(present[k, 1])
            if not allowed_edges[j, i]:
                continue
            if max_parents is not None and adj[:, i].sum() >= int(max_parents):
                continue
            adj[i, j] = 0
            adj[j, i] = 1
            if is_dag(adj):
                applied += 1
            else:
                adj[j, i] = 0
                adj[i, j] = 1
    assert is_dag(adj), "perturbation produced a non-DAG"
    return adj


# ---------------------------------------------------------------------------
# ESS and R-hat.
# ---------------------------------------------------------------------------


def _ess_ips(x: np.ndarray) -> float:
    """Effective sample size via Geyer's initial-positive-sequence estimator.

    Handles the common degenerate case ``var(x) == 0`` by returning the
    sample size (no mixing penalty for a constant chain).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float(n)
    v = float(np.var(x, ddof=0))
    if v <= 0.0:
        return float(n)
    x_c = x - x.mean()
    # FFT-based autocorrelation up to lag n-1.
    m = 1 << (int(np.ceil(np.log2(2 * n))))
    F = np.fft.rfft(x_c, n=m)
    acf = np.fft.irfft(F * np.conj(F), n=m)[:n] / (n * v)
    # Sum of consecutive pairs rho_{2k} + rho_{2k+1}; Geyer's theorem says
    # these should be non-negative; truncate at first non-positive pair.
    pair_sum = acf[0::2] + np.concatenate([acf[1::2], [0.0]])[: acf[0::2].size]
    # Find the first index where the positive-sequence pair becomes <= 0.
    cutoff = pair_sum.size
    for k in range(1, pair_sum.size):
        if pair_sum[k] <= 0.0:
            cutoff = k
            break
    tau = -1.0 + 2.0 * float(np.sum(pair_sum[:cutoff]))
    tau = max(tau, 1.0)  # never claim super-efficient sampling.
    return float(n) / tau


def _rhat_edgewise(chain_samples: Sequence[np.ndarray]) -> np.ndarray:
    """Classic (non-split) Gelman-Rubin R-hat per edge across chains.

    ``chain_samples`` is a list of (S, p, p) 0/1 arrays (one per chain,
    same S).  Constant edges return 1 only when every chain has the same
    constant value. Constant chains with different values report infinity.
    """
    m = len(chain_samples)
    if m < 2:
        p = chain_samples[0].shape[-1] if chain_samples else 0
        return np.ones((p, p), dtype=float)
    S = min(int(c.shape[0]) for c in chain_samples)
    if S < 2:
        p = chain_samples[0].shape[-1]
        return np.ones((p, p), dtype=float)
    X = np.stack([c[:S].astype(float) for c in chain_samples], axis=0)
    chain_means = X.mean(axis=1)  # (m, p, p)
    overall = chain_means.mean(axis=0)  # (p, p)
    B = (S / (m - 1.0)) * np.sum((chain_means - overall) ** 2, axis=0)
    W = X.var(axis=1, ddof=1).mean(axis=0)
    var_hat = ((S - 1.0) / S) * W + B / S
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_hat / W)
    # If every chain is internally constant and the chain means agree, the
    # edge is deterministically sampled and R-hat is 1.  If internally
    # constant chains disagree with one another, within-chain variance is zero
    # but between-chain variance is positive, which is non-convergence; report
    # infinity instead of hiding it as 1.
    const_within = W == 0.0
    means_disagree = np.ptp(chain_means, axis=0) > 0.0
    rhat = np.where(const_within & ~means_disagree, 1.0, rhat)
    rhat = np.where(const_within & means_disagree, np.inf, rhat)
    rhat = np.where(np.isnan(rhat), 1.0, rhat)
    return rhat


# ---------------------------------------------------------------------------
# Hybrid random parent-set resample move (Madigan & York 1995 Section 4).
#
# One iteration:
#   1. pick target node j uniformly from 0..p-1.
#   2. for every candidate i != j, flip the edge (i, j) independently w.p.
#      resample_flip.
#   3. if the resulting graph is acyclic, accept with MH probability
#         alpha = exp( [log S(G'; j) - log S(G; j)]
#                    + sum_i [I'(i,j) (log pi_ij - log(1-pi_ij))
#                             - I(i,j)  (log pi_ij - log(1-pi_ij))] )
#      (the proposal ratio log q(G|G')/q(G'|G) is zero because each flip is
#      symmetric: q = (1/p) * 0.3^|F| * 0.7^(|C|-|F|) with F = symmetric
#      difference between the old and new parent columns of j, which is the
#      same set whether we go forward or backward).
#
# The move only touches the parent set of node j, so only S(G; j) changes
# (other nodes' local scores cancel in the delta) and only column j of adj
# changes -- so acyclicity can be checked on the whole graph with is_dag.
# ---------------------------------------------------------------------------


def _hybrid_resample_parents(
    adj: np.ndarray,
    j: int,
    data: np.ndarray,
    node_types: Sequence[str],
    log_pi: np.ndarray,
    log_1m: np.ndarray,
    resample_flip: float,
    cache: dict,
    rng: np.random.Generator,
    hyper: dict,
    max_parents: Optional[int] = None,
    allowed_edges: Optional[np.ndarray] = None,
) -> Tuple[bool, float, float]:
    """Attempt a random parent-set resample at node ``j``.

    Returns ``(accepted, d_score, d_prior)``.  If accepted, ``adj`` is
    mutated in place; otherwise ``adj`` is unchanged.

    Candidates are all ``i != j`` (uniform-fallback scheme).  Each flip is
    independent with probability ``resample_flip``.  Cycle-creating
    proposals are rejected (treated as alpha = 0).
    """
    p = adj.shape[0]
    if allowed_edges is None:
        allowed_col = np.ones(p, dtype=bool)
        allowed_col[j] = False
    else:
        allowed_col = allowed_edges[:, j].astype(bool, copy=True)
        allowed_col[j] = False
    # Old parent column of j.
    old_col = adj[:, j].copy()

    # Flip each candidate edge with probability resample_flip.
    # Candidates are all i != j; mask out the diagonal.
    flip_draw = rng.random(p) < resample_flip
    flip_draw &= allowed_col
    new_col = old_col.copy()
    new_col[flip_draw] ^= 1  # XOR flip
    new_col[~allowed_col] = 0
    if max_parents is not None and int(new_col.sum()) > int(max_parents):
        return False, 0.0, 0.0

    # If nothing flipped, proposal = current; alpha = 1 (no-op).  Count as
    # accepted so the per-type rate reflects the true fraction of iterations
    # that end at the proposal.
    if not flip_draw.any():
        return True, 0.0, 0.0

    # Apply tentatively to adj and check acyclicity.
    adj[:, j] = new_col
    if not is_dag(adj):
        # Reject: restore and bail.
        adj[:, j] = old_col
        return False, 0.0, 0.0

    # Score delta at node j.
    old_parents = np.flatnonzero(old_col).tolist()
    new_parents = np.flatnonzero(new_col).tolist()
    old_score = score_node(j, old_parents, data, node_types, cache=cache, **hyper)
    new_score = score_node(j, new_parents, data, node_types, cache=cache, **hyper)
    d_score = float(new_score - old_score)

    # Prior delta at column j: independent Bernoulli over candidate edges.
    # Non-candidate entries have log_pi = log_1m = 0 on the diagonal so the
    # sum over all rows is equivalent to the sum over candidates.
    new_mask = new_col.astype(bool)
    old_mask = old_col.astype(bool)
    new_prior_col = np.where(new_mask, log_pi[:, j], log_1m[:, j]).sum()
    old_prior_col = np.where(old_mask, log_pi[:, j], log_1m[:, j]).sum()
    d_prior = float(new_prior_col - old_prior_col)

    # Proposal ratio is symmetric (see module docstring on this move) so
    # log q(G|G') - log q(G'|G) = 0.
    log_alpha = d_score + d_prior
    log_u = np.log(rng.random() + 1e-300)
    if log_u < log_alpha:
        return True, d_score, d_prior
    # Reject: restore old column.
    adj[:, j] = old_col
    return False, 0.0, 0.0


def _gibbs_resample_parents(
    adj: np.ndarray,
    j: int,
    data: np.ndarray,
    node_types: Sequence[str],
    log_pi: np.ndarray,
    log_1m: np.ndarray,
    max_parents: int,
    cache: dict,
    rng: np.random.Generator,
    hyper: dict,
    allowed_edges: Optional[np.ndarray] = None,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[bool, float, float]:
    """Exact conditional parent-set update for one node.

    Holding all outgoing edges and all other parent columns fixed, removing
    column ``j`` makes every subset of nodes not reachable from ``j`` legal.
    Sampling that subset from its local score times Bernoulli edge prior is
    a Gibbs step, so no Metropolis accept/reject correction is needed.
    """
    p = adj.shape[0]
    cap = int(max_parents)
    if cap < 0:
        raise ValueError("max_parents must be non-negative")

    old_col = adj[:, j].copy()
    base = adj.copy()
    base[:, j] = 0
    reach = _ancestors_matrix(base)
    if allowed_edges is None:
        allowed_col = np.ones(p, dtype=bool)
        allowed_col[j] = False
    else:
        allowed_col = allowed_edges[:, j].astype(bool, copy=True)
        allowed_col[j] = False
    candidates = [
        int(i) for i in range(p) if allowed_col[i] and i != j and not bool(reach[j, i])
    ]
    cap = min(cap, len(candidates))
    n_parent_sets = int(sum(comb(len(candidates), size) for size in range(cap + 1)))
    emit_parent_progress = progress is not None and n_parent_sets >= 500
    parent_started_at = time.time()
    if emit_parent_progress:
        progress(
            {
                "event": "exact_parent_sets_start",
                "target_node": int(j),
                "candidate_parents": int(len(candidates)),
                "parent_cap": int(cap),
                "parent_sets_total": n_parent_sets,
                "old_parent_count": int(old_col.sum()),
            }
        )

    old_parents = tuple(int(i) for i in np.flatnonzero(old_col))
    old_score = score_node(j, old_parents, data, node_types, cache=cache, **hyper)
    old_prior_col = float(
        np.where(old_col.astype(bool), log_pi[:, j], log_1m[:, j]).sum()
    )

    absent_prior = float(log_1m[candidates, j].sum()) if candidates else 0.0
    log_odds = log_pi[:, j] - log_1m[:, j]
    parent_sets: List[Tuple[int, ...]] = []
    log_weights: List[float] = []
    scored_sets = 0
    last_parent_progress_at = parent_started_at
    for size in range(cap + 1):
        for parents in combinations(candidates, size):
            parent_sets.append(tuple(int(i) for i in parents))
            score = score_node(j, parents, data, node_types, cache=cache, **hyper)
            prior = absent_prior + (
                float(log_odds[list(parents)].sum()) if parents else 0.0
            )
            log_weights.append(float(score + prior))
            scored_sets += 1
            now = time.time()
            if emit_parent_progress and now - last_parent_progress_at >= 10.0:
                progress(
                    {
                        "event": "exact_parent_sets_progress",
                        "target_node": int(j),
                        "candidate_parents": int(len(candidates)),
                        "parent_cap": int(cap),
                        "parent_sets_scored": int(scored_sets),
                        "parent_sets_total": n_parent_sets,
                        "parent_elapsed_s": float(now - parent_started_at),
                    }
                )
                last_parent_progress_at = now

    weights_log = np.asarray(log_weights, dtype=float)
    weights = np.exp(weights_log - float(weights_log.max()))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Gibbs parent-set weights are not finite")
    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
    idx = min(idx, len(parent_sets) - 1)

    new_parents = parent_sets[idx]
    new_col = np.zeros(p, dtype=np.int64)
    if new_parents:
        new_col[list(new_parents)] = 1

    adj[:, j] = new_col
    assert is_dag(adj), "Gibbs parent-set update produced a non-DAG"

    new_score = score_node(j, new_parents, data, node_types, cache=cache, **hyper)
    new_prior_col = float(
        np.where(new_col.astype(bool), log_pi[:, j], log_1m[:, j]).sum()
    )
    if emit_parent_progress:
        progress(
            {
                "event": "exact_parent_sets_complete",
                "target_node": int(j),
                "candidate_parents": int(len(candidates)),
                "parent_cap": int(cap),
                "parent_sets_scored": int(scored_sets),
                "parent_sets_total": n_parent_sets,
                "selected_parent_count": int(len(new_parents)),
                "parent_elapsed_s": float(time.time() - parent_started_at),
            }
        )
    return True, float(new_score - old_score), float(new_prior_col - old_prior_col)


def _gibbs_resample_edge_pair(
    adj: np.ndarray,
    i: int,
    j: int,
    data: np.ndarray,
    node_types: Sequence[str],
    log_pi: np.ndarray,
    log_1m: np.ndarray,
    max_parents: Optional[int],
    cache: dict,
    rng: np.random.Generator,
    hyper: dict,
    allowed_edges: Optional[np.ndarray] = None,
) -> Tuple[bool, float, float]:
    """Exact conditional update for the unordered pair ``{i, j}``.

    The three states are no edge, ``i -> j``, and ``j -> i``.  Illegal
    cyclic or parent-cap-violating states are omitted.  Sampling from the
    remaining conditional probabilities gives a rejection-free Gibbs move.
    """
    if i == j:
        return True, 0.0, 0.0

    old_i_parents = tuple(int(k) for k in np.flatnonzero(adj[:, i]))
    old_j_parents = tuple(int(k) for k in np.flatnonzero(adj[:, j]))
    old_score = score_node(i, old_i_parents, data, node_types, cache=cache, **hyper)
    old_score += score_node(j, old_j_parents, data, node_types, cache=cache, **hyper)
    old_prior = float(log_pi[i, j] if adj[i, j] else log_1m[i, j]) + float(
        log_pi[j, i] if adj[j, i] else log_1m[j, i]
    )

    base = adj.copy()
    base[i, j] = 0
    base[j, i] = 0
    reach = _ancestors_matrix(base)
    base_i_parents = tuple(int(k) for k in np.flatnonzero(base[:, i]))
    base_j_parents = tuple(int(k) for k in np.flatnonzero(base[:, j]))
    base_i_score = score_node(i, base_i_parents, data, node_types, cache=cache, **hyper)
    base_j_score = score_node(j, base_j_parents, data, node_types, cache=cache, **hyper)

    states: List[Tuple[int, float, float, Tuple[int, ...], Tuple[int, ...]]] = []
    none_prior = float(log_1m[i, j] + log_1m[j, i])
    states.append(
        (0, base_i_score + base_j_score, none_prior, base_i_parents, base_j_parents)
    )

    j_parent_count = len(base_j_parents)
    allow_ij = True if allowed_edges is None else bool(allowed_edges[i, j])
    allow_ji = True if allowed_edges is None else bool(allowed_edges[j, i])

    if (
        allow_ij
        and not bool(reach[j, i])
        and (max_parents is None or j_parent_count < int(max_parents))
    ):
        parents_j = tuple(sorted(base_j_parents + (int(i),)))
        score_j = score_node(j, parents_j, data, node_types, cache=cache, **hyper)
        prior = float(log_pi[i, j] + log_1m[j, i])
        states.append((1, base_i_score + score_j, prior, base_i_parents, parents_j))

    i_parent_count = len(base_i_parents)
    if (
        allow_ji
        and not bool(reach[i, j])
        and (max_parents is None or i_parent_count < int(max_parents))
    ):
        parents_i = tuple(sorted(base_i_parents + (int(j),)))
        score_i = score_node(i, parents_i, data, node_types, cache=cache, **hyper)
        prior = float(log_1m[i, j] + log_pi[j, i])
        states.append((2, score_i + base_j_score, prior, parents_i, base_j_parents))

    log_weights = np.asarray([score + prior for _, score, prior, _, _ in states])
    weights = np.exp(log_weights - float(log_weights.max()))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("edge-pair Gibbs weights are not finite")
    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
    idx = min(idx, len(states) - 1)
    state, new_score, new_prior, _, _ = states[idx]

    adj[i, j] = 0
    adj[j, i] = 0
    if state == 1:
        adj[i, j] = 1
    elif state == 2:
        adj[j, i] = 1
    assert is_dag(adj), "edge-pair Gibbs update produced a non-DAG"

    return True, float(new_score - old_score), float(new_prior - old_prior)


def _internal_prior(
    adj: np.ndarray,
    nodes: Sequence[int],
    log_pi: np.ndarray,
    log_1m: np.ndarray,
) -> float:
    total = 0.0
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            total += float(log_pi[u, v] if adj[u, v] else log_1m[u, v])
    return float(total)


def _gibbs_resample_node_block(
    adj: np.ndarray,
    nodes: Sequence[int],
    data: np.ndarray,
    node_types: Sequence[str],
    log_pi: np.ndarray,
    log_1m: np.ndarray,
    max_parents: Optional[int],
    cache: dict,
    rng: np.random.Generator,
    hyper: dict,
    allowed_edges: Optional[np.ndarray] = None,
) -> Tuple[bool, float, float]:
    """Exact Gibbs update for all directed edges inside a small node block."""
    block = tuple(int(x) for x in nodes)
    if len(set(block)) < 2:
        return True, 0.0, 0.0

    old_score = 0.0
    for node in block:
        parents = tuple(int(k) for k in np.flatnonzero(adj[:, node]))
        old_score += score_node(node, parents, data, node_types, cache=cache, **hyper)
    old_prior = _internal_prior(adj, block, log_pi, log_1m)

    base = adj.copy()
    for u in block:
        for v in block:
            if u != v:
                base[u, v] = 0

    unordered_pairs = list(combinations(block, 2))
    states: List[Tuple[np.ndarray, float, float]] = []
    log_weights: List[float] = []
    for choices in product((0, 1, 2), repeat=len(unordered_pairs)):
        cand = base.copy()
        for choice, (u, v) in zip(choices, unordered_pairs):
            if choice == 1:
                if allowed_edges is not None and not bool(allowed_edges[u, v]):
                    cand = None
                    break
                cand[u, v] = 1
            elif choice == 2:
                if allowed_edges is not None and not bool(allowed_edges[v, u]):
                    cand = None
                    break
                cand[v, u] = 1
        if cand is None:
            continue
        if max_parents is not None and int(cand[:, block].sum(axis=0).max()) > int(
            max_parents
        ):
            continue
        if not is_dag(cand):
            continue
        score = 0.0
        for node in block:
            parents = tuple(int(k) for k in np.flatnonzero(cand[:, node]))
            score += score_node(node, parents, data, node_types, cache=cache, **hyper)
        prior = _internal_prior(cand, block, log_pi, log_1m)
        states.append((cand, float(score), float(prior)))
        log_weights.append(float(score + prior))

    if not states:
        raise RuntimeError("node-block Gibbs update had no legal states")

    weights_log = np.asarray(log_weights, dtype=float)
    weights = np.exp(weights_log - float(weights_log.max()))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("node-block Gibbs weights are not finite")
    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
    idx = min(idx, len(states) - 1)
    new_adj, new_score, new_prior = states[idx]

    for u in block:
        for v in block:
            if u != v:
                adj[u, v] = new_adj[u, v]
    assert is_dag(adj), "node-block Gibbs update produced a non-DAG"
    return True, float(new_score - old_score), float(new_prior - old_prior)


# ---------------------------------------------------------------------------
# Core single-chain sampler
# ---------------------------------------------------------------------------


def _run_chain(
    data: np.ndarray,
    node_types: Sequence[str],
    start_adj: np.ndarray,
    log_pi: np.ndarray,
    log_1m: np.ndarray,
    n_samples: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
    cache: dict,
    hyper: dict,
    progress_callback: Optional[ProgressCallback] = None,
    chain_index: int = 0,
    n_chains: int = 1,
    progress_interval: Optional[int] = None,
    hybrid_prob: float = 0.1,
    resample_flip: float = 0.3,
    resample_flip_burn: Optional[float] = None,
    exact_parent_resample: bool = False,
    max_parents: Optional[int] = None,
    edge_resample_prob: float = 0.0,
    block_resample_prob: float = 0.0,
    block_size: int = 3,
    allowed_edges: Optional[np.ndarray] = None,
) -> dict:
    """Run one MCMC chain.  Returns a dict of per-chain outputs."""
    adj = start_adj.astype(np.int64, copy=True)
    # Anchor current log score and prior.
    cur_score = float(score_dag(adj, data, node_types, cache=cache, **hyper))
    cur_prior = _log_prior(adj, log_pi, log_1m)
    n_add, n_rem, n_rev, _reach = _count_neighbourhood(
        adj,
        max_parents=max_parents,
        allowed_edges=allowed_edges,
    )
    cur_nmoves = n_add + n_rem + n_rev

    total_iters = burn_in + n_samples * thin
    samples: List[np.ndarray] = []
    log_post_kept: List[float] = []

    prop = {
        "add": 0,
        "remove": 0,
        "reverse": 0,
        "hybrid": 0,
        "edge_gibbs": 0,
        "block_gibbs": 0,
    }
    accept = {"add": 0, "remove": 0, "reverse": 0, "hybrid": 0, "edge_gibbs": 0}
    accept["block_gibbs"] = 0
    progress_every = max(
        1,
        int(progress_interval)
        if progress_interval is not None
        else max(1, total_iters // 20),
    )
    chain_started_at = time.time()
    last_progress_at = chain_started_at

    def _keep_sample_if_due(it: int) -> None:
        if it >= burn_in and ((it - burn_in) % thin == 0):
            samples.append(adj.copy())
            log_post_kept.append(cur_score + cur_prior)

    def _accept_rate_snapshot() -> dict[str, float]:
        rates = {k: (accept[k] / prop[k]) if prop[k] > 0 else 0.0 for k in prop}
        mh_prop = prop["add"] + prop["remove"] + prop["reverse"]
        mh_acc = accept["add"] + accept["remove"] + accept["reverse"]
        rates["metropolis_hastings"] = mh_acc / mh_prop if mh_prop > 0 else 0.0
        total_prop = sum(prop.values())
        total_acc = sum(accept.values())
        rates["overall"] = total_acc / total_prop if total_prop > 0 else 0.0
        return rates

    def _emit_progress(
        it: int,
        event: str,
        *,
        force: bool = False,
        move_type: str | None = None,
        accepted: bool | None = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        nonlocal last_progress_at
        if progress_callback is None:
            return
        now = time.time()
        completed = max(0, it + 1)
        if not force and completed < total_iters:
            if completed % progress_every != 0 and now - last_progress_at < 30.0:
                return
        elapsed = max(now - chain_started_at, 1e-9)
        iter_per_s = completed / elapsed if completed > 0 else 0.0
        remaining = max(total_iters - completed, 0)
        eta_s = remaining / iter_per_s if iter_per_s > 0.0 else float("nan")
        phase = "burn_in" if completed <= burn_in else "sampling"
        payload: dict[str, Any] = {
            "event": event,
            "chain": int(chain_index + 1),
            "n_chains": int(n_chains),
            "iter": int(completed),
            "total_iters": int(total_iters),
            "burn_in": int(burn_in),
            "thin": int(thin),
            "phase": phase,
            "kept": int(len(samples)),
            "target_samples": int(n_samples),
            "elapsed_s": float(now - chain_started_at),
            "eta_s": float(eta_s),
            "iter_per_s": float(iter_per_s),
            "score": float(cur_score),
            "prior": float(cur_prior),
            "log_post": float(cur_score + cur_prior),
            "edges": int(adj.sum()),
            "n_add": int(n_add),
            "n_remove": int(n_rem),
            "n_reverse": int(n_rev),
            "n_moves": int(cur_nmoves),
            "move_type": move_type,
            "accepted": accepted,
            "proposals": dict(prop),
            "accepts": dict(accept),
            "accept_rate": _accept_rate_snapshot(),
            "score_cache_entries": int(len(cache)),
        }
        if extra:
            payload.update(extra)
        progress_callback(payload)
        last_progress_at = time.time()

    def _after_iteration(
        it: int,
        move_type: str,
        accepted: bool | None = None,
    ) -> None:
        _keep_sample_if_due(it)
        _emit_progress(
            it,
            "iteration",
            move_type=move_type,
            accepted=accepted,
        )

    _emit_progress(-1, "chain_start", force=True)

    # Effective hybrid flip rate per iteration: during burn-in we optionally
    # use ``resample_flip_burn`` (if provided) to take larger exploratory
    # jumps, then step down to ``resample_flip`` for sampling.  Detailed
    # balance still holds: each kernel applied at a given iteration is
    # symmetric in its proposal density, so the chain's stationary
    # distribution under the final kernel is unchanged by the burn-in
    # schedule (burn-in samples are discarded regardless).
    for it in range(total_iters):
        flip_it = (
            resample_flip
            if (resample_flip_burn is None or it >= burn_in)
            else float(resample_flip_burn)
        )
        kernel_u = rng.random()
        if block_resample_prob > 0.0 and kernel_u < block_resample_prob:
            size = min(max(2, int(block_size)), data.shape[1])
            nodes = rng.choice(data.shape[1], size=size, replace=False)
            prop["block_gibbs"] += 1
            accepted, d_score, d_prior = _gibbs_resample_node_block(
                adj,
                [int(x) for x in nodes],
                data,
                node_types,
                log_pi,
                log_1m,
                max_parents,
                cache,
                rng,
                hyper,
                allowed_edges=allowed_edges,
            )
            if accepted:
                cur_score += d_score
                cur_prior += d_prior
                n_add, n_rem, n_rev, _reach = _count_neighbourhood(
                    adj,
                    max_parents=max_parents,
                    allowed_edges=allowed_edges,
                )
                cur_nmoves = n_add + n_rem + n_rev
                accept["block_gibbs"] += 1
            _after_iteration(it, "block_gibbs", accepted=accepted)
            continue

        edge_cutoff = block_resample_prob + edge_resample_prob
        if edge_resample_prob > 0.0 and kernel_u < edge_cutoff:
            i, j = rng.choice(data.shape[1], size=2, replace=False)
            prop["edge_gibbs"] += 1
            accepted, d_score, d_prior = _gibbs_resample_edge_pair(
                adj,
                int(i),
                int(j),
                data,
                node_types,
                log_pi,
                log_1m,
                max_parents,
                cache,
                rng,
                hyper,
                allowed_edges=allowed_edges,
            )
            if accepted:
                cur_score += d_score
                cur_prior += d_prior
                n_add, n_rem, n_rev, _reach = _count_neighbourhood(
                    adj,
                    max_parents=max_parents,
                    allowed_edges=allowed_edges,
                )
                cur_nmoves = n_add + n_rem + n_rev
                accept["edge_gibbs"] += 1
            _after_iteration(it, "edge_gibbs", accepted=accepted)
            continue

        # Hybrid parent-set resample move with probability ``hybrid_prob``.
        if (
            hybrid_prob > 0.0
            and kernel_u < block_resample_prob + edge_resample_prob + hybrid_prob
        ):
            j_target = int(rng.integers(0, data.shape[1]))
            prop["hybrid"] += 1
            if exact_parent_resample:
                if max_parents is None:
                    raise ValueError(
                        "exact_parent_resample requires a finite max_parents"
                    )

                def _parent_progress(payload: dict[str, Any]) -> None:
                    _emit_progress(
                        it,
                        str(payload.get("event", "exact_parent_sets")),
                        force=True,
                        move_type="hybrid",
                        extra=payload,
                    )

                accepted, d_score, d_prior = _gibbs_resample_parents(
                    adj,
                    j_target,
                    data,
                    node_types,
                    log_pi,
                    log_1m,
                    int(max_parents),
                    cache,
                    rng,
                    hyper,
                    allowed_edges=allowed_edges,
                    progress=(
                        _parent_progress if progress_callback is not None else None
                    ),
                )
            else:
                accepted, d_score, d_prior = _hybrid_resample_parents(
                    adj,
                    j_target,
                    data,
                    node_types,
                    log_pi,
                    log_1m,
                    flip_it,
                    cache,
                    rng,
                    hyper,
                    max_parents=max_parents,
                    allowed_edges=allowed_edges,
                )
            if accepted:
                cur_score += d_score
                cur_prior += d_prior
                # Column j changed: refresh neighbourhood & reach.
                n_add, n_rem, n_rev, _reach = _count_neighbourhood(
                    adj,
                    max_parents=max_parents,
                    allowed_edges=allowed_edges,
                )
                cur_nmoves = n_add + n_rem + n_rev
                accept["hybrid"] += 1
            _after_iteration(it, "hybrid", accepted=accepted)
            continue

        if cur_nmoves == 0:
            # Degenerate; nothing to do besides record.
            _after_iteration(it, "none", accepted=None)
            continue

        # Pick move type by proportional weight.
        u = rng.integers(0, cur_nmoves)
        if u < n_add:
            mtype = "add"
            i, j = _sample_add_target(
                adj,
                _reach,
                rng,
                max_parents=max_parents,
                allowed_edges=allowed_edges,
            )
        elif u < n_add + n_rem:
            mtype = "remove"
            i, j = _sample_remove_target(adj, rng)
        else:
            mtype = "reverse"
            sel = _sample_reverse_target(
                adj,
                _reach,
                rng,
                max_parents=max_parents,
                allowed_edges=allowed_edges,
            )
            # Should always succeed because n_rev > 0 here.
            assert sel is not None, "sampler state inconsistent"
            i, j = sel

        prop[mtype] += 1

        # Score delta.
        if mtype == "add":
            d_score = score_delta_add_edge(
                i, j, adj, data, node_types, cache=cache, **hyper
            )
            d_prior = float(log_pi[i, j] - log_1m[i, j])
        elif mtype == "remove":
            d_score = score_delta_remove_edge(
                i, j, adj, data, node_types, cache=cache, **hyper
            )
            d_prior = float(log_1m[i, j] - log_pi[i, j])
        else:  # reverse
            d_score = score_delta_reverse_edge(
                i, j, adj, data, node_types, cache=cache, **hyper
            )
            # adj[i, j] -> adj[j, i]; i,j lost presence, j,i gained presence.
            d_prior = float(
                (log_1m[i, j] - log_pi[i, j]) + (log_pi[j, i] - log_1m[j, i])
            )

        # Propose.
        if mtype == "add":
            adj[i, j] = 1
        elif mtype == "remove":
            adj[i, j] = 0
        else:
            adj[i, j] = 0
            adj[j, i] = 1

        # Neighbourhood of proposed graph.
        new_add, new_rem, new_rev, new_reach = _count_neighbourhood(
            adj,
            max_parents=max_parents,
            allowed_edges=allowed_edges,
        )
        new_nmoves = new_add + new_rem + new_rev

        # Acceptance on the log scale.
        log_alpha = d_score + d_prior + np.log(cur_nmoves) - np.log(max(new_nmoves, 1))
        log_u = np.log(rng.random() + 1e-300)

        accepted_move = False
        if log_u < log_alpha and new_nmoves > 0:
            # Accept.
            cur_score += d_score
            cur_prior += d_prior
            n_add, n_rem, n_rev = new_add, new_rem, new_rev
            cur_nmoves = new_nmoves
            _reach = new_reach
            accept[mtype] += 1
            accepted_move = True
        else:
            # Revert.
            if mtype == "add":
                adj[i, j] = 0
            elif mtype == "remove":
                adj[i, j] = 1
            else:
                adj[j, i] = 0
                adj[i, j] = 1

        _after_iteration(it, mtype, accepted=accepted_move)

    # Keep only the first n_samples (the tail may briefly exceed due to edge
    # rounding: e.g. if total_iters % thin != 0).
    if len(samples) > n_samples:
        samples = samples[:n_samples]
        log_post_kept = log_post_kept[:n_samples]

    _emit_progress(total_iters - 1, "chain_complete", force=True)

    return {
        "samples": samples,
        "log_post": np.asarray(log_post_kept, dtype=float),
        "prop": prop,
        "accept": accept,
        "final_adj": adj.copy(),
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_structure_mcmc(
    data,
    node_types: Sequence[str],
    start_adj: np.ndarray,
    pi_prior: np.ndarray,
    n_samples: int = 2000,
    burn_in: int = 1000,
    thin: int = 5,
    n_chains: int = 4,
    rng: Optional[np.random.Generator] = None,
    progress: bool | ProgressCallback = False,
    progress_interval: Optional[int] = None,
    perturb_flips: int = 5,
    hybrid_prob: float = 0.1,
    resample_flip: float = 0.3,
    resample_flip_burn: Optional[float] = None,
    exact_parent_resample: bool = False,
    max_parents: Optional[int] = None,
    edge_resample_prob: float = 0.0,
    block_resample_prob: float = 0.0,
    block_size: int = 3,
    allowed_edges: Optional[np.ndarray] = None,
    **hyper,
) -> MCMCResult:
    """Structure MCMC over DAGs with an MrDAG edge-inclusion prior.

    Parameters
    ----------
    data : array-like (n, p)
        Design matrix in the same column order as ``node_types``.
    node_types : sequence of str (length p)
        ``"continuous"`` / ``"binary"`` / ``"survival"`` per column.
    start_adj : array-like (p, p)
        Initial acyclic adjacency (e.g. the DAGSLAM MAP estimate).
    pi_prior : array-like (p, p)
        Per-edge Bernoulli prior.  ``NaN`` entries are replaced by 0.5.
    n_samples : int
        Number of post-burn-in samples to retain per chain (after thinning).
    burn_in : int
        Number of burn-in iterations per chain.
    thin : int
        Thin factor: one sample is kept every ``thin`` iterations
        post-burn-in.  Total iterations per chain = burn_in + n_samples * thin.
    n_chains : int
        Independent chains, run sequentially (no multiprocessing).
    rng : numpy.random.Generator, optional
        Parent RNG; per-chain streams are spawned from it.
    progress : bool or callable
        If True, print structured progress during each chain. If callable,
        call it with a progress payload dict.
    progress_interval : int, optional
        Emit one iteration heartbeat every ``progress_interval`` iterations
        per chain. Time-based heartbeats still emit at least every 30 seconds.
    perturb_flips : int
        Target number of random edge flips (kept acyclic) applied to
        ``start_adj`` to form each chain's initial graph.  The first
        chain starts from an exact copy of ``start_adj``.
    hybrid_prob : float, default 0.1
        Per-iteration probability of proposing a parent-set resample
        (Madigan & York 1995 Section 4) instead of a single-edge move.
        ``0.0`` disables the hybrid move.
    edge_resample_prob : float, default 0.0
        Per-iteration probability of an exact Gibbs update over one
        unordered node pair: no edge, ``i -> j``, or ``j -> i``.  This
        is useful for sharp posteriors where random add/remove/reverse
        proposals mostly reject.
    block_resample_prob : float, default 0.0
        Per-iteration probability of an exact Gibbs update over all
        directed edges inside a small node block.
    block_size : int, default 3
        Number of nodes in the block Gibbs update.
    resample_flip : float, default 0.3
        Per-candidate flip probability inside the hybrid move.  With a
        symmetric-flip proposal the forward / backward densities cancel
        regardless of this value; we expose it for tuning the move's
        effective step size.
    resample_flip_burn : float, optional
        If provided, the hybrid-move flip probability used during the
        burn-in window (``it < burn_in``).  After burn-in the move
        switches to ``resample_flip``.  A larger value here (e.g. 0.1)
        lets chains take big exploratory jumps during burn-in so they
        concentrate in the same posterior mode before the sampling
        phase starts, which helps R-hat on multi-modal targets.
    exact_parent_resample : bool, default False
        If True, the hybrid move is an exact Gibbs update over legal
        parent sets for one node.  This requires ``max_parents``.
    max_parents : int, optional
        Maximum number of incoming edges per node.  When set, both
        single-edge moves and exact parent-set resampling stay within
        this bounded DAG space.
    allowed_edges : array-like, optional
        Boolean ``(p, p)`` structural mask.  False entries are forbidden
        directed edges across every proposal kernel.

    Returns
    -------
    MCMCResult
    """
    X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    _n, p = X.shape
    if len(node_types) != p:
        raise ValueError(f"node_types length {len(node_types)} != p={p}")
    start_adj = np.asarray(start_adj, dtype=np.int64)
    if start_adj.shape != (p, p):
        raise ValueError(f"start_adj has shape {start_adj.shape}, expected ({p}, {p})")
    if not is_dag(start_adj):
        raise ValueError("start_adj is not acyclic")
    allowed = _prepare_allowed_edges(allowed_edges, p)
    if np.any((start_adj != 0) & (~allowed)):
        bad = np.argwhere((start_adj != 0) & (~allowed))
        i, j = (int(bad[0, 0]), int(bad[0, 1]))
        raise ValueError(f"start_adj contains forbidden edge ({i}, {j})")
    if max_parents is not None:
        max_parents = int(max_parents)
        if max_parents < 0:
            raise ValueError("max_parents must be non-negative")
        max_start_parents = int(start_adj.sum(axis=0).max(initial=0))
        if max_start_parents > max_parents:
            raise ValueError(
                f"start_adj has a node with {max_start_parents} parents, "
                f"exceeding max_parents={max_parents}"
            )
    if exact_parent_resample and max_parents is None:
        raise ValueError("exact_parent_resample requires max_parents")
    edge_resample_prob = float(edge_resample_prob)
    block_resample_prob = float(block_resample_prob)
    hybrid_prob = float(hybrid_prob)
    if edge_resample_prob < 0.0 or hybrid_prob < 0.0 or block_resample_prob < 0.0:
        raise ValueError(
            "block_resample_prob, edge_resample_prob, and hybrid_prob must be non-negative"
        )
    if block_resample_prob + edge_resample_prob + hybrid_prob > 1.0:
        raise ValueError(
            "block_resample_prob + edge_resample_prob + hybrid_prob must be <= 1"
        )
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    if rng is None:
        rng = np.random.default_rng()
    log_pi, log_1m = _prepare_prior(pi_prior, p)

    # Per-chain RNG streams.
    if hasattr(rng, "spawn"):
        child_rngs = rng.spawn(n_chains)
    else:
        seeds = rng.integers(0, 2**31 - 1, size=n_chains)
        child_rngs = [np.random.default_rng(int(s)) for s in seeds]

    # Shared score cache across chains (scores are graph-state-free).
    cache: dict = {}
    emit = _progress_callback(progress)
    sampler_started_at = time.time()
    total_iters_per_chain = int(burn_in + n_samples * thin)
    if emit is not None:
        emit(
            {
                "event": "sampler_start",
                "chain": 0,
                "n_chains": int(n_chains),
                "iter": 0,
                "total_iters": total_iters_per_chain,
                "burn_in": int(burn_in),
                "thin": int(thin),
                "phase": "setup",
                "kept": 0,
                "target_samples": int(n_samples),
                "elapsed_s": 0.0,
                "eta_s": float("nan"),
                "iter_per_s": 0.0,
                "score": 0.0,
                "prior": 0.0,
                "log_post": 0.0,
                "edges": int(start_adj.sum()),
                "n": int(_n),
                "p": int(p),
                "n_allowed_edges": int(allowed.sum()),
                "start_edges": int(start_adj.sum()),
                "samples_per_chain": int(n_samples),
                "edge_resample_prob": float(edge_resample_prob),
                "parent_resample_prob": float(hybrid_prob),
                "block_resample_prob": float(block_resample_prob),
                "exact_parent_resample": bool(exact_parent_resample),
                "max_parents": None if max_parents is None else int(max_parents),
            }
        )

    all_chain_samples: List[np.ndarray] = []
    all_log_post: List[np.ndarray] = []
    prop_totals = {
        "add": 0,
        "remove": 0,
        "reverse": 0,
        "hybrid": 0,
        "edge_gibbs": 0,
        "block_gibbs": 0,
    }
    accept_totals = {
        "add": 0,
        "remove": 0,
        "reverse": 0,
        "hybrid": 0,
        "edge_gibbs": 0,
        "block_gibbs": 0,
    }
    mean_lp_per_chain: List[float] = []

    for c in range(n_chains):
        if c == 0:
            start_c = start_adj.copy()
        else:
            start_c = _perturb_dag(
                start_adj,
                perturb_flips,
                child_rngs[c],
                max_parents=max_parents,
                allowed_edges=allowed,
            )
        chain_out = _run_chain(
            X,
            node_types,
            start_c,
            log_pi,
            log_1m,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            rng=child_rngs[c],
            cache=cache,
            hyper=hyper,
            progress_callback=emit,
            chain_index=c,
            n_chains=n_chains,
            progress_interval=progress_interval,
            hybrid_prob=hybrid_prob,
            resample_flip=resample_flip,
            resample_flip_burn=resample_flip_burn,
            exact_parent_resample=exact_parent_resample,
            max_parents=max_parents,
            edge_resample_prob=edge_resample_prob,
            block_resample_prob=block_resample_prob,
            block_size=block_size,
            allowed_edges=allowed,
        )
        # Stack this chain's samples into a (S, p, p) array.
        if chain_out["samples"]:
            arr = np.stack(chain_out["samples"], axis=0).astype(np.int8)
        else:
            arr = np.zeros((0, p, p), dtype=np.int8)
        all_chain_samples.append(arr)
        all_log_post.append(chain_out["log_post"])
        for k in prop_totals:
            prop_totals[k] += chain_out["prop"][k]
            accept_totals[k] += chain_out["accept"][k]
        if chain_out["log_post"].size:
            mean_lp_per_chain.append(float(chain_out["log_post"].mean()))
        else:
            mean_lp_per_chain.append(float("nan"))
        if emit is not None:
            elapsed = time.time() - sampler_started_at
            emit(
                {
                    "event": "chain_stacked",
                    "chain": int(c + 1),
                    "n_chains": int(n_chains),
                    "iter": total_iters_per_chain,
                    "total_iters": total_iters_per_chain,
                    "burn_in": int(burn_in),
                    "thin": int(thin),
                    "phase": "chain_done",
                    "kept": int(arr.shape[0]),
                    "target_samples": int(n_samples),
                    "elapsed_s": float(elapsed),
                    "eta_s": float("nan"),
                    "iter_per_s": float(
                        ((c + 1) * total_iters_per_chain) / max(elapsed, 1e-9)
                    ),
                    "score": 0.0,
                    "prior": 0.0,
                    "log_post": float(mean_lp_per_chain[-1]),
                    "edges": int(arr[-1].sum()) if arr.shape[0] else 0,
                    "score_cache_entries": int(len(cache)),
                    "chain_samples": int(arr.shape[0]),
                }
            )

    # Concatenate samples across chains for edge_probs and the flat sample list.
    if sum(a.shape[0] for a in all_chain_samples) > 0:
        flat = np.concatenate(all_chain_samples, axis=0)
    else:
        flat = np.zeros((0, p, p), dtype=np.int8)
    if flat.shape[0] > 0:
        edge_probs = flat.astype(np.float64).mean(axis=0)
    else:
        edge_probs = np.zeros((p, p), dtype=float)
    np.fill_diagonal(edge_probs, 0.0)

    flat_samples_list = [flat[k] for k in range(flat.shape[0])]
    log_post_concat = np.concatenate(all_log_post) if all_log_post else np.zeros(0)

    # Acceptance rates per move type.
    accept_rate = {
        k: (accept_totals[k] / prop_totals[k]) if prop_totals[k] > 0 else 0.0
        for k in prop_totals
    }
    total_prop = sum(prop_totals.values())
    total_accept = sum(accept_totals.values())
    accept_rate["overall"] = total_accept / total_prop if total_prop > 0 else 0.0
    # ``overall`` mixes Metropolis-Hastings single-edge moves with Gibbs
    # blocks; Gibbs steps record an "accept" whenever the resampled state
    # is taken (no MH reject), so a high overall accept rate mostly
    # reflects the Gibbs share rather than MH efficiency.  Report the
    # MH-only rate alongside it -- this is the diagnostic users want for
    # tuning.
    mh_keys = ("add", "remove", "reverse")
    mh_prop = sum(prop_totals[k] for k in mh_keys)
    mh_acc = sum(accept_totals[k] for k in mh_keys)
    accept_rate["metropolis_hastings"] = mh_acc / mh_prop if mh_prop > 0 else 0.0
    if emit is not None:
        elapsed = time.time() - sampler_started_at
        emit(
            {
                "event": "diagnostics_start",
                "chain": int(n_chains),
                "n_chains": int(n_chains),
                "iter": total_iters_per_chain,
                "total_iters": total_iters_per_chain,
                "burn_in": int(burn_in),
                "thin": int(thin),
                "phase": "diagnostics",
                "kept": int(flat.shape[0]),
                "target_samples": int(n_samples * n_chains),
                "elapsed_s": float(elapsed),
                "eta_s": float("nan"),
                "iter_per_s": float(
                    (n_chains * total_iters_per_chain) / max(elapsed, 1e-9)
                ),
                "score": 0.0,
                "prior": 0.0,
                "log_post": float(log_post_concat.mean())
                if log_post_concat.size
                else float("nan"),
                "edges": int(flat[-1].sum()) if flat.shape[0] else 0,
                "score_cache_entries": int(len(cache)),
                "proposals": dict(prop_totals),
                "accepts": dict(accept_totals),
                "accept_rate": dict(accept_rate),
            }
        )

    # R-hat and ESS diagnostics.
    # We compute R-hat both on the directed-edge indicators and on the
    # undirected skeleton (adj | adj.T).  Directed R-hat is the primary
    # ``max_rhat`` diagnostic because downstream summaries are directed
    # causal edge/path probabilities and the prior carries directed MR
    # evidence.  Skeleton R-hat remains useful for separating orientation
    # mixing problems from skeleton mixing problems.
    if n_chains > 1 and all(a.shape[0] > 1 for a in all_chain_samples):
        rhat_mat = _rhat_edgewise(all_chain_samples)
        skel_chains = [
            ((c.astype(np.int8) | c.transpose(0, 2, 1).astype(np.int8)) > 0)
            for c in all_chain_samples
        ]
        rhat_skel = _rhat_edgewise(skel_chains)
        # Upper-triangle mask of off-diagonal: skeleton is symmetric so
        # we take the max over i < j only.
        triu = np.triu(np.ones((p, p), dtype=bool), k=1)
        rhat_skel_scored = rhat_skel[triu]
        finite_skel = np.isfinite(rhat_skel_scored)
        max_rhat_skel = (
            float(np.nanmax(rhat_skel_scored)) if rhat_skel_scored.size else 1.0
        )
        max_finite_rhat_skel = (
            float(np.max(rhat_skel_scored[finite_skel]))
            if np.any(finite_skel)
            else float("nan")
        )
        n_infinite_rhat_skel = int(np.sum(np.isposinf(rhat_skel_scored)))
        off = ~np.eye(p, dtype=bool)
        rhat_dir_scored = rhat_mat[off]
        finite_dir = np.isfinite(rhat_dir_scored)
        max_rhat_dir = (
            float(np.nanmax(rhat_dir_scored)) if rhat_dir_scored.size else 1.0
        )
        max_finite_rhat_dir = (
            float(np.max(rhat_dir_scored[finite_dir]))
            if np.any(finite_dir)
            else float("nan")
        )
        n_infinite_rhat_dir = int(np.sum(np.isposinf(rhat_dir_scored)))
        max_rhat = max_rhat_dir
    else:
        rhat_mat = np.ones((p, p), dtype=float)
        rhat_skel = np.ones((p, p), dtype=float)
        max_rhat = 1.0
        max_rhat_dir = 1.0
        max_rhat_skel = 1.0
        max_finite_rhat_dir = 1.0
        max_finite_rhat_skel = 1.0
        n_infinite_rhat_dir = 0
        n_infinite_rhat_skel = 0

    # Per-edge ESS: sum the per-chain ESS rather than computing on the
    # concatenated trace.  Concatenating chains that occupy different
    # posterior modes inserts a step at each chain boundary which inflates
    # the empirical autocorrelation and collapses the IPS estimator's ESS
    # toward O(n_modes).  The sum of independent per-chain ESS is the
    # standard practice (Vehtari et al. 2021, "Rank-normalization,
    # folding, and localization").
    eligible_chains = [c for c in all_chain_samples if c.shape[0] >= 4]
    if eligible_chains:
        ess_mat = np.zeros((p, p), dtype=float)
        for i in range(p):
            for j in range(p):
                if i == j:
                    ess_mat[i, j] = float(flat.shape[0])
                    continue
                ess_mat[i, j] = float(
                    sum(_ess_ips(c[:, i, j].astype(float)) for c in eligible_chains)
                )
        off = ~np.eye(p, dtype=bool)
        min_ess = float(np.min(ess_mat[off])) if off.any() else float(flat.shape[0])
    else:
        ess_mat = np.full((p, p), float(flat.shape[0]))
        min_ess = float(flat.shape[0])

    diagnostics = {
        "accept_rate": accept_rate,
        "proposals_per_type": dict(prop_totals),
        "accepts_per_type": dict(accept_totals),
        "mean_log_posterior_per_chain": mean_lp_per_chain,
        "rhat_per_edge": rhat_mat,
        "rhat_per_skeleton_edge": rhat_skel,
        "max_rhat": max_rhat,
        "max_rhat_directed": max_rhat_dir,
        "max_rhat_skeleton": max_rhat_skel,
        "max_finite_rhat_directed": max_finite_rhat_dir,
        "max_finite_rhat_skeleton": max_finite_rhat_skel,
        "n_infinite_rhat_directed": n_infinite_rhat_dir,
        "n_infinite_rhat_skeleton": n_infinite_rhat_skel,
        "ess_per_edge": ess_mat,
        "min_ess": min_ess,
        "n_chains": int(n_chains),
        "n_samples_per_chain": [int(a.shape[0]) for a in all_chain_samples],
        "burn_in": int(burn_in),
        "thin": int(thin),
        "exact_parent_resample": bool(exact_parent_resample),
        "max_parents": None if max_parents is None else int(max_parents),
        "edge_resample_prob": float(edge_resample_prob),
        "parent_resample_prob": float(hybrid_prob),
        "block_resample_prob": float(block_resample_prob),
        "block_size": int(block_size),
        "n_allowed_edges": int(allowed.sum()),
    }
    if emit is not None:
        elapsed = time.time() - sampler_started_at
        emit(
            {
                "event": "diagnostics_complete",
                "chain": int(n_chains),
                "n_chains": int(n_chains),
                "iter": total_iters_per_chain,
                "total_iters": total_iters_per_chain,
                "burn_in": int(burn_in),
                "thin": int(thin),
                "phase": "complete",
                "kept": int(flat.shape[0]),
                "target_samples": int(n_samples * n_chains),
                "elapsed_s": float(elapsed),
                "eta_s": 0.0,
                "iter_per_s": float(
                    (n_chains * total_iters_per_chain) / max(elapsed, 1e-9)
                ),
                "score": 0.0,
                "prior": 0.0,
                "log_post": float(log_post_concat.mean())
                if log_post_concat.size
                else float("nan"),
                "edges": int(flat[-1].sum()) if flat.shape[0] else 0,
                "score_cache_entries": int(len(cache)),
                "proposals": dict(prop_totals),
                "accepts": dict(accept_totals),
                "accept_rate": dict(accept_rate),
                "max_rhat_directed": float(max_rhat_dir),
                "max_rhat_skeleton": float(max_rhat_skel),
                "min_ess": float(min_ess),
                "samples_total": int(flat.shape[0]),
            }
        )

    return MCMCResult(
        samples=flat_samples_list,
        log_post=log_post_concat,
        edge_probs=edge_probs,
        diagnostics=diagnostics,
    )


__all__ = ["MCMCResult", "run_structure_mcmc"]
