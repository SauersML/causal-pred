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

In addition to the single-edge moves, a **random parent-set resample**
hybrid move (Madigan & York 1995 Section 4) is proposed on a fraction
``hybrid_prob`` of iterations.  One step picks a target node j
uniformly, flips each candidate edge (i, j) into/out of the parent set
independently with probability ``resample_flip``, rejects any proposal
that would create a cycle, and otherwise accepts with

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
    ``max_rhat`` statistic is the skeleton R-hat: DAGs that are Markov-
    equivalent (share a CPDAG) differ on directed-edge indicators but
    agree on the skeleton, so the skeleton R-hat is the appropriate
    convergence diagnostic for observational DAG inference, which can
    only identify the equivalence class.
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
from typing import List, Optional, Sequence, Tuple

import numpy as np

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


# ---------------------------------------------------------------------------
# Acyclicity utilities
# ---------------------------------------------------------------------------


def _is_reachable(adj: np.ndarray, src: int, dst: int) -> bool:
    """True iff there is a directed path ``src -> ... -> dst``.

    ``src == dst`` returns True (self-reachable).  Uses a numpy-backed
    DFS over the adjacency matrix.
    """
    if src == dst:
        return True
    p = adj.shape[0]
    visited = np.zeros(p, dtype=bool)
    visited[src] = True
    stack: List[int] = [int(src)]
    while stack:
        u = stack.pop()
        nbrs = np.flatnonzero(adj[u])
        for v in nbrs:
            vv = int(v)
            if vv == dst:
                return True
            if not visited[vv]:
                visited[vv] = True
                stack.append(vv)
    return False


def _is_dag(adj: np.ndarray) -> bool:
    """Kahn's algorithm DAG check."""
    in_deg = adj.sum(axis=0).astype(int).copy()
    p = adj.shape[0]
    stack = [int(i) for i in range(p) if in_deg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in np.flatnonzero(adj[u]):
            vv = int(v)
            in_deg[vv] -= 1
            if in_deg[vv] == 0:
                stack.append(vv)
    return seen == p


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


def _addable_mask(adj: np.ndarray, reach: np.ndarray) -> np.ndarray:
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
    return no_edge & (~reach.T)


def _reversible_mask(adj: np.ndarray, reach: np.ndarray) -> np.ndarray:
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
    return A & (~other_path)


def _count_neighbourhood(
    adj: np.ndarray, reach: Optional[np.ndarray] = None
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
    n_add = int(_addable_mask(adj, reach).sum())
    n_remove = int(adj.sum())
    n_reverse = int(_reversible_mask(adj, reach).sum())
    return n_add, n_remove, n_reverse, reach


def _sample_add_target(
    adj: np.ndarray, reach: np.ndarray, rng: np.random.Generator
) -> Tuple[int, int]:
    """Sample (i, j) uniformly from the addable set."""
    idxs = np.argwhere(_addable_mask(adj, reach))
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


def _sample_remove_target(adj: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    idxs = np.argwhere(adj == 1)
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


def _sample_reverse_target(
    adj: np.ndarray, reach: np.ndarray, rng: np.random.Generator
) -> Optional[Tuple[int, int]]:
    """Uniform sample from reversible edges (or None if none)."""
    idxs = np.argwhere(_reversible_mask(adj, reach))
    if idxs.shape[0] == 0:
        return None
    k = int(rng.integers(0, idxs.shape[0]))
    return int(idxs[k, 0]), int(idxs[k, 1])


# ---------------------------------------------------------------------------
# Initial-graph perturbation for multi-chain starts.
# ---------------------------------------------------------------------------


def _perturb_dag(
    start_adj: np.ndarray, n_flips: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a perturbed acyclic copy of ``start_adj``.

    We attempt up to ``4 * n_flips`` random single-edge moves; each move is
    kept only if the resulting graph is still acyclic.  Keeps the flipped
    count close to ``n_flips`` while never returning a non-DAG.
    """
    adj = np.asarray(start_adj, dtype=np.int64).copy()
    p = adj.shape[0]
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
            adj[i, j] = 1
            if _is_dag(adj):
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
            adj[i, j] = 0
            adj[j, i] = 1
            if _is_dag(adj):
                applied += 1
            else:
                adj[j, i] = 0
                adj[i, j] = 1
    assert _is_dag(adj), "perturbation produced a non-DAG"
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
    same S).  Constant edges (variance 0 within every chain) return 1.
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
        rhat = np.sqrt(var_hat / np.where(W > 0, W, 1.0))
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    # Edges that are constant in every chain cannot have R-hat; set to 1.
    const_edge = W == 0.0
    rhat = np.where(const_edge, 1.0, rhat)
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
# changes -- so acyclicity can be checked on the whole graph with _is_dag.
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
) -> Tuple[bool, float, float]:
    """Attempt a random parent-set resample at node ``j``.

    Returns ``(accepted, d_score, d_prior)``.  If accepted, ``adj`` is
    mutated in place; otherwise ``adj`` is unchanged.

    Candidates are all ``i != j`` (uniform-fallback scheme).  Each flip is
    independent with probability ``resample_flip``.  Cycle-creating
    proposals are rejected (treated as alpha = 0).
    """
    p = adj.shape[0]
    # Old parent column of j.
    old_col = adj[:, j].copy()

    # Flip each candidate edge with probability resample_flip.
    # Candidates are all i != j; mask out the diagonal.
    flip_draw = rng.random(p) < resample_flip
    flip_draw[j] = False  # self-loop never a candidate
    new_col = old_col.copy()
    new_col[flip_draw] ^= 1  # XOR flip

    # If nothing flipped, proposal = current; alpha = 1 (no-op).  Count as
    # accepted so the per-type rate reflects the true fraction of iterations
    # that end at the proposal.
    if not flip_draw.any():
        return True, 0.0, 0.0

    # Apply tentatively to adj and check acyclicity.
    adj[:, j] = new_col
    if not _is_dag(adj):
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
    progress: bool,
    hyper: dict,
    hybrid_prob: float = 0.1,
    resample_flip: float = 0.3,
    resample_flip_burn: Optional[float] = None,
) -> dict:
    """Run one MCMC chain.  Returns a dict of per-chain outputs."""
    data.shape[1]
    adj = start_adj.astype(np.int64, copy=True)
    # Anchor current log score and prior.
    cur_score = float(score_dag(adj, data, node_types, cache=cache, **hyper))
    cur_prior = _log_prior(adj, log_pi, log_1m)
    n_add, n_rem, n_rev, _reach = _count_neighbourhood(adj)
    cur_nmoves = n_add + n_rem + n_rev

    total_iters = burn_in + n_samples * thin
    samples: List[np.ndarray] = []
    log_post_kept: List[float] = []

    prop = {"add": 0, "remove": 0, "reverse": 0, "hybrid": 0}
    accept = {"add": 0, "remove": 0, "reverse": 0, "hybrid": 0}

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
        # Hybrid parent-set resample move with probability ``hybrid_prob``.
        if hybrid_prob > 0.0 and rng.random() < hybrid_prob:
            j_target = int(rng.integers(0, data.shape[1]))
            prop["hybrid"] += 1
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
            )
            if accepted:
                cur_score += d_score
                cur_prior += d_prior
                # Column j changed: refresh neighbourhood & reach.
                n_add, n_rem, n_rev, _reach = _count_neighbourhood(adj)
                cur_nmoves = n_add + n_rem + n_rev
                accept["hybrid"] += 1
            if it >= burn_in and ((it - burn_in) % thin == 0):
                samples.append(adj.copy())
                log_post_kept.append(cur_score + cur_prior)
            continue

        if cur_nmoves == 0:
            # Degenerate; nothing to do besides record.
            if it >= burn_in and ((it - burn_in) % thin == 0):
                samples.append(adj.copy())
                log_post_kept.append(cur_score + cur_prior)
            continue

        # Pick move type by proportional weight.
        u = rng.integers(0, cur_nmoves)
        if u < n_add:
            mtype = "add"
            i, j = _sample_add_target(adj, _reach, rng)
        elif u < n_add + n_rem:
            mtype = "remove"
            i, j = _sample_remove_target(adj, rng)
        else:
            mtype = "reverse"
            sel = _sample_reverse_target(adj, _reach, rng)
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
        new_add, new_rem, new_rev, new_reach = _count_neighbourhood(adj)
        new_nmoves = new_add + new_rem + new_rev

        # Acceptance on the log scale.
        log_alpha = d_score + d_prior + np.log(cur_nmoves) - np.log(max(new_nmoves, 1))
        log_u = np.log(rng.random() + 1e-300)

        if log_u < log_alpha and new_nmoves > 0:
            # Accept.
            cur_score += d_score
            cur_prior += d_prior
            n_add, n_rem, n_rev = new_add, new_rem, new_rev
            cur_nmoves = new_nmoves
            _reach = new_reach
            accept[mtype] += 1
        else:
            # Revert.
            if mtype == "add":
                adj[i, j] = 0
            elif mtype == "remove":
                adj[i, j] = 1
            else:
                adj[j, i] = 0
                adj[i, j] = 1

        if it >= burn_in and ((it - burn_in) % thin == 0):
            samples.append(adj.copy())
            log_post_kept.append(cur_score + cur_prior)

        if progress and (it + 1) % max(1, total_iters // 10) == 0:
            print(
                f"[mcmc] iter {it + 1}/{total_iters} "
                f"score={cur_score:.2f} prior={cur_prior:.2f}"
            )

    # Keep only the first n_samples (the tail may briefly exceed due to edge
    # rounding: e.g. if total_iters % thin != 0).
    if len(samples) > n_samples:
        samples = samples[:n_samples]
        log_post_kept = log_post_kept[:n_samples]

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
    progress: bool = False,
    perturb_flips: int = 5,
    hybrid_prob: float = 0.1,
    resample_flip: float = 0.3,
    resample_flip_burn: Optional[float] = None,
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
    progress : bool
        If True, occasional progress prints during each chain.
    perturb_flips : int
        Target number of random edge flips (kept acyclic) applied to
        ``start_adj`` to form each chain's initial graph.  The first
        chain starts from an exact copy of ``start_adj``.
    hybrid_prob : float, default 0.1
        Per-iteration probability of proposing a random parent-set
        resample (Madigan & York 1995 Section 4) instead of a single-
        edge move.  ``0.0`` disables the hybrid move.
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

    Returns
    -------
    MCMCResult
    """
    X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    n, p = X.shape
    if len(node_types) != p:
        raise ValueError(f"node_types length {len(node_types)} != p={p}")
    start_adj = np.asarray(start_adj, dtype=np.int64)
    if start_adj.shape != (p, p):
        raise ValueError(f"start_adj has shape {start_adj.shape}, expected ({p}, {p})")
    if not _is_dag(start_adj):
        raise ValueError("start_adj is not acyclic")
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

    all_chain_samples: List[np.ndarray] = []
    all_log_post: List[np.ndarray] = []
    prop_totals = {"add": 0, "remove": 0, "reverse": 0, "hybrid": 0}
    accept_totals = {"add": 0, "remove": 0, "reverse": 0, "hybrid": 0}
    mean_lp_per_chain: List[float] = []

    for c in range(n_chains):
        if c == 0:
            start_c = start_adj.copy()
        else:
            start_c = _perturb_dag(start_adj, perturb_flips, child_rngs[c])
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
            progress=progress,
            hyper=hyper,
            hybrid_prob=hybrid_prob,
            resample_flip=resample_flip,
            resample_flip_burn=resample_flip_burn,
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

    # R-hat and ESS diagnostics.
    # We compute R-hat both on the directed-edge indicators and on the
    # undirected skeleton (adj | adj.T).  The skeleton R-hat is the
    # primary convergence diagnostic reported as ``max_rhat``, because it
    # is invariant under Markov equivalence: two DAGs with the same
    # CPDAG differ on directed-edge indicators (which inflates the
    # directed R-hat) but agree on the skeleton.  When the skeleton has
    # converged but the directed edges have not, the chain has found
    # the right Markov class even if it has not resolved all edge
    # orientations, which is a well-known fundamental non-identifiability
    # of DAG inference from observational data.
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
        max_rhat_skel = float(np.nanmax(rhat_skel[triu])) if triu.any() else 1.0
        off = ~np.eye(p, dtype=bool)
        max_rhat_dir = float(np.nanmax(rhat_mat[off])) if off.any() else 1.0
        max_rhat = max_rhat_skel
    else:
        rhat_mat = np.ones((p, p), dtype=float)
        rhat_skel = np.ones((p, p), dtype=float)
        max_rhat = 1.0
        max_rhat_dir = 1.0
        max_rhat_skel = 1.0

    # Per-edge ESS (across all chains concatenated).
    if flat.shape[0] >= 4:
        ess_mat = np.zeros((p, p), dtype=float)
        for i in range(p):
            for j in range(p):
                if i == j:
                    ess_mat[i, j] = float(flat.shape[0])
                    continue
                ess_mat[i, j] = _ess_ips(flat[:, i, j].astype(float))
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
        "ess_per_edge": ess_mat,
        "min_ess": min_ess,
        "n_chains": int(n_chains),
        "n_samples_per_chain": [int(a.shape[0]) for a in all_chain_samples],
        "burn_in": int(burn_in),
        "thin": int(thin),
    }

    return MCMCResult(
        samples=flat_samples_list,
        log_post=log_post_concat,
        edge_probs=edge_probs,
        diagnostics=diagnostics,
    )


__all__ = ["MCMCResult", "run_structure_mcmc"]
