"""DAGSLAM hill-climbing structure search.

Implements the tabu-augmented greedy hill-climber of Zhao & Jia
(*DAGSLAM*, BMC Med Res Methodol 2025) on top of the mixed-variable
marginal-likelihood score in :mod:`causal_pred.scoring.mixed`.

Algorithm
---------

1. Start from the empty DAG (optionally from a random sparse DAG for
   restarts > 1).  Random starts draw every forward edge in a random
   topological order with probability 0.1, which guarantees acyclicity
   by construction.

2. At every iteration we enumerate candidate moves:
     * edge addition (i, j) for every ordered pair where j does not
       already have ``max_parents`` parents and adding i -> j does not
       create a cycle;
     * edge deletion of any currently-present edge;
     * edge reversal of any currently-present edge whose flip keeps the
       DAG acyclic AND keeps both endpoints within their parent-count
       cap.
   The delta log score for every candidate is computed through the
   cached ``score_delta_*`` functions, so repeat evaluations are O(1).

3. Apply the best-improving NON-tabu move.  Ties are broken in favour
   of the move that leaves fewer edges (preferring sparsity).

4. A tabu list of the last ``tabu_tenure`` applied moves (default 10)
   forbids any move whose application would reverse a recently-applied
   one (adding a just-deleted edge, deleting a just-added edge, or
   re-reversing a just-reversed edge), UNLESS the candidate's delta is
   more than 1.0 log-units better than the best tabu-approved
   alternative -- the Glover (1989) aspiration criterion.

5. Stopping: no improving non-tabu move for ``2 * n_nodes`` consecutive
   steps, or ``max_iter`` iterations reached.

6. Multiple restarts are run from different starting graphs with a
   shared ``cache`` (dramatic speedup: the same local parent-set
   scores recur).  The global best DAG across restarts is returned;
   every restart's trajectory is in ``result.trace``.

Public API mirrors the docstring in :mod:`causal_pred.dagslam.__init__`:

    run_dagslam(data, node_types, max_parents=6, max_iter=500,
                restarts=5, tabu_tenure=10, rng=None, verbose=False,
                pi_prior=None, **hyper) -> DAGSLAMResult

References
----------
* Zhao, Z. & Jia, W. (2025), "DAGSLAM ...", *BMC Medical Research
  Methodology*.  (Greedy hill-climbing with tabu list + random
  restarts for structure learning on mixed-type biobank data.)
* Glover, F. (1989), "Tabu Search -- Part I", *ORSA J. Comput.* 1(3),
  190-206.  (Aspiration criterion.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..graph import is_dag, is_reachable, would_create_cycle
from ..scoring.mixed import (
    score_dag,
    score_delta_add_edge,
    score_delta_remove_edge,
    score_delta_reverse_edge,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DAGSLAMResult:
    adjacency: np.ndarray  # (p, p) 0/1 DAG
    log_score: float  # score_dag(adjacency)
    trace: List[dict] = field(default_factory=list)
    score_cache: dict = field(default_factory=dict)
    n_edges: int = 0


# ---------------------------------------------------------------------------
# Edge prior utilities
# ---------------------------------------------------------------------------


def _prepare_edge_logit_prior(
    pi_prior: Optional[np.ndarray],
    p: int,
) -> np.ndarray:
    """Return per-edge log-odds for the optional MrDAG prior.

    Non-finite entries mean "no MrDAG evidence" and map to Bernoulli(0.5),
    which has zero log-odds.  The diagonal is forced to zero because
    self-edges are structurally forbidden.
    """
    if pi_prior is None:
        edge_logit = np.zeros((p, p), dtype=float)
    else:
        pi = np.asarray(pi_prior, dtype=float).copy()
        if pi.shape != (p, p):
            raise ValueError(f"pi_prior has shape {pi.shape}, expected ({p}, {p})")
        pi = np.where(np.isfinite(pi), pi, 0.5)
        np.fill_diagonal(pi, 0.5)
        pi = np.clip(pi, 1e-12, 1.0 - 1e-12)
        edge_logit = np.log(pi) - np.log1p(-pi)

    np.fill_diagonal(edge_logit, 0.0)
    return edge_logit


def _relative_log_prior(adj: np.ndarray, edge_logit: np.ndarray) -> float:
    """Graph prior score up to a graph-independent Bernoulli constant."""
    return float(np.sum(adj * edge_logit))


def _delta_prior_of_move(
    move: Tuple[str, int, int],
    edge_logit: np.ndarray,
) -> float:
    """Relative log-prior delta for one add, remove, or reverse move."""
    kind, i, j = move
    if kind == "add":
        return float(edge_logit[i, j])
    if kind == "remove":
        return float(-edge_logit[i, j])
    if kind == "reverse":
        return float(edge_logit[j, i] - edge_logit[i, j])
    raise ValueError(f"unknown move kind {kind!r}")


# ---------------------------------------------------------------------------
# Random sparse acyclic start
# ---------------------------------------------------------------------------


def _prepare_allowed_edges(allowed_edges: Optional[np.ndarray], p: int) -> np.ndarray:
    """Return a boolean structural mask for candidate directed edges."""
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


def _random_sparse_dag(
    p: int,
    density: float,
    rng: np.random.Generator,
    allowed_edges: np.ndarray,
) -> np.ndarray:
    """Draw a random DAG by fixing a random topological order and
    including each forward edge independently with probability ``density``.
    Acyclic by construction.
    """
    order = rng.permutation(p)
    adj = np.zeros((p, p), dtype=np.int64)
    for a_idx in range(p):
        for b_idx in range(a_idx + 1, p):
            u = int(order[a_idx])
            v = int(order[b_idx])
            if allowed_edges[u, v] and rng.random() < density:
                adj[u, v] = 1
    return adj


# ---------------------------------------------------------------------------
# Moves and tabu list
# ---------------------------------------------------------------------------

# A move is encoded as a tuple (kind, i, j):
#   ("add",     i, j)   apply: adj[i, j] = 1
#   ("remove",  i, j)   apply: adj[i, j] = 0
#   ("reverse", i, j)   apply: flip i->j to j->i


def _reverse_move(move: Tuple[str, int, int]) -> Tuple[str, int, int]:
    """The move that undoes ``move``.  Used to test tabu membership."""
    kind, i, j = move
    if kind == "add":
        return ("remove", i, j)
    if kind == "remove":
        return ("add", i, j)
    if kind == "reverse":
        # Undoing a reversal of i->j (which produced j->i) is itself a
        # reversal of j->i back to i->j.
        return ("reverse", j, i)
    raise ValueError(f"unknown move kind {kind!r}")


def _enumerate_moves(
    adj: np.ndarray,
    max_parents: int,
    allowed_edges: np.ndarray,
) -> List[Tuple[str, int, int]]:
    """Structurally-legal moves from ``adj``.

    Structurally legal means: no self-loops; acyclic; parent-count cap
    respected at both endpoints (for reversals, the new parent of the
    flipped edge also gets checked).

    Tabu filtering is applied separately at the selection stage so that
    the aspiration criterion can be evaluated on the full candidate set.
    """
    p = adj.shape[0]
    parent_counts = adj.sum(axis=0)
    moves: List[Tuple[str, int, int]] = []

    # Deletions.
    for i, j in np.argwhere(adj == 1):
        moves.append(("remove", int(i), int(j)))

    # Reversals.  Flip i->j to j->i; acyclicity test is performed on the
    # adj with i->j removed so the *new* path must not include the old
    # edge.
    for i, j in np.argwhere(adj == 1):
        i = int(i)
        j = int(j)
        if not allowed_edges[j, i]:
            continue
        if parent_counts[i] >= max_parents:
            continue
        adj[i, j] = 0
        reaches = is_reachable(adj, i, j)
        adj[i, j] = 1
        if not reaches:
            moves.append(("reverse", i, j))

    # Additions.
    for j in range(p):
        if parent_counts[j] >= max_parents:
            continue
        for i in range(p):
            if i == j:
                continue
            if adj[i, j] or adj[j, i]:
                # Skip if the edge already exists (handled above), or the
                # reverse edge does (handled via "reverse").
                continue
            if not allowed_edges[i, j]:
                continue
            if not would_create_cycle(adj, i, j):
                moves.append(("add", i, j))

    return moves


def _delta_of_move(
    move: Tuple[str, int, int],
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: dict,
    hyper: dict,
    edge_logit: np.ndarray,
) -> float:
    kind, i, j = move
    if kind == "add":
        d_score = score_delta_add_edge(
            i, j, adj, data, node_types, cache=cache, **hyper
        )
    elif kind == "remove":
        d_score = score_delta_remove_edge(
            i, j, adj, data, node_types, cache=cache, **hyper
        )
    elif kind == "reverse":
        d_score = score_delta_reverse_edge(
            i, j, adj, data, node_types, cache=cache, **hyper
        )
    else:
        raise ValueError(f"unknown move kind {kind!r}")
    return float(d_score + _delta_prior_of_move(move, edge_logit))


def _apply_move(adj: np.ndarray, move: Tuple[str, int, int]) -> None:
    kind, i, j = move
    if kind == "add":
        adj[i, j] = 1
    elif kind == "remove":
        adj[i, j] = 0
    elif kind == "reverse":
        adj[i, j] = 0
        adj[j, i] = 1
    else:
        raise ValueError(f"unknown move kind {kind!r}")


# ---------------------------------------------------------------------------
# Single hill-climb
# ---------------------------------------------------------------------------


def _single_climb(
    data: np.ndarray,
    node_types: Sequence[str],
    start_adj: np.ndarray,
    max_parents: int,
    max_iter: int,
    tabu_tenure: int,
    cache: dict,
    hyper: dict,
    verbose: bool,
    edge_logit: np.ndarray,
    allowed_edges: np.ndarray,
) -> Tuple[np.ndarray, float, dict]:
    """One hill-climb from ``start_adj``.  Mutates ``cache``.

    Returns (best_adj_of_restart, best_score_of_restart, trace_dict).
    """
    p = data.shape[1]
    adj = start_adj.astype(np.int64, copy=True)
    # Anchor the running score once; then maintain it via delta updates.
    cur_score = float(
        score_dag(adj, data, node_types, cache=cache, **hyper)
        + _relative_log_prior(adj, edge_logit)
    )

    best_adj = adj.copy()
    best_score = cur_score
    tabu: List[Tuple[str, int, int]] = []
    stagnation = 0
    stagnation_limit = 2 * p
    path_scores: List[float] = [cur_score]
    skipped_moves: List[dict] = []
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        moves = _enumerate_moves(adj, max_parents, allowed_edges)
        if not moves:
            if verbose:
                print(f"[dagslam] iter {it}: no structurally legal move.")
            break

        tabu_set = {_reverse_move(m) for m in tabu}

        non_tabu_best: Optional[Tuple[float, Tuple[str, int, int], int]] = None
        tabu_best: Optional[Tuple[float, Tuple[str, int, int], int]] = None

        for m in moves:
            try:
                d = _delta_of_move(m, adj, data, node_types, cache, hyper, edge_logit)
            except Exception as exc:  # defensive -- never propagate
                skipped_moves.append(
                    {
                        "iter": it,
                        "move": m,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not np.isfinite(d):
                skipped_moves.append(
                    {
                        "iter": it,
                        "move": m,
                        "reason": f"non-finite delta {d}",
                    }
                )
                continue

            if m[0] == "add":
                n_edge_delta = 1
            elif m[0] == "remove":
                n_edge_delta = -1
            else:
                n_edge_delta = 0

            candidate = (d, m, n_edge_delta)

            is_tabu = m in tabu_set
            if is_tabu:
                if (
                    tabu_best is None
                    or d > tabu_best[0]
                    or (d == tabu_best[0] and n_edge_delta < tabu_best[2])
                ):
                    tabu_best = candidate
            else:
                if (
                    non_tabu_best is None
                    or d > non_tabu_best[0]
                    or (d == non_tabu_best[0] and n_edge_delta < non_tabu_best[2])
                ):
                    non_tabu_best = candidate

        # Selection with aspiration.
        chosen: Optional[Tuple[float, Tuple[str, int, int], int]] = None
        if non_tabu_best is not None and non_tabu_best[0] > 0:
            chosen = non_tabu_best
        if tabu_best is not None:
            nt_val = non_tabu_best[0] if non_tabu_best is not None else -np.inf
            if tabu_best[0] > 0 and tabu_best[0] > nt_val + 1.0:
                chosen = tabu_best

        if chosen is None:
            stagnation += 1
            path_scores.append(cur_score)
            if stagnation >= stagnation_limit:
                if verbose:
                    print(f"[dagslam] iter {it}: stagnated; stop.")
                break
            continue

        d, move, _ = chosen
        _apply_move(adj, move)
        cur_score += d
        path_scores.append(cur_score)
        stagnation = 0

        tabu.append(move)
        if len(tabu) > tabu_tenure:
            tabu.pop(0)

        if cur_score > best_score + 1e-12:
            best_score = cur_score
            best_adj = adj.copy()

        if verbose:
            print(
                f"[dagslam] iter {it}: {move} d={d:+.4f} "
                f"score={cur_score:.4f} best={best_score:.4f}"
            )

    trace = {
        "best_score": float(best_score),
        "final_score": float(cur_score),
        "n_iter": int(n_iter),
        "path_scores": np.asarray(path_scores, dtype=np.float64),
        "skipped_moves": skipped_moves,
    }
    return best_adj, float(best_score), trace


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_dagslam(
    data,
    node_types: Sequence[str],
    max_parents: int = 6,
    max_iter: int = 500,
    restarts: int = 5,
    tabu_tenure: int = 10,
    rng=None,
    verbose: bool = False,
    pi_prior: Optional[np.ndarray] = None,
    allowed_edges: Optional[np.ndarray] = None,
    **hyper,
) -> DAGSLAMResult:
    """DAGSLAM hill-climbing structure search.

    Parameters
    ----------
    data : array-like, shape (n, p)
        Design matrix in the ordering of ``node_types``.  Cast to float64.
    node_types : sequence of str, length p
        ``"continuous"`` / ``"binary"`` / ``"survival"`` per column.
    max_parents : int, default 6
        Maximum indegree allowed for any node.
    max_iter : int, default 500
        Hard cap on iterations per restart.
    restarts : int, default 5
        Number of independent hill-climbs.  The first starts from the
        empty DAG; subsequent restarts start from random sparse DAGs
        drawn with the supplied RNG (density 0.1).
    tabu_tenure : int, default 10
        Length of the tabu list.
    rng : numpy.random.Generator, optional
        If ``None``, ``np.random.default_rng()`` is used.
    verbose : bool
        If True, prints progress.
    pi_prior : array-like, shape (p, p), optional
        Per-edge MrDAG inclusion probabilities.  Finite entries add an
        independent Bernoulli log-odds prior to the greedy search objective;
        non-finite entries are treated as neutral 0.5 probabilities.
    allowed_edges : array-like, shape (p, p), optional
        Boolean structural mask. False entries are never proposed as
        additions or reversals.
    **hyper
        Extra hyperparameters forwarded to the scoring module (e.g.
        ``alpha_mu`` for BGe, ``tau2`` for the Laplace logistic).

    Returns
    -------
    DAGSLAMResult
    """
    X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    n, p = X.shape
    edge_logit = _prepare_edge_logit_prior(pi_prior, p)
    allowed = _prepare_allowed_edges(allowed_edges, p)
    if len(node_types) != p:
        raise ValueError(
            f"node_types has length {len(node_types)} but data has {p} columns"
        )
    if rng is None:
        rng = np.random.default_rng()

    cache: dict = {}
    trace: List[dict] = []

    global_best_adj = np.zeros((p, p), dtype=np.int64)
    global_best_score = -np.inf

    n_restarts = max(1, int(restarts))
    for r in range(n_restarts):
        if r == 0:
            start_adj = np.zeros((p, p), dtype=np.int64)
        else:
            start_adj = _random_sparse_dag(
                p,
                density=0.1,
                rng=rng,
                allowed_edges=allowed,
            )
            # Respect max_parents in the starting state by randomly
            # dropping excess incoming edges.
            for j in range(p):
                incoming = np.flatnonzero(start_adj[:, j])
                if len(incoming) > max_parents:
                    drop = rng.choice(
                        incoming,
                        size=len(incoming) - max_parents,
                        replace=False,
                    )
                    start_adj[drop, j] = 0

        best_adj_r, best_score_r, info = _single_climb(
            data=X,
            node_types=node_types,
            start_adj=start_adj,
            max_parents=max_parents,
            max_iter=max_iter,
            tabu_tenure=tabu_tenure,
            cache=cache,
            hyper=hyper,
            verbose=verbose,
            edge_logit=edge_logit,
            allowed_edges=allowed,
        )

        info["restart"] = r
        info["start_n_edges"] = int(start_adj.sum())
        info["final_n_edges"] = int(best_adj_r.sum())
        trace.append(info)

        if best_score_r > global_best_score:
            global_best_score = best_score_r
            global_best_adj = best_adj_r.copy()

    # Strictly acyclic invariant (MCMC depends on it).
    assert is_dag(global_best_adj), (
        "DAGSLAM produced a non-acyclic graph -- internal invariant violated"
    )

    # Final authoritative rescore.
    final_score = float(
        score_dag(global_best_adj, X, node_types, cache=cache, **hyper)
        + _relative_log_prior(global_best_adj, edge_logit)
    )
    if not np.isfinite(final_score):
        raise RuntimeError("DAGSLAM produced a non-finite final score")
    assert abs(final_score - global_best_score) < 1e-6, (
        f"tracked best_score {global_best_score:.6f} disagrees with "
        f"re-scored {final_score:.6f} (delta "
        f"{final_score - global_best_score:.2e})"
    )

    return DAGSLAMResult(
        adjacency=global_best_adj.astype(np.int64, copy=False),
        log_score=final_score,
        trace=trace,
        score_cache=cache,
        n_edges=int(global_best_adj.sum()),
    )


__all__ = ["DAGSLAMResult", "run_dagslam"]
