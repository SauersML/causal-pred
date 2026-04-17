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
                **hyper) -> DAGSLAMResult

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
# Acyclicity utilities (DFS-based, exact)
# ---------------------------------------------------------------------------


def _is_reachable(adj: np.ndarray, src: int, dst: int) -> bool:
    """Return True iff there is a directed path ``src -> ... -> dst`` in
    ``adj``.  Exact DFS.  Self-reachability (``src == dst``) returns True.
    """
    if src == dst:
        return True
    p = adj.shape[0]
    visited = np.zeros(p, dtype=bool)
    visited[src] = True
    stack = [int(src)]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            v = int(v)
            if v == dst:
                return True
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return False


def _creates_cycle_if_add(adj: np.ndarray, i: int, j: int) -> bool:
    """Adding i -> j creates a cycle iff j can already reach i."""
    if i == j:
        return True
    return _is_reachable(adj, j, i)


def _is_dag(adj: np.ndarray) -> bool:
    """Kahn's algorithm; True iff the graph is acyclic."""
    in_deg = adj.sum(axis=0).astype(int).copy()
    stack = [i for i in range(adj.shape[0]) if in_deg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in np.flatnonzero(adj[u]):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                stack.append(int(v))
    return seen == adj.shape[0]


# ---------------------------------------------------------------------------
# Random sparse acyclic start
# ---------------------------------------------------------------------------


def _random_sparse_dag(p: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a random DAG by fixing a random topological order and
    including each forward edge independently with probability ``density``.
    Acyclic by construction.
    """
    order = rng.permutation(p)
    adj = np.zeros((p, p), dtype=np.int64)
    for a_idx in range(p):
        for b_idx in range(a_idx + 1, p):
            if rng.random() < density:
                u = int(order[a_idx])
                v = int(order[b_idx])
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
        if parent_counts[i] >= max_parents:
            continue
        adj[i, j] = 0
        reaches = _is_reachable(adj, i, j)
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
            if not _creates_cycle_if_add(adj, i, j):
                moves.append(("add", i, j))

    return moves


def _delta_of_move(
    move: Tuple[str, int, int],
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: dict,
    hyper: dict,
) -> float:
    kind, i, j = move
    if kind == "add":
        return score_delta_add_edge(i, j, adj, data, node_types, cache=cache, **hyper)
    if kind == "remove":
        return score_delta_remove_edge(
            i, j, adj, data, node_types, cache=cache, **hyper
        )
    if kind == "reverse":
        return score_delta_reverse_edge(
            i, j, adj, data, node_types, cache=cache, **hyper
        )
    raise ValueError(f"unknown move kind {kind!r}")


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
) -> Tuple[np.ndarray, float, dict]:
    """One hill-climb from ``start_adj``.  Mutates ``cache``.

    Returns (best_adj_of_restart, best_score_of_restart, trace_dict).
    """
    p = data.shape[1]
    adj = start_adj.astype(np.int64, copy=True)
    # Anchor the running score once; then maintain it via delta updates.
    cur_score = float(score_dag(adj, data, node_types, cache=cache, **hyper))

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
        moves = _enumerate_moves(adj, max_parents)
        if not moves:
            if verbose:
                print(f"[dagslam] iter {it}: no structurally legal move.")
            break

        tabu_set = {_reverse_move(m) for m in tabu}

        non_tabu_best: Optional[Tuple[float, Tuple[str, int, int], int]] = None
        tabu_best: Optional[Tuple[float, Tuple[str, int, int], int]] = None

        for m in moves:
            try:
                d = _delta_of_move(m, adj, data, node_types, cache, hyper)
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
    **hyper
        Extra hyperparameters forwarded to the scoring module (e.g.
        ``alpha_mu`` for BGe, ``tau2`` for the Laplace logistic).

    Returns
    -------
    DAGSLAMResult
    """
    X = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    n, p = X.shape
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
            start_adj = _random_sparse_dag(p, density=0.1, rng=rng)
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
        )

        info["restart"] = r
        info["start_n_edges"] = int(start_adj.sum())
        info["final_n_edges"] = int(best_adj_r.sum())
        trace.append(info)

        if best_score_r > global_best_score:
            global_best_score = best_score_r
            global_best_adj = best_adj_r.copy()

    # Strictly acyclic invariant (MCMC depends on it).
    assert _is_dag(global_best_adj), (
        "DAGSLAM produced a non-acyclic graph -- internal invariant violated"
    )

    # Final authoritative rescore.
    final_score = float(score_dag(global_best_adj, X, node_types, cache=cache, **hyper))
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
