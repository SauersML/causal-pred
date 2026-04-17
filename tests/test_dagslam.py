"""Tests for the DAGSLAM hill-climbing structure search.

Covers:
  1. output is a strict DAG with {0,1} entries and zero diagonal;
  2. DAGSLAM closes >=90% of the log-score gap between the empty DAG
     and the ground-truth DAG on ``medium_data``;
  3. DAGSLAM recovers >=70% of the CANONICAL_EDGES (up to Markov
     equivalence -- reversals that live in the same v-structure-free
     skeleton component as the canonical edge count as recovered);
  4. multi-restart (restarts=5) is no worse than single-restart
     (restarts=1);
  5. max_parents is respected;
  6. runtime budget: n=3000, p=18, restarts=3, max_iter=500 under 2 min;
  7. the returned score_cache can re-score a perturbed DAG in <100 ms
     (evidence that cache hits dominate).
"""

from __future__ import annotations

import time

import numpy as np

from causal_pred.data.nodes import (
    CANONICAL_EDGES,
    NODE_INDEX,
)
from causal_pred.dagslam.search import (
    DAGSLAMResult,
    _is_dag,
    run_dagslam,
)
from causal_pred.scoring.mixed import score_dag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary_dag(adj: np.ndarray) -> bool:
    """Strict DAG-ness: entries in {0,1}, zero diagonal, acyclic."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return False
    if not np.all((adj == 0) | (adj == 1)):
        return False
    if np.any(np.diag(adj) != 0):
        return False
    return _is_dag(adj)


def _skeleton_edges(adj: np.ndarray):
    """Unordered set of skeleton edges (undirected)."""
    edges = set()
    for i, j in np.argwhere(adj == 1):
        i = int(i)
        j = int(j)
        edges.add((min(i, j), max(i, j)))
    return edges


# ---------------------------------------------------------------------------
# 1. Output is a DAG
# ---------------------------------------------------------------------------


def test_output_is_dag(small_data):
    rng = np.random.default_rng(0)
    res = run_dagslam(
        small_data.X,
        small_data.node_types,
        max_parents=5,
        max_iter=300,
        restarts=2,
        rng=rng,
    )
    assert isinstance(res, DAGSLAMResult)
    A = res.adjacency
    assert _is_binary_dag(A), "returned adjacency is not a valid 0/1 DAG"
    assert A.dtype.kind in ("i", "u"), (
        f"adjacency dtype should be integer, got {A.dtype}"
    )
    # Cross-check with networkx if available.
    try:
        import networkx as nx

        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(G)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 2. Beats empty: closes >=90% of the empty->truth gap.
#
# Interpretation on the log-scale: we define the relative closure
#
#     frac = (s_dagslam - s_empty) / (s_truth - s_empty)
#
# and require frac >= 0.9.  This is a well-defined ratio of the
# log-marginal-likelihood improvements achievable by adding structure
# to the empty graph; frac = 1 means DAGSLAM recovered the truth score
# exactly, frac > 1 means it found a *better*-scoring graph than the
# ground truth (which can happen because ground truth is only one of
# several Markov-equivalent DAGs under the score).
# ---------------------------------------------------------------------------


def test_beats_empty_on_synthetic_data(medium_data):
    rng = np.random.default_rng(123)
    X = medium_data.X
    nt = medium_data.node_types
    adj_truth = medium_data.ground_truth_adj
    adj_empty = np.zeros_like(adj_truth)

    s_truth = float(score_dag(adj_truth, X, nt))
    s_empty = float(score_dag(adj_empty, X, nt))

    t0 = time.perf_counter()
    res = run_dagslam(
        X,
        nt,
        max_parents=6,
        max_iter=500,
        restarts=3,
        rng=rng,
    )
    elapsed = time.perf_counter() - t0

    assert np.isfinite(res.log_score)
    assert s_truth > s_empty + 100.0, (
        f"precondition failed: truth {s_truth:.1f} not meaningfully above "
        f"empty {s_empty:.1f}"
    )
    gap = s_truth - s_empty
    frac = (res.log_score - s_empty) / gap
    assert frac >= 0.9, (
        f"DAGSLAM closed only {frac * 100:.1f}% of the empty->truth gap "
        f"(dagslam={res.log_score:.1f} empty={s_empty:.1f} truth={s_truth:.1f}) "
        f"after {elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# 3. Canonical-edge recovery >=70% (up to Markov equivalence).
#
# Rationale: a score-equivalent (BGe) or Laplace-over-logistic
# approximation is roughly score-equivalent on Markov-equivalent DAGs;
# a directed edge can legitimately appear reversed in the learned DAG
# if its v-structure pattern is unobservable.  We count an edge as
# recovered if EITHER direction is present; that convention is
# documented here.
# ---------------------------------------------------------------------------


def test_recovers_majority_of_canonical_edges(medium_data):
    rng = np.random.default_rng(7)
    X = medium_data.X
    nt = medium_data.node_types
    res = run_dagslam(
        X,
        nt,
        max_parents=6,
        max_iter=500,
        restarts=3,
        rng=rng,
    )
    A = res.adjacency
    hits = 0
    missed = []
    reversed_hits = []
    for p_name, c_name in CANONICAL_EDGES:
        i = NODE_INDEX[p_name]
        j = NODE_INDEX[c_name]
        if A[i, j] == 1:
            hits += 1
        elif A[j, i] == 1:
            hits += 1
            reversed_hits.append((p_name, c_name))
        else:
            missed.append((p_name, c_name))
    frac = hits / len(CANONICAL_EDGES)
    assert frac >= 0.7, (
        f"only {hits}/{len(CANONICAL_EDGES)} canonical edges recovered "
        f"({frac * 100:.0f}%). missed={missed} reversed={reversed_hits}"
    )


# ---------------------------------------------------------------------------
# 4. Multi-restart log-score >= single-restart log-score.
# ---------------------------------------------------------------------------


def test_restart_improves_over_single(small_data):
    rng1 = np.random.default_rng(2024)
    rng5 = np.random.default_rng(2024)
    r1 = run_dagslam(
        small_data.X,
        small_data.node_types,
        max_parents=5,
        max_iter=300,
        restarts=1,
        rng=rng1,
    )
    r5 = run_dagslam(
        small_data.X,
        small_data.node_types,
        max_parents=5,
        max_iter=300,
        restarts=5,
        rng=rng5,
    )
    assert r5.log_score >= r1.log_score - 1e-6, (
        f"multi-restart score {r5.log_score:.3f} worse than "
        f"single-restart {r1.log_score:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. max_parents is respected.
# ---------------------------------------------------------------------------


def test_respects_max_parents(small_data):
    rng = np.random.default_rng(5)
    res = run_dagslam(
        small_data.X,
        small_data.node_types,
        max_parents=2,
        max_iter=200,
        restarts=2,
        rng=rng,
    )
    in_degree = res.adjacency.sum(axis=0)
    assert in_degree.max() <= 2, f"found node with {in_degree.max()} parents (cap=2)"


# ---------------------------------------------------------------------------
# 6. Runtime budget.
# ---------------------------------------------------------------------------


def test_runtime_budget(medium_data):
    rng = np.random.default_rng(9)
    t0 = time.perf_counter()
    run_dagslam(
        medium_data.X,
        medium_data.node_types,
        max_parents=6,
        max_iter=500,
        restarts=3,
        rng=rng,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 120.0, f"DAGSLAM too slow: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# 7. Cache usable downstream.
# ---------------------------------------------------------------------------


def test_score_cache_is_usable_downstream(small_data):
    rng = np.random.default_rng(11)
    res = run_dagslam(
        small_data.X,
        small_data.node_types,
        max_parents=5,
        max_iter=200,
        restarts=2,
        rng=rng,
    )
    cache = res.score_cache
    assert isinstance(cache, dict) and len(cache) > 0

    # Perturb: remove a random present edge, add a random legal edge.
    A = res.adjacency.copy()
    present = np.argwhere(A == 1)
    if present.size:
        idx = int(rng.integers(0, present.shape[0]))
        i, j = int(present[idx, 0]), int(present[idx, 1])
        A[i, j] = 0
    # (A may still be a DAG either way.)

    t0 = time.perf_counter()
    s = score_dag(A, small_data.X, small_data.node_types, cache=cache)
    elapsed = time.perf_counter() - t0
    assert np.isfinite(s)
    assert elapsed < 0.1, (
        f"cache-warmed rescore too slow: {elapsed * 1e3:.2f}ms "
        f"(cache has {len(cache)} entries)"
    )
