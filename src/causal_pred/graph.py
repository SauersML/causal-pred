"""Canonical graph utilities for the causal-pred pipeline.

Single source of truth for the small set of DAG primitives that are
needed by MrDAG, DAGSLAM, and structure MCMC.  All callers operate on
dense ``(p, p)`` adjacency matrices where ``adj[i, j] == 1`` denotes the
directed edge ``i -> j`` and ``adj[i, j] == 0`` denotes its absence.

A "move" is the tuple ``(kind, i, j)`` with ``kind`` in ``"add"``,
``"delete"``, or ``"reverse"``.  ``"delete"`` is the canonical name; the
DAGSLAM module historically used ``"remove"`` and accepts that spelling
through its own thin wrapper to avoid touching the score-cache plumbing.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


Move = Tuple[str, int, int]


# ---------------------------------------------------------------------------
# Reachability and acyclicity
# ---------------------------------------------------------------------------


def is_reachable(adj: np.ndarray, src: int, dst: int) -> bool:
    """True iff there is a directed path ``src -> ... -> dst`` in ``adj``.

    ``src == dst`` returns True (self-reachable).  Implemented as an
    iterative DFS over the dense adjacency matrix.
    """
    if src == dst:
        return True
    p = adj.shape[0]
    visited = np.zeros(p, dtype=bool)
    visited[src] = True
    stack: List[int] = [int(src)]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            vv = int(v)
            if vv == dst:
                return True
            if not visited[vv]:
                visited[vv] = True
                stack.append(vv)
    return False


def would_create_cycle(adj: np.ndarray, parent: int, child: int) -> bool:
    """True iff adding ``parent -> child`` to ``adj`` would create a cycle.

    A self-loop (``parent == child``) is treated as a cycle.  Otherwise,
    the addition cycles iff ``child`` can already reach ``parent``.
    """
    if parent == child:
        return True
    return is_reachable(adj, child, parent)


def is_dag(adj: np.ndarray) -> bool:
    """Kahn's-algorithm DAG check; True iff ``adj`` is acyclic."""
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
# Move application / reversal
# ---------------------------------------------------------------------------


def apply_move(adj: np.ndarray, move: Move) -> None:
    """Apply ``move`` in place to ``adj``.

    Move kinds:
      * ``"add"``:     ``adj[i, j] = 1``
      * ``"delete"``:  ``adj[i, j] = 0``
      * ``"reverse"``: flip ``i -> j`` to ``j -> i``
    """
    kind, i, j = move
    if kind == "add":
        adj[i, j] = 1
    elif kind == "delete":
        adj[i, j] = 0
    elif kind == "reverse":
        adj[i, j] = 0
        adj[j, i] = 1
    else:
        raise ValueError(f"unknown move kind {kind!r}")


def revert_move(adj: np.ndarray, move: Move) -> None:
    """Undo ``move`` in place on ``adj``.  Inverse of :func:`apply_move`."""
    kind, i, j = move
    if kind == "add":
        adj[i, j] = 0
    elif kind == "delete":
        adj[i, j] = 1
    elif kind == "reverse":
        adj[j, i] = 0
        adj[i, j] = 1
    else:
        raise ValueError(f"unknown move kind {kind!r}")


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def parents_of(adj: np.ndarray, child: int) -> np.ndarray:
    """Return the indices of nodes with a directed edge into ``child``."""
    return np.flatnonzero(adj[:, child])


def edge_list(adj: np.ndarray, names: Sequence[str]) -> List[Tuple[str, str]]:
    """Return ``[(parent_name, child_name), ...]`` for every present edge."""
    out: List[Tuple[str, str]] = []
    for i, parent in enumerate(names):
        for j, child in enumerate(names):
            if int(adj[i, j]) == 1:
                out.append((str(parent), str(child)))
    return out


__all__ = [
    "Move",
    "is_reachable",
    "would_create_cycle",
    "is_dag",
    "apply_move",
    "revert_move",
    "parents_of",
    "edge_list",
]
