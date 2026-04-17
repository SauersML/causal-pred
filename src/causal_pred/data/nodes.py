"""Canonical node set for the T2D causal model.

This module is the single source of truth for:
  * the node names,
  * each node's type (continuous / binary / survival),
  * the ground-truth causal structure used by the synthetic data generator
    and by the validation framework's ``known-edges`` check.

Every other component (MrDAG, DAGSLAM, MCMC, GAM, validation) imports
``NODES``, ``NODE_TYPES``, and ``KNOWN_EDGES`` from here so that node indices
are consistent across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Node:
    name: str
    kind: str  # "continuous" | "binary" | "survival"
    description: str


# Ordered so that any topological sort of the synthetic DAG will respect it.
# Genetic/demographic roots first, then mediators, then the T2D outcome.
NODES: Tuple[Node, ...] = (
    # --- genetic roots (continuous, standardised polygenic scores) ---
    Node("PGS_T2D", "continuous", "Polygenic score for T2D"),
    Node("PGS_BMI", "continuous", "Polygenic score for body-mass index"),
    Node("PGS_LDL", "continuous", "Polygenic score for LDL cholesterol"),
    Node("PGS_HbA1c", "continuous", "Polygenic score for HbA1c"),
    # --- demographic roots ---
    Node("age", "continuous", "Current age (years)"),
    Node("sex", "binary", "1 = male, 0 = female"),
    Node("ancestry_PC1", "continuous", "Genetic-ancestry PC1"),
    # --- family history ---
    Node("family_history_T2D", "binary", "Any first-degree relative with T2D"),
    # --- lifestyle / environment ---
    Node("years_smoking", "continuous", "Cumulative years of smoking"),
    Node("physical_activity", "continuous", "Average MET-hours per week"),
    Node("diet_quality", "continuous", "Healthy-Eating-Index z-score"),
    # --- clinical mediators ---
    Node("BMI", "continuous", "Body-mass index (kg/m^2)"),
    Node("LDL", "continuous", "LDL cholesterol (mmol/L)"),
    Node("HbA1c", "continuous", "Glycated haemoglobin (%)"),
    Node("systolic_BP", "continuous", "Systolic blood pressure (mmHg)"),
    # --- related disease states ---
    Node("hypertension", "binary", "Hypertension diagnosis"),
    Node("cardiovascular_disease", "binary", "Prior CVD event"),
    # --- outcome ---
    Node("T2D", "survival", "Time-to-T2D with right-censoring"),
)

NODE_NAMES: Tuple[str, ...] = tuple(n.name for n in NODES)
NODE_INDEX: dict = {n.name: i for i, n in enumerate(NODES)}
NODE_TYPES: Tuple[str, ...] = tuple(n.kind for n in NODES)
N_NODES: int = len(NODES)


# ---------------------------------------------------------------------------
# Ground-truth causal edges.
#
# Used by:
#   data/synthetic.py    -> samples are generated from exactly these edges,
#   validation/known_edges.py -> recovery rate is measured against them.
#
# Edges are directed (parent -> child).  We separate "canonical" edges (in
# textbook / MR literature) from "synthetic-only" edges we introduce to make
# the data richer; both are simulated, but only the canonical ones count for
# the known-edge benchmark.
# ---------------------------------------------------------------------------

CANONICAL_EDGES: Tuple[Tuple[str, str], ...] = (
    # genetic -> phenotype
    ("PGS_BMI", "BMI"),
    ("PGS_LDL", "LDL"),
    ("PGS_HbA1c", "HbA1c"),
    ("PGS_T2D", "T2D"),
    # lifestyle -> phenotype  (well-established)
    ("years_smoking", "cardiovascular_disease"),
    ("physical_activity", "BMI"),
    ("diet_quality", "BMI"),
    ("diet_quality", "LDL"),
    # clinical mediators -> outcome
    ("BMI", "T2D"),
    ("BMI", "hypertension"),
    ("BMI", "HbA1c"),
    ("HbA1c", "T2D"),
    ("systolic_BP", "cardiovascular_disease"),
    # age is a driver of almost everything
    ("age", "T2D"),
    ("age", "hypertension"),
    ("age", "cardiovascular_disease"),
    ("age", "systolic_BP"),
    # family history captures unmeasured genetic+environmental shared liab.
    ("family_history_T2D", "T2D"),
    # hypertension -> CVD
    ("hypertension", "cardiovascular_disease"),
    ("hypertension", "systolic_BP"),
)

# Extra edges used only by the simulator (mild confounders / demographic
# effects that are realistic but not "textbook causal").
SYNTHETIC_ONLY_EDGES: Tuple[Tuple[str, str], ...] = (
    ("sex", "BMI"),
    ("sex", "systolic_BP"),
    ("sex", "cardiovascular_disease"),
    ("ancestry_PC1", "LDL"),
    ("ancestry_PC1", "HbA1c"),
)

ALL_GROUND_TRUTH_EDGES: Tuple[Tuple[str, str], ...] = (
    CANONICAL_EDGES + SYNTHETIC_ONLY_EDGES
)


def edges_as_index_pairs(edges) -> List[Tuple[int, int]]:
    """Convert (parent_name, child_name) edges to (i, j) index pairs."""
    return [(NODE_INDEX[p], NODE_INDEX[c]) for p, c in edges]


def adjacency_from_edges(edges) -> "object":
    """Return a dense 0/1 adjacency matrix for the given edges (numpy)."""
    import numpy as np

    A = np.zeros((N_NODES, N_NODES), dtype=int)
    for p, c in edges:
        A[NODE_INDEX[p], NODE_INDEX[c]] = 1
    return A


__all__ = [
    "Node",
    "NODES",
    "NODE_NAMES",
    "NODE_INDEX",
    "NODE_TYPES",
    "N_NODES",
    "CANONICAL_EDGES",
    "SYNTHETIC_ONLY_EDGES",
    "ALL_GROUND_TRUTH_EDGES",
    "edges_as_index_pairs",
    "adjacency_from_edges",
]
