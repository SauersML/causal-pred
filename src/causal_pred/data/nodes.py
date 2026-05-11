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
    label: str  # pretty display name for plots / log strings
    exogenous: bool = False


# Ordered so that any topological sort of the synthetic DAG will respect it.
# Genetic/demographic roots first, then mediators, then the T2D outcome.
#
# ``name`` is the canonical lowercase snake_case identifier used for every
# array column, dict key, and string comparison in the codebase.  ``label``
# is the pretty display name used only by plots and log messages -- it is
# data, not a code path.
NODES: Tuple[Node, ...] = (
    # --- genetic roots (continuous, standardised polygenic scores) ---
    Node("pgs_t2d", "continuous", "Polygenic score for T2D", "PGS T2D", exogenous=True),
    Node("pgs_bmi", "continuous", "Polygenic score for body-mass index", "PGS BMI", exogenous=True),
    Node("pgs_ldl", "continuous", "Polygenic score for LDL cholesterol", "PGS LDL", exogenous=True),
    Node("pgs_hba1c", "continuous", "Polygenic score for HbA1c", "PGS HbA1c", exogenous=True),
    # --- demographic roots ---
    Node("age", "continuous", "Current age (years)", "age", exogenous=True),
    Node("sex", "binary", "1 = male, 0 = female", "sex", exogenous=True),
    Node("ancestry_pc1", "continuous", "Genetic-ancestry PC1", "ancestry PC1", exogenous=True),
    # --- family history ---
    Node(
        "family_history_t2d",
        "binary",
        "Any first-degree relative with T2D",
        "family history T2D",
        exogenous=True,
    ),
    # --- lifestyle / environment ---
    Node("years_smoking", "continuous", "Cumulative years of smoking", "years smoking"),
    Node("physical_activity", "continuous", "Average MET-hours per week", "physical activity"),
    Node("diet_quality", "continuous", "Healthy-Eating-Index z-score", "diet quality"),
    # --- clinical mediators ---
    Node("bmi", "continuous", "Body-mass index (kg/m^2)", "BMI"),
    Node("ldl_cholesterol", "continuous", "LDL cholesterol (mmol/L)", "LDL cholesterol"),
    Node("hba1c", "continuous", "Glycated haemoglobin (%)", "HbA1c"),
    Node("systolic_bp", "continuous", "Systolic blood pressure (mmHg)", "systolic BP"),
    # --- related disease states ---
    Node("hypertension", "binary", "Hypertension diagnosis", "hypertension"),
    Node("cardiovascular_disease", "binary", "Prior CVD event", "cardiovascular disease"),
    # --- outcome ---
    Node("type2_diabetes", "survival", "Time-to-T2D with right-censoring", "T2D"),
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
    ("pgs_bmi", "bmi"),
    ("pgs_ldl", "ldl_cholesterol"),
    ("pgs_hba1c", "hba1c"),
    ("pgs_t2d", "type2_diabetes"),
    # lifestyle -> phenotype  (well-established)
    ("years_smoking", "cardiovascular_disease"),
    ("physical_activity", "bmi"),
    ("diet_quality", "bmi"),
    ("diet_quality", "ldl_cholesterol"),
    # clinical mediators -> outcome
    ("bmi", "type2_diabetes"),
    ("bmi", "hypertension"),
    ("bmi", "hba1c"),
    ("systolic_bp", "cardiovascular_disease"),
    # age is a driver of almost everything
    ("age", "type2_diabetes"),
    ("age", "hypertension"),
    ("age", "cardiovascular_disease"),
    ("age", "systolic_bp"),
    # family history captures unmeasured genetic+environmental shared liab.
    ("family_history_t2d", "type2_diabetes"),
    # hypertension -> CVD
    ("hypertension", "cardiovascular_disease"),
)

# Extra edges used only by the simulator (mild confounders / demographic
# effects that are realistic but not "textbook causal").
SYNTHETIC_ONLY_EDGES: Tuple[Tuple[str, str], ...] = (
    ("sex", "bmi"),
    ("sex", "systolic_bp"),
    ("sex", "cardiovascular_disease"),
    ("ancestry_pc1", "ldl_cholesterol"),
    ("ancestry_pc1", "hba1c"),
    # Simulated risk-marker/diagnostic dependencies, excluded from the clean
    # literature gold-standard benchmark.
    ("hba1c", "type2_diabetes"),
    ("hypertension", "systolic_bp"),
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
