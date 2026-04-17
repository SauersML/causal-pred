"""Sanity checks for the synthetic AoU-shaped data generator."""

import numpy as np

from causal_pred.data.nodes import (
    NODE_NAMES,
    NODE_TYPES,
    N_NODES,
    CANONICAL_EDGES,
    NODE_INDEX,
)


def test_shapes_and_columns(small_data):
    d = small_data
    assert d.X.shape == (800, N_NODES)
    assert d.time.shape == (800,)
    assert d.event.shape == (800,)
    assert tuple(d.columns) == NODE_NAMES
    assert tuple(d.node_types) == NODE_TYPES


def test_event_rate_reasonable(medium_data):
    d = medium_data
    rate = d.event.mean()
    # >=10% events and <=80% (we want non-trivial learning and non-trivial
    # censoring).
    assert 0.10 < rate < 0.80


def test_time_positive(small_data):
    assert np.all(small_data.time > 0)


def test_binary_columns_are_binary(small_data):
    for i, t in enumerate(NODE_TYPES):
        if t == "binary":
            vals = np.unique(small_data.X[:, i])
            assert set(vals.tolist()).issubset({0.0, 1.0})


def test_bmi_increases_t2d_risk(medium_data):
    """One-sided sanity: people with BMI > 30 should have higher event rate."""
    bmi = medium_data.X[:, NODE_INDEX["BMI"]]
    high = medium_data.event[bmi > 30]
    low = medium_data.event[bmi < 25]
    assert high.mean() > low.mean()


def test_ground_truth_adjacency_contains_canonical(small_data):
    A = small_data.ground_truth_adj
    for p, c in CANONICAL_EDGES:
        assert A[NODE_INDEX[p], NODE_INDEX[c]] == 1, (p, c)
