"""Tests for the MrDAG edge-inclusion-probability estimator.

Covers:
  * return-type and shape
  * probabilities in [0, 1] where defined
  * NaN outside the MR trait set
  * recovery of BMI -> T2D (MR-positive, known from literature)
  * non-recovery of LDL -> T2D (MR-null, known from literature)
  * diagonal NaN or 0
  * between-chain agreement + Gelman-Rubin R-hat
  * real-literature GWAS loader
  * DAG-implied total-effect utility: no path -> T(i, j) = 0
  * cycle rejection in the MCMC machinery
"""

import numpy as np
import pytest

from causal_pred.data.nodes import NODE_INDEX, NODE_NAMES, N_NODES
from causal_pred.data.gwas import MR_EXPOSURES, MR_OUTCOMES
from causal_pred.mrdag.pipeline import (
    run_mrdag,
    MrDAGResult,
    _compute_T,
    _creates_cycle_if_add,
    _is_reachable,
)


# ---------------------------------------------------------------------------
# Module-scoped fixture so we only run MCMC once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mrdag_result():
    from causal_pred.data.gwas import simulate_gwas

    gwas = simulate_gwas(rng=np.random.default_rng(7))
    return run_mrdag(
        gwas,
        rng=np.random.default_rng(2024),
        n_iter=6000,
        n_chains=4,
        n_burn=1000,
        thin=5,
    )


# ---------------------------------------------------------------------------
# Basic return-type / shape / domain tests.
# ---------------------------------------------------------------------------


def test_return_type_and_shape(mrdag_result):
    res = mrdag_result
    assert isinstance(res, MrDAGResult)
    assert res.pi.shape == (N_NODES, N_NODES)
    assert tuple(res.nodes) == NODE_NAMES
    assert res.n_chains == 4


def test_probabilities_in_unit_interval(mrdag_result):
    pi = mrdag_result.pi
    defined = ~np.isnan(pi)
    assert np.all(pi[defined] >= 0.0 - 1e-12)
    assert np.all(pi[defined] <= 1.0 + 1e-12)


def test_diagonal_is_nan_or_zero(mrdag_result):
    pi = mrdag_result.pi
    for i in range(N_NODES):
        v = pi[i, i]
        assert np.isnan(v) or v == 0.0, f"pi[{i},{i}] = {v}"


def test_entries_outside_mr_set_are_nan(mrdag_result):
    pi = mrdag_result.pi
    mr_traits = set(MR_EXPOSURES) | set(MR_OUTCOMES)
    for i, name_i in enumerate(NODE_NAMES):
        for j, name_j in enumerate(NODE_NAMES):
            if i == j:
                continue
            if name_i not in mr_traits or name_j not in mr_traits:
                assert np.isnan(pi[i, j]), (
                    f"Expected NaN at ({name_i}, {name_j}) outside MR set, "
                    f"got {pi[i, j]}"
                )


# ---------------------------------------------------------------------------
# Known MR edges / non-edges.
# ---------------------------------------------------------------------------


def test_bmi_to_t2d_high(mrdag_result):
    pi = mrdag_result.pi
    val = pi[NODE_INDEX["BMI"], NODE_INDEX["T2D"]]
    assert np.isfinite(val)
    assert val > 0.7, f"Expected BMI->T2D > 0.7, got {val:.3f}"


def test_ldl_to_t2d_low(mrdag_result):
    pi = mrdag_result.pi
    val = pi[NODE_INDEX["LDL"], NODE_INDEX["T2D"]]
    assert np.isfinite(val)
    assert val < 0.3, f"Expected LDL->T2D < 0.3, got {val:.3f}"


# ---------------------------------------------------------------------------
# Chain diagnostics.
# ---------------------------------------------------------------------------


def test_between_chain_agreement(mrdag_result):
    diag = mrdag_result.diagnostics
    assert diag["between_chain_max_abs_diff"] < 0.25, (
        f"max|diff| = {diag['between_chain_max_abs_diff']:.3f}"
    )


def test_rhat_bound(mrdag_result):
    """Gelman-Rubin R-hat on per-edge indicators should be within 1.2."""
    diag = mrdag_result.diagnostics
    assert "max_rhat_on_allowed" in diag
    assert diag["max_rhat_on_allowed"] <= 1.2, (
        f"max R-hat = {diag['max_rhat_on_allowed']:.3f}"
    )


def test_diagnostics_present(mrdag_result):
    diag = mrdag_result.diagnostics
    for key in (
        "accept_rates",
        "mean_log_posterior_per_chain",
        "rhat_per_edge",
        "max_rhat_on_allowed",
        "between_chain_max_abs_diff",
        "n_candidate_edges",
    ):
        assert key in diag
    assert len(diag["accept_rates"]) == mrdag_result.n_chains
    for ar in diag["accept_rates"]:
        assert 0.0 <= ar <= 1.0


# ---------------------------------------------------------------------------
# Real-literature GWAS loader.
# ---------------------------------------------------------------------------


def test_real_gwas_loads():
    from causal_pred.data.real_gwas import load_real_gwas

    g = load_real_gwas()
    assert g.betas.shape == g.ses.shape
    i_bmi = g.exposure_index("BMI")
    j_t2d = g.outcome_index("T2D")
    assert np.isfinite(g.betas[i_bmi, j_t2d])
    assert g.betas[i_bmi, j_t2d] > 0.4, (
        f"Literature BMI->T2D beta should be > 0.4, got {g.betas[i_bmi, j_t2d]}"
    )
    i_ldl = g.exposure_index("LDL")
    assert np.isfinite(g.betas[i_ldl, j_t2d])
    assert abs(g.betas[i_ldl, j_t2d]) < 0.15, (
        f"Literature LDL->T2D beta should be ~0, got {g.betas[i_ldl, j_t2d]}"
    )
    # Circular pair should have been dropped.
    i_hba = g.exposure_index("HbA1c")
    assert np.isnan(g.betas[i_hba, j_t2d])


def test_run_mrdag_accepts_real_gwas():
    """``run_mrdag`` should work on a literature-based ``RealGWASSummary``."""
    from causal_pred.data.real_gwas import load_real_gwas

    g = load_real_gwas()
    res = run_mrdag(
        g,
        rng=np.random.default_rng(11),
        n_iter=3000,
        n_chains=2,
        n_burn=500,
        thin=5,
    )
    assert res.pi.shape == (N_NODES, N_NODES)
    # BMI -> T2D should come out high even on the real data.
    bmi_t2d = res.pi[NODE_INDEX["BMI"], NODE_INDEX["T2D"]]
    assert np.isfinite(bmi_t2d)
    assert bmi_t2d > 0.7, f"real-data BMI->T2D = {bmi_t2d:.3f}"
    # LDL -> T2D should remain low.
    ldl_t2d = res.pi[NODE_INDEX["LDL"], NODE_INDEX["T2D"]]
    assert np.isfinite(ldl_t2d)
    assert ldl_t2d < 0.3, f"real-data LDL->T2D = {ldl_t2d:.3f}"


# ---------------------------------------------------------------------------
# Total-effect (path) utility.
# ---------------------------------------------------------------------------


def test_path_effect_zero_when_no_path():
    """If G has no directed path i -> j, T(G, B)[i, j] must be 0."""
    n = 4
    adj = np.zeros((n, n), dtype=int)
    B = np.zeros((n, n), dtype=float)
    # Add a chain 0 -> 1 -> 2 but leave node 3 disconnected.
    adj[0, 1] = 1
    B[0, 1] = 0.3
    adj[1, 2] = 1
    B[1, 2] = 0.5
    T = _compute_T(adj, B)
    # 0 -> 2 exists (product = 0.15)
    assert abs(T[0, 2] - 0.15) < 1e-9
    # 0 -> 3: no path
    assert abs(T[0, 3]) < 1e-12
    # 3 -> 0: no path
    assert abs(T[3, 0]) < 1e-12
    # 2 -> 0: no path (backwards)
    assert abs(T[2, 0]) < 1e-12


# ---------------------------------------------------------------------------
# Cycle rejection.
# ---------------------------------------------------------------------------


def test_cycle_rejection_in_detector():
    """_creates_cycle_if_add must reject moves that close a directed cycle."""
    n = 4
    adj = np.zeros((n, n), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    adj[2, 3] = 1
    # Adding 3 -> 0 closes the cycle 0 -> 1 -> 2 -> 3 -> 0.
    assert _creates_cycle_if_add(adj, 3, 0) is True
    # Adding 0 -> 2 does not create a cycle (it's just a shortcut).
    assert _creates_cycle_if_add(adj, 0, 2) is False
    # Self-loops are cycles.
    assert _creates_cycle_if_add(adj, 2, 2) is True


def test_reachability_consistency():
    n = 3
    adj = np.zeros((n, n), dtype=int)
    adj[0, 1] = 1
    adj[1, 2] = 1
    assert _is_reachable(adj, 0, 2) is True
    assert _is_reachable(adj, 2, 0) is False


def test_mcmc_never_produces_cycle(mrdag_result):
    """The returned pi must be consistent with acyclic sampling: an
    exact check is expensive, but we can at least check that no two
    candidate edges with high inclusion form a 2-cycle i<->j."""
    pi = mrdag_result.pi
    defined = ~np.isnan(pi)
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if defined[i, j] and defined[j, i]:
                # Under an acyclic posterior, both directions being simul-
                # taneously high-probability is impossible because each
                # sample G is a DAG.  A softer mixing-quality sanity
                # check: the sum of the two marginal probabilities must
                # not exceed 1 by much (they are marginals of mutually
                # exclusive indicator events in every DAG sample).
                s = pi[i, j] + pi[j, i]
                assert s <= 1.0 + 1e-6, (
                    f"pi[{i},{j}] + pi[{j},{i}] = {s:.3f} > 1 "
                    "(DAG sampler produced contradictory marginals)"
                )
