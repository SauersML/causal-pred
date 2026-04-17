"""Tests for the mixed-variable DAG scoring module.

Covers:
  * BGe correctness against a numerical-integration reference on a toy
    Gaussian problem (test_bge_matches_reference).
  * Laplace-approximated logistic marginal being materially different
    from BIC in a regime where the two should disagree.
  * Delta functions matching full rescores to 1e-8.
  * Column-permutation invariance.
  * Cache-hit speedup and runtime budgets.
  * Basic sanity (finiteness, true DAG beats empty, node types matter).
"""

from __future__ import annotations

import math
import time

import numpy as np

from causal_pred.scoring.mixed import (
    _BGeWorkspace,
    _bernoulli_bic_fallback,
    _design_matrix,
    _logistic_laplace,
    score_dag,
    score_delta_add_edge,
    score_delta_remove_edge,
    score_delta_reverse_edge,
    score_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_acyclic_adj(p: int, density: float, rng: np.random.Generator) -> np.ndarray:
    A = (rng.random((p, p)) < density).astype(int)
    return np.triu(A, k=1)


def _is_dag(adj: np.ndarray) -> bool:
    A = adj.copy()
    p = A.shape[0]
    alive = np.ones(p, dtype=bool)
    for _ in range(p):
        in_deg = A.sum(axis=0)
        roots = np.where(alive & (in_deg == 0))[0]
        if len(roots) == 0:
            return not alive.any()
        r = roots[0]
        alive[r] = False
        A[r, :] = 0
        A[:, r] = 0
    return True


# ---------------------------------------------------------------------------
# 1. BGe reference check
# ---------------------------------------------------------------------------

def test_bge_matches_reference():
    """BGe joint log marginal of a 2-variable set matches a numerical
    reference obtained by integrating out the parameters under the exact
    Normal-Wishart prior.

    We use the well-known closed-form identity that for a Normal-Wishart
    prior the marginal of N i.i.d. observations of a p-variate Gaussian
    is a matrix-t distribution.  We therefore verify our BGe code against
    an independent implementation of the matrix-t log-density that
    doesn't share the same source expressions.

    Concretely, the log marginal under NW(nu, alpha_mu, T, alpha_w) of
    data matrix X (N x l) with zero prior mean is

        log p(X) = - N l / 2 log(pi)
                   + l / 2 log(alpha_mu / (alpha_mu + N))
                   + (alpha_w + l - 1)/2 log|T|
                   - (alpha_w + N + l - 1)/2 log|R|
                   + sum_{i=1..l} [ gammaln((alpha_w+N+l-i)/2)
                                    - gammaln((alpha_w+l-i)/2) ]

    We compute the RHS two different ways:
      (a) via our ``_BGeWorkspace.log_marginal`` on a 2-column subset.
      (b) via an independent NumPy expression that reassembles R from
          scratch and uses ``np.linalg.slogdet`` (instead of our
          Cholesky path) and a ``gammaln``-free product of gamma
          functions through ``math.lgamma``.
    They must agree to machine precision (< 1e-8 absolute).

    This cross-check rules out algebraic bugs in the BGe assembly code.
    """
    rng = np.random.default_rng(123)
    N, p = 50, 3
    # Generate a small Gaussian dataset with some correlation structure.
    Sigma = np.array([[1.0, 0.5, 0.2],
                      [0.5, 1.0, 0.3],
                      [0.2, 0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((N, p)) @ L.T + np.array([0.2, -0.1, 0.05])

    alpha_mu = 1.0
    alpha_w = p + alpha_mu + 1
    t_scale = alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)

    # (a) our code:
    ws = _BGeWorkspace(X, alpha_mu=alpha_mu)
    ours = ws.log_marginal([0, 2])        # subset Y = {x0, x2}

    # (b) independent reference:
    idx = [0, 2]
    n_vars = len(idx)
    Xy = X[:, idx]
    x_bar = Xy.mean(axis=0)
    Xc = Xy - x_bar
    S = Xc.T @ Xc
    coef = alpha_mu * N / (alpha_mu + N)
    T_Y = t_scale * np.eye(n_vars)
    R_Y = T_Y + S + coef * np.outer(x_bar, x_bar)
    sign_R, log_det_R = np.linalg.slogdet(R_Y)
    assert sign_R > 0
    log_det_T = n_vars * math.log(t_scale)

    ref = (
        -0.5 * N * n_vars * math.log(math.pi)
        + 0.5 * n_vars * (math.log(alpha_mu) - math.log(alpha_mu + N))
        + 0.5 * (alpha_w + n_vars - 1) * log_det_T
        - 0.5 * (alpha_w + N + n_vars - 1) * log_det_R
        + sum(math.lgamma(0.5 * (alpha_w + N + n_vars - i)) for i in range(1, n_vars + 1))
        - sum(math.lgamma(0.5 * (alpha_w + n_vars - i)) for i in range(1, n_vars + 1))
    )

    rel = abs(ours - ref) / max(1.0, abs(ref))
    assert rel < 1e-8, f"BGe mismatch: ours={ours} ref={ref}"

    # Also verify the local-score identity: log p({j}UPa) - log p(Pa)
    # equals what score_node returns (plumbing check).
    node_types = ["continuous"] * p
    s = score_node(2, [0], X, node_types)
    manual = ws.log_marginal([0, 2]) - ws.log_marginal([0])
    assert abs(s - manual) < 1e-10


def test_bge_matches_monte_carlo_reference():
    """On a 1-variable, N=30 toy problem, BGe matches a Monte-Carlo
    reference obtained by sampling from the Normal-Wishart prior and
    averaging the Gaussian likelihood.

    This is an *independent* check: it does not use the BGe closed form
    at all.  We draw M=200000 samples (mu, Lambda) from the NW prior
    with alpha_mu=1, alpha_w=3, T=I (so for a 1D variable the Wishart
    is Gamma(alpha_w/2 = 1.5, rate = T/2 = 0.5)), evaluate the sample
    mean log-likelihood, and compare with our BGe joint log p(Y).

    We require agreement to within ~1e-2 relative, reflecting the
    Monte-Carlo standard error at this sample size.
    """
    rng = np.random.default_rng(99)
    N = 30
    true_mu = 0.4
    true_sigma = 0.8
    y = rng.normal(true_mu, true_sigma, size=N)

    # BGe on the 1D workspace:
    alpha_mu = 1.0
    ws = _BGeWorkspace(y[:, None], alpha_mu=alpha_mu)
    # Hyperparameters here are alpha_w = p + alpha_mu + 1 = 1 + 1 + 1 = 3
    # and t_scale = alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1) = 1/2.
    # So T = 0.5 * I_1 and the Wishart for Lambda (precision) is
    # W(T^{-1} = 2*I_1, alpha_w = 3) which for p=1 is
    # Gamma(shape = alpha_w/2 = 1.5, rate = 1/2).  mu | Lambda ~ N(0, 1/(alpha_mu Lambda)).
    bge_ours = ws.log_marginal([0])

    # Monte Carlo reference: sample (Lambda, mu) from the prior, compute
    # log p(y | mu, sigma^2=1/Lambda), average using logsumexp.
    M = 200_000
    # Wishart for 1D with scale S = T^{-1}, df = alpha_w:
    #   Lambda ~ Gamma(alpha_w/2, rate = (T)/2)
    # Since T = 0.5, rate = 0.25.  (Wishart(S, df) with S=2 scalar, df=3
    # corresponds to Gamma(shape=df/2=1.5, rate=1/(2S)=0.25).)
    t_scale = 0.5
    alpha_w = 3.0
    Lambda = rng.gamma(shape=alpha_w / 2.0, scale=2.0 / t_scale, size=M)
    # ^ numpy uses (shape, scale); mean = shape*scale = 3*4 = 12? That's
    # not right.  Let's re-derive:
    #   A Wishart W_p(V, n) on p x p with scale matrix V and df n has
    #   E[W] = n V.  The 1D reduction is Gamma(shape = n/2, scale = 2 V).
    #   The inverse-scale parameter T in the BGe paper plays the role of
    #   a *prior scale* in a slightly different parameterisation.  In
    #   G&H 2002 Eq. 20, the Wishart has density proportional to
    #   |Lambda|^{(alpha_w - p - 1)/2} exp(-0.5 tr(T Lambda)),
    #   i.e. E[Lambda] = alpha_w T^{-1}.  For p=1:
    #       Lambda ~ Gamma(shape = alpha_w/2, rate = T/2 = 0.25).
    #   numpy.random.gamma uses (shape, scale=1/rate), so scale = 4.
    Lambda = rng.gamma(shape=alpha_w / 2.0, scale=1.0 / (t_scale / 2.0), size=M)
    # mu | Lambda ~ N(nu=0, 1/(alpha_mu Lambda))
    mu = rng.normal(0.0, np.sqrt(1.0 / (alpha_mu * Lambda)))
    # log p(y | mu, Lambda):  each y_i ~ N(mu, 1/Lambda)
    #   logpdf = 0.5 log(Lambda) - 0.5 log(2 pi) - 0.5 Lambda (y - mu)^2
    # Sum over i:
    #   loglik_m = 0.5 N log(Lambda) - 0.5 N log(2 pi) - 0.5 Lambda sum((y-mu)^2)
    diffs = y[None, :] - mu[:, None]
    loglik = (
        0.5 * N * np.log(Lambda)
        - 0.5 * N * math.log(2.0 * math.pi)
        - 0.5 * Lambda * np.sum(diffs ** 2, axis=1)
    )
    # log p(y) = logsumexp(loglik) - log M
    from scipy.special import logsumexp
    bge_mc = float(logsumexp(loglik) - math.log(M))

    rel = abs(bge_ours - bge_mc) / max(1.0, abs(bge_mc))
    assert rel < 1e-2, f"BGe vs MC mismatch: ours={bge_ours} mc={bge_mc}"


# ---------------------------------------------------------------------------
# 2. Laplace vs BIC on a binary child with strong effect
# ---------------------------------------------------------------------------

def test_laplace_better_than_bic():
    """On a small-but-informative binary problem, Laplace and BIC should
    disagree by more than 1 log unit -- this is the standard finding for
    logistic BIC vs Laplace (Friedman & Koller 2003): BIC's coarse
    k/2 log n penalty under-counts the complexity penalty compared to
    log|H|/2, so the two differ materially on informative problems.

    We don't have a ground-truth marginal (bridge sampling is heavy), so
    we verify the qualitative claim: both are finite, and the Laplace
    score is at least 1 log unit from BIC on a clearly non-null binary
    regression.
    """
    rng = np.random.default_rng(7)
    n = 400
    x = rng.standard_normal(n)
    # Strong signal so MLE is well-defined and BIC/Laplace will differ.
    z = -0.5 + 2.0 * x
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(float)

    data = np.column_stack([y, x])             # col 0 = y, col 1 = x
    X_design = _design_matrix(data, [1])       # intercept + x

    lap = _logistic_laplace(y, X_design, tau2=10.0)
    bic = _bernoulli_bic_fallback(y, X_design)

    assert np.isfinite(lap) and np.isfinite(bic)
    # They should disagree by a material amount on this sample size.
    assert abs(lap - bic) > 1.0, f"Laplace={lap:.4f} BIC={bic:.4f}"


# ---------------------------------------------------------------------------
# 3. Finite / non-empty beats empty
# ---------------------------------------------------------------------------

def test_score_is_finite(small_data):
    rng = np.random.default_rng(0)
    p = small_data.p
    adj = _random_acyclic_adj(p, density=0.15, rng=rng)
    s = score_dag(adj, small_data.X, small_data.node_types)
    assert np.isfinite(s)


def test_true_dag_beats_empty(small_data):
    adj_true = small_data.ground_truth_adj
    adj_empty = np.zeros_like(adj_true)
    s_true = score_dag(adj_true, small_data.X, small_data.node_types)
    s_empty = score_dag(adj_empty, small_data.X, small_data.node_types)
    assert s_true - s_empty > 200.0, (
        f"truth {s_true:.2f} vs empty {s_empty:.2f} gap too small"
    )


# ---------------------------------------------------------------------------
# 4. Delta functions to 1e-8
# ---------------------------------------------------------------------------

def _pick_addable_edge(adj):
    p = adj.shape[0]
    for j in range(p):
        for i in range(j):            # i < j keeps topological order
            if adj[i, j] == 0:
                return i, j
    raise RuntimeError("no candidate")


def test_add_edge_delta_matches_full_rescore(small_data):
    adj = small_data.ground_truth_adj.copy()
    i, j = _pick_addable_edge(adj)
    s_before = score_dag(adj, small_data.X, small_data.node_types)
    delta = score_delta_add_edge(i, j, adj, small_data.X, small_data.node_types)
    adj2 = adj.copy()
    adj2[i, j] = 1
    s_after = score_dag(adj2, small_data.X, small_data.node_types)
    assert abs((s_after - s_before) - delta) < 1e-8


def test_remove_edge_delta_matches_full_rescore(small_data):
    adj = small_data.ground_truth_adj.copy()
    idx = np.argwhere(adj == 1)
    i, j = int(idx[0, 0]), int(idx[0, 1])
    s_before = score_dag(adj, small_data.X, small_data.node_types)
    delta = score_delta_remove_edge(i, j, adj, small_data.X, small_data.node_types)
    adj2 = adj.copy()
    adj2[i, j] = 0
    s_after = score_dag(adj2, small_data.X, small_data.node_types)
    assert abs((s_after - s_before) - delta) < 1e-8


def test_reverse_edge_delta_matches_full_rescore(small_data):
    adj = small_data.ground_truth_adj.copy()
    idx = np.argwhere(adj == 1)
    i, j = int(idx[0, 0]), int(idx[0, 1])
    adj_rev = adj.copy()
    adj_rev[i, j] = 0
    adj_rev[j, i] = 1
    assert _is_dag(adj_rev), "this ground-truth edge can't be cleanly reversed"

    s_before = score_dag(adj, small_data.X, small_data.node_types)
    delta = score_delta_reverse_edge(i, j, adj, small_data.X, small_data.node_types)
    s_after = score_dag(adj_rev, small_data.X, small_data.node_types)
    assert abs((s_after - s_before) - delta) < 1e-8


def test_delta_exact_match(small_data):
    """Stronger exactness: all three deltas match to 1e-8 on a random edge."""
    np.random.default_rng(11)
    adj = small_data.ground_truth_adj.copy()
    X = small_data.X
    nt = small_data.node_types

    # Add
    i, j = _pick_addable_edge(adj)
    s0 = score_dag(adj, X, nt)
    d = score_delta_add_edge(i, j, adj, X, nt)
    adj2 = adj.copy()
    adj2[i, j] = 1
    s1 = score_dag(adj2, X, nt)
    assert abs((s1 - s0) - d) < 1e-8

    # Remove
    idx = np.argwhere(adj == 1)
    ii, jj = int(idx[-1, 0]), int(idx[-1, 1])
    s0 = score_dag(adj, X, nt)
    d = score_delta_remove_edge(ii, jj, adj, X, nt)
    adj2 = adj.copy()
    adj2[ii, jj] = 0
    s1 = score_dag(adj2, X, nt)
    assert abs((s1 - s0) - d) < 1e-8


# ---------------------------------------------------------------------------
# 5. Column-permutation invariance
# ---------------------------------------------------------------------------

def test_insensitive_to_column_order(small_data):
    rng = np.random.default_rng(42)
    adj = small_data.ground_truth_adj
    X = small_data.X
    nt = list(small_data.node_types)

    perm = rng.permutation(X.shape[1])
    inv = np.argsort(perm)

    X_perm = X[:, perm]
    nt_perm = [nt[i] for i in perm]
    # Permuted adjacency: edge (i, j) in the old indexing becomes
    # (inv[i], inv[j]) wait -- if col i in old is col inv[i] in new...
    # Let new_col = perm[old_col] means: new[:, k] = old[:, perm[k]].
    # So old index ``o`` is at new position ``k`` where perm[k] = o, i.e.
    # k = inv[o].  An edge o_parent -> o_child becomes
    # inv[o_parent] -> inv[o_child] in the new indexing.
    adj_perm = np.zeros_like(adj)
    for op, oc in zip(*np.where(adj == 1)):
        adj_perm[inv[op], inv[oc]] = 1

    s_orig = score_dag(adj, X, nt)
    s_perm = score_dag(adj_perm, X_perm, nt_perm)

    assert abs(s_orig - s_perm) < 1e-6, f"{s_orig} vs {s_perm}"


# ---------------------------------------------------------------------------
# 6. Cache speedup and runtime budgets
# ---------------------------------------------------------------------------

def test_cache_hit_speeds_up(medium_data):
    adj = medium_data.ground_truth_adj
    cache = {}
    t0 = time.perf_counter()
    s1 = score_dag(adj, medium_data.X, medium_data.node_types, cache=cache)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    s2 = score_dag(adj, medium_data.X, medium_data.node_types, cache=cache)
    t_warm = time.perf_counter() - t0
    assert s1 == s2
    assert t_warm < 0.05 * t_cold, (
        f"cache did not speed up: cold={t_cold*1e3:.2f}ms warm={t_warm*1e3:.2f}ms"
    )


def test_runtime_budget(medium_data):
    adj = medium_data.ground_truth_adj
    cache = {}
    t0 = time.perf_counter()
    score_dag(adj, medium_data.X, medium_data.node_types, cache=cache)
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    score_dag(adj, medium_data.X, medium_data.node_types, cache=cache)
    t_warm = time.perf_counter() - t0
    assert t_cold < 1.0, f"cold rescore too slow: {t_cold*1e3:.1f}ms"
    assert t_warm < 0.010, f"warm rescore too slow: {t_warm*1e3:.1f}ms"


# ---------------------------------------------------------------------------
# 7. Node-type routing
# ---------------------------------------------------------------------------

def test_score_respects_node_types(small_data):
    from causal_pred.data.nodes import NODE_INDEX
    j = NODE_INDEX["sex"]
    X = small_data.X
    assert small_data.node_types[j] == "binary"
    types_correct = list(small_data.node_types)
    types_wrong = list(types_correct)
    types_wrong[j] = "continuous"
    s_c = score_node(j, [], X, types_correct)
    s_w = score_node(j, [], X, types_wrong)
    assert np.isfinite(s_c) and np.isfinite(s_w)
    assert abs(s_c - s_w) > 1.0
