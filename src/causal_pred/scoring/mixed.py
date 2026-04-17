"""Mixed-variable DAG scoring (continuous + binary + survival).

Score forms (peer-review quality)
---------------------------------

* **Continuous node** j with parent set Pa is scored with the **exact
  BGe marginal likelihood** (Geiger & Heckerman 2002; Kuipers, Moffa &
  Heckerman 2014 correction) under a Normal-Wishart prior.

  With hyperparameters

      alpha_mu = 1
      alpha_w  = p + alpha_mu + 1            (the smallest integer value
                                              that keeps the Wishart
                                              proper; standard default)
      T        = t * I_p                      with
      t        = alpha_mu (alpha_w - p - 1) / (alpha_mu + 1)

  the joint log marginal likelihood of a subset ``Y`` of variables
  (|Y| = l) is, using the Kuipers-corrected constant,

      log p(Y) = - N l / 2 * log(pi)
                 + l / 2 * log(alpha_mu / (alpha_mu + N))
                 + (alpha_w + l - 1) / 2 * log|T_Y|
                 - (alpha_w + N + l - 1) / 2 * log|R_Y|
                 + sum_{i=1}^{l} log Gamma((alpha_w + N + l - i) / 2)
                 - sum_{i=1}^{l} log Gamma((alpha_w + l - i) / 2)

  where

      R = T + S + alpha_mu N / (alpha_mu + N) (x_bar - nu)(x_bar - nu)^T
      S = sum_{k=1}^N (x_k - x_bar)(x_k - x_bar)^T           (sample scatter)
      nu = 0                                                  (prior mean)

  The local BGe score for node j given parents Pa is then

      log_score(j | Pa) = log p({j} U Pa) - log p(Pa).

  References:
    Geiger, D. & Heckerman, D. (2002), "Parameter Priors for Directed
      Acyclic Graphical Models and the Characterization of Several
      Probability Distributions", *Ann. Stat.* 30(5), 1412--1440,
      Equation 22.
    Kuipers, J., Moffa, G. & Heckerman, D. (2014), "Addendum on the
      scoring of Gaussian directed acyclic graphical models",
      *Ann. Stat.* 42(4), 1689--1691, Equation 2 (this paper fixes a
      constant in the G&H 2002 formula; we use the corrected form).

* **Binary node** (and **survival** node, on the event indicator) with
  parent set Pa is scored with a **Laplace approximation** to the
  logistic-regression log marginal likelihood under a Gaussian prior
  on the coefficients, N(0, tau^2 I):

      log p(y | X) ~= log p(y | X, beta_MAP)
                     + log N(beta_MAP | 0, tau^2 I)
                     + k/2 * log(2 pi)
                     - 1/2 * log det(H)

  where H is the negative Hessian of the joint log-posterior at
  beta_MAP, i.e. H = X^T diag(p (1-p)) X + (1/tau^2) I,
  and k = dim(beta) = |Pa| + 1 (intercept is included).  The intercept
  is *not* penalised (its ridge coefficient is 0).

  beta_MAP is found by Newton-IRLS iteration to machine precision.  A
  numerically stable log-sigmoid ``- logaddexp(0, -z)`` is used, and
  ``log det(H)`` is obtained from a Cholesky factorisation with up to
  three rounds of diagonal jitter (10^-8 -> 10^-6 -> 10^-4) on
  Cholesky failure.  If ``n < 2k`` we fall back to BIC because the
  Laplace approximation can become unstable in that regime.

  Reference:
    Friedman, N. & Koller, D. (2003), "Being Bayesian about network
      structure: A Bayesian approach to structure discovery in
      Bayesian networks", *Machine Learning* 50, 95--125 -- uses the
      same Laplace-around-MAP construction for discrete local scores.

Caching
-------
Scores are deterministic in ``(j, frozenset(parents))`` and are
memoised in a user-supplied ``cache`` dict.  Calling ``score_dag``
once warms the cache; the delta functions then only recompute the
affected child(ren).  This is the hot path for structure MCMC.
"""

from __future__ import annotations

from typing import Iterable, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Hyperparameter defaults
# ---------------------------------------------------------------------------

_ALPHA_MU_DEFAULT = 1.0
# Laplace prior standard deviation on logistic coefficients (non-intercept).
# tau^2 = 10.0 ==> tau ~ 3.16; weakly informative on standardised features
# but enough to tame separation.
_TAU2_DEFAULT = 10.0
# Numerical stability
_RIDGE_XX = 1e-8  # added to X'X for BGe-adjacent computations
_JITTER_TRIES = (1e-8, 1e-6, 1e-4)
_NEWTON_TOL = 1e-10
_NEWTON_MAX_ITER = 100


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parents_key(parents: Iterable[int]) -> frozenset:
    """Canonical, hashable key for a parent set."""
    return frozenset(int(p) for p in parents)


def _safe_logdet_chol(M: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (log|M|, L) for SPD ``M`` via Cholesky with jitter retry.

    Attempts Cholesky with increasing diagonal jitter on failure.  Raises
    ``np.linalg.LinAlgError`` if none succeed.
    """
    n = M.shape[0]
    last_err = None
    for jitter in _JITTER_TRIES:
        try:
            A = M + jitter * np.eye(n)
            L = np.linalg.cholesky(A)
            # log|M| = 2 * sum(log(diag(L)))
            return 2.0 * float(np.sum(np.log(np.diag(L)))), L
        except np.linalg.LinAlgError as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# BGe local score for continuous nodes
#
# Implementation follows Kuipers, Moffa & Heckerman (2014) Equation 2.
# The local score for j | Pa is computed as the DIFFERENCE of two joint
# log marginal likelihoods, log p({j} U Pa) - log p(Pa); this cancels
# the expensive ``-N l / 2 log(pi)`` constant only partially (different
# l on either side) but both calls are O(l^3) for the Cholesky of the
# restricted R matrix so the cost is fine.
# ---------------------------------------------------------------------------


class _BGeWorkspace:
    """Cached data summaries used by every BGe local-score call.

    We precompute the centred design (X - x_bar) and its scatter
    ``S = sum (x - x_bar)(x - x_bar)^T`` once -- these only depend on
    ``data``, not on the graph -- and reuse them for every local score.
    The matrix we actually slice into on each call is ``R = T + S``
    (the prior mean nu = 0 in our standardised prior, so the rank-one
    correction in the full BGe formula is exactly ``S_bar = alpha_mu N /
    (alpha_mu + N) * x_bar x_bar^T``, which we add separately).
    """

    __slots__ = ("N", "p", "alpha_mu", "alpha_w", "t_scale", "R", "log_t")

    def __init__(self, data: np.ndarray, alpha_mu: float = _ALPHA_MU_DEFAULT):
        X = np.asarray(data, dtype=np.float64)
        N, p = X.shape
        self.N = int(N)
        self.p = int(p)
        self.alpha_mu = float(alpha_mu)
        self.alpha_w = float(p + alpha_mu + 1)  # standard default
        self.t_scale = float(
            self.alpha_mu * (self.alpha_w - p - 1.0) / (self.alpha_mu + 1.0)
        )
        # log|T_Y| = l * log(t_scale) because T = t_scale * I_p.
        self.log_t = float(np.log(self.t_scale))
        # R = T + S + (alpha_mu N / (alpha_mu + N)) x_bar x_bar^T   (nu=0)
        x_bar = X.mean(axis=0)  # (p,)
        Xc = X - x_bar
        S = Xc.T @ Xc  # (p, p) scatter
        coef = self.alpha_mu * N / (self.alpha_mu + N)
        # Prior mean nu=0 ==> (x_bar - nu)(x_bar - nu)^T = outer(x_bar, x_bar)
        bar_outer = np.outer(x_bar, x_bar)
        self.R = self.t_scale * np.eye(p) + S + coef * bar_outer

    # --- public API ------------------------------------------------------

    def log_marginal(self, idx: Sequence[int]) -> float:
        """log p(Y_idx) under the Normal-Wishart marginal (Kuipers 2014, Eq. 2).

        ``idx`` is an ordered list of variable indices defining the
        subset ``Y``.  Returns the joint log marginal likelihood of
        those variables.
        """
        N = self.N
        n_vars = len(idx)
        if n_vars == 0:
            return 0.0  # empty product -> log 1 = 0

        alpha_mu = self.alpha_mu
        alpha_w = self.alpha_w

        # Slice the shared R matrix; |T_Y| = t_scale^n_vars because T is t*I.
        R_Y = self.R[np.ix_(idx, idx)]
        log_det_R, _ = _safe_logdet_chol(R_Y)
        log_det_T = n_vars * self.log_t

        # Constant & log-gamma parts.
        term_const = -0.5 * N * n_vars * np.log(np.pi)
        term_mu = 0.5 * n_vars * (np.log(alpha_mu) - np.log(alpha_mu + N))
        term_T = 0.5 * (alpha_w + n_vars - 1.0) * log_det_T
        term_R = -0.5 * (alpha_w + N + n_vars - 1.0) * log_det_R

        # sum_{i=1}^{n_vars} [ log Gamma((alpha_w + N + n_vars - i)/2)
        #                     - log Gamma((alpha_w + n_vars - i)/2) ]
        i = np.arange(1, n_vars + 1, dtype=np.float64)
        term_gamma = float(
            np.sum(gammaln(0.5 * (alpha_w + N + n_vars - i)))
            - np.sum(gammaln(0.5 * (alpha_w + n_vars - i)))
        )

        return float(term_const + term_mu + term_T + term_R + term_gamma)

    def local_score(self, j: int, parents: Sequence[int]) -> float:
        """BGe local score: log p({j} U Pa) - log p(Pa)."""
        pa = list(parents)
        joint_idx = [j] + pa
        return self.log_marginal(joint_idx) - self.log_marginal(pa)


# The BGe workspace precomputes (R, T, x_bar) in O(N p^2) once per array.
#
# Historical note: we used to memoise workspaces in a module-level dict keyed
# by (id(data), shape, alpha_mu).  But Python's id() can be reused after a
# previous array is garbage collected, so a later test could receive a stale
# workspace whose R matrix was built from entirely different data.  The
# symptom was an order-dependent failure of
# ``test_insensitive_to_column_order``.  The workspace build is O(N p^2)
# (dominated by X^T X on the scatter), which is ~1ms at our scales --
# well below the O(l^3) log-marginal Cholesky cost per local-score call --
# so building fresh per ``score_dag`` is both correct and fast.
#
# We instead attach the workspace to the user-supplied ``cache`` mapping
# when one is provided, so MCMC (which reuses the same ``data`` and
# ``cache`` across thousands of calls) still pays the O(N p^2) build only
# once per chain.


def _get_bge_workspace(
    data: np.ndarray, alpha_mu: float, cache: Optional[MutableMapping] = None
) -> _BGeWorkspace:
    if cache is not None:
        # Namespaced key so it cannot collide with a ``(j, frozenset)``
        # local-score key.
        ws_key = ("__bge_workspace__", data.shape, float(alpha_mu))
        ws = cache.get(ws_key)
        if ws is None:
            ws = _BGeWorkspace(data, alpha_mu=alpha_mu)
            cache[ws_key] = ws
        return ws
    return _BGeWorkspace(data, alpha_mu=alpha_mu)


# ---------------------------------------------------------------------------
# Laplace approximation for binary (and survival) nodes
# ---------------------------------------------------------------------------


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable log sigmoid(z) = -log(1 + exp(-z))."""
    # logaddexp(0, -z) = log(1 + exp(-z))
    return -np.logaddexp(0.0, -z)


def _design_matrix(data: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    n = data.shape[0]
    if len(parents) == 0:
        return np.ones((n, 1), dtype=np.float64)
    X = np.empty((n, len(parents) + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = np.asarray(data[:, list(parents)], dtype=np.float64)
    return X


def _logistic_laplace(
    y: np.ndarray,
    X: np.ndarray,
    tau2: float = _TAU2_DEFAULT,
) -> float:
    """Laplace-approximated log marginal likelihood for logistic reg.

    log p(y | X) approx
        sum_i [ y_i z_i - logaddexp(0, z_i) ] |_{beta=beta_MAP}
      + log N(beta_MAP | 0, tau^2 I_prior)
      + (k/2) log(2 pi) - (1/2) log|H|

    where H = X^T diag(p(1-p)) X + Lambda,
          Lambda = diag([0, 1/tau^2, ..., 1/tau^2])  (intercept unpenalised),
    and beta_MAP is found by Newton-IRLS to machine precision.  The
    Gaussian prior Normaliser includes only the k-1 penalised
    coefficients (not the intercept, which has an improper flat prior);
    this keeps the score invariant to shifts of y's prevalence.
    """
    y = y.astype(np.float64, copy=False)
    n, k = X.shape

    # Prior precision (diagonal): zero on the intercept, 1/tau^2 elsewhere.
    prec = np.zeros(k, dtype=np.float64)
    if k > 1:
        prec[1:] = 1.0 / tau2

    beta = np.zeros(k, dtype=np.float64)

    prev_loss = np.inf
    for _ in range(_NEWTON_MAX_ITER):
        z = X @ beta
        # p_i = sigmoid(z_i); use stable expit-equivalent: 1/(1+exp(-z)).
        # np.exp handles overflow by going to inf -> p -> 0; the next
        # (1 - p) subtraction is exactly 1 in that regime.
        p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        w = p * (1.0 - p)
        # Gradient of negative joint log-posterior:
        #   -g = X^T (p - y) + prec * beta
        grad = X.T @ (p - y) + prec * beta
        # Hessian:  H = X^T diag(w) X + diag(prec)
        Xw = X * w[:, None]
        H = X.T @ Xw
        H[np.diag_indices_from(H)] += prec
        # Solve H delta = grad
        try:
            _, L = _safe_logdet_chol(H)
            delta = cho_solve((L, True), grad)
        except np.linalg.LinAlgError:
            # Very rare; fall back to pinv-based solve.
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        # Backtracking line search on the negative log-posterior.
        neg_log_post = _neg_log_posterior(beta, X, y, prec)
        step = 1.0
        for _ls in range(30):
            new_beta = beta - step * delta
            new_loss = _neg_log_posterior(new_beta, X, y, prec)
            if new_loss < neg_log_post - 1e-12:
                break
            step *= 0.5
        else:
            new_beta = beta  # no improvement -> stop
            new_loss = neg_log_post
        beta = new_beta

        if abs(prev_loss - new_loss) < _NEWTON_TOL * max(1.0, abs(prev_loss)):
            break
        prev_loss = new_loss

    # ---- evaluate Laplace log marginal ----
    z = X @ beta
    # log-lik = sum [y_i z_i - log(1 + exp(z_i))]  via logaddexp for stability
    log_lik = float(np.sum(y * z - np.logaddexp(0.0, z)))

    # Gaussian prior log pdf on penalised coefs:
    #   log N(beta_pen | 0, tau^2 I) = - (k_pen / 2) log(2 pi tau^2)
    #                                  - 0.5 * ||beta_pen||^2 / tau^2
    if k > 1:
        beta_pen = beta[1:]
        k_pen = k - 1
        log_prior = (
            -0.5 * k_pen * np.log(2.0 * np.pi * tau2)
            - 0.5 * float(beta_pen @ beta_pen) / tau2
        )
    else:
        log_prior = 0.0

    # Hessian at MAP
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    w = p * (1.0 - p)
    Xw = X * w[:, None]
    H = X.T @ Xw
    H[np.diag_indices_from(H)] += prec
    # For the intercept, prec is 0 and if n is tiny H can be singular; the
    # Cholesky retry handles rare near-singularity with jitter.
    log_det_H, _ = _safe_logdet_chol(H)

    # Laplace formula:
    log_marg = log_lik + log_prior + 0.5 * k * np.log(2.0 * np.pi) - 0.5 * log_det_H
    return float(log_marg)


def _neg_log_posterior(beta, X, y, prec) -> float:
    z = X @ beta
    # -log-lik:  sum logaddexp(0, z) - y*z
    neg_ll = float(np.sum(np.logaddexp(0.0, z)) - float(y @ z))
    # -log-prior (Gaussian, excluding constants since we only compare):
    neg_lp = 0.5 * float(np.sum(prec * beta * beta))
    return neg_ll + neg_lp


def _bernoulli_bic_fallback(y: np.ndarray, X: np.ndarray) -> float:
    """BIC fallback for binary nodes when n < 2 k (Laplace unstable)."""
    from sklearn.linear_model import LogisticRegression

    n, k = X.shape
    if k == 1 or y.sum() == 0 or y.sum() == n:
        s = float(y.sum())
        p = s / n
        p_c = min(max(p, 1e-12), 1.0 - 1e-12)
        log_lik = s * np.log(p_c) + (n - s) * np.log1p(-p_c)
        return float(log_lik - 0.5 * 1 * np.log(n))
    # A tiny L2 (large C) for numerical stability only; intercept is in X[:,0]
    # so we disable sklearn's own intercept.
    model = LogisticRegression(
        C=1e6,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
        tol=1e-8,
    ).fit(X, y.astype(int))
    w = model.coef_[0]
    z = X @ w
    log_lik = float(np.sum(y * z - np.logaddexp(0.0, z)))
    return float(log_lik - 0.5 * k * np.log(n))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_node(
    j: int,
    parents: Iterable[int],
    data: np.ndarray,
    node_types: Sequence[str],
    cache: Optional[MutableMapping] = None,
    **hyper,
) -> float:
    """Log marginal likelihood of node ``j`` given ``parents``.

    Routing by ``node_types[j]``:
      * ``"continuous"``: exact BGe (see module docstring).
      * ``"binary"`` / ``"survival"``: Laplace approximation to logistic
        regression marginal likelihood (see module docstring).
    """
    key = (int(j), _parents_key(parents))
    if cache is not None and key in cache:
        return cache[key]

    kind = node_types[j]
    parents_list = sorted(key[1])

    if kind == "continuous":
        alpha_mu = float(hyper.get("alpha_mu", _ALPHA_MU_DEFAULT))
        ws = _get_bge_workspace(
            np.asarray(data, dtype=np.float64),
            alpha_mu,
            cache=cache,
        )
        s = ws.local_score(j, parents_list)
    elif kind in ("binary", "survival"):
        y = np.asarray(data[:, j], dtype=np.float64)
        X = _design_matrix(data, parents_list)
        tau2 = float(hyper.get("tau2", _TAU2_DEFAULT))
        n, k = X.shape
        if n < 2 * k:
            s = _bernoulli_bic_fallback(y, X)
        else:
            s = _logistic_laplace(y, X, tau2=tau2)
    else:
        raise ValueError(f"Unknown node_type {kind!r} for node {j}")

    if cache is not None:
        cache[key] = s
    return float(s)


def _parents_of(adj: np.ndarray, j: int) -> np.ndarray:
    return np.nonzero(adj[:, j])[0]


def score_dag(
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: Optional[MutableMapping] = None,
    **hyper,
) -> float:
    """Sum of local scores over every node of the DAG.

    Calling this once populates ``cache`` with every ``(j, frozenset(pa_j))``
    currently in the graph, pre-warming the cache for MCMC.
    """
    adj = np.asarray(adj)
    p = adj.shape[1]
    total = 0.0
    for j in range(p):
        pa = _parents_of(adj, j)
        total += score_node(j, pa, data, node_types, cache=cache, **hyper)
    return float(total)


# ---------------------------------------------------------------------------
# Delta scores for MCMC moves.  Each delta = new_total - old_total.
# Because local scores are per-node, only the child (or children) whose
# parent set changes need to be rescored; the rest cancel in the sum.
# ---------------------------------------------------------------------------


def score_delta_add_edge(
    i: int,
    j: int,
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: Optional[MutableMapping] = None,
    **hyper,
) -> float:
    pa_old = set(_parents_of(adj, j).tolist())
    if i in pa_old:
        return 0.0
    pa_new = pa_old | {int(i)}
    old = score_node(j, pa_old, data, node_types, cache=cache, **hyper)
    new = score_node(j, pa_new, data, node_types, cache=cache, **hyper)
    return float(new - old)


def score_delta_remove_edge(
    i: int,
    j: int,
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: Optional[MutableMapping] = None,
    **hyper,
) -> float:
    pa_old = set(_parents_of(adj, j).tolist())
    if i not in pa_old:
        return 0.0
    pa_new = pa_old - {int(i)}
    old = score_node(j, pa_old, data, node_types, cache=cache, **hyper)
    new = score_node(j, pa_new, data, node_types, cache=cache, **hyper)
    return float(new - old)


def score_delta_reverse_edge(
    i: int,
    j: int,
    adj: np.ndarray,
    data: np.ndarray,
    node_types: Sequence[str],
    cache: Optional[MutableMapping] = None,
    **hyper,
) -> float:
    """Delta for reversing i -> j to j -> i.

    Two nodes change parent sets: ``j`` loses ``i``, ``i`` gains ``j``.
    """
    pa_j_old = set(_parents_of(adj, j).tolist())
    pa_i_old = set(_parents_of(adj, i).tolist())
    if i not in pa_j_old:
        return 0.0
    pa_j_new = pa_j_old - {int(i)}
    pa_i_new = pa_i_old | {int(j)}
    old_j = score_node(j, pa_j_old, data, node_types, cache=cache, **hyper)
    new_j = score_node(j, pa_j_new, data, node_types, cache=cache, **hyper)
    old_i = score_node(i, pa_i_old, data, node_types, cache=cache, **hyper)
    new_i = score_node(i, pa_i_new, data, node_types, cache=cache, **hyper)
    return float((new_j - old_j) + (new_i - old_i))


__all__ = [
    "score_node",
    "score_dag",
    "score_delta_add_edge",
    "score_delta_remove_edge",
    "score_delta_reverse_edge",
]
