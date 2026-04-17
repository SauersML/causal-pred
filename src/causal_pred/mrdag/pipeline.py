"""MrDAG: joint Mendelian-randomisation + Bayesian DAG structure learning.

References
----------
* Zuber V, Lewin A, Levin MG, Haycock PC, Burgess S.  "Bayesian causal
  graphical model for joint Mendelian randomization analysis of multiple
  exposures and outcomes."  Genet Epidemiol, 2025.
* Madigan D, York J.  "Bayesian graphical models for discrete data."
  Int Stat Rev, 1995.
* Giudici P, Castelo R.  "Improving Markov chain Monte Carlo model
  search for data mining."  Machine Learning, 2003.
* Wakefield J.  "Bayes factors for genome-wide association studies:
  comparison with P-values."  Genet Epidemiol, 2009.

Model
-----
Let ``G`` be a binary (p, p) adjacency matrix on the MR trait set with
no self-loops and no directed cycles.  Let ``B`` be the real-valued
weighted adjacency supported on the edges of ``G``.  The DAG-implied
total-effect matrix is

    T(G, B) = (I - B)^{-1} - I

whose (i, j) entry is the sum over directed i -> ... -> j paths of the
product of edge weights.  Given the "first-stage" approximation of
Zuber et al. (2025), we treat observed IVW cells as conditionally
independent given ``G``:

    beta_ij | G, B  ~  Normal(T_ij(G, B), se_ij^2).

For the DIRECT edge i -> j we analytically marginalise its weight
under a **competing-hypothesis** mixture prior conditional on the
edge being present: H1 ``b_ij ~ N(0, W)`` (a non-negligible causal
effect) versus a practical-null H0 ``b_ij ~ N(0, eps^2)`` with
``eps << sqrt(W)``.  Integrating both out against the Gaussian
likelihood gives a closed-form log Bayes factor (a Wakefield-type
ABF generalised to a non-point null).  This yields POSITIVE log BF
for residuals large relative to ``eps`` and NEGATIVE log BF for
residuals small relative to ``eps`` with tight SEs — the latter is
the "evidence-against" behaviour required to rule out MR-null edges
such as LDL -> T2D that would otherwise register mild |z| > 2 from
pleiotropic noise.  The BF is evaluated on the residual between the
observed IVW beta and the indirect-path contribution ``mu_ij``,
where

    mu_ij = T_ij(G, B_indirect)

is computed from plug-in posterior-mean weights on every OTHER edge.
Concretely the log-likelihood of one observed cell is

    log p(beta_ij | G)
      = log N(beta_ij | mu_ij, se_ij^2)                 (null residual)
      + 1[(i, j) in G] * log_bf(r_ij, se_ij, W, eps^2)

with

    log_bf(r, se, W, eps^2)
      = 0.5 * (log(se^2 + eps^2) - log(se^2 + W))
      + 0.5 * r^2 * (1 / (se^2 + eps^2) - 1 / (se^2 + W))

where r_ij = beta_ij - mu_ij is the direct-edge residual.  Summed
across observed cells and combined with the Bernoulli(pi0) edge prior

    log P(G) = k(G) * log(pi0) + (K - k(G)) * log(1 - pi0)

this gives the target log posterior

    log P(G | data)
      = const + log P(G)
      + sum_{(i,j) obs} [ log N(beta_ij | mu_ij, se_ij^2)
                        + 1[(i,j) in G] * log_bf(r_ij, se_ij, W, eps^2) ].

This is the "marginalise edge weights under the slab using a closed-
form Gaussian Bayes factor, accumulated along the DAG-implied paths"
formulation of MrDAG.  Direct edges are integrated over (so strong
prior regularisation kicks in), while indirect paths are treated with
plug-in point estimates so their contribution to mu_ij properly
reduces the residual at downstream cells.

Structure MCMC (Madigan-York 1995; Giudici-Castelo 2003)
--------------------------------------------------------
Moves: add, delete, reverse.  We enumerate the legal-move set N(G)
(respecting acyclicity and the allowed candidate-edge slots), pick
uniformly, and accept via Metropolis-Hastings with the Giudici-
Castelo neighbourhood-size correction

    alpha = min(1, exp(delta_logpost) * |N(G)| / |N(G')|).

This guarantees detailed balance on the DAG space.

Numerics
--------
Cycle detection is exact DFS reachability.  Per-cell Wakefield log
BFs are computed in closed form.  Plug-in weights b_hat_ij are the
Wakefield posterior means (W / (W + se_ij^2)) * beta_ij.  The
indirect total-effect matrix is obtained by solving (I - B) X = I on
the restricted MR trait set; p is small (<= 20) so this is cheap.
Multiple chains are run from the empty DAG with RNGs spawned from the
user-supplied seed; Gelman-Rubin R-hat on per-edge indicators is
reported in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.nodes import NODE_NAMES


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MrDAGResult:
    pi: np.ndarray  # (p, p) edge-inclusion probabilities, NODE_NAMES order
    nodes: Sequence[str]  # matches NODE_NAMES ordering
    n_chains: int
    diagnostics: dict


# ---------------------------------------------------------------------------
# Wakefield Bayes factor and posterior-mean weight
# ---------------------------------------------------------------------------

_LOG_2PI = float(np.log(2.0 * np.pi))


def _wakefield_log_bf(beta: float, se: float, W: float, eps2: float = 0.0) -> float:
    """Slab-vs-practical-null log Bayes factor.

    H1 (non-negligible causal effect): theta ~ N(0, W)
    H0 (practical null): theta ~ N(0, eps2)

    Both are integrated analytically against N(beta | theta, se^2),
    giving a marginal N(beta | 0, se^2 + V) and

        log BF_10 = 0.5 * (log(se^2 + eps2) - log(se^2 + W))
                  + 0.5 * beta^2 * (1/(se^2 + eps2) - 1/(se^2 + W)).

    When ``eps2 = 0`` this collapses to the classical Wakefield (2009)
    BF against a point null.  A strictly positive ``eps2`` installs a
    "minimum meaningful effect" scale: residuals much smaller than
    sqrt(eps2) with tight SEs now yield a NEGATIVE log BF (evidence
    against a non-negligible direct edge), which is the behaviour the
    literature reports for MR-null edges such as LDL -> T2D.
    """
    if not (np.isfinite(beta) and np.isfinite(se)) or se <= 0.0:
        return 0.0
    v = se * se
    v1 = v + W
    v0 = v + eps2
    if v1 <= 0.0 or v0 <= 0.0:
        return 0.0
    return 0.5 * (np.log(v0) - np.log(v1)) + 0.5 * (beta * beta) * (1.0 / v0 - 1.0 / v1)


def _posterior_mean_weight(beta: float, se: float, W: float) -> float:
    """Mean of the slab posterior on a single-edge direct effect."""
    if not (np.isfinite(beta) and np.isfinite(se)) or se <= 0.0:
        return 0.0
    v = se * se
    return (W / (W + v)) * beta


# ---------------------------------------------------------------------------
# DAG / reachability utilities
# ---------------------------------------------------------------------------


def _is_reachable(adj: np.ndarray, src: int, dst: int) -> bool:
    """True iff there is a directed path src -> ... -> dst in ``adj``."""
    if src == dst:
        return True
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    stack = [src]
    visited[src] = True
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            if v == dst:
                return True
            if not visited[v]:
                visited[v] = True
                stack.append(int(v))
    return False


def _creates_cycle_if_add(adj: np.ndarray, i: int, j: int) -> bool:
    """Adding i -> j creates a cycle iff j can already reach i."""
    if i == j:
        return True
    return _is_reachable(adj, j, i)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _compute_T(adj: np.ndarray, B_weights: np.ndarray) -> np.ndarray:
    """Return T(G, B) = (I - B)^{-1} - I with B = adj * B_weights."""
    n = adj.shape[0]
    B = adj.astype(float) * B_weights
    M = np.eye(n) - B
    inv = np.linalg.solve(M, np.eye(n))
    return inv - np.eye(n)


def _log_posterior(
    adj: np.ndarray,
    b_plugin: np.ndarray,
    obs_beta: np.ndarray,
    obs_se: np.ndarray,
    obs_mask: np.ndarray,
    log_bf_direct: np.ndarray,
    allowed: np.ndarray,
    pi0: float,
    W: float,
    eps2: float = 0.0,
) -> float:
    """Target log posterior log P(G | data) up to constants.

    log P(G | data) = log P(G) + sum over observed cells of
        [ log N(beta_ij | mu_ij, se_ij^2)
          + 1[(i,j) edge present] * wakefield_log_bf(r_ij, se_ij, W) ],

    where mu_ij = T_ij(G, B_indirect) with B_indirect zeroed on the
    direct (i, j) entry.

    We compute this as:
      mu = T(G, B_plugin)
      r = beta - mu
      # For present direct edges the plug-in weight inflates mu_ij by
      # b_hat_ij through the matrix inverse; we undo this on the direct
      # cell so r_ij is a pure "residual with weight = 0" quantity.
      # On a DAG, T_ij with B_ij set to b equals T_ij_noedge * (1) + b
      # when we *linearise*; more precisely, the exact correction is
      # obtained from the Neumann series.  We use the exact rank-1
      # correction: removing b from the (i, j) entry of B reduces
      # T_ij by b * (1 + T_ji(G, B)) ~ b (since T_ji is small on a DAG
      # where j does not reach i).  On a DAG j cannot reach i through
      # G (by acyclicity), so T_ji == 0 exactly and the correction is
      # exactly b_hat_ij.
    """
    # 1) Indirect-path mu with all plug-in weights.
    T = _compute_T(adj, b_plugin)
    # 2) For cells where the direct edge is present, subtract b_hat_ij
    # from T_ij to get mu_ij (the indirect-only contribution).  On a
    # DAG, j cannot reach i (acyclicity), so the rank-1 correction is
    # exact: T_ij - b_hat_ij on a DAG with edge i -> j.
    direct_present = (adj == 1) & allowed
    mu = T.copy()
    mu[direct_present] -= b_plugin[direct_present]

    # 3) Residuals on observed cells.
    r = obs_beta - mu
    r_obs = r[obs_mask]
    se_obs = obs_se[obs_mask]
    v = se_obs * se_obs
    log_null = -0.5 * float(np.sum(r_obs * r_obs / v + np.log(v) + _LOG_2PI))

    # 4) Wakefield BF contribution for present direct edges on
    # observed cells.  log_bf_direct is *recomputed* here from the
    # residual r_ij rather than the static beta_ij, so the slab
    # integration correctly conditions on what remains after indirect
    # paths.
    bf_contrib = 0.0
    if direct_present.any():
        rd = r[direct_present]
        sed = obs_se[direct_present]
        # Vectorised slab-vs-practical-null log BF (see _wakefield_log_bf).
        vd = sed * sed
        v1 = vd + W
        v0 = vd + eps2
        bf = 0.5 * (np.log(v0) - np.log(v1)) + 0.5 * (rd * rd) * (1.0 / v0 - 1.0 / v1)
        bf_contrib = float(bf.sum())

    # 5) Structure prior.
    K = int(allowed.sum())
    k = int(direct_present.sum())
    if 0.0 < pi0 < 1.0:
        log_prior = k * np.log(pi0) + (K - k) * np.log(1.0 - pi0)
    else:
        log_prior = 0.0

    return log_null + bf_contrib + log_prior


# ---------------------------------------------------------------------------
# Legal-move enumeration (Giudici-Castelo)
# ---------------------------------------------------------------------------


def _legal_moves(adj: np.ndarray, allowed: np.ndarray) -> List[Tuple[str, int, int]]:
    legal: List[Tuple[str, int, int]] = []
    # Deletes.
    for i, j in np.argwhere(adj == 1):
        legal.append(("delete", int(i), int(j)))
    # Reverses: only if reversed slot is allowed and acyclic.
    for i, j in np.argwhere(adj == 1):
        i = int(i)
        j = int(j)
        if not allowed[j, i] or adj[j, i] == 1:
            continue
        adj[i, j] = 0
        ok = not _is_reachable(adj, i, j)
        adj[i, j] = 1
        if ok:
            legal.append(("reverse", i, j))
    # Adds.
    for i, j in np.argwhere(allowed):
        i = int(i)
        j = int(j)
        if adj[i, j] == 1 or adj[j, i] == 1:
            continue
        if not _is_reachable(adj, j, i):
            legal.append(("add", i, j))
    return legal


def _apply_move(adj: np.ndarray, move: Tuple[str, int, int]) -> None:
    mtype, i, j = move
    if mtype == "add":
        adj[i, j] = 1
    elif mtype == "delete":
        adj[i, j] = 0
    elif mtype == "reverse":
        adj[i, j] = 0
        adj[j, i] = 1


def _revert_move(adj: np.ndarray, move: Tuple[str, int, int]) -> None:
    mtype, i, j = move
    if mtype == "add":
        adj[i, j] = 0
    elif mtype == "delete":
        adj[i, j] = 1
    elif mtype == "reverse":
        adj[j, i] = 0
        adj[i, j] = 1


# ---------------------------------------------------------------------------
# Single-chain MCMC
# ---------------------------------------------------------------------------


def _run_chain(
    obs_beta: np.ndarray,
    obs_se: np.ndarray,
    obs_mask: np.ndarray,
    b_plugin: np.ndarray,
    log_bf_direct: np.ndarray,
    allowed: np.ndarray,
    pi0: float,
    W: float,
    n_iter: int,
    n_burn: int,
    thin: int,
    rng: np.random.Generator,
    eps2: float = 0.0,
):
    n = obs_beta.shape[0]
    adj = np.zeros((n, n), dtype=np.int8)

    cur_lp = _log_posterior(
        adj, b_plugin, obs_beta, obs_se, obs_mask, log_bf_direct, allowed, pi0, W, eps2
    )
    cur_legal = _legal_moves(adj, allowed)
    cur_nmoves = len(cur_legal)

    samples: List[np.ndarray] = []
    lp_sum = 0.0
    lp_n = 0
    n_accept = 0
    n_prop = 0

    for it in range(n_iter):
        if cur_nmoves == 0:
            if it >= n_burn and ((it - n_burn) % thin == 0):
                samples.append(adj.copy())
                lp_sum += cur_lp
                lp_n += 1
            continue

        idx = int(rng.integers(0, cur_nmoves))
        move = cur_legal[idx]

        _apply_move(adj, move)
        new_legal = _legal_moves(adj, allowed)
        new_nmoves = len(new_legal)
        new_lp = _log_posterior(
            adj,
            b_plugin,
            obs_beta,
            obs_se,
            obs_mask,
            log_bf_direct,
            allowed,
            pi0,
            W,
            eps2,
        )

        log_alpha = (new_lp - cur_lp) + np.log(cur_nmoves) - np.log(new_nmoves)
        log_u = np.log(rng.random() + 1e-300)
        n_prop += 1
        if log_u < log_alpha:
            cur_lp = new_lp
            cur_legal = new_legal
            cur_nmoves = new_nmoves
            n_accept += 1
        else:
            _revert_move(adj, move)

        if it >= n_burn and ((it - n_burn) % thin == 0):
            samples.append(adj.copy())
            lp_sum += cur_lp
            lp_n += 1

    if samples:
        sample_arr = np.stack(samples, axis=0).astype(np.int8)
    else:
        sample_arr = np.zeros((0, n, n), dtype=np.int8)
    accept = n_accept / max(n_prop, 1)
    mean_lp = lp_sum / max(lp_n, 1)
    return sample_arr, accept, mean_lp


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat on per-edge indicators
# ---------------------------------------------------------------------------


def _rhat_per_edge(chain_samples: List[np.ndarray]) -> np.ndarray:
    m = len(chain_samples)
    if m < 2:
        n = chain_samples[0].shape[1] if chain_samples else 0
        return np.ones((n, n))
    S = min(c.shape[0] for c in chain_samples)
    stacked = np.stack([c[:S] for c in chain_samples], axis=0).astype(float)
    chain_means = stacked.mean(axis=1)
    overall = chain_means.mean(axis=0)
    B = (S / (m - 1.0)) * np.sum((chain_means - overall) ** 2, axis=0)
    chain_vars = stacked.var(axis=1, ddof=1)
    Wv = chain_vars.mean(axis=0)
    var_hat = ((S - 1.0) / S) * Wv + B / S
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(var_hat / Wv)
    rhat = np.where(np.isfinite(rhat), rhat, 1.0)
    return rhat


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_mrdag(
    gwas,
    nodes: Sequence[str] = NODE_NAMES,
    rng: Optional[np.random.Generator] = None,
    n_iter: int = 6000,
    n_chains: int = 4,
    n_burn: int = 1000,
    thin: int = 5,
    prior_incl: float = 0.05,
    prior_effect_var: float = 0.25**2,
    min_effect_scale: float = 0.075,
    **kwargs,
) -> MrDAGResult:
    """Run MrDAG on an IVW summary-statistics object.

    Accepts either a simulated ``GWASSummary`` or a literature-based
    ``RealGWASSummary``; both expose ``exposures``, ``outcomes``,
    ``betas``, ``ses``, ``n_snps``.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    p = len(nodes)
    node_index = {name: i for i, name in enumerate(nodes)}

    # ------------------------------------------------------------------
    # 1) MR trait set + candidate edge mask.
    # ------------------------------------------------------------------
    mr_traits = tuple(dict.fromkeys(list(gwas.exposures) + list(gwas.outcomes)))
    mr_idx = np.array([node_index[t] for t in mr_traits], dtype=int)
    in_mr = np.zeros(p, dtype=bool)
    in_mr[mr_idx] = True

    obs_beta = np.full((p, p), np.nan, dtype=float)
    obs_se = np.full((p, p), np.nan, dtype=float)
    for a, exp in enumerate(gwas.exposures):
        for b, out in enumerate(gwas.outcomes):
            if exp == out:
                continue
            be = gwas.betas[a, b]
            se = gwas.ses[a, b]
            if not (np.isfinite(be) and np.isfinite(se)) or se <= 0.0:
                continue
            i = node_index[exp]
            j = node_index[out]
            obs_beta[i, j] = be
            obs_se[i, j] = se

    obs_mask = np.isfinite(obs_beta) & np.isfinite(obs_se) & (obs_se > 0)
    allowed = obs_mask.copy()

    W = float(prior_effect_var)
    eps2 = float(min_effect_scale) ** 2

    # Plug-in posterior-mean weights per candidate edge.
    b_plugin = np.zeros((p, p), dtype=float)
    log_bf_direct = np.zeros((p, p), dtype=float)
    for i, j in np.argwhere(obs_mask):
        i = int(i)
        j = int(j)
        b_plugin[i, j] = _posterior_mean_weight(obs_beta[i, j], obs_se[i, j], W)
        log_bf_direct[i, j] = _wakefield_log_bf(obs_beta[i, j], obs_se[i, j], W, eps2)

    # Replace NaN entries in obs arrays with safe zeros (obs_mask protects them).
    obs_beta_clean = np.where(obs_mask, obs_beta, 0.0)
    obs_se_clean = np.where(obs_mask, obs_se, 1.0)

    # ------------------------------------------------------------------
    # 2) Multi-chain MCMC.
    # ------------------------------------------------------------------
    if hasattr(rng, "spawn"):
        child_rngs = rng.spawn(n_chains)
    else:
        seeds = rng.integers(0, 2**31 - 1, size=n_chains)
        child_rngs = [np.random.default_rng(int(s)) for s in seeds]

    chain_samples: List[np.ndarray] = []
    accept_rates: List[float] = []
    mean_lps: List[float] = []
    for c in range(n_chains):
        samples, acc, mlp = _run_chain(
            obs_beta_clean,
            obs_se_clean,
            obs_mask,
            b_plugin,
            log_bf_direct,
            allowed,
            pi0=prior_incl,
            W=W,
            n_iter=n_iter,
            n_burn=n_burn,
            thin=thin,
            rng=child_rngs[c],
            eps2=eps2,
        )
        chain_samples.append(samples)
        accept_rates.append(acc)
        mean_lps.append(mlp)

    chain_probs = [
        c.mean(axis=0) if c.shape[0] > 0 else np.zeros((p, p)) for c in chain_samples
    ]

    total = sum(c.shape[0] for c in chain_samples)
    if total > 0:
        pi_mr = np.concatenate(chain_samples, axis=0).mean(axis=0)
    else:
        pi_mr = np.zeros((p, p))

    max_abs_diff = 0.0
    if len(chain_probs) >= 2:
        for a in range(len(chain_probs)):
            for b in range(a + 1, len(chain_probs)):
                d = np.abs(chain_probs[a] - chain_probs[b])[allowed]
                if d.size:
                    max_abs_diff = max(max_abs_diff, float(d.max()))

    rhat = _rhat_per_edge(chain_samples)
    max_rhat_on_allowed = float(rhat[allowed].max()) if allowed.any() else 1.0

    # ------------------------------------------------------------------
    # 3) Assemble (p, p) pi in NODE_NAMES order.
    # ------------------------------------------------------------------
    pi = np.full((p, p), np.nan, dtype=float)
    mr_mask = np.outer(in_mr, in_mr)
    pi[mr_mask] = 0.0
    pi[allowed] = pi_mr[allowed]
    np.fill_diagonal(pi, 0.0)

    diagnostics = {
        "accept_rates": accept_rates,
        "mean_log_posterior_per_chain": mean_lps,
        "between_chain_max_abs_diff": max_abs_diff,
        "rhat_per_edge": rhat,
        "max_rhat_on_allowed": max_rhat_on_allowed,
        "n_candidate_edges": int(allowed.sum()),
        "mr_traits": mr_traits,
        "prior_incl": prior_incl,
        "prior_effect_var": prior_effect_var,
        "min_effect_scale": float(min_effect_scale),
        "n_iter": n_iter,
        "n_burn": n_burn,
        "thin": thin,
        "n_samples_per_chain": [int(c.shape[0]) for c in chain_samples],
    }

    return MrDAGResult(
        pi=pi,
        nodes=tuple(nodes),
        n_chains=n_chains,
        diagnostics=diagnostics,
    )


__all__ = ["MrDAGResult", "run_mrdag"]
