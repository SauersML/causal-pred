# legacy reference -- survival.py uses the gam Python library (PyO3 binding
# to SauersML/gam) as its backend.  This from-scratch NUTS is kept for
# documentation and offline experiments; it is NOT imported on the default
# survival GAM path.
"""From-scratch No-U-Turn Sampler (Hoffman & Gelman 2014).

Implements the recursive-doubling, multinomial-weighted variant of NUTS
(Algorithm 5 of `arxiv:1111.4246v2`), with:

  * a leapfrog integrator in a diagonal mass-matrix metric,
  * dual-averaging step-size adaptation (Algorithm 6) during a warm-up
    phase, with target acceptance ``delta``,
  * Stan-style windowed diagonal mass-matrix adaptation: running mean
    and variance of posterior samples are accumulated in expanding
    windows, the diagonal mass matrix is set to the sample variance at
    the end of each window, and the step size is re-initialised,
  * divergence detection: a leapfrog step with log-weight ratio below
    ``-1000`` (Stan default) is flagged and terminates the current
    trajectory.

The sampler is deliberately numpy-only: ``log_posterior(q) -> (logp,
grad_logp)`` is the sole dependency on the target model, and it is
user-supplied.  All adaptation is deterministic given an RNG seed.

References
----------
Hoffman, M. D. & Gelman, A. (2014).  The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.  JMLR 15:1593-1623.

Stan Development Team (2016).  Stan Reference Manual, section on
adaptation windows and diagonal mass matrix estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Welford online mean/variance (for mass-matrix adaptation)
# ---------------------------------------------------------------------------


class WelfordEstimator:
    """Running diagonal variance estimator (Welford algorithm)."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def add(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self, regularise: bool = True) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        var = self.M2 / (self.n - 1)
        if regularise:
            # Stan's regularised covariance: shrink toward identity for
            # short windows.  See Stan Reference Manual; this is the
            # standard biased shrinkage estimator used in the default
            # adaptation.
            w = self.n / (self.n + 5.0)
            var = w * var + (1.0 - w) * 1e-3 * np.ones_like(var)
        var = np.maximum(var, 1e-8)
        return var

    def reset(self) -> None:
        self.n = 0
        self.mean[:] = 0.0
        self.M2[:] = 0.0


# ---------------------------------------------------------------------------
# NUTS state
# ---------------------------------------------------------------------------


@dataclass
class NUTSState:
    q: np.ndarray  # position (parameters)
    logp: float
    grad: np.ndarray
    step_size: float = 0.1
    mass_inv: np.ndarray = field(default=None)  # diagonal M^{-1}
    # dual-averaging bookkeeping
    mu: float = 0.0
    log_eps_bar: float = 0.0
    H_bar: float = 0.0

    def __post_init__(self):
        if self.mass_inv is None:
            self.mass_inv = np.ones_like(self.q)


# ---------------------------------------------------------------------------
# Leapfrog
# ---------------------------------------------------------------------------


def leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    eps: float,
    mass_inv: np.ndarray,
    logp_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
):
    """One leapfrog step of size ``eps`` for Hamiltonian dynamics with
    diagonal mass matrix ``M``.

        p_half = p + 0.5 * eps * grad_logp(q)
        q_new  = q + eps * M^{-1} p_half
        p_new  = p_half + 0.5 * eps * grad_logp(q_new)

    Returns ``(q_new, p_new, logp_new, grad_new)``.
    """
    p_half = p + 0.5 * eps * grad
    q_new = q + eps * mass_inv * p_half
    logp_new, grad_new = logp_and_grad(q_new)
    p_new = p_half + 0.5 * eps * grad_new
    return q_new, p_new, logp_new, grad_new


# ---------------------------------------------------------------------------
# Heuristic initial step size (Algorithm 4 of Hoffman & Gelman 2014)
# ---------------------------------------------------------------------------


def find_reasonable_step_size(
    q: np.ndarray,
    logp: float,
    grad: np.ndarray,
    mass_inv: np.ndarray,
    logp_and_grad,
    rng: np.random.Generator,
    eps: float = 1.0,
) -> float:
    """Double/halve ``eps`` until one leapfrog step changes the
    log joint by ``log 2`` — Hoffman & Gelman 2014 Alg. 4.
    """
    mass = 1.0 / mass_inv
    p = rng.standard_normal(q.shape[0]) * np.sqrt(mass)
    H0 = logp - 0.5 * np.sum(p * p * mass_inv)
    q_new, p_new, logp_new, grad_new = leapfrog(
        q, p, grad, eps, mass_inv, logp_and_grad
    )
    H_new = logp_new - 0.5 * np.sum(p_new * p_new * mass_inv)
    log_ratio = H_new - H0
    if not np.isfinite(log_ratio):
        log_ratio = -np.inf
    a = 1.0 if log_ratio > np.log(0.5) else -1.0
    for _ in range(100):
        if a * log_ratio <= -a * np.log(2.0):
            break
        eps = eps * (2.0**a)
        q_new, p_new, logp_new, grad_new = leapfrog(
            q, p, grad, eps, mass_inv, logp_and_grad
        )
        H_new = logp_new - 0.5 * np.sum(p_new * p_new * mass_inv)
        log_ratio = H_new - H0
        if not np.isfinite(log_ratio):
            log_ratio = -np.inf
    return eps


# ---------------------------------------------------------------------------
# NUTS recursion (multinomial / Algorithm 5)
# ---------------------------------------------------------------------------

_MAX_EXPONENT = 700.0  # guard against overflow in exp()


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) in a numerically stable way."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


@dataclass
class _SubTree:
    q_minus: np.ndarray
    p_minus: np.ndarray
    grad_minus: np.ndarray
    q_plus: np.ndarray
    p_plus: np.ndarray
    grad_plus: np.ndarray
    q_prop: np.ndarray
    logp_prop: float
    grad_prop: np.ndarray
    log_w: float  # log total weight in subtree (for multinomial pick)
    n_leapfrog: int
    s: bool  # continue flag (not U-turn, not divergent)
    sum_accept: float
    n_alpha: int
    diverged: bool


def _uturn(
    q_minus: np.ndarray,
    q_plus: np.ndarray,
    p_minus: np.ndarray,
    p_plus: np.ndarray,
    mass_inv: np.ndarray,
) -> bool:
    """Standard U-turn condition (Hoffman & Gelman 2014 eq. 9).

    Returns True if we've turned (i.e. should stop).
    """
    dq = q_plus - q_minus
    # dot products with momenta scaled by M^{-1} (velocity direction).
    v_minus = mass_inv * p_minus
    v_plus = mass_inv * p_plus
    return (np.dot(dq, v_minus) < 0.0) or (np.dot(dq, v_plus) < 0.0)


def _build_tree(
    q: np.ndarray,
    p: np.ndarray,
    grad: np.ndarray,
    logp: float,
    log_u: float,
    v: int,
    j: int,
    eps: float,
    H0: float,
    mass_inv: np.ndarray,
    logp_and_grad,
    rng: np.random.Generator,
    max_delta_h: float = 1000.0,
) -> _SubTree:
    """Recursively build one side of the NUTS tree (multinomial sampling).

    ``log_u`` is passed through for API compatibility but the
    multinomial variant uses accumulated log-weights instead of the
    slice indicator; we keep it ``-inf`` here.  The Hamiltonian
    divergence threshold is ``max_delta_h`` (Stan default 1000).
    """
    if j == 0:
        # Base case: one leapfrog step in direction v.
        q_new, p_new, logp_new, grad_new = leapfrog(
            q, p, grad, v * eps, mass_inv, logp_and_grad
        )
        if not np.isfinite(logp_new):
            diverged = True
            log_w = -np.inf
            alpha = 0.0
            s = False
            return _SubTree(
                q_minus=q_new,
                p_minus=p_new,
                grad_minus=grad_new,
                q_plus=q_new,
                p_plus=p_new,
                grad_plus=grad_new,
                q_prop=q_new,
                logp_prop=logp_new,
                grad_prop=grad_new,
                log_w=log_w,
                n_leapfrog=1,
                s=s,
                sum_accept=alpha,
                n_alpha=1,
                diverged=True,
            )
        H_new = logp_new - 0.5 * np.sum(p_new * p_new * mass_inv)
        log_w = H_new - H0  # multinomial weight
        delta_h = H0 - H_new
        diverged = not np.isfinite(delta_h) or (delta_h > max_delta_h)
        s = not diverged
        alpha = min(1.0, np.exp(min(log_w, 0.0))) if np.isfinite(log_w) else 0.0
        return _SubTree(
            q_minus=q_new,
            p_minus=p_new,
            grad_minus=grad_new,
            q_plus=q_new,
            p_plus=p_new,
            grad_plus=grad_new,
            q_prop=q_new,
            logp_prop=logp_new,
            grad_prop=grad_new,
            log_w=log_w,
            n_leapfrog=1,
            s=s,
            sum_accept=alpha,
            n_alpha=1,
            diverged=diverged,
        )

    # Recurse: build left subtree.
    left = _build_tree(
        q,
        p,
        grad,
        logp,
        log_u,
        v,
        j - 1,
        eps,
        H0,
        mass_inv,
        logp_and_grad,
        rng,
        max_delta_h,
    )
    if not left.s:
        return left
    # Build right subtree from the appropriate endpoint.
    if v == -1:
        right = _build_tree(
            left.q_minus,
            left.p_minus,
            left.grad_minus,
            left.logp_prop,
            log_u,
            v,
            j - 1,
            eps,
            H0,
            mass_inv,
            logp_and_grad,
            rng,
            max_delta_h,
        )
        q_minus, p_minus, grad_minus = right.q_minus, right.p_minus, right.grad_minus
        q_plus, p_plus, grad_plus = left.q_plus, left.p_plus, left.grad_plus
    else:
        right = _build_tree(
            left.q_plus,
            left.p_plus,
            left.grad_plus,
            left.logp_prop,
            log_u,
            v,
            j - 1,
            eps,
            H0,
            mass_inv,
            logp_and_grad,
            rng,
            max_delta_h,
        )
        q_minus, p_minus, grad_minus = left.q_minus, left.p_minus, left.grad_minus
        q_plus, p_plus, grad_plus = right.q_plus, right.p_plus, right.grad_plus

    # Multinomial: pick proposal from the combined tree proportional to
    # exp(log_w).  If right diverged or is done, its weight is -inf.
    log_w_total = _log_add(left.log_w, right.log_w)
    if log_w_total == -np.inf:
        prob_right = 0.0
    else:
        # Stable: P(right) = exp(right.log_w - log_w_total)
        prob_right = np.exp(right.log_w - log_w_total)
        if not np.isfinite(prob_right):
            prob_right = 0.0
    if rng.random() < prob_right:
        q_prop = right.q_prop
        logp_prop = right.logp_prop
        grad_prop = right.grad_prop
    else:
        q_prop = left.q_prop
        logp_prop = left.logp_prop
        grad_prop = left.grad_prop

    s_new = (
        left.s and right.s and not _uturn(q_minus, q_plus, p_minus, p_plus, mass_inv)
    )
    sum_accept = left.sum_accept + right.sum_accept
    n_alpha = left.n_alpha + right.n_alpha
    diverged = left.diverged or right.diverged

    return _SubTree(
        q_minus=q_minus,
        p_minus=p_minus,
        grad_minus=grad_minus,
        q_plus=q_plus,
        p_plus=p_plus,
        grad_plus=grad_plus,
        q_prop=q_prop,
        logp_prop=logp_prop,
        grad_prop=grad_prop,
        log_w=log_w_total,
        n_leapfrog=left.n_leapfrog + right.n_leapfrog,
        s=s_new,
        sum_accept=sum_accept,
        n_alpha=n_alpha,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------


@dataclass
class NUTSResult:
    samples: np.ndarray  # (n_samples, dim)
    step_sizes: np.ndarray  # (n_samples,) final step size per draw
    tree_depths: np.ndarray  # (n_samples,)
    n_divergences: int
    final_step_size: float
    final_mass_inv: np.ndarray


def run_nuts(
    logp_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    q0: np.ndarray,
    n_samples: int,
    warmup: int,
    rng: np.random.Generator,
    target_accept: float = 0.8,
    max_tree_depth: int = 10,
    progress: bool = False,
    adapt_mass: bool = True,
) -> NUTSResult:
    """Run NUTS with dual-averaging and windowed mass-matrix adaptation.

    Parameters
    ----------
    logp_and_grad : callable
        Function mapping ``q -> (log p(q), grad log p(q))``.  Must
        return ``(-inf, any)`` for invalid points (the caller is
        responsible; the sampler treats a non-finite ``logp`` as a
        divergence).
    q0 : ndarray
        Initial position.
    n_samples : int
        Number of *post-warmup* samples.
    warmup : int
        Number of warm-up iterations used for dual-averaging and
        mass-matrix adaptation.
    """
    q = q0.astype(float).copy()
    dim = q.shape[0]
    logp, grad = logp_and_grad(q)
    if not np.isfinite(logp):
        raise ValueError("Initial log-posterior is not finite.")

    # Initial mass matrix (identity) and step size.
    mass_inv = np.ones(dim)
    eps = find_reasonable_step_size(q, logp, grad, mass_inv, logp_and_grad, rng)
    mu = np.log(10.0 * eps)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # Mass-matrix adaptation windows (Stan-style): warmup split as
    # init_buffer (15% <= 75), term_buffer (10% <= 50), windowed middle.
    init_buffer = min(75, max(3, int(0.15 * warmup)))
    term_buffer = min(50, max(3, int(0.10 * warmup)))
    if init_buffer + term_buffer >= warmup:
        init_buffer = max(1, warmup // 5)
        term_buffer = max(1, warmup // 5)
    win_start = init_buffer
    win_end = warmup - term_buffer
    # Expanding windows with doubling length.
    windows: List[Tuple[int, int]] = []
    cur = win_start
    length = 25
    while cur < win_end:
        nxt = min(cur + length, win_end)
        # Last window absorbs any remainder.
        if nxt + 2 * length > win_end:
            nxt = win_end
        windows.append((cur, nxt))
        cur = nxt
        length *= 2
    welford = WelfordEstimator(dim)
    window_idx = 0

    samples = np.zeros((n_samples, dim))
    step_sizes = np.zeros(n_samples)
    tree_depths = np.zeros(n_samples, dtype=int)
    n_divergences = 0
    n_divergences_warmup = 0

    total = warmup + n_samples
    for it in range(total):
        is_warmup = it < warmup

        # Resample momentum and evaluate Hamiltonian.
        mass = 1.0 / mass_inv
        p = rng.standard_normal(dim) * np.sqrt(mass)
        H0 = logp - 0.5 * np.sum(p * p * mass_inv)

        # Slice variable for multinomial sampler is unused; pass -inf.
        log_u = -np.inf
        q_minus = q.copy()
        p_minus = p.copy()
        grad_minus = grad.copy()
        q_plus = q.copy()
        p_plus = p.copy()
        grad_plus = grad.copy()
        q_prop = q.copy()
        logp_prop = logp
        grad_prop = grad.copy()
        log_w_total = 0.0  # log weight of the initial point is 0

        j = 0
        s = True
        this_diverged = False
        sum_accept = 0.0
        n_alpha = 0
        while s and j < max_tree_depth:
            v = 1 if rng.random() < 0.5 else -1
            if v == -1:
                sub = _build_tree(
                    q_minus,
                    p_minus,
                    grad_minus,
                    logp_prop,
                    log_u,
                    v,
                    j,
                    eps,
                    H0,
                    mass_inv,
                    logp_and_grad,
                    rng,
                )
                q_minus, p_minus, grad_minus = sub.q_minus, sub.p_minus, sub.grad_minus
            else:
                sub = _build_tree(
                    q_plus,
                    p_plus,
                    grad_plus,
                    logp_prop,
                    log_u,
                    v,
                    j,
                    eps,
                    H0,
                    mass_inv,
                    logp_and_grad,
                    rng,
                )
                q_plus, p_plus, grad_plus = sub.q_plus, sub.p_plus, sub.grad_plus

            # Accept subtree proposal with prob exp(sub.log_w - log_w_total)
            if sub.s:
                if log_w_total == -np.inf:
                    prob = 1.0
                elif sub.log_w == -np.inf:
                    prob = 0.0
                else:
                    prob = min(1.0, np.exp(sub.log_w - log_w_total))
                if rng.random() < prob:
                    q_prop = sub.q_prop
                    logp_prop = sub.logp_prop
                    grad_prop = sub.grad_prop
            log_w_total = _log_add(log_w_total, sub.log_w)

            sum_accept += sub.sum_accept
            n_alpha += sub.n_alpha
            this_diverged = this_diverged or sub.diverged

            s = sub.s and not _uturn(q_minus, q_plus, p_minus, p_plus, mass_inv)
            j += 1

        # Accept the proposal drawn from the whole tree.
        q = q_prop
        logp = logp_prop
        grad = grad_prop

        if this_diverged:
            if is_warmup:
                n_divergences_warmup += 1
            else:
                n_divergences += 1

        # --- adaptation ---
        if is_warmup:
            if n_alpha == 0:
                alpha_bar = 0.0
            else:
                alpha_bar = sum_accept / n_alpha
            # Dual averaging (Hoffman & Gelman Alg. 6).
            m = it + 1
            H_bar = (1.0 - 1.0 / (m + t0)) * H_bar + (1.0 / (m + t0)) * (
                target_accept - alpha_bar
            )
            log_eps = mu - (np.sqrt(m) / gamma) * H_bar
            eps = float(np.exp(log_eps))
            eta = m ** (-kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar

            # Mass-matrix adaptation: accumulate samples only in the
            # "middle" portion (between init_buffer and warmup - term_buffer).
            if adapt_mass and win_start <= it < win_end:
                welford.add(q)
                # Check if we're at the end of the current window.
                if window_idx < len(windows) and it + 1 == windows[window_idx][1]:
                    new_var = welford.variance(regularise=True)
                    mass_inv = new_var
                    welford.reset()
                    window_idx += 1
                    # Re-init step size after mass-matrix update.
                    try:
                        eps = find_reasonable_step_size(
                            q, logp, grad, mass_inv, logp_and_grad, rng
                        )
                    except Exception:
                        pass
                    mu = np.log(10.0 * eps)
                    H_bar = 0.0
                    log_eps_bar = 0.0
        else:
            if it == warmup:
                # Freeze step size at the dual-averaging smoothed value.
                eps = float(np.exp(log_eps_bar))
            idx = it - warmup
            samples[idx] = q
            step_sizes[idx] = eps
            tree_depths[idx] = j

        if progress and (it % max(1, total // 20) == 0):
            phase = "warmup" if is_warmup else "sample"
            print(f"[nuts] {phase} {it}/{total} eps={eps:.4f} depth={j}")

    return NUTSResult(
        samples=samples,
        step_sizes=step_sizes,
        tree_depths=tree_depths,
        n_divergences=n_divergences,
        final_step_size=float(eps),
        final_mass_inv=mass_inv,
    )


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat and effective sample size
# ---------------------------------------------------------------------------


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    """Split-R-hat for a stack of chains, shape (n_chains, n_samples, dim).

    Uses the classical (non-rank-normalised) split-R-hat: each chain is
    split in half, then the potential scale reduction factor is computed.
    """
    chains = np.asarray(chains)
    if chains.ndim == 2:
        chains = chains[:, :, None]
    n_chains, n_samp, dim = chains.shape
    # Split each chain.
    half = n_samp // 2
    if half < 2:
        return np.full(dim, np.nan)
    a = chains[:, :half, :]
    b = chains[:, half : 2 * half, :]
    split = np.concatenate([a, b], axis=0)  # (2 * n_chains, half, dim)
    split.shape[0]
    n = split.shape[1]
    chain_mean = split.mean(axis=1)  # (m, dim)
    chain_var = split.var(axis=1, ddof=1)  # (m, dim)
    W = chain_var.mean(axis=0)
    B = n * chain_mean.var(axis=0, ddof=1)
    var_hat = (1 - 1.0 / n) * W + B / n
    out = np.sqrt(var_hat / np.where(W < 1e-12, 1e-12, W))
    return out


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Effective sample size per Geyer's initial monotone-sequence estimator
    applied to each parameter, across all chains.

    Input: ``(n_chains, n_samples, dim)``.  Output: ``(dim,)``.
    """
    chains = np.asarray(chains)
    if chains.ndim == 2:
        chains = chains[:, :, None]
    n_chains, n_samp, dim = chains.shape
    out = np.zeros(dim)
    for d in range(dim):
        x = chains[:, :, d]
        n = x.shape[1]
        if n < 4:
            out[d] = n * n_chains
            continue
        # Pool across chains: compute mean within-chain autocorrelations.
        chain_mean = x.mean(axis=1, keepdims=True)
        chain_var = x.var(axis=1, ddof=1, keepdims=True)
        W = chain_var.mean()
        if W < 1e-12:
            out[d] = n * n_chains
            continue
        # Autocov up to n-1 lags, averaged across chains.
        rho = np.zeros(n)
        for c in range(n_chains):
            xc = x[c] - chain_mean[c, 0]
            # Use FFT-based autocorrelation.
            fx = np.fft.rfft(np.concatenate([xc, np.zeros_like(xc)]))
            acov = np.fft.irfft(fx * np.conj(fx))[:n].real / n
            rho += acov / n_chains
        rho /= rho[0] if rho[0] != 0 else 1.0
        # Sum of pairs until negative (Geyer initial positive).
        tau = 1.0
        for t in range(1, n - 1, 2):
            s = rho[t] + rho[t + 1]
            if s < 0:
                break
            tau += 2.0 * s
        tau = max(tau, 1.0)
        out[d] = n_chains * n / tau
    return out


__all__ = [
    "NUTSResult",
    "run_nuts",
    "find_reasonable_step_size",
    "leapfrog",
    "gelman_rubin",
    "effective_sample_size",
    "WelfordEstimator",
]
