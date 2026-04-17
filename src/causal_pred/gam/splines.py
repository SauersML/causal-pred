# legacy reference -- survival.py uses the gam Python library (PyO3 binding
# to SauersML/gam) as its backend.  This hand-rolled P-spline module is kept
# for documentation and offline experiments; it is NOT imported on the
# default survival GAM path.
"""Penalised cubic B-spline (P-spline) bases with Wood-2011 reparametrisation.

This module is isolated from the NUTS sampler and the survival likelihood so
that it can be unit-tested on its own.  Given a 1-D covariate ``x``, we build:

    * a cubic B-spline basis ``B(x)`` with ``K`` interior knots placed at the
      empirical quantiles of ``x`` (Eilers & Marx 1996, "Flexible smoothing
      with B-splines and penalties", Stat. Sci. 11(2):89-121),
    * the second-order difference operator ``D`` acting on spline
      coefficients, and the corresponding penalty matrix ``P = D' D``.

The naive P-spline prior ``beta ~ N(0, tau^2 P^+)`` is rank-deficient: ``P``
has a null space (the constant and linear vectors for a second-difference
penalty).  Wood (2011, JRSS-B 73(1):3-36, "Fast stable restricted maximum
likelihood and marginal likelihood estimation of semiparametric generalized
linear models") recommends reparametrising so that the null space is
handled as a set of unpenalised fixed effects and the range space becomes a
Gaussian random effect with scalar precision.  We implement that reparam
here; the survival GAM then puts a Gaussian prior on the "random" block
only, absorbing the constant null-space column into the intercept and the
linear null-space column into an ordinary (weakly-penalised) slope.

The centring constraint ``sum(B @ beta) = 0`` used by ``mgcv`` to
identify the smooth is applied *after* the reparametrisation so the
unpenalised fixed-effect column is orthogonal to the intercept.  The
reparametrised design we return therefore has:

    Z = [ z_lin | Z_ran ]            # shape (n, K+3)  after centring
    beta = [ b_lin ; b_ran ]         # first column is the unpenalised
                                      # linear null-space direction,
                                      # the remaining columns have prior
                                      # b_ran ~ N(0, tau^2 I).

This is the standard "mixed-model form" of a P-spline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# B-spline basis
# ---------------------------------------------------------------------------


def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Return (n, n_basis) B-spline design matrix using Cox-de Boor recursion.

    ``knots`` must be the full augmented knot vector (length n_basis + degree + 1).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    n_basis = len(t) - degree - 1
    n = x.shape[0]

    # Degree-0 (piecewise constant) basis: B_{i,0}(x) = 1 if t_i <= x < t_{i+1}
    np.zeros((n, n_basis + degree), dtype=float)
    # We'll iterate up to the required degree; start by creating the full
    # zero-degree basis on len(t)-1 intervals, then recurse.
    n_intervals = len(t) - 1
    B0 = np.zeros((n, n_intervals), dtype=float)
    for i in range(n_intervals):
        left = t[i]
        right = t[i + 1]
        if right <= left:
            continue
        mask = (x >= left) & (x < right)
        B0[mask, i] = 1.0
    # Include the right endpoint in the last non-degenerate interval
    # (otherwise x == knots[-1] evaluates to all zeros).
    last_valid = -1
    for i in range(n_intervals):
        if t[i + 1] > t[i]:
            last_valid = i
    if last_valid >= 0:
        B0[x >= t[last_valid + 1], last_valid] = 1.0

    B_prev = B0
    for d in range(1, degree + 1):
        n_cur = n_intervals - d
        B_cur = np.zeros((n, n_cur), dtype=float)
        for i in range(n_cur):
            denom1 = t[i + d] - t[i]
            denom2 = t[i + d + 1] - t[i + 1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 0:
                term1 = (x - t[i]) / denom1 * B_prev[:, i]
            if denom2 > 0:
                term2 = (t[i + d + 1] - x) / denom2 * B_prev[:, i + 1]
            B_cur[:, i] = term1 + term2
        B_prev = B_cur
    return B_prev  # shape (n, n_basis)


def build_knots(x: np.ndarray, n_interior: int, degree: int) -> np.ndarray:
    """Place ``n_interior`` interior knots at empirical quantiles of ``x``.

    A cubic B-spline basis with ``K`` interior knots has ``K + degree + 1``
    basis functions once the boundary knots are repeated ``degree + 1`` times.
    Following Eilers & Marx we use interior knots at equispaced quantiles;
    this behaves much better than equispaced knots when ``x`` has a
    non-uniform empirical distribution.
    """
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if n_interior <= 0:
        interior = np.array([], dtype=float)
    else:
        qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        interior = np.quantile(x, qs)
    # small pad so that x lies strictly inside the knot range
    pad = 1e-6 * max(hi - lo, 1.0)
    lo_b = lo - pad
    hi_b = hi + pad
    # repeat boundary knots degree+1 times
    knots = np.concatenate(
        [
            np.full(degree + 1, lo_b),
            interior,
            np.full(degree + 1, hi_b),
        ]
    )
    return knots


def difference_matrix(n: int, order: int = 2) -> np.ndarray:
    """Return the ``order``-th forward-difference operator of shape (n-order, n)."""
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


# ---------------------------------------------------------------------------
# Wood-2011 mixed-model reparametrisation
# ---------------------------------------------------------------------------


@dataclass
class SmoothSpec:
    """Reparametrised smooth term for a single continuous covariate.

    Attributes
    ----------
    Z : (n, m) ndarray
        The reparametrised design matrix after mean-centring and
        null-space separation.  The first column ``Z[:, 0]`` corresponds
        to the *unpenalised* linear null-space direction; the remaining
        ``m - 1`` columns correspond to the penalised (random-effect)
        basis on which we put the Gaussian prior ``N(0, tau^2 I)``.
    knots : ndarray
        The augmented knot vector (for use by ``evaluate``).
    degree : int
        Spline degree (typically 3).
    U_pen : (n_basis, m-1) ndarray
        Eigenvector block corresponding to the range space of ``P``,
        scaled by 1/sqrt(eigenvalue), so ``Z_ran = B_centred @ U_pen``.
    U_nul : (n_basis, 1) ndarray
        Eigenvector block corresponding to the (non-constant part of the)
        null space of ``P``, so ``Z_lin = B_centred @ U_nul``.
    constraint : (n_basis,) ndarray
        Centring constraint vector ``c`` such that ``c' beta = 0``.  We
        absorb it via a QR-style projection and return ``B_centred``.
    mean : float
        Mean of the raw covariate (for standardisation).
    std : float
        Std of the raw covariate.
    """

    Z: np.ndarray
    knots: np.ndarray
    degree: int
    U_pen: np.ndarray
    U_nul: np.ndarray
    Q_constraint: np.ndarray  # (n_basis, n_basis-1) absorbing matrix
    mean: float
    std: float

    @property
    def n_unpen(self) -> int:
        """Number of unpenalised columns (linear null-space directions)."""
        return self.U_nul.shape[1]

    @property
    def n_pen(self) -> int:
        return self.U_pen.shape[1]


def build_smooth(
    x: np.ndarray, n_interior_knots: int = 10, degree: int = 3, penalty_order: int = 2
) -> SmoothSpec:
    """Construct the reparametrised P-spline smooth for a 1-D covariate.

    Implements the mixed-model reparametrisation of Wood (2011): eigen-
    decompose the penalty ``P = D' D``, split eigenvectors into null-space
    (eigenvalue zero) and range-space blocks, and transform coefficients
    so the random-effect block has precision matrix ``tau^{-2} I``.  The
    sum-to-zero centring constraint ``1' B beta = 0`` is applied first so
    the constant null-space direction coincides with the model intercept
    and is dropped.
    """
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-12:
        std = 1.0
    xs = (x - mean) / std

    knots = build_knots(xs, n_interior_knots, degree)
    B = _bspline_basis(xs, knots, degree)  # (n, K)
    K = B.shape[1]

    # Sum-to-zero constraint: c = colmeans(B) (so 1'B beta = 0 with beta in the
    # reduced basis).  We absorb it by QR of c so Q has K-1 columns orthogonal
    # to c, and work with B_c = B @ Q.
    c = B.sum(axis=0) / B.shape[0]  # (K,)
    # Householder: build orthonormal basis of c-perp.
    c_norm = c / np.linalg.norm(c)
    identity = np.eye(K)
    H = identity - 2.0 * np.outer(c_norm, c_norm)  # reflector (K, K)
    # Columns 1..K-1 of H span c-perp.  (Column 0 spans span(c).)
    Q = H[:, 1:]  # (K, K-1)
    B_c = B @ Q  # (n, K-1)

    # Build penalty in the constrained space: P_c = Q' P Q.
    D = difference_matrix(K, order=penalty_order)  # (K-order, K)
    P_full = D.T @ D
    P_c = Q.T @ P_full @ Q  # (K-1, K-1)

    # Eigendecompose P_c.  Separate null space (eigenvalue ~ 0) from range.
    evals, evecs = np.linalg.eigh(P_c)
    # Numerical tolerance: relative to the largest eigenvalue.
    tol = max(evals[-1], 1.0) * 1e-8
    nul_mask = evals < tol
    pen_mask = ~nul_mask

    # Null-space block: unpenalised, typically 1 column for order-2 penalty
    # after the constant has been absorbed (only the linear direction
    # remains).
    U_nul_raw = evecs[:, nul_mask]  # (K-1, n_unpen)
    # Range-space block: scale by 1/sqrt(eval) so prior on the new coeffs
    # is N(0, tau^2 I) with beta_ran = sqrt(eval) * coef (mgcv-style).
    sqrt_lam = np.sqrt(evals[pen_mask])
    U_pen_raw = evecs[:, pen_mask] / sqrt_lam[np.newaxis, :]  # (K-1, n_pen)

    # Design-matrix columns (un-normalised scale):
    Z_nul = B_c @ U_nul_raw  # (n, n_unpen)
    Z_pen = B_c @ U_pen_raw  # (n, n_pen)
    Z = np.concatenate([Z_nul, Z_pen], axis=1)  # (n, K-1)

    return SmoothSpec(
        Z=Z,
        knots=knots,
        degree=degree,
        U_pen=U_pen_raw,
        U_nul=U_nul_raw,
        Q_constraint=Q,
        mean=mean,
        std=std,
    )


def evaluate_smooth(spec: SmoothSpec, x_new: np.ndarray) -> np.ndarray:
    """Return the reparametrised design matrix ``Z_new`` for new ``x_new``.

    Uses the standardisation, centring, and null-space/range-space
    decomposition stored in ``spec`` so the output lines up column-for-
    column with the one used at fit time.
    """
    x_new = np.asarray(x_new, dtype=float)
    xs = (x_new - spec.mean) / spec.std
    B = _bspline_basis(xs, spec.knots, spec.degree)
    B_c = B @ spec.Q_constraint
    Z_nul = B_c @ spec.U_nul
    Z_pen = B_c @ spec.U_pen
    return np.concatenate([Z_nul, Z_pen], axis=1)


def fit_pspline_ridge(
    x: np.ndarray,
    y: np.ndarray,
    n_interior_knots: int = 10,
    degree: int = 3,
    penalty_order: int = 2,
    lam: float = 1.0,
) -> Tuple[np.ndarray, SmoothSpec, float]:
    """Maximum-a-posteriori P-spline fit by penalised least squares.

    Minimises ``||y - intercept - Z @ beta||^2 + lam * ||beta_pen||^2``
    where ``beta_pen`` is the vector of penalised coefficients in the
    mixed-model parametrisation.  Useful for unit-testing the spline
    module independently of NUTS.
    """
    spec = build_smooth(x, n_interior_knots, degree, penalty_order)
    n = spec.Z.shape[0]
    # Append an intercept column.
    Xdes = np.concatenate([np.ones((n, 1)), spec.Z], axis=1)
    p = Xdes.shape[1]
    # Penalty: zero on intercept + n_unpen cols, lam on the rest.
    d = np.zeros(p)
    d[1 + spec.n_unpen :] = lam
    A = Xdes.T @ Xdes + np.diag(d)
    coef = np.linalg.solve(A, Xdes.T @ y)
    intercept = coef[0]
    beta = coef[1:]
    return beta, spec, intercept


__all__ = [
    "SmoothSpec",
    "build_smooth",
    "evaluate_smooth",
    "build_knots",
    "difference_matrix",
    "fit_pspline_ridge",
]
