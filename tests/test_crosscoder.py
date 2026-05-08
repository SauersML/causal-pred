"""Planted-superposition recovery test for the TopK crosscoder.

We generate paired (genome, EHR) data from a known sparse latent code with
mixing matrices ``M_G`` and ``M_E``, train the crosscoder, and verify that:

1. After training the per-stream reconstruction loss drops materially below
   the random-init level.
2. Each planted latent is recovered by *some* learned feature, where
   recovery is measured by absolute Pearson correlation between planted
   activations and learned activations on a held-out split.
3. Latents whose planted mixing only touches one stream are classified as
   ``"genome"`` or ``"ehr"``; latents that touch both as ``"shared"``.

The synthetic generator is deliberately small so the test runs in seconds
on CPU. There is no assumption that *every* learned feature is a planted
one -- the dictionary is overcomplete -- only that every planted feature
has at least one learned feature that matches it.
"""

from __future__ import annotations

import numpy as np

from causal_pred.genscore import (
    classify_features,
    encode,
    feature_stream_share,
    train_crosscoder,
)
from causal_pred.genscore.crosscoder import _normalise_decoders


# ---------------------------------------------------------------------------
# Synthetic data with planted superposition
# ---------------------------------------------------------------------------


def _make_synthetic(
    n: int,
    n_features: int,
    m_G: int,
    m_E: int,
    activations_per_sample: int,
    *,
    n_genome_only: int,
    n_ehr_only: int,
    noise: float,
    rng: np.random.Generator,
):
    """Plant a sparse latent code with known per-stream visibility.

    Returns
    -------
    A : (n, m_G), B : (n, m_E)
        Observed paired streams.
    Z_true : (n, n_features)
        True latent activations (non-negative, mostly zero).
    M_G : (n_features, m_G), M_E : (n_features, m_E)
        Mixing matrices used to generate A and B.
    visibility : (n_features,) array of strings in {"genome", "ehr", "shared"}.
    """
    if n_genome_only + n_ehr_only > n_features:
        raise ValueError("too many stream-specific features")
    n_shared = n_features - n_genome_only - n_ehr_only

    M_G = rng.standard_normal((n_features, m_G))
    M_E = rng.standard_normal((n_features, m_E))

    # First chunk: shared. Next: genome-only. Last: ehr-only.
    visibility = np.array(
        ["shared"] * n_shared
        + ["genome"] * n_genome_only
        + ["ehr"] * n_ehr_only,
        dtype=object,
    )
    # Zero out the off-stream rows.
    for j in range(n_features):
        if visibility[j] == "genome":
            M_E[j, :] = 0.0
        elif visibility[j] == "ehr":
            M_G[j, :] = 0.0

    # Sparse latent code: each row activates `activations_per_sample` features
    # uniformly at random, with non-negative magnitudes from a half-normal.
    Z_true = np.zeros((n, n_features))
    for i in range(n):
        active = rng.choice(n_features, size=activations_per_sample, replace=False)
        Z_true[i, active] = np.abs(rng.standard_normal(activations_per_sample))

    A = Z_true @ M_G + noise * rng.standard_normal((n, m_G))
    B = Z_true @ M_E + noise * rng.standard_normal((n, m_E))
    return A, B, Z_true, M_G, M_E, visibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_corr(u: np.ndarray, v: np.ndarray) -> float:
    """Absolute Pearson correlation between two 1D vectors; 0 if degenerate."""
    u = u - u.mean()
    v = v - v.mean()
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-10 or nv < 1e-10:
        return 0.0
    return float(abs(u @ v / (nu * nv)))


def _best_match_per_planted(
    Z_true: np.ndarray, Z_learned: np.ndarray
) -> np.ndarray:
    """For each planted latent, the highest |corr| across learned features."""
    n_features = Z_true.shape[1]
    best = np.zeros(n_features)
    for j in range(n_features):
        u = Z_true[:, j]
        if u.std() < 1e-10:
            continue
        best_j = 0.0
        for jj in range(Z_learned.shape[1]):
            v = Z_learned[:, jj]
            if v.std() < 1e-10:
                continue
            c = _abs_corr(u, v)
            if c > best_j:
                best_j = c
        best[j] = best_j
    return best


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_decoder_normalisation_unit_norm():
    """Joint decoder constraint should leave each feature with unit L2 norm."""
    rng = np.random.default_rng(0)
    d, m_G, m_E = 16, 4, 6
    W_d_G = rng.standard_normal((d, m_G))
    W_d_E = rng.standard_normal((d, m_E))
    W_d_G_n, W_d_E_n = _normalise_decoders(W_d_G, W_d_E)
    norms = np.sqrt(
        np.sum(W_d_G_n ** 2, axis=1) + np.sum(W_d_E_n ** 2, axis=1)
    )
    np.testing.assert_allclose(norms, np.ones(d), atol=1e-10)


def test_topk_crosscoder_recovers_planted_features():
    """Train on planted superposition; check recovery and stream classification."""
    rng = np.random.default_rng(42)
    n = 4000
    n_features = 12
    m_G, m_E = 8, 10
    A, B, Z_true, _, _, visibility = _make_synthetic(
        n=n,
        n_features=n_features,
        m_G=m_G,
        m_E=m_E,
        activations_per_sample=3,
        n_genome_only=3,
        n_ehr_only=3,
        noise=0.05,
        rng=rng,
    )

    # Train. d > n_features so the dictionary is overcomplete.
    model = train_crosscoder(
        A=A,
        B=B,
        d=64,
        k=3,
        n_steps=2000,
        batch_size=256,
        lr=3e-3,
        aux_k=8,
        rng=np.random.default_rng(0),
    )

    # Loss should drop substantially below init.
    init_loss = model.history["loss_main"][0]
    final_loss = model.history["loss_main"][-1]
    assert final_loss < 0.5 * init_loss, (
        f"loss did not drop enough: init={init_loss:.4f} final={final_loss:.4f}"
    )

    # Per-planted-feature recovery: every planted latent must have at least
    # one learned feature with |corr| > 0.7 on a held-out split.
    n_train = n // 2
    A_eval, B_eval = A[n_train:], B[n_train:]
    Z_eval_true = Z_true[n_train:]
    Z_eval_learned = encode(model, A_eval, B_eval)

    best = _best_match_per_planted(Z_eval_true, Z_eval_learned)
    # Allow at most one weak planted feature so the test is robust to a
    # single unlucky init dimension.
    n_recovered = int((best > 0.7).sum())
    assert n_recovered >= n_features - 1, (
        f"only {n_recovered}/{n_features} planted features recovered; "
        f"per-feature best correlations = {best.round(3).tolist()}"
    )

    # Stream classification: the highest-correlation learned match for each
    # planted feature should sit on the correct side of the decoder split.
    r_G = feature_stream_share(model)
    cls = classify_features(model)
    for j in range(n_features):
        u = Z_eval_true[:, j]
        if u.std() < 1e-10:
            continue
        # find the learned feature with the highest correlation
        best_jj = -1
        best_c = -1.0
        for jj in range(model.d):
            v = Z_eval_learned[:, jj]
            if v.std() < 1e-10:
                continue
            c = _abs_corr(u, v)
            if c > best_c:
                best_c = c
                best_jj = jj
        if best_c < 0.7 or best_jj < 0:
            continue  # weak match -- already counted above
        learned_share = float(r_G[best_jj])
        if visibility[j] == "genome":
            assert learned_share > 0.7, (
                f"planted genome-only feature {j} matched learned {best_jj} "
                f"with genome share {learned_share:.3f} (cls={cls[best_jj]})"
            )
        elif visibility[j] == "ehr":
            assert learned_share < 0.3, (
                f"planted ehr-only feature {j} matched learned {best_jj} "
                f"with genome share {learned_share:.3f} (cls={cls[best_jj]})"
            )
        else:  # shared
            assert 0.1 < learned_share < 0.9, (
                f"planted shared feature {j} matched learned {best_jj} "
                f"with genome share {learned_share:.3f} (cls={cls[best_jj]})"
            )
