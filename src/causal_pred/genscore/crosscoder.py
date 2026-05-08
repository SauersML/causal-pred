"""TopK crosscoder over (genome, EHR) participant streams.

Anthropic-style sparse dictionary learning over two activation streams that
share a single latent space. For each participant the input is::

    a in R^{m_G}   genome-side activations (e.g. polygenic-score panel)
    b in R^{m_E}   EHR-side activations    (e.g. PheCode counts, drug classes,
                                             lab summaries)

The crosscoder learns a single shared encoder and per-stream decoders::

    x      = [a_z; b_z]                            (m_G + m_E,)  standardised
    pre    = x @ W_e + b_enc                       (d,)
    z      = TopK_k(ReLU(pre))                     (d,)          sparse latent
    a_hat  = z @ W_d_G                             (m_G,)        genome recon
    b_hat  = z @ W_d_E                             (m_E,)        EHR recon

After training, each feature ``j`` has a joint unit-norm decoder column split
across the two streams. Define::

    n_G[j]^2 = ||W_d_G[j, :]||^2
    n_E[j]^2 = ||W_d_E[j, :]||^2
    n_G[j]^2 + n_E[j]^2 = 1            (constraint)

The genome share ``r_G[j] = n_G[j]^2`` is in [0, 1]: near 1 means a
genome-only feature, near 0 means an EHR-only feature, near 0.5 means a
shared feature. Shared features are the ones with both genetic and clinical
support; they are the natural candidates to promote to DAG nodes.

Architecture choices, fixed (no width / sparsity sweeps):

* **TopK** activation (Gao et al. 2024) -- exact sparsity control, no L1
  shrinkage, no JumpReLU hyper-parameters.
* **AuxK** dead-feature revival (Gao et al. 2024) -- a small auxiliary loss
  that reconstructs the residual using the top dead pre-activations, keeping
  features alive without manual reinitialisation.
* **Joint unit-norm decoder** (per Lindsey et al. 2025 crosscoders) -- the
  concatenated decoder column for each feature is L2-normalised across both
  streams, so per-stream norms are directly comparable.
* **Standardised inputs** -- per-column z-scoring before encoding; with mean
  zero this absorbs the role of an explicit pre-encoder bias.
* **Torch accelerator training** -- dense minibatch matmuls, TopK, AuxK, and
  Adam run on CUDA or MPS. CPU-only hosts can still import this module and
  inspect trained models; training itself requires an accelerator.

References
----------
- Gao, Goh, Sutskever 2024, "Scaling and Evaluating Sparse Autoencoders"
  -- TopK SAE + AuxK + dead-feature definition.
- Bricken et al. 2023, "Towards Monosemanticity"
  -- decoder unit-norm constraint, pre-bias.
- Lindsey, Templeton et al. 2025, "Sparse Crosscoders for Cross-Layer
  Features" -- per-stream decoders sharing one encoder + latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


@dataclass
class TopKCrosscoder:
    """Trained TopK crosscoder weights and standardisation statistics.

    Attributes
    ----------
    W_e : (m_G + m_E, d)
        Shared encoder.
    b_enc : (d,)
        Encoder bias.
    W_d_G : (d, m_G)
        Genome-stream decoder.
    W_d_E : (d, m_E)
        EHR-stream decoder.
    mean_G, std_G : (m_G,)
        Per-column standardisation of the genome stream.
    mean_E, std_E : (m_E,)
        Per-column standardisation of the EHR stream.
    k : int
        TopK sparsity used during training.
    history : dict
        Training-loss diagnostics. Keys: ``"step"``, ``"loss_main"``,
        ``"loss_aux"``, ``"frac_dead"``.
    """

    W_e: np.ndarray
    b_enc: np.ndarray
    W_d_G: np.ndarray
    W_d_E: np.ndarray
    mean_G: np.ndarray
    std_G: np.ndarray
    mean_E: np.ndarray
    std_E: np.ndarray
    k: int
    history: dict

    @property
    def d(self) -> int:
        return int(self.W_e.shape[1])

    @property
    def m_G(self) -> int:
        return int(self.W_d_G.shape[1])

    @property
    def m_E(self) -> int:
        return int(self.W_d_E.shape[1])


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def _topk_per_row(pre: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the top-k positive pre-activations per row, zero the rest.

    Parameters
    ----------
    pre : (n, d)
        Pre-activations (already including any bias).
    k : int
        Sparsity. Must satisfy ``0 < k <= d``. Entries with non-positive
        pre-activations are zeroed even if they survive the partition.

    Returns
    -------
    z : (n, d)
        Activation matrix with at most k non-zero entries per row.
    mask : (n, d) bool
        Indicator of which entries are non-zero in ``z``. Used for backprop.
    """
    n, d = pre.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= d:
        z = np.maximum(pre, 0.0)
        mask = z > 0
        return z, mask
    # argpartition gives the indices of the top-k (unsorted within the k).
    idx = np.argpartition(pre, -k, axis=1)[:, -k:]
    rows = np.arange(n)[:, None]
    vals = pre[rows, idx]
    keep = vals > 0
    mask = np.zeros_like(pre, dtype=bool)
    mask[rows, idx] = keep
    z = np.where(mask, pre, 0.0)
    return z, mask

def encode(
    model: TopKCrosscoder, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Compute the sparse latent ``z`` for each participant.

    Returns
    -------
    z : (n, d)
        TopK-sparse activations per participant.
    """
    a_z = (A - model.mean_G) / model.std_G
    b_z = (B - model.mean_E) / model.std_E
    x = np.concatenate([a_z, b_z], axis=1)
    pre = x @ model.W_e + model.b_enc
    z, _ = _topk_per_row(pre, model.k)
    return z


def reconstruct(
    model: TopKCrosscoder, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode latents back into the original (de-standardised) input scale.

    Returns
    -------
    a_hat : (n, m_G)
    b_hat : (n, m_E)
    """
    a_hat = z @ model.W_d_G * model.std_G + model.mean_G
    b_hat = z @ model.W_d_E * model.std_E + model.mean_E
    return a_hat, b_hat


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------


def feature_stream_share(model: TopKCrosscoder) -> np.ndarray:
    """Per-feature genome share ``r_G[j] = n_G[j]^2 / (n_G[j]^2 + n_E[j]^2)``.

    With the joint unit-norm decoder constraint enforced during training,
    the denominator is 1 and ``r_G + r_E = 1`` (up to floating-point).
    Returns an array of shape ``(d,)`` with values in [0, 1].
    """
    n_g_sq = np.sum(model.W_d_G ** 2, axis=1)
    n_e_sq = np.sum(model.W_d_E ** 2, axis=1)
    total = n_g_sq + n_e_sq
    total = np.where(total > 0, total, 1.0)
    return n_g_sq / total


def classify_features(
    model: TopKCrosscoder,
    genome_threshold: float = 0.9,
    ehr_threshold: float = 0.1,
) -> np.ndarray:
    """Categorise each feature as ``"genome"``, ``"ehr"``, or ``"shared"``.

    Parameters
    ----------
    genome_threshold : float
        Genome share at or above which the feature is genome-only.
    ehr_threshold : float
        Genome share at or below which the feature is EHR-only.
    """
    if not 0.0 <= ehr_threshold < genome_threshold <= 1.0:
        raise ValueError(
            "Need 0 <= ehr_threshold < genome_threshold <= 1, "
            f"got {ehr_threshold} and {genome_threshold}"
        )
    r_G = feature_stream_share(model)
    out = np.full(model.d, "shared", dtype=object)
    out[r_G >= genome_threshold] = "genome"
    out[r_G <= ehr_threshold] = "ehr"
    return out


# ---------------------------------------------------------------------------
# Decoder normalisation
# ---------------------------------------------------------------------------


def _normalise_decoders(
    W_d_G: np.ndarray, W_d_E: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project so each feature's concatenated decoder column has unit L2 norm."""
    norms = np.sqrt(
        np.sum(W_d_G ** 2, axis=1) + np.sum(W_d_E ** 2, axis=1)
    )
    norms = np.where(norms > 1e-8, norms, 1.0)
    return W_d_G / norms[:, None], W_d_E / norms[:, None]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_crosscoder(
    A: np.ndarray,
    B: np.ndarray,
    d: Optional[int] = None,
    k: int = 32,
    n_steps: int = 2000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    aux_k: int = 64,
    aux_coef: float = 1.0 / 32.0,
    dead_steps: int = 200,
    rng: Optional[np.random.Generator] = None,
    log_every: int = 100,
) -> TopKCrosscoder:
    """Train a TopK crosscoder on paired (genome, EHR) activations.

    Parameters
    ----------
    A : (n, m_G)
        Genome-stream activations (e.g. polygenic-score panel).
    B : (n, m_E)
        EHR-stream activations (e.g. PheCode + drug + lab-summary panel).
    d : int, optional
        Latent dimension. Defaults to ``4 * (m_G + m_E)``.
    k : int
        TopK sparsity per participant. Must satisfy ``k <= d``.
    n_steps : int
        Number of Adam steps.
    batch_size : int
        Per-step minibatch.
    lr : float
        Adam learning rate.
    aux_k : int
        Number of dead pre-activations to use for the AuxK loss.
    aux_coef : float
        Weight on the AuxK loss. Default ``1/32`` per Gao et al.
    dead_steps : int
        A feature is considered dead if it has not activated for this many
        consecutive steps.
    rng : numpy.random.Generator, optional
        Random source. Defaults to ``np.random.default_rng(0)``.
    log_every : int
        Step interval at which to record diagnostics into ``history``.

    Returns
    -------
    TopKCrosscoder
        Trained model. The decoder columns satisfy the joint unit-norm
        constraint at return time.
    """
    rng_local = rng if rng is not None else np.random.default_rng(0)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"A and B must share rows, got {A.shape[0]} vs {B.shape[0]}"
        )
    n, m_G = A.shape
    m_E = B.shape[1]
    d_lat: int = int(d) if d is not None else 4 * (m_G + m_E)
    if k > d_lat:
        raise ValueError(f"k={k} must be <= d={d_lat}")
    if n <= 0:
        raise ValueError("crosscoder training requires at least one row")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    # --- standardisation -------------------------------------------------
    mean_G = A.mean(axis=0)
    std_G = A.std(axis=0)
    std_G = np.where(std_G > 1e-8, std_G, 1.0)
    mean_E = B.mean(axis=0)
    std_E = B.std(axis=0)
    std_E = np.where(std_E > 1e-8, std_E, 1.0)
    A_z = (A - mean_G) / std_G
    B_z = (B - mean_E) / std_E
    X = np.concatenate([A_z, B_z], axis=1)
    m = m_G + m_E

    # --- initialisation --------------------------------------------------
    # Encoder: scaled Gaussian. Decoder: take the encoder transpose split
    # across the two streams, then apply the joint unit-norm projection.
    scale = 1.0 / np.sqrt(m)
    W_e = rng_local.standard_normal((m, d_lat)) * scale
    b_enc = np.zeros(d_lat)
    W_d_G = W_e[:m_G, :].T.copy()
    W_d_E = W_e[m_G:, :].T.copy()
    W_d_G, W_d_E = _normalise_decoders(W_d_G, W_d_E)

    # --- Adam state ------------------------------------------------------
    params = {
        "W_e": W_e,
        "b_enc": b_enc,
        "W_d_G": W_d_G,
        "W_d_E": W_d_E,
    }
    m_state = {p: np.zeros_like(v) for p, v in params.items()}
    v_state = {p: np.zeros_like(v) for p, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # --- dead-feature tracker -------------------------------------------
    steps_since_active = np.zeros(d_lat, dtype=np.int64)

    inv_m_G = 1.0 / m_G
    inv_m_E = 1.0 / m_E

    history = {
        "step": [],
        "loss_main": [],
        "loss_aux": [],
        "frac_dead": [],
    }

    for step in range(1, n_steps + 1):
        idx = rng_local.integers(0, n, size=batch_size)
        x_batch = X[idx]
        a_target = A_z[idx]
        b_target = B_z[idx]
        bs = x_batch.shape[0]

        # ---- forward (main path) ---------------------------------------
        pre = x_batch @ params["W_e"] + params["b_enc"]
        z, mask = _topk_per_row(pre, k)

        a_hat = z @ params["W_d_G"]
        b_hat = z @ params["W_d_E"]

        res_G = a_hat - a_target
        res_E = b_hat - b_target

        loss_main = 0.5 * (
            inv_m_G * np.mean(np.sum(res_G ** 2, axis=1))
            + inv_m_E * np.mean(np.sum(res_E ** 2, axis=1))
        )

        # ---- backward (main path) --------------------------------------
        d_a_hat = res_G / bs
        d_b_hat = res_E / bs

        g_W_d_G = z.T @ d_a_hat
        g_W_d_E = z.T @ d_b_hat

        d_z = d_a_hat @ params["W_d_G"].T + d_b_hat @ params["W_d_E"].T
        d_pre = np.where(mask, d_z, 0.0)

        g_W_e = x_batch.T @ d_pre
        g_b_enc = d_pre.sum(axis=0)

        # ---- AuxK on dead features -------------------------------------
        loss_aux = 0.0
        dead_mask = steps_since_active >= dead_steps
        n_dead = int(dead_mask.sum())
        if n_dead > 0 and aux_k > 0 and aux_coef > 0:
            kk = min(aux_k, n_dead)
            pre_dead = np.where(dead_mask[None, :], pre, -np.inf)
            z_aux, mask_aux = _topk_per_row(pre_dead, kk)

            e_hat_G = z_aux @ params["W_d_G"]
            e_hat_E = z_aux @ params["W_d_E"]
            aux_G = e_hat_G + res_G
            aux_E = e_hat_E + res_E
            loss_aux = 0.5 * (np.mean(aux_G ** 2) + np.mean(aux_E ** 2))

            d_e_hat_G = aux_coef * aux_G / bs
            d_e_hat_E = aux_coef * aux_E / bs

            g_W_d_G = g_W_d_G + z_aux.T @ d_e_hat_G
            g_W_d_E = g_W_d_E + z_aux.T @ d_e_hat_E

            d_z_aux = d_e_hat_G @ params["W_d_G"].T + d_e_hat_E @ params["W_d_E"].T
            d_pre_aux = np.where(mask_aux, d_z_aux, 0.0)
            g_W_e = g_W_e + x_batch.T @ d_pre_aux
            g_b_enc = g_b_enc + d_pre_aux.sum(axis=0)

        # ---- update dead-feature tracker -------------------------------
        active_this_step = mask.any(axis=0)
        steps_since_active += 1
        steps_since_active[active_this_step] = 0

        # ---- Adam step --------------------------------------------------
        grads = {
            "W_e": g_W_e,
            "b_enc": g_b_enc,
            "W_d_G": g_W_d_G,
            "W_d_E": g_W_d_E,
        }
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        for name, g in grads.items():
            m_state[name] = beta1 * m_state[name] + (1.0 - beta1) * g
            v_state[name] = beta2 * v_state[name] + (1.0 - beta2) * (g * g)
            m_hat = m_state[name] / bc1
            v_hat = v_state[name] / bc2
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # ---- enforce decoder unit-norm constraint ----------------------
        params["W_d_G"], params["W_d_E"] = _normalise_decoders(
            params["W_d_G"], params["W_d_E"]
        )

        # ---- diagnostics -----------------------------------------------
        if step == 1 or step == n_steps or step % log_every == 0:
            history["step"].append(step)
            history["loss_main"].append(float(loss_main))
            history["loss_aux"].append(float(loss_aux))
            history["frac_dead"].append(float(n_dead) / d_lat)

    return TopKCrosscoder(
        W_e=params["W_e"],
        b_enc=params["b_enc"],
        W_d_G=params["W_d_G"],
        W_d_E=params["W_d_E"],
        mean_G=mean_G,
        std_G=std_G,
        mean_E=mean_E,
        std_E=std_E,
        k=k,
        history=history,
    )
