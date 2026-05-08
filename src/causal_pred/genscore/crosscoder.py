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
from typing import Optional, Sequence, Tuple

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
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    *,
    batch_size: Optional[int] = None,
    device: str | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Compute the sparse latent ``z`` for each participant.

    Passing ``device="auto"``, ``"cuda"``, or ``"mps"`` runs the dense
    encoding matmuls on the accelerator in batches. Leaving ``device`` as
    ``None`` uses the NumPy path, which is useful for small already-trained
    models and keeps non-training utilities usable on CPU-only hosts.

    Returns
    -------
    z : (n, d)
        TopK-sparse activations per participant.
    """
    if device is not None:
        return encode_batched(
            model,
            A,
            B,
            batch_size=batch_size or 65536,
            device=device,
            dtype=dtype,
        )

    a_z = (A - model.mean_G) / model.std_G
    b_z = (B - model.mean_E) / model.std_E
    x = np.concatenate([a_z, b_z], axis=1)
    pre = x @ model.W_e + model.b_enc
    z, _ = _topk_per_row(pre, model.k)
    return z


def encode_batched(
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    *,
    batch_size: int = 65536,
    device: str | None = "auto",
    dtype: str = "float32",
) -> np.ndarray:
    """Encode rows in accelerator batches and return all latent activations."""
    import torch

    resolved = _resolve_torch_device(device)
    torch_dtype = _torch_float_dtype(dtype)
    _configure_torch_matmul(resolved)
    tensors = _torch_model_tensors(model, resolved, torch_dtype)

    A_np = np.asarray(A)
    B_np = np.asarray(B)
    if A_np.shape[0] != B_np.shape[0]:
        raise ValueError(
            f"A and B must share rows, got {A_np.shape[0]} vs {B_np.shape[0]}"
        )
    n = A_np.shape[0]
    z_out = np.empty((n, model.d), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            a = torch.as_tensor(A_np[start:stop], dtype=torch_dtype, device=resolved)
            b = torch.as_tensor(B_np[start:stop], dtype=torch_dtype, device=resolved)
            x = torch.cat(
                (
                    (a - tensors["mean_G"]) / tensors["std_G"],
                    (b - tensors["mean_E"]) / tensors["std_E"],
                ),
                dim=1,
            )
            pre = x @ tensors["W_e"] + tensors["b_enc"]
            z, _ = _torch_topk_per_row(pre, model.k)
            z_out[start:stop] = z.float().cpu().numpy()
    return z_out


def encode_selected_batched(
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    feature_indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int = 65536,
    device: str | None = "auto",
    dtype: str = "float32",
) -> np.ndarray:
    """Encode only selected latent columns without materialising full ``z``."""
    import torch

    resolved = _resolve_torch_device(device)
    torch_dtype = _torch_float_dtype(dtype)
    _configure_torch_matmul(resolved)
    tensors = _torch_model_tensors(model, resolved, torch_dtype)

    A_np = np.asarray(A)
    B_np = np.asarray(B)
    if A_np.shape[0] != B_np.shape[0]:
        raise ValueError(
            f"A and B must share rows, got {A_np.shape[0]} vs {B_np.shape[0]}"
        )
    idx_np = np.asarray(feature_indices, dtype=np.int64)
    idx = torch.as_tensor(idx_np, dtype=torch.long, device=resolved)
    n = A_np.shape[0]
    z_out = np.empty((n, idx_np.size), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            a = torch.as_tensor(A_np[start:stop], dtype=torch_dtype, device=resolved)
            b = torch.as_tensor(B_np[start:stop], dtype=torch_dtype, device=resolved)
            x = torch.cat(
                (
                    (a - tensors["mean_G"]) / tensors["std_G"],
                    (b - tensors["mean_E"]) / tensors["std_E"],
                ),
                dim=1,
            )
            pre = x @ tensors["W_e"] + tensors["b_enc"]
            z, _ = _torch_topk_per_row(pre, model.k)
            z_out[start:stop] = z.index_select(1, idx).float().cpu().numpy()
    return z_out


def activation_rate_batched(
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    *,
    batch_size: int = 65536,
    device: str | None = "auto",
    dtype: str = "float32",
) -> np.ndarray:
    """Compute per-feature activation rates on the accelerator in batches."""
    import torch

    resolved = _resolve_torch_device(device)
    torch_dtype = _torch_float_dtype(dtype)
    _configure_torch_matmul(resolved)
    tensors = _torch_model_tensors(model, resolved, torch_dtype)

    A_np = np.asarray(A)
    B_np = np.asarray(B)
    if A_np.shape[0] != B_np.shape[0]:
        raise ValueError(
            f"A and B must share rows, got {A_np.shape[0]} vs {B_np.shape[0]}"
        )
    n = A_np.shape[0]
    counts = torch.zeros(model.d, dtype=torch.float32, device=resolved)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            a = torch.as_tensor(A_np[start:stop], dtype=torch_dtype, device=resolved)
            b = torch.as_tensor(B_np[start:stop], dtype=torch_dtype, device=resolved)
            x = torch.cat(
                (
                    (a - tensors["mean_G"]) / tensors["std_G"],
                    (b - tensors["mean_E"]) / tensors["std_E"],
                ),
                dim=1,
            )
            pre = x @ tensors["W_e"] + tensors["b_enc"]
            _, mask = _torch_topk_per_row(pre, model.k)
            counts += mask.sum(dim=0, dtype=torch.float32)
    return (counts / float(n)).cpu().numpy()


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
    device: str | None = "auto",
    dtype: str = "float32",
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
    device : str, optional
        ``"auto"`` selects CUDA first, then MPS. CPU is rejected for training.
    dtype : {"float32", "float16", "bfloat16"}
        Floating dtype used for accelerator tensors. ``float32`` is the
        default because it is robust and uses TF32 matmul on supported CUDA
        devices.

    Returns
    -------
    TopKCrosscoder
        Trained model. The decoder columns satisfy the joint unit-norm
        constraint at return time.
    """
    import torch

    rng_local = rng if rng is not None else np.random.default_rng(0)
    resolved = _resolve_torch_device(device)
    torch_dtype = _torch_float_dtype(dtype)
    _configure_torch_matmul(resolved)

    A_np = np.ascontiguousarray(np.asarray(A, dtype=np.float32))
    B_np = np.ascontiguousarray(np.asarray(B, dtype=np.float32))
    if A_np.shape[0] != B_np.shape[0]:
        raise ValueError(
            f"A and B must share rows, got {A_np.shape[0]} vs {B_np.shape[0]}"
        )
    n, m_G = A_np.shape
    m_E = B_np.shape[1]
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
    A_t = torch.as_tensor(A_np, dtype=torch_dtype, device=resolved)
    B_t = torch.as_tensor(B_np, dtype=torch_dtype, device=resolved)
    mean_G = A_t.mean(dim=0)
    std_G = A_t.std(dim=0, unbiased=False).clamp_min(1e-8)
    mean_E = B_t.mean(dim=0)
    std_E = B_t.std(dim=0, unbiased=False).clamp_min(1e-8)
    A_z = (A_t - mean_G) / std_G
    B_z = (B_t - mean_E) / std_E
    del A_t, B_t
    m = m_G + m_E

    # --- initialisation --------------------------------------------------
    # Encoder: scaled Gaussian. Decoder: take the encoder transpose split
    # across the two streams, then apply the joint unit-norm projection.
    seed = int(rng_local.integers(0, 2**63 - 1))
    torch.manual_seed(seed)
    gen = torch.Generator(device=resolved).manual_seed(seed) if resolved.type == "cuda" else None
    scale = 1.0 / np.sqrt(m)
    randn_kwargs = {"dtype": torch_dtype, "device": resolved}
    if gen is not None:
        randn_kwargs["generator"] = gen
    W_e = torch.randn((m, d_lat), **randn_kwargs)
    W_e.mul_(scale)
    b_enc = torch.zeros(d_lat, dtype=torch_dtype, device=resolved)
    W_d_G = W_e[:m_G, :].T.contiguous().clone()
    W_d_E = W_e[m_G:, :].T.contiguous().clone()
    _normalise_decoders_torch(W_d_G, W_d_E)

    # --- Adam state ------------------------------------------------------
    params = {
        "W_e": W_e,
        "b_enc": b_enc,
        "W_d_G": W_d_G,
        "W_d_E": W_d_E,
    }
    m_state = {p: torch.zeros_like(v, dtype=torch.float32) for p, v in params.items()}
    v_state = {p: torch.zeros_like(v, dtype=torch.float32) for p, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # --- dead-feature tracker -------------------------------------------
    steps_since_active = torch.zeros(d_lat, dtype=torch.int64, device=resolved)

    inv_m_G = 1.0 / m_G
    inv_m_E = 1.0 / m_E

    history = {
        "step": [],
        "loss_main": [],
        "loss_aux": [],
        "frac_dead": [],
    }

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            randint_kwargs = {"dtype": torch.long, "device": resolved}
            if gen is not None:
                randint_kwargs["generator"] = gen
            idx = torch.randint(n, (batch_size,), **randint_kwargs)
            a_target = A_z.index_select(0, idx)
            b_target = B_z.index_select(0, idx)
            x_batch = torch.cat((a_target, b_target), dim=1)
            bs = x_batch.shape[0]

            # ---- forward (main path) -----------------------------------
            pre = x_batch @ params["W_e"] + params["b_enc"]
            z, mask = _torch_topk_per_row(pre, k)

            a_hat = z @ params["W_d_G"]
            b_hat = z @ params["W_d_E"]

            res_G = a_hat - a_target
            res_E = b_hat - b_target

            loss_main = 0.5 * (
                inv_m_G * torch.mean(torch.sum(res_G * res_G, dim=1))
                + inv_m_E * torch.mean(torch.sum(res_E * res_E, dim=1))
            )

            # ---- backward (main path) ----------------------------------
            d_a_hat = res_G / bs
            d_b_hat = res_E / bs

            g_W_d_G = z.T @ d_a_hat
            g_W_d_E = z.T @ d_b_hat

            d_z = d_a_hat @ params["W_d_G"].T + d_b_hat @ params["W_d_E"].T
            d_pre = torch.where(mask, d_z, torch.zeros_like(d_z))

            g_W_e = x_batch.T @ d_pre
            g_b_enc = d_pre.sum(dim=0)

            # ---- AuxK on dead features ---------------------------------
            loss_aux = pre.new_zeros(())
            dead_mask = steps_since_active >= dead_steps
            if step >= dead_steps and aux_k > 0 and aux_coef > 0:
                kk = min(aux_k, d_lat)
                pre_dead = pre.masked_fill(~dead_mask[None, :], float("-inf"))
                z_aux, mask_aux = _torch_topk_per_row(pre_dead, kk)

                e_hat_G = z_aux @ params["W_d_G"]
                e_hat_E = z_aux @ params["W_d_E"]
                row_aux = mask_aux.any(dim=1).to(torch_dtype)[:, None]
                aux_G = (e_hat_G + res_G) * row_aux
                aux_E = (e_hat_E + res_E) * row_aux
                loss_aux = 0.5 * (
                    torch.mean(aux_G * aux_G) + torch.mean(aux_E * aux_E)
                )

                d_e_hat_G = aux_coef * aux_G / bs
                d_e_hat_E = aux_coef * aux_E / bs

                g_W_d_G = g_W_d_G + z_aux.T @ d_e_hat_G
                g_W_d_E = g_W_d_E + z_aux.T @ d_e_hat_E

                d_z_aux = (
                    d_e_hat_G @ params["W_d_G"].T
                    + d_e_hat_E @ params["W_d_E"].T
                )
                d_pre_aux = torch.where(mask_aux, d_z_aux, torch.zeros_like(d_z_aux))
                g_W_e = g_W_e + x_batch.T @ d_pre_aux
                g_b_enc = g_b_enc + d_pre_aux.sum(dim=0)

            # ---- update dead-feature tracker ---------------------------
            active_this_step = mask.any(dim=0)
            steps_since_active.add_(1)
            steps_since_active.masked_fill_(active_this_step, 0)

            # ---- Adam step ----------------------------------------------
            grads = {
                "W_e": g_W_e,
                "b_enc": g_b_enc,
                "W_d_G": g_W_d_G,
                "W_d_E": g_W_d_E,
            }
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            for name, g in grads.items():
                g32 = g.float()
                m_state[name].mul_(beta1).add_(g32, alpha=1.0 - beta1)
                v_state[name].mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)
                m_hat = m_state[name] / bc1
                denom = (v_state[name] / bc2).sqrt().add_(eps)
                params[name].add_((-lr * (m_hat / denom)).to(params[name].dtype))

            # ---- enforce decoder unit-norm constraint ------------------
            _normalise_decoders_torch(params["W_d_G"], params["W_d_E"])

            # ---- diagnostics -------------------------------------------
            if step == 1 or step == n_steps or step % log_every == 0:
                n_dead = int(dead_mask.sum().item())
                history["step"].append(step)
                history["loss_main"].append(float(loss_main.item()))
                history["loss_aux"].append(float(loss_aux.item()))
                history["frac_dead"].append(float(n_dead) / d_lat)

    return TopKCrosscoder(
        W_e=params["W_e"].float().cpu().numpy(),
        b_enc=params["b_enc"].float().cpu().numpy(),
        W_d_G=params["W_d_G"].float().cpu().numpy(),
        W_d_E=params["W_d_E"].float().cpu().numpy(),
        mean_G=mean_G.float().cpu().numpy(),
        std_G=std_G.float().cpu().numpy(),
        mean_E=mean_E.float().cpu().numpy(),
        std_E=std_E.float().cpu().numpy(),
        k=k,
        history=history,
        device=str(resolved),
        dtype=dtype,
    )
