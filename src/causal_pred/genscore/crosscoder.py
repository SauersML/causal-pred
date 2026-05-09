"""GPU BatchTopK multi-view crosscoder over paired genome/EHR streams.

The trainer in this module is intentionally no longer a concatenated NumPy
autoencoder. It learns separate latent banks:

``shared``
    Decodes into both streams and is the only bank promoted downstream.
``genome_private``
    Encodes/decodes genome-only signal.
``ehr_private``
    Encodes/decodes EHR-only signal.

Each minibatch is trained through three views of the same participants:

``z_AB = encode(A, B)``, ``z_A = encode(A, 0)``, and ``z_B = encode(0, B)``.
The loss combines full reconstruction, cross reconstruction, shared-bank
alignment, an optional same-person contrastive term, and AuxK dead-latent
revival. Activations use BatchTopK by default so the batch has fixed average
sparsity while individual participants may use fewer or more latents.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CROSSCODER_FIT_STATE_VERSION = 2

BANK_SHARED = 0
BANK_GENOME_PRIVATE = 1
BANK_EHR_PRIVATE = 2

_BINARY_EHR_KINDS = {"condition", "drug", "lab_missing"}
_COUNT_EHR_KINDS = {"utilisation"}


@dataclass
class TopKCrosscoder:
    """Trained multi-view crosscoder weights and standardisation statistics."""

    W_e: np.ndarray
    b_enc: np.ndarray
    W_d_G: np.ndarray
    W_d_E: np.ndarray
    mean_G: np.ndarray
    std_G: np.ndarray
    mean_E: np.ndarray
    std_E: np.ndarray
    k: int
    latent_bank: np.ndarray
    activation_kind: str
    history: dict
    ehr_feature_kinds: Tuple[str, ...] = ()
    device: str = "unknown"

    @property
    def d(self) -> int:
        return int(self.W_e.shape[1])

    @property
    def m_G(self) -> int:
        return int(self.W_d_G.shape[1])

    @property
    def m_E(self) -> int:
        return int(self.W_d_E.shape[1])

    @property
    def shared_indices(self) -> np.ndarray:
        return np.flatnonzero(self.latent_bank == BANK_SHARED)


def _topk_per_row(pre: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the top-k positive pre-activations per row."""
    n, d = pre.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= d:
        z = np.maximum(pre, 0.0)
        return z, z > 0
    idx = np.argpartition(pre, -k, axis=1)[:, -k:]
    rows = np.arange(n)[:, None]
    vals = pre[rows, idx]
    keep = vals > 0
    mask = np.zeros_like(pre, dtype=bool)
    mask[rows, idx] = keep
    return np.where(mask, pre, 0.0), mask


def _batch_topk_np(
    pre: np.ndarray,
    k: int,
    *,
    row_cap_multiplier: float,
    allowed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores = np.maximum(pre, 0.0)
    if allowed is not None:
        scores = np.where(allowed[None, :], scores, 0.0)
    n, d = scores.shape
    take = min(scores.size, max(1, int(n * k)))
    flat = scores.reshape(-1)
    if take >= flat.size:
        mask = scores > 0
    else:
        idx = np.argpartition(flat, -take)[-take:]
        idx = idx[flat[idx] > 0]
        mask = np.zeros(flat.size, dtype=bool)
        mask[idx] = True
        mask = mask.reshape(scores.shape)
    if row_cap_multiplier > 0:
        cap = min(d, max(k, int(np.ceil(float(k) * row_cap_multiplier))))
        if cap < d and mask.any():
            row_scores = np.where(mask, scores, 0.0)
            idx = np.argpartition(row_scores, -cap, axis=1)[:, -cap:]
            cap_mask = np.zeros_like(mask)
            rows = np.arange(n)[:, None]
            cap_mask[rows, idx] = True
            mask &= cap_mask
    return np.where(mask, pre, 0.0), mask


def _activate_np(
    pre: np.ndarray,
    k: int,
    activation_kind: str,
    *,
    row_cap_multiplier: float = 4.0,
    allowed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if allowed is not None:
        pre = np.where(allowed[None, :], pre, -np.inf)
    if activation_kind == "topk":
        return _topk_per_row(pre, k)
    if activation_kind == "batch_topk":
        return _batch_topk_np(
            pre,
            k,
            row_cap_multiplier=row_cap_multiplier,
            allowed=allowed,
        )
    raise ValueError(f"unknown activation_kind {activation_kind!r}")


def encode(
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    *,
    view: str = "both",
) -> np.ndarray:
    """Compute sparse latents for paired rows.

    ``view`` is one of ``"both"``, ``"genome"``, or ``"ehr"`` and mirrors the
    training views. Single-stream views can only activate the shared bank plus
    that stream's private bank.

    Routes through the GPU when CUDA is available. The full panel encode
    materialises ``pre = (n, d)`` float64 plus an equally-sized ``scores``
    intermediate inside batch_topk - around 5-7 GB host RAM in the AoU
    cohort - which OOM-killed the notebook when ``select_shared_features``
    called this three times in a row. On the T4 the same allocations live
    in 15 GB VRAM and only the final ``z`` returns to host RAM.
    """
    if torch.cuda.is_available():
        return _encode_torch(model, A, B, view=view, device=torch.device("cuda"))
    a_z = (np.asarray(A, dtype=np.float64) - model.mean_G) / model.std_G
    b_z = (np.asarray(B, dtype=np.float64) - model.mean_E) / model.std_E
    if view == "genome":
        b_z = np.zeros_like(b_z)
        allowed = model.latent_bank != BANK_EHR_PRIVATE
    elif view == "ehr":
        a_z = np.zeros_like(a_z)
        allowed = model.latent_bank != BANK_GENOME_PRIVATE
    elif view == "both":
        allowed = None
    else:
        raise ValueError(f"unknown encode view {view!r}")
    pre = np.concatenate([a_z, b_z], axis=1) @ model.W_e + model.b_enc
    z, _ = _activate_np(
        pre,
        model.k,
        model.activation_kind,
        allowed=allowed,
    )
    return z


def _encode_torch(
    model: TopKCrosscoder,
    A: np.ndarray,
    B: np.ndarray,
    *,
    view: str,
    device: torch.device,
) -> np.ndarray:
    import math

    A_t = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float64)).to(device)
    B_t = torch.from_numpy(np.ascontiguousarray(B, dtype=np.float64)).to(device)
    mean_G = torch.from_numpy(np.ascontiguousarray(model.mean_G, dtype=np.float64)).to(device)
    std_G = torch.from_numpy(np.ascontiguousarray(model.std_G, dtype=np.float64)).to(device)
    mean_E = torch.from_numpy(np.ascontiguousarray(model.mean_E, dtype=np.float64)).to(device)
    std_E = torch.from_numpy(np.ascontiguousarray(model.std_E, dtype=np.float64)).to(device)
    W_e = torch.from_numpy(np.ascontiguousarray(model.W_e, dtype=np.float64)).to(device)
    b_enc = torch.from_numpy(np.ascontiguousarray(model.b_enc, dtype=np.float64)).to(device)

    a_z = (A_t - mean_G) / std_G
    b_z = (B_t - mean_E) / std_E
    del A_t, B_t

    if view == "genome":
        b_z = torch.zeros_like(b_z)
        latent_bank = torch.from_numpy(np.ascontiguousarray(model.latent_bank)).to(device)
        allowed = latent_bank != BANK_EHR_PRIVATE
    elif view == "ehr":
        a_z = torch.zeros_like(a_z)
        latent_bank = torch.from_numpy(np.ascontiguousarray(model.latent_bank)).to(device)
        allowed = latent_bank != BANK_GENOME_PRIVATE
    elif view == "both":
        allowed = None
    else:
        raise ValueError(f"unknown encode view {view!r}")

    pre = torch.cat([a_z, b_z], dim=1) @ W_e + b_enc
    del a_z, b_z, W_e, b_enc, mean_G, std_G, mean_E, std_E

    n, d = int(pre.shape[0]), int(pre.shape[1])
    k = int(model.k)
    activation_kind = str(model.activation_kind)

    if allowed is not None:
        pre = torch.where(
            allowed.unsqueeze(0), pre, torch.full_like(pre, float("-inf"))
        )

    if activation_kind == "topk":
        if k >= d:
            mask = pre > 0
        else:
            topk_vals, topk_idx = torch.topk(pre, k, dim=1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(1, topk_idx, topk_vals > 0)
        z = torch.where(mask, pre, torch.zeros_like(pre))
    elif activation_kind == "batch_topk":
        scores = torch.relu(pre)
        if allowed is not None:
            scores = torch.where(
                allowed.unsqueeze(0), scores, torch.zeros_like(scores)
            )
        take = min(scores.numel(), max(1, n * k))
        flat = scores.flatten()
        if take >= flat.numel():
            mask = scores > 0
        else:
            topk_vals, topk_idx = torch.topk(flat, take, sorted=False)
            keep_idx = topk_idx[topk_vals > 0]
            mask_flat = torch.zeros_like(flat, dtype=torch.bool)
            mask_flat[keep_idx] = True
            mask = mask_flat.view(n, d)
        del flat
        row_cap_multiplier = 4.0
        cap = min(d, max(k, int(math.ceil(k * row_cap_multiplier))))
        if cap < d and bool(mask.any()):
            row_scores = torch.where(mask, scores, torch.zeros_like(scores))
            _, cap_idx = torch.topk(row_scores, cap, dim=1)
            cap_mask = torch.zeros_like(mask)
            cap_mask.scatter_(1, cap_idx, True)
            mask &= cap_mask
            del row_scores, cap_idx, cap_mask
        z = torch.where(mask, pre, torch.zeros_like(pre))
        del scores, mask
    else:
        raise ValueError(f"unknown activation_kind {activation_kind!r}")

    del pre
    out = z.cpu().numpy()
    del z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _ehr_kind_masks(kinds: Sequence[str], m_E: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not kinds:
        kinds = ("gaussian",) * m_E
    if len(kinds) != m_E:
        raise ValueError(f"ehr_feature_kinds has length {len(kinds)} but m_E={m_E}")
    kind_arr = np.asarray([str(k) for k in kinds], dtype=object)
    binary = np.isin(kind_arr, list(_BINARY_EHR_KINDS))
    count = np.isin(kind_arr, list(_COUNT_EHR_KINDS))
    gaussian = ~(binary | count)
    return gaussian, binary, count


def reconstruct(
    model: TopKCrosscoder,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode latents into genome values and EHR values/probabilities."""
    z = np.asarray(z, dtype=np.float64)
    a_hat = z @ model.W_d_G * model.std_G + model.mean_G
    b_raw = z @ model.W_d_E
    b_hat = np.empty_like(b_raw, dtype=np.float64)
    gaussian, binary, count = _ehr_kind_masks(model.ehr_feature_kinds, model.m_E)
    if gaussian.any():
        b_hat[:, gaussian] = b_raw[:, gaussian] * model.std_E[gaussian] + model.mean_E[gaussian]
    if binary.any():
        prevalence = np.clip(model.mean_E[binary], 1e-4, 1.0 - 1e-4)
        prior_logit = np.log(prevalence / (1.0 - prevalence))
        b_hat[:, binary] = 1.0 / (1.0 + np.exp(-(b_raw[:, binary] + prior_logit[None, :])))
    if count.any():
        b_hat[:, count] = np.expm1(np.logaddexp(0.0, b_raw[:, count]))
    return a_hat, b_hat


def feature_stream_share(model: TopKCrosscoder) -> np.ndarray:
    """Per-feature genome share from decoder norms."""
    n_g_sq = np.sum(model.W_d_G ** 2, axis=1)
    n_e_sq = np.sum(model.W_d_E ** 2, axis=1)
    total = np.where(n_g_sq + n_e_sq > 0, n_g_sq + n_e_sq, 1.0)
    return n_g_sq / total


def classify_features(
    model: TopKCrosscoder,
    genome_threshold: float = 0.9,
    ehr_threshold: float = 0.1,
) -> np.ndarray:
    if not 0.0 <= ehr_threshold < genome_threshold <= 1.0:
        raise ValueError(
            "Need 0 <= ehr_threshold < genome_threshold <= 1, "
            f"got {ehr_threshold} and {genome_threshold}"
        )
    r_g = feature_stream_share(model)
    out = np.full(model.d, "shared", dtype=object)
    out[model.latent_bank == BANK_GENOME_PRIVATE] = "genome_private"
    out[model.latent_bank == BANK_EHR_PRIVATE] = "ehr_private"
    out[(model.latent_bank == BANK_SHARED) & (r_g >= genome_threshold)] = "genome"
    out[(model.latent_bank == BANK_SHARED) & (r_g <= ehr_threshold)] = "ehr"
    return out


def _normalise_decoders(
    W_d_G: np.ndarray,
    W_d_E: np.ndarray,
    latent_bank: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    Wg = np.asarray(W_d_G).copy()
    We = np.asarray(W_d_E).copy()
    if latent_bank is not None:
        bank = np.asarray(latent_bank)
        Wg[bank == BANK_EHR_PRIVATE, :] = 0.0
        We[bank == BANK_GENOME_PRIVATE, :] = 0.0
    norms = np.sqrt(np.sum(Wg ** 2, axis=1) + np.sum(We ** 2, axis=1))
    norms = np.where(norms > 1e-8, norms, 1.0)
    return Wg / norms[:, None], We / norms[:, None]


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    with part.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(part, path)


def save_crosscoder_fit_state(path: str | os.PathLike, state: dict[str, Any]) -> None:
    meta = {
        "version": CROSSCODER_FIT_STATE_VERSION,
        "step": int(state["step"]),
        "target_steps": int(state["target_steps"]),
        "n": int(state["n"]),
        "m_G": int(state["m_G"]),
        "m_E": int(state["m_E"]),
        "d": int(state["d"]),
        "k": int(state["k"]),
        "batch_size": int(state["batch_size"]),
        "lr": float(state["lr"]),
        "aux_k": int(state["aux_k"]),
        "aux_coef": float(state["aux_coef"]),
        "dead_steps": int(state["dead_steps"]),
        "activation_kind": str(state["activation_kind"]),
        "row_cap_multiplier": float(state["row_cap_multiplier"]),
        "shared_fraction": float(state["shared_fraction"]),
        "cross_reconstruction_coef": float(state["cross_reconstruction_coef"]),
        "shared_alignment_coef": float(state["shared_alignment_coef"]),
        "contrastive_coef": float(state["contrastive_coef"]),
        "weight_decay": float(state["weight_decay"]),
        "validation_fraction": float(state["validation_fraction"]),
        "mixed_likelihood": bool(state["mixed_likelihood"]),
        "train_dtype": str(np.dtype(state["train_dtype"])),
        "ehr_feature_kinds": list(state["ehr_feature_kinds"]),
    }
    _atomic_npz(
        Path(path),
        meta_json=np.array(json.dumps(meta)),
        history_json=np.array(json.dumps(state["history"])),
        rng_state_json=np.array(json.dumps(state["rng_state"])),
        latent_bank=np.asarray(state["latent_bank"], dtype=np.int8),
        W_e=np.asarray(state["W_e"]),
        b_enc=np.asarray(state["b_enc"]),
        W_d_G=np.asarray(state["W_d_G"]),
        W_d_E=np.asarray(state["W_d_E"]),
        mean_G=np.asarray(state["mean_G"], dtype=np.float64),
        std_G=np.asarray(state["std_G"], dtype=np.float64),
        mean_E=np.asarray(state["mean_E"], dtype=np.float64),
        std_E=np.asarray(state["std_E"], dtype=np.float64),
        steps_since_active=np.asarray(state["steps_since_active"], dtype=np.int64),
        ever_active=np.asarray(state["ever_active"], dtype=bool),
        train_idx=np.asarray(state["train_idx"], dtype=np.int64),
        val_idx=np.asarray(state["val_idx"], dtype=np.int64),
        **{f"adam_m_{k}": np.asarray(v) for k, v in state["adam_m"].items()},
        **{f"adam_v_{k}": np.asarray(v) for k, v in state["adam_v"].items()},
    )


def load_crosscoder_fit_state(path: str | os.PathLike) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"].item()))
        version = int(meta.get("version", -1))
        if version != CROSSCODER_FIT_STATE_VERSION:
            raise ValueError(
                f"crosscoder checkpoint version {version} != {CROSSCODER_FIT_STATE_VERSION}"
            )
        return {
            **meta,
            "history": json.loads(str(z["history_json"].item())),
            "rng_state": json.loads(str(z["rng_state_json"].item())),
            "latent_bank": z["latent_bank"].astype(np.int8),
            "W_e": np.asarray(z["W_e"]),
            "b_enc": np.asarray(z["b_enc"]),
            "W_d_G": np.asarray(z["W_d_G"]),
            "W_d_E": np.asarray(z["W_d_E"]),
            "mean_G": z["mean_G"].astype(np.float64),
            "std_G": z["std_G"].astype(np.float64),
            "mean_E": z["mean_E"].astype(np.float64),
            "std_E": z["std_E"].astype(np.float64),
            "steps_since_active": z["steps_since_active"].astype(np.int64),
            "ever_active": z["ever_active"].astype(bool),
            "train_idx": z["train_idx"].astype(np.int64),
            "val_idx": z["val_idx"].astype(np.int64),
            "adam_m": {
                "W_e": np.asarray(z["adam_m_W_e"]),
                "b_enc": np.asarray(z["adam_m_b_enc"]),
                "W_d_G": np.asarray(z["adam_m_W_d_G"]),
                "W_d_E": np.asarray(z["adam_m_W_d_E"]),
            },
            "adam_v": {
                "W_e": np.asarray(z["adam_v_W_e"]),
                "b_enc": np.asarray(z["adam_v_b_enc"]),
                "W_d_G": np.asarray(z["adam_v_W_d_G"]),
                "W_d_E": np.asarray(z["adam_v_W_d_E"]),
            },
        }


def _resolve_device(device: str) -> torch.device:
    device = str(device)
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("crosscoder training requires CUDA or MPS; pass device='cpu' for deterministic unit tests")
    out = torch.device(device)
    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if out.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested MPS but torch.backends.mps.is_available() is false")
    return out


def _make_latent_bank(d: int, shared_fraction: float) -> np.ndarray:
    if not 0.0 < shared_fraction < 1.0:
        raise ValueError(f"shared_fraction must be in (0, 1), got {shared_fraction}")
    n_shared = max(1, min(d - 2, int(round(d * shared_fraction))))
    remaining = d - n_shared
    n_genome = max(1, remaining // 2)
    n_ehr = d - n_shared - n_genome
    if n_ehr <= 0:
        n_ehr = 1
        n_genome -= 1
    return np.asarray(
        [BANK_SHARED] * n_shared
        + [BANK_GENOME_PRIVATE] * n_genome
        + [BANK_EHR_PRIVATE] * n_ehr,
        dtype=np.int8,
    )


def _torch_activate(
    pre: torch.Tensor,
    k: int,
    activation_kind: str,
    *,
    row_cap_multiplier: float,
    allowed: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if allowed is not None:
        pre = pre.masked_fill(~allowed[None, :], torch.finfo(pre.dtype).min)
    scores = torch.relu(pre)
    n, d = scores.shape
    if activation_kind == "topk":
        kk = min(k, d)
        vals, idx = torch.topk(scores, kk, dim=1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, vals > 0)
    elif activation_kind == "batch_topk":
        take = min(scores.numel(), max(1, int(n * k)))
        vals, idx = torch.topk(scores.reshape(-1), take)
        idx = idx[vals > 0]
        mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
        if idx.numel():
            mask[idx] = True
        mask = mask.reshape_as(scores)
        if row_cap_multiplier > 0:
            cap = min(d, max(k, int(np.ceil(float(k) * row_cap_multiplier))))
            if cap < d:
                selected_scores = torch.where(mask, scores, torch.zeros_like(scores))
                _vals, row_idx = torch.topk(selected_scores, cap, dim=1)
                cap_mask = torch.zeros_like(mask)
                cap_mask.scatter_(1, row_idx, True)
                mask = mask & cap_mask
    else:
        raise ValueError(f"unknown activation_kind {activation_kind!r}")
    return torch.where(mask, pre, torch.zeros_like(pre)), mask


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def _ehr_recon_metrics(
    b_z: np.ndarray,
    b_raw: np.ndarray,
    b_hat: np.ndarray,
    *,
    gaussian: np.ndarray,
    binary: np.ndarray,
    count: np.ndarray,
    mean_E: np.ndarray,
) -> dict[str, float]:
    """Per-kind reconstruction quality.

    ``b_hat`` is the model's raw decoder output ``z @ W_d_E``. In mixed-
    likelihood mode that's a z-score for Gaussian columns, a logit for
    binary columns, and a log-rate for count columns. Comparing it
    directly against the z-scored target with a single R^2 is meaningless
    when most columns are binary - the binary z-scores are huge spikes
    (~1/sqrt(prevalence)) while the logits are O(1), so the residuals
    explode and drive R^2 deeply negative even when classification is
    perfect.

    Compute each kind in its own native space:
      * Gaussian: standard R^2 on z-scored values.
      * Binary: Brier score against the actual {0,1} target after
        sigmoid(logit + prior_logit). Lower is better; 0.0 = perfect,
        prevalence*(1-prevalence) = predicting the prior.
      * Count: R^2 in log1p(rate) space against log1p(target), since
        that's what the loss already uses.
    """
    out: dict[str, float] = {}
    if gaussian.any():
        out["r2_ehr_gaussian"] = _r2(b_z[:, gaussian], b_hat[:, gaussian])
    else:
        out["r2_ehr_gaussian"] = float("nan")

    if binary.any():
        prevalence = np.clip(mean_E[binary], 1e-4, 1.0 - 1e-4)
        prior_logit = np.log(prevalence / (1.0 - prevalence))
        prob = 1.0 / (1.0 + np.exp(-(b_hat[:, binary] + prior_logit[None, :])))
        target = b_raw[:, binary]
        out["brier_ehr_binary"] = float(np.mean((prob - target) ** 2))
        prior_brier = float(
            np.mean((prevalence[None, :] - target) ** 2)
        )
        out["brier_lift_vs_prior"] = prior_brier - out["brier_ehr_binary"]
    else:
        out["brier_ehr_binary"] = float("nan")
        out["brier_lift_vs_prior"] = float("nan")

    if count.any():
        target = np.log1p(np.clip(b_raw[:, count], 0.0, None))
        pred = np.log1p(np.clip(np.expm1(np.logaddexp(0.0, b_hat[:, count])), 0.0, None))
        out["r2_ehr_count_logspace"] = _r2(target, pred)
    else:
        out["r2_ehr_count_logspace"] = float("nan")

    return out


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
    progress: Optional[Callable[[str], None]] = None,
    warm_start: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[str | os.PathLike] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_callback: Optional[Callable[[Path, int], None]] = None,
    train_dtype: str | np.dtype = "float32",
    device: str = "auto",
    activation_kind: str = "batch_topk",
    row_cap_multiplier: float = 4.0,
    shared_fraction: float = 0.5,
    cross_reconstruction_coef: float = 0.35,
    shared_alignment_coef: float = 0.05,
    contrastive_coef: float = 0.02,
    contrastive_temperature: float = 0.2,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.1,
    ehr_feature_kinds: Optional[Sequence[str]] = None,
    mixed_likelihood: bool = True,
) -> TopKCrosscoder:
    """Train the GPU multi-view BatchTopK crosscoder."""
    rng_local = rng if rng is not None else np.random.default_rng(0)
    dtype_np = np.dtype(train_dtype)
    if dtype_np != np.dtype("float32"):
        raise ValueError("GPU crosscoder training uses float32 tensors")
    if activation_kind not in {"batch_topk", "topk"}:
        raise ValueError(f"unknown activation_kind {activation_kind!r}")
    if n_steps <= 0 or batch_size <= 0:
        raise ValueError("n_steps and batch_size must be positive")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    if min(cross_reconstruction_coef, shared_alignment_coef, contrastive_coef, weight_decay) < 0:
        raise ValueError("loss coefficients and weight_decay must be non-negative")

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"A and B must share rows, got {A.shape[0]} vs {B.shape[0]}")
    n, m_G = A.shape
    m_E = B.shape[1]
    if n < 2:
        raise ValueError("crosscoder training requires at least two rows")
    d_lat = int(d) if d is not None else 4 * (m_G + m_E)
    if d_lat < 3:
        raise ValueError("d must be at least 3 for shared/private banks")
    if k > d_lat:
        raise ValueError(f"k={k} must be <= d={d_lat}")

    input_kinds = tuple(str(x) for x in (ehr_feature_kinds or ("gaussian",) * m_E))
    model_kinds = input_kinds if mixed_likelihood else ("gaussian",) * m_E
    gaussian_e_np, binary_e_np, count_e_np = _ehr_kind_masks(model_kinds, m_E)
    binary_pos_weight_np = np.ones(m_E, dtype=np.float32)
    if binary_e_np.any():
        prevalence = np.clip(B[:, binary_e_np].mean(axis=0), 1e-4, 1.0 - 1e-4)
        binary_pos_weight_np[binary_e_np] = np.clip(
            (1.0 - prevalence) / prevalence,
            1.0,
            50.0,
        ).astype(np.float32)

    mean_G = A.mean(axis=0)
    std_G = np.where(A.std(axis=0) > 1e-8, A.std(axis=0), 1.0)
    mean_E = B.mean(axis=0)
    std_E = np.where(B.std(axis=0) > 1e-8, B.std(axis=0), 1.0)
    A_z64 = (A - mean_G) / std_G
    B_z64 = (B - mean_E) / std_E
    X64 = np.ascontiguousarray(np.concatenate([A_z64, B_z64], axis=1))
    del A_z64, B_z64

    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None
    if warm_start is None and checkpoint_path_obj is not None and checkpoint_path_obj.is_file():
        warm_start = load_crosscoder_fit_state(checkpoint_path_obj)
    checkpoint_interval = int(checkpoint_every or log_every)
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_every must be positive")

    torch_device = _resolve_device(device)
    torch_dtype = torch.float32
    X = torch.as_tensor(X64, dtype=torch_dtype, device=torch_device)
    B_raw = torch.as_tensor(B, dtype=torch_dtype, device=torch_device)
    gaussian_e = torch.as_tensor(gaussian_e_np, dtype=torch.bool, device=torch_device)
    binary_e = torch.as_tensor(binary_e_np, dtype=torch.bool, device=torch_device)
    count_e = torch.as_tensor(count_e_np, dtype=torch.bool, device=torch_device)
    binary_pos_weight = torch.as_tensor(
        binary_pos_weight_np[binary_e_np],
        dtype=torch_dtype,
        device=torch_device,
    )
    binary_prior_logit = torch.as_tensor(
        np.log(
            np.clip(mean_E[binary_e_np], 1e-4, 1.0 - 1e-4)
            / (1.0 - np.clip(mean_E[binary_e_np], 1e-4, 1.0 - 1e-4))
        ),
        dtype=torch_dtype,
        device=torch_device,
    )
    inv_m_G = 1.0 / float(m_G)
    inv_m_E = 1.0 / float(max(m_E, 1))

    if warm_start is not None:
        train_idx_np = np.asarray(warm_start["train_idx"], dtype=np.int64)
        val_idx_np = np.asarray(warm_start["val_idx"], dtype=np.int64)
    else:
        n_val = int(round(n * validation_fraction))
        if validation_fraction > 0.0:
            n_val = min(max(1, n_val), n - 1)
        perm = rng_local.permutation(n)
        val_idx_np = perm[:n_val] if n_val else np.asarray([], dtype=np.int64)
        train_idx_np = perm[n_val:] if n_val else perm
    train_idx_all = torch.as_tensor(train_idx_np, dtype=torch.long, device=torch_device)
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=torch_device)

    def _as_param(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch_dtype, device=torch_device).clone().detach().requires_grad_(True)

    if warm_start is None:
        latent_bank = _make_latent_bank(d_lat, shared_fraction)
        scale = 1.0 / np.sqrt(m_G + m_E)
        W_e_np = (rng_local.standard_normal((m_G + m_E, d_lat)) * scale).astype(np.float32)
        b_enc_np = np.zeros(d_lat, dtype=np.float32)
        W_d_G_np = W_e_np[:m_G, :].T.copy()
        W_d_E_np = W_e_np[m_G:, :].T.copy()
        W_d_G_np, W_d_E_np = _normalise_decoders(W_d_G_np, W_d_E_np, latent_bank)
        params = {
            "W_e": _as_param(W_e_np),
            "b_enc": _as_param(b_enc_np),
            "W_d_G": _as_param(W_d_G_np),
            "W_d_E": _as_param(W_d_E_np),
        }
        adam_m = {name: torch.zeros_like(value) for name, value in params.items()}
        adam_v = {name: torch.zeros_like(value) for name, value in params.items()}
        steps_since_active = np.zeros(d_lat, dtype=np.int64)
        ever_active = np.zeros(d_lat, dtype=bool)
        start_step = 1
        history = {
            "step": [],
            "loss_main": [],
            "loss_val": [],
            "loss_aux": [],
            "loss_cross": [],
            "loss_align": [],
            "loss_contrastive": [],
            "frac_dead": [],
            "avg_l0_batch": [],
            "frac_active_batch": [],
            "ever_active_count": [],
            "r2_genome_val": [],
            "r2_ehr_gaussian_val": [],
            "brier_ehr_binary_val": [],
            "brier_lift_vs_prior_val": [],
            "r2_ehr_count_logspace_val": [],
            "cross_r2_ehr_gaussian_from_genome_val": [],
            "cross_brier_ehr_binary_from_genome_val": [],
            "cross_brier_lift_vs_prior_from_genome_val": [],
            "cross_r2_genome_from_ehr_val": [],
            "frac_shared_decoder": [],
            "negative_control_margin_val": [],
            "device": str(torch_device),
            "activation_kind": activation_kind,
        }
    else:
        expected = {
            "n": n,
            "m_G": m_G,
            "m_E": m_E,
            "d": d_lat,
            "k": k,
            "batch_size": batch_size,
            "activation_kind": activation_kind,
        }
        mismatches = [
            f"{key}: checkpoint={warm_start[key]} current={val}"
            for key, val in expected.items()
            if warm_start[key] != val
        ]
        if mismatches:
            raise ValueError("crosscoder checkpoint does not match this fit: " + "; ".join(mismatches))
        latent_bank = np.asarray(warm_start["latent_bank"], dtype=np.int8)
        params = {
            "W_e": _as_param(np.asarray(warm_start["W_e"], dtype=np.float32)),
            "b_enc": _as_param(np.asarray(warm_start["b_enc"], dtype=np.float32)),
            "W_d_G": _as_param(np.asarray(warm_start["W_d_G"], dtype=np.float32)),
            "W_d_E": _as_param(np.asarray(warm_start["W_d_E"], dtype=np.float32)),
        }
        adam_m = {
            name: torch.as_tensor(value, dtype=torch_dtype, device=torch_device)
            for name, value in warm_start["adam_m"].items()
        }
        adam_v = {
            name: torch.as_tensor(value, dtype=torch_dtype, device=torch_device)
            for name, value in warm_start["adam_v"].items()
        }
        steps_since_active = np.asarray(warm_start["steps_since_active"], dtype=np.int64).copy()
        ever_active = np.asarray(warm_start["ever_active"], dtype=bool).copy()
        history = dict(warm_start["history"])
        rng_local = np.random.default_rng()
        rng_local.bit_generator.state = warm_start["rng_state"]
        start_step = int(warm_start["step"]) + 1

    bank = torch.as_tensor(latent_bank, dtype=torch.long, device=torch_device)
    shared_mask = bank == BANK_SHARED
    genome_allowed = bank != BANK_EHR_PRIVATE
    ehr_allowed = bank != BANK_GENOME_PRIVATE
    decoder_mask_g = (bank != BANK_EHR_PRIVATE).to(torch_dtype)[:, None]
    decoder_mask_e = (bank != BANK_GENOME_PRIVATE).to(torch_dtype)[:, None]
    encoder_mask = torch.ones((m_G + m_E, d_lat), dtype=torch_dtype, device=torch_device)
    encoder_mask[m_G:, bank == BANK_GENOME_PRIVATE] = 0.0
    encoder_mask[:m_G, bank == BANK_EHR_PRIVATE] = 0.0

    def _decode(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return z @ params["W_d_G"], z @ params["W_d_E"]

    def _loss_genome(a_hat: torch.Tensor, a_target: torch.Tensor) -> torch.Tensor:
        return 0.5 * inv_m_G * torch.mean(torch.sum((a_hat - a_target) ** 2, dim=1))

    def _loss_ehr(
        b_hat: torch.Tensor,
        b_z_target: torch.Tensor,
        b_raw_target: torch.Tensor,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        weights: list[float] = []
        if bool(gaussian_e.any()):
            pieces.append(0.5 * torch.mean((b_hat[:, gaussian_e] - b_z_target[:, gaussian_e]) ** 2))
            weights.append(float(int(gaussian_e.sum().item())) * inv_m_E)
        if bool(binary_e.any()):
            pieces.append(
                F.binary_cross_entropy_with_logits(
                    b_hat[:, binary_e] + binary_prior_logit,
                    b_raw_target[:, binary_e],
                    pos_weight=binary_pos_weight,
                )
            )
            weights.append(float(int(binary_e.sum().item())) * inv_m_E)
        if bool(count_e.any()):
            target = torch.log1p(torch.clamp_min(b_raw_target[:, count_e], 0.0))
            pred = F.softplus(b_hat[:, count_e])
            pieces.append(0.5 * torch.mean((pred - target) ** 2))
            weights.append(float(int(count_e.sum().item())) * inv_m_E)
        if not pieces:
            return torch.zeros((), dtype=torch_dtype, device=torch_device)
        return sum(w * p for w, p in zip(weights, pieces))

    def _standard_aux_loss(a_hat: torch.Tensor, b_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x_hat = torch.cat([a_hat, b_hat], dim=1)
        return 0.5 * torch.mean((x_hat - target) ** 2)

    @torch.no_grad()
    def _project_params() -> None:
        params["W_e"].mul_(encoder_mask)
        params["W_d_G"].mul_(decoder_mask_g)
        params["W_d_E"].mul_(decoder_mask_e)
        norm = torch.sqrt(
            torch.sum(params["W_d_G"] ** 2, dim=1)
            + torch.sum(params["W_d_E"] ** 2, dim=1)
        ).clamp_min(1e-8)
        params["W_d_G"].div_(norm[:, None])
        params["W_d_E"].div_(norm[:, None])

    def _make_state(step: int) -> dict[str, Any]:
        return {
            "step": int(step),
            "target_steps": int(n_steps),
            "n": int(n),
            "m_G": int(m_G),
            "m_E": int(m_E),
            "d": int(d_lat),
            "k": int(k),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "aux_k": int(aux_k),
            "aux_coef": float(aux_coef),
            "dead_steps": int(dead_steps),
            "activation_kind": activation_kind,
            "row_cap_multiplier": float(row_cap_multiplier),
            "shared_fraction": float(shared_fraction),
            "cross_reconstruction_coef": float(cross_reconstruction_coef),
            "shared_alignment_coef": float(shared_alignment_coef),
            "contrastive_coef": float(contrastive_coef),
            "weight_decay": float(weight_decay),
            "validation_fraction": float(validation_fraction),
            "mixed_likelihood": bool(mixed_likelihood),
            "ehr_feature_kinds": model_kinds,
            "latent_bank": latent_bank.copy(),
            "W_e": params["W_e"].detach().cpu().numpy(),
            "b_enc": params["b_enc"].detach().cpu().numpy(),
            "W_d_G": params["W_d_G"].detach().cpu().numpy(),
            "W_d_E": params["W_d_E"].detach().cpu().numpy(),
            "mean_G": mean_G.copy(),
            "std_G": std_G.copy(),
            "mean_E": mean_E.copy(),
            "std_E": std_E.copy(),
            "adam_m": {name: value.detach().cpu().numpy() for name, value in adam_m.items()},
            "adam_v": {name: value.detach().cpu().numpy() for name, value in adam_v.items()},
            "steps_since_active": steps_since_active.copy(),
            "ever_active": ever_active.copy(),
            "train_idx": train_idx_np.copy(),
            "val_idx": val_idx_np.copy(),
            "history": history,
            "rng_state": rng_local.bit_generator.state,
            "train_dtype": str(dtype_np),
        }

    def _write_checkpoint(step: int) -> None:
        if checkpoint_path_obj is None:
            return
        save_crosscoder_fit_state(checkpoint_path_obj, _make_state(step))
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint_path_obj, int(step))

    @torch.no_grad()
    def _eval_on(idx: torch.Tensor) -> dict[str, float]:
        if idx.numel() == 0:
            idx = train_idx_all[: min(train_idx_all.numel(), batch_size)]
        x = X[idx]
        b_raw = B_raw[idx]
        a_t = x[:, :m_G]
        b_t = x[:, m_G:]
        x_a = x.clone()
        x_a[:, m_G:] = 0.0
        x_b = x.clone()
        x_b[:, :m_G] = 0.0
        z_ab, _ = _torch_activate(x @ params["W_e"] + params["b_enc"], k, activation_kind, row_cap_multiplier=row_cap_multiplier)
        z_a, _ = _torch_activate(x_a @ params["W_e"] + params["b_enc"], k, activation_kind, row_cap_multiplier=row_cap_multiplier, allowed=genome_allowed)
        z_b, _ = _torch_activate(x_b @ params["W_e"] + params["b_enc"], k, activation_kind, row_cap_multiplier=row_cap_multiplier, allowed=ehr_allowed)
        a_hat, b_hat = _decode(z_ab)
        a_from_b, _ = _decode(z_b)
        _, b_from_a = _decode(z_a)
        val_loss = _loss_genome(a_hat, a_t) + _loss_ehr(b_hat, b_t, b_raw)
        a_np = a_t.detach().cpu().numpy()
        b_np = b_t.detach().cpu().numpy()
        a_hat_np = a_hat.detach().cpu().numpy()
        b_hat_np = b_hat.detach().cpu().numpy()
        a_from_b_np = a_from_b.detach().cpu().numpy()
        b_from_a_np = b_from_a.detach().cpu().numpy()
        shared = shared_mask
        if bool(shared.any()):
            za = z_a[:, shared]
            zb = z_b[:, shared]
            matched = torch.mean(za * zb)
            perm_idx = torch.arange(zb.shape[0] - 1, -1, -1, device=zb.device)
            shuffled = torch.mean(za * zb[perm_idx])
            margin = (matched - shuffled).detach().cpu().item()
        else:
            margin = float("nan")
        b_raw_np = b_raw.detach().cpu().numpy()
        ehr_metrics = _ehr_recon_metrics(
            b_np, b_raw_np, b_hat_np,
            gaussian=gaussian_e_np, binary=binary_e_np, count=count_e_np,
            mean_E=mean_E,
        )
        cross_ehr = _ehr_recon_metrics(
            b_np, b_raw_np, b_from_a_np,
            gaussian=gaussian_e_np, binary=binary_e_np, count=count_e_np,
            mean_E=mean_E,
        )
        return {
            "loss": float(val_loss.detach().cpu().item()),
            "r2_genome": _r2(a_np, a_hat_np),
            "cross_r2_genome_from_ehr": _r2(a_np, a_from_b_np),
            "r2_ehr_gaussian": float(ehr_metrics["r2_ehr_gaussian"]),
            "brier_ehr_binary": float(ehr_metrics["brier_ehr_binary"]),
            "brier_lift_vs_prior": float(ehr_metrics["brier_lift_vs_prior"]),
            "r2_ehr_count_logspace": float(ehr_metrics["r2_ehr_count_logspace"]),
            "cross_r2_ehr_gaussian_from_genome": float(cross_ehr["r2_ehr_gaussian"]),
            "cross_brier_ehr_binary_from_genome": float(cross_ehr["brier_ehr_binary"]),
            "cross_brier_lift_vs_prior_from_genome": float(cross_ehr["brier_lift_vs_prior"]),
            "negative_control_margin": float(margin),
        }

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_start = time.time()
    _project_params()

    if progress is not None:
        progress(
            "[crosscoder] start "
            f"device={torch_device} n={n} train={train_idx_np.size} val={val_idx_np.size} "
            f"m_G={m_G} m_E={m_E} d={d_lat} k_avg={k} activation={activation_kind} "
            f"banks=shared:{int((latent_bank == BANK_SHARED).sum())},"
            f"genome:{int((latent_bank == BANK_GENOME_PRIVATE).sum())},"
            f"ehr:{int((latent_bank == BANK_EHR_PRIVATE).sum())} "
            f"steps={start_step}-{n_steps} batch={batch_size}"
        )

    for step in range(start_step, n_steps + 1):
        draw = rng_local.integers(0, train_idx_np.size, size=batch_size)
        idx = train_idx_all[torch.as_tensor(draw, dtype=torch.long, device=torch_device)]
        x = X[idx]
        b_raw = B_raw[idx]
        a_t = x[:, :m_G]
        b_t = x[:, m_G:]
        x_a = x.clone()
        x_a[:, m_G:] = 0.0
        x_b = x.clone()
        x_b[:, :m_G] = 0.0

        pre_ab = x @ params["W_e"] + params["b_enc"]
        pre_a = x_a @ params["W_e"] + params["b_enc"]
        pre_b = x_b @ params["W_e"] + params["b_enc"]
        z_ab, mask_ab = _torch_activate(pre_ab, k, activation_kind, row_cap_multiplier=row_cap_multiplier)
        z_a, _ = _torch_activate(pre_a, k, activation_kind, row_cap_multiplier=row_cap_multiplier, allowed=genome_allowed)
        z_b, _ = _torch_activate(pre_b, k, activation_kind, row_cap_multiplier=row_cap_multiplier, allowed=ehr_allowed)

        a_hat, b_hat = _decode(z_ab)
        loss_main = _loss_genome(a_hat, a_t) + _loss_ehr(b_hat, b_t, b_raw)

        a_from_b, _ = _decode(z_b)
        _, b_from_a = _decode(z_a)
        loss_cross = _loss_genome(a_from_b, a_t) + _loss_ehr(b_from_a, b_t, b_raw)

        loss_align = torch.zeros((), dtype=torch_dtype, device=torch_device)
        loss_contrastive = torch.zeros((), dtype=torch_dtype, device=torch_device)
        if bool(shared_mask.any()):
            za_s = z_a[:, shared_mask]
            zb_s = z_b[:, shared_mask]
            zab_s = z_ab[:, shared_mask]
            loss_align = (
                F.mse_loss(za_s, zab_s.detach())
                + F.mse_loss(zb_s, zab_s.detach())
                + F.mse_loss(za_s, zb_s)
            ) / 3.0
            if contrastive_coef > 0.0 and za_s.shape[0] > 1:
                za_n = F.normalize(za_s + 1e-6, dim=1)
                zb_n = F.normalize(zb_s + 1e-6, dim=1)
                logits = (za_n @ zb_n.T) / float(contrastive_temperature)
                labels = torch.arange(logits.shape[0], device=torch_device)
                loss_contrastive = 0.5 * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)
                )

        loss_aux = torch.zeros((), dtype=torch_dtype, device=torch_device)
        dead_mask_np = steps_since_active >= dead_steps
        n_dead = int(dead_mask_np.sum())
        if n_dead > 0 and aux_k > 0 and aux_coef > 0.0:
            dead_allowed = torch.as_tensor(dead_mask_np, dtype=torch.bool, device=torch_device)
            z_aux, _ = _torch_activate(
                pre_ab,
                min(aux_k, n_dead),
                "topk",
                row_cap_multiplier=row_cap_multiplier,
                allowed=dead_allowed,
            )
            a_aux, b_aux = _decode(z_aux)
            residual = x - torch.cat([a_hat.detach(), b_hat.detach()], dim=1)
            loss_aux = _standard_aux_loss(a_aux, b_aux, residual)

        loss = (
            loss_main
            + cross_reconstruction_coef * loss_cross
            + shared_alignment_coef * loss_align
            + contrastive_coef * loss_contrastive
            + aux_coef * loss_aux
        )
        loss.backward()

        active_this_step = mask_ab.detach().any(dim=0).cpu().numpy()
        steps_since_active += 1
        steps_since_active[active_this_step] = 0
        ever_active |= active_this_step

        with torch.no_grad():
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            for name, param in params.items():
                grad = param.grad
                if grad is None:
                    continue
                if weight_decay > 0.0 and name != "b_enc":
                    grad = grad + float(weight_decay) * param
                adam_m[name].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                adam_v[name].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = adam_m[name] / bc1
                v_hat = adam_v[name] / bc2
                param.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps), value=-lr)
                param.grad = None
            _project_params()

        if step == 1 or step == n_steps or step % log_every == 0:
            eval_metrics = _eval_on(val_idx)
            g_sq = torch.sum(params["W_d_G"].detach() ** 2, dim=1).cpu().numpy()
            e_sq = torch.sum(params["W_d_E"].detach() ** 2, dim=1).cpu().numpy()
            share = g_sq / np.maximum(g_sq + e_sq, 1e-8)
            frac_shared_decoder = float(((share >= 0.2) & (share <= 0.8)).mean())
            avg_l0 = float(mask_ab.detach().sum(dim=1).float().mean().cpu().item())
            frac_active_batch = float(active_this_step.mean())
            history["step"].append(int(step))
            history["loss_main"].append(float(loss_main.detach().cpu().item()))
            history["loss_val"].append(float(eval_metrics["loss"]))
            history["loss_aux"].append(float(loss_aux.detach().cpu().item()))
            history["loss_cross"].append(float(loss_cross.detach().cpu().item()))
            history["loss_align"].append(float(loss_align.detach().cpu().item()))
            history["loss_contrastive"].append(float(loss_contrastive.detach().cpu().item()))
            history["frac_dead"].append(float(n_dead) / float(d_lat))
            history["avg_l0_batch"].append(avg_l0)
            history["frac_active_batch"].append(frac_active_batch)
            history["ever_active_count"].append(int(ever_active.sum()))
            history["r2_genome_val"].append(float(eval_metrics["r2_genome"]))
            history["r2_ehr_gaussian_val"].append(float(eval_metrics["r2_ehr_gaussian"]))
            history["brier_ehr_binary_val"].append(float(eval_metrics["brier_ehr_binary"]))
            history["brier_lift_vs_prior_val"].append(float(eval_metrics["brier_lift_vs_prior"]))
            history["r2_ehr_count_logspace_val"].append(float(eval_metrics["r2_ehr_count_logspace"]))
            history["cross_r2_ehr_gaussian_from_genome_val"].append(float(eval_metrics["cross_r2_ehr_gaussian_from_genome"]))
            history["cross_brier_ehr_binary_from_genome_val"].append(float(eval_metrics["cross_brier_ehr_binary_from_genome"]))
            history["cross_brier_lift_vs_prior_from_genome_val"].append(float(eval_metrics["cross_brier_lift_vs_prior_from_genome"]))
            history["cross_r2_genome_from_ehr_val"].append(float(eval_metrics["cross_r2_genome_from_ehr"]))
            history["frac_shared_decoder"].append(frac_shared_decoder)
            history["negative_control_margin_val"].append(float(eval_metrics["negative_control_margin"]))
            if progress is not None:
                progress(
                    f"[crosscoder] step={step}/{n_steps} "
                    f"loss_main={history['loss_main'][-1]:.5f} "
                    f"loss_val={history['loss_val'][-1]:.5f} "
                    f"loss_cross={history['loss_cross'][-1]:.5f} "
                    f"loss_align={history['loss_align'][-1]:.5f} "
                    f"r2_G={history['r2_genome_val'][-1]:.3f} "
                    f"r2_E_gauss={history['r2_ehr_gaussian_val'][-1]:.3f} "
                    f"brier_E_bin={history['brier_ehr_binary_val'][-1]:.4f} "
                    f"brier_lift={history['brier_lift_vs_prior_val'][-1]:+.4f} "
                    f"cross_r2_G_from_E={history['cross_r2_genome_from_ehr_val'][-1]:.3f} "
                    f"avg_l0={avg_l0:.2f} dead={history['frac_dead'][-1]:.3f} "
                    f"neg_margin={history['negative_control_margin_val'][-1]:.5f} "
                    f"elapsed={time.time() - t_start:.1f}s"
                )
        if step == n_steps or step % checkpoint_interval == 0:
            _write_checkpoint(step)

    if start_step > n_steps:
        _write_checkpoint(start_step - 1)

    if progress is not None:
        progress(
            "[crosscoder] done "
            f"device={torch_device} steps={n_steps} elapsed={time.time() - t_start:.1f}s "
            f"final_loss_val={history['loss_val'][-1] if history['loss_val'] else float('nan'):.5f} "
            f"ever_active={int(ever_active.sum())}/{d_lat}"
        )

    return TopKCrosscoder(
        W_e=params["W_e"].detach().cpu().numpy().astype(np.float64),
        b_enc=params["b_enc"].detach().cpu().numpy().astype(np.float64),
        W_d_G=params["W_d_G"].detach().cpu().numpy().astype(np.float64),
        W_d_E=params["W_d_E"].detach().cpu().numpy().astype(np.float64),
        mean_G=mean_G,
        std_G=std_G,
        mean_E=mean_E,
        std_E=std_E,
        k=int(k),
        latent_bank=latent_bank.astype(np.int8, copy=False),
        activation_kind=activation_kind,
        history=history,
        ehr_feature_kinds=model_kinds,
        device=str(torch_device),
    )
