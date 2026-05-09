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

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

CROSSCODER_FIT_STATE_VERSION = 1


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
# Fit-state checkpointing
# ---------------------------------------------------------------------------


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists():
        part.unlink()
    with part.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(part, path)


def save_crosscoder_fit_state(path: str | os.PathLike, state: dict[str, Any]) -> None:
    """Persist a resumable crosscoder training state without pickle."""
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
        "stream_dropout_prob": float(state["stream_dropout_prob"]),
        "cross_stream_align_coef": float(state["cross_stream_align_coef"]),
        "decoder_balance_coef": float(state["decoder_balance_coef"]),
        "train_dtype": str(np.dtype(state["train_dtype"])),
    }
    _atomic_npz(
        Path(path),
        meta_json=np.array(json.dumps(meta)),
        history_json=np.array(json.dumps(state["history"])),
        rng_state_json=np.array(json.dumps(state["rng_state"])),
        W_e=np.asarray(state["W_e"]),
        b_enc=np.asarray(state["b_enc"]),
        W_d_G=np.asarray(state["W_d_G"]),
        W_d_E=np.asarray(state["W_d_E"]),
        mean_G=np.asarray(state["mean_G"], dtype=np.float64),
        std_G=np.asarray(state["std_G"], dtype=np.float64),
        mean_E=np.asarray(state["mean_E"], dtype=np.float64),
        std_E=np.asarray(state["std_E"], dtype=np.float64),
        adam_m_W_e=np.asarray(state["adam_m"]["W_e"]),
        adam_m_b_enc=np.asarray(state["adam_m"]["b_enc"]),
        adam_m_W_d_G=np.asarray(state["adam_m"]["W_d_G"]),
        adam_m_W_d_E=np.asarray(state["adam_m"]["W_d_E"]),
        adam_v_W_e=np.asarray(state["adam_v"]["W_e"]),
        adam_v_b_enc=np.asarray(state["adam_v"]["b_enc"]),
        adam_v_W_d_G=np.asarray(state["adam_v"]["W_d_G"]),
        adam_v_W_d_E=np.asarray(state["adam_v"]["W_d_E"]),
        steps_since_active=np.asarray(state["steps_since_active"], dtype=np.int64),
        ever_active=np.asarray(state["ever_active"], dtype=bool),
    )


def load_crosscoder_fit_state(path: str | os.PathLike) -> dict[str, Any]:
    """Load a crosscoder fit state written by :func:`save_crosscoder_fit_state`."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"].item()))
        version = int(meta.get("version", -1))
        if version != CROSSCODER_FIT_STATE_VERSION:
            raise ValueError(
                f"crosscoder checkpoint version {version} != "
                f"{CROSSCODER_FIT_STATE_VERSION}"
            )
        return {
            "step": int(meta["step"]),
            "target_steps": int(meta["target_steps"]),
            "n": int(meta["n"]),
            "m_G": int(meta["m_G"]),
            "m_E": int(meta["m_E"]),
            "d": int(meta["d"]),
            "k": int(meta["k"]),
            "batch_size": int(meta["batch_size"]),
            "lr": float(meta["lr"]),
            "aux_k": int(meta["aux_k"]),
            "aux_coef": float(meta["aux_coef"]),
            "dead_steps": int(meta["dead_steps"]),
            "stream_dropout_prob": float(meta["stream_dropout_prob"]),
            "cross_stream_align_coef": float(meta["cross_stream_align_coef"]),
            "decoder_balance_coef": float(meta["decoder_balance_coef"]),
            "train_dtype": str(meta["train_dtype"]),
            "history": json.loads(str(z["history_json"].item())),
            "rng_state": json.loads(str(z["rng_state_json"].item())),
            "W_e": np.asarray(z["W_e"]),
            "b_enc": np.asarray(z["b_enc"]),
            "W_d_G": np.asarray(z["W_d_G"]),
            "W_d_E": np.asarray(z["W_d_E"]),
            "mean_G": z["mean_G"].astype(np.float64),
            "std_G": z["std_G"].astype(np.float64),
            "mean_E": z["mean_E"].astype(np.float64),
            "std_E": z["std_E"].astype(np.float64),
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
            "steps_since_active": z["steps_since_active"].astype(np.int64),
            "ever_active": z["ever_active"].astype(bool),
        }


def _validate_warm_start(
    state: dict[str, Any],
    *,
    n: int,
    m_G: int,
    m_E: int,
    d_lat: int,
    k: int,
    batch_size: int,
    lr: float,
    aux_k: int,
    aux_coef: float,
    dead_steps: int,
    stream_dropout_prob: float,
    cross_stream_align_coef: float,
    decoder_balance_coef: float,
    train_dtype: np.dtype,
    mean_G: np.ndarray,
    std_G: np.ndarray,
    mean_E: np.ndarray,
    std_E: np.ndarray,
) -> None:
    expected = {
        "n": n,
        "m_G": m_G,
        "m_E": m_E,
        "d": d_lat,
        "k": k,
        "batch_size": batch_size,
        "aux_k": aux_k,
        "dead_steps": dead_steps,
    }
    mismatches = [
        f"{name}: checkpoint={int(state[name])} current={value}"
        for name, value in expected.items()
        if int(state[name]) != int(value)
    ]
    if mismatches:
        raise ValueError(
            "crosscoder checkpoint does not match this fit: "
            + "; ".join(mismatches)
        )
    if not np.isclose(float(state["lr"]), float(lr), rtol=0.0, atol=1e-15):
        raise ValueError(
            "crosscoder checkpoint lr does not match this fit: "
            f"checkpoint={float(state['lr'])} current={float(lr)}"
        )
    if not np.isclose(float(state["aux_coef"]), float(aux_coef), rtol=0.0, atol=1e-15):
        raise ValueError(
            "crosscoder checkpoint aux_coef does not match this fit: "
            f"checkpoint={float(state['aux_coef'])} current={float(aux_coef)}"
        )
    for name, current in (
        ("stream_dropout_prob", stream_dropout_prob),
        ("cross_stream_align_coef", cross_stream_align_coef),
        ("decoder_balance_coef", decoder_balance_coef),
    ):
        if not np.isclose(float(state[name]), float(current), rtol=0.0, atol=1e-15):
            raise ValueError(
                f"crosscoder checkpoint {name} does not match this fit: "
                f"checkpoint={float(state[name])} current={float(current)}"
            )
    checkpoint_dtype = np.dtype(state["train_dtype"])
    if checkpoint_dtype != np.dtype(train_dtype):
        raise ValueError(
            "crosscoder checkpoint train_dtype does not match this fit: "
            f"checkpoint={checkpoint_dtype} current={np.dtype(train_dtype)}"
        )
    for name, got, want in (
        ("mean_G", np.asarray(state["mean_G"]), mean_G),
        ("std_G", np.asarray(state["std_G"]), std_G),
        ("mean_E", np.asarray(state["mean_E"]), mean_E),
        ("std_E", np.asarray(state["std_E"]), std_E),
    ):
        if got.shape != want.shape or not np.allclose(got, want, rtol=0.0, atol=1e-12):
            raise ValueError(f"crosscoder checkpoint {name} does not match this fit")


def _make_fit_state(
    *,
    step: int,
    target_steps: int,
    n: int,
    m_G: int,
    m_E: int,
    d_lat: int,
    k: int,
    batch_size: int,
    lr: float,
    aux_k: int,
    aux_coef: float,
    dead_steps: int,
    stream_dropout_prob: float,
    cross_stream_align_coef: float,
    decoder_balance_coef: float,
    params: dict[str, np.ndarray],
    mean_G: np.ndarray,
    std_G: np.ndarray,
    mean_E: np.ndarray,
    std_E: np.ndarray,
    m_state: dict[str, np.ndarray],
    v_state: dict[str, np.ndarray],
    steps_since_active: np.ndarray,
    ever_active: np.ndarray,
    history: dict,
    rng: np.random.Generator,
    train_dtype: np.dtype,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "target_steps": int(target_steps),
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
        "stream_dropout_prob": float(stream_dropout_prob),
        "cross_stream_align_coef": float(cross_stream_align_coef),
        "decoder_balance_coef": float(decoder_balance_coef),
        "W_e": params["W_e"].copy(),
        "b_enc": params["b_enc"].copy(),
        "W_d_G": params["W_d_G"].copy(),
        "W_d_E": params["W_d_E"].copy(),
        "mean_G": mean_G.copy(),
        "std_G": std_G.copy(),
        "mean_E": mean_E.copy(),
        "std_E": std_E.copy(),
        "adam_m": {name: value.copy() for name, value in m_state.items()},
        "adam_v": {name: value.copy() for name, value in v_state.items()},
        "steps_since_active": steps_since_active.copy(),
        "ever_active": ever_active.copy(),
        "history": {name: list(values) for name, values in history.items()},
        "rng_state": rng.bit_generator.state,
        "train_dtype": str(np.dtype(train_dtype)),
    }


def _eval_reconstruction_metrics(
    X_eval: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    m_G: int,
    k: int,
    inv_m_G: float,
    inv_m_E: float,
) -> dict[str, float]:
    a_target = X_eval[:, :m_G]
    b_target = X_eval[:, m_G:]
    pre = X_eval @ params["W_e"] + params["b_enc"]
    z, _mask = _topk_per_row(pre, k)
    a_hat = z @ params["W_d_G"]
    b_hat = z @ params["W_d_E"]
    res_G = a_hat - a_target
    res_E = b_hat - b_target
    ss_res_g = float(np.sum(res_G.astype(np.float64) ** 2))
    ss_res_e = float(np.sum(res_E.astype(np.float64) ** 2))
    ss_tot_g = float(np.sum(a_target.astype(np.float64) ** 2))
    ss_tot_e = float(np.sum(b_target.astype(np.float64) ** 2))
    loss = 0.5 * (
        inv_m_G * float(np.mean(np.sum(res_G ** 2, axis=1)))
        + inv_m_E * float(np.mean(np.sum(res_E ** 2, axis=1)))
    )
    return {
        "loss": loss,
        "r2_genome": 1.0 - ss_res_g / ss_tot_g if ss_tot_g > 0.0 else float("nan"),
        "r2_ehr": 1.0 - ss_res_e / ss_tot_e if ss_tot_e > 0.0 else float("nan"),
        "frac_active": float((z > 0).any(axis=0).mean()),
    }


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
    progress: Optional[Callable[[str], None]] = None,
    warm_start: Optional[dict[str, Any]] = None,
    checkpoint_path: Optional[str | os.PathLike] = None,
    checkpoint_every: Optional[int] = None,
    checkpoint_callback: Optional[Callable[[Path, int], None]] = None,
    train_dtype: str | np.dtype = "float32",
    stream_dropout_prob: float = 0.0,
    cross_stream_align_coef: float = 0.0,
    decoder_balance_coef: float = 0.0,
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
    warm_start : dict, optional
        Fit state from :func:`load_crosscoder_fit_state`.
    checkpoint_path : path-like, optional
        Local file used for periodic fit-state checkpoints. If it already
        exists and ``warm_start`` is not supplied, training resumes from it.
    checkpoint_every : int, optional
        Step interval for writing ``checkpoint_path``. Defaults to
        ``log_every``.
    checkpoint_callback : callable, optional
        Called as ``callback(path, step)`` after each checkpoint write.
    train_dtype : {"float32", "float64"}
        Numeric dtype used for the minibatch optimizer state. ``float32`` is
        materially faster and smaller for the biobank-scale panel path.
    stream_dropout_prob : float
        Per-row probability of zeroing exactly one input stream before
        encoding while still reconstructing both streams. This forces
        latents to carry cross-stream predictive signal instead of splitting
        into two independent autoencoders.
    cross_stream_align_coef : float
        Penalty weight for aligning genome-only and EHR-only encoder
        pre-activations on the same participants.
    decoder_balance_coef : float
        Penalty weight for nudging each decoder's genome share toward 0.5.
        Keep this mild so stream-specific features can still exist.

    Returns
    -------
    TopKCrosscoder
        Trained model. The decoder columns satisfy the joint unit-norm
        constraint at return time.
    """
    rng_local = rng if rng is not None else np.random.default_rng(0)
    dtype = np.dtype(train_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"train_dtype must be float32 or float64, got {dtype}")
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
    stream_dropout_prob = float(stream_dropout_prob)
    cross_stream_align_coef = float(cross_stream_align_coef)
    decoder_balance_coef = float(decoder_balance_coef)
    if not 0.0 <= stream_dropout_prob < 1.0:
        raise ValueError(
            "stream_dropout_prob must be in [0, 1), "
            f"got {stream_dropout_prob}"
        )
    if cross_stream_align_coef < 0.0 or decoder_balance_coef < 0.0:
        raise ValueError(
            "cross_stream_align_coef and decoder_balance_coef must be non-negative"
        )

    # --- standardisation -------------------------------------------------
    mean_G = A.mean(axis=0)
    std_G = A.std(axis=0)
    std_G = np.where(std_G > 1e-8, std_G, 1.0)
    mean_E = B.mean(axis=0)
    std_E = B.std(axis=0)
    std_E = np.where(std_E > 1e-8, std_E, 1.0)
    A_z64 = (A - mean_G) / std_G
    B_z64 = (B - mean_E) / std_E
    X = np.ascontiguousarray(np.concatenate([A_z64, B_z64], axis=1), dtype=dtype)
    del A_z64, B_z64
    A_z = X[:, :m_G]
    B_z = X[:, m_G:]
    m = m_G + m_E

    checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None
    if warm_start is None and checkpoint_path_obj is not None and checkpoint_path_obj.is_file():
        warm_start = load_crosscoder_fit_state(checkpoint_path_obj)
    checkpoint_interval = int(checkpoint_every or log_every)
    if checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_every must be positive, got {checkpoint_interval}"
        )

    # --- initialisation / resume -----------------------------------------
    start_step = 1
    if warm_start is None:
        # Encoder: scaled Gaussian. Decoder: take the encoder transpose split
        # across the two streams, then apply the joint unit-norm projection.
        scale = 1.0 / np.sqrt(m)
        W_e = (rng_local.standard_normal((m, d_lat)) * scale).astype(dtype)
        b_enc = np.zeros(d_lat, dtype=dtype)
        W_d_G = W_e[:m_G, :].T.copy()
        W_d_E = W_e[m_G:, :].T.copy()
        W_d_G, W_d_E = _normalise_decoders(W_d_G, W_d_E)
        params = {
            "W_e": W_e,
            "b_enc": b_enc,
            "W_d_G": W_d_G,
            "W_d_E": W_d_E,
        }
        m_state = {p: np.zeros_like(v) for p, v in params.items()}
        v_state = {p: np.zeros_like(v) for p, v in params.items()}
        steps_since_active = np.zeros(d_lat, dtype=np.int64)
        history = {
            "step": [],
            "loss_main": [],
            "loss_eval": [],
            "loss_aux": [],
            "loss_align": [],
            "loss_balance": [],
            "frac_dead": [],
            "frac_active_batch": [],
            "ever_active_count": [],
            "r2_genome_eval": [],
            "r2_ehr_eval": [],
            "frac_shared_decoder": [],
        }
        ever_active = np.zeros(d_lat, dtype=bool)
    else:
        _validate_warm_start(
            warm_start,
            n=n,
            m_G=m_G,
            m_E=m_E,
            d_lat=d_lat,
            k=k,
            batch_size=batch_size,
            lr=lr,
            aux_k=aux_k,
            aux_coef=aux_coef,
            dead_steps=dead_steps,
            stream_dropout_prob=stream_dropout_prob,
            cross_stream_align_coef=cross_stream_align_coef,
            decoder_balance_coef=decoder_balance_coef,
            train_dtype=dtype,
            mean_G=mean_G,
            std_G=std_G,
            mean_E=mean_E,
            std_E=std_E,
        )
        params = {
            "W_e": np.asarray(warm_start["W_e"], dtype=dtype).copy(),
            "b_enc": np.asarray(warm_start["b_enc"], dtype=dtype).copy(),
            "W_d_G": np.asarray(warm_start["W_d_G"], dtype=dtype).copy(),
            "W_d_E": np.asarray(warm_start["W_d_E"], dtype=dtype).copy(),
        }
        m_state = {
            name: np.asarray(value, dtype=dtype).copy()
            for name, value in warm_start["adam_m"].items()
        }
        v_state = {
            name: np.asarray(value, dtype=dtype).copy()
            for name, value in warm_start["adam_v"].items()
        }
        steps_since_active = np.asarray(
            warm_start["steps_since_active"], dtype=np.int64
        ).copy()
        history = {
            name: list(warm_start["history"].get(name, []))
            for name in (
                "step",
                "loss_main",
                "loss_eval",
                "loss_aux",
                "loss_align",
                "loss_balance",
                "frac_dead",
                "frac_active_batch",
                "ever_active_count",
                "r2_genome_eval",
                "r2_ehr_eval",
                "frac_shared_decoder",
            )
        }
        ever_active = np.asarray(warm_start["ever_active"], dtype=bool).copy()
        rng_local = np.random.default_rng()
        rng_local.bit_generator.state = warm_start["rng_state"]
        start_step = int(warm_start["step"]) + 1
        params["W_d_G"], params["W_d_E"] = _normalise_decoders(
            params["W_d_G"], params["W_d_E"]
        )

    def _write_checkpoint(step: int) -> None:
        if checkpoint_path_obj is None:
            return
        state = _make_fit_state(
            step=step,
            target_steps=n_steps,
            n=n,
            m_G=m_G,
            m_E=m_E,
            d_lat=d_lat,
            k=k,
            batch_size=batch_size,
            lr=lr,
            aux_k=aux_k,
            aux_coef=aux_coef,
            dead_steps=dead_steps,
            stream_dropout_prob=stream_dropout_prob,
            cross_stream_align_coef=cross_stream_align_coef,
            decoder_balance_coef=decoder_balance_coef,
            params=params,
            mean_G=mean_G,
            std_G=std_G,
            mean_E=mean_E,
            std_E=std_E,
            m_state=m_state,
            v_state=v_state,
            steps_since_active=steps_since_active,
            ever_active=ever_active,
            history=history,
            rng=rng_local,
            train_dtype=dtype,
        )
        save_crosscoder_fit_state(checkpoint_path_obj, state)
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint_path_obj, int(step))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    inv_m_G = 1.0 / m_G
    inv_m_E = 1.0 / m_E
    eval_n = min(n, max(batch_size, min(4096, n)))
    eval_idx = np.linspace(0, n - 1, num=eval_n, dtype=np.int64)
    X_eval = X[eval_idx]

    t_start = time.time()
    if progress is not None:
        if start_step > 1:
            progress(
                "[crosscoder] warm start "
                f"checkpoint_step={start_step - 1} target_steps={n_steps}"
            )
        if start_step <= n_steps:
            progress(
                "[crosscoder] start "
                f"n={n} m_G={m_G} m_E={m_E} d={d_lat} k={k} "
                f"steps={start_step}-{n_steps} batch={batch_size} lr={lr:g} "
                f"aux_k={aux_k} dead_steps={dead_steps} dtype={dtype} "
                f"stream_dropout={stream_dropout_prob:.2f} "
                f"align_coef={cross_stream_align_coef:g} "
                f"balance_coef={decoder_balance_coef:g}"
            )
        else:
            progress(
                "[crosscoder] checkpoint complete "
                f"completed_steps={start_step - 1} target_steps={n_steps}"
            )

    for step in range(start_step, n_steps + 1):
        idx = rng_local.integers(0, n, size=batch_size)
        x_batch = X[idx]
        a_target = A_z[idx]
        b_target = B_z[idx]
        bs = x_batch.shape[0]
        if stream_dropout_prob > 0.0:
            x_encode = x_batch.copy()
            drop = rng_local.random(bs) < stream_dropout_prob
            if bool(drop.any()):
                drop_genome = drop & (rng_local.random(bs) < 0.5)
                drop_ehr = drop & ~drop_genome
                x_encode[drop_genome, :m_G] = 0.0
                x_encode[drop_ehr, m_G:] = 0.0
        else:
            x_encode = x_batch

        # ---- forward (main path) ---------------------------------------
        pre = x_encode @ params["W_e"] + params["b_enc"]
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
        d_a_hat = (inv_m_G / bs) * res_G
        d_b_hat = (inv_m_E / bs) * res_E

        g_W_d_G = z.T @ d_a_hat
        g_W_d_E = z.T @ d_b_hat

        d_z = d_a_hat @ params["W_d_G"].T + d_b_hat @ params["W_d_E"].T
        d_pre = np.where(mask, d_z, 0.0)

        g_W_e = x_encode.T @ d_pre
        g_b_enc = d_pre.sum(axis=0)
        loss_align = 0.0
        if cross_stream_align_coef > 0.0:
            pre_g = a_target @ params["W_e"][:m_G, :] + params["b_enc"]
            pre_e = b_target @ params["W_e"][m_G:, :] + params["b_enc"]
            diff_pre = pre_g - pre_e
            loss_align = 0.5 * cross_stream_align_coef * float(
                np.mean(diff_pre ** 2)
            )
            d_diff = (cross_stream_align_coef / (bs * d_lat)) * diff_pre
            g_W_e[:m_G, :] = g_W_e[:m_G, :] + a_target.T @ d_diff
            g_W_e[m_G:, :] = g_W_e[m_G:, :] - b_target.T @ d_diff

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
            loss_aux = 0.5 * (
                inv_m_G * np.mean(np.sum(aux_G ** 2, axis=1))
                + inv_m_E * np.mean(np.sum(aux_E ** 2, axis=1))
            )

            d_e_hat_G = (aux_coef * inv_m_G / bs) * aux_G
            d_e_hat_E = (aux_coef * inv_m_E / bs) * aux_E

            g_W_d_G = g_W_d_G + z_aux.T @ d_e_hat_G
            g_W_d_E = g_W_d_E + z_aux.T @ d_e_hat_E

            d_z_aux = d_e_hat_G @ params["W_d_G"].T + d_e_hat_E @ params["W_d_E"].T
            d_pre_aux = np.where(mask_aux, d_z_aux, 0.0)
            g_W_e = g_W_e + x_encode.T @ d_pre_aux
            g_b_enc = g_b_enc + d_pre_aux.sum(axis=0)
        loss_balance = 0.0
        if decoder_balance_coef > 0.0:
            g_sq = np.sum(params["W_d_G"] ** 2, axis=1)
            e_sq = np.sum(params["W_d_E"] ** 2, axis=1)
            total_sq = np.maximum(g_sq + e_sq, np.asarray(1e-8, dtype=dtype))
            share = g_sq / total_sq
            delta_share = share - 0.5
            loss_balance = decoder_balance_coef * float(np.mean(delta_share ** 2))
            denom = total_sq ** 2
            scale_bal = (2.0 * decoder_balance_coef / d_lat) * delta_share
            g_W_d_G = g_W_d_G + (
                scale_bal[:, None]
                * (2.0 * params["W_d_G"] * e_sq[:, None] / denom[:, None])
            )
            g_W_d_E = g_W_d_E - (
                scale_bal[:, None]
                * (2.0 * params["W_d_E"] * g_sq[:, None] / denom[:, None])
            )

        # ---- update dead-feature tracker -------------------------------
        active_this_step = mask.any(axis=0)
        steps_since_active += 1
        steps_since_active[active_this_step] = 0
        ever_active |= active_this_step

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
            g = np.asarray(g, dtype=dtype)
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
            frac_active_batch = float(active_this_step.mean())
            eval_metrics = _eval_reconstruction_metrics(
                X_eval,
                params,
                m_G=m_G,
                k=k,
                inv_m_G=inv_m_G,
                inv_m_E=inv_m_E,
            )
            history["step"].append(step)
            history["loss_main"].append(float(loss_main))
            history["loss_eval"].append(float(eval_metrics["loss"]))
            history["loss_aux"].append(float(loss_aux))
            history["loss_align"].append(float(loss_align))
            history["loss_balance"].append(float(loss_balance))
            history["frac_dead"].append(float(n_dead) / d_lat)
            history["frac_active_batch"].append(frac_active_batch)
            history["ever_active_count"].append(int(ever_active.sum()))
            history["r2_genome_eval"].append(float(eval_metrics["r2_genome"]))
            history["r2_ehr_eval"].append(float(eval_metrics["r2_ehr"]))
            g_sq_diag = np.sum(params["W_d_G"] ** 2, axis=1)
            e_sq_diag = np.sum(params["W_d_E"] ** 2, axis=1)
            share_diag = g_sq_diag / np.maximum(g_sq_diag + e_sq_diag, 1e-8)
            frac_shared_decoder = float(
                ((share_diag >= 0.2) & (share_diag <= 0.8)).mean()
            )
            history["frac_shared_decoder"].append(frac_shared_decoder)
            if progress is not None:
                elapsed = time.time() - t_start
                progress(
                    f"[crosscoder] step={step}/{n_steps} "
                    f"loss_main={float(loss_main):.5f} "
                    f"loss_eval={float(eval_metrics['loss']):.5f} "
                    f"loss_aux={float(loss_aux):.5f} "
                    f"loss_align={float(loss_align):.5f} "
                    f"loss_balance={float(loss_balance):.5f} "
                    f"r2_G={float(eval_metrics['r2_genome']):.3f} "
                    f"r2_E={float(eval_metrics['r2_ehr']):.3f} "
                    f"shared_decoder={frac_shared_decoder:.3f} "
                    f"frac_dead={float(n_dead) / d_lat:.3f} "
                    f"frac_active_batch={frac_active_batch:.3f} "
                    f"ever_active={int(ever_active.sum())}/{d_lat} "
                    f"elapsed={elapsed:.1f}s"
                )
        if step == n_steps or step % checkpoint_interval == 0:
            _write_checkpoint(step)

    if progress is not None:
        elapsed = time.time() - t_start
        final_loss_main = history["loss_main"][-1] if history["loss_main"] else float("nan")
        final_loss_eval = history["loss_eval"][-1] if history["loss_eval"] else float("nan")
        final_loss_aux = history["loss_aux"][-1] if history["loss_aux"] else float("nan")
        final_loss_align = history["loss_align"][-1] if history["loss_align"] else float("nan")
        final_loss_balance = history["loss_balance"][-1] if history["loss_balance"] else float("nan")
        final_frac_dead = history["frac_dead"][-1] if history["frac_dead"] else float("nan")
        final_r2_g = history["r2_genome_eval"][-1] if history["r2_genome_eval"] else float("nan")
        final_r2_e = history["r2_ehr_eval"][-1] if history["r2_ehr_eval"] else float("nan")
        final_frac_shared = history["frac_shared_decoder"][-1] if history["frac_shared_decoder"] else float("nan")
        progress(
            "[crosscoder] done "
            f"steps={n_steps} elapsed={elapsed:.1f}s "
            f"final_loss_main={final_loss_main:.5f} "
            f"final_loss_eval={final_loss_eval:.5f} "
            f"final_loss_aux={final_loss_aux:.5f} "
            f"final_loss_align={final_loss_align:.5f} "
            f"final_loss_balance={final_loss_balance:.5f} "
            f"final_r2_G={final_r2_g:.3f} "
            f"final_r2_E={final_r2_e:.3f} "
            f"final_shared_decoder={final_frac_shared:.3f} "
            f"final_frac_dead={final_frac_dead:.3f} "
            f"ever_active={int(ever_active.sum())}/{d_lat}"
        )
    if start_step > n_steps:
        _write_checkpoint(start_step - 1)

    return TopKCrosscoder(
        W_e=params["W_e"].astype(np.float64),
        b_enc=params["b_enc"].astype(np.float64),
        W_d_G=params["W_d_G"].astype(np.float64),
        W_d_E=params["W_d_E"].astype(np.float64),
        mean_G=mean_G,
        std_G=std_G,
        mean_E=mean_E,
        std_E=std_E,
        k=k,
        history=history,
    )
