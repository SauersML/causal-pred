"""Integration of TopK crosscoder features into the causal-pred pipeline.

This module is the bridge between the dictionary-learning side of the
project (:mod:`causal_pred.genscore.crosscoder`) and the causal stack
(:mod:`causal_pred.data`, :mod:`causal_pred.dagslam`,
:mod:`causal_pred.mcmc`, :mod:`causal_pred.gam`). Its job is to:

  1. **Align** the (genome PRS, EHR) panels with the participant order of
     a cohort :class:`SyntheticDataset`.
  2. **Train** a TopK crosscoder on the aligned panels.
  3. **Select** which learned features to promote into the DAG. The
     selection rule is structural, not semantic: features whose decoder
     norm is meaningfully spread across both streams (cross-modal
     coherence) and which actually fire for enough participants are kept.
  4. **Augment** the dataset by appending the selected features as new
     continuous nodes. The downstream MrDAG / DAGSLAM / structure-MCMC /
     survival-GAM stages then characterise the features by their inferred
     causal position and measured hazard -- no labels required.

Nothing about steps 3 and 4 depends on auto-interp. The features speak
through the DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.cohort import EhrPanel
from ..data.synthetic import SyntheticDataset
from .crosscoder import (
    TopKCrosscoder,
    encode,
    feature_stream_share,
    train_crosscoder,
)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignedPanels:
    """Row-aligned (genome, EHR) activation matrices for the crosscoder.

    Rows are in ``person_id`` order, restricted to participants present in
    both panels.
    """

    A: np.ndarray              # (n, m_G)
    B: np.ndarray              # (n, m_E)
    person_id: np.ndarray      # (n,) string


def align_panels_by_iid(
    person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
) -> AlignedPanels:
    """Inner-join the cohort, the PRS frame, and the EHR panel by IID.

    Parameters
    ----------
    person_ids : sequence of str
        Cohort participant order (length n0). The output preserves this
        order, restricted to participants present in both panels.
    prs_df : pandas.DataFrame
        DataFrame indexed by IID (string), one float column per polygenic
        score. Output of :func:`causal_pred.data.polygenic.score_panel`.
    ehr_panel : EhrPanel
        EHR-stream panel from :func:`causal_pred.data.cohort.build_ehr_panel`.

    Returns
    -------
    AlignedPanels
    """
    pid = np.asarray([str(p) for p in person_ids])
    ehr_pid = np.asarray([str(p) for p in ehr_panel.person_id])
    ehr_index = {p: i for i, p in enumerate(ehr_pid.tolist())}

    prs_index = {str(ix): i for i, ix in enumerate(prs_df.index.tolist())}

    keep_rows: list[int] = []
    keep_pids: list[str] = []
    for p in pid.tolist():
        if p in prs_index and p in ehr_index:
            keep_rows.append(0)  # placeholder, real indices computed below
            keep_pids.append(p)

    if not keep_pids:
        raise ValueError(
            "no participants are present in both the PRS frame and the EHR panel"
        )

    A_full = prs_df.to_numpy(dtype=np.float64)
    A = A_full[[prs_index[p] for p in keep_pids], :]
    B = ehr_panel.matrix[[ehr_index[p] for p in keep_pids], :]

    return AlignedPanels(
        A=np.ascontiguousarray(A, dtype=np.float64),
        B=np.ascontiguousarray(B, dtype=np.float64),
        person_id=np.asarray(keep_pids),
    )


# ---------------------------------------------------------------------------
# Training facade
# ---------------------------------------------------------------------------


def fit_panel_crosscoder(
    panels: AlignedPanels,
    *,
    d: Optional[int] = None,
    k: int = 32,
    n_steps: int = 4000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    aux_k: int = 64,
    aux_coef: float = 1.0 / 32.0,
    dead_steps: int = 200,
    rng: Optional[np.random.Generator] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> TopKCrosscoder:
    """Train the TopK crosscoder on aligned panels, with biobank-friendly
    defaults (longer schedule, smaller learning rate). Thin wrapper around
    :func:`train_crosscoder`."""
    return train_crosscoder(
        A=panels.A,
        B=panels.B,
        d=d,
        k=k,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        aux_k=aux_k,
        aux_coef=aux_coef,
        dead_steps=dead_steps,
        rng=rng,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


@dataclass
class FeatureSelection:
    """Indices and metadata of crosscoder features promoted to DAG nodes."""

    indices: np.ndarray             # (k_promote,) int
    names: Tuple[str, ...]          # length k_promote, e.g. "feat_0042"
    genome_share: np.ndarray        # (k_promote,) float, in [0, 1]
    activation_rate: np.ndarray     # (k_promote,) float, fraction of
                                    # participants with z[:, j] > 0


def _feature_activation_rate(z: np.ndarray) -> np.ndarray:
    """Per-feature fraction of rows where the activation is strictly positive."""
    return (z > 0).mean(axis=0)


def select_shared_features(
    model: TopKCrosscoder,
    panels: AlignedPanels,
    *,
    n_promote: int = 32,
    genome_share_min: float = 0.2,
    genome_share_max: float = 0.8,
    min_activation_rate: float = 0.01,
    name_prefix: str = "feat",
) -> FeatureSelection:
    """Pick the most promotable features for downstream DAG inference.

    Promotion rule:

      * **Cross-modal**: ``genome_share_min <= r_G[j] <= genome_share_max``
        keeps features whose decoder norm is meaningfully spread across
        both streams (a structural cross-modal coherence test that does
        not require labels).
      * **Active enough**: ``activation_rate[j] >= min_activation_rate``
        drops features that fire so rarely they are useless as DAG
        covariates and would push the BGe / Laplace marginal-likelihood
        scores into a degenerate regime.
      * **Top by joint strength**: among the survivors, take the
        ``n_promote`` features whose product
        ``r_G[j] * (1 - r_G[j]) * activation_rate[j]`` is largest. This
        favours features that are simultaneously well-balanced across
        streams *and* often-active.

    Parameters
    ----------
    model : TopKCrosscoder
    panels : AlignedPanels
    n_promote : int
        Target number of features to promote.
    genome_share_min, genome_share_max : float
        Bracketing thresholds on per-feature genome share.
    min_activation_rate : float
        Minimum fraction of participants for whom ``z[:, j] > 0``.
    name_prefix : str
        Used to construct DAG node names ``"{prefix}_{idx:04d}"``.

    Returns
    -------
    FeatureSelection
    """
    if not 0.0 <= genome_share_min < genome_share_max <= 1.0:
        raise ValueError(
            "need 0 <= genome_share_min < genome_share_max <= 1, "
            f"got {genome_share_min} and {genome_share_max}"
        )

    z = encode(model, panels.A, panels.B)
    rates = _feature_activation_rate(z)
    r_G = feature_stream_share(model)

    eligible = (
        (r_G >= genome_share_min)
        & (r_G <= genome_share_max)
        & (rates >= min_activation_rate)
    )
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size == 0:
        raise RuntimeError(
            "no features satisfy the cross-modal + activation criteria; "
            "consider relaxing thresholds or training longer"
        )

    score = r_G * (1.0 - r_G) * rates
    eligible_score = score[eligible_idx]
    order = np.argsort(eligible_score)[::-1]
    take = min(n_promote, eligible_idx.size)
    chosen = eligible_idx[order[:take]]
    chosen_sorted = np.sort(chosen)

    names = tuple(f"{name_prefix}_{int(j):04d}" for j in chosen_sorted)
    return FeatureSelection(
        indices=chosen_sorted.astype(np.int64),
        names=names,
        genome_share=r_G[chosen_sorted].astype(np.float64),
        activation_rate=rates[chosen_sorted].astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Dataset augmentation
# ---------------------------------------------------------------------------


@dataclass
class AugmentationResult:
    """Augmented dataset plus provenance metadata.

    The augmented dataset has the same row order as the input
    ``base_dataset``, restricted to participants for whom the panel
    activations could be computed (i.e. who survived
    :func:`align_panels_by_iid`).
    """

    dataset: SyntheticDataset
    feature_selection: FeatureSelection
    feature_activations: np.ndarray   # (n, k_promote) post-TopK
    kept_person_id: np.ndarray        # (n,) string
    base_n: int                       # rows of base_dataset
    augmented_n: int                  # rows after panel alignment


def augment_dataset_with_features(
    base_dataset: SyntheticDataset,
    base_person_ids: Sequence[str],
    panels: AlignedPanels,
    model: TopKCrosscoder,
    selection: FeatureSelection,
) -> AugmentationResult:
    """Append crosscoder feature activations as new continuous DAG nodes.

    The ``base_dataset`` is restricted (by row) to the participants present
    in ``panels.person_id`` (i.e. those who appear in both the PRS frame
    and the EHR panel). For each surviving participant the post-TopK
    activations for the selected features are appended as new columns of
    ``X``. Node types are extended with ``"continuous"`` per appended
    column. The ground-truth adjacency is padded with zeros (the new
    features have no a-priori known edges to / from the cohort columns).

    Parameters
    ----------
    base_dataset : SyntheticDataset
        Cohort dataset (cohort columns + survival placeholders) before
        augmentation.
    base_person_ids : sequence of str
        Per-row IID for ``base_dataset.X``. Must satisfy
        ``len(base_person_ids) == base_dataset.n``.
    panels : AlignedPanels
        Output of :func:`align_panels_by_iid`. Defines the participants
        whose panel activations are available.
    model : TopKCrosscoder
        Trained crosscoder used to encode panels into latent activations.
    selection : FeatureSelection
        Output of :func:`select_shared_features`.

    Returns
    -------
    AugmentationResult
    """
    if len(base_person_ids) != base_dataset.n:
        raise ValueError(
            f"base_person_ids has length {len(base_person_ids)} but "
            f"base_dataset has {base_dataset.n} rows"
        )

    base_pid = np.asarray([str(p) for p in base_person_ids])
    panel_pid = panels.person_id.astype(str)
    panel_index = {p: i for i, p in enumerate(panel_pid.tolist())}

    keep_base_rows: list[int] = []
    keep_panel_rows: list[int] = []
    for i, p in enumerate(base_pid.tolist()):
        j = panel_index.get(p)
        if j is None:
            continue
        keep_base_rows.append(i)
        keep_panel_rows.append(j)

    if not keep_base_rows:
        raise RuntimeError(
            "no overlap between base_person_ids and panels.person_id"
        )

    keep_base_arr = np.asarray(keep_base_rows, dtype=np.int64)
    keep_panel_arr = np.asarray(keep_panel_rows, dtype=np.int64)

    # Encode the panel rows we are keeping, then take the selected feature
    # columns to use as new design-matrix columns.
    A_keep = panels.A[keep_panel_arr]
    B_keep = panels.B[keep_panel_arr]
    z = encode(model, A_keep, B_keep)
    feat = z[:, selection.indices].astype(np.float64, copy=False)

    X_keep = base_dataset.X[keep_base_arr]
    X_aug = np.concatenate([X_keep, feat], axis=1)

    new_columns = tuple(base_dataset.columns) + tuple(selection.names)
    new_node_types = tuple(base_dataset.node_types) + ("continuous",) * len(
        selection.names
    )

    p_old = base_dataset.p
    p_new = X_aug.shape[1]
    if base_dataset.ground_truth_adj.size:
        gt_padded = np.zeros((p_new, p_new), dtype=int)
        gt_padded[:p_old, :p_old] = base_dataset.ground_truth_adj
    else:
        gt_padded = np.zeros((p_new, p_new), dtype=int)

    aug_dataset = SyntheticDataset(
        X=X_aug,
        time=base_dataset.time[keep_base_arr].copy(),
        event=base_dataset.event[keep_base_arr].copy(),
        columns=new_columns,
        node_types=new_node_types,
        ground_truth_adj=gt_padded,
    )

    return AugmentationResult(
        dataset=aug_dataset,
        feature_selection=selection,
        feature_activations=feat,
        kept_person_id=base_pid[keep_base_arr],
        base_n=base_dataset.n,
        augmented_n=int(X_aug.shape[0]),
    )


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def run_genscore(
    base_dataset: SyntheticDataset,
    base_person_ids: Sequence[str],
    prs_df: pd.DataFrame,
    ehr_panel: EhrPanel,
    *,
    n_promote: int = 32,
    genome_share_min: float = 0.2,
    genome_share_max: float = 0.8,
    min_activation_rate: float = 0.01,
    crosscoder_kwargs: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[AugmentationResult, TopKCrosscoder]:
    """Run the full panel-alignment -> crosscoder -> promotion -> augmentation
    loop and return the augmented dataset together with the trained model.
    """
    panels = align_panels_by_iid(base_person_ids, prs_df, ehr_panel)
    if progress is not None:
        progress(
            f"[crosscoder] panels aligned n={panels.A.shape[0]} "
            f"m_G={panels.A.shape[1]} m_E={panels.B.shape[1]}"
        )
    model = fit_panel_crosscoder(
        panels,
        rng=rng,
        progress=progress,
        **(crosscoder_kwargs or {}),
    )
    selection = select_shared_features(
        model,
        panels,
        n_promote=n_promote,
        genome_share_min=genome_share_min,
        genome_share_max=genome_share_max,
        min_activation_rate=min_activation_rate,
    )
    result = augment_dataset_with_features(
        base_dataset=base_dataset,
        base_person_ids=base_person_ids,
        panels=panels,
        model=model,
        selection=selection,
    )
    return result, model


__all__ = [
    "AlignedPanels",
    "AugmentationResult",
    "FeatureSelection",
    "align_panels_by_iid",
    "augment_dataset_with_features",
    "fit_panel_crosscoder",
    "run_genscore",
    "select_shared_features",
]
