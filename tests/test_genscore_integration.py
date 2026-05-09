"""End-to-end test for the crosscoder integration loop.

This test exercises ``align_panels_by_iid -> fit_panel_crosscoder ->
select_shared_features -> augment_dataset_with_features`` on a tiny
synthetic cohort so the whole machinery is verified in seconds without
needing gnomon, OMOP frames, or a real GAM. The test asserts the
contract that downstream stages depend on:

  1. The augmented :class:`SyntheticDataset` carries the new feature
     columns at the *end* of ``X``, with ``"continuous"`` node-types.
  2. Row count drops to the IID intersection (no empty rows).
  3. ``ground_truth_adj`` is padded with zeros to the new dimensionality.
  4. The kept ``person_id`` order matches the input ``base_person_ids``
     order, restricted to the intersection.

The synthetic generator plants a structured shared-latent code so we
also confirm at least one feature gets selected with a meaningful
(neither-genome-only-nor-EHR-only) decoder split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_pred.data.cohort import EhrPanel
from causal_pred.data.synthetic import SyntheticDataset
from causal_pred.genscore import (
    align_panels_by_iid,
    augment_dataset_with_features,
    fit_panel_crosscoder,
    run_genscore,
    select_shared_features,
)


def _synthetic_panels(n: int, m_G: int, m_E: int, n_shared: int, rng):
    """Build matched (PRS frame, EHR panel) with a planted shared code.

    The shared latents act on both streams, so a well-trained crosscoder
    should pick at least one cross-modal feature.
    """
    Z = np.abs(rng.standard_normal((n, n_shared)))
    Z *= (rng.uniform(size=Z.shape) < 0.3)  # sparsify
    M_G = rng.standard_normal((n_shared, m_G))
    M_E = rng.standard_normal((n_shared, m_E))
    A = Z @ M_G + 0.05 * rng.standard_normal((n, m_G))
    B = Z @ M_E + 0.05 * rng.standard_normal((n, m_E))
    person_ids = np.array([f"p{i:05d}" for i in range(n)])
    prs_df = pd.DataFrame(
        A,
        index=pd.Index(person_ids, name="IID"),
        columns=[f"PRS_{j}" for j in range(m_G)],
    )
    ehr_panel = EhrPanel(
        matrix=B,
        person_id=person_ids,
        feature_names=tuple(f"ehr_{j}" for j in range(m_E)),
        feature_kinds=tuple("lab_mean" for _ in range(m_E)),
    )
    return person_ids, prs_df, ehr_panel


def _synthetic_base_dataset(person_ids, rng) -> SyntheticDataset:
    """Tiny cohort dataset matching the cohort-CSV contract."""
    n = len(person_ids)
    p = 3
    X = rng.standard_normal((n, p))
    columns = ("type2_diabetes", "bmi", "hba1c")
    node_types = ("binary", "continuous", "continuous")
    X[:, 0] = (X[:, 0] > 0).astype(float)  # binary first column
    return SyntheticDataset(
        X=X,
        time=np.zeros(n, dtype=float),
        event=np.zeros(n, dtype=int),
        columns=columns,
        node_types=node_types,
        ground_truth_adj=np.zeros((p, p), dtype=int),
    )


def test_full_integration_loop_augments_and_preserves_contract():
    rng = np.random.default_rng(7)
    n_total = 600
    person_ids, prs_df, ehr_panel = _synthetic_panels(
        n=n_total, m_G=8, m_E=10, n_shared=4, rng=rng
    )
    base_dataset = _synthetic_base_dataset(person_ids, rng)
    base_pids = list(person_ids)

    aug_result, model = run_genscore(
        base_dataset=base_dataset,
        base_person_ids=base_pids,
        prs_df=prs_df,
        ehr_panel=ehr_panel,
        n_promote=8,
        genome_share_min=0.15,
        genome_share_max=0.85,
        min_activation_rate=0.05,
        crosscoder_kwargs=dict(
            d=64,
            k=4,
            n_steps=700,
            batch_size=128,
            lr=3e-3,
            device="cpu",
            contrastive_coef=0.0,
        ),
        rng=np.random.default_rng(0),
    )

    # 1. Augmented dataset shape contract.
    p_old = base_dataset.p
    sel = aug_result.feature_selection
    n_promoted = int(sel.indices.size)
    assert n_promoted >= 1
    aug = aug_result.dataset
    assert aug.X.shape[1] == p_old + n_promoted
    assert aug.columns[:p_old] == base_dataset.columns
    assert aug.node_types[:p_old] == base_dataset.node_types
    assert all(t == "continuous" for t in aug.node_types[p_old:])
    assert aug.X.shape[0] == aug_result.augmented_n
    assert aug_result.augmented_n == n_total  # all overlap in this test

    # 2. ground_truth_adj is padded with zeros.
    assert aug.ground_truth_adj.shape == (p_old + n_promoted, p_old + n_promoted)
    assert aug.ground_truth_adj.sum() == 0

    # 3. Promoted feature names appear at the tail of columns.
    assert tuple(aug.columns[p_old:]) == sel.names

    # 4. Promoted features have meaningful cross-modal split.
    assert np.all(sel.genome_share > 0.15)
    assert np.all(sel.genome_share < 0.85)

    # 5. Activation rates respect the floor.
    assert np.all(sel.activation_rate >= 0.05)

    # 6. Loss decreased relative to init.
    losses = model.history["loss_main"]
    assert losses[-1] < losses[0]


def test_alignment_drops_participants_missing_from_either_panel():
    """The aligned matrices should only contain the intersection of IIDs."""
    rng = np.random.default_rng(0)
    n_total = 100
    person_ids, prs_df, ehr_panel = _synthetic_panels(
        n=n_total, m_G=4, m_E=5, n_shared=3, rng=rng
    )
    # Drop the last 10 from PRS and the first 10 from EHR.
    prs_df = prs_df.iloc[:-10].copy()
    ehr_panel = EhrPanel(
        matrix=ehr_panel.matrix[10:],
        person_id=ehr_panel.person_id[10:],
        feature_names=ehr_panel.feature_names,
        feature_kinds=ehr_panel.feature_kinds,
    )
    panels = align_panels_by_iid(person_ids, prs_df, ehr_panel)
    assert panels.A.shape[0] == panels.B.shape[0]
    assert panels.A.shape[0] == n_total - 20  # both 10-tail drops
    # Order preserved relative to person_ids.
    expected = person_ids[10 : n_total - 10]
    np.testing.assert_array_equal(panels.person_id.astype(str), expected)


def test_augment_handles_partial_overlap_with_base_dataset():
    """Base dataset rows missing from panels should be dropped in the output."""
    rng = np.random.default_rng(1)
    n_total = 120
    person_ids, prs_df, ehr_panel = _synthetic_panels(
        n=n_total, m_G=4, m_E=5, n_shared=3, rng=rng
    )
    base_dataset = _synthetic_base_dataset(person_ids, rng)

    # Strip the last 30 IIDs out of the PRS panel only -> they survive in
    # base_dataset and ehr_panel but not in the aligned panels.
    prs_df = prs_df.iloc[:-30].copy()

    panels = align_panels_by_iid(person_ids, prs_df, ehr_panel)
    model = fit_panel_crosscoder(
        panels,
        d=32,
        k=3,
        n_steps=250,
        batch_size=64,
        lr=3e-3,
        device="cpu",
        contrastive_coef=0.0,
        rng=np.random.default_rng(0),
    )
    sel = select_shared_features(
        model,
        panels,
        n_promote=4,
        genome_share_min=0.1,
        genome_share_max=0.9,
        min_activation_rate=0.01,
    )
    aug = augment_dataset_with_features(
        base_dataset=base_dataset,
        base_person_ids=list(person_ids),
        panels=panels,
        model=model,
        selection=sel,
    )
    assert aug.augmented_n == n_total - 30
    assert aug.dataset.X.shape[0] == n_total - 30
    np.testing.assert_array_equal(
        aug.kept_person_id.astype(str), person_ids[: n_total - 30]
    )
