"""Smoke tests for ``causal_pred.genscore.plots``.

We train a tiny crosscoder on a planted synthetic cohort, then verify
that every figure function in the public surface produces a non-empty
file when written to disk via :func:`save_all_genscore_plots`. The
tests exercise the saver's "skip silently" behaviour with partial
inputs as well.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd

from causal_pred.data.cohort import EhrPanel
from causal_pred.genscore import (
    align_panels_by_iid,
    fit_panel_crosscoder,
    select_shared_features,
)
from causal_pred.genscore.plots import (
    GenscorePlotInputs,
    save_all_genscore_plots,
)


def _train_tiny_crosscoder():
    rng = np.random.default_rng(13)
    n, m_g, m_e, n_shared = 400, 6, 12, 3
    Z = np.abs(rng.standard_normal((n, n_shared)))
    Z *= rng.uniform(size=Z.shape) < 0.3
    M_G = rng.standard_normal((n_shared, m_g))
    M_E = rng.standard_normal((n_shared, m_e))
    A = Z @ M_G + 0.05 * rng.standard_normal((n, m_g))
    B = Z @ M_E + 0.05 * rng.standard_normal((n, m_e))
    pids = np.array([f"p{i:05d}" for i in range(n)])
    prs_df = pd.DataFrame(
        A, index=pd.Index(pids, name="IID"),
        columns=[f"PRS_{j}" for j in range(m_g)],
    )
    kinds = ["lab_mean"] * m_e
    ehr = EhrPanel(
        matrix=B,
        person_id=pids,
        feature_names=tuple(f"ehr_{j}" for j in range(m_e)),
        feature_kinds=tuple(kinds),
    )
    panels = align_panels_by_iid(pids, prs_df, ehr)
    model = fit_panel_crosscoder(
        panels,
        d=48, k=4, n_steps=600, batch_size=64, lr=3e-3,
        device="cpu",
        contrastive_coef=0.0,
        rng=np.random.default_rng(0),
    )
    selection = select_shared_features(
        model, panels,
        n_promote=6,
        genome_share_min=0.15,
        genome_share_max=0.85,
        min_activation_rate=0.02,
    )
    return model, panels, selection, prs_df, ehr


def test_save_all_genscore_plots_writes_files():
    model, panels, selection, prs_df, ehr = _train_tiny_crosscoder()
    inputs = GenscorePlotInputs(
        model=model,
        panels=panels,
        selection=selection,
        prs_columns=tuple(prs_df.columns),
        ehr_columns=ehr.feature_names,
        ehr_kinds=ehr.feature_kinds,
        history=model.history,
        band=(0.15, 0.85),
        min_activation_rate=0.02,
    )

    with tempfile.TemporaryDirectory() as td:
        saved = save_all_genscore_plots(td, inputs)

        # All ten plot kinds should fire on a fully-populated input.
        expected = {
            "crosscoder_genome_share",
            "crosscoder_decoder_geometry",
            "crosscoder_training",
            "crosscoder_selection",
            "crosscoder_decoder_heatmap",
            "crosscoder_dossier",
            "crosscoder_coactivation",
            "crosscoder_reconstruction",
            "crosscoder_ridgeline",
            "crosscoder_overview",
        }
        assert expected <= set(saved), (
            f"missing plots: {expected - set(saved)}"
        )

        # Every file we said we wrote must exist and be non-empty.
        for name, (png, pdf) in saved.items():
            assert os.path.isfile(png), f"{name} png missing"
            assert os.path.isfile(pdf), f"{name} pdf missing"
            assert os.path.getsize(png) > 1000, f"{name} png suspiciously small"
            assert os.path.getsize(pdf) > 1000, f"{name} pdf suspiciously small"


def test_save_all_genscore_plots_skips_missing_inputs():
    """With only a history dict, only the training-dynamics plot should fire."""
    history = {
        "step": [1, 100, 200],
        "loss_main": [1.0, 0.5, 0.2],
        "loss_aux": [0.8, 0.4, 0.1],
        "frac_dead": [0.5, 0.2, 0.05],
        "frac_active_batch": [0.05, 0.1, 0.12],
        "ever_active_count": [10, 50, 100],
    }
    inputs = GenscorePlotInputs(history=history)
    with tempfile.TemporaryDirectory() as td:
        saved = save_all_genscore_plots(td, inputs)
        assert set(saved) == {"crosscoder_training"}
