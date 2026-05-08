"""Sparse dictionary learning over participant-level streams.

This subpackage hosts mechanistic-interpretability-style decomposition of
the high-dimensional participant signals that flow into the rest of the
causal-pred pipeline. The first (and currently only) tool is a TopK
crosscoder over paired (genome, EHR) activation streams; see
:mod:`causal_pred.genscore.crosscoder`.

The intent is that features discovered here become first-class nodes in
:mod:`causal_pred.data.nodes`, with their causal structure recovered by
the existing MrDAG / DAGSLAM / structure-MCMC stack and their effect on
survival quantified by :mod:`causal_pred.gam.survival`. No auto-interp,
no labels: a feature is defined by its position in the inferred causal
graph and its measured hazard, not by a name.
"""

from .crosscoder import (
    TopKCrosscoder,
    classify_features,
    encode,
    feature_stream_share,
    reconstruct,
    train_crosscoder,
)
from .integrate import (
    AlignedPanels,
    AugmentationResult,
    FeatureSelection,
    align_panels_by_iid,
    augment_dataset_with_features,
    fit_panel_crosscoder,
    run_genscore,
    select_shared_features,
)

__all__ = [
    "AlignedPanels",
    "AugmentationResult",
    "FeatureSelection",
    "TopKCrosscoder",
    "align_panels_by_iid",
    "augment_dataset_with_features",
    "classify_features",
    "encode",
    "feature_stream_share",
    "fit_panel_crosscoder",
    "reconstruct",
    "run_genscore",
    "select_shared_features",
    "train_crosscoder",
]
