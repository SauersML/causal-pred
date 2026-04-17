"""Known-edge recovery check for the MCMC stage's posterior edge-inclusion
probability matrix.

Given a ground-truth set of directed edges (from the causal-disease
literature for T2D), we measure

  * the recovery rate (fraction of ground-truth edges with posterior
    probability >= tau) at several thresholds tau,
  * AUROC and AUPRC, treating the probability matrix as a soft binary
    classifier for the ground-truth adjacency,
  * Matthews correlation coefficient at each threshold.

Significance is assessed against a *degree-preserving* permutation null:
the off-diagonal, non-NaN entries of ``edge_probs`` are shuffled across
positions, which preserves the overall density (marginal mass) of the
probability matrix.  This is the right null for this test -- a dense
probability matrix will trivially ``recover`` any ground-truth edge set.

NaN entries in ``edge_probs`` mean ``no Mendelian-randomisation evidence
available`` and are *masked out*: they are neither scored as predictions
nor used to build the null.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Small NaN-safe AUROC / AUPRC implementations.
# ---------------------------------------------------------------------------


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U identity.

    ``AUROC = P(score_i > score_j) + 0.5 P(score_i == score_j)``
    for a random positive ``i`` and random negative ``j``.  Computed
    from mid-ranks of ``scores`` to handle ties exactly.
    """
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mid-ranks, 1-based.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores, dtype=float)
    # Assign the average of positions to tied values.
    sorted_scores = scores[order]
    # Use a simple tie-handler.
    i = 0
    rpos = np.empty_like(scores, dtype=float)
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        rpos[i : j + 1] = avg_rank
        i = j + 1
    ranks[order] = rpos
    sum_pos = float(ranks[y].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auprc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve.

    Uses the ``step interpolation`` (average-precision) estimator:
    AP = sum_k (R_k - R_{k-1}) P_k, the standard AUPRC definition used
    by scikit-learn.
    """
    y = y.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    i = 0
    while i < scores.size:
        # Consume all equal-score tied samples together (canonical choice:
        # precision and recall are evaluated at the end of each tied block).
        j = i
        while j + 1 < scores.size and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        block = y_sorted[i : j + 1]
        tp += int(block.sum())
        fp += int(block.size - block.sum())
        recall = tp / n_pos
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return float(ap)


def _mcc(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Matthews correlation coefficient for binary predictions."""
    y = y.astype(bool)
    yp = y_pred.astype(bool)
    tp = int(np.sum(yp & y))
    tn = int(np.sum(~yp & ~y))
    fp = int(np.sum(yp & ~y))
    fn = int(np.sum(~yp & y))
    denom = np.sqrt(float((tp + fp)) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denom


# ---------------------------------------------------------------------------
# Main API.
# ---------------------------------------------------------------------------


def known_edge_recovery(
    edge_probs,
    ground_truth_edges: Iterable[Tuple[str, str]],
    node_names: Sequence[str],
    n_permute: int = 1000,
    rng: Optional[np.random.Generator] = None,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
) -> dict:
    """Known-edge recovery of a posterior edge-inclusion probability matrix.

    Parameters
    ----------
    edge_probs : (p, p) matrix.  ``edge_probs[i, j]`` is the posterior
        probability that a directed edge ``i -> j`` is present.  The
        diagonal is ignored.  ``NaN`` entries mean ``no MR evidence``
        and are masked out of both the observed statistic and the null.
    ground_truth_edges : iterable of ``(parent_name, child_name)`` pairs.
    node_names : ordered list/tuple of node names; positions in
        ``edge_probs`` correspond to their index here.
    n_permute : permutation replicates for the null.
    rng : optional ``np.random.Generator``.
    thresholds : probabilities at which recovery rates are evaluated.

    Returns
    -------
    dict -- see module docstring for the full schema.  Always contains a
    ``null_model`` key describing the permutation null.
    """
    if rng is None:
        rng = np.random.default_rng()
    P = np.asarray(edge_probs, dtype=float)
    p = P.shape[0]
    if P.shape != (p, p):
        raise ValueError("edge_probs must be a square (p, p) matrix")
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Build ground-truth adjacency in the same index space.
    A = np.zeros((p, p), dtype=bool)
    gt_pairs = []
    for parent, child in ground_truth_edges:
        if parent not in name_to_idx or child not in name_to_idx:
            raise KeyError(f"edge ({parent}, {child}) references unknown node")
        i, j = name_to_idx[parent], name_to_idx[child]
        A[i, j] = True
        gt_pairs.append((parent, child, i, j))

    # Off-diagonal mask.  NaN entries are masked out (not scored, not
    # permuted) -- the null preserves the set of *available* probability
    # values only.
    off = ~np.eye(p, dtype=bool)
    nan_mask = np.isnan(P)
    usable = off & ~nan_mask

    # Scored classification targets (only over usable cells).
    scores_all = P[usable]
    labels_all = A[usable]
    n_usable = int(usable.sum())

    # AUROC / AUPRC over the usable cells.
    auroc = _auroc(scores_all, labels_all)
    auprc = _auprc(scores_all, labels_all)

    thresholds = tuple(float(t) for t in thresholds)

    # Observed recovery rate at each threshold: fraction of ground-truth
    # edges with probability >= tau.  Ground-truth edges whose entry is
    # NaN (no MR info) are excluded from both numerator and denominator.
    gt_probs = np.array([P[i, j] for _, _, i, j in gt_pairs], dtype=float)
    valid_gt = ~np.isnan(gt_probs)
    n_valid_gt = int(valid_gt.sum())

    observed_rate = {}
    observed_mcc = {}
    for tau in thresholds:
        if n_valid_gt == 0:
            observed_rate[tau] = float("nan")
        else:
            observed_rate[tau] = float(np.mean(gt_probs[valid_gt] >= tau))
        # MCC at this threshold uses all usable cells.
        y_pred = scores_all >= tau
        observed_mcc[tau] = _mcc(y_pred, labels_all)

    # -- Permutation null: shuffle the usable cell values across usable
    # positions.  This preserves the *marginal mass* of edge_probs and
    # therefore controls for the overall density of the matrix.
    positions = np.flatnonzero(usable.ravel())  # flat indices into (p*p,)
    values = scores_all.copy()

    # Flat index of each ground-truth edge into the usable value array.
    # We track which usable cells are ground-truth.
    gt_flat = np.array([i * p + j for _, _, i, j in gt_pairs], dtype=np.int64)
    # For permutation, we need to find, for each ground-truth edge, the
    # permuted value in its cell.  We permute the *values* array and then
    # assemble a permuted probability matrix.  To keep the loop tight we
    # do the math purely with numpy.

    # Pre-compute a position-to-ordinal map so we can look up the ordinal
    # within ``values`` for each ground-truth edge's cell.
    pos_to_ord = -np.ones(p * p, dtype=np.int64)
    pos_to_ord[positions] = np.arange(positions.size)
    gt_ord = pos_to_ord[gt_flat]  # -1 where edge's cell is NaN / diagonal.

    # For MCC / recovery rate / AUROC / AUPRC under the null we simulate
    # by permuting ``values``.
    recov_null = np.empty((n_permute, len(thresholds)), dtype=float)
    mcc_null = np.empty((n_permute, len(thresholds)), dtype=float)
    auroc_null = np.empty(n_permute, dtype=float)
    auprc_null = np.empty(n_permute, dtype=float)
    # Per-edge: track fraction of permutations that yield >= observed value
    # (two-sided via the |obs - mean| formulation below).
    per_edge_null_vals = np.empty((n_permute, len(gt_pairs)), dtype=float)

    for b in range(n_permute):
        perm = rng.permutation(values)
        # Recovery rate under null: look at entries that end up at
        # ground-truth positions.
        gt_perm_vals = np.where(gt_ord >= 0, perm[np.maximum(gt_ord, 0)], np.nan)
        per_edge_null_vals[b] = gt_perm_vals
        valid_perm = ~np.isnan(gt_perm_vals)
        for k, tau in enumerate(thresholds):
            if valid_perm.sum() == 0:
                recov_null[b, k] = float("nan")
            else:
                recov_null[b, k] = float(np.mean(gt_perm_vals[valid_perm] >= tau))
            y_pred = perm >= tau
            mcc_null[b, k] = _mcc(y_pred, labels_all)
        # AUROC/AUPRC under null: should average to 0.5 / base rate.
        auroc_null[b] = _auroc(perm, labels_all)
        auprc_null[b] = _auprc(perm, labels_all)

    # Two-sided permutation p-values.  Two-sided via the centred |.| stat
    # (p = mean[|T* - E[T*]| >= |T_obs - E[T*]|]).
    def _two_sided_p(obs: float, draws: np.ndarray) -> float:
        draws = draws[~np.isnan(draws)]
        if draws.size == 0:
            return float("nan")
        mu = float(np.mean(draws))
        # +1 smoothing so an observation never has p = 0 exactly.
        num = 1 + int(np.sum(np.abs(draws - mu) >= np.abs(obs - mu) - 1e-12))
        den = draws.size + 1
        return num / den

    recov_pvals = {}
    mcc_pvals = {}
    for k, tau in enumerate(thresholds):
        recov_pvals[tau] = _two_sided_p(observed_rate[tau], recov_null[:, k])
        mcc_pvals[tau] = _two_sided_p(observed_mcc[tau], mcc_null[:, k])

    # Per-edge p-values: how extreme is the observed probability at this
    # cell vs.  what would be expected under a random shuffle?
    per_edge = {}
    for idx_e, (parent, child, i, j) in enumerate(gt_pairs):
        obs = float(P[i, j])
        draws = per_edge_null_vals[:, idx_e]
        per_edge[(parent, child)] = {
            "probability": obs,
            "p_value": _two_sided_p(obs, draws) if not np.isnan(obs) else float("nan"),
            "masked": bool(np.isnan(obs)),
        }

    null_model = {
        "type": "degree_preserving_label_permutation",
        "description": (
            "Permute the non-NaN off-diagonal values of edge_probs across "
            "non-NaN off-diagonal positions; preserves the marginal mass "
            "(total probability and value multiset) of the matrix."
        ),
        "n_permute": int(n_permute),
        "n_usable_cells": n_usable,
    }

    return {
        "thresholds": thresholds,
        "observed_recovery": observed_rate,
        "recovery_pvalue": recov_pvals,
        "mcc": observed_mcc,
        "mcc_pvalue": mcc_pvals,
        "auroc": float(auroc),
        "auprc": float(auprc),
        "auroc_null_mean": float(np.nanmean(auroc_null)),
        "auprc_null_mean": float(np.nanmean(auprc_null)),
        "per_edge": per_edge,
        "null_model": null_model,
        "n_ground_truth_edges": len(gt_pairs),
        "n_valid_ground_truth_edges": n_valid_gt,
    }


__all__ = ["known_edge_recovery"]
