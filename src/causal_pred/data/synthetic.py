"""Synthetic All-of-Us-shaped individual-level data generator.

The simulator samples from a DAG with exactly the edges declared in
``nodes.ALL_GROUND_TRUTH_EDGES``.  The conditional forms are chosen so that

  * continuous nodes are (possibly non-linear) Gaussian given parents,
  * binary nodes are logistic given parents,
  * the T2D outcome is modelled as time-to-event with right-censoring
    following a Weibull AFT model whose location is a sum of smooth
    functions of its parents (this is exactly what the survival-GAM
    downstream tries to recover).

Calling ``simulate(n, rng=...)`` returns a :class:`SyntheticDataset`
with three attributes:

    X          : (n, p) numpy array in NODE_NAMES order
    time       : (n,)  observed follow-up time (years since age 40)
    event      : (n,)  1 = T2D observed, 0 = censored
    censored   : boolean alias for event==0

For nodes declared ``survival`` the corresponding column of ``X``
holds the event indicator and a sibling column ``{name}_event`` is NOT
added; the censoring-aware survival outcome is exposed via the separate
``time`` / ``event`` attributes and used by structure learning and the
GAM stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .nodes import (
    NODE_NAMES,
    NODE_INDEX,
    NODE_TYPES,
    N_NODES,
    ALL_GROUND_TRUTH_EDGES,
)


@dataclass
class SyntheticDataset:
    X: np.ndarray  # (n, p) design matrix in NODE_NAMES order
    time: np.ndarray  # (n,) time-to-T2D (years since baseline)
    event: np.ndarray  # (n,) 1 = event, 0 = censored
    columns: tuple  # NODE_NAMES
    node_types: tuple  # NODE_TYPES
    ground_truth_adj: np.ndarray  # (p, p) 0/1 true adjacency

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def to_dict(self) -> dict:
        return {name: self.X[:, i] for i, name in enumerate(self.columns)}


# ---- non-linear smooth helpers ---------------------------------------------


def _smooth_bmi_effect(bmi: np.ndarray) -> np.ndarray:
    """Piecewise-smooth BMI effect: flat below 25, rising steeply above."""
    z = (bmi - 25.0) / 5.0
    return 0.7 * np.log1p(np.exp(z)) - 0.3  # soft-plus around 25


def _smooth_age_effect(age: np.ndarray) -> np.ndarray:
    return 0.04 * (age - 40.0) + 0.0004 * np.maximum(age - 55.0, 0.0) ** 2


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---- main simulator --------------------------------------------------------


def simulate(
    n: int = 5000,
    censoring_rate: float = 0.55,
    followup_years: float = 25.0,
    rng: Optional[np.random.Generator] = None,
) -> SyntheticDataset:
    """Sample ``n`` individuals from the ground-truth DAG."""
    if rng is None:
        rng = np.random.default_rng(0)

    # ---- roots ----
    pgs_t2d = rng.standard_normal(n)
    pgs_bmi = rng.standard_normal(n)
    pgs_ldl = rng.standard_normal(n)
    pgs_hba1c = rng.standard_normal(n)
    age = rng.uniform(40.0, 75.0, size=n)
    sex = rng.binomial(1, 0.5, size=n)  # 1 = male
    ancestry_pc1 = rng.standard_normal(n)

    # family history: partly correlated with T2D PGS to mimic shared liab.
    fh_logit = -0.8 + 0.6 * pgs_t2d + 0.2 * pgs_bmi
    family_history_t2d = rng.binomial(1, _logistic(fh_logit))

    # lifestyle
    years_smoking = np.maximum(
        0.0,
        rng.normal(loc=3.0 + 0.5 * sex, scale=6.0, size=n),
    )
    physical_activity = rng.normal(loc=15.0 - 0.05 * (age - 50.0), scale=8.0, size=n)
    diet_quality = rng.normal(loc=0.0 + 0.05 * (age - 50.0) / 10.0, scale=1.0, size=n)

    # bmi <- pgs_bmi + sex + physical_activity + diet_quality
    bmi = (
        27.0
        + 1.8 * pgs_bmi
        + 1.2 * sex
        - 0.15 * (physical_activity - 15.0)
        - 0.6 * diet_quality
        + rng.normal(0.0, 3.5, size=n)
    )
    bmi = np.clip(bmi, 15.0, 55.0)

    # ldl_cholesterol <- pgs_ldl + ancestry_pc1 + diet_quality
    ldl_cholesterol = (
        3.0
        + 0.45 * pgs_ldl
        + 0.15 * ancestry_pc1
        - 0.20 * diet_quality
        + rng.normal(0.0, 0.6, size=n)
    )
    ldl_cholesterol = np.clip(ldl_cholesterol, 1.0, 9.0)

    # hba1c <- pgs_hba1c + ancestry_pc1 + smooth(bmi)
    hba1c = (
        5.2
        + 0.35 * pgs_hba1c
        + 0.05 * ancestry_pc1
        + 0.25 * _smooth_bmi_effect(bmi)
        + rng.normal(0.0, 0.35, size=n)
    )
    hba1c = np.clip(hba1c, 4.0, 14.0)

    # hypertension <- age + bmi
    htn_logit = (
        -3.5 + 0.05 * (age - 50.0) + 0.18 * (bmi - 25.0) + rng.normal(0.0, 0.3, size=n)
    )
    hypertension = rng.binomial(1, _logistic(htn_logit))

    # systolic_bp <- age + sex + hypertension
    systolic_bp = (
        115.0
        + 0.5 * (age - 50.0)
        + 3.0 * sex
        + 15.0 * hypertension
        + rng.normal(0.0, 10.0, size=n)
    )

    # CVD <- age + sex + years_smoking + systolic_bp + hypertension
    cvd_logit = (
        -4.5
        + 0.06 * (age - 50.0)
        + 0.4 * sex
        + 0.04 * years_smoking
        + 0.015 * (systolic_bp - 120.0)
        + 0.6 * hypertension
    )
    cardiovascular_disease = rng.binomial(1, _logistic(cvd_logit))

    # ---- T2D survival model ----
    # Linear predictor for log-hazard scale (larger = earlier event).
    eta = (
        -2.4
        + 0.55 * pgs_t2d
        + 0.45 * family_history_t2d
        + 0.9 * _smooth_bmi_effect(bmi)
        + 0.60 * (hba1c - 5.5)
        + _smooth_age_effect(age) * 0.6
    )
    # Weibull AFT: T = exp(-eta / shape) * Exp(1)^(1/shape)
    shape = 1.6
    u = rng.uniform(1e-9, 1.0, size=n)
    latent_time = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    latent_time *= 10.0  # rescale to ~years

    # random censoring
    if censoring_rate <= 0.0:
        # No censoring: observe the latent event time for everyone.
        time = latent_time.copy()
        event = np.ones(n, dtype=int)
    else:
        c_rate = -np.log(1.0 - censoring_rate) / followup_years
        censor_time = rng.exponential(1.0 / c_rate, size=n)
        admin = np.full(n, followup_years)
        censor_time = np.minimum(censor_time, admin)

        time = np.minimum(latent_time, censor_time)
        event = (latent_time <= censor_time).astype(int)

    # ---- assemble matrix in NODE_NAMES order ----
    X = np.empty((n, N_NODES), dtype=float)
    values = {
        "pgs_t2d": pgs_t2d,
        "pgs_bmi": pgs_bmi,
        "pgs_ldl": pgs_ldl,
        "pgs_hba1c": pgs_hba1c,
        "age": age,
        "sex": sex.astype(float),
        "ancestry_pc1": ancestry_pc1,
        "family_history_t2d": family_history_t2d.astype(float),
        "years_smoking": years_smoking,
        "physical_activity": physical_activity,
        "diet_quality": diet_quality,
        "bmi": bmi,
        "ldl_cholesterol": ldl_cholesterol,
        "hba1c": hba1c,
        "systolic_bp": systolic_bp,
        "hypertension": hypertension.astype(float),
        "cardiovascular_disease": cardiovascular_disease.astype(float),
        "type2_diabetes": event.astype(float),
    }
    for i, name in enumerate(NODE_NAMES):
        X[:, i] = values[name]

    # ground-truth adjacency
    adj = np.zeros((N_NODES, N_NODES), dtype=int)
    for p_name, c_name in ALL_GROUND_TRUTH_EDGES:
        adj[NODE_INDEX[p_name], NODE_INDEX[c_name]] = 1

    return SyntheticDataset(
        X=X,
        time=time,
        event=event,
        columns=NODE_NAMES,
        node_types=NODE_TYPES,
        ground_truth_adj=adj,
    )


__all__ = ["SyntheticDataset", "simulate"]
