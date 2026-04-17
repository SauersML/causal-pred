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
holds ``time`` and a sibling column ``{name}_event`` is NOT added; the
survival outcome is exposed via the separate ``time`` / ``event``
attributes.  Upstream structure learners treat the T2D node as binary
(event indicator) during DAG search and use (``time``, ``event``) only
inside the GAM stage.
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
    PGS_T2D = rng.standard_normal(n)
    PGS_BMI = rng.standard_normal(n)
    PGS_LDL = rng.standard_normal(n)
    PGS_HbA1c = rng.standard_normal(n)
    age = rng.uniform(40.0, 75.0, size=n)
    sex = rng.binomial(1, 0.5, size=n)  # 1 = male
    ancestry_PC1 = rng.standard_normal(n)

    # family history: partly correlated with T2D PGS to mimic shared liab.
    fh_logit = -0.8 + 0.6 * PGS_T2D + 0.2 * PGS_BMI
    family_history_T2D = rng.binomial(1, _logistic(fh_logit))

    # lifestyle
    years_smoking = np.maximum(
        0.0,
        rng.normal(loc=3.0 + 0.5 * sex, scale=6.0, size=n),
    )
    physical_activity = rng.normal(loc=15.0 - 0.05 * (age - 50.0), scale=8.0, size=n)
    diet_quality = rng.normal(loc=0.0 + 0.05 * (age - 50.0) / 10.0, scale=1.0, size=n)

    # BMI <- PGS_BMI + sex + physical_activity + diet_quality
    BMI = (
        27.0
        + 1.8 * PGS_BMI
        + 1.2 * sex
        - 0.15 * (physical_activity - 15.0)
        - 0.6 * diet_quality
        + rng.normal(0.0, 3.5, size=n)
    )
    BMI = np.clip(BMI, 15.0, 55.0)

    # LDL <- PGS_LDL + ancestry_PC1 + diet_quality
    LDL = (
        3.0
        + 0.45 * PGS_LDL
        + 0.15 * ancestry_PC1
        - 0.20 * diet_quality
        + rng.normal(0.0, 0.6, size=n)
    )
    LDL = np.clip(LDL, 1.0, 9.0)

    # HbA1c <- PGS_HbA1c + ancestry_PC1 + smooth(BMI)
    HbA1c = (
        5.2
        + 0.35 * PGS_HbA1c
        + 0.05 * ancestry_PC1
        + 0.25 * _smooth_bmi_effect(BMI)
        + rng.normal(0.0, 0.35, size=n)
    )
    HbA1c = np.clip(HbA1c, 4.0, 14.0)

    # hypertension <- age + BMI
    htn_logit = (
        -3.5 + 0.05 * (age - 50.0) + 0.18 * (BMI - 25.0) + rng.normal(0.0, 0.3, size=n)
    )
    hypertension = rng.binomial(1, _logistic(htn_logit))

    # systolic BP <- age + sex + hypertension
    systolic_BP = (
        115.0
        + 0.5 * (age - 50.0)
        + 3.0 * sex
        + 15.0 * hypertension
        + rng.normal(0.0, 10.0, size=n)
    )

    # CVD <- age + sex + years_smoking + systolic_BP + hypertension
    cvd_logit = (
        -4.5
        + 0.06 * (age - 50.0)
        + 0.4 * sex
        + 0.04 * years_smoking
        + 0.015 * (systolic_BP - 120.0)
        + 0.6 * hypertension
    )
    cardiovascular_disease = rng.binomial(1, _logistic(cvd_logit))

    # ---- T2D survival model ----
    # Linear predictor for log-hazard scale (larger = earlier event).
    eta = (
        -2.4
        + 0.55 * PGS_T2D
        + 0.45 * family_history_T2D
        + 0.9 * _smooth_bmi_effect(BMI)
        + 0.60 * (HbA1c - 5.5)
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
        "PGS_T2D": PGS_T2D,
        "PGS_BMI": PGS_BMI,
        "PGS_LDL": PGS_LDL,
        "PGS_HbA1c": PGS_HbA1c,
        "age": age,
        "sex": sex.astype(float),
        "ancestry_PC1": ancestry_PC1,
        "family_history_T2D": family_history_T2D.astype(float),
        "years_smoking": years_smoking,
        "physical_activity": physical_activity,
        "diet_quality": diet_quality,
        "BMI": BMI,
        "LDL": LDL,
        "HbA1c": HbA1c,
        "systolic_BP": systolic_BP,
        "hypertension": hypertension.astype(float),
        "cardiovascular_disease": cardiovascular_disease.astype(float),
        "T2D": event.astype(float),  # DAG search treats T2D as binary
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
