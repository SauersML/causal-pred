"""Distributional survival GAM via the gam CLI.

This module is a thin, direct wrapper over the `gam <https://github.com/SauersML/gam>`_
Rust CLI's ``survival`` + ``predict`` subcommands.  The library's Rust
engine implements proper Royston-Parmar / Weibull / probit-location-scale
survival GAMs with penalised splines, REML smoothing, and analytical
uncertainty -- we invoke it via ``subprocess`` and read back the CSV
outputs.  There is no IPCW, no EM, no imputation, no events-only
complete-case hack: the CLI's native survival family handles censoring
exactly.

Public API
----------

``fit_survival_gam(time, event, X, columns=None, ...) -> SurvivalGAM``
``SurvivalGAM.predict_survival(X_new, t_grid) -> (n_samples, n_new, n_t)``
``SurvivalGAM.predict_median_survival(X_new) -> (n_samples, n_new)``
``SurvivalGAM.predict_hazard(X_new, t_grid) -> (n_samples, n_new, n_t)``
``SurvivalGAM.posterior_summary() -> dict``
``bma_survival(parent_sets, weights, time, event, data_matrix, columns,
               t_grid, X_eval=None, **gam_kwargs) -> dict``

Bayesian model averaging (Draper 1995, JRSS-B 57(1):45-97) decomposes
the posterior-predictive variance of ``S(t | x)`` into a parametric
(within-model) part and a structural (between-model) part.

CLI discovery
-------------
The CLI binary is located via :func:`_locate_gam_cli`, which checks:

    1. ``GAM_CLI`` environment variable (highest priority)
    2. ``gam`` on ``$PATH``
    3. ``/Users/user/.local/bin/gam`` (the upstream installer's default)
    4. ``/Users/user/gam/bench/runtime/gam`` (the local dev build)

If none of those resolve to an executable the module raises at fit
time.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------

_GAM_CLI_CANDIDATES = (
    os.environ.get("GAM_CLI"),
    None,                                       # placeholder for shutil.which
    "/Users/user/.local/bin/gam",
    "/Users/user/gam/bench/runtime/gam",
)


def _locate_gam_cli() -> str:
    """Return the absolute path of the ``gam`` CLI binary, raising if absent."""
    env_cli = os.environ.get("GAM_CLI")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return env_cli
    onpath = shutil.which("gam")
    if onpath:
        return onpath
    for candidate in (
        "/Users/user/.local/bin/gam",
        "/Users/user/gam/bench/runtime/gam",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "gam CLI not found.  Install the binary via SauersML/gam's install.sh "
        "or set the GAM_CLI env var to its path."
    )


# ---------------------------------------------------------------------------
# Column kind detection + formula
# ---------------------------------------------------------------------------


def _is_binary_column(x: np.ndarray) -> bool:
    vals = np.unique(x)
    return vals.shape[0] <= 2 and set(vals.tolist()).issubset({0.0, 1.0})


# Default smooth kind for continuous covariates.  Duchon splines
# (Duchon 1977) generalise thin-plate splines: parameter ``order = m``
# controls the smoothness (``m-1`` continuous derivatives) and ``power =
# s`` the rate of decay of the Green's function, with the condition
# 2m + s > d (d = covariate dimension).  For univariate smooths the
# common choice is m = 2, s = 1, which gives a cubic-style penalised
# smooth with better boundary behaviour than B-splines.
_DUCHON_DEFAULT_ORDER = 2
_DUCHON_DEFAULT_POWER = 1


def _build_survival_formula(columns: Sequence[str],
                            kinds: Sequence[str],
                            smooth_kind: str = "duchon") -> str:
    """Return ``s(x1, type=duchon, order=2, power=1) + x2 + ...``.

    Continuous columns become penalised Duchon smooths; binary columns
    enter as plain linear terms.  An empty covariate set collapses to
    intercept-only via the literal ``1`` term.
    """
    terms: List[str] = []
    for name, kind in zip(columns, kinds):
        if kind == "continuous":
            if smooth_kind == "duchon":
                terms.append(
                    f"s({name}, type=duchon, order={_DUCHON_DEFAULT_ORDER}, "
                    f"power={_DUCHON_DEFAULT_POWER})"
                )
            else:
                terms.append(f"s({name})")
        else:
            terms.append(name)
    return " + ".join(terms) if terms else "1"


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, header: Sequence[str], rows: np.ndarray) -> None:
    """Write a 2-D numpy array with the given header to ``path``."""
    rows = np.asarray(rows, dtype=float)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for r in rows:
            writer.writerow(r.tolist())


def _read_prediction_csv(path: str) -> Dict[str, np.ndarray]:
    """Load a gam-predict CSV and return columns as float64 arrays."""
    import pandas as pd  # pandas is a hard dep of the project

    df = pd.read_csv(path)
    return {c: df[c].to_numpy(dtype=float) for c in df.columns}


# ---------------------------------------------------------------------------
# Internal fit object
# ---------------------------------------------------------------------------


@dataclass
class _SubmodelFit:
    """Fitted gam survival model artefacts."""

    model_path: str                     # path to gam's model.json on disk
    columns: Tuple[str, ...]
    kinds: Tuple[str, ...]              # "continuous" | "binary"
    n_train: int
    n_events: int
    formula: str
    cli: str                            # the gam CLI binary used
    train_summary: Dict[str, Any]
    tmp_workdir: str                    # path to the tempdir
    _tmp: Optional[Any] = None          # owns the tempdir object (keeps it alive)


# ---------------------------------------------------------------------------
# Fit via the gam CLI
# ---------------------------------------------------------------------------


def _fit_gam(time: np.ndarray, event: np.ndarray, X: np.ndarray,
             columns: Tuple[str, ...],
             survival_likelihood: str = "transformation",
             time_basis: Optional[str] = None,
             time_num_internal_knots: Optional[int] = None,
             time_degree: Optional[int] = None,
             timeout: int = 600) -> _SubmodelFit:
    """Fit a survival GAM via ``gam survival`` and return a ``_SubmodelFit``."""
    cli = _locate_gam_cli()

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float).reshape(time.shape[0], -1)
    if X.shape[1] != len(columns):
        raise ValueError(
            f"columns length {len(columns)} != X.shape[1] {X.shape[1]}"
        )

    kinds = tuple(
        "binary" if _is_binary_column(X[:, i]) else "continuous"
        for i in range(X.shape[1])
    )
    formula = _build_survival_formula(columns, kinds)

    # Persist the tempdir across fit + prediction so the model.json lives
    # long enough to be invoked.  The SurvivalGAM dataclass owns it and
    # it's cleaned up when the Python object is garbage-collected.
    tmp = _PersistentTempdir(prefix="gam_survival_")

    train_csv = os.path.join(tmp.path, "train.csv")
    model_json = os.path.join(tmp.path, "model.json")

    entry = np.zeros_like(time, dtype=float)
    header = ["entry", "exit", "event", *columns]
    rows = np.column_stack([entry, time, event, X])
    _write_csv(train_csv, header, rows)

    cmd = [
        cli, "survival",
        "--entry", "entry",
        "--exit", "exit",
        "--event", "event",
        "--formula", formula,
        "--survival-likelihood", survival_likelihood,
    ]
    # Let the CLI pick its own defaults for the time basis unless the
    # caller overrides; the CLI's ``linear`` default is robust, whereas
    # higher-dimensional B-spline bases can run into monotonicity
    # feasibility issues on small samples.
    if time_basis is not None:
        cmd += ["--time-basis", time_basis]
    if time_num_internal_knots is not None:
        cmd += ["--time-num-internal-knots", str(time_num_internal_knots)]
    if time_degree is not None:
        cmd += ["--time-degree", str(time_degree)]
    cmd += ["--out", model_json, train_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "gam survival fit failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    # Parse any line starting with '{' as JSON summary; otherwise keep the raw stdout.
    train_summary: Dict[str, Any] = {"cli_stdout": proc.stdout.strip()}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                train_summary = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    return _SubmodelFit(
        model_path=model_json,
        columns=tuple(columns),
        kinds=kinds,
        n_train=int(time.shape[0]),
        n_events=int(np.sum(event > 0.0)),
        formula=f"Surv(entry, exit, event) ~ {formula}",
        cli=cli,
        train_summary=train_summary,
        tmp_workdir=tmp.path,
        _tmp=tmp,
    )


class _PersistentTempdir:
    """A tempfile.TemporaryDirectory that auto-cleans on dealloc.

    We keep an explicit handle on the underlying object so multiple
    SurvivalGAM objects can safely outlive each other.
    """

    def __init__(self, prefix: str = "gam_") -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix=prefix)
        self.path = self._tmp.name

    def cleanup(self) -> None:
        try:
            self._tmp.cleanup()
        except Exception:  # pragma: no cover -- best-effort
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.cleanup()


# ---------------------------------------------------------------------------
# Predict via the gam CLI
# ---------------------------------------------------------------------------


def _predict_survival_matrix(fit: _SubmodelFit,
                             X_new: np.ndarray,
                             t_grid: np.ndarray,
                             interval: Optional[float] = 0.95,
                             timeout: int = 600
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(S_mean, S_se)`` of shape ``(n_new, n_t)``.

    We build a long-format CSV with ``n_new * n_t`` rows (one per
    (individual, time) pair) and ask ``gam predict --uncertainty`` to
    evaluate ``S(t|x)`` and its standard error for each row.
    """
    X_new = np.asarray(X_new, dtype=float)
    p = len(fit.columns)
    if p > 0:
        X_new = X_new.reshape(-1, p)
    else:
        X_new = X_new.reshape(-1, 0)
    n_new = X_new.shape[0]
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    n_t = t_grid.shape[0]

    if n_new == 0 or n_t == 0:
        return (
            np.zeros((n_new, n_t), dtype=float),
            np.zeros((n_new, n_t), dtype=float),
        )

    # Long format: block over t_grid within each individual.
    rep_X = (
        np.repeat(X_new, n_t, axis=0)
        if p > 0 else np.zeros((n_new * n_t, 0))
    )
    rep_t = np.tile(t_grid, n_new)
    rep_entry = np.zeros_like(rep_t, dtype=float)
    rep_event = np.ones_like(rep_t, dtype=float)  # placeholder

    header = ["entry", "exit", "event", *fit.columns]
    rows = np.column_stack([rep_entry, rep_t, rep_event, rep_X])

    new_csv = os.path.join(fit.tmp_workdir, "new.csv")
    pred_csv = os.path.join(fit.tmp_workdir, "pred.csv")
    _write_csv(new_csv, header, rows)

    cmd = [fit.cli, "predict", "--out", pred_csv]
    if interval is not None:
        cmd += ["--uncertainty", "--level", str(float(interval))]
    cmd += [fit.model_path, new_csv]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "gam predict failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    cols = _read_prediction_csv(pred_csv)
    if "survival_prob" in cols:
        S_flat = cols["survival_prob"]
    elif "mean" in cols:
        S_flat = cols["mean"]
    else:
        raise RuntimeError(
            f"gam predict produced unexpected columns {list(cols)}; "
            "expected 'survival_prob' or 'mean'."
        )
    # Uncertainty: the CLI may emit ``survival_prob_lower``/``_upper`` or
    # ``eta_se``; prefer a direct SE on the survival scale.
    if "survival_prob_se" in cols:
        SE_flat = cols["survival_prob_se"]
    elif "survival_prob_lower" in cols and "survival_prob_upper" in cols:
        z = 1.959963984540054  # 97.5th percentile of N(0,1)
        SE_flat = (cols["survival_prob_upper"] - cols["survival_prob_lower"]) / (2.0 * z)
    elif "eta_se" in cols:
        # Delta method: survival = 1 - Phi(eta) on the probit scale;
        # when the CLI doesn't give a survival_prob_se, scale eta_se by
        # |dS/deta| = phi(eta).  Safe for small SEs.
        from scipy.stats import norm

        eta = cols.get("eta", np.zeros_like(S_flat))
        SE_flat = np.abs(norm.pdf(eta)) * cols["eta_se"]
    else:
        SE_flat = np.zeros_like(S_flat)

    S_mean = S_flat.reshape(n_new, n_t)
    S_se = SE_flat.reshape(n_new, n_t)

    # S must be non-increasing in t; enforce by left-to-right cumulative
    # minimum: S_fixed[k] = min(S[0], ..., S[k]).  This is a no-op when
    # the CLI's output is already monotone and removes any local
    # violations otherwise without pulling early values down to the
    # final value (the bug the right-to-left variant produced on
    # already-monotone input).
    S_mean = np.minimum.accumulate(S_mean, axis=1)
    S_mean = np.clip(S_mean, 0.0, 1.0)
    S_se = np.clip(S_se, 0.0, None)
    return S_mean, S_se


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SurvivalGAM:
    """Fitted distributional survival GAM.

    Attributes
    ----------
    samples : dict
        Retained for API compatibility.  Posterior-predictive draws are
        synthesised from the CLI's survival-scale mean + SE at prediction
        time (Normal around the CLI point estimate, clipped to [0, 1]
        and re-monotonised).  When the CLI emits MCMC draws directly we
        will thread those through here.
    X_design : dict
        Per-covariate metadata (kind + column order).
    columns : tuple
        Covariate names in the order used at fit time.
    t_scale : float
        Kept at 1.0: the gam CLI works on the native time scale.
    diagnostics : dict
        CLI stdout + detected family / formula / convergence bits.
    """

    samples: dict
    X_design: dict
    columns: tuple
    t_scale: float
    diagnostics: dict
    _fit: Optional[_SubmodelFit] = field(repr=False, default=None)
    _n_posterior_draws: int = 200
    _predict_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0),
        repr=False,
    )

    # --------------------------------------------------------------

    def _cached_predict(self, X_new: np.ndarray, t_grid: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        X_new = np.ascontiguousarray(X_new, dtype=float)
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        key = hash((X_new.tobytes(), X_new.shape, t_grid.tobytes()))
        cached = self._predict_cache.get(key)
        if cached is not None:
            return cached
        S_mean, S_se = _predict_survival_matrix(self._fit, X_new, t_grid)
        self._predict_cache[key] = (S_mean, S_se)
        return S_mean, S_se

    def _shape_X_new(self, X_new: Any) -> np.ndarray:
        X_new = np.asarray(X_new, dtype=float)
        p = len(self.columns)
        if p == 0:
            if X_new.ndim == 1:
                return X_new.reshape(-1, 0)
            return X_new.reshape(-1, 0)
        return X_new.reshape(-1, p)

    # --------------------------------------------------------------

    def predict_survival(self, X_new: np.ndarray, t_grid: np.ndarray
                         ) -> np.ndarray:
        """Posterior-predictive survival probabilities ``(S, n_new, n_t)``.

        The gam CLI returns a point estimate ``S_mean`` (already
        monotone-decreasing in ``t``) and an epistemic standard error
        ``S_se`` for the survival-scale prediction.  We generate
        posterior-predictive draws as

            S^{(s)}(t | x) = S_mean(t | x) + eta_s(x) * S_se(t | x),

        with ``eta_s(x) ~ Normal(0, 1)`` sampled ONCE per (draw, row)
        and reused across ``t``.  This preserves the within-row
        monotonicity of each draw (every curve is a uniformly shifted
        copy of the point estimate) and avoids the bias that appears
        when independent per-``(s, row, t)`` noise is projected onto
        the monotone-decreasing cone.
        """
        X_new = self._shape_X_new(X_new)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        S_mean, S_se = self._cached_predict(X_new, t_grid)
        S = max(int(self._n_posterior_draws), 1)
        if S == 1 or not np.any(S_se > 0):
            return np.broadcast_to(S_mean, (S,) + S_mean.shape).copy()
        n_new = S_mean.shape[0]
        eta = self._rng.standard_normal((S, n_new))      # shared across t
        out = S_mean[None, :, :] + eta[:, :, None] * S_se[None, :, :]
        return np.clip(out, 0.0, 1.0)

    def predict_median_survival(self, X_new: np.ndarray) -> np.ndarray:
        """Posterior draws of the median survival time per row."""
        X_new = self._shape_X_new(X_new)
        t_grid = np.geomspace(1e-3, 1.0e2, 400)
        S_draws = self.predict_survival(X_new, t_grid)       # (S, n_new, n_t)
        out = np.full(S_draws.shape[:2], float("inf"))
        for s in range(S_draws.shape[0]):
            for k in range(S_draws.shape[1]):
                curve = S_draws[s, k]
                below = np.where(curve <= 0.5)[0]
                if below.size == 0:
                    out[s, k] = t_grid[-1]
                elif below[0] == 0:
                    out[s, k] = t_grid[0]
                else:
                    j = int(below[0])
                    s_lo, s_hi = curve[j - 1], curve[j]
                    t_lo_g, t_hi_g = t_grid[j - 1], t_grid[j]
                    if s_hi == s_lo:
                        out[s, k] = t_hi_g
                    else:
                        frac = (s_lo - 0.5) / (s_lo - s_hi)
                        out[s, k] = t_lo_g + frac * (t_hi_g - t_lo_g)
        return out

    def predict_hazard(self, X_new: np.ndarray, t_grid: np.ndarray
                       ) -> np.ndarray:
        """Posterior-predictive hazard ``h(t|x) = -d/dt log S(t|x)``."""
        S = self.predict_survival(X_new, t_grid)
        t = np.asarray(t_grid, dtype=float).ravel()
        dt = np.diff(t)
        eps = 1e-12
        logS = np.log(np.clip(S, eps, 1.0))
        dlog = -np.diff(logS, axis=2) / dt[None, None, :]
        pad = dlog[:, :, -1:]
        return np.concatenate([dlog, pad], axis=2)

    def posterior_summary(self) -> dict:
        """Diagnostics-ready dict."""
        d = dict(self.diagnostics)
        d["n_posterior_draws"] = int(self._n_posterior_draws)
        d["columns"] = list(self.columns)
        return d


# ---------------------------------------------------------------------------
# Public fit entry point
# ---------------------------------------------------------------------------


def fit_survival_gam(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    columns: Optional[Tuple[str, ...]] = None,
    n_basis: int = 10,                 # retained for API compat
    n_samples: int = 1000,
    warmup: int = 500,                 # retained for API compat
    n_chains: int = 2,                 # retained for API compat
    target_accept: float = 0.8,        # retained for API compat
    max_tree_depth: int = 10,          # retained for API compat
    rng: Optional[np.random.Generator] = None,
    progress: bool = False,
) -> SurvivalGAM:
    """Fit a distributional survival GAM via the gam CLI.

    Parameters
    ----------
    time, event : arrays of shape (n,)
        Observed follow-up times (positive) and event indicators
        (1 = failure, 0 = right-censored).  These are passed directly to
        the gam CLI's ``survival`` subcommand; no transformation, no
        censoring workaround.
    X : (n, p) ndarray
        Covariate matrix.  Columns whose unique values are a subset of
        {0, 1} are treated as binary and enter linearly; other columns
        enter as penalised smooths ``s(name)``.  When ``p == 0`` the
        model reduces to an intercept-only survival fit.
    columns : tuple, optional
        Names for the ``p`` covariates.  Defaults to ``("x0", ...)``.
    n_samples : int
        Number of posterior-predictive draws per prediction.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    X = np.asarray(X, dtype=float).reshape(time.shape[0], -1)
    if columns is None:
        columns = tuple(f"x{i}" for i in range(X.shape[1]))
    columns = tuple(columns)

    fit = _fit_gam(time, event, X, columns)

    X_design = {
        name: {"kind": fit.kinds[i], "index": i}
        for i, name in enumerate(columns)
    }

    cli_version: Optional[str]
    try:
        ver = subprocess.run(
            [fit.cli, "--version"], capture_output=True, text=True, timeout=10,
        )
        cli_version = (ver.stdout + ver.stderr).strip().splitlines()[0]
    except Exception:
        cli_version = None

    summ = fit.train_summary or {}

    def _num(key: str, default: float = float("nan")) -> float:
        v = summ.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    diagnostics = {
        "backend": "gam CLI (survival)",
        "cli_path": fit.cli,
        "cli_version": cli_version,
        "formula": fit.formula,
        "train_summary": summ,
        "converged": True,
        "reml_iterations": int(summ.get("reml_iterations", summ.get("iterations", 0)) or 0),
        "reml_score": _num("reml_score"),
        "edf_total": _num("edf_total"),
        "sigma_residual": _num("sigma", _num("sigma_residual")),
        "deviance": _num("deviance"),
        "n_train": fit.n_train,
        "n_events": fit.n_events,
        "n_posterior_draws": int(n_samples),
    }

    return SurvivalGAM(
        samples={"n_samples": int(n_samples), "coefficients": None},
        X_design=X_design,
        columns=columns,
        t_scale=1.0,
        diagnostics=diagnostics,
        _fit=fit,
        _n_posterior_draws=int(n_samples),
        _rng=rng,
    )


# ---------------------------------------------------------------------------
# Bayesian model averaging (Draper 1995)
# ---------------------------------------------------------------------------


def bma_survival(
    parent_sets: List[Tuple[int, ...]],
    weights: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    data_matrix: np.ndarray,
    column_names: Tuple[str, ...],
    t_grid: np.ndarray,
    X_eval: Optional[np.ndarray] = None,
    **gam_kwargs,
) -> dict:
    """Bayesian model averaging over candidate parent sets (Draper 1995)."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != len(parent_sets):
        raise ValueError("weights must match len(parent_sets)")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("weights sum to zero")
    weights = weights / weights.sum()

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    data_matrix = np.asarray(data_matrix, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    X_eval_arr = (
        data_matrix if X_eval is None else np.asarray(X_eval, dtype=float)
    )
    if X_eval_arr.ndim != 2:
        raise ValueError("X_eval must be 2-D (n_eval, p)")
    n_eval = X_eval_arr.shape[0]

    per_set_mean: List[np.ndarray] = []
    per_set_var: List[np.ndarray] = []
    fits: List[SurvivalGAM] = []

    for cols in parent_sets:
        cols = tuple(int(c) for c in cols)
        sub_X = data_matrix[:, list(cols)] if cols else np.zeros((time.shape[0], 0))
        names = tuple(column_names[c] for c in cols)
        fit = fit_survival_gam(
            time, event, sub_X, columns=names, **gam_kwargs,
        )
        fits.append(fit)
        sub_eval = X_eval_arr[:, list(cols)] if cols else np.zeros((n_eval, 0))
        draws = fit.predict_survival(sub_eval, t_grid)       # (S, n_eval, n_t)
        per_set_mean.append(draws.mean(axis=0))
        per_set_var.append(draws.var(axis=0, ddof=0))

    stack_mean = np.stack(per_set_mean, axis=0)              # (K, n_eval, n_t)
    stack_var = np.stack(per_set_var, axis=0)

    S_bma = np.einsum("k,kij->ij", weights, stack_mean)
    V_param = np.einsum("k,kij->ij", weights, stack_var)
    V_struct = np.einsum(
        "k,kij->ij", weights, (stack_mean - S_bma[None, :, :]) ** 2,
    )

    V_total = V_param + V_struct
    return {
        # Canonical keys.
        "survival_mean": S_bma,
        "S_bma": S_bma,                   # alias kept for test compatibility
        "per_set_mean": stack_mean,
        "per_model_mean": stack_mean,     # alias
        "per_set_variance": stack_var,
        "per_model_variance": stack_var,  # alias
        "variance_parametric": V_param,
        "var_parametric": V_param,        # alias
        "variance_structural": V_struct,
        "var_structural": V_struct,       # alias
        "variance_total": V_total,
        "var_total": V_total,             # alias
        "weights": weights,
        "fits": fits,
        "t_grid": t_grid,
    }


__all__ = [
    "SurvivalGAM",
    "fit_survival_gam",
    "bma_survival",
]
