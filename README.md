# Causal-Pred: Causal Disease Modeling with Uncertainty

End-to-end implementation of the project proposed in `paper/main.tex`: predict
disease *and* its causes on biobank-scale individual-level data, with explicit
structural uncertainty over DAG parent sets. The gamfit survival backend
fits right-censored Gompertz-Makeham GAMLSS survival models through gamfit and
reports gamfit delta-method response-scale uncertainty, which is combined with
structural parent-set uncertainty.

Target disease: **Type 2 Diabetes (T2D)**. T2D has well-characterised causal
parents (BMI, HbA1c, lifestyle, genetic risk), strong Mendelian-randomisation
evidence (BMI -> T2D, LDL -> T2D null, etc.), and mature polygenic scores,
which makes the validation framework meaningful even on synthetic data.

## Pipeline

External GWAS summary stats feed MrDAG, which produces an edge-inclusion
prior. AoU cohort CSV plus microarray genotypes feed gnomon-scored PRS nodes
and TopK crosscoder features. DAGSLAM uses the MrDAG prior and data score to
warm-start a structure MCMC, whose posterior parent sets are then passed to
the gamfit survival GAM. The output is per-person survival curves and causal
pathway probabilities.

## Layout

- `src/causal_pred/data/` — synthetic biobank-shape data, real GWAS table, gnomon wrapper
- `src/causal_pred/mrdag/` — MrDAG: produces edge-inclusion-probability matrix pi
- `src/causal_pred/scoring/` — mixed-type marginal-likelihood scores (BGe / Laplace)
- `src/causal_pred/dagslam/` — DAGSLAM hill-climber for the warm-start DAG
- `src/causal_pred/mcmc/` — structure MCMC with MrDAG prior
- `src/causal_pred/gam/` — survival GAM (SauersML/gam backend)
- `src/causal_pred/validation/` — known-edge checks, Nagelkerke R^2, calibration, time-dep AUC
- `src/causal_pred/plots.py` — figure helpers (heatmaps, calibration, survival fans)
- `src/causal_pred/pipeline.py` — end-to-end orchestration
- `scripts/` — runnable drivers (`run_full_pipeline.py`, `generate_figures.py`, `benchmark.py`)
- `tests/` — pytest tests for every component
- `outputs/` — JSON / PNG artefacts produced by the pipeline
- `paper/` — LaTeX paper and build script

## Quickstart

In the terminal, setup and run with one command:

```sh
if [ -d causal-pred/.git ]; then git -C causal-pred stash push -m aou-local-uv-lock -- uv.lock >/dev/null || true; git -C causal-pred pull --ff-only; else git clone https://github.com/SauersML/causal-pred.git causal-pred; fi && bash causal-pred/scripts/bootstrap_aou.sh
```

The bootstrap verifies AoU workspace inputs, installs local tools, syncs locked
dependencies, prepares or restores PRS/EHR intermediates, and runs the single
causal pipeline.

## Install

The project uses `uv` for environment management. Clone the repo, then:

```sh
uv sync --dev
```

This resolves all runtime and dev dependencies, including the Rust-backed
`gam` library.

## Run

```sh
uv run python scripts/run_full_pipeline.py
```

There are no command-line flags; edit the constants in
`src/causal_pred/pipeline.py` when the production configuration changes.

## MR prior source (OpenGWAS)

The MrDAG prior is built from two-sample IVW estimates over a curated trait
set, produced by `src/causal_pred/data/opengwas.py::load_live_gwas`. Set
`OPENGWAS_JWT` (token from <https://opengwas.io/profile/>) to refresh from
the OpenGWAS REST API. The on-disk cache under `data/mr_cache/` is keyed by
`(exposure, outcome, p, r2, kb)` so repeat runs do not re-hit the API.

## Tests

```sh
uv run pytest -q
```

Parallel:

```sh
uv run pytest -q -n auto
```

## Benchmarks

```sh
uv run python scripts/benchmark.py
```

Results are written to `outputs/benchmark.json`.

## Figures

```sh
uv run python scripts/generate_figures.py
```

Renders standalone PNG/PDF figures from `outputs/summary.json` and the saved
NumPy artefacts into `outputs/plots/`.

## Paper rebuild

```sh
make -C paper paper
```

`build_paper.py` stamps numeric values from `outputs/summary.json` into
`main.tex`. The Makefile no-ops gracefully if `latexmk` and `pdflatex` are
both missing.
