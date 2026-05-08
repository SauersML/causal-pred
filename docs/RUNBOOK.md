# Runbook

Operational instructions for installing, running, and troubleshooting
`causal-pred`. Python commands go through `uv run` so the environment
stays consistent with `uv.lock`.

## Install

The project uses `uv` for environment management. Clone the repo, then:

```sh
uv sync --dev
```

This resolves all runtime and dev dependencies, including the Rust-backed
`gam` library. If `gam` fails to build, see the troubleshooting section
below for `rustc missing`.

## AoU Workbench bootstrap

The real-data path expects the complete-case T2D node file at
`$WORKSPACE_BUCKET/data/t2d_initial_nodes_complete.csv`. On the AoU
Researcher Workbench, the bootstrap verifies the workspace inputs, installs
the local toolchain, syncs dependencies, installs `gnomon`, and runs the
single pipeline. If the cohort CSV has no follow-up time/event columns, the
pipeline builds the required cached survival outcome from OMOP follow-up and
T2D diagnosis frames instead of skipping GAM:

```sh
bash causal-pred/scripts/bootstrap_aou.sh
```

This writes `outputs/summary.json` and the edge artefacts under `outputs/`,
then mirrors those small outputs to `$WORKSPACE_BUCKET/results/causal-pred/latest`.

## Run the full pipeline

The pipeline is a single fixed path: cohort wide CSV (resolved via the
local `data/` cache, then `$WORKSPACE_BUCKET/data/`) -> load/build cached
microarray PRS with `gnomon` -> build a baseline-censored EHR panel -> promote
shared genome/EHR crosscoder features -> MrDAG priors -> DAGSLAM hill-climb ->
structure MCMC -> gamfit survival GAM over posterior parent sets -> per-person
survival curves and causal pathway probabilities under `outputs/`. There are
no command-line flags and `run_pipeline()` takes no configuration arguments;
edit the constants in `src/causal_pred/pipeline.py` when the production
configuration changes.

Reusable intermediates live under `data/intermediates/causal-pred/` and are
mirrored to `$WORKSPACE_BUCKET/intermediates/causal-pred/` when the real AoU
workspace bucket is available. This includes the cohort-aligned PRS panel,
the raw `gnomon` `.sscore` output under `gnomon_score/`, OMOP condition/drug
long-frame parquets plus the BigQuery-aggregated curated LOINC measurement
summary under `omop/`, the EHR panel, crosscoder feature matrices, DAGSLAM,
MCMC, and survival GAM outputs. The PLINK genotype files are intentionally not
uploaded or mirrored. Cache keys include the inputs and pipeline constants, so
precomputed results are loaded only when they match the current run.

```sh
uv run python scripts/run_full_pipeline.py
```

## MR prior source (OpenGWAS)

The MrDAG prior is built from two-sample IVW estimates over a curated
trait set (BMI, LDL, HbA1c, SBP, lifetime smoking, physical activity,
hypertension, T2D, CAD). Source selection is controlled by:

- `OPENGWAS_JWT` -- if set, the pipeline runs **live** two-sample IVW
  via the OpenGWAS REST API (study IDs in
  `src/causal_pred/data/opengwas.py::OPENGWAS_STUDY_IDS`). Get a token
  at <https://opengwas.io/profile/>.
- `MR_GWAS_SOURCE=literature` -- forces the offline fallback table in
  `src/causal_pred/data/real_gwas.py`.
- Default: live when a JWT is present, literature otherwise.

Live IVW results are cached per `(exposure, outcome, p, r2, kb, pop)`
combination under `data/mr_cache/`, so repeat runs do not re-hit the
API. The MrDAG cache key includes the source label so live and
literature runs do not collide.

## Paper rebuild

The paper is a LaTeX document under `paper/`. `build_paper.py` stamps
numeric values from `outputs/summary.json` into `main.tex`, and the
Makefile invokes a TeX toolchain if one is on PATH:

```sh
make -C paper paper
```

You can also do a quick rebuild (no pipeline rerun) with:

```sh
make -C paper quick
```

Copy the plot PNGs into `paper/figures/` with:

```sh
make -C paper figures
```

The Makefile no-ops gracefully if `latexmk` and `pdflatex` are both
missing.

## Benchmarks

```sh
uv run python scripts/benchmark.py
uv run python scripts/benchmark.py --sizes 500,1000,2000
```

Results are written to `outputs/benchmark.json`.

## Report generation

```sh
uv run python scripts/generate_report.py
```

This renders a short HTML/Markdown report from `outputs/summary.json`.

## Polygenic scoring via `gnomon`

`data/polygenic.py` shells out to `/Users/user/.local/bin/gnomon` for
scoring, ancestry projection, and sample-term inference. A typical
invocation from Python covers these cases; you can also call `gnomon`
directly for a quick sanity check:

```sh
gnomon --help
gnomon score --manifest outputs/manifest.tsv --out outputs/prs.tsv
gnomon project --pcs reference.pcs --in samples.vcf --out outputs/pcs.tsv
```

If you are reimplementing scoring in Python because `gnomon` is awkward,
stop. Fix the `gnomon` call site instead.

## Tests

Run the full suite:

```sh
uv run pytest -q
```

Module slices:

```sh
uv run pytest tests/test_mcmc.py -q
uv run pytest tests/test_gam.py -q
uv run pytest tests/test_pipeline.py -q
uv run pytest tests/test_docs.py -q
uv run pytest tests/test_paper.py -q
```

Parallel:

```sh
uv run pytest -q -n auto
```

## Troubleshooting

### `ImportError: gam`
The `SauersML/gam` Rust extension did not build. Run:

```sh
rustup default stable
uv sync --dev --reinstall-package gam
```

Do NOT swap in the legacy pure-Python GAM library; see CLAUDE.md.

### `gnomon` missing
The wrapper raises `NotImplementedError` naming the missing binary. Install
it to `/Users/user/.local/bin/gnomon` and re-run. Nothing in this repo
should substitute a Python reimplementation of the scoring pipeline.

### `rustc` missing
`SauersML/gam` needs a Rust toolchain to build its wheel. Install with:

```sh
uv run python -c "import gam; print(gam.__version__)"
```

If that fails with a linker or `cc` error, install rustup and rerun
`uv sync --dev`.

### R-hat larger than 1.1 in MCMC
Too few MCMC samples or bad warm-start. Bump `MCMC_SAMPLES` /
`MCMC_BURN_IN` in `src/causal_pred/pipeline.py` and rerun
`scripts/run_full_pipeline.py`.

### Memory pressure
Reduce `MCMC_SAMPLES` and `MCMC_CHAINS` in `src/causal_pred/pipeline.py`.

### `NotImplementedError: partial availability`
Triggered when a stage tries to run on an incomplete input set (eg a real
biobank loader that has BMI but not HbA1c). Fix the loader to match the
column convention in `docs/ARCHITECTURE.md`.

### Cleanup

```sh
uv run python -c "import shutil; shutil.rmtree('outputs', ignore_errors=True)"
```

Removes cached outputs so the next pipeline run regenerates everything.
