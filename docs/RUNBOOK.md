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
Researcher Workbench, the bootstrap verifies that object, installs the
local toolchain, syncs dependencies, copies the CSV into `data/`, and runs
the pipeline:

```sh
bash causal-pred/scripts/bootstrap_aou.sh
```

This writes `outputs/summary.json` and the edge artefacts under `outputs/`.

## Run the full pipeline

The pipeline is a single fixed path: cohort wide CSV (resolved via the
local `data/` cache, then `$WORKSPACE_BUCKET/data/`) -> DAGSLAM hill-climb
-> structure MCMC -> save artefacts under `outputs/`. There are no
command-line flags; defaults are the configuration. To override hyperparameters, edit
``run_pipeline``'s defaults in `src/causal_pred/pipeline.py`.

```sh
uv run python scripts/run_full_pipeline.py
```

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
should fall back to a Python reimplementation of the scoring pipeline.

### `rustc` missing
`SauersML/gam` needs a Rust toolchain to build its wheel. Install with:

```sh
uv run python -c "import gam; print(gam.__version__)"
```

If that fails with a linker or `cc` error, install rustup and rerun
`uv sync --dev`.

### R-hat larger than 1.1 in MCMC
Too few MCMC samples or bad warm-start. Bump ``mcmc_samples`` /
``mcmc_burn_in`` defaults in
``src/causal_pred/pipeline.py``'s ``run_pipeline`` signature, or supply
a better warm-start via ``dagslam`` and rerun
``scripts/run_full_pipeline.py``.

### Memory pressure
Reduce ``mcmc_samples`` and ``mcmc_chains`` defaults in
``run_pipeline``. The distributional GAM is the largest
allocator; dropping NUTS draws from 2000 to 500 is usually enough.

### `NotImplementedError: partial availability`
Triggered when a stage tries to run on an incomplete input set (eg a real
biobank loader that has BMI but not HbA1c). Fix the loader to match the
column convention in `docs/ARCHITECTURE.md`, or pass `--allow-partial` if
the stage is explicitly designed for it.

### Cleanup

```sh
uv run python -c "import shutil; shutil.rmtree('outputs', ignore_errors=True)"
```

Removes cached outputs so the next pipeline run regenerates everything.
