# Runbook

Operational instructions for installing, running, and troubleshooting
`causal-pred`. Every shell command below goes through `uv run` so the
environment stays consistent with `uv.lock`.

## Install

The project uses `uv` for environment management. Clone the repo, then:

```sh
uv sync --dev
```

This resolves all runtime and dev dependencies, including the Rust-backed
`gam` library. If `gam` fails to build, see the troubleshooting section
below for `rustc missing`.

## Run the demo

The fastest sanity check. Runs the pipeline end-to-end on synthetic data
with small sample size:

```sh
uv run python -m causal_pred.pipeline
```

This writes `outputs/summary.json` and a handful of PNGs under
`outputs/plots/`.

## Run the full pipeline

The peer-review grade run. Larger `n`, more MCMC samples, more NUTS draws
from the distributional GAM, and complete validation metrics:

```sh
uv run python scripts/run_full_pipeline.py
```

Useful flags (inspect `--help` for the full list):

```sh
uv run python scripts/run_full_pipeline.py --help
uv run python scripts/run_full_pipeline.py --seed 17
uv run python scripts/run_full_pipeline.py --n-samples 50000
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
Too few MCMC samples or bad warm-start. Increase `--mcmc-iter` or supply
a better warm-start via `dagslam`:

```sh
uv run python scripts/run_full_pipeline.py --mcmc-iter 50000
```

### Memory pressure
Reduce `--n-samples` and `--n-posterior-draws` in
`scripts/run_full_pipeline.py`. The distributional GAM is the largest
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
