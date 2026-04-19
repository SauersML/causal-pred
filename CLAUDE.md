# CLAUDE.md

Notes for future Claude sessions working on this repo.

## First principle

**Always use `uv run`.** Never invoke plain `python`, `python3`, or
`pytest`. This project is `uv`-managed and the virtualenv in `.venv/` is
only consistent with itself when commands go through `uv run`.

## What this repo is

An end-to-end Bayesian pipeline (MrDAG -> DAGSLAM -> structure MCMC ->
distributional survival GAM -> validation) for individual-level disease
risk modelling on biobank-shape data, with explicit uncertainty at both
structural (DAG) and parametric (GAM) levels.

## Where to look

- `docs/ARCHITECTURE.md` -- module map, data flow, extension points.
  Read this before touching more than one module.
- `docs/RUNBOOK.md` -- install, run, regenerate, troubleshoot.
- `docs/MATHEMATICAL_NOTES.md` -- equation-by-equation transcription of
  MrDAG and DAGSLAM and the approximations we make.
- `README.md` -- short entry point with quickstart.

## External tools

- **`gnomon` CLI** at `/Users/user/.local/bin/gnomon` -- used by
  `causal_pred.data.polygenic` for polygenic scoring, HWE-PCA ancestry
  projection, and sample-term inference. Never reimplement scoring in
  Python; shell out to `gnomon`.
- **`gam` Python library** (`SauersML/gam`, installed from git via
  `pyproject.toml`) -- Rust-backed GAM engine used by
  `causal_pred.gam.survival`. Building it requires `rustup`.

## Testing

```sh
uv run pytest -q
```

Individual module tests:

```sh
uv run pytest tests/test_mcmc.py -q
uv run pytest tests/test_pipeline.py -q
uv run pytest tests/test_docs.py -q
```

## Hard rules

- **Do not reintroduce the legacy pure-Python GAM library whose import
  name starts with the letters "p" and "y".** We intentionally depend on
  `SauersML/gam` (Rust backend). Every past attempt to swap in that older
  library caused regressions; if you find yourself reaching for it, stop
  and re-read `docs/ARCHITECTURE.md` under "How to swap the GAM backend".
  This is a hard tripwire.
- **Don't edit `src/causal_pred/data/real_gwas.py` without a citation.**
  The table is literature-derived. New or changed cells must carry a
  PMID/DOI for the reported IVW beta / SE / SNP count. Cells without a
  defensible published value stay as `LITERATURE_UNAVAILABLE`.
- **Don't use `pip install ...`.** Use `uv add <pkg>` (runtime dep) or
  `uv add --dev <pkg>` (dev dep). If you see a stray `pip` reference
  anywhere, that's a bug -- file a task.
- **Don't skip `uv run`.** Even one-off Python commands go through it.

## When tests fail in files you didn't touch

Multiple teammates own different modules. If `tests/test_<x>.py` fails
after a change you made to a different module, do **not** patch the test
blindly. Find the teammate who owns that module (see
`~/.claude/teams/causal-pred/config.json`) and coordinate. Common
ownership:

- `mcmc/` -> `mcmc` teammate
- `data/polygenic.py` -> `polygenic` teammate
- `docs/MATHEMATICAL_NOTES.md` -> `literature` teammate
- `pipeline.py`, `scripts/run_full_pipeline.py` -> `integrator` teammate
- `scripts/benchmark.py` -> `benchmarks` teammate
- `paper/` -> `paper` teammate
- `plots/` -> `plots` teammate
- `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, this file -> `docs` teammate

## Conventions

- Time is in years since baseline (cohort age 40 in the synthetic
  generator).
- Continuous covariates stay on their clinical scale (BMI kg/m^2, LDL
  mmol/L, HbA1c %, BP mmHg); standardise inside scorers when needed, not
  at ingest.
- `node_types` is a length-`p` tuple of
  `{"continuous", "binary", "survival"}` in `NODE_NAMES` order -- that
  ordering is the single source of truth.
- Every stage takes an explicit `rng` argument; do not re-seed inside a
  stage, derive child RNGs from the passed one.
- JSON output is written via `_json_sanitise` in `pipeline.py`; numpy
  scalars and arrays must go through it.

## Reproducibility checklist

1. Update the stage module's `__init__.py` docstring.
2. Update `docs/ARCHITECTURE.md` (module section + data-flow table).
3. Update `pipeline.py`'s call site.
4. Run `uv run pytest -q` and ensure the full suite is green.
5. If the change affects reported metrics, re-run
   `uv run python -m causal_pred.pipeline` and inspect the diff in
   `outputs/summary.json`.
