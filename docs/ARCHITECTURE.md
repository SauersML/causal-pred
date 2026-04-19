# Architecture

This document describes the module layout of `causal-pred`, the data flow
between stages, the extension points, and reproducibility notes. Read it
before touching more than one module.

## High-level diagram

```text
  +--------------------+      +-----------------------+
  |  External GWAS     |      |  Biobank individual   |
  |  summary stats     |      |  level matrix (X, Y)  |
  |  (per-exposure IV) |      |  (synthetic generator |
  |                    |      |   or real loader)     |
  +--------+-----------+      +-----------+-----------+
           |                              |
           v                              v
     +-----+------+               +-------+---------+
     |  mrdag/    |               |  data/          |
     |  edge prior|               |  synthetic.py   |
     |  matrix pi |               |  polygenic.py   |
     +-----+------+               +-------+---------+
           |                              |
           +---------------+--------------+
                           v
                   +-------+--------+
                   |  dagslam/      |
                   |  warm-start DAG|
                   +-------+--------+
                           |
                           v
                   +-------+--------+   scoring/  (BGe / Laplace)
                   |  mcmc/         |<-- mixed-type marginal likelihood
                   |  structure MCMC|
                   +-------+--------+
                           |
                   posterior parent sets
                           |
                           v
                   +-------+--------+
                   |  gam/ survival |
                   |  (SauersML/gam)|
                   +-------+--------+
                           |
                           v
                   +-------+--------+     +--------+
                   |  validation/   |---->| plots/ |
                   +----------------+     +--------+
                           |
                           v
                   outputs/summary.json
                   outputs/plots/*.png
```

## Module by module

### `data/`
Ingest layer. `synthetic.py` generates an AoU-shaped matrix with a known
ground-truth DAG so the pipeline and its metrics are meaningful end-to-end.
`nodes.py` is the single source of truth for `NODE_NAMES` (length `p`) and
the parallel `node_types` tuple. `real_gwas.py` holds literature-derived
IVW beta / SE / SNP-count cells with PMID/DOI. `polygenic.py` shells out to
the `gnomon` CLI for scoring and HWE-PCA ancestry projection; it never
reimplements scoring in Python.

### `mrdag/`
Produces an edge-inclusion posterior matrix `pi` of shape `(p, p)` from
per-exposure GWAS summary statistics using the MrDAG model (Zuber et al.
2025) with the Wakefield 2009 spike-vs-practical-null Bayes factor.

### `scoring/`
Mixed-type marginal likelihoods for scoring a DAG against data. Gaussian
nodes use the BGe score (Geiger-Heckerman 2002, Kuipers-Moffa-Heckerman
2014). Binary nodes use a Laplace-approximated logistic marginal. Survival
nodes defer to `gam/` for an EB / NUTS approximation to the marginal.

### `dagslam/`
Hill-climbing search through DAG space given a score function and an edge
prior. Used to produce the warm-start DAG for MCMC.

### `mcmc/`
Structure MCMC over DAGs with the Giudici-Castelo neighbourhood correction,
MrDAG edge prior, and BGe/Laplace marginal likelihood. Emits samples that
are read downstream for parent-set posterior estimates.

### `gam/`
Distributional survival GAM wrappers around the `SauersML/gam` Rust engine.
`splines.py` builds the P-spline bases (Eilers-Marx 1996), `nuts.py`
produces NUTS samples from the posterior, `survival.py` composes the
distributional parameterisation (RigbyStasinopoulos 2005) on survival time.

### `validation/`
Known-edge recall/AUROC, Nagelkerke R-squared, Brier decomposition
(Murphy 1973), ECE (Naeini et al. 2015), time-dependent AUC
(Heagerty 2000, Uno 2007).

### `plots.py`
Matplotlib figure helpers for edge heatmaps, calibration curves, and
survival fans.

### `pipeline.py`
Stage orchestrator. Takes an RNG, invokes each stage with a child RNG,
writes `outputs/summary.json` through `_json_sanitise` and an analogous
plots directory.

### `benchmarks.py`
Throughput / scaling sweep runner used by `scripts/benchmark.py`.

## Data flow table

| Stage      | Input shape / units                          | Output shape / units                         |
|------------|----------------------------------------------|----------------------------------------------|
| `data`     | --                                           | `X (n, p)`, `time (n,)` years, `event (n,)`  |
| `mrdag`    | `betas (p, p)`, `ses (p, p)`                 | `pi (p, p)` edge-inclusion probs             |
| `dagslam`  | `X, pi, score_fn`                            | `G0` adjacency `(p, p)` in {0,1}             |
| `mcmc`     | `X, pi, G0, node_types`                      | `samples (S, p, p)`, `accept`, `rhat`        |
| `gam`      | `X, parent_sets, time, event`                | per-subject survival curves, posterior draws |
| `validation` | ground-truth DAG + per-subject risks        | metrics dict                                 |

Time is in years since cohort baseline (age 40 in the synthetic generator).
Continuous covariates stay on their clinical scale; scorers standardise
internally where needed. `node_types` is a length-`p` tuple of
`{"continuous", "binary", "survival"}` in `NODE_NAMES` order.

## Extension points

### How to add a new disease
1. Add the outcome node to `NODE_NAMES` in `data/nodes.py` with a
   `"survival"` type and declare its upstream causes.
2. Add IV cells (beta, se, SNP count, PMID) to `data/real_gwas.py` for each
   exposure that has a published Mendelian-randomisation estimate onto the
   new outcome. Cells without a published value stay as
   `LITERATURE_UNAVAILABLE`.
3. Update the synthetic generator's ground-truth DAG so validation metrics
   stay meaningful.
4. Rerun `uv run python -m causal_pred.pipeline` and inspect the diff in
   `outputs/summary.json`.

### How to swap the GAM backend
The only supported GAM backend is the `SauersML/gam` Rust engine pinned in
`pyproject.toml`. Do not reach for the legacy pure-Python library whose
import name begins with "py" -- it has repeatedly caused regressions and is
a hard tripwire (see CLAUDE.md). If you genuinely need a new backend:
1. Implement the two public functions used by `gam/survival.py`
   (`fit_posterior`, `posterior_predict`) behind a thin adapter module.
2. Gate the import in `gam/__init__.py` so test collection still works
   when the new backend is missing.
3. Add a CI toggle and run the full suite with both backends before
   making the switch default.

### How `gnomon` slots in
`data/polygenic.py` constructs a temporary working directory, writes VCF or
plink manifests that `gnomon` expects, and shells out to
`/Users/user/.local/bin/gnomon` for (a) per-sample polygenic scoring,
(b) HWE-PCA ancestry projection and (c) sample-term inference. The Python
code never recomputes any of these; it only parses the JSON/TSV output and
caches it alongside the input path. If `gnomon` is absent the wrapper
raises a clear `NotImplementedError` naming the missing binary rather than
falling back to an inexact Python reimplementation.

### How to retarget at a different biobank
1. Write a loader that yields `(X, time, event, node_types)` matching the
   column order in `data/nodes.py`. Keep clinical units.
2. Point `pipeline.py` at the new loader behind a `--data` flag rather
   than editing the synthetic generator.
3. Provide per-ancestry HWE-PCA eigenvectors so `gnomon` can project new
   samples into the reference PC space.
4. Re-examine the `real_gwas.py` IVs: some will not transport (eg HbA1c
   effects differ by assay). Replace or null out as needed.

## Reproducibility notes

* `uv.lock` pins the full dependency graph. `uv sync --dev` reproduces the
  environment byte-for-byte.
* Every stage takes an explicit `rng` argument. Children are derived via
  `numpy.random.Generator.spawn`-style splitting; no stage re-seeds.
* Building `SauersML/gam` requires a working Rust toolchain
  (`rustup default stable`). The wheel is compiled once per environment.
* `outputs/summary.json` is written through `_json_sanitise` in
  `pipeline.py`. Do not bypass it; numpy scalars and arrays must be
  converted or the file becomes non-round-trippable.
* The `gnomon` binary is not pinned by `uv`. Record its version in the
  pipeline summary when running on real data so the artefact is auditable.
