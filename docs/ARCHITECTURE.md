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
Ingest layer. There is one data-loading entry point used by
`pipeline.run_pipeline`: the real complete-case cohort CSV loader. It
returns the same `SyntheticDataset` shape used by the downstream stages, so
tests can still exercise synthetic matrices without changing the production
path. `synthetic.py` generates an AoU-shaped matrix with a known
ground-truth DAG -- still used by tests and by historical benchmarks.
`nodes.py` is the single source of truth for the synthetic schema's
`NODE_NAMES` (length 18) and the parallel `node_types` tuple.
`real_gwas.py` holds literature-derived IVW beta / SE / SNP-count cells
with PMID/DOI. `polygenic.py` shells out to the `gnomon` CLI for
scoring and HWE-PCA ancestry projection; it never reimplements scoring
in Python. `cohort.py` is the cohort-data branch of the same loader:
it accepts OMOP-shaped condition + measurement frames for cohort CSV
construction and uses a curated LOINC measurement catalog for the
production EHR stream. The cohort builder labels rows by DAG node
(BMI, HbA1c, HDL, LDL, triglycerides, systolic BP), normalises units
(mmol/L -> mg/dL, mmol/mol -> NGSP %), drops
physiologically impossible values, removes extreme outliers per-node
via a conservative IQR rule, collapses repeated measures by median,
and merges with a binary T2D node from the condition frame to produce
a participant-level wide CSV with core clinical, demographic, and lifestyle
columns. If that CSV lacks follow-up time/event columns, the pipeline builds a
required cached survival outcome from OMOP visit baseline, observation-period
end, and first post-baseline T2D diagnosis before DAGSLAM/MCMC/GAM. The
production cache is
`data/t2d_initial_nodes_complete.csv`;
`resolve_cohort_csv` checks that local file first, then copies the same
canonical filename from `$WORKSPACE_BUCKET/data/` via `gsutil cp`.
`load_cohort_dataset_with_person_ids` is the production loader because the
single pipeline aligns microarray-derived PRS and baseline-censored EHR
features back to participant IDs before DAGSLAM -> MCMC -> GAM. The EHR
measurement path is aggregated in BigQuery before download; it never pulls
raw measurement histories for the full AoU cohort.

### `genscore/`
Mechanistic-interpretability-style feature discovery. `crosscoder.py` learns
TopK sparse features shared between the PRS stream and the EHR stream.
`integrate.py` promotes cross-modal, active features into the DAG as
continuous nodes so the causal stack can estimate where those learned
mechanisms sit and how much they affect survival.

### `mrdag/`
Produces an edge-inclusion posterior matrix `pi` of shape `(p, p)` from
per-exposure GWAS summary statistics using the MrDAG model (Zuber et al.
2025) with the Wakefield 2009 spike-vs-practical-null Bayes factor.

### `scoring/`
Mixed-type marginal likelihoods for scoring a DAG against data. Gaussian
nodes use the BGe score (Geiger-Heckerman 2002, Kuipers-Moffa-Heckerman
2014). Binary nodes use a Laplace-approximated logistic marginal. Survival
nodes use a fixed-horizon IPCW logistic local likelihood during graph scoring;
the full `gam/` survival model is fit after structure sampling.

### `dagslam/`
Hill-climbing search through DAG space given a score function and an edge
prior. Used to produce the warm-start DAG for MCMC. The production path also
passes a structural edge mask so fixed roots such as PRS and derived feature
nodes cannot receive incoming edges, and the T2D outcome cannot point back to
baseline predictors.

### `mcmc/`
Structure MCMC over DAGs with the Giudici-Castelo neighbourhood correction,
MrDAG edge prior, and BGe/Laplace marginal likelihood. Emits samples that
are read downstream for parent-set posterior estimates. Every proposal kernel
uses the same structural edge mask as DAGSLAM, so posterior samples stay in
the biologically admissible graph space.

### `gam/`
Survival GAM wrappers around the `gamfit` Python bindings for the
`SauersML/gam` Rust engine. The production pipeline fits gamfit location-scale
survival models on posterior parent sets sampled by structure MCMC plus the
target's median-probability parent model, queries gamfit survival curves and
response-scale standard errors, then averages per-person survival curves by
parent-set posterior probability.

### `validation/`
Known-edge recall/AUROC, Nagelkerke R-squared, Brier decomposition
(Murphy 1973), ECE (Naeini et al. 2015), time-dependent AUC
(Heagerty 2000, Uno 2007).

### `plots.py`
Matplotlib figure helpers for edge heatmaps, calibration curves, and
survival fans.

### `pipeline.py`
Single production stage orchestrator. It resolves cohort/PRS/EHR inputs,
promotes crosscoder features, runs MrDAG, DAGSLAM, structure MCMC, survival
GAM parent-set averaging, validation, and causal-pathway extraction, then
writes `outputs/summary.json` through `_json_sanitise`.

### `benchmarks.py`
Throughput / scaling sweep runner used by `scripts/benchmark.py`.

## Data flow table

| Stage      | Input shape / units                          | Output shape / units                         |
|------------|----------------------------------------------|----------------------------------------------|
| `data`     | --                                           | `X (n, p)`, `time (n,)` years, `event (n,)`  |
| `mrdag`    | `betas (p, p)`, `ses (p, p)`                 | `pi (p, p)` edge-inclusion probs             |
| `dagslam`  | `X, pi, score_fn`                            | `G0` adjacency `(p, p)` in {0,1}             |
| `mcmc`     | `X, pi, G0, node_types`                      | `samples (S, p, p)`, `accept`, `rhat`        |
| `gam`      | `X, parent_sets, time, event`                | per-person survival/risk curves + uncertainty |
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
1. Implement the `SurvivalGAM` methods used by the pipeline behind a thin
   adapter module.
2. Replace the required dependency in `pyproject.toml` and update the lockfile.
3. Run the full suite and a benchmark smoke test before making the switch.

### How `gnomon` slots in
`data/polygenic.py` constructs a temporary working directory, writes VCF or
plink manifests that `gnomon` expects, and shells out to
`/Users/user/.local/bin/gnomon` for (a) per-sample polygenic scoring,
(b) HWE-PCA ancestry projection and (c) sample-term inference. The Python
code never recomputes any of these; it only parses the JSON/TSV output and
caches it alongside the input path. If `gnomon` is absent the wrapper
raises a clear `NotImplementedError` naming the missing binary.

### How to retarget at a different biobank
1. Write a loader that yields the cohort CSV plus stable participant IDs.
   Keep clinical units and map the model columns in `pipeline.py`.
2. Update the single loader/constants in `pipeline.py`; do not add a second
   production entry path.
3. Provide per-ancestry HWE-PCA eigenvectors so `gnomon` can project new
   samples into the reference PC space.
4. Re-examine the `real_gwas.py` IVs: some will not transport (eg HbA1c
   effects differ by assay). Replace or null out as needed.

## Reproducibility notes

* `uv.lock` pins the full dependency graph. `uv sync --dev` reproduces the
  environment byte-for-byte.
* Every stage takes an explicit `rng` argument. Children are derived via
  `numpy.random.Generator.spawn`-style splitting; no stage re-seeds.
* `gamfit` ships wheels for supported platforms. A Rust toolchain is only
  needed when no compatible wheel exists.
* `outputs/summary.json` is written through `_json_sanitise` in
  `pipeline.py`. Do not bypass it; numpy scalars and arrays must be
  converted or the file becomes non-round-trippable.
* The `gnomon` binary is not pinned by `uv`. Record its version in the
  pipeline summary when running on real data so the artefact is auditable.
