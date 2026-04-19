# Causal-Pred: Causal Disease Modeling with Uncertainty

End-to-end implementation of the project proposed in `paper/main.tex`: predict
disease *and* its causes on biobank-scale individual-level data, with explicit
uncertainty at both structural (DAG) and parametric (GAM) levels.

Target disease: **Type 2 Diabetes (T2D)**. T2D has well-characterised causal
parents (BMI, HbA1c, lifestyle, genetic risk), strong Mendelian-randomisation
evidence (BMI -> T2D, LDL -> T2D null, etc.), and mature polygenic scores,
which makes the validation framework meaningful even on synthetic data.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map, data flow, and
  extension points (add a new disease, swap the GAM backend, retarget at a
  different biobank).
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — install, run, regenerate, troubleshoot.
- [`docs/MATHEMATICAL_NOTES.md`](docs/MATHEMATICAL_NOTES.md) — equation-level
  transcription of MrDAG and DAGSLAM with all approximations labelled.
- [`CLAUDE.md`](CLAUDE.md) — notes for future Claude sessions; start here if
  you are automating work on this repo.

## Pipeline

```text
  External GWAS            Biobank (SIMULATED HERE: data/synthetic.py)
  summary stats            individual-level matrix
        |                            |
        v                            v
      MrDAG  -- edge priors pi -->  DAGSLAM -- starting DAG -->
                                        |
                                        v
                            Structure MCMC over DAGs
                                        |
                            posterior parent sets
                                        |
                                        v
                        Distributional Survival GAM
                        (SauersML/gam, REML + NUTS)
                                        |
                                        v
                    survival curves  +  causal-pathway probs
```

## Layout

```text
src/causal_pred/
  data/        synthetic biobank-shape data, real GWAS table, gnomon wrapper
  mrdag/       MrDAG: produces edge-inclusion-probability matrix pi
  scoring/     mixed-type marginal-likelihood scores (BGe / Laplace)
  dagslam/     DAGSLAM hill-climber for the warm-start DAG
  mcmc/        structure MCMC with MrDAG prior
  gam/         distributional survival GAM (SauersML/gam backend)
  validation/  known-edge checks, Nagelkerke R^2, calibration, time-dep AUC
  plots.py     figure helpers (heatmaps, calibration, survival fans)
  pipeline.py  end-to-end orchestration

scripts/       runnable drivers (run_full_pipeline.py, generate_report.py, benchmark.py)
tests/         pytest tests for every component
outputs/       JSON / PNG artefacts produced by the pipeline
paper/         LaTeX paper and build script
docs/          architecture, runbook, mathematical notes
```

## Quickstart

Install (requires `uv` and a working Rust toolchain for the `gam` extension):

```sh
uv sync --dev
```

Run the demo end-to-end on synthetic data:

```sh
uv run python -m causal_pred.pipeline
```

Run the full peer-review-grade pipeline:

```sh
uv run python scripts/run_full_pipeline.py
```

Run all tests:

```sh
uv run pytest -q
```

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for full instructions, CLI flags,
real-genotype scoring with `gnomon`, and troubleshooting.

## Dependencies

`numpy`, `scipy`, `pandas`, `scikit-learn`, `networkx`, `statsmodels`,
`matplotlib`, and the Rust-backed `gam` Python library from
[SauersML/gam](https://github.com/SauersML/gam).

## Disclaimer

*All of Us* individual-level data is **not** usable here — it requires a
Researcher Workbench and cannot be downloaded. This repo generates an
AoU-shaped synthetic dataset with known ground-truth causal structure so the
full pipeline runs end-to-end and validation metrics are computable. The
pipeline is written to ingest real biobank data once a local loader supplies
it in the column convention documented in `docs/ARCHITECTURE.md`.
