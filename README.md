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

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map, data flow, and
  extension points (add a new disease, retarget at a different biobank).
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — install, run, regenerate, troubleshoot.
- [`docs/MATHEMATICAL_NOTES.md`](docs/MATHEMATICAL_NOTES.md) — equation-level
  transcription of MrDAG and DAGSLAM with all approximations labelled.

## Pipeline

```text
  External GWAS            AoU cohort CSV + microarray genotypes
  summary stats            + baseline-censored EHR stream
        |                            |
        v                            v
      MrDAG  -- edge priors pi -->  gnomon-scored PRS nodes
                     |              + TopK crosscoder features
                     +----------+--------------+
                                v
                  DAGSLAM warm start -> Structure MCMC
                                |
                                v
                    posterior parent sets + pathways
                                |
                                v
                    gamfit survival GAM
                                |
                                v
         per-person survival curves + causal pathway probabilities
```

## Layout

```text
src/causal_pred/
  data/        synthetic biobank-shape data, real GWAS table, gnomon wrapper
  mrdag/       MrDAG: produces edge-inclusion-probability matrix pi
  scoring/     mixed-type marginal-likelihood scores (BGe / Laplace)
  dagslam/     DAGSLAM hill-climber for the warm-start DAG
  mcmc/        structure MCMC with MrDAG prior
  gam/         survival GAM (SauersML/gam backend)
  validation/  known-edge checks, Nagelkerke R^2, calibration, time-dep AUC
  plots.py     figure helpers (heatmaps, calibration, survival fans)
  pipeline.py  end-to-end orchestration

scripts/       runnable drivers (run_full_pipeline.py, generate_figures.py, benchmark.py)
tests/         pytest tests for every component
outputs/       JSON / PNG artefacts produced by the pipeline
paper/         LaTeX paper and build script
docs/          architecture, runbook, mathematical notes
```

## Quickstart

In the terminal, setup and run with one command:

```sh
if [ -d causal-pred/.git ]; then git -C causal-pred stash push -m aou-local-uv-lock -- uv.lock >/dev/null || true; git -C causal-pred pull --ff-only; else git clone https://github.com/SauersML/causal-pred.git causal-pred; fi && bash causal-pred/scripts/bootstrap_aou.sh
```

The bootstrap verifies AoU workspace inputs, installs local tools, syncs locked
dependencies, prepares or restores PRS/EHR intermediates, and runs the single
causal pipeline.

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for full instructions and
troubleshooting.
