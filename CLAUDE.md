# Claude Notes

Use `uv run` for Python commands so the repo-local environment and lockfile
are respected.

The production path is `uv run python scripts/run_full_pipeline.py`. It builds
or restores `gnomon` polygenic scores, promotes genome/EHR crosscoder features,
runs MrDAG -> DAGSLAM -> MCMC, then fits the `gamfit` survival GAM.

`gnomon` is the only supported genotype scoring tool. Fix its call site if
scoring breaks; do not reimplement PRS scoring in Python.

The survival backend is the Rust-backed `gamfit` package. Do not replace it
with older Python GAM libraries.
