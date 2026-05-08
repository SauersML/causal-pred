#!/usr/bin/env bash
# One-shot bootstrap for the AoU Researcher Workbench.
#
#   curl -fsSL https://raw.githubusercontent.com/SauersML/causal-pred/main/scripts/bootstrap_aou.sh | bash
#
# Steps (each is idempotent):
#   1. install `uv` if missing
#   2. clone (or fast-forward) the causal-pred repo
#   3. `uv sync --dev` to install Python deps into a private .venv
#   4. install the `gnomon` CLI (via gnomon's install.sh) if missing
#   5. fetch the AoU v8 microarray PLINK triple into $GENO_DIR
#      (skipped when FETCH_GENOTYPES=0; resumes individual missing files)
#   6. run the pipeline (skipped when RUN_PIPELINE=0)
#
# Overridable env-vars:
#   REPO_DIR           where to clone (default: $HOME/causal-pred)
#   GENO_DIR           where to cache arrays.{bed,bim,fam} (default: $REPO_DIR/genomes)
#   FETCH_GENOTYPES    1 (default) downloads ~181 GiB; set 0 to skip
#   RUN_PIPELINE       1 (default) runs the cohort pipeline; set 0 to skip
#
# Required env-vars (set automatically inside the AoU workbench):
#   GOOGLE_PROJECT     billing project for the requester-pays bucket
#   WORKSPACE_BUCKET   used by resolve_cohort_csv to fetch the cohort CSV

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/causal-pred}"
GENO_DIR="${GENO_DIR:-$REPO_DIR/genomes}"
FETCH_GENOTYPES="${FETCH_GENOTYPES:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. repo --------------------------------------------------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
    log "updating repo at $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only
else
    log "cloning repo into $REPO_DIR"
    git clone https://github.com/SauersML/causal-pred.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# 3. python deps -------------------------------------------------------------
log "uv sync --dev"
uv sync --dev

# 4. gnomon ------------------------------------------------------------------
if ! command -v gnomon >/dev/null 2>&1; then
    log "installing gnomon via install.sh"
    curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash
fi
export PATH="$HOME/.local/bin:$PATH"

# 5. genotypes ---------------------------------------------------------------
if [[ "$FETCH_GENOTYPES" == "1" ]]; then
    if [[ -z "${GOOGLE_PROJECT:-}" ]]; then
        log "WARNING: GOOGLE_PROJECT is not set; cannot bill the requester-pays bucket -- skipping genotype fetch"
    else
        log "fetching AoU v8 microarray PLINK files into $GENO_DIR (resumable)"
        mkdir -p "$GENO_DIR"
        GENO_DIR="$GENO_DIR" uv run python - <<'PY'
import os
from causal_pred.data.cohort import resolve_aou_genotypes
bed = resolve_aou_genotypes(cache_dir=os.environ["GENO_DIR"])
print(f"genotypes ready: {bed}")
PY
    fi
fi

# 6. pipeline ----------------------------------------------------------------
if [[ "$RUN_PIPELINE" == "1" ]]; then
    log "running pipeline"
    uv run python scripts/run_full_pipeline.py
fi

log "done. artefacts in $REPO_DIR/outputs"
