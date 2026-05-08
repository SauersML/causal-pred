#!/usr/bin/env bash
# One-shot real-data training bootstrap for the AoU Researcher Workbench.
#
# Preferred from a checkout:
#
#   bash causal-pred/scripts/bootstrap_aou.sh
#
# Fresh workbench:
#
#   curl -fsSL https://raw.githubusercontent.com/SauersML/causal-pred/main/scripts/bootstrap_aou.sh | bash
#
# Required AoU inputs:
#
#   $WORKSPACE_BUCKET/data/t2d_initial_nodes_complete.csv
#     or $WORKSPACE_BUCKET/data/t2d_initial_nodes_complete_case.csv
#   $GOOGLE_PROJECT for requester-pays microarray downloads
#
# Steps:
#   1. assert AoU workspace variables exist
#   2. install `uv` if missing
#   3. install a minimal stable Rust toolchain if missing (`gamfit` builds from source)
#   4. locate this checkout, or clone it into $HOME/causal-pred on a fresh workbench
#   5. `uv sync --locked --dev` into the repo-local .venv
#   6. install `gnomon` if missing
#   7. run the single real pipeline; it prepares/loads cached PRS, builds
#      EHR crosscoder features, runs MrDAG -> DAGSLAM -> MCMC -> GAM, and
#      mirrors small artefacts to WORKSPACE_BUCKET

set -euo pipefail

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

[[ -n "${WORKSPACE_BUCKET:-}" ]] || die "WORKSPACE_BUCKET is not set; run this inside the AoU Researcher Workbench"
[[ -n "${GOOGLE_PROJECT:-}" ]] || die "GOOGLE_PROJECT is not set; AoU microarray downloads are requester-pays"
need_cmd curl
need_cmd git
need_cmd gsutil

WORKSPACE_BUCKET="${WORKSPACE_BUCKET%/}"

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Rust --------------------------------------------------------------------
if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
    log "installing stable Rust toolchain"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
elif command -v rustup >/dev/null 2>&1; then
    log "ensuring stable Rust toolchain is active"
    rustup default stable
fi

# 3. repo --------------------------------------------------------------------
repo_from_script=""
script_source="${BASH_SOURCE[0]}"
if [[ -f "$script_source" ]]; then
    script_dir="$(cd "$(dirname "$script_source")" && pwd)"
    repo_candidate="$(cd "$script_dir/.." && pwd)"
    if [[ -f "$repo_candidate/pyproject.toml" && -d "$repo_candidate/src/causal_pred" ]]; then
        repo_from_script="$repo_candidate"
    fi
fi

if [[ -n "$repo_from_script" ]]; then
    REPO_DIR="$repo_from_script"
elif [[ -f "$PWD/pyproject.toml" && -d "$PWD/src/causal_pred" ]]; then
    REPO_DIR="$PWD"
else
    REPO_DIR="$HOME/causal-pred"
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        [[ ! -e "$REPO_DIR" ]] || die "$REPO_DIR exists but is not a git checkout"
        log "cloning repo into $REPO_DIR"
        git clone https://github.com/SauersML/causal-pred.git "$REPO_DIR"
    fi
fi
log "using repo at $REPO_DIR"
cd "$REPO_DIR"

# 4. python deps -------------------------------------------------------------
log "uv sync --locked --dev"
uv sync --locked --dev

# 5. gnomon ------------------------------------------------------------------
if ! command -v gnomon >/dev/null 2>&1; then
    log "installing gnomon via install.sh"
    curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash
fi
export PATH="$HOME/.local/bin:$PATH"

# 6. pipeline ----------------------------------------------------------------
log "running real AoU causal pipeline"
uv run python scripts/run_full_pipeline.py

log "done. artefacts in $REPO_DIR/outputs"
