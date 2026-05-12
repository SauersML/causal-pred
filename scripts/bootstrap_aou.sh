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
#   3. install a minimal stable Rust toolchain if a compatible `gamfit` wheel is unavailable
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

# 4. pull latest -------------------------------------------------------------
# Fast-forward only so a dirty / diverged checkout fails loudly instead of
# silently running stale code (which previously masked fixes that had been
# pushed to main between bootstrap invocations).
log "git pull --ff-only"
git pull --ff-only

# 5. python deps -------------------------------------------------------------
# Lock everything else to the committed pins, but always pull the newest
# gamfit available on PyPI. The Rust survival GAM is the most actively
# evolving dependency in the stack, and we never want a stale wheel
# silently masking fixes that have been published since the last `uv lock`.
log "uv sync --upgrade-package gamfit --dev (gamfit pinned to latest)"
uv sync --upgrade-package gamfit --dev

# 5. gnomon ------------------------------------------------------------------
log "installing gnomon via install.sh"
curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"

# 6. kill any stale pipeline before launching a new one ---------------------
# Python's module system is interpreter-local: once causal_pred.gam.survival
# (or any other module) is imported into a long-lived process, a later
# `git pull` does NOT cause the running process to pick up the new code,
# and loky workers it forks inherit the same stale in-memory modules. The
# symptom is "I pulled and re-ran bootstrap but the fits still take the
# old amount of time" -- this happens whenever a previous bootstrap left a
# pipeline + loky zombies running. We hard-kill those here so the next
# `uv run python` invocation is a brand-new interpreter with the freshly
# pulled source on disk.
log "killing any stale pipeline + loky workers from previous invocations"
pkill -9 -f "scripts/run_full_pipeline" 2>/dev/null || true
pkill -9 -f "joblib.externals.loky" 2>/dev/null || true
# Give the OS a moment to reap; then verify nothing is left.
sleep 2
if ps -ef | grep -E "run_full_pipeline|joblib.externals.loky" | grep -v grep >/dev/null; then
    log "WARNING: stale pipeline processes survived pkill -9; listing"
    ps -ef | grep -E "run_full_pipeline|joblib.externals.loky" | grep -v grep
    die "refusing to start a new pipeline alongside stale workers"
fi

# 7. pipeline ----------------------------------------------------------------
# Always mirror stdout+stderr to bootstrap.log so the status-check cell can
# read the Rust optimizer trace, BFGS convergence lines and panic messages
# *without* relying on the operator remembering to `tee` the invocation.
# Forgetting the tee made every silent worker death unrecoverable post-hoc;
# baking it into the script is the only way to make that impossible.
BOOTSTRAP_LOG="$REPO_DIR/bootstrap.log"
log "git head before launch: $(git -C "$REPO_DIR" log --oneline -1)"
log "gam/survival.py k=5 grep matches: $(grep -c 'k=5' "$REPO_DIR/src/causal_pred/gam/survival.py")"
log "running real AoU causal pipeline (mirroring to $BOOTSTRAP_LOG)"
# -u forces unbuffered stdout/stderr. PYTHONUNBUFFERED is also set inside the
# pipeline entrypoint as a belt-and-suspenders so progress logs from
# DAGSLAM / MCMC / BMA appear in real time on this non-TTY harness instead
# of sitting in the OS pipe buffer for minutes. `stdbuf -oL -eL` line-buffers
# the `tee` side so partial Rust log lines don't sit in the pipe for hours.
PYTHONUNBUFFERED=1 stdbuf -oL -eL uv run python -u scripts/run_full_pipeline.py 2>&1 \
    | stdbuf -oL -eL tee "$BOOTSTRAP_LOG"
# `pipefail` is on (`set -euo pipefail`), so a nonzero pipeline exit propagates.

log "done. artefacts in $REPO_DIR/outputs"
