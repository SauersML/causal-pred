"""Tests for the paper/ subdirectory.

These tests are intentionally lightweight so they pass on a stock CI
machine without a TeX Live install.  The only heavy test
(`test_paper_compiles`) is skipped unless pdflatex/latexmk is on PATH.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
BIB = PAPER_DIR / "bib.bib"


# ---------------------------------------------------------------------------
# Compilation (skipped without a LaTeX toolchain)
# ---------------------------------------------------------------------------


def _latex_tool():
    for name in ("latexmk", "pdflatex"):
        path = shutil.which(name)
        if path:
            return name, path
    return None, None


@pytest.mark.skipif(
    _latex_tool()[0] is None, reason="neither latexmk nor pdflatex on PATH"
)
def test_paper_compiles(tmp_path):
    """End-to-end: copy paper/ to tmp, run make quick, assert exit 0."""
    scratch = tmp_path / "paper"
    shutil.copytree(PAPER_DIR, scratch)
    # Copy the build_paper.py's expected summary location so stamping works
    name, _ = _latex_tool()
    if name == "latexmk":
        cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    proc = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (
        f"{name} failed:\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )
    assert (scratch / "main.pdf").exists()


# ---------------------------------------------------------------------------
# Citation resolution
# ---------------------------------------------------------------------------

_CITE_REGEX = re.compile(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_BIBKEY_REGEX = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)", flags=re.MULTILINE)


def test_all_citations_resolve():
    tex = MAIN_TEX.read_text()
    bib = BIB.read_text()
    bib_keys = set(_BIBKEY_REGEX.findall(bib))

    cited: set[str] = set()
    for match in _CITE_REGEX.finditer(tex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                cited.add(key)

    assert cited, "No citations found in main.tex"
    missing = cited - bib_keys
    assert not missing, (
        f"{len(missing)} cite keys missing from bib.bib: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# build_paper.py stamping
# ---------------------------------------------------------------------------


def test_newcommand_stamping(tmp_path):
    """paper/build_paper.py must substitute values from a mock summary."""
    import sys

    sys.path.insert(0, str(PAPER_DIR))
    try:
        import build_paper  # noqa: E402
    finally:
        sys.path.pop(0)

    summary = {
        "dataset": {"n": 5000, "p": 18, "event_rate": 0.123},
        "metrics": {
            "auroc": 0.82,
            "time_dep_auc": {"5": 0.80, "10": 0.82, "15": 0.81},
            "brier": 0.09,
            "ibs": 0.11,
            "nagelkerke_r2": 0.41,
            "ece": 0.04,
            "edge_recall": 0.67,
            "edge_auroc": 0.91,
        },
        "mcmc": {
            "n_samples": 10000,
            "n_chains": 4,
            "max_rhat": 1.05,
            "mean_accept": 0.24,
        },
        "runtime_seconds": 125.5,
    }
    tex = MAIN_TEX.read_text()
    stamped = build_paper.stamp_tex(tex, summary)

    # Key assertions: numeric values appear in stamped source
    assert r"\providecommand{\auroc}{0.820}" in stamped
    assert r"\providecommand{\aurocTen}{0.820}" in stamped
    assert r"\providecommand{\nSamples}{5,000}" in stamped
    assert r"\providecommand{\nNodes}{18}" in stamped
    assert r"\providecommand{\mcmcChains}{4}" in stamped
    assert r"\providecommand{\mcmcRhat}{1.05}" in stamped
    # Percentage is \%-escaped
    assert r"\providecommand{\edgeRecall}{67.0\%}" in stamped
    # Idempotent: running stamp twice produces the same result
    assert build_paper.stamp_tex(stamped, summary) == stamped


def test_build_paper_missing_summary_is_tbd(tmp_path):
    """If summary.json is absent, every macro falls back to \\fbox{TBD}."""
    import sys

    sys.path.insert(0, str(PAPER_DIR))
    try:
        import build_paper  # noqa: E402
    finally:
        sys.path.pop(0)

    tex = MAIN_TEX.read_text()
    stamped = build_paper.stamp_tex(tex, {})
    assert r"\providecommand{\auroc}{\fbox{TBD}}" in stamped
    assert r"\providecommand{\nSamples}{\fbox{TBD}}" in stamped


def test_build_paper_cli(tmp_path):
    """The CLI writes a stamped file and exits cleanly."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset": {"n": 1000, "p": 18, "event_rate": 0.2},
                "metrics": {"auroc": 0.75},
            }
        )
    )
    out_path = tmp_path / "main.tex"

    proc = subprocess.run(
        [
            "python",
            str(PAPER_DIR / "build_paper.py"),
            "--summary",
            str(summary_path),
            "--tex",
            str(MAIN_TEX),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    stamped = out_path.read_text()
    assert r"\providecommand{\auroc}{0.750}" in stamped
    assert r"\providecommand{\nSamples}{1,000}" in stamped
