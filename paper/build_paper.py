r"""Stamp dynamic numeric values into paper/main.tex from outputs/summary.json.

Usage
-----
    uv run python paper/build_paper.py \
        [--summary outputs/summary.json] \
        [--tex paper/main.tex] \
        [--out paper/main.tex]

The script walks a fixed mapping from summary-JSON keys to LaTeX
`\newcommand` macros that are already used in `main.tex`, and rewrites
the `\providecommand{<name>}{...}` block near the top of main.tex with
the up-to-date values.  Entries missing from the summary are rendered
as an `\fbox{TBD}` placeholder so the PDF stays compilable.

Design goals
------------
* Deterministic: the same (summary, tex) inputs always yield the same
  output file (byte-for-byte).
* Idempotent: running twice in a row is a no-op.
* Safe if the summary file is absent: leave all macros as TBD.

The build script does NOT invoke pdflatex -- the Makefile does that.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

# Mapping: LaTeX macro name -> (dotted key path into summary.json, format).
# - Dotted path "a.b.c" means summary["a"]["b"]["c"].
# - Format is a Python format spec (applied via format(value, fmt)) used only
#   for floats; ints and strings are stringified directly.
# - Missing or NaN values fall back to \fbox{TBD}.
MACRO_MAP: tuple[tuple[str, str, str], ...] = (
    ("nSamples", "dataset.n", ",d"),
    ("nNodes", "dataset.p", "d"),
    ("eventRate", "dataset.event_rate", ".3f"),
    ("auroc", "metrics.auroc", ".3f"),
    ("aurocFive", "metrics.time_dep_auc.5", ".3f"),
    ("aurocTen", "metrics.time_dep_auc.10", ".3f"),
    ("aurocFifteen", "metrics.time_dep_auc.15", ".3f"),
    ("brier", "metrics.brier", ".3f"),
    ("ibs", "metrics.ibs", ".3f"),
    ("nagelkerke", "metrics.nagelkerke_r2", ".3f"),
    ("ece", "metrics.ece", ".3f"),
    ("edgeRecall", "metrics.edge_recall", ".1%"),
    ("edgeAuroc", "metrics.edge_auroc", ".3f"),
    ("mcmcIter", "mcmc.n_samples", ",d"),
    ("mcmcChains", "mcmc.n_chains", "d"),
    ("mcmcRhat", "mcmc.max_rhat", ".2f"),
    ("mcmcAccept", "mcmc.mean_accept", ".2f"),
    ("runtimeSeconds", "runtime_seconds", ".1f"),
)

TBD = r"\fbox{TBD}"

# Block delimiters in main.tex.  Everything between these two comment lines
# is replaced on each build.
BLOCK_START = (
    "% ---------------------------------------------------------------------------\n"
    "% Dynamic numeric values.  These are overwritten at build time by\n"
    "% paper/build_paper.py, which reads outputs/summary.json.  The defaults\n"
    "% below keep the document compilable even without a pipeline run.\n"
    "% ---------------------------------------------------------------------------\n"
)
BLOCK_END_MARK = "% Convenient math shortcuts"


def _lookup(summary: Mapping[str, Any], path: str) -> Any:
    """Walk a dotted path through nested dicts; return None if missing."""
    node: Any = summary
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return None
    return node


def _fmt(value: Any, fmt: str) -> str:
    """Format ``value`` with ``fmt``; fall back to TBD for nan/None."""
    if value is None:
        return TBD
    try:
        if isinstance(value, float) and math.isnan(value):
            return TBD
    except TypeError:
        pass
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


def build_macro_block(summary: Mapping[str, Any]) -> str:
    r"""Return the replacement ``\providecommand`` block as a single string."""
    lines = [BLOCK_START.rstrip("\n")]
    for macro, key, fmt in MACRO_MAP:
        value = _lookup(summary, key)
        rendered = _fmt(value, fmt)
        # LaTeX-escape percent signs that come out of `.1%` format
        rendered = rendered.replace("%", r"\%")
        lines.append(r"\providecommand{\%s}{%s}" % (macro, rendered))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# File rewrite
# ---------------------------------------------------------------------------

_BLOCK_REGEX = re.compile(
    r"% -{10,}\s*\n"
    r"% Dynamic numeric values\..*?"  # up to \providecommand lines
    r"(?:^\\providecommand\{.*?\}\{.*?\}\s*\n)+",
    flags=re.DOTALL | re.MULTILINE,
)


def stamp_tex(tex_source: str, summary: Mapping[str, Any]) -> str:
    """Return ``tex_source`` with its dynamic-macros block replaced."""
    new_block = build_macro_block(summary)
    if _BLOCK_REGEX.search(tex_source):
        # Use a function replacement so backslashes in new_block are not
        # interpreted as re back-references.
        return _BLOCK_REGEX.sub(lambda _m: new_block, tex_source, count=1)
    # No block found -- insert just before the math-shortcuts comment, or
    # failing that, just after \documentclass.
    anchor = tex_source.find(BLOCK_END_MARK)
    if anchor == -1:
        anchor = tex_source.find(r"\begin{document}")
    if anchor == -1:
        return new_block + tex_source
    return tex_source[:anchor] + new_block + tex_source[anchor:]


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=repo_root / "outputs" / "summary.json"
    )
    parser.add_argument("--tex", type=Path, default=repo_root / "paper" / "main.tex")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path; defaults to --tex (in-place).",
    )
    args = parser.parse_args(argv)

    if args.summary.exists():
        with args.summary.open() as fh:
            summary = json.load(fh)
    else:
        summary = {}

    tex_source = args.tex.read_text()
    stamped = stamp_tex(tex_source, summary)
    out_path = args.out or args.tex
    out_path.write_text(stamped)
    print(
        f"[build_paper] wrote {out_path} ({len(stamped):,} bytes, "
        f"summary keys={list(summary)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
