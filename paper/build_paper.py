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
# Dotted paths look up against a merged {summary.json, benchmarks.json}
# namespace where benchmark entries sit under "baselines.<name>.<key>" and
# time_dep_auc values are also surfaced as ``time_dep_auc_at_10`` for
# convenience (nearest grid point to 10 y).
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
    # -- benchmark table (from outputs/benchmarks.json) --------------------
    ("benchKMNagelkerke", "baselines.kaplan_meier.nagelkerke_at_10y", ".3f"),
    ("benchKMTdAuc", "baselines.kaplan_meier.time_dep_auc_at_10", ".3f"),
    ("benchKMIbs", "baselines.kaplan_meier.ibs", ".3f"),
    ("benchCoxNagelkerke", "baselines.cox_ph.nagelkerke_at_10y", ".3f"),
    ("benchCoxTdAuc", "baselines.cox_ph.time_dep_auc_at_10", ".3f"),
    ("benchCoxIbs", "baselines.cox_ph.ibs", ".3f"),
    ("benchLogisticNagelkerke", "baselines.naive_logistic.nagelkerke_at_10y", ".3f"),
    ("benchLogisticTdAuc", "baselines.naive_logistic.time_dep_auc_at_10", ".3f"),
    ("benchLogisticIbs", "baselines.naive_logistic.ibs", ".3f"),
    ("benchMRAuprc", "baselines.mr_ivw.edge_auprc", ".3f"),
    ("benchMRAuroc", "baselines.mr_ivw.edge_auroc", ".3f"),
    ("benchCausalPredNagelkerke", "baselines.causal_pred.nagelkerke_at_10y", ".3f"),
    ("benchCausalPredTdAuc", "baselines.causal_pred.time_dep_auc_at_10", ".3f"),
    ("benchCausalPredIbs", "baselines.causal_pred.ibs", ".3f"),
    ("benchCausalPredEdgeAuprc", "baselines.causal_pred.edge_auprc", ".3f"),
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
    """Walk a dotted path through nested dicts; return None if missing.

    Supports a small number of synthetic keys on top of raw dict walking:
    * ``time_dep_auc_at_<T>`` under a baseline node resolves to the entry of
      ``time_dep_auc.auc`` whose matching ``time_dep_auc.times`` entry is
      nearest to ``T`` (integer years).
    """
    node: Any = summary
    parts = path.split(".")
    for i, part in enumerate(parts):
        if (
            isinstance(node, Mapping)
            and part.startswith("time_dep_auc_at_")
            and "time_dep_auc" in node
        ):
            try:
                target = float(part.split("_at_", 1)[1])
            except ValueError:
                return None
            td = node["time_dep_auc"]
            if not isinstance(td, Mapping):
                return None
            times = td.get("times") or []
            aucs = td.get("auc") or []
            if not times or len(times) != len(aucs):
                return None
            best = min(
                range(len(times)),
                key=lambda k: abs(float(times[k]) - target),
            )
            return aucs[best]
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


def _merge(dst: dict, src: Mapping[str, Any]) -> dict:
    """Shallow-merge ``src`` into ``dst`` without overwriting existing dict
    subtrees; nested dicts are merged recursively so that summary.json and
    benchmarks.json can share the same namespace (e.g. both contributing a
    ``dataset`` section) without clobbering each other."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, Mapping):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        default=None,
        help=(
            "Summary JSON to stamp from; may be given multiple times. "
            "Defaults to outputs/summary.json and outputs/benchmarks.json."
        ),
    )
    parser.add_argument("--tex", type=Path, default=repo_root / "paper" / "main.tex")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path; defaults to --tex (in-place).",
    )
    args = parser.parse_args(argv)

    summary_paths: list[Path]
    if args.summary:
        summary_paths = list(args.summary)
    else:
        summary_paths = [
            repo_root / "outputs" / "summary.json",
            repo_root / "outputs" / "benchmarks.json",
        ]

    summary: dict = {}
    loaded: list[str] = []
    for path in summary_paths:
        if not path.exists():
            continue
        with path.open() as fh:
            _merge(summary, json.load(fh))
        loaded.append(str(path))

    tex_source = args.tex.read_text()
    stamped = stamp_tex(tex_source, summary)
    out_path = args.out or args.tex
    out_path.write_text(stamped)
    print(
        f"[build_paper] wrote {out_path} ({len(stamped):,} bytes, "
        f"summary sources={loaded}, top-level keys={list(summary)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
