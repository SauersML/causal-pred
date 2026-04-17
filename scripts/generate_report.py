"""Render a self-contained HTML report from ``outputs/``.

Reads:

* ``outputs/summary.json`` -- pipeline validation metrics and stage status.
* ``outputs/run_config.json`` -- run configuration dict.
* ``outputs/plots/*.png`` -- per-artefact plots written by the pipeline.

Writes:

* ``outputs/report.html`` -- single self-contained HTML document with
  inlined base64 PNGs, a metrics table, stage timings, and run config.

Example
-------

    uv run python scripts/generate_report.py
    uv run python scripts/generate_report.py --outputs-dir some/dir
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _inline_png(path: str) -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, dict):
        return html.escape(json.dumps(v, indent=None))
    return html.escape(str(v))


def _kv_rows(d: Dict[str, Any]) -> str:
    out: List[str] = []
    for k, v in d.items():
        out.append(f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>")
    return "\n".join(out)


def _image_block(name: str, src: str) -> str:
    return (
        f"<figure class='plot'>"
        f"<figcaption>{html.escape(name)}</figcaption>"
        f"<img alt='{html.escape(name)}' src='{src}'/>"
        f"</figure>"
    )


def build_report(outputs_dir: str) -> str:
    summary = _load_json(os.path.join(outputs_dir, "summary.json"))
    run_config = _load_json(os.path.join(outputs_dir, "run_config.json"))

    plots_dir = os.path.join(outputs_dir, "plots")
    plot_items: List[str] = []
    if os.path.isdir(plots_dir):
        for fname in sorted(os.listdir(plots_dir)):
            if fname.lower().endswith(".png"):
                src = _inline_png(os.path.join(plots_dir, fname))
                plot_items.append(_image_block(fname, src))

    validation = summary.get("validation", {}) or {}
    stage_status = summary.get("stage_status", {}) or {}
    timings = summary.get("timings", {}) or {}
    data_summary = summary.get("data_summary", {}) or {}
    parent_sets = summary.get("parent_sets", []) or []

    metrics_flat: Dict[str, Any] = {
        "target_node": summary.get("target_node", "n/a"),
        "n": data_summary.get("n", "n/a"),
        "event_rate": data_summary.get("event_rate", "n/a"),
        "Nagelkerke R^2 (t=10y)": validation.get("nagelkerke_r2_at_10y", "n/a"),
    }
    calib = validation.get("calibration_at_10y", {}) or {}
    if isinstance(calib, dict):
        metrics_flat["ECE (t=10y)"] = calib.get("ece", "n/a")
        metrics_flat["Brier (t=10y, point)"] = calib.get("brier", "n/a")
    recov = validation.get("known_edge_recovery", {}) or {}
    if isinstance(recov, dict):
        metrics_flat["Edge recovery AUROC"] = recov.get("auroc", "n/a")
        metrics_flat["Edge recovery AUPRC"] = recov.get("auprc", "n/a")
    ibs = validation.get("brier", {})
    if isinstance(ibs, dict):
        metrics_flat["Integrated Brier Score"] = ibs.get("ibs", "n/a")

    tda = validation.get("time_dependent_auc", {}) or {}
    if isinstance(tda, dict) and "auc" in tda:
        eval_times = tda.get("eval_times", [])
        auc_values = tda.get("auc", [])
        for tt, aa in zip(eval_times, auc_values):
            metrics_flat[f"AUC(t={_fmt(tt)})"] = aa

    parent_set_rows = []
    for ps in parent_sets:
        names = ps.get("parent_names", [])
        w = ps.get("weight", 0.0)
        parent_set_rows.append(
            f"<tr><td>{html.escape(', '.join(names)) or '&empty;'}</td>"
            f"<td>{_fmt(w)}</td></tr>"
        )

    style = """
    body { font-family: -apple-system, system-ui, sans-serif;
           max-width: 1100px; margin: 1.5rem auto; padding: 0 1rem;
           color: #222; line-height: 1.45; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 0.3rem; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #ccc;
         padding-bottom: 0.2rem; }
    table { border-collapse: collapse; margin: 0.5rem 0 1rem 0; }
    th, td { padding: 4px 10px; text-align: left; vertical-align: top;
             border-bottom: 1px solid #eee; font-size: 0.92rem; }
    th { background: #f6f6f6; font-weight: 600; }
    .plot { display: inline-block; margin: 0.5rem; vertical-align: top; }
    .plot img { max-width: 480px; border: 1px solid #ccc;
                 background: #fff; }
    figcaption { font-size: 0.85rem; color: #555; margin-bottom: 4px; }
    code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
    pre { background: #f6f6f6; padding: 0.6rem; border-radius: 4px;
          overflow-x: auto; font-size: 0.8rem; }
    .status-ok { color: #1a7f37; font-weight: 600; }
    .status-placeholder { color: #b08800; }
    .status-error { color: #b22; }
    """

    def _status_class(s: str) -> str:
        if s == "ok":
            return "status-ok"
        if s.startswith("placeholder"):
            return "status-placeholder"
        return "status-error"

    stage_rows = "\n".join(
        f"<tr><th>{html.escape(k)}</th>"
        f"<td class='{_status_class(str(v))}'>{html.escape(str(v))}</td>"
        f"<td>{_fmt(timings.get(k))}</td></tr>"
        for k, v in stage_status.items()
    )

    plots_html = (
        "\n".join(plot_items) if plot_items else ("<p><em>No plots found.</em></p>")
    )
    parent_sets_html = (
        "<table><tr><th>parents</th><th>weight</th></tr>"
        + "\n".join(parent_set_rows)
        + "</table>"
        if parent_set_rows
        else "<p><em>No parent sets recorded.</em></p>"
    )

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'/>
<title>causal-pred pipeline report</title>
<style>{style}</style>
</head>
<body>
<h1>causal-pred pipeline report</h1>
<p>Target node: <code>{html.escape(str(summary.get("target_node", "?")))}</code>
   | n = <code>{_fmt(data_summary.get("n"))}</code>
   | event rate = <code>{_fmt(data_summary.get("event_rate"))}</code></p>

<h2>Metrics</h2>
<table>{_kv_rows(metrics_flat)}</table>

<h2>Stage status and timings (s)</h2>
<table>
<tr><th>stage</th><th>status</th><th>time (s)</th></tr>
{stage_rows}
</table>

<h2>Top parent sets (BMA)</h2>
{parent_sets_html}

<h2>Plots</h2>
{plots_html}

<h2>Run configuration</h2>
<pre>{html.escape(json.dumps(run_config, indent=2, sort_keys=True))}</pre>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_report",
        description="Render outputs/report.html from the pipeline artefacts.",
    )
    default_outputs = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs")
    )
    parser.add_argument("--outputs-dir", default=default_outputs)
    parser.add_argument(
        "--report-path", default=None, help="defaults to <outputs-dir>/report.html"
    )
    args = parser.parse_args(argv)

    report_path = args.report_path or os.path.join(args.outputs_dir, "report.html")
    doc = build_report(args.outputs_dir)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write(doc)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
