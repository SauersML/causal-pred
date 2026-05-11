"""Run the single end-to-end causal-prediction pipeline."""

from __future__ import annotations

import os
import sys

# Force line-buffered stdout/stderr BEFORE importing the pipeline. The AoU
# Jupyter harness pipes stdout to a non-TTY, which makes CPython default to
# block-buffering -- and that produces minutes of silence during long
# stages (DAGSLAM, MCMC, BMA) where individual logger.info emits sit in
# the OS pipe buffer until a 4-8 KB threshold flushes them out. Setting
# this once at process start beats sprinkling sys.stdout.flush() everywhere.
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    # Python < 3.7; fall back to raw fd writes from the logging side.
    pass

from causal_pred.pipeline import main  # noqa: E402  -- must follow buffering setup


if __name__ == "__main__":
    raise SystemExit(main())
