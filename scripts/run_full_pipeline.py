"""Run the end-to-end causal-prediction pipeline.

Single fixed path: cohort CSV (resolved via the local-then-bucket cache)
-> DAGSLAM -> structure MCMC -> save artefacts under ``outputs/``. There
are no command-line flags; defaults are the configuration. To override
hyperparameters, edit ``run_pipeline``'s defaults in
:mod:`causal_pred.pipeline`.

Example
-------

    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from causal_pred.pipeline import (
    DEFAULT_OUTPUT_DIR,
    run_pipeline,
    save_result,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv  # unused; this entry point takes no arguments
    result = run_pipeline()
    save_result(result, outdir=DEFAULT_OUTPUT_DIR, run_config={})
    print(f"\nArtefacts written to {DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
