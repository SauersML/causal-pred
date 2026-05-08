"""Run the single real AoU microarray-PRS pipeline.

Single fixed path: cohort CSV + cached gnomon-scored PRS matrix -> DAGSLAM
-> structure MCMC -> save artefacts under ``outputs/``. Run
``scripts/prepare_aou_prs.py`` first; this entry point deliberately fails if
``data/aou_prs_panel.csv.gz`` is absent.

Example
-------

    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from causal_pred.pipeline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRS_PATH,
    run_pipeline,
    save_result,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv  # unused; this entry point takes no arguments
    result = run_pipeline()
    save_result(
        result,
        outdir=DEFAULT_OUTPUT_DIR,
        run_config={
            "mode": "aou_microarray_prs",
            "prs_path": DEFAULT_PRS_PATH,
            "n_prs_nodes": 8,
            "prs_max_missing": 0.2,
        },
    )
    print(f"\nArtefacts written to {DEFAULT_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
