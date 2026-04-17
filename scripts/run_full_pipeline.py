"""Run the end-to-end causal-prediction pipeline at "full" scale.

Same CLI as ``python -m causal_pred.pipeline`` but with defaults tuned for
a meaningful run on a laptop (n=3000, mcmc_iter=2000, gam_samples=500,
gam_warmup=250).  Use this when you want to regenerate the canonical
artefacts in ``outputs/``.

Example
-------

    uv run python scripts/run_full_pipeline.py
    uv run python scripts/run_full_pipeline.py --n 5000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from causal_pred.pipeline import (
    DEFAULT_OUTPUT_DIR,
    run_pipeline,
    save_result,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_full_pipeline",
        description="Full-scale end-to-end causal-prediction pipeline run.",
    )
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--mcmc-iter", type=int, default=2000)
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--gam-samples", type=int, default=500)
    parser.add_argument("--gam-warmup", type=int, default=250)
    parser.add_argument("--target", default="T2D")
    parser.add_argument("--use-real-gwas", action="store_true", default=True)
    parser.add_argument("--no-real-gwas", dest="use_real_gwas", action="store_false")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args(argv)

    run_config = vars(args).copy()
    result = run_pipeline(
        n=args.n,
        use_real_gwas=args.use_real_gwas,
        mcmc_iter=args.mcmc_iter,
        mcmc_chains=args.mcmc_chains,
        gam_samples=args.gam_samples,
        gam_warmup=args.gam_warmup,
        target_node=args.target,
        seed=args.seed,
        verbose=args.verbose,
    )
    save_result(result, outdir=args.output_dir, run_config=run_config)
    print(f"\nFull-scale artefacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
