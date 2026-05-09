"""Shared test fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``src/`` importable without installing.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def small_data():
    from causal_pred.data.synthetic import simulate

    return simulate(n=800, rng=np.random.default_rng(42))


@pytest.fixture
def medium_data():
    from causal_pred.data.synthetic import simulate

    return simulate(n=3000, rng=np.random.default_rng(42))


@pytest.fixture
def small_gwas():
    from causal_pred.data.opengwas import OpenGWASClient, load_live_gwas

    cache_dir = Path(ROOT) / "data" / "mr_cache"
    return load_live_gwas(
        client=OpenGWASClient(token=None),
        cache_dir=cache_dir,
    )
