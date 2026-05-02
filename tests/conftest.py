"""Shared fixtures for the baycomp_plotting test suite."""
from __future__ import annotations

import baycomp as bc
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def correlated_ttest_posterior(rng):
    base = rng.normal(0.85, 0.015, 10)
    acc_a = np.clip(base + rng.normal(0.0, 0.005, 10), 0, 1)
    acc_b = np.clip(base + rng.normal(0.015, 0.005, 10), 0, 1)
    return bc.CorrelatedTTest(acc_a, acc_b, rope=0.01, runs=1)


@pytest.fixture
def signed_rank_posterior(rng):
    mu = rng.uniform(0.65, 0.92, 25)
    mean_a = np.clip(mu + rng.normal(0.0, 0.005, 25), 0, 1)
    mean_b = np.clip(mu + rng.normal(0.012, 0.005, 25), 0, 1)
    return bc.SignedRankTest(mean_a, mean_b, rope=0.01, random_state=0)
