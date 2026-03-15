"""Shared fixtures for ATFT test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def simple_1d_points():
    """5 sorted points on a line with known gaps.

    Points: [1.0, 2.0, 4.0, 7.0, 11.0]
    Gaps:   [1.0, 2.0, 3.0, 4.0]
    Sorted gaps (desc): [4.0, 3.0, 2.0, 1.0]
    """
    return np.array([[1.0], [2.0], [4.0], [7.0], [11.0]], dtype=np.float64)


@pytest.fixture
def uniform_1d_points():
    """10 uniformly spaced points (all gaps = 1.0).

    Gini coefficient of gaps should be 0.0 (perfect equality).
    """
    return np.array([[float(i)] for i in range(10)], dtype=np.float64)
