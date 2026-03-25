"""Tests for atft.analysis.pair_correlation.

Three tests:
  1. test_poisson_r2_is_flat       — Poisson points → r₂ ≈ 1
  2. test_correlation_energy_poisson_near_zero — Poisson → E ≈ 0
  3. test_correlation_energy_gue_positive      — GUE theoretical r₂ → E > 0
"""
from __future__ import annotations

import numpy as np
import pytest

from atft.analysis.pair_correlation import (
    correlation_energy,
    pair_correlation_function,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _poisson_points(n: int = 2000, seed: int = 0) -> np.ndarray:
    """Generate n Poisson (uniform) points on [0, n]."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, float(n), n))


def _gue_theoretical_r2(s: np.ndarray) -> np.ndarray:
    """GUE pair correlation function: r₂(s) = 1 − (sin(πs)/(πs))²."""
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc_sq = np.where(
            np.abs(s) < 1e-12,
            1.0,
            (np.sin(np.pi * s) / (np.pi * s)) ** 2,
        )
    return 1.0 - sinc_sq


# ── tests ────────────────────────────────────────────────────────────────────

def test_poisson_r2_is_flat():
    """Poisson point set → r₂(s) ≈ 1 for all s > 0.

    We test over s ∈ [0.5, 3.5] to avoid end-of-histogram artefacts.
    Tolerance is generous (atol=0.2) because we need relatively few points
    to keep tests fast; the deviation shrinks with N.
    """
    pts = _poisson_points(n=2000, seed=7)
    s_c, r2 = pair_correlation_function(pts, n_bins=100, s_max=4.0)

    # Focus on the interior range
    interior = (s_c >= 0.5) & (s_c <= 3.5)
    r2_interior = r2[interior]

    assert len(r2_interior) > 0, "No bins in the interior range [0.5, 3.5]"
    # Mean close to 1
    assert abs(np.mean(r2_interior) - 1.0) < 0.2, (
        f"Poisson r₂ mean = {np.mean(r2_interior):.3f}, expected ≈ 1.0 (atol=0.2)"
    )
    # No wild outliers
    assert np.all(np.abs(r2_interior - 1.0) < 1.0), (
        "Some Poisson r₂ bins deviate by > 1.0 from baseline"
    )


def test_correlation_energy_poisson_near_zero():
    """Poisson points → correlation_energy ≈ 0 (small relative to GUE)."""
    pts = _poisson_points(n=2000, seed=13)
    s_c, r2 = pair_correlation_function(pts, n_bins=100, s_max=4.0)
    ds = s_c[1] - s_c[0]
    E = correlation_energy(r2, ds)

    # Energy must be finite and non-negative
    assert np.isfinite(E), f"Poisson correlation energy is not finite: {E}"
    assert E >= 0.0, f"Correlation energy must be non-negative, got {E}"

    # Compare to the GUE theoretical energy for context.
    # GUE theoretical r₂ = 1 - sinc²(πs) deviates substantially from 1.
    s_c_th = np.linspace(0.01, 4.0, 400)
    r2_gue_th = _gue_theoretical_r2(s_c_th)
    ds_th = s_c_th[1] - s_c_th[0]
    E_gue_th = correlation_energy(r2_gue_th, ds_th)

    # Poisson energy should be much smaller than GUE theoretical
    assert E < E_gue_th, (
        f"Poisson energy ({E:.4f}) should be < GUE theoretical energy ({E_gue_th:.4f})"
    )


def test_correlation_energy_gue_positive():
    """GUE theoretical r₂(s) = 1 − (sinπs/πs)² gives positive correlation energy.

    r₂ dips below 1 near s=0 (level repulsion), so ∫|r₂−1|² ds > 0.
    """
    s = np.linspace(0.01, 4.0, 500)
    r2 = _gue_theoretical_r2(s)
    ds = s[1] - s[0]
    E = correlation_energy(r2, ds)

    assert E > 0.0, (
        f"GUE theoretical correlation energy should be > 0, got {E}"
    )
    # Sanity: level-repulsion hole means the energy is non-trivial
    assert E > 0.05, (
        f"GUE theoretical correlation energy seems too small: {E:.4f}"
    )
