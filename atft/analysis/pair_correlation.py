"""Pair correlation analysis for point sets.

Computes r₂(s) — the two-point correlation function — and derived statistics
used for ATFT novelty testing: does the sheaf Laplacian spectral sum S carry
information beyond what pair correlations alone encode?
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def pair_correlation_function(
    points: np.ndarray,
    n_bins: int = 100,
    s_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the pair correlation function r₂(s) for a 1D sorted point set.

    Computes the ratio of the observed spacing density to the Poisson
    (uniform) baseline.  After unfolding to mean spacing = 1, the
    Poisson baseline is p(s) = e^{-s}, so r₂(s) = observed_density(s) /
    e^{-s}.  In practice we use the all-pair distance histogram normalized
    by the expected count for Poisson points at the same density.

    Procedure
    ---------
    1. Unfold so that the mean nearest-neighbour spacing equals 1.
    2. Compute *all* pairwise distances (GPU-accelerated when torch is
       available and points fit in VRAM; falls back to CPU numpy).
    3. Bin into `n_bins` bins on [0, s_max].
    4. Normalize by the Poisson prediction: for N points spread over
       length L the expected number of pairs with separation in [s, s+ds]
       is N(N-1)/2 * (2/L) * ds  (two-sided, flattened to positive axis).

    Parameters
    ----------
    points : 1D sorted array of point positions (need not be mean-1 spaced).
    n_bins : number of histogram bins on [0, s_max].
    s_max  : upper limit of the spacing axis (in units of mean spacing).

    Returns
    -------
    s_centers : (n_bins,) array of bin centres.
    r2        : (n_bins,) array of r₂(s) values.
    """
    pts = np.sort(np.asarray(points, dtype=np.float64).ravel())
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 points to compute r₂(s).")

    # --- unfold to mean spacing = 1 -----------------------------------------
    mean_sp = float(np.mean(np.diff(pts)))
    if mean_sp <= 0:
        raise ValueError("Mean spacing must be positive.")
    pts_unf = pts / mean_sp          # now mean spacing ≈ 1
    L = pts_unf[-1] - pts_unf[0]    # total length after unfolding

    # --- pairwise distances --------------------------------------------------
    if _TORCH_AVAILABLE:
        t = torch.tensor(pts_unf, dtype=torch.float32)
        # all-pairs absolute difference; keep upper triangle only
        diffs = torch.abs(t.unsqueeze(1) - t.unsqueeze(0))
        # upper-triangle mask (i < j)
        mask = torch.ones(n, n, dtype=torch.bool).triu(diagonal=1)
        distances = diffs[mask].numpy()
    else:
        # CPU fallback — upper triangle only
        i_idx, j_idx = np.triu_indices(n, k=1)
        distances = np.abs(pts_unf[i_idx] - pts_unf[j_idx])

    # --- histogram -----------------------------------------------------------
    bins = np.linspace(0.0, s_max, n_bins + 1)
    counts, _ = np.histogram(distances, bins=bins)
    s_centers = 0.5 * (bins[:-1] + bins[1:])
    ds = bins[1] - bins[0]

    # Observed density (pairs per unit s, normalized by number of pairs)
    n_pairs = n * (n - 1) / 2
    obs_density = counts / (n_pairs * ds)

    # Poisson prediction for all-pair density: for uniform points on [0, L]
    # the pair density is 2/L for s in [0, L], 0 beyond.
    # With mean spacing = 1 we have L >> s_max in practice.
    poisson_density = np.where(s_centers <= L, 2.0 / L, 1e-30)

    # r₂ = observed / Poisson
    r2 = np.where(poisson_density > 0, obs_density / poisson_density, 0.0)

    return s_centers, r2


def correlation_energy(r2: np.ndarray, ds: float = 1.0) -> float:
    """Return ∫|r₂(s) − 1|² ds — total deviation from Poisson baseline.

    Parameters
    ----------
    r2 : pair correlation values on a uniform s-grid.
    ds : bin width (set to 1 for dimensionless energy).

    Returns
    -------
    energy : non-negative float.
    """
    return float(np.sum((r2 - 1.0) ** 2) * ds)


def predict_S_from_r2(
    r2_source: np.ndarray,
    r2_reference: np.ndarray,
    S_reference: float,
    ds: float = 1.0,
) -> float:
    """Predict S for a source from its pair-correlation energy.

    If S is entirely determined by pair correlations, the hypothesis is:

        S_source / S_ref ≈ f(E_source / E_ref)

    The simplest (linear) model:

        S_predicted = S_ref * (E_source / E_ref)

    where E = ∫|r₂(s) − 1|² ds.

    Parameters
    ----------
    r2_source    : r₂ values for the source whose S we want to predict.
    r2_reference : r₂ values for the reference source with known S.
    S_reference  : known spectral sum for the reference source.
    ds           : bin width used in both r₂ arrays.

    Returns
    -------
    S_predicted : float.
    """
    E_source = correlation_energy(r2_source, ds)
    E_ref = correlation_energy(r2_reference, ds)
    if E_ref == 0.0:
        # Reference is Poisson — cannot normalize; return S_ref as fallback
        return S_reference
    return S_reference * (E_source / E_ref)


def nearest_neighbour_distribution(
    points: np.ndarray,
    n_bins: int = 100,
    s_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute p(s) — nearest-neighbour spacing distribution.

    Parameters
    ----------
    points : 1D point array.
    n_bins : histogram bins on [0, s_max].
    s_max  : upper limit.

    Returns
    -------
    s_centers, p_s : arrays of shape (n_bins,).
    """
    pts = np.sort(np.asarray(points, dtype=np.float64).ravel())
    spacings = np.diff(pts)
    mean_sp = float(np.mean(spacings))
    spacings_norm = spacings / mean_sp   # unfold to mean = 1

    bins = np.linspace(0.0, s_max, n_bins + 1)
    counts, _ = np.histogram(spacings_norm, bins=bins, density=True)
    s_centers = 0.5 * (bins[:-1] + bins[1:])
    return s_centers, counts


def number_variance(
    points: np.ndarray,
    L_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the number variance Σ²(L) for a 1D point set.

    Σ²(L) = <n(L)²> − <n(L)>²

    where n(L) is the number of points in a randomly placed window of
    length L (in units of mean spacing).

    Uses a sliding window over the unfolded spectrum.

    Parameters
    ----------
    points   : 1D array of point positions.
    L_values : window lengths to probe (in mean-spacing units).
                Defaults to np.linspace(0.5, 10, 40).

    Returns
    -------
    L_values, sigma2 : arrays of shape (len(L_values),).
    """
    if L_values is None:
        L_values = np.linspace(0.5, 10.0, 40)

    pts = np.sort(np.asarray(points, dtype=np.float64).ravel())
    mean_sp = float(np.mean(np.diff(pts)))
    pts_unf = pts / mean_sp
    total_L = pts_unf[-1] - pts_unf[0]

    sigma2 = np.empty(len(L_values))
    for k, L in enumerate(L_values):
        if L >= total_L:
            sigma2[k] = np.nan
            continue
        # slide window in steps of 0.1 mean spacings
        step = 0.1
        starts = np.arange(pts_unf[0], pts_unf[-1] - L, step)
        if len(starts) < 2:
            sigma2[k] = np.nan
            continue
        counts = np.array([
            int(np.sum((pts_unf >= x) & (pts_unf < x + L)))
            for x in starts
        ])
        sigma2[k] = float(np.var(counts))

    return L_values, sigma2
