#!/usr/bin/env python
"""RH Control Test: Zeta zeros vs Random points.

The symmetrized spectral sum S_avg(sigma) = [spec_sum(sigma) + spec_sum(1-sigma)] / 2
peaks at sigma=1/2 for zeta zeros. Does it also peak for random points?

If YES: the peak is an artifact of the functional equation generator (tautological)
If NO: the peak is a genuine signal of the zeros' arithmetic structure
"""
import sys
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def compute_sigma_profile(zeros, K, sigma_grid, epsilon, mode="fe"):
    """Compute spec_sum profile and symmetrized version."""
    results = {}
    for sigma in sigma_grid:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = SheafLaplacian(builder, zeros, transport_mode=mode)
        eigs = lap.smallest_eigenvalues(epsilon, m=20)
        results[sigma] = float(np.sum(eigs))
    return results


def analyze_profile(name, sigma_grid, results):
    """Print and analyze a sigma profile."""
    print(f"\n  {name}:")
    print(f"  {'sigma':>8} | {'spec_sum':>10} | {'sym_avg':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")

    sym_avgs = []
    sym_sigmas = []
    for sigma in sigma_grid:
        ss = results[sigma]
        s_mirror = round(1.0 - sigma, 3)
        if s_mirror in results:
            sym_avg = (ss + results[s_mirror]) / 2
        else:
            sym_avg = ss
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"  {sigma:8.3f} | {ss:10.4f} | {sym_avg:10.4f}{marker}")
        if sigma <= 0.5:
            sym_avgs.append(sym_avg)
            sym_sigmas.append(sigma)

    sym_avgs = np.array(sym_avgs)
    sym_sigmas = np.array(sym_sigmas)
    peak_idx = int(np.argmax(sym_avgs))
    peak_sigma = sym_sigmas[peak_idx]

    # Compute "peakiness": how much does sigma=0.5 stand out?
    if len(sym_avgs) > 1:
        half_idx = len(sym_avgs) - 1  # sigma=0.5 is last
        relative_peak = sym_avgs[half_idx] / sym_avgs[0] if sym_avgs[0] > 0 else 0
        contrast = (sym_avgs[half_idx] - sym_avgs[0]) / sym_avgs[half_idx] if sym_avgs[half_idx] > 0 else 0
    else:
        relative_peak = 1.0
        contrast = 0.0

    print(f"  Peak sym_avg at sigma = {peak_sigma:.3f} (value = {sym_avgs[peak_idx]:.4f})")
    print(f"  Contrast (S(0.5)-S(0.25))/S(0.5) = {contrast:.4f}")
    return peak_sigma, contrast


def main():
    print("="*70)
    print("  RH CONTROL TEST: Zeta Zeros vs Random Points")
    print("  Normalized FE transport, symmetrized spectral sum")
    print("="*70)

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))

    sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                           0.55, 0.60, 0.65, 0.70, 0.75])

    K = 6
    epsilon = 2.5

    # Load zeta zeros
    cloud = source.generate(30)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    N = len(zeta_zeros)
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))

    print(f"\nN={N}, K={K}, eps={epsilon}")
    print(f"Zeta zeros: range [{zeta_zeros.min():.2f}, {zeta_zeros.max():.2f}], "
          f"mean_spacing={mean_spacing:.4f}")

    # Compute zeta zero profile
    print(f"\nComputing zeta zero profile...")
    sys.stdout.flush()
    zeta_results = compute_sigma_profile(zeta_zeros, K, sigma_grid, epsilon, mode="fe")
    zeta_peak, zeta_contrast = analyze_profile("ZETA ZEROS", sigma_grid, zeta_results)

    # Test multiple random point sets
    print(f"\nComputing random point profiles...")
    sys.stdout.flush()

    rng = np.random.default_rng(42)
    n_random_trials = 5
    random_peaks = []
    random_contrasts = []

    for trial in range(n_random_trials):
        # Random uniform points with same range and count
        random_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), N))
        results = compute_sigma_profile(random_pts, K, sigma_grid, epsilon, mode="fe")
        peak, contrast = analyze_profile(f"RANDOM trial {trial+1}", sigma_grid, results)
        random_peaks.append(peak)
        random_contrasts.append(contrast)

    # Also test GUE-spaced points (same spacing statistics, no arithmetic structure)
    print(f"\nComputing GUE-like point profiles...")
    sys.stdout.flush()
    gue_peaks = []
    gue_contrasts = []

    for trial in range(n_random_trials):
        # GUE-spaced: Wigner surmise for nearest-neighbor spacing
        # p(s) = (pi/2)*s*exp(-pi*s^2/4)
        spacings = []
        for _ in range(N - 1):
            # Inverse CDF sampling of Wigner surmise
            u = rng.random()
            s = np.sqrt(-4 / np.pi * np.log(1 - u))
            spacings.append(s * mean_spacing)
        gue_pts = np.cumsum([zeta_zeros.min()] + spacings)
        results = compute_sigma_profile(gue_pts, K, sigma_grid, epsilon, mode="fe")
        peak, contrast = analyze_profile(f"GUE trial {trial+1}", sigma_grid, results)
        gue_peaks.append(peak)
        gue_contrasts.append(contrast)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Source':>20} | {'peak sigma':>12} | {'contrast':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Zeta zeros':>20} | {zeta_peak:12.3f} | {zeta_contrast:12.4f}")
    for i, (p, c) in enumerate(zip(random_peaks, random_contrasts)):
        print(f"  {f'Random {i+1}':>20} | {p:12.3f} | {c:12.4f}")
    for i, (p, c) in enumerate(zip(gue_peaks, gue_contrasts)):
        print(f"  {f'GUE {i+1}':>20} | {p:12.3f} | {c:12.4f}")

    # Final verdict
    zeta_at_half = zeta_results[0.50]
    random_avg_contrast = np.mean(random_contrasts)
    gue_avg_contrast = np.mean(gue_contrasts)

    print(f"\n  Average contrast:")
    print(f"    Zeta zeros:  {zeta_contrast:.4f}")
    print(f"    Random:      {random_avg_contrast:.4f}")
    print(f"    GUE:         {gue_avg_contrast:.4f}")

    if zeta_contrast > random_avg_contrast * 1.5:
        print(f"\n  ** ZETA ZEROS SHOW STRONGER CRITICAL LINE SIGNAL **")
        print(f"  Contrast ratio (zeta/random): {zeta_contrast/random_avg_contrast:.2f}x")
    else:
        print(f"\n  Signal strength is comparable across point types.")
        print(f"  The peak at sigma=0.5 may be a geometric artifact of the FE generator.")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
