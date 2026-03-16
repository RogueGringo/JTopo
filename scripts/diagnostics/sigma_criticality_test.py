#!/usr/bin/env python
"""Sigma-criticality test: multiple metrics to detect sigma=1/2 signal.

Tests several normalized and ratio-based metrics that strip out the
trivial 1/p^sigma scaling to reveal the genuine sigma-dependent signal.

Metrics:
1. Raw spec_sum (for reference — monotonic for resonant, inverted-U for FE)
2. Normalized spec_sum: spec_sum / expected_scale(sigma)
3. Spectral gap ratio: 2nd eigenvalue / 1st non-zero eigenvalue
4. Kernel dimension (at fixed threshold)
5. Non-unitarity cost: ||UU†-I|| weighted spectral sum
6. Functional equation asymmetry: |spec_sum(sigma) - spec_sum(1-sigma)|
"""
import sys
import time
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def expected_generator_scale(primes, sigma):
    """Expected scaling factor: sum of (log(p)/p^sigma)^2."""
    return sum((np.log(p) / p**sigma)**2 for p in primes)


def test_sigma_criticality(zeros, K, sigma_grid, epsilon, label, mode="fe"):
    """Run comprehensive sigma-criticality test at one epsilon."""
    N = len(zeros)
    print(f"\n{'='*70}")
    print(f"  {label} (N={N}, K={K}, eps={epsilon}, mode={mode})")
    print(f"{'='*70}")

    primes = TransportMapBuilder(K=K, sigma=0.5).primes
    print(f"  Primes: {primes}")

    results = {}
    for sigma in sigma_grid:
        t0 = time.time()
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = SheafLaplacian(builder, zeros, transport_mode=mode)
        eigs = lap.smallest_eigenvalues(epsilon, m=20)
        elapsed = time.time() - t0

        spec_sum = float(np.sum(eigs))
        scale = expected_generator_scale(primes, sigma) if primes else 1.0
        normalized = spec_sum / scale if scale > 0 else 0.0
        min_eig = float(eigs[0])
        # Kernel count at fixed threshold
        kernel_count = int(np.sum(eigs < 1e-6))
        # Spectral gap: first non-near-zero eigenvalue
        nonzero = eigs[eigs > 1e-6]
        spec_gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0
        # Non-unitarity cost
        U_test = builder.transport_fe(1.0) if mode == "fe" else builder.transport_resonant(1.0)
        unitarity_dev = float(np.linalg.norm(U_test @ U_test.conj().T - np.eye(K)))

        results[sigma] = {
            "spec_sum": spec_sum,
            "normalized": normalized,
            "kernel_count": kernel_count,
            "spec_gap": spec_gap,
            "unitarity_dev": unitarity_dev,
            "scale": scale,
        }

    # Table 1: Raw and normalized metrics
    print(f"\n  {'sigma':>8} | {'raw_sum':>10} | {'scale':>10} | {'norm_sum':>10} | "
          f"{'kernel':>6} | {'spec_gap':>10} | {'||UU^H-I||':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")

    for sigma in sigma_grid:
        r = results[sigma]
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"  {sigma:8.3f} | {r['spec_sum']:10.4f} | {r['scale']:10.4f} | "
              f"{r['normalized']:10.4f} | {r['kernel_count']:6d} | "
              f"{r['spec_gap']:10.6f} | {r['unitarity_dev']:10.6f}{marker}")

    # Analysis
    raw_sums = np.array([results[s]["spec_sum"] for s in sigma_grid])
    norm_sums = np.array([results[s]["normalized"] for s in sigma_grid])
    kernels = np.array([results[s]["kernel_count"] for s in sigma_grid])
    gaps = np.array([results[s]["spec_gap"] for s in sigma_grid])

    print(f"\n  Peak analysis:")
    print(f"    Raw spec_sum:  min at sigma={sigma_grid[np.argmin(raw_sums)]:.3f}, "
          f"max at sigma={sigma_grid[np.argmax(raw_sums)]:.3f}")
    print(f"    Norm spec_sum: min at sigma={sigma_grid[np.argmin(norm_sums)]:.3f}, "
          f"max at sigma={sigma_grid[np.argmax(norm_sums)]:.3f}")
    if kernels.max() > kernels.min():
        print(f"    Kernel count:  max at sigma={sigma_grid[np.argmax(kernels)]:.3f}")
    else:
        print(f"    Kernel count:  constant at {kernels[0]}")
    print(f"    Spec gap:      min at sigma={sigma_grid[np.argmin(gaps)]:.3f}, "
          f"max at sigma={sigma_grid[np.argmax(gaps)]:.3f}")

    # Functional equation asymmetry
    if mode == "fe":
        print(f"\n  Functional equation asymmetry: |spec_sum(s) - spec_sum(1-s)|")
        print(f"  {'sigma':>8} | {'D(sigma)':>12}")
        print(f"  {'-'*8}-+-{'-'*12}")
        D_values = []
        D_sigmas = []
        for sigma in sigma_grid:
            s_mirror = round(1 - sigma, 3)
            if s_mirror in results:
                D = abs(results[sigma]["spec_sum"] - results[s_mirror]["spec_sum"])
                D_values.append(D)
                D_sigmas.append(sigma)
                marker = " <-- zero!" if D < 1e-8 else ""
                print(f"  {sigma:8.3f} | {D:12.6f}{marker}")
        if D_values:
            min_D_idx = int(np.argmin(D_values))
            print(f"  Minimum asymmetry at sigma = {D_sigmas[min_D_idx]:.3f}")

    sys.stdout.flush()
    return results


def main():
    print("ATFT Sigma-Criticality Test")
    print("Comprehensive multi-metric analysis of sigma-dependence\n")

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))

    sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.48,
                           0.50,
                           0.52, 0.55, 0.60, 0.65, 0.70, 0.75])

    cloud = source.generate(30)
    zeros_30 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]

    # Test 1: FE transport at eps=2.5
    test_sigma_criticality(zeros_30, K=6, sigma_grid=sigma_grid,
                           epsilon=2.5, label="FE K=6 eps=2.5", mode="fe")

    # Test 2: FE transport at eps=3.0 (more constraints)
    test_sigma_criticality(zeros_30, K=6, sigma_grid=sigma_grid,
                           epsilon=3.0, label="FE K=6 eps=3.0", mode="fe")

    # Test 3: Resonant transport (for comparison) at eps=2.5
    test_sigma_criticality(zeros_30, K=6, sigma_grid=sigma_grid,
                           epsilon=2.5, label="Resonant K=6 eps=2.5", mode="resonant")

    # Test 4: Compare zeta zeros vs RANDOM points (the crucial test)
    print(f"\n\n{'='*70}")
    print(f"  CONTROL TEST: Zeta zeros vs Random points")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)
    N = len(zeros_30)
    # Generate random points with same mean spacing
    mean_spacing = np.mean(np.diff(np.sort(zeros_30)))
    random_zeros = np.sort(rng.uniform(zeros_30.min(), zeros_30.max(), N))

    print(f"\n  Zeta zeros: mean_spacing = {mean_spacing:.4f}")
    print(f"  Random pts: mean_spacing = {np.mean(np.diff(random_zeros)):.4f}")

    results_zeta = test_sigma_criticality(
        zeros_30, K=6, sigma_grid=sigma_grid,
        epsilon=2.5, label="ZETA ZEROS K=6 eps=2.5", mode="fe")

    results_random = test_sigma_criticality(
        random_zeros, K=6, sigma_grid=sigma_grid,
        epsilon=2.5, label="RANDOM POINTS K=6 eps=2.5", mode="fe")

    # Compare
    print(f"\n\n  COMPARISON: normalized spec_sum")
    print(f"  {'sigma':>8} | {'zeta norm':>12} | {'random norm':>12} | {'ratio':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for sigma in sigma_grid:
        nz = results_zeta[sigma]["normalized"]
        nr = results_random[sigma]["normalized"]
        ratio = nz / nr if nr > 0 else float('inf')
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"  {sigma:8.3f} | {nz:12.4f} | {nr:12.4f} | {ratio:10.4f}{marker}")


if __name__ == "__main__":
    main()
