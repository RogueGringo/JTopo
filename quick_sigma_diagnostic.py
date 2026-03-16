#!/usr/bin/env python
"""Quick diagnostic: spectral sum vs sigma at fixed epsilon values.

Instead of the full epsilon sweep (slow), compute the 20 smallest
eigenvalues of the sheaf Laplacian at a few key epsilon values and
compare across sigma. The spec_sum metric isn't capped by m.
"""
import sys
import time
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def run_quick_diagnostic(zeros, K, sigma_grid, eps_values, label):
    """Compute spectral sums at fixed epsilon for each sigma."""
    N = len(zeros)
    print(f"\n{'='*70}")
    print(f"  {label} (N={N}, K={K}, dim={N*K*K})")
    print(f"{'='*70}")

    builder0 = TransportMapBuilder(K=K, sigma=0.5)
    print(f"  Primes: {builder0.primes}")
    print(f"  log(primes): {[f'{np.log(p):.4f}' for p in builder0.primes]}")

    for eps in eps_values:
        print(f"\n  --- epsilon = {eps:.2f} ---")
        print(f"  {'sigma':>8} | {'spec_sum':>12} | {'min_eig':>12} | {'2nd_eig':>12} | {'spec_gap':>12}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        spec_sums = []
        min_eigs = []
        spec_gaps = []

        for sigma in sigma_grid:
            t0 = time.time()
            builder = TransportMapBuilder(K=K, sigma=sigma)
            lap = SheafLaplacian(builder, zeros, transport_mode="resonant")
            eigs = lap.smallest_eigenvalues(eps, m=20)
            elapsed = time.time() - t0

            spec_sum = float(np.sum(eigs))
            min_eig = float(eigs[0])
            # Find first non-kernel eigenvalue
            nonzero = eigs[eigs > 1e-8]
            spec_gap = float(nonzero[0]) if len(nonzero) > 0 else 0.0

            spec_sums.append(spec_sum)
            min_eigs.append(min_eig)
            spec_gaps.append(spec_gap)

            marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
            print(f"  {sigma:8.3f} | {spec_sum:12.6f} | {min_eig:12.2e} | "
                  f"{eigs[1] if len(eigs) > 1 else 0:12.2e} | "
                  f"{spec_gap:12.6f}{marker}")
            sys.stdout.flush()

        spec_sums = np.array(spec_sums)
        min_eigs = np.array(min_eigs)
        spec_gaps = np.array(spec_gaps)

        # Analysis
        min_spec_idx = int(np.argmin(spec_sums))
        min_gap_idx = int(np.argmin(spec_gaps))
        print(f"\n  Min spec_sum at sigma = {sigma_grid[min_spec_idx]:.3f} "
              f"(value = {spec_sums[min_spec_idx]:.6f})")
        print(f"  Min spec_gap at sigma = {sigma_grid[min_gap_idx]:.3f} "
              f"(value = {spec_gaps[min_gap_idx]:.6f})")

        # Variation
        spec_range = spec_sums.max() - spec_sums.min()
        if spec_range > 1e-6:
            print(f"  Spec_sum range: {spec_range:.6f} >> SIGMA-DEPENDENT!")
        else:
            print(f"  Spec_sum range: {spec_range:.2e} (no variation)")

    sys.stdout.flush()


def main():
    print("ATFT Quick Sigma Diagnostic: Spectral Sum at Fixed Epsilon")
    print("(Resonant dependency-graph transport)\n")

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))

    sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.48,
                           0.50,
                           0.52, 0.55, 0.60, 0.65, 0.70, 0.75])

    # Scale 1: N=30, K=6 (very fast)
    cloud = source.generate(30)
    zeros_30 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_quick_diagnostic(zeros_30, K=6, sigma_grid=sigma_grid,
                         eps_values=[1.0, 1.5, 2.0, 2.5],
                         label="FAST: N=30, K=6")

    # Scale 2: N=30, K=10 (more primes)
    run_quick_diagnostic(zeros_30, K=10, sigma_grid=sigma_grid,
                         eps_values=[1.0, 1.5, 2.0],
                         label="FAST: N=30, K=10")

    # Scale 3: N=50, K=6
    cloud = source.generate(50)
    zeros_50 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_quick_diagnostic(zeros_50, K=6, sigma_grid=sigma_grid,
                         eps_values=[1.5, 2.0],
                         label="MED: N=50, K=6")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
