#!/usr/bin/env python
"""ATFT Phase 2: Sigma-sweep with resonant (curved) transport.

Now that the flat connection obstruction is broken via dependency-graph
transport (each edge binds to its resonant prime), the sheaf Laplacian
spectrum genuinely depends on sigma.

This script sweeps sigma across the critical strip and tests the
Sigma-Criticality Hypothesis: does beta_0^F(epsilon) peak at sigma=1/2?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.sheaf_ph import SheafPH
from atft.topology.transport_maps import TransportMapBuilder


def divider(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")
    sys.stdout.flush()


def run_sigma_sweep(zeros, K, sigma_grid, eps_grid, label):
    """Run the full sigma sweep with resonant transport."""
    divider(f"{label} (N={len(zeros)}, K={K})")

    # Show prime landscape
    builder = TransportMapBuilder(K=K, sigma=0.5)
    print(f"Primes up to K={K}: {builder.primes}")
    print(f"log(primes): {[f'{np.log(p):.4f}' for p in builder.primes]}")
    print()

    n_sigma = len(sigma_grid)
    n_eps = len(eps_grid)

    # Compute sheaf Betti curves for each sigma
    heatmap = np.zeros((n_sigma, n_eps), dtype=np.int64)
    spectral_sums = np.zeros((n_sigma, n_eps), dtype=np.float64)

    for s_idx, sigma in enumerate(sigma_grid):
        t0 = time.time()
        builder = TransportMapBuilder(K=K, sigma=sigma)
        ph = SheafPH(builder, zeros, transport_mode="resonant")
        curve = ph.sweep(eps_grid, m=20)
        heatmap[s_idx, :] = curve.kernel_dimensions
        spectral_sums[s_idx, :] = np.sum(curve.smallest_eigenvalues, axis=1)
        elapsed = time.time() - t0

        # Summary stats
        max_kernel = int(np.max(curve.kernel_dimensions[1:]))  # skip eps=0
        integral = float(np.sum(curve.kernel_dimensions[1:]))
        marker = " <-- sigma=1/2" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"  sigma={sigma:.3f}: max_kernel={max_kernel:4d}, "
              f"integral={integral:10.0f}, "
              f"time={elapsed:.1f}s{marker}")
        sys.stdout.flush()

    print()

    # Find peak sigma
    # Metric 1: max kernel dimension across epsilon
    max_per_sigma = np.max(heatmap[:, 1:], axis=1)
    peak_idx_max = int(np.argmax(max_per_sigma))
    peak_sigma_max = float(sigma_grid[peak_idx_max])

    # Metric 2: integral of kernel dimension curve
    integral_per_sigma = np.sum(heatmap[:, 1:], axis=1)
    peak_idx_int = int(np.argmax(integral_per_sigma))
    peak_sigma_int = float(sigma_grid[peak_idx_int])

    # Metric 3: minimum total spectral sum (most global sections)
    spec_sum_per_sigma = np.sum(spectral_sums[:, 1:], axis=1)
    peak_idx_spec = int(np.argmin(spec_sum_per_sigma))
    peak_sigma_spec = float(sigma_grid[peak_idx_spec])

    print(f"Results:")
    print(f"  Peak max_kernel at sigma = {peak_sigma_max:.3f} (value = {max_per_sigma[peak_idx_max]})")
    print(f"  Peak integral   at sigma = {peak_sigma_int:.3f} (value = {integral_per_sigma[peak_idx_int]})")
    print(f"  Min spectral_sum at sigma = {peak_sigma_spec:.3f} (value = {spec_sum_per_sigma[peak_idx_spec]:.2f})")

    # Check if any metric peaks uniquely at sigma=1/2
    sigma_half_idx = int(np.argmin(np.abs(sigma_grid - 0.5)))
    print(f"\n  Values at sigma=0.5 (index {sigma_half_idx}):")
    print(f"    max_kernel = {max_per_sigma[sigma_half_idx]}")
    print(f"    integral   = {integral_per_sigma[sigma_half_idx]}")
    print(f"    spec_sum   = {spec_sum_per_sigma[sigma_half_idx]:.2f}")

    # Variation check
    kernel_range = int(max_per_sigma.max() - max_per_sigma.min())
    integral_range = int(integral_per_sigma.max() - integral_per_sigma.min())
    print(f"\n  Sigma-dependence check:")
    print(f"    max_kernel range: {kernel_range}")
    print(f"    integral range:   {integral_range}")
    if kernel_range > 0 or integral_range > 0:
        print(f"    >> SIGMA-DEPENDENT! The resonant connection works!")
    else:
        print(f"    >> No sigma variation detected.")

    # Detailed table
    print(f"\n  {'sigma':>8} | {'max_kernel':>10} | {'integral':>10} | {'spec_sum':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for s_idx, sigma in enumerate(sigma_grid):
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"  {sigma:8.3f} | {max_per_sigma[s_idx]:10d} | "
              f"{integral_per_sigma[s_idx]:10d} | "
              f"{spec_sum_per_sigma[s_idx]:12.2f}{marker}")

    sys.stdout.flush()
    return heatmap


def main():
    print("""
    +==================================================================+
    |  ATFT Phase 2: Resonant Transport Sigma-Sweep                     |
    |                                                                    |
    |  Dependency-graph transport: each edge binds to the prime whose    |
    |  log(p) best matches the gap. Non-commuting generators create     |
    |  genuine curvature. The flat connection is BROKEN.                 |
    |                                                                    |
    |  Testing: does sheaf Betti number peak at sigma = 1/2?            |
    +==================================================================+
    """)
    sys.stdout.flush()

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))

    sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.48,
                           0.50,
                           0.52, 0.55, 0.60, 0.65, 0.70, 0.75])

    eps_grid = np.linspace(0, 3.0, 50)

    # ================================================================
    # Scale 1: N=50, K=6 (quick test)
    # ================================================================
    cloud = source.generate(50)
    zeros_50 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_sigma_sweep(zeros_50, K=6, sigma_grid=sigma_grid,
                    eps_grid=eps_grid, label="SCALE 1: Quick test")

    # ================================================================
    # Scale 2: N=100, K=6
    # ================================================================
    cloud = source.generate(100)
    zeros_100 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_sigma_sweep(zeros_100, K=6, sigma_grid=sigma_grid,
                    eps_grid=eps_grid, label="SCALE 2: N=100")

    # ================================================================
    # Scale 3: N=100, K=10 (more primes: 2,3,5,7)
    # ================================================================
    run_sigma_sweep(zeros_100, K=10, sigma_grid=sigma_grid,
                    eps_grid=eps_grid, label="SCALE 3: K=10")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
