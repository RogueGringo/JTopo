#!/usr/bin/env python
"""ATFT Phase 2: Riemann Hypothesis test via twisted spectral holonomy.

KEY INSIGHT: The twisted Laplacian L_omega on a graph detects holonomy —
non-trivial phase accumulation around cycles. On a tree/path, there are
NO cycles, so L_omega always has kernel dim = #components regardless of omega.

For the sigma signal to appear, we need epsilon LARGE ENOUGH that the
Vietoris-Rips graph has CYCLES (triangles, quadrilaterals). For unfolded
zeros with mean spacing ~1, this requires eps >= 2.0.

At sufficiently large epsilon, the spectrum of L_omega depends on omega,
which depends on sigma. The question: does resonance peak at sigma=1/2?
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from atft.experiments.phase2a_abelian import Phase2aAbelian
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder


def divider(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")
    sys.stdout.flush()


def analyze_twisted_spectrum(zeros, K, sigma, epsilon):
    """For a given sigma and epsilon, compute detailed twisted spectral data."""
    N = len(zeros)
    builder = TransportMapBuilder(K=K, sigma=sigma)
    eigenvalues_A = builder.eigenvalues()

    # Collect distinct nonzero frequencies with multiplicities
    freq_mult = {}  # omega -> multiplicity
    for k in range(K):
        for l in range(K):
            omega = eigenvalues_A[k] - eigenvalues_A[l]
            if abs(omega) > 1e-12:
                key = round(omega, 10)
                freq_mult[key] = freq_mult.get(key, 0) + 1

    # For each distinct frequency, compute spectral data
    total_kernel = 0
    weighted_gap = 0.0  # sum of (multiplicity * min_eigenvalue)
    spectral_sum = 0.0  # sum of all eigenvalues across all twisted blocks

    for omega, mult in sorted(freq_mult.items()):
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega, epsilon)
        eigs = np.linalg.eigvalsh(L)
        n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
        min_eig = float(eigs[0])
        total_kernel += n_kernel * mult
        weighted_gap += mult * min_eig
        spectral_sum += mult * float(np.sum(eigs))

    return {
        "total_kernel": total_kernel,
        "weighted_gap": weighted_gap,
        "spectral_sum": spectral_sum,
        "n_distinct_freq": len(freq_mult),
    }


def run_holonomy_analysis(zeros, K, sigma_grid, eps_values, label):
    """Analyze twisted holonomy at cycle-creating epsilon values."""
    divider(f"{label} (N={len(zeros)}, K={K})")

    N = len(zeros)

    # First, characterize the graph topology at each epsilon
    print("Graph topology per epsilon:")
    for eps in eps_values:
        n_edges = 0
        for i in range(N):
            for j in range(i + 1, N):
                if zeros[j] - zeros[i] > eps:
                    break
                n_edges += 1
        n_cycles = n_edges - (N - 1)  # first Betti number (approx)
        avg_degree = 2 * n_edges / N
        print(f"  eps={eps:.1f}: {n_edges} edges, ~{max(0,n_cycles)} cycles, "
              f"avg_degree={avg_degree:.1f}")
    print()

    for eps in eps_values:
        print(f"--- epsilon = {eps:.1f} ---")

        # Detailed analysis for 3 representative sigma values
        for s_detail in [0.3, 0.5, 0.7]:
            builder = TransportMapBuilder(K=K, sigma=s_detail)
            eig_A = builder.eigenvalues()

            # Pick the dominant nontrivial frequency
            freqs = {}
            for k in range(K):
                for l in range(K):
                    om = eig_A[k] - eig_A[l]
                    if abs(om) > 1e-12:
                        key = round(abs(om), 8)
                        freqs[key] = freqs.get(key, 0) + 1

            print(f"  sigma={s_detail}: {len(freqs)} distinct |omega| values, "
                  f"largest={max(freqs.keys()):.4f}")

            # Show spectrum of L_omega at largest nontrivial frequency
            omega_max = max(freqs.keys())
            L = Phase2aAbelian._build_twisted_laplacian(zeros, omega_max, eps)
            eigs = np.linalg.eigvalsh(L)
            n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
            print(f"    L_omega(omega={omega_max:.4f}): kernel={n_kernel}, "
                  f"min_eig={eigs[0]:.6f}, 2nd={eigs[1]:.6f}, "
                  f"3rd={eigs[2]:.6f}")

        # Full sigma sweep at this epsilon
        print(f"\n  Sigma sweep at eps={eps:.1f}:")
        print(f"  {'sigma':>8} | {'total_kernel':>12} | {'weighted_gap':>14} | "
              f"{'spectral_sum':>14}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}")

        all_kernels = []
        all_gaps = []
        all_sums = []

        for sigma in sigma_grid:
            data = analyze_twisted_spectrum(zeros, K, sigma, eps)
            all_kernels.append(data["total_kernel"])
            all_gaps.append(data["weighted_gap"])
            all_sums.append(data["spectral_sum"])
            marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
            print(f"  {sigma:8.3f} | {data['total_kernel']:12d} | "
                  f"{data['weighted_gap']:14.6f} | "
                  f"{data['spectral_sum']:14.2f}{marker}")

        all_kernels = np.array(all_kernels)
        all_gaps = np.array(all_gaps)
        all_sums = np.array(all_sums)

        # Check for sigma-dependence
        kernel_range = all_kernels.max() - all_kernels.min()
        gap_range = all_gaps.max() - all_gaps.min()
        sum_range = all_sums.max() - all_sums.min()
        print(f"\n  Variation: kernel_range={kernel_range}, "
              f"gap_range={gap_range:.6f}, sum_range={sum_range:.4f}")

        if kernel_range > 0 or gap_range > 1e-8:
            peak_k = sigma_grid[np.argmax(all_kernels)]
            min_g = sigma_grid[np.argmin(all_gaps)]
            print(f"  >> SIGMA-DEPENDENT! Peak kernel at sigma={peak_k:.3f}, "
                  f"min gap at sigma={min_g:.3f}")
        else:
            print(f"  >> No sigma dependence detected at this epsilon")

        print()
        sys.stdout.flush()


def main():
    print("""
    +==================================================================+
    |  ATFT Phase 2: Twisted Holonomy Analysis for RH                  |
    |                                                                  |
    |  The twisted Laplacian L_omega detects holonomy: non-trivial     |
    |  phase winding around graph cycles. Holonomy depends on omega    |
    |  which depends on sigma. Sigma-dependence requires CYCLES.       |
    |                                                                  |
    |  Testing at eps >= 2.0 where Rips graph has triangles.           |
    +==================================================================+
    """)
    sys.stdout.flush()

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))

    sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.48,
                           0.50,
                           0.52, 0.55, 0.60, 0.65, 0.70, 0.75])

    # Epsilon values that create cycles
    eps_values = [2.0, 2.5, 3.0, 4.0, 5.0]

    # ================================================================
    # Scale 1: N=100, K=6
    # ================================================================
    cloud = source.generate(100)
    zeros_100 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_holonomy_analysis(zeros_100, K=6, sigma_grid=sigma_grid,
                          eps_values=eps_values, label="SCALE 1")

    # ================================================================
    # Scale 2: N=300, K=8
    # ================================================================
    cloud = source.generate(300)
    zeros_300 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    run_holonomy_analysis(zeros_300, K=8, sigma_grid=sigma_grid,
                          eps_values=[2.5, 3.0, 4.0], label="SCALE 2")

    # ================================================================
    # Scale 3: N=500, K=10
    # ================================================================
    cloud = source.generate(500)
    zeros_500 = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    # Fewer eps values for the larger scale (N=500 means N^2 cost per L_omega)
    run_holonomy_analysis(zeros_500, K=10, sigma_grid=sigma_grid,
                          eps_values=[3.0, 4.0], label="SCALE 3")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
