#!/usr/bin/env python
"""Diagnostic: examine the actual eigenvalue spectrum of the sheaf Laplacian.

The kernel_dim counting is saturating at m. We need to see:
1. What the actual eigenvalues ARE (not just how many are below tol)
2. Whether they differ between sigma values
3. Whether the Frobenius norm tolerance is too loose
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def main():
    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))
    cloud = source.generate(30)
    zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    N = len(zeros)

    print(f"N = {N} zeros")
    print(f"Zeros range: [{zeros.min():.2f}, {zeros.max():.2f}]")
    print(f"Mean spacing: {np.mean(np.diff(np.sort(zeros))):.4f}")

    # Test with K=3
    K = 3
    dim = N * K * K
    print(f"\nK={K}, dim={dim}")

    sigma_vals = [0.3, 0.5, 0.7]
    eps_vals = [0.5, 1.0, 1.5, 2.0, 2.5]

    for sigma in sigma_vals:
        print(f"\n{'='*60}")
        print(f"  sigma = {sigma}")
        print(f"{'='*60}")

        builder = TransportMapBuilder(K=K, sigma=sigma)
        eig_A = builder.eigenvalues()
        print(f"  A(sigma) eigenvalues: {eig_A}")
        print(f"  Eigenvalue spread: {eig_A.max() - eig_A.min():.4f}")

        # Check how nontrivial the transport is at delta_gamma = 1.0
        U = builder.transport(1.0)
        deviation = np.linalg.norm(U - np.eye(K))
        print(f"  ||U(1.0) - I|| = {deviation:.4f}")

        lap = SheafLaplacian(builder, zeros)

        for eps in eps_vals:
            m = min(20, dim - 1)
            eigs = lap.smallest_eigenvalues(eps, m=m)
            fnorm = lap.frobenius_norm_estimate(eps)
            tol = fnorm * 1e-6 if fnorm > 0 else 1e-10
            n_kernel = int(np.sum(eigs < tol))
            n_edges = len(lap._edges(eps))

            print(f"\n  eps={eps:.1f}: {n_edges} edges, fnorm={fnorm:.2f}, tol={tol:.2e}")
            print(f"    kernel_dim (< tol): {n_kernel}")
            print(f"    20 smallest eigenvalues:")
            for i in range(0, min(20, len(eigs)), 5):
                chunk = eigs[i:i+5]
                formatted = [f"{v:.2e}" for v in chunk]
                print(f"      [{i:2d}-{i+len(chunk)-1:2d}]: {', '.join(formatted)}")

        sys.stdout.flush()

    # Now test: what if we look at the SUM of smallest eigenvalues
    # (spectral gap aggregate) rather than kernel count?
    print(f"\n\n{'='*60}")
    print(f"  SPECTRAL GAP ANALYSIS")
    print(f"{'='*60}")
    print(f"\nSum of 20 smallest eigenvalues at eps=1.0:")

    sigma_fine = np.linspace(0.3, 0.7, 11)
    for sigma in sigma_fine:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.0, m=20)
        total = np.sum(eigs)
        nonzero = eigs[eigs > 1e-10]
        spectral_gap = nonzero[0] if len(nonzero) > 0 else 0.0
        marker = " <-- sigma=1/2" if abs(sigma - 0.5) < 0.01 else ""
        bar_len = int(40 * total / max(total, 0.001))
        print(f"  sigma={sigma:.2f}: sum={total:.6f}, "
              f"gap={spectral_gap:.6f}, "
              f"n_nonzero={len(nonzero)}{marker}")

    # Also try K=6 for more nontrivial transport
    print(f"\n\n{'='*60}")
    print(f"  K=6 ANALYSIS (more nontrivial transport)")
    print(f"{'='*60}")
    K = 6
    dim = N * K * K
    print(f"K={K}, dim={dim}")

    for sigma in [0.3, 0.5, 0.7]:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        eig_A = builder.eigenvalues()
        U = builder.transport(1.0)
        deviation = np.linalg.norm(U - np.eye(K))
        print(f"\n  sigma={sigma}: A eigenvalues={np.array2string(eig_A, precision=3)}")
        print(f"    ||U(1.0)-I|| = {deviation:.4f}")

    print(f"\nSum of 20 smallest eigenvalues at eps=1.0 (K=6):")
    for sigma in sigma_fine:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.0, m=20)
        total = np.sum(eigs)
        nonzero = eigs[eigs > 1e-10]
        spectral_gap = nonzero[0] if len(nonzero) > 0 else 0.0
        marker = " <-- sigma=1/2" if abs(sigma - 0.5) < 0.01 else ""
        print(f"  sigma={sigma:.2f}: sum={total:.6f}, "
              f"gap={spectral_gap:.6f}, "
              f"n_nonzero={len(nonzero)}{marker}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
