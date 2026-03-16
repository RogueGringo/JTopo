#!/usr/bin/env python
"""Phase 3b: High-Frequency GPU Sweep.

Uses GPUSheafLaplacian with K=50+ (15+ primes) to test whether
higher-frequency prime harmonics sharpen the spectral signature
at sigma=0.5 into a true phase transition.

This is the focused "zeta-only" scout run. Once we confirm the
signature sharpens, we run full controls.

Usage:
    python -u -m atft.experiments.phase3b_gpu_sweep 2>&1 | tee output/phase3b_gpu_K50.log
    python -u -m atft.experiments.phase3b_gpu_sweep --K 80 2>&1 | tee output/phase3b_gpu_K80.log
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from numpy.typing import NDArray

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.gpu_sheaf_laplacian import GPUSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def main():
    parser = argparse.ArgumentParser(description="Phase 3b GPU High-Frequency Sweep")
    parser.add_argument("--K", type=int, default=50, help="Fiber dimension (default: 50)")
    parser.add_argument("--N", type=int, default=9877, help="Number of zeros (default: 9877)")
    parser.add_argument("--k-eig", type=int, default=50, help="Number of eigenvalues (default: 50)")
    parser.add_argument("--scout", action="store_true", help="Quick scout: few sigma/eps, zeta only")
    args = parser.parse_args()

    K = args.K
    N = args.N
    k_eig = args.k_eig

    if args.scout:
        sigma_grid = np.array([0.25, 0.40, 0.50, 0.60, 0.75])
        epsilon_grid = np.array([3.0, 5.0])
    else:
        sigma_grid = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75])
        epsilon_grid = np.array([2.0, 3.0, 4.0, 5.0])

    # Count primes up to K
    primes = []
    for n in range(2, K + 1):
        if all(n % d != 0 for d in range(2, int(n**0.5) + 1)):
            primes.append(n)

    print("=" * 70)
    print("  ATFT PHASE 3b: GPU HIGH-FREQUENCY SWEEP")
    print(f"  K={K} ({len(primes)} primes: {primes})")
    print("=" * 70)
    print(f"\n  N={N}, K={K}, dim={N*K}, k_eig={k_eig}")
    print(f"  sigma_grid: {sigma_grid}")
    print(f"  epsilon_grid: {epsilon_grid}")
    print(f"  Mode: {'scout' if args.scout else 'full'}")

    # Check GPU
    try:
        import cupy as cp
        mem = cp.cuda.Device(0).mem_info
        print(f"\n  GPU: {mem[0]/1e9:.2f} GB free / {mem[1]/1e9:.2f} GB VRAM")
    except Exception as e:
        print(f"\n  WARNING: GPU check failed: {e}")

    # Load zeta zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeta zeros: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")

    # --- Zeta Zeros Sweep ---
    print(f"\n  [ZETA ZEROS — K={K}]")
    results: dict[tuple[float, float], dict[str, float]] = {}

    for sigma in sigma_grid:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        for eps in epsilon_grid:
            t0 = time.time()
            lap = GPUSheafLaplacian(builder, zeta_zeros, transport_mode="superposition")
            eigs = lap.smallest_eigenvalues(eps, k=k_eig)
            s = float(np.sum(eigs))
            tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
            beta_0 = int(np.sum(eigs < tau))
            elapsed = time.time() - t0
            results[(sigma, eps)] = {'spectral_sum': s, 'kernel_dim': beta_0}
            print(f"    sigma={sigma:.2f} eps={eps:.1f}: S={s:.6f} b0={beta_0} ({elapsed:.1f}s)")
            sys.stdout.flush()

    # --- Analysis ---
    print(f"\n  {'=' * 60}")
    print(f"  K={K} SPECTRAL SUM TABLE")
    print(f"  {'=' * 60}")

    # Print table: sigma rows, epsilon columns
    header = f"  {'sigma':>6} |" + "".join(f" {'eps='+str(e):>12}" for e in epsilon_grid)
    print(header)
    print(f"  {'-' * 6}-+-" + "-" * (13 * len(epsilon_grid)))

    for sigma in sigma_grid:
        row = f"  {sigma:6.2f} |"
        for eps in epsilon_grid:
            s = results.get((sigma, eps), {}).get('spectral_sum', 0.0)
            row += f" {s:12.6f}"
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(row + marker)

    # Compute contrast at each epsilon
    print(f"\n  Contrast C = [S(0.5) - S(0.25)] / S(0.5):")
    for eps in epsilon_grid:
        s_half = results.get((0.50, eps), {}).get('spectral_sum', 0.0)
        s_quarter = results.get((0.25, eps), {}).get('spectral_sum', 0.0)
        if abs(s_half) > 1e-15:
            c = (s_half - s_quarter) / s_half
        else:
            c = 0.0
        print(f"    eps={eps:.1f}: C={c:.4f}")

    # Check for peak at sigma=0.5
    print(f"\n  Peak detection (does S peak at sigma=0.50?):")
    for eps in epsilon_grid:
        sums = [(sigma, results.get((sigma, eps), {}).get('spectral_sum', 0.0))
                for sigma in sigma_grid]
        peak_sigma, peak_s = max(sums, key=lambda x: x[1])
        s_at_half = results.get((0.50, eps), {}).get('spectral_sum', 0.0)
        is_peak = abs(peak_sigma - 0.5) < 1e-6
        print(f"    eps={eps:.1f}: peak at sigma={peak_sigma:.2f} (S={peak_s:.6f}), "
              f"S(0.5)={s_at_half:.6f} {'PEAK' if is_peak else 'not peak'}")

    # Symmetry check: S(sigma) vs S(1-sigma)
    print(f"\n  Functional equation symmetry S(sigma) vs S(1-sigma):")
    for eps in epsilon_grid:
        print(f"    eps={eps:.1f}:")
        for sigma in sigma_grid:
            if sigma < 0.5:
                s_mirror = round(1.0 - sigma, 3)
                s_val = results.get((sigma, eps), {}).get('spectral_sum', 0.0)
                s_mirr = results.get((s_mirror, eps), {}).get('spectral_sum', 0.0)
                ratio = s_val / s_mirr if abs(s_mirr) > 1e-15 else float('inf')
                print(f"      S({sigma:.2f})/S({s_mirror:.2f}) = {ratio:.4f}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
