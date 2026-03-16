#!/usr/bin/env python
"""Phase 3c: K=100 GPU Sweep on RTX 4080 (12GB VRAM).

25 primes up to 97 -- the critical test for Fourier sharpening.
N=2000 with aggressive VRAM cleanup between points.

VRAM budget (K=100, N=2000):
  eps=3.0: ~6.5 GB peak (COO phase) -- comfortable
  eps=5.0: ~11.6 GB peak (COO phase) -- tight, may OOM

Usage:
  python -u -m atft.experiments.phase3c_gpu_k100 2>&1 | tee output/phase3c_k100_N2000.log
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.gpu_sheaf_laplacian import GPUSheafLaplacian


def gpu_cleanup():
    """Force CuPy to release all cached VRAM."""
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def run_point(zeros, K, sigma, epsilon, k_eig, label):
    """Run one (sigma, epsilon) point with VRAM cleanup."""
    import cupy as cp
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = GPUSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(epsilon, k=k_eig)
        s = float(np.sum(eigs))
        tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
        beta_0 = int(np.sum(eigs < tau))
        elapsed = time.time() - t0
        mem = cp.cuda.Device(0).mem_info
        print(f"  [{label}] sigma={sigma:.3f} eps={epsilon:.1f}: "
              f"S={s:.6f} b0={beta_0} ({elapsed:.1f}s) "
              f"[{mem[0]/1e9:.1f}GB free]")
        sys.stdout.flush()
        result = {"sigma": float(sigma), "epsilon": float(epsilon),
                  "spectral_sum": s, "kernel_dim": beta_0,
                  "eigs_top5": eigs[:5].tolist(), "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{label}] sigma={sigma:.3f} eps={epsilon:.1f}: "
              f"FAILED ({e}) ({elapsed:.1f}s)")
        sys.stdout.flush()
        result = None
    finally:
        gpu_cleanup()
    return result


def generate_gue_points(n, mean_spacing, start, rng):
    """GUE (beta=2) Wigner surmise via rejection sampling."""
    spacings = []
    c_reject = 2.0
    for _ in range(n - 1):
        while True:
            s = rng.rayleigh(scale=np.sqrt(np.pi / 8))
            target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            sigma2 = np.pi / 8
            proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
            if proposal > 0 and rng.random() < target / (c_reject * proposal):
                spacings.append(s * mean_spacing)
                break
    return np.cumsum(np.array([start] + spacings))


def main():
    K = 100
    N = 2000
    k_eig = 20

    # Finer grid near 0.50 to resolve the peak
    sigma_grid = np.array([0.25, 0.35, 0.40, 0.44, 0.46, 0.48, 0.49,
                           0.50, 0.51, 0.52, 0.54, 0.56, 0.60, 0.65, 0.75])
    # eps=3.0 first (safe), then eps=5.0 (tight VRAM)
    epsilon_grid = np.array([3.0, 5.0])

    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    print("=" * 70)
    print(f"  ATFT PHASE 3c: K=100 GPU SWEEP (25 primes)")
    print(f"  K={K}, N={N}, dim={N*K}, k_eig={k_eig}")
    print(f"  Primes: {primes}")
    print(f"  sigma grid: {len(sigma_grid)} points, "
          f"[{sigma_grid[0]:.2f} ... {sigma_grid[-1]:.2f}]")
    print(f"  epsilon grid: {list(epsilon_grid)}")
    n_pts = len(sigma_grid) * len(epsilon_grid)
    print(f"  Total points: {n_pts} zeta + {n_pts} random + {n_pts} GUE = {3*n_pts}")
    print("=" * 70)

    # GPU check
    import cupy as cp
    mem = cp.cuda.Device(0).mem_info
    print(f"  GPU: {mem[0]/1e9:.2f} GB free / {mem[1]/1e9:.2f} GB VRAM")

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeros loaded: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")
    sys.stdout.flush()

    all_results = {}

    # ---- Run each source for eps=3.0 (safe), then eps=5.0 (tight) ----
    sources = [
        ("Zeta", zeta_zeros),
    ]

    # Generate controls upfront (CPU, no VRAM)
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)))
    gue_pts = generate_gue_points(len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng)
    sources.append(("Random", rand_pts))
    sources.append(("GUE", gue_pts))

    for source_label, pts in sources:
        print(f"\n  [{source_label.upper()} -- K={K}, N={N}]")
        all_results[source_label] = {}
        for eps in epsilon_grid:
            for sigma in sigma_grid:
                r = run_point(pts, K, float(sigma), float(eps), k_eig, source_label)
                if r is not None:
                    all_results[source_label][(float(sigma), float(eps))] = r

    # ---- ANALYSIS ----
    print(f"\n{'='*70}")
    print(f"  RESULTS -- K={K}, N={N}")
    print(f"{'='*70}")

    zeta_results = all_results.get("Zeta", {})
    rand_results = all_results.get("Random", {})
    gue_results = all_results.get("GUE", {})

    for eps in epsilon_grid:
        print(f"\n  eps={eps:.1f}:")
        print(f"  {'sigma':>7} {'S(zeta)':>12} {'S(rand)':>12} {'S(GUE)':>12} {'zeta/rand':>10}")
        print(f"  {'-------':>7} {'------------':>12} {'------------':>12} {'------------':>12} {'----------':>10}")
        for sigma in sigma_grid:
            key = (float(sigma), float(eps))
            sz = zeta_results.get(key, {}).get("spectral_sum", float('nan'))
            sr = rand_results.get(key, {}).get("spectral_sum", float('nan'))
            sg = gue_results.get(key, {}).get("spectral_sum", float('nan'))
            ratio = sz / sr if sr > 1e-10 else float('inf')
            marker = " <--" if abs(sigma - 0.50) < 0.005 else ""
            print(f"  {sigma:7.3f} {sz:12.6f} {sr:12.6f} {sg:12.6f} {ratio:10.1f}{marker}")

    # Contrast and peak detection
    for eps in epsilon_grid:
        zeta_vals = {}
        for sigma in sigma_grid:
            key = (float(sigma), float(eps))
            if key in zeta_results:
                zeta_vals[float(sigma)] = zeta_results[key]["spectral_sum"]

        if not zeta_vals:
            continue

        sigmas = sorted(zeta_vals.keys())
        zv = [zeta_vals[s] for s in sigmas]

        peak_idx = int(np.argmax(zv))
        peak_sigma = sigmas[peak_idx]
        peak_val = zv[peak_idx]

        # Parabolic interpolation
        if 0 < peak_idx < len(sigmas) - 1:
            sl, sc, sr = zv[peak_idx-1], zv[peak_idx], zv[peak_idx+1]
            denom = 2 * (sl - 2*sc + sr)
            if abs(denom) > 1e-15:
                delta = sigmas[peak_idx] - sigmas[peak_idx-1]
                offset = delta * (sl - sr) / denom
                peak_sigma = sigmas[peak_idx] + offset

        s_min = min(zv)
        contrast = (peak_val - s_min) / peak_val if peak_val > 0 else 0

        print(f"\n  eps={eps:.1f}: peak_sigma={peak_sigma:.4f}, S(peak)={peak_val:.6f}, "
              f"contrast={contrast:.4f}")

        if 0.50 in zeta_vals:
            s50 = zeta_vals[0.50]
            if peak_val > 0:
                print(f"         S(0.50)={s50:.6f} vs S(peak)={peak_val:.6f} "
                      f"(ratio={s50/peak_val:.4f})")

    print(f"\n  DONE")


if __name__ == "__main__":
    main()
