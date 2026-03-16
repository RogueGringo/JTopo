#!/usr/bin/env python
"""Phase 3 Distributed Sweep — Multi-Machine Parameter Partitioning.

Each machine runs the same script with different --role and --backend flags.
Results are written to JSON files that can be collected and merged.

Roles:
  control-cpu   — K=20 random/GUE control trials (CPU only, for 1060 laptop)
  gpu-k50       — K=50 zeta + controls (GPU, for 4080/5070)
  gpu-k100      — K=100 zeta + controls (GPU, for A100 on RunPod)
  gpu-k200      — K=200 zeta only scout (GPU, for A100 on RunPod)

Usage:
  # On the GTX 1060 laptop (CPU only, K=20 controls):
  python -u -m atft.experiments.phase3_distributed --role control-cpu \\
      --trials 3,4,5 2>&1 | tee output/k20_controls_1060.log

  # On the RTX 5070 desktop (GPU, K=50 full sweep):
  python -u -m atft.experiments.phase3_distributed --role gpu-k50 \\
      2>&1 | tee output/k50_sweep_5070.log

  # On RunPod A100 (GPU, K=100 definitive sweep):
  python -u -m atft.experiments.phase3_distributed --role gpu-k100 \\
      2>&1 | tee output/k100_sweep_a100.log

  # On RunPod A100 (GPU, K=200 scout):
  python -u -m atft.experiments.phase3_distributed --role gpu-k200 \\
      2>&1 | tee output/k200_scout_a100.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder


def generate_gue_points(
    n: int, mean_spacing: float, start: float, rng: np.random.Generator
) -> NDArray[np.float64]:
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


def run_sweep(
    zeros: NDArray[np.float64],
    K: int,
    sigma_grid: NDArray[np.float64],
    epsilon_grid: NDArray[np.float64],
    k_eig: int,
    backend: str,
    label: str,
) -> dict:
    """Run sigma x epsilon sweep, returns results dict."""
    results = {}

    for sigma in sigma_grid:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        for eps in epsilon_grid:
            t0 = time.time()

            if backend == "gpu":
                from atft.topology.gpu_sheaf_laplacian import GPUSheafLaplacian
                lap = GPUSheafLaplacian(builder, zeros, transport_mode="superposition")
            else:
                from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
                lap = SparseSheafLaplacian(
                    builder, zeros, transport_mode="superposition", normalize=True,
                )

            eigs = lap.smallest_eigenvalues(eps, k=k_eig)
            s = float(np.sum(eigs))
            tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
            beta_0 = int(np.sum(eigs < tau))
            elapsed = time.time() - t0

            key = f"{sigma:.3f}_{eps:.1f}"
            results[key] = {
                "sigma": float(sigma),
                "epsilon": float(eps),
                "spectral_sum": s,
                "kernel_dim": beta_0,
                "time_s": elapsed,
            }
            print(f"  [{label}] sigma={sigma:.2f} eps={eps:.1f}: "
                  f"S={s:.6f} b0={beta_0} ({elapsed:.1f}s)")
            sys.stdout.flush()

    return results


# ---- Role configurations ----

ROLES = {
    "control-cpu": {
        "K": 20,
        "N": 9877,
        "k_eig": 100,
        "backend": "cpu",
        "sigma_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50,
                       0.52, 0.55, 0.60, 0.65, 0.70, 0.75],
        "epsilon_grid": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "run_zeta": False,
        "n_random": 5,
        "n_gue": 5,
    },
    "gpu-k50": {
        "K": 50,
        "N": 9877,
        "k_eig": 20,
        "backend": "gpu",
        "sigma_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50,
                       0.52, 0.55, 0.60, 0.65, 0.70, 0.75],
        "epsilon_grid": [2.0, 3.0, 4.0, 5.0],
        "run_zeta": True,
        "n_random": 2,
        "n_gue": 2,
    },
    "gpu-k100": {
        "K": 100,
        "N": 9877,
        "k_eig": 20,
        "backend": "gpu",
        "sigma_grid": [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50,
                       0.52, 0.55, 0.60, 0.65, 0.70, 0.75],
        "epsilon_grid": [3.0, 5.0],
        "run_zeta": True,
        "n_random": 1,
        "n_gue": 1,
    },
    "gpu-k200": {
        "K": 200,
        "N": 9877,
        "k_eig": 10,
        "backend": "gpu",
        "sigma_grid": [0.25, 0.40, 0.50, 0.60, 0.75],
        "epsilon_grid": [3.0, 5.0],
        "run_zeta": True,
        "n_random": 0,
        "n_gue": 0,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Distributed Sweep")
    parser.add_argument("--role", required=True, choices=ROLES.keys())
    parser.add_argument("--trials", type=str, default=None,
                        help="Comma-separated trial indices (e.g., '3,4,5')")
    parser.add_argument("--zeta-only", action="store_true",
                        help="Skip random and GUE controls (budget-conscious mode)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ROLES[args.role]
    K = cfg["K"]
    N = cfg["N"]
    k_eig = cfg["k_eig"]
    backend = cfg["backend"]
    sigma_grid = np.array(cfg["sigma_grid"])
    epsilon_grid = np.array(cfg["epsilon_grid"])

    # Count primes
    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    print("=" * 70)
    print(f"  ATFT PHASE 3 DISTRIBUTED — Role: {args.role}")
    print(f"  K={K} ({len(primes)} primes), N={N}, backend={backend}")
    print("=" * 70)
    sys.stdout.flush()

    # Check GPU if needed
    if backend == "gpu":
        try:
            import cupy as cp
            mem = cp.cuda.Device(0).mem_info
            print(f"  GPU: {mem[0]/1e9:.2f} GB free / {mem[1]/1e9:.2f} GB VRAM")
        except Exception as e:
            print(f"  WARNING: GPU unavailable ({e}), falling back to CPU")
            cfg["backend"] = "cpu"
            backend = "cpu"

    # Load zeta zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"  Zeros: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    all_results = {"role": args.role, "K": K, "N": N, "primes": primes}

    # --- Zeta zeros ---
    if cfg["run_zeta"]:
        print(f"\n  [ZETA ZEROS — K={K}]")
        all_results["zeta"] = run_sweep(
            zeta_zeros, K, sigma_grid, epsilon_grid, k_eig, backend, "Zeta"
        )

    # --- Random controls ---
    n_random = 0 if args.zeta_only else cfg["n_random"]
    n_gue = 0 if args.zeta_only else cfg["n_gue"]
    trial_indices = None
    if args.trials:
        trial_indices = [int(x) for x in args.trials.split(",")]

    all_results["random"] = {}
    for trial in range(1, n_random + 1):
        if trial_indices and trial not in trial_indices:
            # Burn the RNG state to keep seeding consistent
            rng.uniform(0, 1, len(zeta_zeros))
            continue
        print(f"\n  [RANDOM trial {trial}]")
        rand_pts = np.sort(rng.uniform(
            zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)
        ))
        all_results["random"][f"trial_{trial}"] = run_sweep(
            rand_pts, K, sigma_grid, epsilon_grid, k_eig, backend, f"Rand-{trial}"
        )

    # --- GUE controls ---
    all_results["gue"] = {}
    for trial in range(1, n_gue + 1):
        if trial_indices and trial not in trial_indices:
            continue
        print(f"\n  [GUE trial {trial}]")
        gue_pts = generate_gue_points(
            len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng
        )
        all_results["gue"][f"trial_{trial}"] = run_sweep(
            gue_pts, K, sigma_grid, epsilon_grid, k_eig, backend, f"GUE-{trial}"
        )

    # --- Save results ---
    out_file = out_dir / f"phase3_{args.role}_K{K}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_file}")
    print("  DONE")


if __name__ == "__main__":
    main()
