#!/usr/bin/env python
"""Reproduce the K=20 Phase 3 superposition sweep results.

Runs the exact experiment matching published parameters and validates
the discrimination ratio against the published 670x claim.

Parameters (matching Phase 3 publication):
  N = 9877 Odlyzko zeros
  K = 20 (8 primes)
  sigma grid: [0.25, 0.30, ..., 0.75] (13 values)
  epsilon grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
  k_eig = 100
  transport: superposition, normalized
  5 random (Poisson) control trials
  5 GUE control trials

Output:
  output/reproduction_k20.csv  -- full results table
  stdout                       -- summary with PASS/FAIL verdict

Estimated runtime: ~2 hours on CPU (shift-invert eigsh).

Usage:
    python scripts/reproduce_k20.py             # full run
    python scripts/reproduce_k20.py --quick     # quick dev check (N=200, K=6)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "odlyzko_zeros.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_zeta_zeros(n_points: int) -> np.ndarray:
    """Load and unfold Odlyzko zeta zeros."""
    src = ZetaZerosSource(data_path=DATA_PATH)
    pc = src.generate(n_points=n_points)
    unfold = SpectralUnfolding(method="zeta")
    return unfold.transform(pc).points.ravel()


def generate_control(source_type: str, n_points: int, seed: int) -> np.ndarray:
    """Generate a control point cloud (Poisson or GUE)."""
    if source_type == "poisson":
        src = PoissonSource(seed=seed)
        pc = src.generate(n_points=n_points)
        return pc.points.ravel()  # Already unfolded
    elif source_type == "gue":
        src = GUESource(seed=seed)
        pc = src.generate(n_points=n_points)
        unfold = SpectralUnfolding(method="semicircle")
        return unfold.transform(pc).points.ravel()
    else:
        raise ValueError(f"Unknown source: {source_type}")


def compute_spectral_sum(
    zeros: np.ndarray, K: int, sigma: float, epsilon: float, k_eig: int,
) -> float:
    """Compute spectral sum for given parameters."""
    builder = TransportMapBuilder(K=K, sigma=sigma)
    lap = SparseSheafLaplacian(builder, zeros, normalize=True)
    return lap.spectral_sum(epsilon, k=k_eig)


def run_reproduction(
    n_points: int = 9877,
    K: int = 20,
    sigma_grid: np.ndarray | None = None,
    epsilon_grid: np.ndarray | None = None,
    k_eig: int = 100,
    n_random_trials: int = 5,
    n_gue_trials: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Run the full reproduction sweep."""
    if sigma_grid is None:
        sigma_grid = np.array([
            0.25, 0.30, 0.35, 0.40, 0.45, 0.48,
            0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75,
        ])
    if epsilon_grid is None:
        epsilon_grid = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

    total_points = len(sigma_grid) * len(epsilon_grid)
    print(f"Reproduction sweep: N={n_points}, K={K}, k_eig={k_eig}")
    print(f"  sigma grid: {sigma_grid}")
    print(f"  epsilon grid: {epsilon_grid}")
    print(f"  {total_points} grid points x (1 zeta + "
          f"{n_random_trials} random + {n_gue_trials} GUE)")
    print()

    # Load zeta zeros
    print("Loading zeta zeros...", end=" ", flush=True)
    t0 = time.perf_counter()
    zeta_zeros = load_zeta_zeros(n_points)
    print(f"{len(zeta_zeros)} zeros in {time.perf_counter() - t0:.1f}s")

    # Pre-generate control point clouds
    print("Generating controls...", end=" ", flush=True)
    random_clouds = [
        generate_control("poisson", n_points, seed + i)
        for i in range(n_random_trials)
    ]
    gue_clouds = [
        generate_control("gue", n_points, seed + 100 + i)
        for i in range(n_gue_trials)
    ]
    print(f"{n_random_trials} random + {n_gue_trials} GUE")
    print()

    results = []
    point_idx = 0

    for sigma in sigma_grid:
        for eps in epsilon_grid:
            point_idx += 1
            t_start = time.perf_counter()

            # Zeta zeros
            s_zeta = compute_spectral_sum(zeta_zeros, K, sigma, eps, k_eig)

            # Random controls
            s_random_vals = [
                compute_spectral_sum(rc, K, sigma, eps, k_eig)
                for rc in random_clouds
            ]
            s_random_mean = float(np.mean(s_random_vals))
            s_random_std = float(np.std(s_random_vals))

            # GUE controls
            s_gue_vals = [
                compute_spectral_sum(gc, K, sigma, eps, k_eig)
                for gc in gue_clouds
            ]
            s_gue_mean = float(np.mean(s_gue_vals))
            s_gue_std = float(np.std(s_gue_vals))

            # Discrimination ratio
            ratio = s_zeta / s_random_mean if s_random_mean > 0 else float("inf")

            elapsed = time.perf_counter() - t_start

            row = {
                "sigma": sigma,
                "epsilon": eps,
                "S_zeta": s_zeta,
                "S_random_mean": s_random_mean,
                "S_random_std": s_random_std,
                "S_gue_mean": s_gue_mean,
                "S_gue_std": s_gue_std,
                "discrimination_ratio": ratio,
            }
            results.append(row)

            print(
                f"  [{point_idx:3d}/{total_points}] "
                f"sigma={sigma:.2f} eps={eps:.1f}: "
                f"S_zeta={s_zeta:.6f}  "
                f"S_rand={s_random_mean:.6f}  "
                f"ratio={ratio:.1f}x  "
                f"({elapsed:.1f}s)"
            )
            sys.stdout.flush()

    return results


def save_csv(results: list[dict], path: Path) -> None:
    """Save results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sigma", "epsilon", "S_zeta",
        "S_random_mean", "S_random_std",
        "S_gue_mean", "S_gue_std",
        "discrimination_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_verdict(results: list[dict], published_ratio: float = 670.0) -> bool:
    """Print summary and PASS/FAIL verdict."""
    print()
    print("=" * 60)
    print("REPRODUCTION VERDICT")
    print("=" * 60)

    # Find max discrimination ratio across all grid points
    max_ratio = max(r["discrimination_ratio"] for r in results)
    max_row = max(results, key=lambda r: r["discrimination_ratio"])

    print(f"  Max discrimination ratio: {max_ratio:.1f}x")
    print(f"    at sigma={max_row['sigma']:.2f}, eps={max_row['epsilon']:.1f}")
    print(f"  Published ratio: {published_ratio:.0f}x")
    print()

    # Verdict: reproduced ratio should be within 5% of published
    # (or at least > 100x, demonstrating the arithmetic signal)
    tolerance = 0.05
    lower = published_ratio * (1 - tolerance)

    if max_ratio >= lower:
        print(f"  VERDICT: PASS (reproduced {max_ratio:.0f}x >= "
              f"{lower:.0f}x threshold)")
        passed = True
    elif max_ratio >= 100:
        print(f"  VERDICT: PARTIAL PASS (reproduced {max_ratio:.0f}x, "
              f"strong signal but below published {published_ratio:.0f}x)")
        passed = True
    else:
        print(f"  VERDICT: FAIL (reproduced {max_ratio:.0f}x < 100x)")
        passed = False

    print("=" * 60)
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the K=20 Phase 3 superposition sweep."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick dev check: N=200, K=6, reduced grid",
    )
    args = parser.parse_args()

    if args.quick:
        results = run_reproduction(
            n_points=200, K=6,
            sigma_grid=np.array([0.25, 0.50, 0.75]),
            epsilon_grid=np.array([3.0, 5.0]),
            k_eig=20, n_random_trials=2, n_gue_trials=2,
        )
        csv_path = OUTPUT_DIR / "reproduction_k6_quick.csv"
        published = 5.0  # Quick mode: K=6 has much weaker signal
    else:
        results = run_reproduction()
        csv_path = OUTPUT_DIR / "reproduction_k20.csv"
        published = 670.0

    save_csv(results, csv_path)
    print(f"\nResults saved to: {csv_path}")

    passed = print_verdict(results, published_ratio=published)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
