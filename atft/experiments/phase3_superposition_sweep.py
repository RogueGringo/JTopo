#!/usr/bin/env python
"""Phase 3 Superposition Sweep: The Definitive Control Test.

Runs the multi-prime superposition transport across zeta zeros, random
points, and GUE points at multiple sigma and epsilon values. Measures
whether the explicit-formula phase interference creates a genuine
arithmetic signal at sigma=0.5.

Usage:
    python -m atft.experiments.phase3_superposition_sweep          # full run
    python -m atft.experiments.phase3_superposition_sweep --quick   # dev mode

See: docs/superpowers/specs/2026-03-15-atft-phase3-superposition-design.md
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


@dataclass
class Phase3Config:
    """Configuration for the Phase 3 superposition sweep."""
    n_points: int = 9877
    K: int = 20
    sigma_grid: NDArray[np.float64] = field(default_factory=lambda: np.array(
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75]
    ))
    epsilon_grid: NDArray[np.float64] = field(default_factory=lambda: np.array(
        [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    ))
    k_eigenvalues: int = 100
    n_random_trials: int = 5
    n_gue_trials: int = 5
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    seed: int = 42

    @classmethod
    def quick(cls) -> Phase3Config:
        """Quick dev mode: small scale for fast iteration."""
        return cls(
            n_points=30,
            K=6,
            sigma_grid=np.array([0.25, 0.50, 0.75]),
            epsilon_grid=np.array([2.0, 3.0]),
            k_eigenvalues=10,
            n_random_trials=2,
            n_gue_trials=2,
        )


def generate_gue_points(
    n: int, mean_spacing: float, start: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Generate GUE-spaced points using rejection sampling of GUE Wigner surmise.

    GUE (beta=2): P(s) = (32/pi^2) * s^2 * exp(-4*s^2/pi)
    Note: NOT the GOE (beta=1) surmise which has linear repulsion.
    """
    spacings = []
    # GUE Wigner surmise: P(s) = (32/pi^2) * s^2 * exp(-4*s^2/pi)
    # Mode at s = sqrt(pi)/2 ~ 0.886, max density ~ 0.738
    # Use Rayleigh envelope for rejection sampling
    c_reject = 2.0  # rejection constant
    for _ in range(n - 1):
        while True:
            # Proposal: Rayleigh(sigma=sqrt(pi/8)) has peak at same location
            s = rng.rayleigh(scale=np.sqrt(np.pi / 8))
            # Target: (32/pi^2) * s^2 * exp(-4*s^2/pi)
            target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            # Rayleigh PDF: (s / sigma^2) * exp(-s^2/(2*sigma^2))
            sigma2 = np.pi / 8
            proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
            if proposal > 0 and rng.random() < target / (c_reject * proposal):
                spacings.append(s * mean_spacing)
                break
    return np.cumsum(np.array([start] + spacings))


def run_sigma_sweep(
    zeros: NDArray[np.float64],
    config: Phase3Config,
    normalize: bool,
    label: str,
) -> dict[tuple[float, float], dict[str, float]]:
    """Run sigma x epsilon sweep for a single point set.

    Returns dict mapping (sigma, epsilon) -> {
        'spectral_sum': float,
        'kernel_dim': int,  # beta_0 = #{lambda_i < tau}
    }
    """
    results: dict[tuple[float, float], dict[str, float]] = {}

    for sigma in config.sigma_grid:
        builder = TransportMapBuilder(K=config.K, sigma=sigma)
        for eps in config.epsilon_grid:
            t0 = time.time()
            lap = SparseSheafLaplacian(
                builder, zeros,
                transport_mode="superposition",
                normalize=normalize,
            )
            eigs = lap.smallest_eigenvalues(eps, k=config.k_eigenvalues)
            s = float(np.sum(eigs))
            # Kernel dimension: eigenvalues below threshold
            tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
            beta_0 = int(np.sum(eigs < tau))
            elapsed = time.time() - t0
            results[(sigma, eps)] = {'spectral_sum': s, 'kernel_dim': beta_0}
            print(f"    sigma={sigma:.2f} eps={eps:.1f}: S={s:.4f} b0={beta_0} ({elapsed:.1f}s)")
            sys.stdout.flush()

    return results


def compute_symmetrized(
    results: dict[tuple[float, float], dict[str, float]],
    sigma_grid: NDArray[np.float64],
    epsilon_grid: NDArray[np.float64],
) -> dict[tuple[float, float], float]:
    """Compute symmetrized spectral sum S_sym = [S(sigma) + S(1-sigma)] / 2."""
    sym: dict[tuple[float, float], float] = {}
    for sigma in sigma_grid:
        s_mirror = round(1.0 - sigma, 3)
        for eps in epsilon_grid:
            s_val = results.get((sigma, eps), {}).get('spectral_sum', 0.0)
            s_mirr = results.get((s_mirror, eps), {}).get('spectral_sum', s_val)
            sym[(sigma, eps)] = (s_val + s_mirr) / 2
    return sym


def compute_contrast(
    results: dict[tuple[float, float], dict[str, float]],
    epsilon_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute contrast C = [S(0.5) - S(0.25)] / S(0.5) at each epsilon."""
    contrasts = []
    for eps in epsilon_grid:
        s_half = results.get((0.50, eps), {}).get('spectral_sum', 0.0)
        s_quarter = results.get((0.25, eps), {}).get('spectral_sum', 0.0)
        if abs(s_half) > 1e-15:
            c = (s_half - s_quarter) / s_half
        else:
            c = 0.0
        contrasts.append(c)
    return np.array(contrasts)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Superposition Sweep")
    parser.add_argument("--quick", action="store_true", help="Quick dev mode")
    args = parser.parse_args()

    config = Phase3Config.quick() if args.quick else Phase3Config()

    print("=" * 70)
    print("  ATFT PHASE 3: SUPERPOSITION & SCALE")
    print("  Multi-prime phase interference control test")
    print("=" * 70)
    print(f"\n  N={config.n_points}, K={config.K}, k_eig={config.k_eigenvalues}")
    print(f"  sigma_grid: {config.sigma_grid}")
    print(f"  epsilon_grid: {config.epsilon_grid}")
    print(f"  Normalization: both True and False")

    # Load zeta zeros
    source = ZetaZerosSource(config.zeta_data_path)
    cloud = source.generate(config.n_points)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"\n  Zeta zeros loaded: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")

    rng = np.random.default_rng(config.seed)

    for normalize in [True, False]:
        norm_label = "NORMALIZED" if normalize else "UNNORMALIZED"
        print(f"\n{'=' * 70}")
        print(f"  {norm_label} SUPERPOSITION")
        print(f"{'=' * 70}")

        # --- Zeta zeros ---
        print(f"\n  [ZETA ZEROS]")
        zeta_results = run_sigma_sweep(zeta_zeros, config, normalize, "Zeta")

        # --- Random points ---
        all_random_results = []
        for trial in range(config.n_random_trials):
            print(f"\n  [RANDOM trial {trial + 1}]")
            rand_pts = np.sort(rng.uniform(
                zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)
            ))
            r = run_sigma_sweep(rand_pts, config, normalize, f"Random {trial+1}")
            all_random_results.append(r)

        # --- GUE points ---
        all_gue_results = []
        for trial in range(config.n_gue_trials):
            print(f"\n  [GUE trial {trial + 1}]")
            gue_pts = generate_gue_points(
                len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng
            )
            r = run_sigma_sweep(gue_pts, config, normalize, f"GUE {trial+1}")
            all_gue_results.append(r)

        # --- Compute contrasts ---
        zeta_contrast = compute_contrast(zeta_results, config.epsilon_grid)
        random_contrasts = [
            compute_contrast(r, config.epsilon_grid) for r in all_random_results
        ]
        gue_contrasts = [
            compute_contrast(r, config.epsilon_grid) for r in all_gue_results
        ]

        # Average control contrasts
        all_control = np.array(random_contrasts + gue_contrasts)  # (n_trials, n_eps)
        mean_control = np.mean(all_control, axis=0)

        # Signal strength R per epsilon
        R_values = np.where(
            np.abs(mean_control) > 1e-15,
            zeta_contrast / mean_control,
            0.0,
        )

        # --- Symmetrized spectral sum ---
        zeta_sym = compute_symmetrized(
            zeta_results, config.sigma_grid, config.epsilon_grid
        )

        # --- Summary ---
        print(f"\n  {'=' * 60}")
        print(f"  SUMMARY ({norm_label})")
        print(f"  {'=' * 60}")

        # Symmetrized spectral sum table (for Phase 2 comparison)
        print(f"\n  Symmetrized spectral sum S_sym at eps={config.epsilon_grid[1]:.1f}:")
        ref_eps = config.epsilon_grid[1]
        for sigma in config.sigma_grid:
            if sigma <= 0.5:
                s_sym = zeta_sym.get((sigma, ref_eps), 0.0)
                marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
                print(f"    sigma={sigma:.2f}: S_sym={s_sym:.4f}{marker}")

        # Kernel dimension at sigma=0.5
        print(f"\n  Kernel dimension beta_0 at sigma=0.5:")
        for eps in config.epsilon_grid:
            b0 = zeta_results.get((0.50, eps), {}).get('kernel_dim', 0)
            print(f"    eps={eps:.1f}: beta_0={b0}")

        print(f"\n  {'epsilon':>8} | {'C(zeta)':>10} | {'C(ctrl)':>10} | {'R':>10}")
        print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
        for i, eps in enumerate(config.epsilon_grid):
            marker = " <--" if R_values[i] > 2.0 else ""
            print(f"  {eps:8.1f} | {zeta_contrast[i]:10.4f} | "
                  f"{mean_control[i]:10.4f} | {R_values[i]:10.4f}{marker}")

        # Outcome determination
        strong_count = int(np.sum(R_values > 2.0))
        weak_count = int(np.sum(R_values > 1.0))
        n_eps = len(config.epsilon_grid)

        print(f"\n  R > 2.0 at {strong_count}/{n_eps} epsilon values")
        print(f"  R > 1.0 at {weak_count}/{n_eps} epsilon values")

        if strong_count > n_eps * 2 // 3:
            print(f"\n  ** OUTCOME A: STRONG SIGNAL **")
            print(f"  The primes are singing at sigma=0.5.")
        elif weak_count > n_eps * 2 // 3:
            print(f"\n  ** OUTCOME B: WEAK SIGNAL **")
            print(f"  Arithmetic structure detected but needs refinement.")
        else:
            print(f"\n  ** OUTCOME C: NO SIGNAL **")
            print(f"  Superposition transport does not distinguish zeta zeros.")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
