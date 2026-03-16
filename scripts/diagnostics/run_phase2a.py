#!/usr/bin/env python
"""Run Phase 2a: Abelian eigenbasis diagnostic.

Usage:
    python run_phase2a.py                # Quick validation (N=100, K=5)
    python run_phase2a.py --production   # Production (N=1000, K=20)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from atft.experiments.phase2a_abelian import Phase2aAbelian
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.visualization.plots import plot_resonance_matrix


def main():
    parser = argparse.ArgumentParser(description="Phase 2a: Abelian Eigenbasis Diagnostic")
    parser.add_argument("--production", action="store_true", help="N=1000, K=20")
    args = parser.parse_args()

    if args.production:
        N, K, n_eps = 1000, 20, 200
        label = "PRODUCTION"
    else:
        N, K, n_eps = 100, 5, 50
        label = "VALIDATION"

    print(f"\n{'='*60}")
    print(f"  ATFT Phase 2a — {label} RUN")
    print(f"  N={N}, K={K}, {n_eps} epsilon steps")
    print(f"{'='*60}\n")

    t0 = time.time()

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))
    cloud = source.generate(N)
    zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]

    builder = TransportMapBuilder(K=K, sigma=0.5)
    diag = Phase2aAbelian(builder, zeros)
    eps_grid = np.linspace(0, 3.0, n_eps)
    results = diag.run(eps_grid)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Distinct frequencies: {results['n_distinct_frequencies']}")
    print(f"Eigenvalues of A(0.5): {results['eigenvalues_A']}")
    print(f"\nResonance matrix:\n{results['resonance_matrix']}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"phase2a_{label.lower()}.png"
    plot_resonance_matrix(results["resonance_matrix"], results["eigenvalues_A"], save_path=fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
