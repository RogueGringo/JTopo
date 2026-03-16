#!/usr/bin/env python
"""Run Phase 2b: Full sigma-sweep experiment.

Usage:
    python run_phase2b.py               # Quick validation (N=20, K=2, 3 sigmas)
    python run_phase2b.py --production  # Production (N=1000, K=20, 9 sigmas)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from atft.experiments.phase2b_sheaf import Phase2bConfig, Phase2bExperiment
from atft.visualization.plots import plot_sigma_peak


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Sigma-Sweep Experiment")
    parser.add_argument("--production", action="store_true", help="Full production run")
    args = parser.parse_args()

    if args.production:
        config = Phase2bConfig()
        label = "PRODUCTION"
    else:
        config = Phase2bConfig(
            n_points=20,
            K=2,
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            n_epsilon_steps=10,
            epsilon_max=3.0,
            m=5,
        )
        label = "VALIDATION"

    print(f"\n{'='*60}")
    print(f"  ATFT Phase 2b — {label} RUN")
    print(f"  N={config.n_points}, K={config.K}")
    print(f"  {len(config.sigma_grid)} sigma values x {config.n_epsilon_steps} epsilon steps")
    print(f"{'='*60}\n")

    t0 = time.time()
    experiment = Phase2bExperiment(config)
    result = experiment.run()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Peak sigma: {result.peak_sigma}")
    print(f"Peak kernel dim: {result.peak_kernel_dim}")
    print(f"Unique peak: {result.is_unique_peak}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"phase2b_{label.lower()}.png"
    plot_sigma_peak(result, save_path=fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
