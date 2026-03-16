#!/usr/bin/env python
"""Run Phase 1 experiment: Zeta vs GUE vs Poisson topological benchmark.

Usage:
    python run_phase1.py                # Quick validation (N=500, M=50)
    python run_phase1.py --production   # Production run (N=1000, M=200)
    python run_phase1.py --full         # Full run (N=2000, M=500)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from atft.experiments.phase1_benchmark import Phase1Config, Phase1Experiment
from atft.visualization.plots import plot_phase1_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: ATFT Riemann Hypothesis Benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--production", action="store_true", help="N=1000, M=200")
    group.add_argument("--full", action="store_true", help="N=2000, M=500")
    group.add_argument("--deep", action="store_true", help="N=10000, M=500")
    group.add_argument("--ultra", action="store_true", help="N=50000, M=300")
    args = parser.parse_args()

    if args.ultra:
        config = Phase1Config(n_points=50_000, ensemble_size=300, n_epsilon_steps=500, zeta_data_path=Path("data/odlyzko_zeros.txt"))
        label = "ULTRA"
    elif args.deep:
        config = Phase1Config(n_points=10_000, ensemble_size=500, n_epsilon_steps=500, zeta_data_path=Path("data/odlyzko_zeros.txt"))
        label = "DEEP"
    elif args.full:
        config = Phase1Config(n_points=2000, ensemble_size=500, zeta_data_path=Path("data/odlyzko_zeros.txt"))
        label = "FULL"
    elif args.production:
        config = Phase1Config(n_points=1000, ensemble_size=200, zeta_data_path=Path("data/odlyzko_zeros.txt"))
        label = "PRODUCTION"
    else:
        config = Phase1Config(n_points=500, ensemble_size=50, zeta_data_path=Path("data/odlyzko_zeros.txt"))
        label = "VALIDATION"

    print(f"\n{'='*60}")
    print(f"  ATFT Phase 1 — {label} RUN")
    print(f"  N={config.n_points} points, M={config.ensemble_size} ensemble samples")
    print(f"  K={config.k_waypoints} waypoints, {config.n_epsilon_steps} epsilon steps")
    print(f"{'='*60}\n")

    t0 = time.time()
    experiment = Phase1Experiment(config)
    results = experiment.run()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s")

    # Generate publication figure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"phase1_{label.lower()}.png"

    print(f"\nGenerating figure: {fig_path}")
    plot_phase1_results(
        zeta_curves=results.zeta_curves,
        gue_curves=results.gue_curves,
        poisson_curves=results.poisson_curves,
        zeta_sig=results.zeta_signature,
        gue_sigs=results.gue_signatures,
        poisson_sig=results.poisson_signatures[0],
        zeta_result=results.zeta_validation,
        poisson_result=results.poisson_validation,
        save_path=fig_path,
    )
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
