#!/usr/bin/env python
"""Phase 3 Result Aggregation & Analysis.

Merges JSON results from distributed machines and computes:
  - Contrast ratios C(σ) = [S(σ) - S(σ_min)] / S(σ) per epsilon
  - Signal-to-noise R = C_zeta / mean(C_controls)
  - Peak σ detection with parabolic interpolation
  - Summary table across all K values

Usage:
  # After collecting all JSON files into output/:
  python scripts/aggregate_results.py output/phase3_*.json

  # Specific files:
  python scripts/aggregate_results.py \
      output/phase3_control-cpu_K20.json \
      output/phase3_gpu-k50_K50.json \
      output/phase3_gpu-k100_K100.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_results(paths: list[str]) -> dict[int, dict]:
    """Load and merge JSON results by K value."""
    merged = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        K = data["K"]
        if K not in merged:
            merged[K] = data
        else:
            # Merge control trials from different machines
            for source_type in ["random", "gue"]:
                if source_type in data:
                    if source_type not in merged[K]:
                        merged[K][source_type] = {}
                    merged[K][source_type].update(data[source_type])
            # Prefer zeta from GPU if available
            if "zeta" in data and ("zeta" not in merged[K] or "gpu" in path.lower()):
                merged[K]["zeta"] = data["zeta"]
    return merged


def extract_sigma_profile(sweep_data: dict, sigma_grid: list[float]) -> dict[float, dict[float, float]]:
    """Extract {epsilon: {sigma: spectral_sum}} from sweep results."""
    profiles = {}
    for key, val in sweep_data.items():
        sigma = val["sigma"]
        eps = val["epsilon"]
        s = val["spectral_sum"]
        if eps not in profiles:
            profiles[eps] = {}
        profiles[eps][sigma] = s
    return profiles


def compute_contrast(profile: dict[float, float]) -> tuple[float, float]:
    """Compute contrast ratio C = [S(peak) - S(min)] / S(peak) and peak sigma."""
    if not profile:
        return 0.0, 0.0
    sigmas = sorted(profile.keys())
    values = [profile[s] for s in sigmas]
    if max(values) <= 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(values))
    s_peak = values[peak_idx]
    s_min = min(values)
    contrast = (s_peak - s_min) / s_peak if s_peak > 0 else 0.0

    # Parabolic interpolation for sub-grid peak
    sigma_peak = sigmas[peak_idx]
    if 0 < peak_idx < len(sigmas) - 1:
        s_l, s_c, s_r = values[peak_idx - 1], values[peak_idx], values[peak_idx + 1]
        denom = 2 * (s_l - 2 * s_c + s_r)
        if abs(denom) > 1e-12:
            delta = (sigmas[peak_idx] - sigmas[peak_idx - 1])
            offset = delta * (s_l - s_r) / denom
            sigma_peak = sigmas[peak_idx] + offset

    return contrast, sigma_peak


def analyze_k(K: int, data: dict):
    """Full analysis for one K value."""
    print(f"\n{'='*70}")
    print(f"  K={K} ({len(data.get('primes', []))} primes), N={data.get('N', '?')}")
    print(f"{'='*70}")

    # Zeta zeros
    if "zeta" not in data:
        print("  No zeta data available.")
        return

    zeta_profiles = extract_sigma_profile(data["zeta"], [])
    epsilons = sorted(zeta_profiles.keys())

    print(f"\n  {'sigma':>6}", end="")
    for eps in epsilons:
        print(f"  {'ε='+str(eps):>10}", end="")
    print()
    print(f"  {'------':>6}", end="")
    for _ in epsilons:
        print(f"  {'----------':>10}", end="")
    print()

    # Collect all sigmas
    all_sigmas = set()
    for prof in zeta_profiles.values():
        all_sigmas.update(prof.keys())
    sigmas = sorted(all_sigmas)

    for sigma in sigmas:
        print(f"  {sigma:6.3f}", end="")
        for eps in epsilons:
            s = zeta_profiles[eps].get(sigma, 0.0)
            print(f"  {s:10.6f}", end="")
        print()

    # Contrast ratios
    print(f"\n  Contrast & Peak Detection:")
    zeta_contrasts = {}
    for eps in epsilons:
        c, peak = compute_contrast(zeta_profiles[eps])
        zeta_contrasts[eps] = c
        marker = " *** σ=0.50!" if abs(peak - 0.50) < 0.03 else ""
        print(f"    ε={eps:.1f}: C={c:.4f}, peak_σ={peak:.3f}{marker}")

    # Controls
    for source_type, label in [("random", "RANDOM"), ("gue", "GUE")]:
        controls = data.get(source_type, {})
        if not controls:
            continue
        n_trials = len(controls)
        print(f"\n  {label} Controls ({n_trials} trials):")
        for eps in epsilons:
            contrasts = []
            for trial_key, trial_data in controls.items():
                prof = extract_sigma_profile(trial_data, [])
                if eps in prof:
                    c, _ = compute_contrast(prof[eps])
                    contrasts.append(c)
            if contrasts:
                mean_c = np.mean(contrasts)
                std_c = np.std(contrasts)
                zeta_c = zeta_contrasts.get(eps, 0)
                R = zeta_c / mean_c if mean_c > 0 else float('inf')
                print(f"    ε={eps:.1f}: mean_C={mean_c:.4f}±{std_c:.4f}, "
                      f"R(zeta/ctrl)={R:.1f}x")

    # Symmetry check: S(0.5-δ) vs S(0.5+δ)
    print(f"\n  σ=0.50 Symmetry Check:")
    for eps in epsilons:
        prof = zeta_profiles.get(eps, {})
        pairs = [(0.25, 0.75), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55), (0.48, 0.52)]
        for s_lo, s_hi in pairs:
            if s_lo in prof and s_hi in prof:
                v_lo, v_hi = prof[s_lo], prof[s_hi]
                asym = abs(v_lo - v_hi) / max(v_lo, v_hi, 1e-12)
                sym = "✓" if asym < 0.05 else "✗"
                if asym > 0.01:  # Only print notable ones
                    print(f"    ε={eps:.1f}: S({s_lo})={v_lo:.6f} vs S({s_hi})={v_hi:.6f} "
                          f"asym={asym:.3f} {sym}")


def summary_table(all_data: dict[int, dict]):
    """Cross-K summary of peak sharpening."""
    print(f"\n{'='*70}")
    print(f"  CROSS-K FOURIER SHARPENING SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'K':>5} {'primes':>6} {'ε':>5} {'peak_σ':>7} {'contrast':>9} {'R':>8}  verdict")
    print(f"  {'-----':>5} {'------':>6} {'-----':>5} {'-------':>7} {'---------':>9} {'--------':>8}  -------")

    for K in sorted(all_data.keys()):
        data = all_data[K]
        n_primes = len(data.get("primes", []))
        if "zeta" not in data:
            continue
        profiles = extract_sigma_profile(data["zeta"], [])
        # Pick best epsilon (highest contrast)
        best_eps, best_c, best_peak = 0, 0, 0
        for eps, prof in profiles.items():
            c, peak = compute_contrast(prof)
            if c > best_c:
                best_eps, best_c, best_peak = eps, c, peak

        # Get R from random controls
        R_str = "N/A"
        controls = data.get("random", {})
        if controls and best_eps > 0:
            ctrl_cs = []
            for trial_data in controls.values():
                prof = extract_sigma_profile(trial_data, [])
                if best_eps in prof:
                    c, _ = compute_contrast(prof[best_eps])
                    ctrl_cs.append(c)
            if ctrl_cs and np.mean(ctrl_cs) > 0:
                R_str = f"{best_c / np.mean(ctrl_cs):.1f}x"

        verdict = ""
        if abs(best_peak - 0.50) < 0.03:
            verdict = "PEAK AT σ=0.50"
        elif best_c > 0.05:
            verdict = f"peak drifted to σ={best_peak:.2f}"
        else:
            verdict = "monotonic (needs more primes)"

        print(f"  {K:5d} {n_primes:6d} {best_eps:5.1f} {best_peak:7.3f} {best_c:9.4f} {R_str:>8s}  {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 distributed results")
    parser.add_argument("files", nargs="+", help="JSON result files")
    parser.add_argument("--save-summary", type=str, default=None,
                        help="Save merged summary to JSON")
    args = parser.parse_args()

    all_data = load_results(args.files)
    print(f"Loaded results for K values: {sorted(all_data.keys())}")

    for K in sorted(all_data.keys()):
        analyze_k(K, all_data[K])

    if len(all_data) > 1:
        summary_table(all_data)

    if args.save_summary:
        with open(args.save_summary, "w") as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in all_data.items()}, f, indent=2)
        print(f"\nMerged results saved to {args.save_summary}")


if __name__ == "__main__":
    main()
