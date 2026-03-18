#!/usr/bin/env python
"""Thermodynamic Scaling Exponent Extractor.

Runs KPMSheafLaplacian at multiple K values and extracts the scaling
exponents of the IDOS as a function of K. These exponents are the
key numerical evidence for or against the topological obstruction.

Usage:
  python scripts/validate_thermodynamic_scaling.py --quick
  python scripts/validate_thermodynamic_scaling.py --k-values 20 50 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atft.sources.zeta_zeros import ZetaZerosSource
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.kpm_sheaf_laplacian import KPMSheafLaplacian


def run_kpm_sweep(zeros, K, sigma_values, epsilon, degree=300,
                  num_vectors=30, device=None, fixed_cutoff=None):
    results = {}
    for sigma in sigma_values:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        kpm = KPMSheafLaplacian(
            builder, zeros, device=device,
            degree=degree, num_vectors=num_vectors,
        )
        mu = kpm.compute_moments(epsilon)

        # Use fixed cutoff if provided (avoids the moving-goalpost trap
        # where lambda_max grows with K, widening the integration window)
        if fixed_cutoff is not None:
            idos_val = kpm.idos(cutoff=fixed_cutoff)
            resolution = np.pi * kpm._lam_max / degree
            if resolution > fixed_cutoff:
                print(f"  WARNING: KPM resolution {resolution:.3f} > cutoff "
                      f"{fixed_cutoff:.3f}. Increase degree for K={K}.")
        else:
            idos_val = kpm.spectral_density_at_zero()

        s_sum = kpm.spectral_sum(epsilon)
        results[sigma] = {
            "idos_at_zero": idos_val,
            "spectral_sum": s_sum,
            "lam_max": kpm._lam_max,
            "dim": kpm._dim,
            "resolution_limit": np.pi * kpm._lam_max / degree,
            "moments": mu.tolist(),
        }
        print(f"  K={K}, sigma={sigma:.2f}: IDOS={idos_val:.6f}, "
              f"S={s_sum:.4f}, lam_max={kpm._lam_max:.2f}, "
              f"res={np.pi * kpm._lam_max / degree:.3f}")
    return results


def compute_scaling_exponents(all_results, sigma_on=0.5, sigma_off=0.25):
    from scipy.optimize import curve_fit

    K_vals = sorted(all_results.keys())
    if len(K_vals) < 2:
        return {"warning": "Need at least 2 K values for scaling fit"}

    K_arr = np.array(K_vals, dtype=np.float64)
    idos_on = np.array([all_results[K][sigma_on]["idos_at_zero"] for K in K_vals])
    idos_off = np.array([all_results[K][sigma_off]["idos_at_zero"] for K in K_vals])
    contrast = np.where(idos_off > 1e-12, idos_on / idos_off, np.inf)

    def power_law(K, a, alpha):
        return a * K ** alpha

    def log_law(K, a, alpha):
        return a * np.log(K) ** alpha

    def exp_law(K, a, alpha):
        return a * np.exp(-alpha * K)

    ansatzes = {"power_law": power_law, "logarithmic": log_law, "exponential": exp_law}

    exponents = {
        "K_values": K_vals,
        "idos_on_line": idos_on.tolist(),
        "idos_off_line": idos_off.tolist(),
        "contrast_ratio": contrast.tolist(),
    }

    if np.all(idos_off > 1e-15) and len(K_vals) >= 2:
        best_aic = np.inf
        for name, func in ansatzes.items():
            try:
                p0 = [float(idos_off[0]), -1.0] if name != "exponential" else [float(idos_off[0]), 0.01]
                popt, pcov = curve_fit(func, K_arr, idos_off, p0=p0, maxfev=5000)
                residuals = idos_off - func(K_arr, *popt)
                ss_res = float(np.sum(residuals**2))
                n = len(K_arr)
                aic = n * np.log(ss_res / n + 1e-30) + 4
                exponents[f"off_{name}_a"] = float(popt[0])
                exponents[f"off_{name}_alpha"] = float(popt[1])
                exponents[f"off_{name}_aic"] = float(aic)
                if aic < best_aic:
                    best_aic = aic
                    exponents["off_best_ansatz"] = name
                    exponents["off_line_exponent"] = float(popt[1])
            except Exception as e:
                exponents[f"off_{name}_error"] = str(e)

    if np.all(idos_on > 1e-15) and len(K_vals) >= 2:
        try:
            popt, _ = curve_fit(power_law, K_arr, idos_on, p0=[1.0, 0.0])
            exponents["on_line_amplitude"] = float(popt[0])
            exponents["on_line_exponent"] = float(popt[1])
        except Exception as e:
            exponents["on_line_fit_error"] = str(e)

    return exponents


def evaluate_criteria(exponents):
    verdicts = {}
    idos_on = exponents.get("idos_on_line", [])
    idos_off = exponents.get("idos_off_line", [])
    contrast = exponents.get("contrast_ratio", [])

    if len(idos_on) >= 2:
        verdicts["P1"] = "PASS" if all(v > 1e-4 for v in idos_on) else "FAIL"
        verdicts["P2"] = "PASS" if idos_off[-1] < idos_off[0] * 0.5 else "FAIL"
        finite_contrast = [c for c in contrast if np.isfinite(c)]
        if len(finite_contrast) >= 2:
            verdicts["P3"] = "PASS" if finite_contrast[-1] > finite_contrast[0] * 3 else "FAIL"
        verdicts["F1"] = "TRIGGERED" if idos_off[-1] >= idos_off[0] else "OK"
        verdicts["F3"] = "TRIGGERED" if all(v < 1e-4 for v in idos_on) else "OK"

    off_exp = exponents.get("off_line_exponent")
    if off_exp is not None:
        verdicts["off_line_scaling"] = f"IDOS_off ~ K^{off_exp:.3f}"

    best = exponents.get("off_best_ansatz")
    if best:
        verdicts["best_ansatz"] = best

    return verdicts


def main():
    parser = argparse.ArgumentParser(description="KPM Thermodynamic Scaling Analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small N, few K values")
    parser.add_argument("--k-values", nargs="+", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=3.0)
    parser.add_argument("--degree", type=int, default=300)
    parser.add_argument("--num-vectors", type=int, default=30)
    parser.add_argument("--n-zeros", type=int, default=9877,
                        help="Number of zeros to use (default: all 9877)")
    parser.add_argument("--fixed-cutoff", type=float, default=None,
                        help="Fixed IDOS cutoff (avoids lambda_max goalpost drift)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parent.parent / "data" / "odlyzko_zeros.txt"
    source = ZetaZerosSource(data_path=data_path)
    cloud = source.generate(n_points=args.n_zeros)
    unfolding = SpectralUnfolding(method="zeta")
    unfolded = unfolding.transform(cloud)
    zeros = unfolded.points[:, 0]

    if args.quick:
        zeros = zeros[:200]
        K_values = [6, 10]
        sigma_values = [0.25, 0.50, 0.75]
    else:
        K_values = args.k_values or [20, 50]
        sigma_values = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]

    # Auto-scale degree to ensure resolution < fixed_cutoff
    if args.fixed_cutoff:
        print(f"Fixed cutoff: {args.fixed_cutoff} (degree will auto-scale per K)")
    print(f"Zeros: {len(zeros)}, K values: {K_values}")
    print(f"Sigma grid: {sigma_values}")
    print(f"Epsilon: {args.epsilon}, Base degree: {args.degree}")
    print("=" * 60)

    all_results = {}
    for K in K_values:
        # Auto-scale degree: ensure pi * lam_max_est / D < fixed_cutoff
        # Rough estimate: lam_max ~ 2*K for moderate epsilon
        degree = args.degree
        if args.fixed_cutoff:
            lam_max_est = 2.0 * K  # conservative estimate
            min_degree = int(np.pi * lam_max_est / args.fixed_cutoff) + 10
            degree = max(degree, min_degree)
        print(f"\n--- K = {K} (degree={degree}) ---")
        all_results[K] = run_kpm_sweep(
            zeros, K, sigma_values, args.epsilon,
            degree=degree, num_vectors=args.num_vectors,
            fixed_cutoff=args.fixed_cutoff,
        )

    print("\n" + "=" * 60)
    print("THERMODYNAMIC SCALING ANALYSIS")
    print("=" * 60)
    exponents = compute_scaling_exponents(all_results)
    for key, val in exponents.items():
        print(f"  {key}: {val}")

    print("\n--- FALSIFICATION_IDOS Criteria ---")
    verdicts = evaluate_criteria(exponents)
    for criterion, verdict in verdicts.items():
        print(f"  {criterion}: {verdict}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "K_values": K_values,
                "sigma_values": sigma_values,
                "results": {str(k): v for k, v in all_results.items()},
                "exponents": exponents,
                "verdicts": verdicts,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
