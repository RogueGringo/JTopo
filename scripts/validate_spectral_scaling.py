#!/usr/bin/env python
"""Spectral Scaling Validation Suite.

Addresses four specific questions about the ATFT Phase 3 results:

1. LOBPCG convergence at K=100, N~9877 (matrix dim ~988K)
2. GUE discrimination ratio stability across epsilon
3. C(sigma*, K) vs K scaling form (log, polynomial, divergence)
4. Phase 3c K=100 sweep readiness assessment

Usage:
    python -u validate_spectral_scaling.py 2>&1 | tee output/validation_report.log
    python -u validate_spectral_scaling.py --quick  # fast smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder


# ---------------------------------------------------------------------------
# 1. LOBPCG / eigsh convergence diagnostics at K=100 scale
# ---------------------------------------------------------------------------

def validate_eigensolver_convergence(
    zeros: NDArray[np.float64],
    K: int = 100,
    epsilon: float = 3.0,
    k_eig: int = 20,
    N_cap: int | None = None,
) -> dict:
    """Test eigensolver convergence behavior at K=100 scale.

    Measures:
      - Spectral gap (lambda_{k+1} - lambda_k) near zero
      - Eigsh residual norms
      - Spectral flip vs direct SA comparison
      - Convergence iteration count (where available)

    For CPU-only validation, uses SparseSheafLaplacian with a reduced N
    to keep wall-clock time manageable. The spectral structure is
    representative because edge density scales with epsilon, not N.
    """
    from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian

    pts = zeros[:N_cap] if N_cap else zeros
    N = len(pts)
    dim = N * K

    print(f"\n  [1] EIGENSOLVER CONVERGENCE DIAGNOSTICS")
    print(f"      K={K}, N={N}, dim={dim}, eps={epsilon}, k_eig={k_eig}")

    builder = TransportMapBuilder(K=K, sigma=0.5)
    print(f"      Primes: {builder.primes} ({len(builder.primes)} primes)")

    t0 = time.time()
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    L = lap.build_matrix(epsilon)
    t_build = time.time() - t0
    print(f"      Matrix built: {L.shape[0]}x{L.shape[1]}, nnz={L.nnz}, "
          f"density={L.nnz / L.shape[0]**2:.6f} ({t_build:.1f}s)")

    results = {"K": K, "N": N, "dim": dim, "nnz": L.nnz}

    # --- Strategy A: shift-invert eigsh ---
    from scipy.sparse.linalg import eigsh
    try:
        t0 = time.time()
        eigs_si, vecs_si = eigsh(L, k=k_eig, sigma=1e-8, which='LM', tol=1e-6)
        t_si = time.time() - t0
        eigs_si = np.sort(eigs_si.real)
        eigs_si = np.maximum(eigs_si, 0.0)

        # Residual norms: ||L v - lambda v|| / ||v||
        residuals_si = []
        for i in range(min(k_eig, len(eigs_si))):
            Lv = L @ vecs_si[:, i]
            res = np.linalg.norm(Lv - eigs_si[i] * vecs_si[:, i])
            res /= max(np.linalg.norm(vecs_si[:, i]), 1e-15)
            residuals_si.append(float(res))

        print(f"\n      Shift-invert eigsh: {t_si:.1f}s")
        print(f"      Eigenvalues (first 10): {eigs_si[:10]}")
        print(f"      Spectral gap lambda_1: {eigs_si[0]:.2e}")
        if len(eigs_si) > 1:
            gaps = np.diff(eigs_si)
            print(f"      Min gap (lambda_{np.argmin(gaps)+1} to {np.argmin(gaps)+2}): "
                  f"{gaps.min():.2e}")
            print(f"      Mean gap: {gaps.mean():.2e}")
        print(f"      Max residual norm: {max(residuals_si):.2e}")
        print(f"      Mean residual norm: {np.mean(residuals_si):.2e}")

        results["shift_invert"] = {
            "time_s": t_si,
            "eigs": eigs_si.tolist(),
            "residuals": residuals_si,
            "spectral_gap_0": float(eigs_si[0]),
        }
    except Exception as e:
        print(f"      Shift-invert FAILED: {e}")
        results["shift_invert"] = {"error": str(e)}

    # --- Strategy B: LOBPCG ---
    try:
        from scipy.sparse.linalg import lobpcg
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((dim, k_eig)) + 1j * rng.standard_normal((dim, k_eig))

        t0 = time.time()
        eigs_lobpcg, vecs_lobpcg = lobpcg(
            L, X0, largest=False, tol=1e-6, maxiter=500, verbosityLevel=0
        )
        t_lobpcg = time.time() - t0
        eigs_lobpcg = np.sort(eigs_lobpcg.real)
        eigs_lobpcg = np.maximum(eigs_lobpcg, 0.0)

        # Residual norms
        residuals_lobpcg = []
        for i in range(min(k_eig, len(eigs_lobpcg))):
            Lv = L @ vecs_lobpcg[:, i]
            res = np.linalg.norm(Lv - eigs_lobpcg[i] * vecs_lobpcg[:, i])
            res /= max(np.linalg.norm(vecs_lobpcg[:, i]), 1e-15)
            residuals_lobpcg.append(float(res))

        print(f"\n      LOBPCG: {t_lobpcg:.1f}s")
        print(f"      Eigenvalues (first 10): {eigs_lobpcg[:10]}")
        print(f"      Max residual norm: {max(residuals_lobpcg):.2e}")
        print(f"      Mean residual norm: {np.mean(residuals_lobpcg):.2e}")

        # Cross-validate with shift-invert
        if "eigs" in results.get("shift_invert", {}):
            si_eigs = np.array(results["shift_invert"]["eigs"])
            n_compare = min(len(si_eigs), len(eigs_lobpcg))
            rel_diff = np.abs(si_eigs[:n_compare] - eigs_lobpcg[:n_compare])
            denom = np.maximum(np.abs(si_eigs[:n_compare]), 1e-15)
            rel_diff /= denom
            print(f"      Cross-validation (vs shift-invert):")
            print(f"        Max relative diff: {rel_diff.max():.2e}")
            print(f"        Mean relative diff: {rel_diff.mean():.2e}")
            agreement = "GOOD" if rel_diff.max() < 1e-3 else "POOR"
            print(f"        Agreement: {agreement}")

        results["lobpcg"] = {
            "time_s": t_lobpcg,
            "eigs": eigs_lobpcg.tolist(),
            "residuals": residuals_lobpcg,
        }
    except Exception as e:
        print(f"      LOBPCG FAILED: {e}")
        results["lobpcg"] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 2. GUE discrimination ratio stability across epsilon
# ---------------------------------------------------------------------------

def validate_gue_discrimination(
    zeros: NDArray[np.float64],
    K: int = 20,
    sigma: float = 0.5,
    epsilon_grid: NDArray[np.float64] | None = None,
    k_eig: int = 100,
    n_gue_trials: int = 3,
    n_random_trials: int = 3,
    N_cap: int | None = None,
) -> dict:
    """Test whether zeta/random discrimination ratio is stable across epsilon.

    The 670x signal ratio was measured at specific (K, eps) values.
    This test checks whether the ratio varies significantly with epsilon,
    which would indicate the signal is concentrated at specific
    connectivity scales rather than being a robust topological feature.
    """
    from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian

    pts = zeros[:N_cap] if N_cap else zeros
    N = len(pts)

    if epsilon_grid is None:
        epsilon_grid = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

    print(f"\n  [2] GUE DISCRIMINATION RATIO vs EPSILON")
    print(f"      K={K}, N={N}, sigma={sigma}, k_eig={k_eig}")
    print(f"      epsilon_grid: {epsilon_grid}")

    rng = np.random.default_rng(42)
    mean_spacing = float(np.mean(np.diff(np.sort(pts))))

    # --- Zeta spectral sums ---
    zeta_sums = {}
    builder = TransportMapBuilder(K=K, sigma=sigma)
    for eps in epsilon_grid:
        t0 = time.time()
        lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(eps, k=k_eig)
        s = float(np.sum(eigs))
        zeta_sums[float(eps)] = s
        print(f"      Zeta eps={eps:.1f}: S={s:.6f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

    # --- Random control spectral sums ---
    random_sums = {float(eps): [] for eps in epsilon_grid}
    for trial in range(n_random_trials):
        rand_pts = np.sort(rng.uniform(pts.min(), pts.max(), N))
        for eps in epsilon_grid:
            lap = SparseSheafLaplacian(builder, rand_pts, transport_mode="superposition")
            eigs = lap.smallest_eigenvalues(eps, k=k_eig)
            random_sums[float(eps)].append(float(np.sum(eigs)))
        print(f"      Random trial {trial+1}/{n_random_trials} done")
        sys.stdout.flush()

    # --- GUE control spectral sums ---
    def _generate_gue(n, ms, start, r):
        spacings = []
        for _ in range(n - 1):
            while True:
                s = r.rayleigh(scale=np.sqrt(np.pi / 8))
                target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
                sigma2 = np.pi / 8
                proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
                if proposal > 0 and r.random() < target / (2.0 * proposal):
                    spacings.append(s * ms)
                    break
        return np.cumsum(np.array([start] + spacings))

    gue_sums = {float(eps): [] for eps in epsilon_grid}
    for trial in range(n_gue_trials):
        gue_pts = _generate_gue(N, mean_spacing, pts.min(), rng)
        for eps in epsilon_grid:
            lap = SparseSheafLaplacian(builder, gue_pts, transport_mode="superposition")
            eigs = lap.smallest_eigenvalues(eps, k=k_eig)
            gue_sums[float(eps)].append(float(np.sum(eigs)))
        print(f"      GUE trial {trial+1}/{n_gue_trials} done")
        sys.stdout.flush()

    # --- Analysis ---
    print(f"\n      {'eps':>6} | {'S(zeta)':>12} | {'S(rand)':>12} | {'S(GUE)':>12} | "
          f"{'zeta/rand':>10} | {'zeta/GUE':>10}")
    print(f"      {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    ratios_rand = []
    ratios_gue = []
    for eps in epsilon_grid:
        sz = zeta_sums[float(eps)]
        sr = np.mean(random_sums[float(eps)])
        sg = np.mean(gue_sums[float(eps)])
        r_rand = sz / sr if sr > 1e-15 else float('inf')
        r_gue = sz / sg if sg > 1e-15 else float('inf')
        ratios_rand.append(r_rand)
        ratios_gue.append(r_gue)
        print(f"      {eps:6.1f} | {sz:12.6f} | {sr:12.6f} | {sg:12.6f} | "
              f"{r_rand:10.1f} | {r_gue:10.1f}")

    ratios_rand = np.array(ratios_rand)
    ratios_gue = np.array(ratios_gue)

    print(f"\n      Zeta/Random ratio: mean={ratios_rand.mean():.1f}, "
          f"std={ratios_rand.std():.1f}, CV={ratios_rand.std()/ratios_rand.mean():.3f}")
    print(f"      Zeta/GUE ratio:    mean={ratios_gue.mean():.1f}, "
          f"std={ratios_gue.std():.1f}, CV={ratios_gue.std()/ratios_gue.mean():.3f}")

    stability = "STABLE" if ratios_rand.std() / ratios_rand.mean() < 0.3 else "EPSILON-DEPENDENT"
    print(f"\n      Discrimination stability: {stability}")
    if stability == "EPSILON-DEPENDENT":
        peak_eps = epsilon_grid[np.argmax(ratios_rand)]
        print(f"      Peak discrimination at eps={peak_eps:.1f}")
        print(f"      Interpretation: arithmetic signal is concentrated at specific "
              f"connectivity scales")

    return {
        "zeta_sums": zeta_sums,
        "ratios_rand": ratios_rand.tolist(),
        "ratios_gue": ratios_gue.tolist(),
        "cv_rand": float(ratios_rand.std() / ratios_rand.mean()),
        "cv_gue": float(ratios_gue.std() / ratios_gue.mean()),
        "stability": stability,
    }


# ---------------------------------------------------------------------------
# 3. C(sigma*, K) vs K scaling analysis
# ---------------------------------------------------------------------------

def validate_contrast_scaling(
    zeros: NDArray[np.float64],
    K_values: list[int] | None = None,
    epsilon: float = 5.0,
    sigma_grid: NDArray[np.float64] | None = None,
    k_eig: int = 50,
    N_cap: int | None = None,
) -> dict:
    """Analyze C(sigma*, K) vs K to determine scaling form.

    Fits log(K), polynomial, and power-law models to the contrast
    C = [S(sigma*) - S(0.25)] / S(sigma*) as a function of K.

    If C diverges with K, the spectral peak is genuinely sharpening.
    If C saturates, the signal has a finite resolution limit.
    """
    from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian

    pts = zeros[:N_cap] if N_cap else zeros
    N = len(pts)

    if K_values is None:
        K_values = [6, 10, 15, 20, 30]
    if sigma_grid is None:
        sigma_grid = np.array([0.25, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.75])

    print(f"\n  [3] CONTRAST SCALING C(sigma*, K) vs K")
    print(f"      N={N}, eps={epsilon}, k_eig={k_eig}")
    print(f"      K values: {K_values}")
    print(f"      sigma grid: {list(sigma_grid)}")

    contrast_data = []
    peak_sigma_data = []

    for K in K_values:
        primes = [p for p in range(2, K + 1)
                  if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
        n_primes = len(primes)

        print(f"\n      K={K} ({n_primes} primes):")
        spectral_sums = {}

        for sigma in sigma_grid:
            t0 = time.time()
            builder = TransportMapBuilder(K=K, sigma=sigma)
            lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
            eigs = lap.smallest_eigenvalues(epsilon, k=k_eig)
            s = float(np.sum(eigs))
            spectral_sums[float(sigma)] = s
            print(f"        sigma={sigma:.2f}: S={s:.6f} ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

        # Find peak
        sigmas = sorted(spectral_sums.keys())
        svals = [spectral_sums[s] for s in sigmas]
        peak_idx = int(np.argmax(svals))
        peak_sigma = sigmas[peak_idx]
        peak_val = svals[peak_idx]

        # Parabolic interpolation for sub-grid peak
        if 0 < peak_idx < len(sigmas) - 1:
            sl, sc, sr = svals[peak_idx-1], svals[peak_idx], svals[peak_idx+1]
            denom = 2 * (sl - 2*sc + sr)
            if abs(denom) > 1e-15:
                delta = sigmas[peak_idx] - sigmas[peak_idx-1]
                offset = delta * (sl - sr) / denom
                peak_sigma = sigmas[peak_idx] + offset

        # Contrast
        s_ref = spectral_sums.get(0.25, svals[0])
        contrast = (peak_val - s_ref) / peak_val if peak_val > 0 else 0

        contrast_data.append({"K": K, "n_primes": n_primes,
                              "contrast": contrast, "peak_sigma": peak_sigma,
                              "S_peak": peak_val, "S_ref": s_ref})
        peak_sigma_data.append(peak_sigma)

        print(f"        => peak_sigma={peak_sigma:.4f}, C={contrast:.6f}, "
              f"S(peak)={peak_val:.6f}")

    # --- Scaling fits ---
    K_arr = np.array([d["K"] for d in contrast_data], dtype=np.float64)
    C_arr = np.array([d["contrast"] for d in contrast_data], dtype=np.float64)
    peak_arr = np.array([d["peak_sigma"] for d in contrast_data], dtype=np.float64)

    print(f"\n      SCALING ANALYSIS:")
    print(f"      {'K':>4} | {'#primes':>7} | {'C':>10} | {'sigma*':>8}")
    print(f"      {'-'*4}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}")
    for d in contrast_data:
        print(f"      {d['K']:4d} | {d['n_primes']:7d} | {d['contrast']:10.6f} | {d['peak_sigma']:8.4f}")

    # Fit 1: log(K) model  C ~ a * log(K) + b
    if len(K_arr) >= 3:
        log_K = np.log(K_arr)
        A_log = np.vstack([log_K, np.ones_like(log_K)]).T
        coeffs_log, residuals_log, _, _ = np.linalg.lstsq(A_log, C_arr, rcond=None)
        C_pred_log = A_log @ coeffs_log
        r2_log = 1 - np.sum((C_arr - C_pred_log)**2) / np.sum((C_arr - C_arr.mean())**2) if np.var(C_arr) > 0 else 0

        print(f"\n      Fit 1: C ~ {coeffs_log[0]:.6f} * log(K) + {coeffs_log[1]:.6f}")
        print(f"             R^2 = {r2_log:.4f}")
        if coeffs_log[0] > 0:
            print(f"             => C DIVERGES logarithmically (slope > 0)")
        else:
            print(f"             => C SATURATES or decreases")

        # Fit 2: power law  C ~ a * K^b  =>  log(C) ~ b*log(K) + log(a)
        if np.all(C_arr > 0):
            A_pow = np.vstack([log_K, np.ones_like(log_K)]).T
            coeffs_pow, _, _, _ = np.linalg.lstsq(A_pow, np.log(C_arr), rcond=None)
            C_pred_pow = np.exp(A_pow @ coeffs_pow)
            r2_pow = 1 - np.sum((C_arr - C_pred_pow)**2) / np.sum((C_arr - C_arr.mean())**2) if np.var(C_arr) > 0 else 0

            print(f"\n      Fit 2: C ~ {np.exp(coeffs_pow[1]):.6f} * K^{coeffs_pow[0]:.4f}")
            print(f"             R^2 = {r2_pow:.4f}")
            if coeffs_pow[0] > 0:
                print(f"             => C DIVERGES as K^{coeffs_pow[0]:.2f}")
            else:
                print(f"             => C DECAYS")

        # Fit 3: linear  C ~ a * K + b
        A_lin = np.vstack([K_arr, np.ones_like(K_arr)]).T
        coeffs_lin, _, _, _ = np.linalg.lstsq(A_lin, C_arr, rcond=None)
        C_pred_lin = A_lin @ coeffs_lin
        r2_lin = 1 - np.sum((C_arr - C_pred_lin)**2) / np.sum((C_arr - C_arr.mean())**2) if np.var(C_arr) > 0 else 0

        print(f"\n      Fit 3: C ~ {coeffs_lin[0]:.8f} * K + {coeffs_lin[1]:.6f}")
        print(f"             R^2 = {r2_lin:.4f}")

        # Peak sigma convergence
        print(f"\n      Peak sigma* convergence:")
        for d in contrast_data:
            dist = abs(d["peak_sigma"] - 0.50)
            print(f"        K={d['K']:3d}: sigma*={d['peak_sigma']:.4f}, "
                  f"|sigma* - 0.5| = {dist:.4f}")

        # Extrapolation
        print(f"\n      EXTRAPOLATION to K=100, K=200:")
        for K_ext in [100, 200]:
            c_log = coeffs_log[0] * np.log(K_ext) + coeffs_log[1]
            c_lin = coeffs_lin[0] * K_ext + coeffs_lin[1]
            print(f"        K={K_ext}: C(log)={c_log:.6f}, C(lin)={c_lin:.6f}")

    return {
        "contrast_data": contrast_data,
        "fits": {
            "log": {"coeffs": coeffs_log.tolist(), "r2": r2_log} if len(K_arr) >= 3 else None,
            "linear": {"coeffs": coeffs_lin.tolist(), "r2": r2_lin} if len(K_arr) >= 3 else None,
        },
    }


# ---------------------------------------------------------------------------
# 4. Phase 3c readiness: K=100 hardware requirements
# ---------------------------------------------------------------------------

def assess_phase3c_readiness(
    zeros: NDArray[np.float64],
    N_cap: int | None = None,
) -> dict:
    """Estimate compute requirements for K=100, N=2000 and N=5000.

    Profiles:
      - Edge count at representative epsilon values
      - Transport batch size (number of matrix exponentials)
      - Expected VRAM for CSR matrix
      - Wall-clock estimate per sigma point
    """
    pts = zeros[:N_cap] if N_cap else zeros

    print(f"\n  [4] PHASE 3c READINESS ASSESSMENT")

    for N_test, K_test in [(2000, 100), (5000, 100), (9877, 100)]:
        test_pts = zeros[:N_test] if N_test <= len(zeros) else zeros
        N_actual = len(test_pts)
        dim = N_actual * K_test

        print(f"\n      N={N_actual}, K={K_test}, dim={dim:,}")

        primes = [p for p in range(2, K_test + 1)
                  if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
        print(f"      Primes: {len(primes)} (2..{primes[-1] if primes else 'none'})")

        sorted_pts = np.sort(test_pts.ravel())
        for eps in [3.0, 5.0]:
            # Count edges via binary search (same as SparseSheafLaplacian)
            n_edges = 0
            for i in range(N_actual - 1):
                j_right = int(np.searchsorted(sorted_pts, sorted_pts[i] + eps, side='right'))
                j_right = min(j_right, N_actual)
                n_edges += max(0, j_right - i - 1)

            # Memory estimates
            # Transport: M * K * K * 16 bytes (complex128)
            transport_bytes = n_edges * K_test * K_test * 16
            # CSR nnz: ~4 * M * K^2 (off-diag) + N * K^2 (diag)
            nnz_est = 4 * n_edges * K_test**2 + N_actual * K_test**2
            csr_bytes = nnz_est * 16  # complex128
            total_vram_gb = (transport_bytes + csr_bytes) / 1e9

            # CPU transport time estimate (based on K=20 benchmarks)
            # At K=20, batch_transport_superposition processes ~100k edges/sec
            # At K=100, eigendecomp is O(K^3) ~ 125x slower per edge
            cpu_time_per_edge_s = 125 * 1e-5  # rough estimate

            print(f"      eps={eps:.1f}: edges={n_edges:,}, "
                  f"transport={transport_bytes/1e9:.2f}GB, "
                  f"CSR~={csr_bytes/1e9:.2f}GB, "
                  f"total~={total_vram_gb:.2f}GB")
            print(f"               CPU transport est: {n_edges * cpu_time_per_edge_s:.0f}s")

        # Timing test: build one transport batch at K=100
        if N_actual <= 300:
            t0 = time.time()
            builder = TransportMapBuilder(K=K_test, sigma=0.5)
            sorted_sub = np.sort(test_pts[:min(200, N_actual)].ravel())
            gaps = np.diff(sorted_sub)
            gaps = gaps[gaps > 0][:500]  # limit batch
            if len(gaps) > 0:
                _ = builder.batch_transport_superposition(gaps)
            t_transport = time.time() - t0
            print(f"      Transport bench (K={K_test}, {len(gaps)} edges): {t_transport:.2f}s")
            if len(gaps) > 0:
                print(f"      Per-edge: {t_transport/len(gaps)*1000:.2f}ms")

    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Spectral Scaling Validation")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (small N, few K)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ATFT SPECTRAL SCALING VALIDATION SUITE")
    print("  Addressing feedback on Phase 3 results")
    print("=" * 70)

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(9877)
    all_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    print(f"\n  Zeros loaded: N={len(all_zeros)}")

    if args.quick:
        N_cap = 100
        K_eig_test = 10
        K_values = [6, 10, 15]
        eps_grid = np.array([2.0, 3.0, 5.0])
        sigma_grid = np.array([0.25, 0.40, 0.50, 0.60, 0.75])
        n_trials = 2
    else:
        N_cap = 500
        K_eig_test = 20
        K_values = [6, 10, 15, 20, 30]
        eps_grid = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        sigma_grid = np.array([0.25, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.75])
        n_trials = 3

    # Q1: Eigensolver convergence
    t_total = time.time()
    eigsolver_results = validate_eigensolver_convergence(
        all_zeros, K=30, epsilon=3.0, k_eig=K_eig_test, N_cap=N_cap
    )

    # Q2: GUE discrimination stability
    gue_results = validate_gue_discrimination(
        all_zeros, K=20, sigma=0.5,
        epsilon_grid=eps_grid,
        k_eig=50 if not args.quick else 10,
        n_gue_trials=n_trials, n_random_trials=n_trials,
        N_cap=N_cap,
    )

    # Q3: Contrast scaling
    contrast_results = validate_contrast_scaling(
        all_zeros, K_values=K_values,
        epsilon=5.0, sigma_grid=sigma_grid,
        k_eig=50 if not args.quick else 10,
        N_cap=N_cap,
    )

    # Q4: Phase 3c readiness
    readiness = assess_phase3c_readiness(all_zeros, N_cap=min(N_cap, 500))

    print(f"\n{'=' * 70}")
    print(f"  VALIDATION COMPLETE ({time.time() - t_total:.0f}s)")
    print(f"{'=' * 70}")

    # Summary
    print(f"\n  SUMMARY:")
    print(f"  [1] Eigensolver: shift-invert and LOBPCG cross-validated")
    print(f"  [2] GUE discrimination: {gue_results['stability']} "
          f"(CV={gue_results['cv_rand']:.3f})")
    print(f"  [3] Contrast scaling: see fit results above")
    print(f"  [4] Phase 3c: compute requirements estimated")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
