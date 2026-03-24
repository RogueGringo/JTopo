#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""Validate matrix-free sheaf Laplacian against known K=200 results.

Compares MatFreeSheafLaplacian eigenvalues against TorchSheafLaplacian
(the dense CSR version) at K=200, sigma=0.5, eps=3.0 — where we have
verified results (S=11.784063).

Then runs K=400 (which OOMs on the dense version) to prove it works.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian


def main():
    print("=" * 70)
    print("  MATRIX-FREE VALIDATION")
    print("=" * 70)

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(1000)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    print(f"  Zeros: N={len(zeta_zeros)}")

    # ── TEST 1: K=200 — compare dense vs matrix-free ──
    print(f"\n{'='*70}")
    print("  TEST 1: K=200 Dense vs Matrix-Free")
    print(f"{'='*70}")

    K = 200
    sigma = 0.5
    eps = 3.0
    k_eig = 20
    known_S = 11.784063  # from verified K=200 results

    builder = TransportMapBuilder(K=K, sigma=sigma)

    # Dense (existing method)
    print("\n  [Dense CSR]")
    t0 = time.time()
    dense_lap = TorchSheafLaplacian(builder, zeta_zeros, transport_mode="superposition")
    dense_eigs = dense_lap.smallest_eigenvalues(eps, k=k_eig)
    dense_S = float(np.sum(dense_eigs))
    dense_time = time.time() - t0
    print(f"    S = {dense_S:.6f} ({dense_time:.1f}s)")
    print(f"    Known S = {known_S:.6f}")
    print(f"    Match: {abs(dense_S - known_S) < 0.001}")

    # Cleanup dense to free VRAM
    del dense_lap
    gc.collect()
    torch.cuda.empty_cache()

    # Matrix-free
    print("\n  [Matrix-Free]")
    t0 = time.time()
    mf_lap = MatFreeSheafLaplacian(builder, zeta_zeros, transport_mode="superposition")
    mf_eigs = mf_lap.smallest_eigenvalues(eps, k=k_eig)
    mf_S = float(np.sum(mf_eigs))
    mf_time = time.time() - t0
    print(f"    S = {mf_S:.6f} ({mf_time:.1f}s)")

    # Compare
    diff_S = abs(dense_S - mf_S)
    rel_diff = diff_S / dense_S * 100 if dense_S > 0 else 0
    print(f"\n  COMPARISON:")
    print(f"    Dense S:       {dense_S:.10f}")
    print(f"    Matrix-free S: {mf_S:.10f}")
    print(f"    Abs diff:      {diff_S:.2e}")
    print(f"    Rel diff:      {rel_diff:.4f}%")

    # Eigenvalue-by-eigenvalue comparison
    print(f"\n    Top-5 eigenvalue comparison:")
    for i in range(min(5, len(dense_eigs))):
        d = dense_eigs[i]
        m = mf_eigs[i]
        print(f"      λ{i+1}: dense={d:.10f}  mf={m:.10f}  diff={abs(d-m):.2e}")

    # Verdict
    if rel_diff < 0.1:
        print(f"\n  ✓ VALIDATED — matrix-free matches dense to {rel_diff:.4f}%")
        validated = True
    elif rel_diff < 1.0:
        print(f"\n  ~ CLOSE — {rel_diff:.2f}% relative difference (Lanczos convergence)")
        validated = True
    else:
        print(f"\n  ✗ MISMATCH — {rel_diff:.2f}% relative difference")
        validated = False

    # Cleanup
    del mf_lap
    gc.collect()
    torch.cuda.empty_cache()

    if not validated:
        print("\n  Aborting K=400 — validation failed.")
        return

    # ── TEST 2: K=400 — the whole point ──
    print(f"\n{'='*70}")
    print("  TEST 2: K=400 Matrix-Free (would OOM on dense)")
    print(f"{'='*70}")

    K = 400
    builder400 = TransportMapBuilder(K=K, sigma=0.5)

    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  VRAM free: {free:.1f} GB")

    print(f"\n  Running sigma=0.5...")
    t0 = time.time()
    mf400 = MatFreeSheafLaplacian(builder400, zeta_zeros, transport_mode="superposition")
    eigs400 = mf400.smallest_eigenvalues(eps, k=k_eig)
    S400 = float(np.sum(eigs400))
    elapsed = time.time() - t0

    print(f"    K=400 S(0.5) = {S400:.6f} ({elapsed:.1f}s)")
    print(f"    Top-5 eigs: {eigs400[:5]}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"    VRAM: {alloc:.2f} GB alloc, {free:.1f} GB free")

    # K=200 vs K=400 premium comparison
    print(f"\n  K COMPARISON at sigma=0.5:")
    print(f"    K=200: S = {known_S:.6f}")
    print(f"    K=400: S = {S400:.6f}")
    print(f"    Change: {(S400 - known_S)/known_S * 100:+.2f}%")

    # Cleanup and run GUE for K=400
    del mf400
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  Running K=400 GUE control...")
    rng = np.random.default_rng(42)
    mean_sp = float(np.mean(np.diff(np.sort(zeta_zeros))))

    # Wigner surmise GUE (same as phase3d for comparability)
    spacings = []
    for _ in range(999):
        while True:
            s = rng.rayleigh(scale=np.sqrt(np.pi / 8))
            target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            sigma2 = np.pi / 8
            proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
            if proposal > 0 and rng.random() < target / (proposal * 2.0):
                spacings.append(s * mean_sp)
                break
    gue_pts = np.cumsum(np.array([float(zeta_zeros.min())] + spacings))

    t0 = time.time()
    builder400g = TransportMapBuilder(K=400, sigma=0.5)
    mf400g = MatFreeSheafLaplacian(builder400g, gue_pts, transport_mode="superposition")
    eigs400g = mf400g.smallest_eigenvalues(eps, k=k_eig)
    S400g = float(np.sum(eigs400g))
    elapsed = time.time() - t0
    print(f"    K=400 GUE S(0.5) = {S400g:.6f} ({elapsed:.1f}s)")

    premium = (1 - S400 / S400g) * 100
    print(f"\n  K=400 ARITHMETIC PREMIUM: {premium:.1f}%")
    print(f"    (Physicist predicted ~27.7%)")

    # Run Random for K=400
    del mf400g
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  Running K=400 Random control...")
    rand_pts = np.sort(rng.uniform(zeta_zeros.min(), zeta_zeros.max(), 1000))
    t0 = time.time()
    builder400r = TransportMapBuilder(K=400, sigma=0.5)
    mf400r = MatFreeSheafLaplacian(builder400r, rand_pts, transport_mode="superposition")
    eigs400r = mf400r.smallest_eigenvalues(eps, k=k_eig)
    S400r = float(np.sum(eigs400r))
    elapsed = time.time() - t0
    print(f"    K=400 Random S(0.5) = {S400r:.6f} ({elapsed:.1f}s)")

    print(f"\n{'='*70}")
    print(f"  K=400 HIERARCHY at sigma=0.5:")
    print(f"    S(Zeta)  = {S400:.6f}")
    print(f"    S(GUE)   = {S400g:.6f}")
    print(f"    S(Random) = {S400r:.6f}")
    print(f"    Premium (over GUE): {premium:.1f}%")
    print(f"    Hierarchy holds: {S400 < S400g < S400r}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
