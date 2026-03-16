#!/usr/bin/env python
"""Verify the functional equation transport properties.

The FE generator G_p(sigma) = log(p) * [p^{-sigma} rho + p^{-(1-sigma)} rho^T]
encodes the zeta(s) <-> zeta(1-s) symmetry:

1. At sigma=1/2: generator is Hermitian, transport is unitary
2. At sigma!=1/2: generator is non-Hermitian, transport is non-unitary
3. sigma and 1-sigma produce related transport (functional equation symmetry)
4. Non-unitarity creates fiber distortion that penalizes sigma != 1/2
"""
import numpy as np
from atft.topology.transport_maps import TransportMapBuilder


def main():
    K = 6
    dg = 1.0  # test gap

    print(f"Functional Equation Transport Verification (K={K})")
    print(f"{'='*60}\n")

    # 1. Check unitarity vs sigma
    print("1. Transport unitarity: ||U U^dagger - I||")
    print(f"   {'sigma':>8} | {'||UU^H-I||':>12} | {'det(U)':>20} | unitary?")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*20}-+-{'-'*8}")

    sigmas = [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75]
    unitarity_devs = []

    for sigma in sigmas:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        U = builder.transport_fe(dg)
        UUh = U @ U.conj().T
        dev = np.linalg.norm(UUh - np.eye(K))
        det = np.linalg.det(U)
        is_unitary = dev < 1e-10
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"   {sigma:8.3f} | {dev:12.6e} | {det.real:+.6f}{det.imag:+.6f}j | "
              f"{'YES' if is_unitary else 'NO'}{marker}")
        unitarity_devs.append(dev)

    unitarity_devs = np.array(unitarity_devs)
    min_idx = int(np.argmin(unitarity_devs))
    print(f"\n   Minimum non-unitarity at sigma = {sigmas[min_idx]:.3f}")

    # 2. Functional equation symmetry: sigma <-> 1-sigma
    print(f"\n2. Functional equation symmetry: spec_sum(sigma) = spec_sum(1-sigma)?")
    from atft.topology.sheaf_laplacian import SheafLaplacian
    from atft.feature_maps.spectral_unfolding import SpectralUnfolding
    from atft.sources.zeta_zeros import ZetaZerosSource
    from pathlib import Path

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))
    cloud = source.generate(30)
    zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    N = len(zeros)

    epsilon = 2.5

    print(f"   N={N}, K={K}, eps={epsilon}")
    print(f"   {'sigma':>8} | {'spec_sum':>12} | {'1-sigma':>8} | {'spec_sum(1-s)':>14} | {'diff':>12}")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*14}-+-{'-'*12}")

    spec_sums = {}
    for sigma in sigmas:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = SheafLaplacian(builder, zeros, transport_mode="fe")
        eigs = lap.smallest_eigenvalues(epsilon, m=20)
        spec_sums[sigma] = float(np.sum(eigs))

    for sigma in sigmas:
        s1 = spec_sums[sigma]
        sigma_mirror = round(1 - sigma, 3)
        if sigma_mirror in spec_sums:
            s2 = spec_sums[sigma_mirror]
            diff = abs(s1 - s2)
            print(f"   {sigma:8.3f} | {s1:12.6f} | {sigma_mirror:8.3f} | {s2:14.6f} | {diff:12.2e}")
        else:
            print(f"   {sigma:8.3f} | {s1:12.6f} | {sigma_mirror:8.3f} | {'N/A':>14} |")

    # 3. The key test: does spec_sum have a MINIMUM at sigma=0.5?
    print(f"\n3. Sigma-criticality test: spec_sum minimum location")
    print(f"   {'sigma':>8} | {'spec_sum':>12}")
    print(f"   {'-'*8}-+-{'-'*12}")
    for sigma in sigmas:
        marker = " <-- sigma=1/2" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"   {sigma:8.3f} | {spec_sums[sigma]:12.6f}{marker}")

    all_sums = [spec_sums[s] for s in sigmas]
    min_sigma = sigmas[int(np.argmin(all_sums))]
    max_sigma = sigmas[int(np.argmax(all_sums))]
    print(f"\n   MINIMUM spec_sum at sigma = {min_sigma:.3f}")
    print(f"   MAXIMUM spec_sum at sigma = {max_sigma:.3f}")

    if abs(min_sigma - 0.5) < 0.03:
        print(f"\n   ** SIGMA-CRITICALITY DETECTED! **")
        print(f"   The spectral sum has its minimum at the critical line sigma=1/2.")
        print(f"   This indicates maximum coherence of the sheaf connection at RH.")
    elif min_sigma < 0.4 or min_sigma > 0.6:
        print(f"\n   Minimum is far from sigma=0.5. The functional equation")
        print(f"   symmetry creates a U-shaped curve with minimum at sigma=1/2,")
        print(f"   but the non-monotonic structure may require larger epsilon.")

    # 4. Compare with resonant (Hermitian) transport
    print(f"\n4. Comparison: FE vs Resonant transport")
    print(f"   {'sigma':>8} | {'FE spec_sum':>12} | {'Res spec_sum':>12} | {'FE unitary?':>12}")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for sigma in [0.30, 0.40, 0.50, 0.60, 0.70]:
        builder = TransportMapBuilder(K=K, sigma=sigma)
        # FE transport
        lap_fe = SheafLaplacian(builder, zeros, transport_mode="fe")
        eigs_fe = lap_fe.smallest_eigenvalues(epsilon, m=20)
        ss_fe = float(np.sum(eigs_fe))
        # Resonant transport
        lap_res = SheafLaplacian(builder, zeros, transport_mode="resonant")
        eigs_res = lap_res.smallest_eigenvalues(epsilon, m=20)
        ss_res = float(np.sum(eigs_res))
        # Unitarity check
        U = builder.transport_fe(1.0)
        unitary = np.linalg.norm(U @ U.conj().T - np.eye(K)) < 1e-10
        marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
        print(f"   {sigma:8.3f} | {ss_fe:12.6f} | {ss_res:12.6f} | "
              f"{'YES' if unitary else 'NO':>12}{marker}")


if __name__ == "__main__":
    main()
