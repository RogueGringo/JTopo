#!/usr/bin/env python
"""Verify: does the twisted Laplacian spectrum depend on omega?

Mathematical claim: For edge weights e^{i*omega*(gamma_j - gamma_i)},
the holonomy around ANY cycle telescopes to 1:

  gap(1,2) + gap(2,3) + ... + gap(k,1)
    = (g2-g1) + (g3-g2) + ... + (g1-gk) = 0

So e^{i*omega*0} = 1. The connection is FLAT. Therefore L_omega is
spectrally equivalent to L_0 for all omega.

If true, the ATFT sheaf framework CANNOT detect sigma-dependence.
"""
import numpy as np
from atft.experiments.phase2a_abelian import Phase2aAbelian


def verify_spectral_invariance():
    """Check if L_omega eigenvalues are omega-independent."""
    zeros = np.array([0.0, 0.7, 1.2, 2.5, 3.1, 4.8, 5.5, 7.2, 8.0, 9.3])
    N = len(zeros)
    epsilon = 3.0  # Create many edges and cycles

    # Count edges and cycles
    n_edges = 0
    for i in range(N):
        for j in range(i+1, N):
            if zeros[j] - zeros[i] <= epsilon:
                n_edges += 1
    n_cycles = n_edges - (N - 1)
    print(f"N={N}, eps={epsilon}: {n_edges} edges, {n_cycles} independent cycles")

    # Compute spectrum for many different omega values
    omegas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.14159, 5.0, 10.0, 100.0]
    print(f"\nSpectrum of L_omega for various omega:")
    print(f"{'omega':>10} | eigenvalues")
    print(f"{'-'*10}-+-{'-'*60}")

    ref_eigs = None
    all_match = True
    for omega in omegas:
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega, epsilon)
        eigs = np.linalg.eigvalsh(L)
        eig_str = ", ".join(f"{v:.6f}" for v in eigs)
        print(f"{omega:10.4f} | {eig_str}")

        if ref_eigs is None:
            ref_eigs = eigs
        elif not np.allclose(eigs, ref_eigs, atol=1e-10):
            all_match = False
            diff = np.max(np.abs(eigs - ref_eigs))
            print(f"  ^^ DIFFERS from omega=0 by {diff:.2e}")

    print(f"\nAll spectra identical: {all_match}")

    # Verify the mathematical reason: holonomy around cycles
    print(f"\n--- HOLONOMY CHECK ---")
    print("For each cycle (i -> j -> k -> i), check if gaps telescope to 0:")

    for i in range(N):
        for j in range(i+1, N):
            if zeros[j] - zeros[i] > epsilon:
                break
            for k in range(j+1, N):
                if zeros[k] - zeros[j] > epsilon:
                    break
                # Triangle (i,j,k): check if (i,k) is also an edge
                if zeros[k] - zeros[i] <= epsilon:
                    gap_sum = (zeros[j]-zeros[i]) + (zeros[k]-zeros[j]) + (zeros[i]-zeros[k])
                    print(f"  Cycle ({i},{j},{k}): gap_sum = {gap_sum:.10f}")
                    if abs(gap_sum) > 1e-14:
                        print(f"  ^^ NON-ZERO HOLONOMY!")

    print(f"\n--- CONCLUSION ---")
    if all_match:
        print("The twisted Laplacian spectrum is INDEPENDENT of omega.")
        print("Reason: the connection gamma_j - gamma_i is FLAT (holonomy = 0).")
        print("The gap sums around every cycle telescope to zero by construction.")
        print()
        print("CONSEQUENCE: The sheaf Laplacian L_F = direct_sum L_{omega_kl}")
        print("decomposes into K^2 copies of the standard graph Laplacian L_0.")
        print("The spectrum of L_F is K^2 copies of spec(L_0), regardless of sigma.")
        print()
        print("The ATFT framework as designed CANNOT detect sigma-dependence.")
        print("The transport U(delta_gamma) = V*diag(e^{i*dg*lambda_k})*V^dagger")
        print("defines a FLAT connection, making holonomy trivial.")
    else:
        print("UNEXPECTED: Some spectra differ. The connection may not be flat.")


if __name__ == "__main__":
    verify_spectral_invariance()
