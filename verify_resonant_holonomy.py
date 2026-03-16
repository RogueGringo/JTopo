#!/usr/bin/env python
"""Verify: resonant transport creates non-trivial holonomy.

The old global transport used A(sigma) on every edge, making all transport
matrices commute. Gap sums telescoped: holonomy = I for all cycles.

The new resonant transport assigns each edge to its closest prime (by log),
using G_{p*}(sigma) instead of A(sigma). Since G_2 and G_3 don't commute,
triangles with edges assigned to different primes have non-trivial holonomy.

This script verifies the fix by:
1. Showing that different edges bind to different primes
2. Computing holonomy around triangles and verifying it is NOT the identity
3. Comparing old (flat) vs new (curved) holonomy
"""
import numpy as np
from atft.topology.transport_maps import TransportMapBuilder


def verify_non_trivial_holonomy():
    """Prove the resonant connection has non-trivial holonomy."""
    K = 6  # Use K=6: primes are 2, 3, 5
    sigma = 0.5

    builder = TransportMapBuilder(K=K, sigma=sigma)

    # Show the prime landscape
    primes = builder.primes
    log_primes = [np.log(p) for p in primes]
    print(f"K={K}, sigma={sigma}")
    print(f"Primes: {primes}")
    print(f"log(primes): {[f'{lp:.4f}' for lp in log_primes]}")
    print()

    # Use realistic unfolded zeros with mean spacing ~1
    zeros = np.array([0.0, 0.7, 1.2, 2.5, 3.1, 4.8, 5.5, 7.2, 8.0, 9.3])
    N = len(zeros)
    epsilon = 3.0

    # Show which edges get assigned to which primes
    print("Edge assignments (resonant prime per edge):")
    print(f"{'Edge':>12} | {'gap':>8} | {'prime':>5} | log(prime)")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*5}-+-{'-'*10}")

    edge_primes = {}
    for i in range(N):
        for j in range(i+1, N):
            gap = zeros[j] - zeros[i]
            if gap > epsilon:
                break
            p = builder.resonant_prime(gap)
            edge_primes[(i,j)] = p
            print(f"  ({i:2d},{j:2d})   | {gap:8.4f} | {p:5d} | {np.log(p):.4f}")

    # Count edges per prime
    prime_counts = {}
    for p in edge_primes.values():
        prime_counts[p] = prime_counts.get(p, 0) + 1
    print(f"\nEdges per prime: {prime_counts}")
    n_distinct = len(prime_counts)
    print(f"Distinct primes used: {n_distinct}")

    if n_distinct < 2:
        print("WARNING: Only one prime used. Need more edge diversity for curvature.")
        print("Try larger epsilon or different zero spacing.")
        return

    # --- HOLONOMY CHECK ---
    print(f"\n{'='*60}")
    print("  HOLONOMY CHECK: Triangles with mixed primes")
    print(f"{'='*60}\n")

    n_nontrivial = 0
    n_triangles = 0

    for i in range(N):
        for j in range(i+1, N):
            if (i,j) not in edge_primes:
                continue
            for k in range(j+1, N):
                if (j,k) not in edge_primes or (i,k) not in edge_primes:
                    continue

                n_triangles += 1
                p_ij = edge_primes[(i,j)]
                p_jk = edge_primes[(j,k)]
                p_ik = edge_primes[(i,k)]

                gap_ij = zeros[j] - zeros[i]
                gap_jk = zeros[k] - zeros[j]
                gap_ik = zeros[k] - zeros[i]

                # Compute holonomy around triangle: U_ij @ U_jk @ U_ki
                # where U_ki = U_ik^dagger (reverse direction)
                U_ij = builder.transport_resonant(gap_ij)
                U_jk = builder.transport_resonant(gap_jk)
                U_ik = builder.transport_resonant(gap_ik)
                U_ki = U_ik.conj().T  # reverse

                holonomy = U_ij @ U_jk @ U_ki
                deviation = np.linalg.norm(holonomy - np.eye(K))

                mixed = len(set([p_ij, p_jk, p_ik])) > 1
                marker = " ** MIXED PRIMES **" if mixed else ""

                if deviation > 1e-10:
                    n_nontrivial += 1
                    print(f"  Triangle ({i},{j},{k}): primes=({p_ij},{p_jk},{p_ik})"
                          f" ||H-I||={deviation:.6f}{marker}")
                else:
                    print(f"  Triangle ({i},{j},{k}): primes=({p_ij},{p_jk},{p_ik})"
                          f" ||H-I||={deviation:.2e}  (trivial){marker}")

    print(f"\nTotal triangles: {n_triangles}")
    print(f"Non-trivial holonomy: {n_nontrivial}")
    print(f"Trivial holonomy: {n_triangles - n_nontrivial}")

    # --- COMPARE OLD vs NEW ---
    print(f"\n{'='*60}")
    print("  COMPARISON: Global (flat) vs Resonant (curved) transport")
    print(f"{'='*60}\n")

    # Pick a triangle with mixed primes for detailed comparison
    for i in range(N):
        for j in range(i+1, N):
            if (i,j) not in edge_primes:
                continue
            for k in range(j+1, N):
                if (j,k) not in edge_primes or (i,k) not in edge_primes:
                    continue
                p_ij = edge_primes[(i,j)]
                p_jk = edge_primes[(j,k)]
                p_ik = edge_primes[(i,k)]
                if len(set([p_ij, p_jk, p_ik])) > 1:
                    gap_ij = zeros[j] - zeros[i]
                    gap_jk = zeros[k] - zeros[j]
                    gap_ik = zeros[k] - zeros[i]

                    # Old: global transport (flat)
                    builder.build_generator_sum()
                    U_ij_old = builder.transport(gap_ij)
                    U_jk_old = builder.transport(gap_jk)
                    U_ik_old = builder.transport(gap_ik)
                    H_old = U_ij_old @ U_jk_old @ U_ik_old.conj().T
                    dev_old = np.linalg.norm(H_old - np.eye(K))

                    # New: resonant transport (curved)
                    U_ij_new = builder.transport_resonant(gap_ij)
                    U_jk_new = builder.transport_resonant(gap_jk)
                    U_ik_new = builder.transport_resonant(gap_ik)
                    H_new = U_ij_new @ U_jk_new @ U_ik_new.conj().T
                    dev_new = np.linalg.norm(H_new - np.eye(K))

                    print(f"Triangle ({i},{j},{k}): primes=({p_ij},{p_jk},{p_ik})")
                    print(f"  Old (global A): ||H-I|| = {dev_old:.2e}  {'FLAT' if dev_old < 1e-10 else 'CURVED'}")
                    print(f"  New (resonant):  ||H-I|| = {dev_new:.6f}  {'FLAT' if dev_new < 1e-10 else 'CURVED'}")
                    print()
                    print(f"  Old holonomy matrix (should be ~I):")
                    print(f"    diag = {np.diag(H_old).real}")
                    print(f"  New holonomy matrix (should NOT be I):")
                    print(f"    diag = {np.diag(H_new).real}")
                    print()

                    # Show the key: G_p generators don't commute
                    G2 = builder.build_generator(2)
                    G3 = builder.build_generator(3)
                    commutator = G2 @ G3 - G3 @ G2
                    comm_norm = np.linalg.norm(commutator)
                    print(f"  ||[G_2, G_3]|| = {comm_norm:.6f}")
                    print(f"  This non-commutativity is the SOURCE of curvature.")
                    return

    print(f"\n{'='*60}")
    print("  CONCLUSION")
    print(f"{'='*60}")
    if n_nontrivial > 0:
        print(f"\nSUCCESS: Resonant transport creates non-trivial holonomy!")
        print(f"The connection is CURVED. {n_nontrivial}/{n_triangles} triangles")
        print(f"have non-zero holonomy due to non-commuting prime generators.")
        print(f"\nThe flat connection obstruction has been BROKEN.")
        print(f"The sheaf Laplacian spectrum can now depend on sigma.")
    else:
        print(f"\nWARNING: All triangles still have trivial holonomy.")


if __name__ == "__main__":
    verify_non_trivial_holonomy()
