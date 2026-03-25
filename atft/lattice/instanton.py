"""BPST instanton configurations on a lattice.

The BPST instanton is an exact solution to the SU(2) Yang-Mills equations
in Euclidean R⁴ with topological charge Q = ±1. It is given analytically:

    A_μ(x) = Im(σ̄_μν (x-x₀)_ν) / ((x-x₀)² + ρ²)

where σ̄_μν are the 't Hooft eta symbols, x₀ is the center, and ρ is the size.

To place it on a lattice: U_μ(x) = exp(i · a · A_μ(x)) where a is the lattice spacing.

Anti-instantons (Q = -1) use η_μν instead of η̄_μν — equivalently, A → -A.
Multi-instantons (|Q| > 1) are constructed by superposition (approximate for well-separated centers).

Reference: Belavin, Polyakov, Schwartz, Tyupkin (1975).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# 't Hooft eta symbols η̄_μν (self-dual, Q=+1)
# η̄_μν^a for a=1,2,3 (su(2) generators), μ,ν=0,1,2,3
# Convention: σ_a = Pauli matrices, σ̄_μν = η̄^a_μν σ_a/2
def eta_bar(a: int, mu: int, nu: int) -> float:
    """'t Hooft eta-bar symbol η̄^a_μν (self-dual)."""
    if mu == nu:
        return 0.0
    # Totally antisymmetric in spatial indices
    if mu == 0:
        return 1.0 if a == nu - 1 else (0.0 if nu - 1 != a else 0.0)
    if nu == 0:
        return -1.0 if a == mu - 1 else 0.0
    # Spatial: ε_{a,mu-1,nu-1}
    eps = {(0,1,2): 1, (1,2,0): 1, (2,0,1): 1,
           (0,2,1): -1, (2,1,0): -1, (1,0,2): -1}
    return float(eps.get((a, mu-1, nu-1), 0))


# Pauli matrices
SIGMA = [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),       # σ₁
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),    # σ₂
    np.array([[1, 0], [0, -1]], dtype=np.complex128),      # σ₃
]


def bpst_gauge_field(x: NDArray, center: NDArray, rho: float, Q: int = 1) -> NDArray:
    """Compute the BPST gauge field A_μ(x) at a single point.

    Args:
        x: (4,) position vector
        center: (4,) instanton center
        rho: instanton size parameter
        Q: topological charge (+1 = instanton, -1 = anti-instanton)

    Returns:
        (4, 2, 2) complex array — A_μ for μ=0,1,2,3, each a 2×2 su(2) matrix
    """
    dx = x - center
    r2 = np.sum(dx**2)
    denom = r2 + rho**2

    A = np.zeros((4, 2, 2), dtype=np.complex128)

    sign = 1.0 if Q > 0 else -1.0

    for mu in range(4):
        for a in range(3):
            coeff = 0.0
            for nu in range(4):
                coeff += sign * eta_bar(a, mu, nu) * dx[nu]
            A[mu] += coeff / denom * SIGMA[a] / 2.0

    return A


def generate_instanton_config(
    lattice_shape: tuple[int, ...],
    Q: int = 1,
    rho: float = 2.0,
    center: NDArray | None = None,
    lattice_spacing: float = 1.0,
) -> dict[int, NDArray]:
    """Generate a BPST instanton configuration on a lattice.

    Args:
        lattice_shape: e.g. (12, 12, 12, 12) for a 12⁴ lattice
        Q: Topological charge (+1, -1, 0, +2, etc.)
        rho: Instanton size in lattice units
        center: (4,) center position. Default: lattice center.
        lattice_spacing: a (lattice spacing parameter)

    Returns:
        Dict {mu: NDArray of shape (*lattice_shape, 2, 2)} — link variables U_μ(x)
    """
    ndim = len(lattice_shape)
    assert ndim == 4, "BPST instantons require 4D lattice"

    if center is None:
        center = np.array([s / 2.0 for s in lattice_shape])

    links = {}

    if Q == 0:
        # Vacuum: all links = identity
        for mu in range(ndim):
            links[mu] = np.tile(np.eye(2, dtype=np.complex128),
                                (*lattice_shape, 1, 1))
        return links

    # For |Q| > 1: superpose well-separated instantons
    centers = [center]
    charges = [np.sign(Q)]

    if abs(Q) > 1:
        # Place additional instantons along the diagonal
        for i in range(1, abs(Q)):
            offset = np.array([lattice_shape[d] / (abs(Q) + 1) * (i + 1)
                               for d in range(4)])
            centers.append(offset)
            charges.append(np.sign(Q))

    for mu in range(ndim):
        U_mu = np.zeros((*lattice_shape, 2, 2), dtype=np.complex128)

        for idx in np.ndindex(*lattice_shape):
            x = np.array(idx, dtype=np.float64) * lattice_spacing

            # Sum gauge fields from all instantons
            A_mu_total = np.zeros((2, 2), dtype=np.complex128)
            for c, q in zip(centers, charges):
                A = bpst_gauge_field(x, c, rho, int(q))
                A_mu_total += A[mu]

            # Exponentiate to get link variable
            # U = exp(i * a * A_mu)
            from scipy.linalg import expm
            U_mu[idx] = expm(1j * lattice_spacing * A_mu_total)

        links[mu] = U_mu

    return links


def measure_topological_charge(config, lattice_shape):
    """Measure the topological charge Q from the field tensor.

    Q = (1/32π²) Σ_x ε_μνρσ Tr(F_μν(x) F_ρσ(x))

    Approximated via clover-leaf plaquette average.
    """
    from atft.lattice.su2 import plaquette

    ndim = len(lattice_shape)
    Q_total = 0.0

    for idx in np.ndindex(*lattice_shape):
        for mu in range(ndim):
            for nu in range(mu + 1, ndim):
                P = plaquette(config, idx, mu, nu, lattice_shape)
                # q_μν = ½ Im Tr P_μν
                q = 0.5 * np.imag(np.trace(P))
                Q_total += q

    # Normalize (approximate)
    Q_total /= (32 * np.pi**2)
    return Q_total
