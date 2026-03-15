"""Multiplicative monoid representation for sheaf transport maps.

Implements the canonical u(K) Lie algebra connection from the ATFT Phase 2 spec:
  - ρ(p): truncated left-regular representation of (Z_{>0}, ×)
  - G_p(σ) = (log p / p^σ)(ρ(p) + ρ(p)†): Hermitian generator
  - A(σ) = Σ G_p(σ): generator sum, eigendecomposed once
  - U(Δγ) = V diag(e^{iΔγ·λ_k}) V†: O(K²) transport shortcut
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _is_prime(n: int) -> bool:
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _primes_up_to(n: int) -> list[int]:
    """Return all primes ≤ n."""
    return [p for p in range(2, n + 1) if _is_prime(p)]


class TransportMapBuilder:
    """Builds the u(K)-valued gauge connection for sheaf transport.

    The construction is canonical: ρ(p) encodes Dirichlet convolution
    as matrix multiplication. The 1/p^σ weighting in G_p(σ) tunes the
    connection to the critical line when σ = 1/2.

    Args:
        K: Fiber dimension (integers 1..K).
        sigma: Critical line parameter for the generator weighting.
        max_prime: Largest prime to include. Defaults to largest prime ≤ K.
    """

    def __init__(self, K: int, sigma: float, max_prime: int | None = None) -> None:
        self._K = K
        self._sigma = sigma
        self._max_prime = max_prime if max_prime is not None else K
        self._primes = _primes_up_to(self._max_prime)

        # Lazily computed and cached
        self._A: NDArray[np.float64] | None = None
        self._V: NDArray[np.complex128] | None = None
        self._Vh: NDArray[np.complex128] | None = None  # V† cached
        self._eigenvals: NDArray[np.float64] | None = None
        self._transport_cache: dict[float, NDArray[np.complex128]] = {}

    @property
    def K(self) -> int:
        return self._K

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def primes(self) -> list[int]:
        return list(self._primes)

    def build_prime_rep(self, p: int) -> NDArray[np.float64]:
        """Sparse K×K partial permutation matrix ρ(p).

        ρ(p)|n⟩ = |pn⟩ if pn ≤ K, else 0.
        """
        if not _is_prime(p):
            raise ValueError(f"{p} is not prime")

        K = self._K
        rho = np.zeros((K, K), dtype=np.float64)
        for n in range(1, K + 1):
            pn = p * n
            if pn <= K:
                rho[pn - 1, n - 1] = 1.0
        return rho

    def build_generator(self, p: int) -> NDArray[np.float64]:
        """Hermitian K×K matrix G_p(σ) = (log p / p^σ)(ρ(p) + ρ(p)†)."""
        rho = self.build_prime_rep(p)
        scale = np.log(p) / p**self._sigma
        return scale * (rho + rho.T)

    def build_generator_sum(self) -> NDArray[np.float64]:
        """A(σ) = Σ_p G_p(σ). Eigendecomposed internally."""
        if self._A is not None:
            return self._A.copy()

        K = self._K
        A = np.zeros((K, K), dtype=np.float64)
        for p in self._primes:
            A += self.build_generator(p)

        self._A = A
        # Eigendecompose: A = V diag(λ) V†
        eigenvals, V = np.linalg.eigh(A)
        self._eigenvals = eigenvals
        self._V = V.astype(np.complex128)
        self._Vh = self._V.conj().T

        return A.copy()

    def eigenvalues(self) -> NDArray[np.float64]:
        """Return the K eigenvalues of A(σ)."""
        if self._eigenvals is None:
            self.build_generator_sum()
        return self._eigenvals.copy()

    def transport(self, delta_gamma: float) -> NDArray[np.complex128]:
        """U = V diag(e^{iΔγ·λ_k}) V†. Returns K×K complex unitary matrix.

        Results are cached by delta_gamma for repeated lookups.
        """
        if self._V is None:
            self.build_generator_sum()

        # Round to avoid float precision creating distinct cache keys
        key = round(delta_gamma, 12)
        cached = self._transport_cache.get(key)
        if cached is not None:
            return cached

        phases = np.exp(1j * delta_gamma * self._eigenvals)
        # U = V @ diag(phases) @ V†  —  O(K²) via broadcasting
        result = (self._V * phases[np.newaxis, :]) @ self._Vh
        self._transport_cache[key] = result
        return result

    def batch_transport(self, delta_gammas: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Compute transport matrices for multiple gaps at once.

        Args:
            delta_gammas: 1D array of M gap values.

        Returns:
            (M, K, K) complex array of transport matrices.
        """
        if self._V is None:
            self.build_generator_sum()

        M = len(delta_gammas)
        K = self._K
        # (M, K) phase matrix
        phases = np.exp(1j * delta_gammas[:, np.newaxis] * self._eigenvals[np.newaxis, :])
        # (M, K, K): V * phases broadcast, then matmul with V†
        return np.einsum('ik,mk,jk->mij', self._V, phases, self._V.conj())
