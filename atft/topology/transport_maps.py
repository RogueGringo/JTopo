"""Multiplicative monoid representation for sheaf transport maps.

Implements the canonical u(K) Lie algebra connection from the ATFT Phase 2 spec:
  - ρ(p): truncated left-regular representation of (Z_{>0}, ×)
  - G_p(σ) = (log p / p^σ)(ρ(p) + ρ(p)†): Hermitian generator
  - A(σ) = Σ G_p(σ): generator sum, eigendecomposed once
  - U(Δγ) = V diag(e^{iΔγ·λ_k}) V†: O(K²) transport shortcut

Transport modes:
  - "global": flat connection using A(σ) — all edges share eigenbasis
  - "resonant": curved connection — each edge binds to resonant prime p*
  - "fe": functional equation connection — encodes ζ(s) ↔ ζ(1-s) symmetry
    G_p^FE(σ) = log(p) * [p^{-σ} ρ(p) + p^{-(1-σ)} ρ(p)^T]
    Hermitian (unitary transport) only at σ = 1/2
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

        # Per-prime eigendecompositions for resonant transport
        self._prime_decomps: dict[int, tuple[
            NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]
        ]] | None = None
        self._log_primes: NDArray[np.float64] | None = None

        # Functional equation per-prime eigendecompositions
        self._fe_decomps: dict[int, tuple[
            NDArray[np.complex128],   # P (right eigenvectors)
            NDArray[np.complex128],   # eigenvalues (may be complex)
            NDArray[np.complex128],   # P_inv
        ]] | None = None

        # Superposition per-prime basis matrices
        self._superposition_bases: NDArray[np.float64] | None = None

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

    # ── Resonant (dependency-graph) transport ─────────────────────────────

    def _ensure_prime_decomps(self) -> None:
        """Eigendecompose each prime generator G_p(sigma) individually.

        This enables per-edge prime assignment: each edge uses the single
        prime generator whose log(p) frequency best matches the gap,
        creating non-commuting transport and genuine holonomy.
        """
        if self._prime_decomps is not None:
            return
        self._prime_decomps = {}
        self._log_primes = (
            np.array([np.log(p) for p in self._primes])
            if self._primes
            else np.array([])
        )
        for p in self._primes:
            G = self.build_generator(p)
            eigenvals, V = np.linalg.eigh(G)
            Vc = V.astype(np.complex128)
            self._prime_decomps[p] = (Vc, eigenvals, Vc.conj().T)

    def resonant_prime(self, delta_gamma: float) -> int:
        """Find the prime whose log frequency best resonates with |delta_gamma|.

        Returns the prime p minimizing |abs(delta_gamma) - log(p)|.
        """
        self._ensure_prime_decomps()
        if not self._primes:
            return 0
        abs_gap = abs(delta_gamma)
        idx = int(np.argmin(np.abs(abs_gap - self._log_primes)))
        return self._primes[idx]

    def transport_resonant(self, delta_gamma: float) -> NDArray[np.complex128]:
        """Transport using only the resonant prime's generator.

        Instead of U = exp(i*dg*A(sigma)) with global A, uses
        U = exp(i*dg*G_{p*}(sigma)) where p* is the prime whose log(p*)
        is closest to |delta_gamma|.

        Different edges get different non-commuting generators,
        yielding non-trivial holonomy around graph cycles.
        """
        self._ensure_prime_decomps()
        if not self._primes:
            return np.eye(self._K, dtype=np.complex128)
        p = self.resonant_prime(delta_gamma)
        V, eigenvals, Vh = self._prime_decomps[p]
        phases = np.exp(1j * delta_gamma * eigenvals)
        return (V * phases[np.newaxis, :]) @ Vh

    def batch_transport_resonant(
        self, delta_gammas: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Batch transport using resonant prime per edge.

        Each gap is assigned to the prime whose log is closest, then
        transport is computed using only that prime's generator. Edges
        are grouped by assigned prime for vectorized computation.

        Returns:
            (M, K, K) complex array of transport matrices.
        """
        self._ensure_prime_decomps()
        M = len(delta_gammas)
        K = self._K

        if M == 0:
            return np.empty((0, K, K), dtype=np.complex128)

        if not self._primes:
            return np.tile(np.eye(K, dtype=np.complex128), (M, 1, 1))

        result = np.empty((M, K, K), dtype=np.complex128)
        abs_gaps = np.abs(delta_gammas)

        # Assign each gap to its resonant prime: argmin |gap - log(p)|
        diffs = np.abs(
            abs_gaps[:, np.newaxis] - self._log_primes[np.newaxis, :]
        )  # (M, P)
        prime_indices = np.argmin(diffs, axis=1)  # (M,)

        # Group by prime for efficient batch computation
        for pidx, p in enumerate(self._primes):
            mask = prime_indices == pidx
            if not np.any(mask):
                continue
            V, eigenvals, Vh = self._prime_decomps[p]
            gaps_p = delta_gammas[mask]
            # (count, K) phase matrix
            phases = np.exp(
                1j * gaps_p[:, np.newaxis] * eigenvals[np.newaxis, :]
            )
            result[mask] = np.einsum('ik,mk,jk->mij', V, phases, V.conj())

        return result

    # ── Functional equation transport ─────────────────────────────────

    def build_generator_fe(self, p: int) -> NDArray[np.float64]:
        """Functional equation generator encoding zeta(s) <-> zeta(1-s).

        G_p^FE(sigma) = log(p) * [p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T]

        Normalized by Frobenius norm so the overall energy is constant
        across sigma. The ONLY sigma-dependence is the internal phase
        geometry: Hermitian (balanced) at sigma=1/2, asymmetric elsewhere.

        At sigma = 1/2: generator is Hermitian, transport is unitary.
        At sigma != 1/2: generator is NOT Hermitian, transport is NOT unitary.
        """
        rho = self.build_prime_rep(p)
        log_p = np.log(p)
        fwd_weight = log_p / p**self._sigma         # p^{-sigma}
        bwd_weight = log_p / p**(1 - self._sigma)    # p^{-(1-sigma)}
        G = fwd_weight * rho + bwd_weight * rho.T
        # Normalize: hold generator energy constant across sigma sweep
        norm = np.linalg.norm(G, ord='fro')
        if norm > 0:
            G = G / norm
        return G

    def _ensure_fe_decomps(self) -> None:
        """Eigendecompose each functional equation generator.

        Uses general (non-symmetric) eigendecomp since G^FE is
        non-Hermitian for sigma != 1/2. Eigenvalues may be complex.
        """
        if self._fe_decomps is not None:
            return
        self._fe_decomps = {}
        if self._log_primes is None:
            self._log_primes = (
                np.array([np.log(p) for p in self._primes])
                if self._primes
                else np.array([])
            )
        for p in self._primes:
            G = self.build_generator_fe(p)
            eigenvals, P = np.linalg.eig(G)
            Pc = P.astype(np.complex128)
            P_inv = np.linalg.inv(Pc)
            self._fe_decomps[p] = (Pc, eigenvals.astype(np.complex128), P_inv)

    def transport_fe(self, delta_gamma: float) -> NDArray[np.complex128]:
        """Transport using functional equation generator of resonant prime.

        U = P diag(exp(i*dg*lambda_k)) P^{-1}

        At sigma=1/2: U is unitary (fibers metrically compatible).
        At sigma!=1/2: U is NOT unitary (fiber distortion).
        """
        self._ensure_fe_decomps()
        if not self._primes:
            return np.eye(self._K, dtype=np.complex128)
        if abs(delta_gamma) < 1e-15:
            return np.eye(self._K, dtype=np.complex128)

        self._ensure_prime_decomps()  # for log_primes
        p = self.resonant_prime(delta_gamma)
        P, eigenvals, P_inv = self._fe_decomps[p]
        phases = np.exp(1j * delta_gamma * eigenvals)
        return (P * phases[np.newaxis, :]) @ P_inv

    def batch_transport_fe(
        self, delta_gammas: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Batch functional equation transport using resonant prime per edge.

        Returns (M, K, K) complex array. NOT unitary for sigma != 1/2.
        """
        self._ensure_fe_decomps()
        self._ensure_prime_decomps()  # for log_primes and prime assignment
        M = len(delta_gammas)
        K = self._K

        if M == 0:
            return np.empty((0, K, K), dtype=np.complex128)

        if not self._primes:
            return np.tile(np.eye(K, dtype=np.complex128), (M, 1, 1))

        result = np.empty((M, K, K), dtype=np.complex128)
        abs_gaps = np.abs(delta_gammas)

        # Assign each gap to its resonant prime
        diffs = np.abs(
            abs_gaps[:, np.newaxis] - self._log_primes[np.newaxis, :]
        )
        prime_indices = np.argmin(diffs, axis=1)

        for pidx, p in enumerate(self._primes):
            mask = prime_indices == pidx
            if not np.any(mask):
                continue
            P, eigenvals, P_inv = self._fe_decomps[p]
            gaps_p = delta_gammas[mask]
            # (count, K) complex phase matrix
            phases = np.exp(
                1j * gaps_p[:, np.newaxis] * eigenvals[np.newaxis, :]
            )
            # U[m] = P @ diag(phases[m]) @ P_inv
            result[mask] = np.einsum('ik,mk,kj->mij', P, phases, P_inv)

        return result

    # ── Superposition (explicit formula) transport ──────────────────────

    def build_superposition_bases(self) -> NDArray[np.float64]:
        """Precompute B_p(sigma) for all primes.

        B_p(sigma) = log(p) * [p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T]

        This is the un-normalized functional equation generator.
        Returns (P, K, K) float64 array where P = number of primes <= K.
        """
        if self._superposition_bases is not None:
            return self._superposition_bases.copy()

        P = len(self._primes)
        K = self._K
        if P == 0:
            self._superposition_bases = np.empty((0, K, K), dtype=np.float64)
            return self._superposition_bases.copy()

        bases = np.zeros((P, K, K), dtype=np.float64)
        for idx, p in enumerate(self._primes):
            rho = self.build_prime_rep(p)
            log_p = np.log(p)
            fwd = log_p / p**self._sigma
            bwd = log_p / p**(1 - self._sigma)
            bases[idx] = fwd * rho + bwd * rho.T

        self._superposition_bases = bases
        # Ensure log_primes is available for phase computation
        if self._log_primes is None:
            self._log_primes = np.array([np.log(p) for p in self._primes])
        return bases.copy()

    def build_generator_superposition(
        self, delta_gamma: float, normalize: bool = True
    ) -> NDArray[np.complex128]:
        """Superposition generator for a single edge.

        A_{ij}(sigma) = sum_p e^{i * dg * log(p)} * B_p(sigma)

        Args:
            delta_gamma: Gap gamma_j - gamma_i for this edge.
            normalize: If True, normalize A by its Frobenius norm.

        Returns:
            (K, K) complex128 matrix.
        """
        bases = self.build_superposition_bases()
        K = self._K

        if len(bases) == 0:
            return np.zeros((K, K), dtype=np.complex128)

        phases = np.exp(1j * delta_gamma * self._log_primes)  # (P,)
        A = np.einsum('p,pij->ij', phases, bases)  # (K, K) complex

        if normalize:
            norm = np.linalg.norm(A, ord='fro')
            if norm > 0:
                A = A / norm

        return A

    def batch_transport_superposition(
        self,
        delta_gammas: NDArray[np.float64],
        normalize: bool = True,
    ) -> NDArray[np.complex128]:
        """Batch superposition transport for M edges.

        Computes U[e] = exp(i * A[e]) where A[e] is the superposition
        generator for each edge. Uses batched eigendecomposition with
        scipy.linalg.expm fallback for defective matrices.

        Args:
            delta_gammas: (M,) array of gap values.
            normalize: If True, Frobenius-normalize each generator.

        Returns:
            (M, K, K) complex128 array of transport matrices.
        """
        bases = self.build_superposition_bases()
        M = len(delta_gammas)
        K = self._K

        if M == 0:
            return np.empty((0, K, K), dtype=np.complex128)

        if len(bases) == 0:
            return np.tile(np.eye(K, dtype=np.complex128), (M, 1, 1))

        # Phase matrix: (M, P) complex
        phases = np.exp(
            1j * delta_gammas[:, np.newaxis] * self._log_primes[np.newaxis, :]
        )

        # Generator batch: (M, K, K) complex via tensor contraction
        A_batch = np.einsum('ep,pij->eij', phases, bases)

        # Optional per-edge Frobenius normalization
        if normalize:
            norms = np.linalg.norm(A_batch.reshape(M, -1), axis=1)
            mask = norms > 0
            A_batch[mask] /= norms[mask, np.newaxis, np.newaxis]

        # Matrix exponential via batched eigendecomposition
        eigenvals, P_mat = np.linalg.eig(A_batch)  # (M,K), (M,K,K)
        P_inv = np.linalg.inv(P_mat)  # (M, K, K)

        # exp(i * A) = P @ diag(exp(i * lambda)) @ P_inv
        exp_eigenvals = np.exp(1j * eigenvals)  # (M, K)
        result = np.einsum('mik,mk,mkj->mij', P_mat, exp_eigenvals, P_inv)

        # Check for defective matrices and fix with expm fallback
        P_norms = np.linalg.norm(P_mat.reshape(M, -1), axis=1)
        P_inv_norms = np.linalg.norm(P_inv.reshape(M, -1), axis=1)
        cond_est = P_norms * P_inv_norms
        defective = cond_est > 1e12

        if np.any(defective):
            from scipy.linalg import expm
            for idx in np.where(defective)[0]:
                result[idx] = expm(1j * A_batch[idx])

        return result
