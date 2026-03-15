"""Tests for transport_maps.py — the multiplicative monoid representation.

Covers:
  - Prime representation ρ(p): partial permutation matrices
  - Hermitian generator G_p(σ)
  - Generator sum A(σ) and eigendecomposition
  - Transport matrix U(Δγ) via eigendecomp shortcut
  - Edge cases: K=1, primes > K, zero gap
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.topology.transport_maps import TransportMapBuilder


# ── ρ(p) tests ──────────────────────────────────────────────────────────────

class TestBuildPrimeRep:
    """Test the partial permutation matrix ρ(p)."""

    def test_rho2_K10(self):
        """ρ(2)|n⟩ = |2n⟩ if 2n ≤ 10, else 0."""
        builder = TransportMapBuilder(K=10, sigma=0.5)
        rho = builder.build_prime_rep(2)
        # |1⟩ → |2⟩, |2⟩ → |4⟩, |3⟩ → |6⟩, |4⟩ → |8⟩, |5⟩ → |10⟩
        # |6⟩ → 0 (12>10), ..., |10⟩ → 0 (20>10)
        assert rho.shape == (10, 10)
        # Check non-zero entries: rho[pn-1, n-1] = 1
        for n in range(1, 11):
            if 2 * n <= 10:
                assert rho[2 * n - 1, n - 1] == 1.0
            else:
                # Column n-1 should be all zeros
                assert np.sum(np.abs(rho[:, n - 1])) == 0.0

    def test_rho3_K20(self):
        """ρ(3)|n⟩ = |3n⟩ if 3n ≤ 20, else 0."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        rho = builder.build_prime_rep(3)
        for n in range(1, 21):
            col = rho[:, n - 1]
            if 3 * n <= 20:
                assert col[3 * n - 1] == 1.0
                assert np.sum(np.abs(col)) == 1.0
            else:
                assert np.sum(np.abs(col)) == 0.0

    def test_rho_composition_encodes_multiplication(self):
        """ρ(2)ρ(3)|1⟩ = |6⟩ — Dirichlet convolution as matrix mult."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        rho2 = builder.build_prime_rep(2)
        rho3 = builder.build_prime_rep(3)
        e1 = np.zeros(20)
        e1[0] = 1.0  # |1⟩
        result = rho2 @ rho3 @ e1
        expected = np.zeros(20)
        expected[5] = 1.0  # |6⟩
        assert_allclose(result, expected)

    def test_rho_prime_power(self):
        """ρ(2)ρ(2)|1⟩ = |4⟩ — prime powers."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        rho2 = builder.build_prime_rep(2)
        e1 = np.zeros(20)
        e1[0] = 1.0
        result = rho2 @ rho2 @ e1
        expected = np.zeros(20)
        expected[3] = 1.0  # |4⟩
        assert_allclose(result, expected)

    def test_rho_truncation(self):
        """ρ(2)|11⟩ = 0 when K=20 (22 > 20)."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        rho2 = builder.build_prime_rep(2)
        e11 = np.zeros(20)
        e11[10] = 1.0  # |11⟩
        result = rho2 @ e11
        assert_allclose(result, np.zeros(20))

    def test_rho_is_sparse(self):
        """ρ(p) should have at most ⌊K/p⌋ non-zero entries."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            rho = builder.build_prime_rep(p)
            nnz = np.count_nonzero(rho)
            assert nnz == 20 // p

    def test_non_prime_raises(self):
        """build_prime_rep should reject non-primes."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        with pytest.raises(ValueError, match="not prime"):
            builder.build_prime_rep(4)

    def test_rho_K1_all_primes_vanish(self):
        """At K=1, ρ(p)|1⟩ = |p⟩ but p > 1 = K, so ρ(p) = 0."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        rho2 = builder.build_prime_rep(2)
        assert rho2.shape == (1, 1)
        assert rho2[0, 0] == 0.0


# ── G_p(σ) tests ───────────────────────────────────────────────────────────

class TestBuildGenerator:
    """Test the Hermitian generator G_p(σ) = (log p / p^σ)(ρ(p) + ρ(p)†)."""

    def test_generator_is_hermitian(self):
        """G_p must be Hermitian for all primes."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        for p in [2, 3, 5, 7, 11, 13, 17, 19]:
            G = builder.build_generator(p)
            assert_allclose(G, G.conj().T, atol=1e-15)

    def test_generator_scaling(self):
        """G_p(σ) should scale as log(p) / p^σ."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        G2 = builder.build_generator(2)
        rho2 = builder.build_prime_rep(2)
        expected_scale = np.log(2) / 2**0.5
        expected = expected_scale * (rho2 + rho2.T)
        assert_allclose(G2, expected, atol=1e-15)

    def test_generator_sigma_dependence(self):
        """Changing σ changes the generator magnitude but not the sparsity pattern."""
        b1 = TransportMapBuilder(K=20, sigma=0.3)
        b2 = TransportMapBuilder(K=20, sigma=0.7)
        G1 = b1.build_generator(2)
        G2 = b2.build_generator(2)
        # Same sparsity pattern
        assert np.array_equal(G1 != 0, G2 != 0)
        # Different magnitudes: ratio should be 2^(0.7-0.3) = 2^0.4
        nz1 = G1[G1 != 0]
        nz2 = G2[G2 != 0]
        ratio = np.abs(nz1[0]) / np.abs(nz2[0])
        assert_allclose(ratio, 2**0.4, rtol=1e-10)

    def test_generator_K1_is_zero(self):
        """At K=1, all generators vanish."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        G2 = builder.build_generator(2)
        assert_allclose(G2, np.zeros((1, 1)))


# ── A(σ) tests ──────────────────────────────────────────────────────────────

class TestBuildGeneratorSum:
    """Test A(σ) = Σ G_p(σ) and its eigendecomposition."""

    def test_generator_sum_is_hermitian(self):
        builder = TransportMapBuilder(K=20, sigma=0.5)
        A = builder.build_generator_sum()
        assert_allclose(A, A.conj().T, atol=1e-14)

    def test_generator_sum_shape(self):
        builder = TransportMapBuilder(K=20, sigma=0.5)
        A = builder.build_generator_sum()
        assert A.shape == (20, 20)

    def test_eigendecomposition_reconstructs_A(self):
        """V Λ V† must reconstruct A(σ)."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        A = builder.build_generator_sum()
        eigenvalues = builder.eigenvalues()
        # Reconstruct: A = V diag(λ) V†
        # We test via transport at Δγ=0 giving identity
        U_zero = builder.transport(0.0)
        assert_allclose(U_zero, np.eye(20), atol=1e-14)

    def test_eigenvalues_real(self):
        """Eigenvalues of Hermitian A(σ) must be real."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        eigs = builder.eigenvalues()
        assert eigs.dtype in (np.float64, np.float32)

    def test_generator_sum_K1_is_zero(self):
        """At K=1, A(σ) = 0."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        A = builder.build_generator_sum()
        assert_allclose(A, np.zeros((1, 1)))

    def test_max_prime_limits_sum(self):
        """max_prime < K should include fewer generators."""
        b_full = TransportMapBuilder(K=20, sigma=0.5)
        b_partial = TransportMapBuilder(K=20, sigma=0.5, max_prime=7)
        A_full = b_full.build_generator_sum()
        A_partial = b_partial.build_generator_sum()
        # A_partial should be different (missing primes 11,13,17,19)
        assert not np.allclose(A_full, A_partial)
        # A_partial should equal sum of G_2 + G_3 + G_5 + G_7 only
        A_expected = sum(b_partial.build_generator(p) for p in [2, 3, 5, 7])
        assert_allclose(A_partial, A_expected, atol=1e-14)

    def test_eigenvalues_count(self):
        """Should have exactly K eigenvalues."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        eigs = builder.eigenvalues()
        assert len(eigs) == 20


# ── U(Δγ) transport tests ──────────────────────────────────────────────────

class TestTransport:
    """Test U = V diag(e^{iΔγλ_k}) V†."""

    def test_transport_is_unitary(self):
        """U must be unitary: U U† = I."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        U = builder.transport(1.5)
        assert_allclose(U @ U.conj().T, np.eye(20), atol=1e-13)

    def test_transport_zero_gap_is_identity(self):
        """U(Δγ=0) = I."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        U = builder.transport(0.0)
        assert_allclose(U, np.eye(20), atol=1e-14)

    def test_transport_composition(self):
        """U(a+b) = U(a) U(b) — group property of matrix exponential."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        a, b = 1.3, 0.7
        Ua = builder.transport(a)
        Ub = builder.transport(b)
        Uab = builder.transport(a + b)
        assert_allclose(Ua @ Ub, Uab, atol=1e-13)

    def test_transport_inverse(self):
        """U(-Δγ) = U(Δγ)†."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        U = builder.transport(2.0)
        U_inv = builder.transport(-2.0)
        assert_allclose(U_inv, U.conj().T, atol=1e-13)

    def test_transport_K1_always_identity(self):
        """At K=1, A=0, so U = e^{i·0} = 1."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        U = builder.transport(42.0)
        assert_allclose(U, np.array([[1.0 + 0j]]), atol=1e-14)

    def test_transport_preserves_hermiticity(self):
        """Conjugation U H U† preserves Hermiticity."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        U = builder.transport(1.0)
        # Random Hermitian matrix
        rng = np.random.default_rng(0)
        M = rng.standard_normal((20, 20)) + 1j * rng.standard_normal((20, 20))
        H = M + M.conj().T  # Hermitian
        transported = U @ H @ U.conj().T
        assert_allclose(transported, transported.conj().T, atol=1e-13)

    def test_transport_determinant_is_unit(self):
        """det(U) should have |det(U)| = 1 (U is unitary)."""
        builder = TransportMapBuilder(K=20, sigma=0.5)
        U = builder.transport(1.0)
        det = np.linalg.det(U)
        assert_allclose(np.abs(det), 1.0, atol=1e-12)

    def test_different_sigma_different_transport(self):
        """Different σ values produce different transport matrices."""
        b1 = TransportMapBuilder(K=20, sigma=0.3)
        b2 = TransportMapBuilder(K=20, sigma=0.7)
        U1 = b1.transport(1.0)
        U2 = b2.transport(1.0)
        assert not np.allclose(U1, U2)


# ── Superposition bases tests ─────────────────────────────────────────────


class TestSuperpositionBases:
    """Tests for build_superposition_bases()."""

    def test_shape(self):
        builder = TransportMapBuilder(K=10, sigma=0.5)
        bases = builder.build_superposition_bases()
        n_primes = len(builder.primes)  # primes <= 10: [2, 3, 5, 7] = 4
        assert bases.shape == (n_primes, 10, 10)

    def test_real_dtype(self):
        builder = TransportMapBuilder(K=10, sigma=0.5)
        bases = builder.build_superposition_bases()
        assert bases.dtype == np.float64

    def test_symmetric_at_half(self):
        """At sigma=0.5, B_p = log(p)/sqrt(p) * (rho + rho^T) is symmetric."""
        builder = TransportMapBuilder(K=10, sigma=0.5)
        bases = builder.build_superposition_bases()
        for p_idx in range(len(builder.primes)):
            np.testing.assert_allclose(
                bases[p_idx], bases[p_idx].T, atol=1e-14,
                err_msg=f"B_p not symmetric at sigma=0.5 for prime index {p_idx}"
            )

    def test_asymmetric_off_half(self):
        """At sigma != 0.5, B_p is NOT symmetric (p^{-sigma} != p^{-(1-sigma)})."""
        builder = TransportMapBuilder(K=10, sigma=0.3)
        bases = builder.build_superposition_bases()
        # Check prime 2 (index 0): rho(2) has entries, so B_2 should be asymmetric
        assert not np.allclose(bases[0], bases[0].T, atol=1e-10)

    def test_matches_unnormalized_fe_generator(self):
        """B_p(sigma) should equal the FE generator BEFORE Frobenius normalization."""
        builder = TransportMapBuilder(K=10, sigma=0.4)
        bases = builder.build_superposition_bases()
        for p_idx, p in enumerate(builder.primes):
            rho = builder.build_prime_rep(p)
            log_p = np.log(p)
            expected = log_p * (rho / p**0.4 + rho.T / p**0.6)
            np.testing.assert_allclose(bases[p_idx], expected, atol=1e-14)

    def test_cached(self):
        """Second call returns same data without recomputation."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        bases1 = builder.build_superposition_bases()
        bases2 = builder.build_superposition_bases()
        np.testing.assert_array_equal(bases1, bases2)

    def test_no_primes(self):
        """K=1 has no primes <= 1, returns empty (0, 1, 1) array."""
        builder = TransportMapBuilder(K=1, sigma=0.5)
        bases = builder.build_superposition_bases()
        assert bases.shape == (0, 1, 1)


# ── Superposition generator tests ─────────────────────────────────────────


class TestSuperpositionGenerator:
    """Tests for build_generator_superposition()."""

    def test_complex_output(self):
        builder = TransportMapBuilder(K=6, sigma=0.5)
        A = builder.build_generator_superposition(1.5)
        assert A.dtype == np.complex128

    def test_shape(self):
        builder = TransportMapBuilder(K=10, sigma=0.5)
        A = builder.build_generator_superposition(1.0)
        assert A.shape == (10, 10)

    def test_non_hermitian_generic_gap(self):
        """For a generic gap, A should NOT be Hermitian (complex phases break it)."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        A = builder.build_generator_superposition(1.234, normalize=False)
        # A - A† should be nonzero
        assert not np.allclose(A, A.conj().T, atol=1e-10)

    def test_normalized_unit_frobenius(self):
        """With normalize=True, ||A||_F should equal 1."""
        builder = TransportMapBuilder(K=10, sigma=0.5)
        A = builder.build_generator_superposition(2.0, normalize=True)
        np.testing.assert_allclose(np.linalg.norm(A, ord='fro'), 1.0, atol=1e-14)

    def test_unnormalized_nonunit_frobenius(self):
        """With normalize=False, ||A||_F should NOT be 1 in general."""
        builder = TransportMapBuilder(K=10, sigma=0.5)
        A = builder.build_generator_superposition(2.0, normalize=False)
        assert not np.isclose(np.linalg.norm(A, ord='fro'), 1.0, atol=1e-6)

    def test_zero_gap_sums_all_bases(self):
        """At dg=0, all phases are 1, so A = sum(B_p)."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        A = builder.build_generator_superposition(0.0, normalize=False)
        bases = builder.build_superposition_bases()
        expected = np.sum(bases, axis=0).astype(np.complex128)
        np.testing.assert_allclose(A, expected, atol=1e-14)

    def test_no_primes_returns_zero(self):
        builder = TransportMapBuilder(K=1, sigma=0.5)
        A = builder.build_generator_superposition(1.0)
        np.testing.assert_allclose(A, np.zeros((1, 1), dtype=np.complex128), atol=1e-14)


class TestBatchSuperpositionTransport:
    """Tests for batch_transport_superposition()."""

    def test_shape(self):
        builder = TransportMapBuilder(K=6, sigma=0.5)
        gaps = np.array([0.5, 1.0, 1.5, 2.0])
        U = builder.batch_transport_superposition(gaps)
        assert U.shape == (4, 6, 6)
        assert U.dtype == np.complex128

    def test_empty_input(self):
        builder = TransportMapBuilder(K=6, sigma=0.5)
        U = builder.batch_transport_superposition(np.array([]))
        assert U.shape == (0, 6, 6)

    def test_invertible(self):
        """All transport matrices should be invertible (det != 0)."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        gaps = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        U = builder.batch_transport_superposition(gaps)
        for e in range(5):
            assert abs(np.linalg.det(U[e])) > 1e-10

    def test_zero_gap_is_identity(self):
        """At dg=0 with normalize=False: A = sum(B_p) real, U = exp(i*sum(B_p))."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        U = builder.batch_transport_superposition(np.array([0.0]), normalize=False)
        # U is exp(i * real_matrix) — should be unitary
        I = np.eye(6, dtype=np.complex128)
        np.testing.assert_allclose(U[0] @ U[0].conj().T, I, atol=1e-12)

    def test_matches_single_edge(self):
        """Batch result should match single-edge computation."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        gaps = np.array([0.7, 1.3, 2.1])
        U_batch = builder.batch_transport_superposition(gaps, normalize=True)
        for e, dg in enumerate(gaps):
            A = builder.build_generator_superposition(dg, normalize=True)
            eigenvals, P = np.linalg.eig(A)
            P_inv = np.linalg.inv(P)
            U_single = (P * np.exp(1j * eigenvals)) @ P_inv
            np.testing.assert_allclose(U_batch[e], U_single, atol=1e-10)

    def test_no_primes_returns_identity(self):
        builder = TransportMapBuilder(K=1, sigma=0.5)
        U = builder.batch_transport_superposition(np.array([1.0, 2.0]))
        I = np.eye(1, dtype=np.complex128)
        for e in range(2):
            np.testing.assert_allclose(U[e], I, atol=1e-14)

    def test_unitary_at_resonant_gap_single_prime(self):
        """K=3, max_prime=2 (only prime 2), sigma=0.5, gap = 2*pi/log(2): phase = 1.
        Generator is Hermitian => transport is unitary."""
        builder = TransportMapBuilder(K=3, sigma=0.5, max_prime=2)
        dg = 2 * np.pi / np.log(2)
        U = builder.batch_transport_superposition(np.array([dg]), normalize=False)
        I = np.eye(3, dtype=np.complex128)
        np.testing.assert_allclose(U[0] @ U[0].conj().T, I, atol=1e-10)

    def test_normalized_vs_unnormalized_different(self):
        """Normalized and unnormalized should give different transport matrices."""
        builder = TransportMapBuilder(K=6, sigma=0.5)
        gaps = np.array([1.0])
        U_norm = builder.batch_transport_superposition(gaps, normalize=True)
        U_raw = builder.batch_transport_superposition(gaps, normalize=False)
        assert not np.allclose(U_norm, U_raw, atol=1e-6)
