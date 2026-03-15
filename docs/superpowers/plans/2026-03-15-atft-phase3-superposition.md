# ATFT Phase 3: Superposition & Scale Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement multi-prime superposition transport with vector fibers and sparse Laplacian at N=10,000 scale, then run a definitive sigma-sweep control test.

**Architecture:** Three components ship together: (1) superposition transport methods in TransportMapBuilder using phase-weighted prime interference, (2) SparseSheafLaplacian with BSR assembly and shift-invert eigsh for C^K vector fibers, (3) experiment script for the definitive zeta vs random vs GUE control test.

**Tech Stack:** Python 3.11+, NumPy >= 2.0 (batched eig/inv on 3D arrays), SciPy >= 1.11 (bsr_matrix, eigsh), existing ATFT Phase 1+2 codebase (171 tests, untouched).

**Spec:** `docs/superpowers/specs/2026-03-15-atft-phase3-superposition-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `atft/topology/transport_maps.py` | Modify | Add superposition transport: bases, generator, batch transport |
| `tests/test_transport_maps.py` | Modify | Add TestSuperpositionBases, TestSuperpositionGenerator, TestBatchSuperpositionTransport |
| `atft/topology/sparse_sheaf_laplacian.py` | Create | SparseSheafLaplacian class: edge discovery, BSR assembly, eigsh solver |
| `tests/test_sparse_sheaf_laplacian.py` | Create | Dense equivalence, Hermiticity, PSD, eigensolver tests |
| `atft/experiments/phase3_superposition_sweep.py` | Create | Massive sigma-sweep control test experiment |

**Unchanged:** All Phase 1+2 files. 171 existing tests must continue to pass.

---

## Chunk 1: Superposition Transport

### Task 1: Superposition Bases & Single-Edge Generator

**Files:**
- Modify: `atft/topology/transport_maps.py`
- Modify: `tests/test_transport_maps.py`

**Context:** The `TransportMapBuilder` class (lines 43-387) already has three transport modes (global, resonant, fe). We add a fourth: "superposition". The per-prime basis matrices `B_p(sigma)` are identical to the un-normalized FE generators from `build_generator_fe()` (line 281). The superposition combines ALL primes with complex phase factors `e^{i*dg*log(p)}`, unlike resonant/fe which each assign ONE prime per edge.

- [ ] **Step 1: Write tests for superposition bases**

Add to `tests/test_transport_maps.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transport_maps.py::TestSuperpositionBases -v`
Expected: FAIL with `AttributeError: 'TransportMapBuilder' object has no attribute 'build_superposition_bases'`

- [ ] **Step 3: Implement build_superposition_bases**

In `atft/topology/transport_maps.py`, add to `__init__` (after line 80, the `self._fe_decomps` declaration):

```python
        # Superposition per-prime basis matrices
        self._superposition_bases: NDArray[np.float64] | None = None
```

Then add the method (after `batch_transport_fe`, around line 387):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport_maps.py::TestSuperpositionBases -v`
Expected: 7 PASSED

- [ ] **Step 5: Write tests for single-edge superposition generator**

Add to `tests/test_transport_maps.py`:

```python
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
```

- [ ] **Step 6: Implement build_generator_superposition**

Add to `atft/topology/transport_maps.py` (after `build_superposition_bases`):

```python
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
```

- [ ] **Step 7: Run all new tests**

Run: `python -m pytest tests/test_transport_maps.py::TestSuperpositionBases tests/test_transport_maps.py::TestSuperpositionGenerator -v`
Expected: 14 PASSED

- [ ] **Step 8: Verify existing tests still pass**

Run: `python -m pytest tests/test_transport_maps.py -v`
Expected: All existing tests + 14 new = all PASSED

- [ ] **Step 9: Commit**

```bash
git add atft/topology/transport_maps.py tests/test_transport_maps.py
git commit -m "feat: add superposition bases and single-edge generator to TransportMapBuilder

Implements B_p(sigma) = log(p)[p^{-sigma}*rho(p) + p^{-(1-sigma)}*rho(p)^T] basis matrices
and A_{ij}(sigma) = sum_p e^{i*dg*log(p)} * B_p(sigma) single-edge generator with
optional Frobenius normalization. Phase 3 superposition transport (part 1/2)."
```

---

### Task 2: Batch Superposition Transport

**Files:**
- Modify: `atft/topology/transport_maps.py`
- Modify: `tests/test_transport_maps.py`

**Context:** This task implements the batch matrix exponential: given M edges with gaps, compute `U[e] = exp(i * A[e])` for all edges simultaneously. Uses batched `np.linalg.eig` + `np.linalg.inv`. Falls back to `scipy.linalg.expm` for defective matrices (cond(P) > 1e12).

- [ ] **Step 1: Write tests for batch superposition transport**

Add to `tests/test_transport_maps.py`:

```python
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
        """K=3 (only prime 2), sigma=0.5, gap = 2*pi/log(2): phase = 1.
        Generator is Hermitian => transport is unitary."""
        builder = TransportMapBuilder(K=3, sigma=0.5)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transport_maps.py::TestBatchSuperpositionTransport -v`
Expected: FAIL with `AttributeError: 'TransportMapBuilder' object has no attribute 'batch_transport_superposition'`

- [ ] **Step 3: Implement batch_transport_superposition**

Add to `atft/topology/transport_maps.py` (after `build_generator_superposition`):

```python
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
        result = np.empty((M, K, K), dtype=np.complex128)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transport_maps.py::TestBatchSuperpositionTransport -v`
Expected: 8 PASSED

- [ ] **Step 5: Run ALL transport tests to verify no regressions**

Run: `python -m pytest tests/test_transport_maps.py -v`
Expected: All existing tests + 22 new tests = all PASSED

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: 171 existing + 22 new = 193 PASSED

- [ ] **Step 7: Commit**

```bash
git add atft/topology/transport_maps.py tests/test_transport_maps.py
git commit -m "feat: add batch superposition transport with matrix exponential

Implements batch_transport_superposition() using phase-weighted prime interference:
A_{ij} = sum_p e^{i*dg*log(p)} * B_p(sigma), U_{ij} = exp(i*A_{ij}).
Batched eigendecomposition with scipy.linalg.expm fallback for defective matrices.
Phase 3 superposition transport (part 2/2)."
```

---

## Chunk 2: Sparse Sheaf Laplacian

### Task 3: SparseSheafLaplacian Skeleton & Edge Discovery

**Files:**
- Create: `atft/topology/sparse_sheaf_laplacian.py`
- Create: `tests/test_sparse_sheaf_laplacian.py`

**Context:** This creates the new module for vector-fiber sheaf Laplacians. The class stores sorted zeros and a TransportMapBuilder. Edge discovery uses the 1D sorted structure: for each point i, binary search for the rightmost j where `zeros[j] - zeros[i] <= epsilon`. This is the same pattern as `SheafLaplacian._get_cached()` (sheaf_laplacian.py lines 81-142) but simplified since we work on pre-sorted zeros and return raw index arrays.

- [ ] **Step 1: Write tests for edge discovery**

Create `tests/test_sparse_sheaf_laplacian.py`:

```python
"""Tests for SparseSheafLaplacian (vector fibers, BSR sparse)."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


class TestEdgeDiscovery:
    """Tests for build_edge_list()."""

    def test_shape(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        assert i_idx.shape == j_idx.shape == gaps.shape
        assert i_idx.ndim == 1

    def test_edges_within_epsilon(self):
        """All discovered gaps should be <= epsilon."""
        zeros = np.array([0.0, 0.8, 1.5, 3.0, 3.2, 5.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.0)
        assert np.all(gaps <= 1.0 + 1e-12)
        assert np.all(gaps > 0)

    def test_edges_complete(self):
        """Should find ALL pairs within epsilon."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.5)
        # Pairs within 0.5: (0,1), (1,2), (2,3), (3,4)
        assert len(i_idx) == 4

    def test_oriented_i_less_j(self):
        """All edges should be oriented i < j."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(2.5)
        assert np.all(i_idx < j_idx)

    def test_eps_zero_no_edges(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.0)
        assert len(i_idx) == 0

    def test_large_eps_complete_graph(self):
        """Large epsilon should connect all pairs."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(100.0)
        # Complete graph on 4 vertices: 4*3/2 = 6 edges
        assert len(i_idx) == 6

    def test_unsorted_input_gets_sorted(self):
        """Input zeros in any order should produce correct edges."""
        zeros_unsorted = np.array([3.0, 1.0, 0.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros_unsorted)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        # After sorting: [0, 1, 2, 3]. Edges within 1.5: (0,1), (1,2), (2,3)
        assert len(i_idx) == 3
        assert np.all(gaps <= 1.5)
```

- [ ] **Step 2: Write the SparseSheafLaplacian skeleton with edge discovery**

Create `atft/topology/sparse_sheaf_laplacian.py`:

```python
"""Sparse sheaf Laplacian with C^K vector fibers.

Implements the Phase 3 vector-valued sheaf Laplacian using BSR sparse
matrices and shift-invert eigsh. Designed for N=10,000 scale with
K=50 fiber dimension.

Key differences from SheafLaplacian (Phase 2):
  - Vector fibers C^K instead of matrix fibers C^{K x K}
  - Explicit sparse matrix (BSR) instead of matrix-free LinearOperator
  - Coboundary: (delta s)_e = U_e s_i - s_j (left multiply, not conjugation)
  - Supports "superposition" transport mode (multi-prime phase interference)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from atft.topology.transport_maps import TransportMapBuilder


class SparseSheafLaplacian:
    """BSR sparse sheaf Laplacian with C^K vector fibers.

    The sheaf Laplacian L = delta_0^dagger delta_0 where
    (delta_0 s)_e = U_e s_i - s_j for oriented edge e = (i -> j).

    Block structure for each edge (i -> j) with transport U:
      L[i,i] += U^dagger U    (diagonal)
      L[j,j] += I_K           (diagonal)
      L[i,j] = -U^dagger      (off-diagonal)
      L[j,i] = -U             (off-diagonal)

    Args:
        builder: TransportMapBuilder providing K, sigma, transport methods.
        zeros: 1D array of (possibly unsorted) unfolded zeta zeros.
        transport_mode: "superposition" (default), "fe", or "resonant".
        normalize: Frobenius-normalize superposition generators (only for
            transport_mode="superposition").
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64],
        transport_mode: str = "superposition",
        normalize: bool = True,
    ) -> None:
        self._builder = builder
        self._zeros = np.sort(zeros.ravel())
        self._N = len(self._zeros)
        self._K = builder.K
        self._transport_mode = transport_mode
        self._normalize = normalize

    @property
    def N(self) -> int:
        return self._N

    @property
    def K(self) -> int:
        return self._K

    def build_edge_list(
        self, epsilon: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Discover all edges in the 1D Rips complex at scale epsilon.

        Uses binary search on sorted zeros for O(N log N + |E|) complexity.

        Returns:
            (i_idx, j_idx, gaps) where each is a 1D array of length |E|.
            All edges satisfy i < j and gaps[e] = zeros[j] - zeros[i] <= epsilon.
        """
        zeros = self._zeros
        N = self._N

        if epsilon <= 0 or N < 2:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        i_parts: list[NDArray[np.int64]] = []
        j_parts: list[NDArray[np.int64]] = []

        for i in range(N - 1):
            max_val = zeros[i] + epsilon
            # Binary search for rightmost j where zeros[j] <= max_val
            j_right = int(np.searchsorted(zeros, max_val, side='right'))
            j_right = min(j_right, N)
            if i + 1 < j_right:
                js = np.arange(i + 1, j_right, dtype=np.int64)
                i_parts.append(np.full(len(js), i, dtype=np.int64))
                j_parts.append(js)

        if i_parts:
            i_idx = np.concatenate(i_parts)
            j_idx = np.concatenate(j_parts)
        else:
            i_idx = np.array([], dtype=np.int64)
            j_idx = np.array([], dtype=np.int64)

        gaps = (
            self._zeros[j_idx] - self._zeros[i_idx]
            if len(i_idx) > 0
            else np.array([], dtype=np.float64)
        )
        return i_idx, j_idx, gaps
```

- [ ] **Step 3: Run edge discovery tests**

Run: `python -m pytest tests/test_sparse_sheaf_laplacian.py::TestEdgeDiscovery -v`
Expected: 7 PASSED

- [ ] **Step 4: Commit**

```bash
git add atft/topology/sparse_sheaf_laplacian.py tests/test_sparse_sheaf_laplacian.py
git commit -m "feat: add SparseSheafLaplacian skeleton with 1D edge discovery

New module for vector-fiber (C^K) sheaf Laplacian using sparse matrices.
Edge discovery uses binary search on sorted zeros for O(N log N) complexity.
Phase 3 sparse engine (part 1/3)."
```

---

### Task 4: BSR Matrix Assembly

**Files:**
- Modify: `atft/topology/sparse_sheaf_laplacian.py`
- Modify: `tests/test_sparse_sheaf_laplacian.py`

**Context:** This task builds the explicit BSR sparse matrix from the transport matrices. The critical validation is a dense equivalence test: at small N and K, manually build the dense (N*K, N*K) Laplacian and verify the sparse version matches. This catches sign errors, transposition errors, and accumulation bugs in the block assembly.

- [ ] **Step 1: Write dense reference helper and tests**

Add to `tests/test_sparse_sheaf_laplacian.py`:

```python
def _build_dense_vector_laplacian(
    zeros: NDArray, builder: TransportMapBuilder, epsilon: float,
    transport_mode: str = "superposition", normalize: bool = True,
) -> NDArray[np.complex128]:
    """Build explicit dense (N*K, N*K) vector-fiber sheaf Laplacian.

    Reference implementation for validating the sparse version.
    """
    sorted_zeros = np.sort(zeros.ravel())
    N = len(sorted_zeros)
    K = builder.K
    L = np.zeros((N * K, N * K), dtype=np.complex128)
    I_K = np.eye(K, dtype=np.complex128)

    for i in range(N):
        for j in range(i + 1, N):
            gap = sorted_zeros[j] - sorted_zeros[i]
            if gap > epsilon:
                break
            # Compute transport for this edge
            if transport_mode == "superposition":
                U = builder.batch_transport_superposition(
                    np.array([gap]), normalize=normalize
                )[0]
            elif transport_mode == "fe":
                U = builder.transport_fe(gap)
            else:
                U = builder.transport_resonant(gap)

            Uh = U.conj().T

            # Off-diagonal: L[i,j] = -U†, L[j,i] = -U
            L[i*K:(i+1)*K, j*K:(j+1)*K] = -Uh
            L[j*K:(j+1)*K, i*K:(i+1)*K] = -U

            # Diagonal: L[i,i] += U†U, L[j,j] += I
            L[i*K:(i+1)*K, i*K:(i+1)*K] += Uh @ U
            L[j*K:(j+1)*K, j*K:(j+1)*K] += I_K

    return L


class TestBuildMatrix:
    """Tests for build_matrix() BSR assembly."""

    def test_dense_equivalence_K3_N5(self):
        """Sparse Laplacian matches dense reference at small scale."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        L_sparse = lap.build_matrix(1.0)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.0, "superposition", True
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)

    def test_dense_equivalence_K6_N4_unnormalized(self):
        """Unnormalized superposition matches dense reference."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.4)
        lap = SparseSheafLaplacian(builder, zeros, normalize=False)
        L_sparse = lap.build_matrix(1.5)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "superposition", False
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)

    def test_hermitian(self):
        """L should equal L†."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        L = lap.build_matrix(1.5).toarray()
        npt.assert_allclose(L, L.conj().T, atol=1e-12)

    def test_positive_semi_definite(self):
        """All eigenvalues should be >= 0."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(1.0).toarray()
        eigenvals = np.linalg.eigvalsh(L)
        assert np.all(eigenvals > -1e-10)

    def test_shape(self):
        N, K = 5, 4
        zeros = np.arange(N, dtype=np.float64)
        builder = TransportMapBuilder(K=K, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(1.5)
        assert L.shape == (N * K, N * K)

    def test_eps_zero_is_zero_matrix(self):
        """No edges => zero Laplacian."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(0.0)
        npt.assert_allclose(L.toarray(), np.zeros((9, 9)), atol=1e-14)

    def test_fe_transport_mode(self):
        """FE transport mode also produces valid Laplacian."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, transport_mode="fe")
        L_sparse = lap.build_matrix(1.5)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "fe", True
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sparse_sheaf_laplacian.py::TestBuildMatrix -v`
Expected: FAIL with `AttributeError: 'SparseSheafLaplacian' object has no attribute 'build_matrix'`

- [ ] **Step 3: Implement build_matrix**

Add to `atft/topology/sparse_sheaf_laplacian.py`:

```python
    def _compute_transport(
        self, gaps: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Compute transport matrices for all edges."""
        if self._transport_mode == "superposition":
            return self._builder.batch_transport_superposition(
                gaps, normalize=self._normalize
            )
        elif self._transport_mode == "fe":
            return self._builder.batch_transport_fe(gaps)
        elif self._transport_mode == "resonant":
            return self._builder.batch_transport_resonant(gaps)
        else:
            raise ValueError(
                f"Unknown transport_mode {self._transport_mode!r}. "
                "Must be 'superposition', 'fe', or 'resonant'."
            )

    def build_matrix(self, epsilon: float) -> sp.csr_matrix:
        """Assemble the N*K x N*K sparse sheaf Laplacian.

        Block structure per edge (i -> j) with transport U:
          L[i,j] = -U†,  L[j,i] = -U   (off-diagonal K x K blocks)
          L[i,i] += U†U,  L[j,j] += I   (diagonal K x K blocks)

        Returns a Hermitian PSD sparse matrix in CSR format.
        """
        N = self._N
        K = self._K
        dim = N * K

        i_idx, j_idx, gaps = self.build_edge_list(epsilon)
        M = len(i_idx)

        if M == 0:
            return sp.csr_matrix((dim, dim), dtype=np.complex128)

        # Compute all transport matrices: (M, K, K)
        U_all = self._compute_transport(gaps)
        Uh_all = np.conj(np.transpose(U_all, (0, 2, 1)))  # U†: (M, K, K)

        # --- Build diagonal blocks: (N, K, K) ---
        diag_blocks = np.zeros((N, K, K), dtype=np.complex128)
        I_K = np.eye(K, dtype=np.complex128)

        # Accumulate U†U at tail vertices (i)
        UhU = Uh_all @ U_all  # (M, K, K) batched matmul
        np.add.at(diag_blocks, i_idx, UhU)

        # Accumulate I at head vertices (j)
        head_degrees = np.bincount(j_idx, minlength=N)
        for v in range(N):
            if head_degrees[v] > 0:
                diag_blocks[v] += head_degrees[v] * I_K

        # --- Collect all blocks ---
        n_blocks = N + 2 * M
        all_rows = np.empty(n_blocks, dtype=np.int64)
        all_cols = np.empty(n_blocks, dtype=np.int64)
        all_data = np.empty((n_blocks, K, K), dtype=np.complex128)

        # Diagonal blocks
        all_rows[:N] = np.arange(N)
        all_cols[:N] = np.arange(N)
        all_data[:N] = diag_blocks

        # Off-diagonal: -U† at (i, j)
        all_rows[N:N+M] = i_idx
        all_cols[N:N+M] = j_idx
        all_data[N:N+M] = -Uh_all

        # Off-diagonal: -U at (j, i)
        all_rows[N+M:] = j_idx
        all_cols[N+M:] = i_idx
        all_data[N+M:] = -U_all

        # --- Expand blocks to element-level COO ---
        rr, cc = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
        row_exp = (all_rows[:, None, None] * K + rr[None, :, :]).ravel()
        col_exp = (all_cols[:, None, None] * K + cc[None, :, :]).ravel()
        data_exp = all_data.ravel()

        L = sp.coo_matrix((data_exp, (row_exp, col_exp)), shape=(dim, dim))
        return L.tocsr()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sparse_sheaf_laplacian.py::TestBuildMatrix -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add atft/topology/sparse_sheaf_laplacian.py tests/test_sparse_sheaf_laplacian.py
git commit -m "feat: add BSR matrix assembly with dense equivalence validation

Implements build_matrix() for vector-fiber sheaf Laplacian. Block structure:
L[i,j] = -U†, L[j,i] = -U, L[i,i] += U†U, L[j,j] += I per edge.
Validated against dense reference at small N. Phase 3 sparse engine (part 2/3)."
```

---

### Task 5: Eigensolver Integration

**Files:**
- Modify: `atft/topology/sparse_sheaf_laplacian.py`
- Modify: `tests/test_sparse_sheaf_laplacian.py`

**Context:** Add shift-invert eigsh with fallback. The primary method `smallest_eigenvalues` returns the k smallest eigenvalues. `spectral_sum` returns their sum — the key metric from the spec. At epsilon=0 (no edges), the Laplacian is zero and all N*K eigenvalues are zero, so we return k zeros directly.

- [ ] **Step 1: Write eigensolver tests**

Add to `tests/test_sparse_sheaf_laplacian.py`:

```python
class TestEigensolver:
    """Tests for smallest_eigenvalues() and spectral_sum()."""

    def test_eigenvalues_sorted_nonneg(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.5, k=10)
        assert len(eigs) == 10
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)  # sorted
        assert np.all(eigs > -1e-10)  # nonneg

    def test_matches_dense_eigenvalues(self):
        """Sparse eigsh should match dense eigh for small problem."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs_sparse = lap.smallest_eigenvalues(1.5, k=10)
        L_dense = lap.build_matrix(1.5).toarray()
        eigs_dense = np.sort(np.linalg.eigvalsh(L_dense))[:10]
        npt.assert_allclose(eigs_sparse, eigs_dense, atol=1e-8)

    def test_eps_zero_returns_zeros(self):
        """At epsilon=0, no edges => all eigenvalues are 0."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(0.0, k=5)
        npt.assert_allclose(eigs, np.zeros(5), atol=1e-14)

    def test_spectral_sum(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.0, k=10)
        s = lap.spectral_sum(1.0, k=10)
        npt.assert_allclose(s, float(np.sum(eigs)), atol=1e-14)

    def test_spectral_sum_eps_zero(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        s = lap.spectral_sum(0.0, k=5)
        assert s == 0.0
```

- [ ] **Step 2: Implement eigensolver methods**

Add to `atft/topology/sparse_sheaf_laplacian.py`:

```python
    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100
    ) -> NDArray[np.float64]:
        """Compute the k smallest eigenvalues of the sheaf Laplacian.

        Uses shift-invert eigsh (targeting eigenvalues near 0) with
        fallback to standard eigsh if shift-invert fails.

        Args:
            epsilon: Rips complex scale parameter.
            k: Number of eigenvalues to compute.

        Returns:
            Sorted 1D array of k smallest eigenvalues (float64).
        """
        N = self._N
        K = self._K
        dim = N * K

        # Degenerate: no edges
        if epsilon <= 0:
            return np.zeros(k, dtype=np.float64)

        L = self.build_matrix(epsilon)

        # Don't request more eigenvalues than matrix dimension allows
        # eigsh requires k < dim for sparse matrices
        k_actual = min(k, dim - 2) if dim > 2 else dim

        if k_actual <= 0:
            return np.zeros(k, dtype=np.float64)

        # If matrix is small enough, use dense eigensolver
        if dim <= 500:
            eigs = np.sort(np.linalg.eigvalsh(L.toarray()).real)
            eigs = np.maximum(eigs[:k], 0.0)
            if len(eigs) < k:
                eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
            return eigs

        # Try shift-invert (targets eigenvalues near sigma)
        try:
            eigs, _ = eigsh(L, k=k_actual, sigma=1e-8, which='LM', tol=1e-6)
            eigs = np.sort(eigs.real)
            # Clamp tiny negatives from numerical noise
            eigs = np.maximum(eigs, 0.0)
        except Exception:
            # Fallback: standard eigsh targeting smallest eigenvalues
            try:
                eigs, _ = eigsh(L, k=k_actual, which='SM', tol=1e-6)
                eigs = np.sort(eigs.real)
                eigs = np.maximum(eigs, 0.0)
            except Exception:
                # Last resort: dense
                eigs = np.sort(np.linalg.eigvalsh(L.toarray()).real)
                eigs = np.maximum(eigs[:k], 0.0)
                if len(eigs) < k:
                    eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
                return eigs

        # Pad with zeros if we got fewer than k
        if len(eigs) < k:
            eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
        return eigs[:k]

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Sum of the k smallest eigenvalues (primary metric).

        This is the total "constraint energy" in the near-kernel of the
        sheaf Laplacian. Lower values indicate more globally consistent
        sections (stronger topological signal).
        """
        return float(np.sum(self.smallest_eigenvalues(epsilon, k)))
```

- [ ] **Step 3: Run eigensolver tests**

Run: `python -m pytest tests/test_sparse_sheaf_laplacian.py::TestEigensolver -v`
Expected: 5 PASSED

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: 171 existing + 22 transport + 12 sparse = 205 PASSED

- [ ] **Step 5: Commit**

```bash
git add atft/topology/sparse_sheaf_laplacian.py tests/test_sparse_sheaf_laplacian.py
git commit -m "feat: add shift-invert eigsh solver to SparseSheafLaplacian

Implements smallest_eigenvalues() with shift-invert eigsh (sigma=1e-8),
fallback to standard eigsh and dense solver. spectral_sum() returns the
primary metric. Phase 3 sparse engine (part 3/3)."
```

---

## Chunk 3: Experiment Script

### Task 6: Phase 3 Superposition Sweep

**Files:**
- Create: `atft/experiments/phase3_superposition_sweep.py`

**Context:** This is the definitive control test. It sweeps sigma and epsilon across zeta zeros, random points, and GUE points using the superposition transport, measuring spectral sums. The script follows the pattern of `phase2b_sheaf.py` but is standalone (configurable, prints progress, computes the contrast ratio R from the spec). No unit tests — this is an experiment script. Include a `--quick` mode for development iteration (K=6, N=30, 3 sigma values).

- [ ] **Step 1: Create the experiment script**

Create `atft/experiments/phase3_superposition_sweep.py`:

```python
#!/usr/bin/env python
"""Phase 3 Superposition Sweep: The Definitive Control Test.

Runs the multi-prime superposition transport across zeta zeros, random
points, and GUE points at multiple sigma and epsilon values. Measures
whether the explicit-formula phase interference creates a genuine
arithmetic signal at sigma=0.5.

Usage:
    python -m atft.experiments.phase3_superposition_sweep          # full run
    python -m atft.experiments.phase3_superposition_sweep --quick   # dev mode

See: docs/superpowers/specs/2026-03-15-atft-phase3-superposition-design.md
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


@dataclass
class Phase3Config:
    """Configuration for the Phase 3 superposition sweep."""
    n_points: int = 9877
    K: int = 50
    sigma_grid: NDArray[np.float64] = field(default_factory=lambda: np.array(
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75]
    ))
    epsilon_grid: NDArray[np.float64] = field(default_factory=lambda: np.array(
        [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    ))
    k_eigenvalues: int = 100
    n_random_trials: int = 5
    n_gue_trials: int = 5
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    seed: int = 42

    @classmethod
    def quick(cls) -> Phase3Config:
        """Quick dev mode: small scale for fast iteration."""
        return cls(
            n_points=30,
            K=6,
            sigma_grid=np.array([0.25, 0.50, 0.75]),
            epsilon_grid=np.array([2.0, 3.0]),
            k_eigenvalues=10,
            n_random_trials=2,
            n_gue_trials=2,
        )


def generate_gue_points(
    n: int, mean_spacing: float, start: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Generate GUE-spaced points using rejection sampling of GUE Wigner surmise.

    GUE (beta=2): P(s) = (32/pi^2) * s^2 * exp(-4*s^2/pi)
    Note: NOT the GOE (beta=1) surmise which has linear repulsion.
    """
    spacings = []
    # GUE Wigner surmise: P(s) = (32/pi^2) * s^2 * exp(-4*s^2/pi)
    # Mode at s = sqrt(pi)/2 ~ 0.886, max density ~ 0.738
    # Use Rayleigh envelope for rejection sampling
    c_reject = 2.0  # rejection constant
    for _ in range(n - 1):
        while True:
            # Proposal: Rayleigh(sigma=sqrt(pi/8)) has peak at same location
            s = rng.rayleigh(scale=np.sqrt(np.pi / 8))
            # Target: (32/pi^2) * s^2 * exp(-4*s^2/pi)
            target = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            # Rayleigh PDF: (s / sigma^2) * exp(-s^2/(2*sigma^2))
            sigma2 = np.pi / 8
            proposal = (s / sigma2) * np.exp(-s**2 / (2 * sigma2))
            if proposal > 0 and rng.random() < target / (c_reject * proposal):
                spacings.append(s * mean_spacing)
                break
    return np.cumsum(np.array([start] + spacings))


def run_sigma_sweep(
    zeros: NDArray[np.float64],
    config: Phase3Config,
    normalize: bool,
    label: str,
) -> dict[tuple[float, float], dict[str, float]]:
    """Run sigma x epsilon sweep for a single point set.

    Returns dict mapping (sigma, epsilon) -> {
        'spectral_sum': float,
        'kernel_dim': int,  # beta_0 = #{lambda_i < tau}
    }
    """
    results: dict[tuple[float, float], dict[str, float]] = {}

    for sigma in config.sigma_grid:
        builder = TransportMapBuilder(K=config.K, sigma=sigma)
        for eps in config.epsilon_grid:
            t0 = time.time()
            lap = SparseSheafLaplacian(
                builder, zeros,
                transport_mode="superposition",
                normalize=normalize,
            )
            eigs = lap.smallest_eigenvalues(eps, k=config.k_eigenvalues)
            s = float(np.sum(eigs))
            # Kernel dimension: eigenvalues below threshold
            tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
            beta_0 = int(np.sum(eigs < tau))
            elapsed = time.time() - t0
            results[(sigma, eps)] = {'spectral_sum': s, 'kernel_dim': beta_0}
            print(f"    sigma={sigma:.2f} eps={eps:.1f}: S={s:.4f} b0={beta_0} ({elapsed:.1f}s)")
            sys.stdout.flush()

    return results


def compute_symmetrized(
    results: dict[tuple[float, float], dict[str, float]],
    sigma_grid: NDArray[np.float64],
    epsilon_grid: NDArray[np.float64],
) -> dict[tuple[float, float], float]:
    """Compute symmetrized spectral sum S_sym = [S(sigma) + S(1-sigma)] / 2."""
    sym: dict[tuple[float, float], float] = {}
    for sigma in sigma_grid:
        s_mirror = round(1.0 - sigma, 3)
        for eps in epsilon_grid:
            s_val = results.get((sigma, eps), {}).get('spectral_sum', 0.0)
            s_mirr = results.get((s_mirror, eps), {}).get('spectral_sum', s_val)
            sym[(sigma, eps)] = (s_val + s_mirr) / 2
    return sym


def compute_contrast(
    results: dict[tuple[float, float], dict[str, float]],
    epsilon_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute contrast C = [S(0.5) - S(0.25)] / S(0.5) at each epsilon."""
    contrasts = []
    for eps in epsilon_grid:
        s_half = results.get((0.50, eps), {}).get('spectral_sum', 0.0)
        s_quarter = results.get((0.25, eps), {}).get('spectral_sum', 0.0)
        if abs(s_half) > 1e-15:
            c = (s_half - s_quarter) / s_half
        else:
            c = 0.0
        contrasts.append(c)
    return np.array(contrasts)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Superposition Sweep")
    parser.add_argument("--quick", action="store_true", help="Quick dev mode")
    args = parser.parse_args()

    config = Phase3Config.quick() if args.quick else Phase3Config()

    print("=" * 70)
    print("  ATFT PHASE 3: SUPERPOSITION & SCALE")
    print("  Multi-prime phase interference control test")
    print("=" * 70)
    print(f"\n  N={config.n_points}, K={config.K}, k_eig={config.k_eigenvalues}")
    print(f"  sigma_grid: {config.sigma_grid}")
    print(f"  epsilon_grid: {config.epsilon_grid}")
    print(f"  Normalization: both True and False")

    # Load zeta zeros
    source = ZetaZerosSource(config.zeta_data_path)
    cloud = source.generate(config.n_points)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    print(f"\n  Zeta zeros loaded: N={len(zeta_zeros)}, mean_spacing={mean_spacing:.4f}")

    rng = np.random.default_rng(config.seed)

    for normalize in [True, False]:
        norm_label = "NORMALIZED" if normalize else "UNNORMALIZED"
        print(f"\n{'=' * 70}")
        print(f"  {norm_label} SUPERPOSITION")
        print(f"{'=' * 70}")

        # --- Zeta zeros ---
        print(f"\n  [ZETA ZEROS]")
        zeta_results = run_sigma_sweep(zeta_zeros, config, normalize, "Zeta")

        # --- Random points ---
        all_random_results = []
        for trial in range(config.n_random_trials):
            print(f"\n  [RANDOM trial {trial + 1}]")
            rand_pts = np.sort(rng.uniform(
                zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros)
            ))
            r = run_sigma_sweep(rand_pts, config, normalize, f"Random {trial+1}")
            all_random_results.append(r)

        # --- GUE points ---
        all_gue_results = []
        for trial in range(config.n_gue_trials):
            print(f"\n  [GUE trial {trial + 1}]")
            gue_pts = generate_gue_points(
                len(zeta_zeros), mean_spacing, zeta_zeros.min(), rng
            )
            r = run_sigma_sweep(gue_pts, config, normalize, f"GUE {trial+1}")
            all_gue_results.append(r)

        # --- Compute contrasts ---
        zeta_contrast = compute_contrast(zeta_results, config.epsilon_grid)
        random_contrasts = [
            compute_contrast(r, config.epsilon_grid) for r in all_random_results
        ]
        gue_contrasts = [
            compute_contrast(r, config.epsilon_grid) for r in all_gue_results
        ]

        # Average control contrasts
        all_control = np.array(random_contrasts + gue_contrasts)  # (n_trials, n_eps)
        mean_control = np.mean(all_control, axis=0)

        # Signal strength R per epsilon
        R_values = np.where(
            np.abs(mean_control) > 1e-15,
            zeta_contrast / mean_control,
            0.0,
        )

        # --- Symmetrized spectral sum ---
        zeta_sym = compute_symmetrized(
            zeta_results, config.sigma_grid, config.epsilon_grid
        )

        # --- Summary ---
        print(f"\n  {'=' * 60}")
        print(f"  SUMMARY ({norm_label})")
        print(f"  {'=' * 60}")

        # Symmetrized spectral sum table (for Phase 2 comparison)
        print(f"\n  Symmetrized spectral sum S_sym at eps={config.epsilon_grid[1]:.1f}:")
        ref_eps = config.epsilon_grid[1]
        for sigma in config.sigma_grid:
            if sigma <= 0.5:
                s_sym = zeta_sym.get((sigma, ref_eps), 0.0)
                marker = " <--" if abs(sigma - 0.5) < 1e-6 else ""
                print(f"    sigma={sigma:.2f}: S_sym={s_sym:.4f}{marker}")

        # Kernel dimension at sigma=0.5
        print(f"\n  Kernel dimension beta_0 at sigma=0.5:")
        for eps in config.epsilon_grid:
            b0 = zeta_results.get((0.50, eps), {}).get('kernel_dim', 0)
            print(f"    eps={eps:.1f}: beta_0={b0}")

        print(f"\n  {'epsilon':>8} | {'C(zeta)':>10} | {'C(ctrl)':>10} | {'R':>10}")
        print(f"  {'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
        for i, eps in enumerate(config.epsilon_grid):
            marker = " <--" if R_values[i] > 2.0 else ""
            print(f"  {eps:8.1f} | {zeta_contrast[i]:10.4f} | "
                  f"{mean_control[i]:10.4f} | {R_values[i]:10.4f}{marker}")

        # Outcome determination
        strong_count = int(np.sum(R_values > 2.0))
        weak_count = int(np.sum(R_values > 1.0))
        n_eps = len(config.epsilon_grid)

        print(f"\n  R > 2.0 at {strong_count}/{n_eps} epsilon values")
        print(f"  R > 1.0 at {weak_count}/{n_eps} epsilon values")

        if strong_count > n_eps * 2 // 3:
            print(f"\n  ** OUTCOME A: STRONG SIGNAL **")
            print(f"  The primes are singing at sigma=0.5.")
        elif weak_count > n_eps * 2 // 3:
            print(f"\n  ** OUTCOME B: WEAK SIGNAL **")
            print(f"  Arithmetic structure detected but needs refinement.")
        else:
            print(f"\n  ** OUTCOME C: NO SIGNAL **")
            print(f"  Superposition transport does not distinguish zeta zeros.")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test in quick mode**

Run: `python -m atft.experiments.phase3_superposition_sweep --quick`
Expected: Completes in < 2 minutes, prints sigma-sweep tables and summary for both normalized and unnormalized modes.

- [ ] **Step 3: Verify full test suite still passes**

Run: `python -m pytest tests/ -v`
Expected: All 205 tests PASSED (171 existing + 34 new, no regressions)

- [ ] **Step 4: Commit**

```bash
git add atft/experiments/phase3_superposition_sweep.py
git commit -m "feat: add Phase 3 superposition sweep experiment script

Implements the definitive control test: zeta zeros vs random vs GUE with
multi-prime superposition transport. Computes contrast ratio C and signal
strength R at each epsilon. Both normalized and unnormalized modes tested.
Includes --quick flag for fast dev iteration (K=6, N=30)."
```

- [ ] **Step 5: Run full experiment (when ready)**

Run: `python -m atft.experiments.phase3_superposition_sweep`
Expected: ~6 hours. Results determine Outcome A (breakthrough), B (promising), or C (negative).

---

## Post-Implementation Checklist

- [ ] All 171 existing tests still pass (no Phase 2 regressions)
- [ ] All new tests pass (~34 new tests)
- [ ] Quick mode (`--quick`) completes in < 2 minutes
- [ ] Dense equivalence validated at small N for both normalized and unnormalized
- [ ] Hermiticity and PSD verified for sparse Laplacian
- [ ] Experiment script prints readable progress and summary tables
