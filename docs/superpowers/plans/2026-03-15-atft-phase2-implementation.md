# ATFT Phase 2: Sheaf-Valued Persistent Homology Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sheaf-valued persistent homology with u(K) Lie algebra fibers and prime-weighted transport maps, enabling the σ-sweep experiment that tests whether RH can be reformulated as a topological phase transition at σ = 1/2.

**Architecture:** A matrix-free `LinearOperator` computes the sheaf Laplacian L_F via on-the-fly coboundary adjoint evaluation (O(|E|·K³) per matvec), fed into LOBPCG/eigsh for kernel dimension extraction. An ε-sweep orchestrator produces sheaf Betti curves β₀^F(ε), and a σ-sweep orchestrator produces the 2D phase diagram β₀^F(ε, σ). Phase 2a (abelian eigenbasis diagnostic) validates the Laplacian against Phase 1's scalar Betti curve before Phase 2b (full non-abelian σ-sweep) runs.

**Tech Stack:** NumPy (complex128 linear algebra), SciPy (LinearOperator, LOBPCG, eigsh), Matplotlib (heatmaps, Betti curves), existing ATFT Phase 1 infrastructure (sources, unfolding, analytical H_0).

**Spec:** `docs/superpowers/specs/2026-03-15-atft-phase2-sheaf-design.md`

**Existing state:** `transport_maps.py` is COMPLETE with 27 passing tests (119 total with Phase 1's 92). All Phase 2 code builds on `TransportMapBuilder`.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| **DONE** | `atft/topology/transport_maps.py` | ρ(p), G_p(σ), A(σ), U(Δγ) — 27 tests passing |
| Modify | `atft/core/types.py` | ADD `SheafBettiCurve`, `SheafValidationResult` frozen dataclasses |
| Create | `tests/test_sheaf_types.py` | Tests for the two new types |
| Create | `atft/topology/sheaf_laplacian.py` | Matrix-free L_F operator, LOBPCG kernel computation |
| Create | `tests/test_sheaf_laplacian.py` | Dense equivalence, K=1 sanity, ε=0 edge case, Hermiticity |
| Create | `atft/topology/sheaf_ph.py` | ε-sweep and σ-sweep orchestrators |
| Create | `tests/test_sheaf_ph.py` | Sweep monotonicity, K=1 reproduces scalar, σ-sweep shape |
| Create | `atft/experiments/phase2a_abelian.py` | Eigenbasis diagnostic: K² scalar Laplacians, resonance matrix |
| Create | `tests/test_phase2a_abelian.py` | Diagonal blocks match Phase 1, resonance matrix shape |
| Create | `atft/experiments/phase2b_sheaf.py` | Full σ-sweep experiment orchestrator |
| Create | `tests/test_phase2b_sheaf.py` | End-to-end integration at tiny scale |
| Modify | `atft/visualization/plots.py` | ADD `plot_sheaf_betti_curves`, `plot_sigma_peak`, `plot_resonance_matrix` |
| Create | `tests/test_phase2_plots.py` | Smoke tests for new plot functions |
| Create | `run_phase2a.py` | CLI runner for Phase 2a abelian diagnostic |
| Create | `run_phase2b.py` | CLI runner for Phase 2b σ-sweep experiment |

---

## Chunk 1: Core Types + Sheaf Laplacian

### Task 1: Add SheafBettiCurve and SheafValidationResult to core types

**Files:**
- Modify: `atft/core/types.py`
- Create: `tests/test_sheaf_types.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sheaf_types.py
"""Tests for Phase 2 sheaf types."""
from __future__ import annotations

import numpy as np
import pytest

from atft.core.types import SheafBettiCurve, SheafValidationResult


class TestSheafBettiCurve:
    def test_creation(self):
        curve = SheafBettiCurve(
            epsilon_grid=np.linspace(0, 3.0, 10),
            kernel_dimensions=np.array([400, 350, 200, 100, 50, 20, 10, 5, 2, 1]),
            smallest_eigenvalues=np.random.rand(10, 5),
            sigma=0.5,
            K=20,
        )
        assert curve.K == 20
        assert curve.sigma == 0.5
        assert len(curve.epsilon_grid) == 10
        assert len(curve.kernel_dimensions) == 10

    def test_frozen(self):
        curve = SheafBettiCurve(
            epsilon_grid=np.array([0.0, 1.0]),
            kernel_dimensions=np.array([10, 5]),
            smallest_eigenvalues=np.zeros((2, 3)),
            sigma=0.5,
            K=5,
        )
        with pytest.raises(AttributeError):
            curve.sigma = 0.7


class TestSheafValidationResult:
    def test_creation(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            epsilon_grid=np.linspace(0, 3.0, 10),
            betti_heatmap=np.zeros((3, 10), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=42,
            is_unique_peak=True,
        )
        assert result.peak_sigma == 0.5
        assert result.is_unique_peak is True
        assert result.betti_heatmap.shape == (3, 10)

    def test_metadata_default(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.5]),
            epsilon_grid=np.array([0.0]),
            betti_heatmap=np.zeros((1, 1), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=0,
            is_unique_peak=True,
        )
        assert result.metadata == {}

    def test_frozen(self):
        result = SheafValidationResult(
            sigma_grid=np.array([0.5]),
            epsilon_grid=np.array([0.0]),
            betti_heatmap=np.zeros((1, 1), dtype=np.int64),
            peak_sigma=0.5,
            peak_kernel_dim=0,
            is_unique_peak=True,
        )
        with pytest.raises(AttributeError):
            result.peak_sigma = 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sheaf_types.py -v`
Expected: FAIL with `ImportError: cannot import name 'SheafBettiCurve'`

- [ ] **Step 3: Add the two new dataclasses to types.py**

Append to `atft/core/types.py` (after the existing `ValidationResult` class):

```python
@dataclass(frozen=True)
class SheafBettiCurve:
    """Sheaf Betti number beta_0^F(epsilon) across filtration scales."""

    epsilon_grid: NDArray[np.float64]
    kernel_dimensions: NDArray[np.int64]
    smallest_eigenvalues: NDArray[np.float64]  # shape (n_steps, m)
    sigma: float
    K: int


@dataclass(frozen=True)
class SheafValidationResult:
    """Output of the sigma-sweep experiment."""

    sigma_grid: NDArray[np.float64]
    epsilon_grid: NDArray[np.float64]
    betti_heatmap: NDArray[np.int64]  # shape (n_sigma, n_epsilon)
    peak_sigma: float
    peak_kernel_dim: int
    is_unique_peak: bool
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sheaf_types.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Run full suite to verify no regressions**

Run: `python -m pytest tests/ -v --tb=short -q`
Expected: 124 passed (119 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add atft/core/types.py tests/test_sheaf_types.py
git commit -m "feat: add SheafBettiCurve and SheafValidationResult types for Phase 2"
```

---

### Task 2: Implement the matrix-free SheafLaplacian

This is the computational core of Phase 2. The sheaf Laplacian L_F is a (N·K²) × (N·K²) Hermitian positive-semidefinite operator that is NEVER assembled as an explicit matrix. Instead, the matrix-vector product y = L_F · x is computed by iterating over edges and applying K×K conjugation directly.

**Files:**
- Create: `atft/topology/sheaf_laplacian.py`
- Create: `tests/test_sheaf_laplacian.py`

**Mathematical background (for the implementer):**
- Spec Section 2.7-2.8: Coboundary operator δ₀ and Laplacian L_F = δ₀†δ₀
- Spec Section 3.5: The matrix-free matvec pseudocode
- The matvec loop: for each edge (i,j) with |γ̃_i - γ̃_j| ≤ ε:
  - `diff = x[j] - U_{ij} @ x[i] @ U_{ij}†`
  - `y[j] += diff`
  - `y[i] -= U_{ij}† @ diff @ U_{ij}`
- x is reshaped from flat (N·K²,) complex to (N, K, K) complex for the edge loop, then flattened back

**Key design decisions:**
- Transport matrices U_{ij} are precomputed and cached per ε value (or per call, since edges change with ε)
- `smallest_eigenvalues` uses LOBPCG first, falls back to eigsh with shift-invert if LOBPCG doesn't converge
- `kernel_dimension` counts eigenvalues below τ = 1e-6 · ‖L_F‖_F (Frobenius norm estimated from edges)
- At ε=0, no edges exist → L_F = 0 → kernel = N·K², bypass eigensolver entirely

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sheaf_laplacian.py
"""Tests for sheaf_laplacian.py — matrix-free L_F operator.

Key tests:
  1. Dense equivalence: matvec matches explicit L_F assembly at tiny scale
  2. K=1 sanity: reduces to standard graph Laplacian
  3. epsilon=0: degenerate case, full kernel
  4. Hermiticity: L_F preserves Hermitian structure
  5. Positive semidefiniteness: all eigenvalues >= 0
  6. Solver fallback: auto mode works
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import LinearOperator

from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def _build_explicit_laplacian_via_coboundary(
    zeros: np.ndarray, builder: TransportMapBuilder, epsilon: float
) -> np.ndarray:
    """Build L_F = δ₀† δ₀ via explicit coboundary assembly."""
    N = len(zeros)
    K = builder.K
    vertex_dim = N * K * K

    # Collect edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(zeros[j] - zeros[i]) <= epsilon:
                edges.append((i, j))

    n_edges = len(edges)
    if n_edges == 0:
        return np.zeros((vertex_dim, vertex_dim), dtype=np.complex128)

    edge_dim = n_edges * K * K

    # Build δ₀ as explicit matrix: edge_dim × vertex_dim
    delta = np.zeros((edge_dim, vertex_dim), dtype=np.complex128)

    for e_idx, (i, j) in enumerate(edges):
        U = builder.transport(zeros[j] - zeros[i])
        Uh = U.conj().T
        for a in range(K):
            for b in range(K):
                row = (e_idx * K + a) * K + b
                # Identity at j: +x[j]_{ab}
                col_j = (j * K + a) * K + b
                delta[row, col_j] += 1.0
                # Transport at i: -(U x[i] U†)_{ab} = -sum_{cd} U_{ac} x[i]_{cd} Uh_{db}
                for c in range(K):
                    for d in range(K):
                        col_i = (i * K + c) * K + d
                        delta[row, col_i] -= U[a, c] * Uh[d, b]

    # L_F = δ₀† δ₀
    return delta.conj().T @ delta


class TestSheafLaplacianMatvec:
    """Test that matvec matches explicit dense assembly."""

    def test_dense_equivalence_K2_N3(self):
        """For N=3, K=2: matvec(x) must match explicit L_F @ x."""
        zeros = np.array([0.0, 0.5, 1.2])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        K = 2
        N = 3
        dim = N * K * K

        # Build explicit L_F
        epsilon = 1.5  # all pairs within range
        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)

        # Random complex input
        rng = np.random.default_rng(42)
        x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        # Compare
        y_dense = L_dense @ x
        y_matvec = lap.matvec(x, epsilon)
        assert_allclose(y_matvec, y_dense, atol=1e-12)

    def test_dense_equivalence_K3_N4(self):
        """Slightly larger: N=4, K=3."""
        zeros = np.array([0.0, 0.3, 0.8, 1.5])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = 4 * 3 * 3
        epsilon = 1.0

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)
        rng = np.random.default_rng(99)
        x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        y_dense = L_dense @ x
        y_matvec = lap.matvec(x, epsilon)
        assert_allclose(y_matvec, y_dense, atol=1e-12)

    def test_matvec_with_partial_edges(self):
        """Only some edges present (not all pairs within epsilon)."""
        zeros = np.array([0.0, 0.5, 2.0, 2.3])  # gap 0→2 = 2.0 > epsilon
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = 4 * 2 * 2
        epsilon = 1.0  # edges: (0,1), (2,3) only

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        y_dense = L_dense @ x
        y_matvec = lap.matvec(x, epsilon)
        assert_allclose(y_matvec, y_dense, atol=1e-12)


class TestSheafLaplacianProperties:
    """Test mathematical properties of L_F."""

    def test_hermitian(self):
        """L_F must be Hermitian: L_F = L_F†."""
        zeros = np.array([0.0, 0.5, 1.2])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = 3 * 2 * 2
        epsilon = 1.5

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)
        assert_allclose(L_dense, L_dense.conj().T, atol=1e-13)

    def test_positive_semidefinite(self):
        """All eigenvalues of L_F must be >= 0."""
        zeros = np.array([0.0, 0.4, 1.0, 1.6])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        epsilon = 2.0

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)
        eigenvalues = np.linalg.eigvalsh(L_dense)
        assert np.all(eigenvalues >= -1e-12)

    def test_kernel_nonempty_at_small_epsilon(self):
        """At small epsilon, few edges → large kernel."""
        zeros = np.array([0.0, 0.1, 5.0, 5.1])  # two clusters
        builder = TransportMapBuilder(K=2, sigma=0.5)
        epsilon = 0.5  # only intra-cluster edges

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon)
        eigenvalues = np.linalg.eigvalsh(L_dense)
        n_kernel = np.sum(eigenvalues < 1e-10)
        # 2 clusters × K² = 2 × 4 = 8 kernel dimensions
        # But each cluster has 1 edge, so kernel within each cluster is K²
        # (global sections on each connected component)
        assert n_kernel >= 2 * 2 * 2  # at least 2 × K² = 8

    def test_matvec_output_shape(self):
        """matvec must return same shape as input."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = 3 * 3 * 3
        x = np.ones(dim, dtype=np.complex128)
        y = lap.matvec(x, epsilon=3.0)
        assert y.shape == (dim,)
        assert y.dtype == np.complex128


class TestSheafLaplacianK1:
    """K=1 sanity check: L_F must reduce to the standard graph Laplacian."""

    def test_k1_matches_graph_laplacian(self):
        """At K=1, A(σ)=0, U=I, so L_F = standard graph Laplacian."""
        zeros = np.array([0.0, 1.0, 2.5, 4.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        N = 4
        epsilon = 2.0  # edges: (0,1), (1,2)

        # Build standard graph Laplacian
        L_graph = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(zeros[j] - zeros[i]) <= epsilon:
                    L_graph[i, i] += 1
                    L_graph[j, j] += 1
                    L_graph[i, j] -= 1
                    L_graph[j, i] -= 1

        # At K=1, each "K×K matrix" is just a scalar, so dim = N
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 0j  # complex for interface compatibility
        y_graph = L_graph @ x.real
        y_sheaf = lap.matvec(x, epsilon)
        assert_allclose(y_sheaf.real, y_graph, atol=1e-14)
        assert_allclose(y_sheaf.imag, 0.0, atol=1e-14)

    def test_k1_eigenvalues_match_graph_laplacian(self):
        """Eigenvalues at K=1 must match standard graph Laplacian eigenvalues."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        epsilon = 1.5  # edges: consecutive pairs only

        # Standard graph Laplacian for path graph
        N = 5
        L_graph = np.zeros((N, N))
        for i in range(N - 1):
            L_graph[i, i] += 1
            L_graph[i + 1, i + 1] += 1
            L_graph[i, i + 1] -= 1
            L_graph[i + 1, i] -= 1
        expected_eigs = np.sort(np.linalg.eigvalsh(L_graph))

        sheaf_eigs = lap.smallest_eigenvalues(epsilon, m=N)
        sheaf_eigs = np.sort(sheaf_eigs.real)
        assert_allclose(sheaf_eigs, expected_eigs, atol=1e-10)


class TestSheafLaplacianEpsilonZero:
    """ε=0 degenerate case: no edges, full kernel."""

    def test_epsilon_zero_matvec_is_zero(self):
        """matvec at ε=0 returns all zeros (no edges, L_F=0)."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = 3 * 2 * 2
        x = np.ones(dim, dtype=np.complex128)
        y = lap.matvec(x, epsilon=0.0)
        assert_allclose(y, np.zeros(dim), atol=1e-15)

    def test_kernel_dimension_at_epsilon_zero(self):
        """kernel_dimension at ε=0 must be N·K² (full space)."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        assert lap.kernel_dimension(epsilon=0.0) == 3 * 3 * 3


class TestAsLinearOperator:
    """Test the LinearOperator wrapper."""

    def test_returns_linear_operator(self):
        zeros = np.array([0.0, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        op = lap.as_linear_operator(epsilon=1.5)
        assert isinstance(op, LinearOperator)
        assert op.shape == (2 * 4, 2 * 4)
        assert op.dtype == np.complex128

    def test_linear_operator_matches_matvec(self):
        """LinearOperator.matvec must match SheafLaplacian.matvec."""
        zeros = np.array([0.0, 0.5, 1.2])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        epsilon = 1.5
        op = lap.as_linear_operator(epsilon)
        dim = 3 * 2 * 2

        rng = np.random.default_rng(42)
        x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

        y_direct = lap.matvec(x, epsilon)
        y_operator = op @ x
        assert_allclose(y_operator, y_direct, atol=1e-15)


class TestSmallestEigenvalues:
    """Test the eigensolver interface."""

    def test_returns_sorted_eigenvalues(self):
        """smallest_eigenvalues must return sorted real values."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=1.0, m=5)
        assert len(eigs) == 5
        assert np.all(np.diff(eigs) >= -1e-12)  # sorted ascending
        assert np.all(eigs >= -1e-12)  # non-negative

    def test_eigsh_solver(self):
        """Explicit eigsh solver must work."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=1.0, m=4, solver="eigsh")
        assert len(eigs) == 4
        assert np.all(eigs >= -1e-12)

    def test_auto_solver(self):
        """Auto solver must not crash."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=1.0, m=3, solver="auto")
        assert len(eigs) == 3


class TestKernelDimension:
    """Test kernel dimension counting."""

    def test_full_kernel_no_edges(self):
        """No edges → full kernel."""
        zeros = np.array([0.0, 100.0])  # gap >> any reasonable ε
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim = lap.kernel_dimension(epsilon=0.001)
        assert dim == 2 * 2 * 2  # N·K²

    def test_kernel_shrinks_with_more_edges(self):
        """More edges → smaller kernel."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        dim_small = lap.kernel_dimension(epsilon=0.6)  # few edges
        dim_large = lap.kernel_dimension(epsilon=2.5)  # many edges
        assert dim_small >= dim_large


class TestExtractGlobalSections:
    """Test global section extraction."""

    def test_returns_list_of_matrices(self):
        """Each global section should be reshaped to (N, K, K)."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        sections = lap.extract_global_sections(epsilon=1.5)
        # Should return a list; each element is (N, K, K) complex
        for sec in sections:
            assert sec.shape == (3, 2, 2)
            assert sec.dtype == np.complex128

    def test_global_section_is_in_kernel(self):
        """L_F @ section must be approximately zero."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        lap = SheafLaplacian(builder, zeros)
        epsilon = 1.5
        sections = lap.extract_global_sections(epsilon=epsilon)
        for sec in sections:
            x = sec.flatten()
            y = lap.matvec(x, epsilon)
            assert_allclose(np.abs(y), 0.0, atol=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sheaf_laplacian.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atft.topology.sheaf_laplacian'`

- [ ] **Step 3: Implement sheaf_laplacian.py**

```python
# atft/topology/sheaf_laplacian.py
"""Matrix-free sheaf Laplacian operator and kernel computation.

Implements L_F = delta_0^dagger delta_0 as a LinearOperator whose matvec
iterates over Rips edges and applies K×K conjugation directly, avoiding
O(K^4) Kronecker memory. Cost: O(|E| * K^3) per matvec.

See spec Section 3.5 for the matvec pseudocode.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

from atft.topology.transport_maps import TransportMapBuilder


class SheafLaplacian:
    """Matrix-free sheaf Laplacian on the Vietoris-Rips complex.

    Args:
        transport_builder: Prebuilt TransportMapBuilder with cached eigendecomp.
        unfolded_zeros: 1D array of N sorted unfolded zeta zeros.
    """

    def __init__(
        self,
        transport_builder: TransportMapBuilder,
        unfolded_zeros: NDArray[np.float64],
    ) -> None:
        self._builder = transport_builder
        self._zeros = np.asarray(unfolded_zeros, dtype=np.float64).ravel()
        self._N = len(self._zeros)
        self._K = transport_builder.K
        self._dim = self._N * self._K * self._K

        # Ensure eigendecomp is cached
        transport_builder.build_generator_sum()

    @property
    def N(self) -> int:
        return self._N

    @property
    def K(self) -> int:
        return self._K

    @property
    def dim(self) -> int:
        return self._dim

    def _edges(self, epsilon: float) -> list[tuple[int, int]]:
        """Return sorted edges (i, j) with i < j and |zeros[j] - zeros[i]| <= epsilon."""
        edges = []
        # Since zeros are sorted, we can break early
        for i in range(self._N):
            for j in range(i + 1, self._N):
                gap = self._zeros[j] - self._zeros[i]
                if gap > epsilon:
                    break  # zeros are sorted, all subsequent j have larger gap
                edges.append((i, j))
        return edges

    def matvec(self, x: NDArray[np.complex128], epsilon: float) -> NDArray[np.complex128]:
        """Compute y = L_F(epsilon) @ x without assembling L_F.

        x is a flat complex array of length N*K*K. It is reshaped to (N, K, K),
        the edge loop computes the coboundary adjoint, and y is flattened back.
        """
        K = self._K
        N = self._N
        x_mat = np.asarray(x, dtype=np.complex128).reshape(N, K, K)
        y_mat = np.zeros_like(x_mat)

        for i, j in self._edges(epsilon):
            delta_gamma = self._zeros[j] - self._zeros[i]
            U = self._builder.transport(delta_gamma)
            Uh = U.conj().T

            # Coboundary mismatch: diff = x[j] - U @ x[i] @ U†
            transported = U @ x_mat[i] @ Uh
            diff = x_mat[j] - transported

            # Adjoint of coboundary:
            y_mat[j] += diff                    # identity side
            y_mat[i] -= Uh @ diff @ U           # transport side

        return y_mat.ravel()

    def as_linear_operator(self, epsilon: float) -> LinearOperator:
        """Wrap matvec as a scipy LinearOperator for LOBPCG/eigsh."""
        def mv(x):
            return self.matvec(x, epsilon)

        return LinearOperator(
            shape=(self._dim, self._dim),
            matvec=mv,
            rmatvec=mv,  # L_F is Hermitian
            dtype=np.complex128,
        )

    def smallest_eigenvalues(
        self,
        epsilon: float,
        m: int = 20,
        solver: str = "auto",
    ) -> NDArray[np.float64]:
        """Return the m smallest eigenvalues of L_F(epsilon).

        Args:
            epsilon: Rips filtration scale.
            m: Number of smallest eigenvalues to compute.
            solver: "lobpcg", "eigsh", or "auto". Auto tries LOBPCG first,
                    falls back to eigsh if it doesn't converge.
        """
        if epsilon <= 0 or len(self._edges(epsilon)) == 0:
            return np.zeros(min(m, self._dim), dtype=np.float64)

        # Clamp m to dimension
        m = min(m, self._dim - 1) if self._dim > 1 else 1

        op = self.as_linear_operator(epsilon)

        if solver == "lobpcg" or solver == "auto":
            try:
                rng = np.random.default_rng(0)
                X0 = rng.standard_normal((self._dim, m)) + 1j * rng.standard_normal((self._dim, m))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eigenvalues, _ = lobpcg(op, X0, largest=False, maxiter=1000, tol=1e-10)
                eigenvalues = np.sort(np.real(eigenvalues))
                return eigenvalues
            except Exception:
                if solver == "lobpcg":
                    raise
                # Fall through to eigsh

        # eigsh fallback: which="SM" finds smallest-magnitude eigenvalues.
        # Note: shift-invert (sigma=0.0, which='LM') would be faster but requires
        # an explicit inverse/factorization, which is unavailable for LinearOperator.
        # which="SM" is the correct fallback for operator-only interfaces.
        eigenvalues = eigsh(op, k=m, which="SM", return_eigenvectors=False)
        return np.sort(np.real(eigenvalues))

    def kernel_dimension(
        self,
        epsilon: float,
        tol: float | None = None,
        m: int = 20,
    ) -> int:
        """Count eigenvalues below tolerance tau.

        If tol is None, uses tau = 1e-6 * frobenius_norm_estimate(L_F).
        """
        if epsilon <= 0 or len(self._edges(epsilon)) == 0:
            return self._dim  # Full kernel: L_F = 0

        eigs = self.smallest_eigenvalues(epsilon, m=m)

        if tol is None:
            tol = self.frobenius_norm_estimate(epsilon) * 1e-6
            if tol == 0:
                tol = 1e-12

        return int(np.sum(np.abs(eigs) < tol))

    def frobenius_norm_estimate(self, epsilon: float) -> float:
        """Estimate ||L_F||_F from edge contributions in O(|E|*K^2)."""
        # Each edge contributes 2 * K^2 to the Frobenius norm squared
        # (identity block + transport block), but we approximate from
        # the number of edges and the transport matrix norms
        edges = self._edges(epsilon)
        if not edges:
            return 0.0

        total = 0.0
        K = self._K
        for i, j in edges:
            # Each edge contributes ~2*K to the diagonal of L_F
            total += 2.0 * K
        return np.sqrt(total * K)

    def extract_global_sections(
        self,
        epsilon: float,
        tol: float | None = None,
    ) -> list[NDArray[np.complex128]]:
        """Return eigenvectors in ker(L_F), reshaped to (N, K, K)."""
        if epsilon <= 0 or len(self._edges(epsilon)) == 0:
            # Full kernel — return standard basis (too large to be useful)
            return []

        m = 20
        m = min(m, self._dim - 1) if self._dim > 1 else 1
        op = self.as_linear_operator(epsilon)

        try:
            rng = np.random.default_rng(0)
            X0 = rng.standard_normal((self._dim, m)) + 1j * rng.standard_normal((self._dim, m))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eigenvalues, eigenvectors = lobpcg(op, X0, largest=False, maxiter=1000, tol=1e-10)
        except Exception:
            eigenvalues, eigenvectors = eigsh(op, k=m, which="SM")

        if tol is None:
            tol = self.frobenius_norm_estimate(epsilon) * 1e-6
            if tol == 0:
                tol = 1e-12

        sections = []
        for i, val in enumerate(np.real(eigenvalues)):
            if abs(val) < tol:
                vec = eigenvectors[:, i]
                sections.append(vec.reshape(self._N, self._K, self._K))

        return sections
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sheaf_laplacian.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Run full suite to verify no regressions**

Run: `python -m pytest tests/ -q --tb=short`
Expected: All tests pass (124 + new sheaf_laplacian tests)

- [ ] **Step 6: Commit**

```bash
git add atft/topology/sheaf_laplacian.py tests/test_sheaf_laplacian.py
git commit -m "feat: implement matrix-free sheaf Laplacian with LOBPCG kernel computation"
```

---

## Chunk 2: SheafPH Orchestrator + Phase 2a Abelian Diagnostic

### Task 3: Implement the SheafPH epsilon/sigma sweep orchestrator

**Files:**
- Create: `atft/topology/sheaf_ph.py`
- Create: `tests/test_sheaf_ph.py`

**Mathematical background:**
- SheafPH.sweep: for each ε in the grid, compute kernel_dimension(ε) → produces SheafBettiCurve
- SheafPH.sigma_sweep: for each σ, rebuild TransportMapBuilder, run sweep → produces 2D heatmap
- At ε=0: bypass eigensolver, record N·K² directly
- The Betti curve β₀^F(ε) should be monotonically non-increasing (more edges = more constraints)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sheaf_ph.py
"""Tests for sheaf_ph.py — epsilon and sigma sweep orchestrators."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.core.types import SheafBettiCurve, SheafValidationResult
from atft.topology.sheaf_ph import SheafPH
from atft.topology.transport_maps import TransportMapBuilder


class TestSheafPHSweep:
    """Test the epsilon sweep."""

    def test_returns_sheaf_betti_curve(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 5.0, 10)
        curve = ph.sweep(eps_grid, m=5)
        assert isinstance(curve, SheafBettiCurve)
        assert len(curve.kernel_dimensions) == 10
        assert curve.sigma == 0.5
        assert curve.K == 2

    def test_epsilon_zero_gives_full_kernel(self):
        """At epsilon=0, kernel dim = N*K^2."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 1.0, 2.0])
        curve = ph.sweep(eps_grid, m=5)
        assert curve.kernel_dimensions[0] == 3 * 2 * 2  # N * K^2 = 12

    def test_betti_curve_monotonically_nonincreasing(self):
        """More edges means tighter constraints — kernel can only shrink."""
        zeros = np.linspace(0, 5, 8)
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 6.0, 15)
        curve = ph.sweep(eps_grid, m=5)
        diffs = np.diff(curve.kernel_dimensions)
        assert np.all(diffs <= 0), f"Betti curve not monotone: {curve.kernel_dimensions}"

    def test_k1_reproduces_scalar_betti(self):
        """At K=1, sheaf Betti curve must match scalar H_0 Betti curve."""
        zeros = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        curve = ph.sweep(eps_grid, m=5)

        # At K=1, sheaf Betti = scalar Betti (number of connected components)
        # Gaps: [1.0, 1.5, 1.5, 2.0]
        # eps=0: 5 components
        # eps=1.0: 4 components (first gap merges)
        # eps=1.5: 2 components (gaps 1.0, 1.5, 1.5 merge)
        # eps=2.0: 1 component
        expected_at_0 = 5
        assert curve.kernel_dimensions[0] == expected_at_0


class TestSheafPHSigmaSweep:
    """Test the sigma sweep (2D heatmap)."""

    def test_returns_2d_array(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 3.0, 5)
        sigma_grid = np.array([0.3, 0.5, 0.7])
        heatmap = ph.sigma_sweep(eps_grid, sigma_grid)
        assert heatmap.shape == (3, 5)

    def test_epsilon_zero_column_constant(self):
        """The ε=0 column should always be N*K^2 regardless of σ."""
        zeros = np.array([0.0, 1.0, 2.0])
        K = 2
        builder = TransportMapBuilder(K=K, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 1.0, 2.0])
        sigma_grid = np.array([0.3, 0.5, 0.7])
        heatmap = ph.sigma_sweep(eps_grid, sigma_grid)
        expected = len(zeros) * K * K
        assert np.all(heatmap[:, 0] == expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sheaf_ph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'atft.topology.sheaf_ph'`

- [ ] **Step 3: Implement sheaf_ph.py**

```python
# atft/topology/sheaf_ph.py
"""Sheaf persistent homology: epsilon and sigma sweep orchestrators.

Produces SheafBettiCurve (single sigma) and 2D heatmaps (sigma sweep)
by wrapping the matrix-free SheafLaplacian kernel computation.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atft.core.types import SheafBettiCurve
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


class SheafPH:
    """Orchestrates epsilon and sigma sweeps for sheaf Betti curves.

    Args:
        transport_builder: TransportMapBuilder for the current (K, sigma).
        unfolded_zeros: 1D array of N sorted unfolded zeros.
    """

    def __init__(
        self,
        transport_builder: TransportMapBuilder,
        unfolded_zeros: NDArray[np.float64],
    ) -> None:
        self._builder = transport_builder
        self._zeros = np.asarray(unfolded_zeros, dtype=np.float64).ravel()
        self._K = transport_builder.K
        self._N = len(self._zeros)

    def sweep(self, epsilon_grid: NDArray[np.float64], m: int = 20) -> SheafBettiCurve:
        """Compute sheaf Betti curve beta_0^F(epsilon) across filtration.

        At epsilon=0, returns N*K^2 without calling the eigensolver.
        """
        lap = SheafLaplacian(self._builder, self._zeros)
        n_steps = len(epsilon_grid)
        kernel_dims = np.zeros(n_steps, dtype=np.int64)
        all_eigs = np.zeros((n_steps, m), dtype=np.float64)

        for idx, eps in enumerate(epsilon_grid):
            if eps <= 0:
                kernel_dims[idx] = self._N * self._K * self._K
                continue

            eigs = lap.smallest_eigenvalues(eps, m=m)
            all_eigs[idx, :len(eigs)] = eigs

            # Derive kernel dim from eigenvalues directly (avoid double eigensolver call)
            tol = lap.frobenius_norm_estimate(eps) * 1e-6
            if tol == 0:
                tol = 1e-12
            kernel_dims[idx] = int(np.sum(np.abs(eigs) < tol))

        return SheafBettiCurve(
            epsilon_grid=np.array(epsilon_grid, dtype=np.float64),
            kernel_dimensions=kernel_dims,
            smallest_eigenvalues=all_eigs,
            sigma=self._builder.sigma,
            K=self._K,
        )

    def sigma_sweep(
        self,
        epsilon_grid: NDArray[np.float64],
        sigma_grid: NDArray[np.float64],
        m: int = 20,
    ) -> NDArray[np.int64]:
        """Compute 2D heatmap beta_0^F(epsilon, sigma).

        Rebuilds TransportMapBuilder at each sigma value.
        Returns array of shape (len(sigma_grid), len(epsilon_grid)).
        """
        heatmap = np.zeros((len(sigma_grid), len(epsilon_grid)), dtype=np.int64)

        for s_idx, sigma in enumerate(sigma_grid):
            builder = TransportMapBuilder(K=self._K, sigma=sigma)
            ph = SheafPH(builder, self._zeros)
            curve = ph.sweep(epsilon_grid, m=m)
            heatmap[s_idx, :] = curve.kernel_dimensions

        return heatmap
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sheaf_ph.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add atft/topology/sheaf_ph.py tests/test_sheaf_ph.py
git commit -m "feat: implement SheafPH epsilon and sigma sweep orchestrators"
```

---

### Task 4: Implement Phase 2a abelian eigenbasis diagnostic

**Files:**
- Create: `atft/experiments/phase2a_abelian.py`
- Create: `tests/test_phase2a_abelian.py`

**Mathematical background (Spec Section 4.1):**
- Eigendecompose A(σ) = VΛV†
- In the eigenbasis, transport is diagonal: V†U_{ij}V = diag(e^{iΔγ·λ_k})
- Each (k,l) entry evolves with frequency ω_{kl} = λ_k - λ_l
- This gives K² independent scalar "twisted" graph Laplacians
- Diagonal entries (k=l): ω=0, reproduces standard graph Laplacian → must match Phase 1's scalar Betti
- Off-diagonal entries: ω≠0, "frequency-twisted" Laplacians
- Output: K×K resonance matrix R where R_{kl} = max_ε dim ker L_{kl}(ε)

**Implementation approach:**
- For each (k,l), build the scalar twisted graph Laplacian L_{kl} as an explicit N×N matrix
  - L_{kl}[i,i] += 1, L_{kl}[j,j] += 1 for each edge (i,j) with gap ≤ ε
  - L_{kl}[i,j] -= e^{iΔγ·ω_{kl}}, L_{kl}[j,i] -= e^{-iΔγ·ω_{kl}}
- Compute eigenvalues and count near-zero ones → kernel dim at each ε
- For diagonal (k=k): ω=0, e^{iΔγ·0} = 1, so L_{kk} is the standard graph Laplacian

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase2a_abelian.py
"""Tests for phase2a_abelian.py — eigenbasis diagnostic.

Key validation: diagonal blocks must reproduce the standard graph Laplacian
(and therefore Phase 1's scalar Betti curve).
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.experiments.phase2a_abelian import Phase2aAbelian
from atft.topology.transport_maps import TransportMapBuilder


class TestTwistedLaplacian:
    """Test the scalar twisted graph Laplacian L_{kl}."""

    def test_zero_frequency_is_graph_laplacian(self):
        """At omega=0, twisted Laplacian must equal standard graph Laplacian."""
        zeros = np.array([0.0, 1.0, 2.5, 4.0])
        epsilon = 2.0
        diag = Phase2aAbelian._build_twisted_laplacian(zeros, 0.0, epsilon)

        # Standard graph Laplacian
        N = len(zeros)
        L_graph = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(zeros[j] - zeros[i]) <= epsilon:
                    L_graph[i, i] += 1
                    L_graph[j, j] += 1
                    L_graph[i, j] -= 1
                    L_graph[j, i] -= 1

        assert_allclose(diag, L_graph, atol=1e-14)

    def test_twisted_laplacian_is_hermitian(self):
        """Twisted Laplacian must be Hermitian for any omega."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega=1.23, epsilon=1.0)
        assert_allclose(L, L.conj().T, atol=1e-14)

    def test_twisted_laplacian_psd(self):
        """Twisted Laplacian must be positive semidefinite."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        L = Phase2aAbelian._build_twisted_laplacian(zeros, omega=0.7, epsilon=1.5)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-12)


class TestResonanceMatrix:
    """Test the full resonance matrix computation."""

    def test_resonance_matrix_shape(self):
        """Resonance matrix must be K×K."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 4.0, 10)
        R = diag.compute_resonance_matrix(eps_grid)
        assert R.shape == (3, 3)

    def test_diagonal_entries_match_scalar_betti(self):
        """Diagonal R_{kk} must reproduce scalar Betti (connected components).

        At omega=0, the twisted Laplacian IS the standard graph Laplacian.
        The max kernel dim over epsilon is N (at eps=0 where there are no edges).
        All K diagonal entries must agree (they all use the same omega=0).
        """
        zeros = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 7.0, 20)
        R = diag.compute_resonance_matrix(eps_grid)

        # All K diagonal entries must be identical (all use omega=0)
        for k in range(3):
            assert R[k, k] == R[0, 0]

        # Additionally verify: at eps=0, kernel = N (trivial full kernel)
        assert R[0, 0] == len(zeros)

        # Verify the zero-frequency Laplacian at a non-trivial epsilon
        # gives the correct connected component count
        # zeros=[0,1,2.5,4,6], eps=1.5: edges (0,1), (2,3) → 3 components
        L = Phase2aAbelian._build_twisted_laplacian(zeros, 0.0, epsilon=1.5)
        eigs = np.linalg.eigvalsh(L)
        n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
        assert n_kernel == 3  # three connected components

    def test_resonance_matrix_symmetric(self):
        """R_{kl} = R_{lk} because omega_{kl} = -omega_{lk} and kernel dim is the same."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=4, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 2.0, 10)
        R = diag.compute_resonance_matrix(eps_grid)
        assert_allclose(R, R.T)


class TestPhase2aRun:
    """Test the full experiment runner."""

    def test_run_returns_results(self):
        """Full run should return resonance matrix and distinct frequency count."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        diag = Phase2aAbelian(builder, zeros)
        eps_grid = np.linspace(0, 4.0, 10)
        results = diag.run(eps_grid)
        assert "resonance_matrix" in results
        assert "n_distinct_frequencies" in results
        assert "eigenvalues_A" in results
        assert results["resonance_matrix"].shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase2a_abelian.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement phase2a_abelian.py**

```python
# atft/experiments/phase2a_abelian.py
"""Phase 2a: Abelian eigenbasis diagnostic.

Decomposes the sheaf Laplacian into K^2 independent scalar twisted
graph Laplacians at eigenfrequency differences of A(sigma). Validates
that the diagonal (zero-frequency) blocks reproduce the standard graph
Laplacian, and produces a resonance matrix showing which prime harmonics
the zero spacings respond to.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atft.topology.transport_maps import TransportMapBuilder


class Phase2aAbelian:
    """Abelian eigenbasis diagnostic for Phase 2a.

    Args:
        transport_builder: TransportMapBuilder with cached eigendecomp.
        unfolded_zeros: 1D array of N sorted unfolded zeros.
    """

    def __init__(
        self,
        transport_builder: TransportMapBuilder,
        unfolded_zeros: NDArray[np.float64],
    ) -> None:
        self._builder = transport_builder
        self._zeros = np.asarray(unfolded_zeros, dtype=np.float64).ravel()
        self._K = transport_builder.K
        self._N = len(self._zeros)

    @staticmethod
    def _build_twisted_laplacian(
        zeros: NDArray[np.float64],
        omega: float,
        epsilon: float,
    ) -> NDArray[np.complex128]:
        """Build the scalar twisted graph Laplacian L_omega(epsilon).

        L_omega is N×N Hermitian:
          L[i,i] += 1, L[j,j] += 1 for each edge
          L[i,j] -= e^{i*delta_gamma*omega}
          L[j,i] -= e^{-i*delta_gamma*omega}
        """
        N = len(zeros)
        L = np.zeros((N, N), dtype=np.complex128)

        for i in range(N):
            for j in range(i + 1, N):
                gap = zeros[j] - zeros[i]
                if gap > epsilon:
                    break  # zeros are sorted
                phase = np.exp(1j * gap * omega)
                L[i, i] += 1.0
                L[j, j] += 1.0
                L[i, j] -= phase
                L[j, i] -= np.conj(phase)

        return L

    def compute_resonance_matrix(
        self, epsilon_grid: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Compute K×K resonance matrix R where R_{kl} = max_eps kernel_dim(L_{omega_kl}).

        Returns integer matrix of shape (K, K).
        """
        eigenvalues = self._builder.eigenvalues()
        K = self._K
        R = np.zeros((K, K), dtype=np.int64)

        for k in range(K):
            for l in range(k, K):
                omega = eigenvalues[k] - eigenvalues[l]
                max_kernel = 0

                for eps in epsilon_grid:
                    if eps <= 0:
                        # At eps=0, no edges, kernel = N
                        max_kernel = max(max_kernel, self._N)
                        continue

                    L = self._build_twisted_laplacian(self._zeros, omega, eps)
                    eigs = np.linalg.eigvalsh(L)
                    n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
                    max_kernel = max(max_kernel, n_kernel)

                R[k, l] = max_kernel
                R[l, k] = max_kernel  # symmetric: omega_{lk} = -omega_{kl}

        return R

    def run(self, epsilon_grid: NDArray[np.float64]) -> dict:
        """Run the full Phase 2a diagnostic.

        Returns dict with:
          - resonance_matrix: K×K int array
          - eigenvalues_A: K eigenvalues of A(sigma)
          - n_distinct_frequencies: number of distinct omega_{kl} values
        """
        eigenvalues = self._builder.eigenvalues()
        R = self.compute_resonance_matrix(epsilon_grid)

        # Count distinct frequencies
        K = self._K
        freqs = set()
        for k in range(K):
            for l in range(K):
                freq = round(eigenvalues[k] - eigenvalues[l], 10)
                freqs.add(abs(freq))
        n_distinct = len(freqs)

        return {
            "resonance_matrix": R,
            "eigenvalues_A": eigenvalues,
            "n_distinct_frequencies": n_distinct,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase2a_abelian.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add atft/experiments/phase2a_abelian.py tests/test_phase2a_abelian.py
git commit -m "feat: implement Phase 2a abelian eigenbasis diagnostic with resonance matrix"
```

---

## Chunk 3: Phase 2b Experiment + Visualization + Runners

### Task 5: Implement Phase 2b sigma-sweep experiment

**Files:**
- Create: `atft/experiments/phase2b_sheaf.py`
- Create: `tests/test_phase2b_sheaf.py`

**Mathematical background (Spec Section 4.2):**
- Sweep σ = [0.30, 0.35, ..., 0.70] × ε = [0, 3.0] (200 steps)
- For each (σ, ε): compute β₀^F via SheafPH
- Produce SheafValidationResult with peak_sigma, is_unique_peak
- Success criterion: peak at σ=0.5, non-trivial kernel only there

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase2b_sheaf.py
"""Tests for phase2b_sheaf.py — full sigma-sweep experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atft.core.types import SheafValidationResult
from atft.experiments.phase2b_sheaf import Phase2bConfig, Phase2bExperiment


class TestPhase2bExperiment:
    """Integration tests at tiny scale."""

    def test_run_returns_validation_result(self):
        """Full run at tiny scale should complete and return SheafValidationResult."""
        config = Phase2bConfig(
            n_points=10,
            K=2,
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            n_epsilon_steps=5,
            epsilon_max=3.0,
            m=3,
            zeta_data_path=Path("data/odlyzko_zeros.txt"),
        )
        experiment = Phase2bExperiment(config)
        result = experiment.run()
        assert isinstance(result, SheafValidationResult)
        assert result.betti_heatmap.shape == (3, 5)
        assert result.peak_sigma in [0.3, 0.5, 0.7]

    def test_epsilon_zero_column(self):
        """First column (ε=0) should always be N*K^2."""
        config = Phase2bConfig(
            n_points=8,
            K=2,
            sigma_grid=np.array([0.4, 0.5, 0.6]),
            n_epsilon_steps=5,
            epsilon_max=3.0,
            m=3,
            zeta_data_path=Path("data/odlyzko_zeros.txt"),
        )
        experiment = Phase2bExperiment(config)
        result = experiment.run()
        expected_full = 8 * 2 * 2  # N * K^2
        assert np.all(result.betti_heatmap[:, 0] == expected_full)

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = Phase2bConfig()
        assert config.K == 20
        assert config.n_points == 1000
        assert len(config.sigma_grid) == 9
        assert config.sigma_grid[4] == 0.5  # middle value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase2b_sheaf.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement phase2b_sheaf.py**

```python
# atft/experiments/phase2b_sheaf.py
"""Phase 2b: Full non-abelian sigma-sweep experiment.

The core falsifiable experiment: sweep sigma across the critical strip
and test whether the sheaf Betti number peaks uniquely at sigma=1/2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atft.core.types import SheafValidationResult
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_ph import SheafPH
from atft.topology.transport_maps import TransportMapBuilder


@dataclass
class Phase2bConfig:
    """Configuration for the Phase 2b sigma-sweep experiment."""

    n_points: int = 1000
    K: int = 20
    sigma_grid: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    )
    n_epsilon_steps: int = 200
    epsilon_max: float = 3.0
    m: int = 20
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    seed: int = 42


class Phase2bExperiment:
    """Orchestrates the full sigma-sweep experiment."""

    def __init__(self, config: Phase2bConfig) -> None:
        self.config = config

    def run(self) -> SheafValidationResult:
        """Execute the sigma-sweep and return validation result."""
        cfg = self.config

        # Load and unfold zeta zeros
        print(f"Loading {cfg.n_points} zeta zeros...")
        source = ZetaZerosSource(cfg.zeta_data_path)
        cloud = source.generate(cfg.n_points)
        unfolded = SpectralUnfolding(method="zeta").transform(cloud)
        zeros = unfolded.points[:, 0]

        eps_grid = np.linspace(0, cfg.epsilon_max, cfg.n_epsilon_steps)

        # Build initial builder just to pass K to SheafPH
        builder = TransportMapBuilder(K=cfg.K, sigma=0.5)
        ph = SheafPH(builder, zeros)

        print(f"Running sigma-sweep: {len(cfg.sigma_grid)} sigma values x {cfg.n_epsilon_steps} epsilon steps")
        heatmap = ph.sigma_sweep(eps_grid, cfg.sigma_grid, m=cfg.m)

        # Find peak
        max_per_sigma = np.max(heatmap[:, 1:], axis=1)  # skip eps=0 (trivial)
        peak_idx = int(np.argmax(max_per_sigma))
        peak_sigma = float(cfg.sigma_grid[peak_idx])
        peak_kernel_dim = int(max_per_sigma[peak_idx])

        # Check uniqueness: is peak_sigma the only maximum?
        is_unique = int(np.sum(max_per_sigma == peak_kernel_dim)) == 1

        return SheafValidationResult(
            sigma_grid=cfg.sigma_grid,
            epsilon_grid=eps_grid,
            betti_heatmap=heatmap,
            peak_sigma=peak_sigma,
            peak_kernel_dim=peak_kernel_dim,
            is_unique_peak=is_unique,
            metadata={
                "n_points": cfg.n_points,
                "K": cfg.K,
                "m": cfg.m,
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase2b_sheaf.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add atft/experiments/phase2b_sheaf.py tests/test_phase2b_sheaf.py
git commit -m "feat: implement Phase 2b sigma-sweep experiment"
```

---

### Task 6: Add Phase 2 visualization functions

**Files:**
- Modify: `atft/visualization/plots.py`
- Create: `tests/test_phase2_plots.py`

**Spec Section 5:**
- Figure 1: Three-panel sheaf Betti curves (single σ, overlay, heatmap)
- Figure 2: σ peak plot (max_ε β₀^F vs σ)
- Figure 3: Phase 2a resonance matrix heatmap

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase2_plots.py
"""Smoke tests for Phase 2 visualization functions."""
from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atft.core.types import SheafBettiCurve, SheafValidationResult
from atft.visualization.plots import (
    plot_sheaf_betti_curves,
    plot_sigma_peak,
    plot_resonance_matrix,
)


def _make_dummy_curve(sigma: float = 0.5) -> SheafBettiCurve:
    n = 20
    return SheafBettiCurve(
        epsilon_grid=np.linspace(0, 3.0, n),
        kernel_dimensions=np.maximum(0, np.arange(n, 0, -1)),
        smallest_eigenvalues=np.random.rand(n, 5),
        sigma=sigma,
        K=5,
    )


def _make_dummy_result() -> SheafValidationResult:
    sigma_grid = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    eps_grid = np.linspace(0, 3.0, 10)
    return SheafValidationResult(
        sigma_grid=sigma_grid,
        epsilon_grid=eps_grid,
        betti_heatmap=np.random.randint(0, 50, size=(5, 10)),
        peak_sigma=0.5,
        peak_kernel_dim=42,
        is_unique_peak=True,
    )


class TestPlotSheafBettiCurves:
    def test_returns_figure(self):
        curves = [_make_dummy_curve(s) for s in [0.3, 0.5, 0.7]]
        fig = plot_sheaf_betti_curves(curves, highlight_sigma=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        curves = [_make_dummy_curve(0.5)]
        path = tmp_path / "test_sheaf.png"
        fig = plot_sheaf_betti_curves(curves, save_path=path)
        assert path.exists()
        plt.close(fig)


class TestPlotSigmaPeak:
    def test_returns_figure(self):
        result = _make_dummy_result()
        fig = plot_sigma_peak(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResonanceMatrix:
    def test_returns_figure(self):
        R = np.random.randint(0, 10, size=(5, 5))
        eigenvalues = np.linspace(-1, 1, 5)
        fig = plot_resonance_matrix(R, eigenvalues)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase2_plots.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_sheaf_betti_curves'`

- [ ] **Step 3: Add Phase 2 plot functions to plots.py**

Append the following functions to `atft/visualization/plots.py`:

```python
def plot_sheaf_betti_curves(
    curves: list,
    highlight_sigma: float | None = 0.5,
    save_path: Path | None = None,
) -> plt.Figure:
    """Three-panel sheaf Betti curve figure.

    Panel A: Highlighted sigma curve
    Panel B: Overlay of all sigma values
    Panel C: Heatmap if multiple curves provided
    """
    n_panels = 3 if len(curves) > 1 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel A: highlighted curve (or single curve)
    ax = axes[0]
    for c in curves:
        if highlight_sigma is not None and abs(c.sigma - highlight_sigma) < 1e-6:
            ax.plot(c.epsilon_grid, c.kernel_dimensions, "b-", linewidth=2,
                    label=f"σ = {c.sigma}")
        else:
            ax.plot(c.epsilon_grid, c.kernel_dimensions, color="grey", alpha=0.4,
                    linewidth=1)
    ax.set_xlabel("ε")
    ax.set_ylabel("β₀ᶠ(ε)")
    ax.set_title("Sheaf Betti Curve")
    ax.legend()

    if n_panels > 1:
        # Panel B: all curves overlaid
        ax = axes[1]
        for c in curves:
            color = "blue" if highlight_sigma and abs(c.sigma - highlight_sigma) < 1e-6 else "grey"
            alpha = 1.0 if color == "blue" else 0.4
            ax.plot(c.epsilon_grid, c.kernel_dimensions, color=color, alpha=alpha,
                    linewidth=1.5, label=f"σ={c.sigma:.2f}" if color == "blue" else None)
        ax.set_xlabel("ε")
        ax.set_ylabel("β₀ᶠ(ε)")
        ax.set_title("σ Overlay")
        ax.legend()

        # Panel C: heatmap
        ax = axes[2]
        sigmas = np.array([c.sigma for c in curves])
        eps = curves[0].epsilon_grid
        heatmap = np.array([c.kernel_dimensions for c in curves])
        im = ax.pcolormesh(eps, sigmas, heatmap, shading="auto", cmap="viridis")
        ax.set_xlabel("ε")
        ax.set_ylabel("σ")
        ax.set_title("Phase Diagram β₀ᶠ(ε, σ)")
        fig.colorbar(im, ax=ax, label="β₀ᶠ")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sigma_peak(
    result,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot max_ε β₀^F(ε, σ) vs σ — the phase transition signature."""
    max_betti = np.max(result.betti_heatmap[:, 1:], axis=1)  # skip eps=0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.sigma_grid, max_betti, "ko-", linewidth=2, markersize=8)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="σ = 1/2")
    ax.set_xlabel("σ")
    ax.set_ylabel("max_ε β₀ᶠ(ε, σ)")
    ax.set_title("σ-Criticality: Topological Phase Transition")
    ax.legend()

    if result.peak_sigma is not None:
        ax.annotate(
            f"Peak: σ = {result.peak_sigma:.2f}\nβ₀ᶠ = {result.peak_kernel_dim}",
            xy=(result.peak_sigma, result.peak_kernel_dim),
            xytext=(result.peak_sigma + 0.05, result.peak_kernel_dim * 0.8),
            arrowprops=dict(arrowstyle="->"),
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_resonance_matrix(
    R: np.ndarray,
    eigenvalues: np.ndarray,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the K×K abelian resonance matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(R, cmap="hot", interpolation="nearest", origin="lower")
    ax.set_xlabel("Eigenbasis index l")
    ax.set_ylabel("Eigenbasis index k")
    ax.set_title("Phase 2a Resonance Matrix R_{kl}")
    fig.colorbar(im, ax=ax, label="max_ε dim ker L_{ω_{kl}}")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase2_plots.py -v --tb=short`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `python -m pytest tests/ -q --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add atft/visualization/plots.py tests/test_phase2_plots.py
git commit -m "feat: add Phase 2 visualization — sheaf Betti, sigma peak, resonance matrix"
```

---

### Task 7: Create CLI runners for Phase 2a and 2b

**Files:**
- Create: `run_phase2a.py`
- Create: `run_phase2b.py`

- [ ] **Step 1: Create run_phase2a.py**

```python
#!/usr/bin/env python
"""Run Phase 2a: Abelian eigenbasis diagnostic.

Usage:
    python run_phase2a.py                # Quick validation (N=100, K=5)
    python run_phase2a.py --production   # Production (N=1000, K=20)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from atft.experiments.phase2a_abelian import Phase2aAbelian
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.visualization.plots import plot_resonance_matrix


def main():
    parser = argparse.ArgumentParser(description="Phase 2a: Abelian Eigenbasis Diagnostic")
    parser.add_argument("--production", action="store_true", help="N=1000, K=20")
    args = parser.parse_args()

    if args.production:
        N, K, n_eps = 1000, 20, 200
        label = "PRODUCTION"
    else:
        N, K, n_eps = 100, 5, 50
        label = "VALIDATION"

    print(f"\n{'='*60}")
    print(f"  ATFT Phase 2a — {label} RUN")
    print(f"  N={N}, K={K}, {n_eps} epsilon steps")
    print(f"{'='*60}\n")

    t0 = time.time()

    source = ZetaZerosSource(Path("data/odlyzko_zeros.txt"))
    cloud = source.generate(N)
    zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]

    builder = TransportMapBuilder(K=K, sigma=0.5)
    diag = Phase2aAbelian(builder, zeros)
    eps_grid = np.linspace(0, 3.0, n_eps)
    results = diag.run(eps_grid)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Distinct frequencies: {results['n_distinct_frequencies']}")
    print(f"Eigenvalues of A(0.5): {results['eigenvalues_A']}")
    print(f"\nResonance matrix:\n{results['resonance_matrix']}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"phase2a_{label.lower()}.png"
    plot_resonance_matrix(results["resonance_matrix"], results["eigenvalues_A"], save_path=fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create run_phase2b.py**

```python
#!/usr/bin/env python
"""Run Phase 2b: Full sigma-sweep experiment.

Usage:
    python run_phase2b.py               # Quick validation (N=20, K=2, 3 sigmas)
    python run_phase2b.py --production  # Production (N=1000, K=20, 9 sigmas)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from atft.experiments.phase2b_sheaf import Phase2bConfig, Phase2bExperiment
from atft.visualization.plots import plot_sigma_peak


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Sigma-Sweep Experiment")
    parser.add_argument("--production", action="store_true", help="Full production run")
    args = parser.parse_args()

    if args.production:
        config = Phase2bConfig()
        label = "PRODUCTION"
    else:
        config = Phase2bConfig(
            n_points=20,
            K=2,
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            n_epsilon_steps=10,
            epsilon_max=3.0,
            m=5,
        )
        label = "VALIDATION"

    print(f"\n{'='*60}")
    print(f"  ATFT Phase 2b — {label} RUN")
    print(f"  N={config.n_points}, K={config.K}")
    print(f"  {len(config.sigma_grid)} sigma values x {config.n_epsilon_steps} epsilon steps")
    print(f"{'='*60}\n")

    t0 = time.time()
    experiment = Phase2bExperiment(config)
    result = experiment.run()
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Peak sigma: {result.peak_sigma}")
    print(f"Peak kernel dim: {result.peak_kernel_dim}")
    print(f"Unique peak: {result.is_unique_peak}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"phase2b_{label.lower()}.png"
    plot_sigma_peak(result, save_path=fig_path)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify runners import correctly**

Run: `python -c "from atft.experiments.phase2a_abelian import Phase2aAbelian; from atft.experiments.phase2b_sheaf import Phase2bConfig; print('OK')"`
Expected: `OK`

Run: `python run_phase2a.py --help && python run_phase2b.py --help`
Expected: Both print usage/help text without errors

- [ ] **Step 4: Run full test suite — final regression check**

Run: `python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass (119 Phase 1 + 27 transport_maps + all new Phase 2 tests)

- [ ] **Step 5: Commit**

```bash
git add run_phase2a.py run_phase2b.py
git commit -m "feat: add CLI runners for Phase 2a and 2b experiments"
```

---

## Dependency Order

```
Task 1 (types) ─────────────────────┐
                                     ▼
Task 2 (sheaf_laplacian) ──────── Task 3 (sheaf_ph)
                                     │
                          ┌──────────┤──────────┐
                          ▼          ▼          ▼
               Task 4 (phase2a) Task 5 (phase2b) Task 6 (plots)
                          │          │          │
                          └──────────┴──────────┘
                                     ▼
                              Task 7 (runners)
```

Task 1 must complete first. Tasks 2 and 1 are sequential prerequisites for everything else. Tasks 4, 5, 6 can be partially parallelized after Task 3. Task 7 is last.
