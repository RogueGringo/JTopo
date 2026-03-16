# ATFT Phase 2: Sheaf-Valued Persistent Homology Design Specification

**Date:** 2026-03-15
**Status:** Approved (Rev 2 — post spec review)
**Authors:** Blake Jones, Claude (Opus 4.6)
**Depends on:** Phase 1 Design Spec (2026-03-15-atft-riemann-hypothesis-design.md)

---

## 1. Overview

### 1.1 Goal

Extend the ATFT framework from scalar persistent homology (Phase 1) to sheaf-valued persistent homology with u(K) Lie algebra fibers, using a canonical prime-weighted gauge connection to test whether the Riemann Hypothesis can be reformulated as a topological phase transition at σ = 1/2.

### 1.2 Phase 1 Foundation

Phase 1 established that zeta zeros are topologically convergent to GUE under scalar H_0 persistence:

| N | D_M(zeta)/D_M(Poisson) | L2 Gini |
|---|------------------------|---------|
| 500 | 0.144 | 0.030 |
| 1,000 | 0.097 | 0.030 |
| 2,000 | 0.088 | 0.025 |
| 10,000 | 0.052 | 0.026 |

The flat Gini L2 (~0.025) proves that the merging rules of zeta zeros are shape-identical to GUE. The residual Mahalanobis distance comes from scale-dependent waypoint features carrying arithmetic noise from small primes — exactly what Phase 2 is designed to resolve.

### 1.3 What Phase 2 Adds

Phase 1 measured the **gaps** between zeros. Phase 2 measures whether the **prime structure** is compatible with the spectral structure. It does this by attaching Hermitian matrix-valued fields to each zero and testing whether these fields can maintain global consistency under prime-weighted parallel transport.

### 1.4 Project Trajectory

- **Phase 1 (complete):** Scalar H_0 benchmark. Zeta vs GUE vs Poisson topological comparison.
- **Phase 2a (this spec):** Abelian diagnostic. Eigenbasis decomposition of A(σ) into K² scalar Laplacians at eigenfrequency differences, validating the matrix-free Laplacian machinery.
- **Phase 2b (this spec):** Non-abelian sheaf PH. Full multiplicative monoid u(K) transport. The σ-sweep experiment.
- **Phase 3 (future):** Rigorous reformulation. Prove that RH is equivalent to a topological waypoint constraint.

---

## 2. Mathematical Foundation

### 2.1 Cellular Sheaf on the Rips Complex

For a 1D point cloud of N unfolded zeta zeros γ̃_1 < γ̃_2 < ... < γ̃_N, the Vietoris-Rips complex at scale ε has:

- **Vertices:** V = {v_1, ..., v_N}, one per zero
- **Edges:** E(ε) = {e_{ij} : |γ̃_i - γ̃_j| ≤ ε}

**Edge orientation convention:** Edges are oriented i → j with i < j (equivalently, γ̃_i < γ̃_j since zeros are sorted). The sheaf Laplacian L_F = δ₀†δ₀ is independent of orientation, but the coboundary requires a consistent convention.

A cellular sheaf F assigns:
- To each vertex v_i: a stalk F(v_i) = Herm(K), the vector space of K×K Hermitian matrices
- To each edge e_{ij}: a stalk F(e_{ij}) = Herm(K)
- Restriction maps: F_{v_i → e_{ij}}: conjugation by U_{ij}; F_{v_j → e_{ij}}: identity. All transport is placed on the tail vertex (see Section 2.6).

### 2.2 The Fiber Space

The fiber dimension K corresponds to the first K positive integers, forming the basis |1⟩, |2⟩, ..., |K⟩ of a K-dimensional Hilbert space. Each stalk carries the space Herm(K) of K×K Hermitian matrices over this basis.

**Phase 2 default:** K = 20 (integers 1–20, containing primes {2, 3, 5, 7, 11, 13, 17, 19}).

### 2.3 The Multiplicative Monoid Representation

For each prime p ≤ K, define the partial permutation matrix:

```
ρ(p)|n⟩ = |pn⟩   if pn ≤ K
ρ(p)|n⟩ = 0       if pn > K
```

This is the truncated left-regular representation of (Z_{>0}, ×). It encodes Dirichlet convolution as matrix multiplication:

```
ρ(2)ρ(3)|1⟩ = ρ(2)|3⟩ = |6⟩
ρ(3)ρ(5)|1⟩ = ρ(3)|5⟩ = |15⟩
ρ(2)ρ(2)|1⟩ = ρ(2)|2⟩ = |4⟩     (prime powers)
ρ(2)|11⟩ = 0                      (22 > K=20, truncated)
```

The sieve-theoretic structure of the primes is baked into the matrix entries. No arbitrary choices are made — the representation is canonical.

### 2.4 The Hermitian Generator

For prime p, define the Hermitian generator parameterized by σ ∈ R:

```
G_p(σ) = (log p / p^σ) · (ρ(p) + ρ(p)†)
```

**Components:**
- `log p` — the von Mangoldt weight Λ(p), encoding the density of the prime's contribution to the explicit formula
- `p^{-σ}` — the critical line weighting from the Riemann explicit formula, where p^{-ρ} = p^{-σ-iγ} decomposes into amplitude p^{-σ} and phase p^{-iγ}
- `ρ(p) + ρ(p)†` — Hermitian symmetrization, ensuring G_p generates unitary transport

**The σ parameter:** At σ = 1/2, this is the unique weighting where the explicit formula's oscillatory terms have the correct amplitude to reconstruct the zero-counting function. If RH is true, σ = 1/2 is the only value producing a non-trivial sheaf Laplacian kernel.

### 2.5 The Transport Matrix

The sum of all generators defines a fixed Hermitian matrix:

```
A(σ) = Σ_{p prime, p ≤ P} G_p(σ)
```

where P is the largest prime ≤ K. For K = 20: P = 19, and the sum runs over 8 primes.

The transport matrix between zeros γ̃_i and γ̃_j is:

```
U_{ij}(σ) = exp(i · (γ̃_j - γ̃_i) · A(σ))
```

**The eigendecomposition shortcut:** Since A(σ) is fixed for a given (K, σ), eigendecompose once:

```
A(σ) = V Λ V†,   Λ = diag(λ_1, ..., λ_K)
```

Then for any edge gap Δγ = γ̃_j - γ̃_i:

```
U_{ij} = V · diag(e^{i·Δγ·λ_k}) · V†
```

**Cost:** One O(K³) eigendecomposition per σ value (negligible at K=20), then O(K²) per edge.

### 2.6 The Restriction Maps

The restriction maps are asymmetric by convention:

```
F_{v_i → e_{ij}}(H) = U_{ij} · H · U_{ij}†    (transport from tail vertex)
F_{v_j → e_{ij}}(H) = H                         (identity at head vertex)
```

All prime-weighted parallel transport is placed on the tail vertex v_i. This is a standard gauge-theoretic convention — the transport matrix U_{ij} carries the Hermitian matrix H from zero γ̃_i to zero γ̃_j via conjugation. The sheaf Laplacian L_F = δ₀†δ₀ is invariant under the choice of which vertex carries the transport.

### 2.7 The Coboundary Operator

The 0-th coboundary map δ₀ takes a 0-cochain (assignment of Herm(K) matrices to vertices) to a 1-cochain (assignment to edges):

```
(δ₀ f)(e_{ij}) = f(v_j) - U_{ij} · f(v_i) · U_{ij}†
```

This measures the "mismatch" between the matrix at v_j and the matrix transported from v_i. A global section satisfies δ₀ f = 0: the field is covariantly constant everywhere.

**Representation note:** The conjugation H → U H U† acts linearly on K×K matrices. In principle, this can be vectorized as vec(UHU†) = (U ⊗ U*) · vec(H), but we do NOT use this Kronecker representation in the implementation. Instead, we use a **matrix-free** approach (Section 3.4) that applies conjugation directly in K×K matrix form, avoiding the O(K⁴) memory cost of explicit Kronecker blocks.

**Hermiticity:** The conjugation action preserves Hermiticity (if H is Hermitian, so is UHU†). The implementation works with general K×K complex matrices for simplicity; the kernel of L_F, when initialized with Hermitian inputs, produces Hermitian outputs.

### 2.8 The Sheaf Laplacian

```
L_F(ε, σ) = δ₀(ε, σ)† · δ₀(ε, σ)
```

This is a (N·K²) × (N·K²) Hermitian positive-semidefinite operator. Its kernel is the space of global sections.

**Dimension at K=20, N=1000:** L_F acts on a 400,000-dimensional complex vector space. It is NEVER assembled as an explicit matrix — instead, the matrix-vector product y = L_F · x is computed via a matrix-free approach (Section 3.4) that iterates over edges and applies K×K conjugation directly. This reduces memory from O(|E|·K⁴) to O(|E|·K²), making the computation feasible on any hardware.

**Degenerate case:** At ε = 0, there are no edges, L_F = 0, and the kernel is the full N·K²-dimensional space. This case is excluded from the sweep grid (no LOBPCG call is made; β₀^F(0) = N·K² is recorded directly).

### 2.9 The Sheaf Betti Curve

As ε increases from 0 to ε_max:
- At ε = 0: no edges. L_F = 0. Kernel is the full N·K²-dimensional space (trivial).
- As ε grows: edges impose transport constraints. The kernel dimension drops as more constraints are added.
- **The critical question:** Does the kernel stabilize at a non-zero dimension for some ε range, or does it collapse to zero?

The sheaf Betti curve β₀^F(ε) = dim ker L_F(ε) tracks this evolution.

### 2.10 The σ-Criticality Hypothesis

**Conjecture:** For the zeta zeros with multiplicative monoid transport:

1. β₀^F(ε, σ = 0.5) plateaus at a non-trivial (non-zero) value for some range of ε
2. β₀^F(ε, σ ≠ 0.5) collapses to zero for all ε beyond the trivial onset
3. The function σ → max_ε β₀^F(ε, σ) has a unique global maximum at σ = 1/2

This reformulates the Riemann Hypothesis as a topological phase transition: the sheaf cohomology is non-trivial if and only if the gauge connection is tuned to the critical line.

---

## 3. Software Architecture

### 3.1 New Modules

Phase 2 adds four new files and extends existing types. The Phase 1 pipeline is untouched — Phase 2 is a parallel pathway sharing the same sources and feature maps but diverging at the topology layer.

```
atft/
├── core/
│   └── types.py                # ADD: SheafBettiCurve, SheafValidationResult
├── topology/
│   ├── transport_maps.py       # NEW: ρ(p), G_p(σ), U_{ij}(σ) construction
│   ├── sheaf_laplacian.py      # NEW: Matrix-free L_F operator + kernel computation
│   └── sheaf_ph.py             # EXTEND: Replace Phase 1 stub with full implementation
├── experiments/
│   ├── phase2a_abelian.py      # NEW: Diagonal U(1)^K diagnostic
│   └── phase2b_sheaf.py        # NEW: Full non-abelian σ-sweep experiment
└── visualization/
    └── plots.py                # EXTEND: Phase 2 figures
```

### 3.2 Data Flow

```
ZetaZerosSource → SpectralUnfolding("zeta") → unfolded zeros γ̃_i
                                                    │
                                                    ▼
                                      TransportMapBuilder(K, σ, primes)
                                                    │
                                        eigendecompose A(σ) = V Λ V†
                                                    │
                                                    ▼
                                      SheafLaplacian.as_linear_operator(ε)
                                                    │
                                        matrix-free L_F via matvec
                                                    │
                                                    ▼
                                      SheafLaplacian.smallest_eigenvalues(ε, m=20)
                                                    │
                                          LOBPCG on LinearOperator
                                                    │
                                                    ▼
                                      SheafPH.sweep(ε_grid, σ_grid)
                                                    │
                                          β₀^F(ε, σ) at each point
                                                    │
                                                    ▼
                                      SheafBettiCurve / σ-heatmap
```

### 3.3 Module Responsibilities

**`transport_maps.py`** — Pure math, no I/O.

```python
class TransportMapBuilder:
    def __init__(self, K: int, sigma: float, max_prime: int | None = None):
        """
        Args:
            K: Fiber dimension (integers 1..K).
            sigma: Critical line parameter for the generator weighting.
            max_prime: Largest prime to include. If None, defaults to the
                largest prime ≤ K. If max_prime > K, primes beyond K
                contribute zero generators (no effect). If max_prime < K,
                only primes up to max_prime are included (useful for
                ablation studies).
        """

    def build_prime_rep(self, p: int) -> np.ndarray:
        """Sparse K×K partial permutation matrix ρ(p)."""

    def build_generator(self, p: int) -> np.ndarray:
        """Hermitian K×K matrix G_p(σ) = (log p / p^σ)(ρ(p) + ρ(p)†)."""

    def build_generator_sum(self) -> np.ndarray:
        """A(σ) = Σ_p G_p(σ). Eigendecomposed internally: A = VΛV†."""

    def transport(self, delta_gamma: float) -> np.ndarray:
        """U = V diag(e^{iΔγ·λ_k}) V†. Returns K×K complex unitary matrix."""

    def eigenvalues(self) -> np.ndarray:
        """Return the K eigenvalues of A(σ), for Phase 2a diagnostic."""
```

**`sheaf_laplacian.py`** — Matrix-free Laplacian operator and kernel computation.

```python
class SheafLaplacian:
    def __init__(self, transport_builder: TransportMapBuilder,
                 unfolded_zeros: np.ndarray): ...

    def matvec(self, x: np.ndarray, epsilon: float) -> np.ndarray:
        """Compute y = L_F(ε) · x without assembling L_F.

        x is reshaped to (N, K, K) complex. For each edge (i,j) with
        |γ̃_i - γ̃_j| ≤ ε:
            diff = x[j] - U_{ij} @ x[i] @ U_{ij}†
            y[j] += diff
            y[i] -= U_{ij}† @ diff @ U_{ij}
        Returns y flattened to (N·K²,).
        """

    def as_linear_operator(self, epsilon: float) -> scipy.sparse.linalg.LinearOperator:
        """Wrap matvec as a LinearOperator for eigsh/LOBPCG."""

    def kernel_dimension(self, epsilon: float, tol: float | None = None,
                         m: int = 20) -> int:
        """Count eigenvalues below tolerance τ.

        If tol is None, uses τ = 1e-6 * frobenius_norm_estimate(L_F).
        The Frobenius norm is estimated from the edge contributions
        in O(|E|·K²) without assembling L_F.
        """

    def smallest_eigenvalues(self, epsilon: float, m: int = 20,
                             solver: str = "auto") -> np.ndarray:
        """Return the m smallest eigenvalues.

        solver: "lobpcg", "eigsh", or "auto". "auto" tries LOBPCG first;
        if it fails to converge within 1000 iterations, falls back to
        eigsh with shift-invert mode (shift=0.0, which='LM').
        """

    def extract_global_sections(self, epsilon: float, tol: float | None = None
                                ) -> list[np.ndarray]:
        """Return eigenvectors in ker(L_F), reshaped to (N, K, K)."""
```

**`sheaf_ph.py`** — The ε-sweep orchestrator.

```python
class SheafPH:
    def __init__(self, transport_builder: TransportMapBuilder,
                 unfolded_zeros: np.ndarray): ...

    def sweep(self, epsilon_grid: np.ndarray, m: int = 20
              ) -> SheafBettiCurve:
        """Compute β₀^F(ε) across the filtration.

        At ε=0, returns N·K² without calling the eigensolver
        (degenerate case: L_F = 0, full kernel).
        """

    def sigma_sweep(self, epsilon_grid: np.ndarray,
                    sigma_grid: np.ndarray
                    ) -> np.ndarray:
        """Compute β₀^F(ε, σ) heatmap. Returns 2D array.

        Rebuilds TransportMapBuilder at each σ value using the same K.
        """
```

**Sanity check (K=1):** At K=1, all prime generators vanish (p·1 > 1 for all primes p ≥ 2), so A(σ) = 0 and U_{ij} = 1. The sheaf Laplacian reduces to the standard graph Laplacian, reproducing Phase 1's scalar Betti curve exactly. This serves as a minimal sanity check.

### 3.4 New Core Types

```python
@dataclass(frozen=True)
class SheafBettiCurve:
    """Sheaf Betti number β₀^F(ε) across filtration scales."""
    epsilon_grid: NDArray[np.float64]
    kernel_dimensions: NDArray[np.int64]
    smallest_eigenvalues: NDArray[np.float64]  # shape (n_steps, m)
    sigma: float
    K: int

@dataclass(frozen=True)
class SheafValidationResult:
    """Output of the σ-sweep experiment."""
    sigma_grid: NDArray[np.float64]
    epsilon_grid: NDArray[np.float64]
    betti_heatmap: NDArray[np.int64]           # shape (n_sigma, n_epsilon)
    peak_sigma: float                           # σ that maximizes max_ε β₀^F
    peak_kernel_dim: int                        # max β₀^F at peak σ
    is_unique_peak: bool                        # True if peak is at σ=0.5 only
    metadata: dict = field(default_factory=dict)
```

### 3.5 The Matrix-Free Approach

This is the Phase 2 architectural equivalent of the analytical H_0 shortcut from Phase 1. Instead of assembling the (N·K²) × (N·K²) sheaf Laplacian L_F as an explicit sparse matrix — which would cost O(|E|·K⁴) memory in complex128 — we implement L_F as a `LinearOperator` whose matrix-vector product is computed on-the-fly.

**The matvec y = L_F · x:**

```
y = zeros(N, K, K)    # complex
for each edge (i, j) with i < j and |γ̃_i - γ̃_j| ≤ ε:
    U = transport(γ̃_j - γ̃_i)         # K×K unitary, O(K²) via eigendecomp shortcut
    diff = x[j] - U @ x[i] @ U†       # K×K complex, O(K³)
    y[j] += diff                       # coboundary adjoint, identity side
    y[i] -= U† @ diff @ U             # coboundary adjoint, transport side
```

**Cost per matvec:** O(|E| × K³). At |E| ≈ 3000 and K = 20: 3000 × 8000 = 24M flops. LOBPCG typically converges in O(100–1000) iterations, giving 2.4B–24B total flops — seconds on a modern CPU.

**Why this works:** LOBPCG and eigsh only require matrix-vector products, not the explicit matrix. The eigendecomposition shortcut (Section 2.5) means each transport U_{ij} is computed in O(K²) from cached V, Λ. The dominant cost per edge is the two K×K matrix multiplies (O(K³)).

### 3.6 Memory Budget (K=20, N=1000, avg degree D≈6 at ε=3)

| Component | Size | Notes |
|-----------|------|-------|
| Precomputed U_{ij} cache | ~3000 × K² × 16B = 19.2 MB | K×K complex128 per edge |
| LOBPCG vectors (m=20) | 20 × N × K² × 16B = 128 MB | Complex working vectors |
| Input/output vectors | 2 × N × K² × 16B = 12.8 MB | x and y = L_F·x |
| Eigendecomposition of A(σ) | K × K × 16B + K × 8B = 6.6 KB | V and Λ, negligible |
| **Total** | **~160 MB** | Trivial on any hardware |

The matrix-free approach eliminates the VRAM bottleneck entirely. K is no longer constrained by memory — the limit is now computation time (each matvec scales as O(|E|·K³)).

**Scaling freedom:** At K=30, memory is ~360 MB. At K=50, ~1 GB. At N=10,000, ~1.6 GB. All are feasible even on CPU. The hard limit is now the LOBPCG convergence time, not memory.

---

## 4. Experimental Design

### 4.1 Phase 2a — Abelian Diagnostic Baseline

**Purpose:** Validate the matrix-free Laplacian implementation by decomposing into K independent scalar problems, and produce a resonance spectrum showing which frequencies of A(σ) the zeros respond to.

**Construction:** The generator sum A(σ) = Σ G_p(σ) is a K×K Hermitian matrix with eigendecomposition A = VΛV†, where Λ = diag(λ_1, ..., λ_K). Working in the eigenbasis of A(σ), the transport matrix is diagonal:

```
V† U_{ij} V = diag(e^{iΔγ·λ_1}, ..., e^{iΔγ·λ_K})
```

In this basis, the coboundary on a K×K matrix H decomposes:

```
(V† U H U† V)_{kl} = e^{iΔγ·(λ_k - λ_l)} (V†HV)_{kl}
```

Each (k,l) entry evolves independently with frequency ω_{kl} = λ_k - λ_l. This yields K² independent scalar "twisted" graph Laplacians.

**Key frequencies:**
- **Diagonal entries (k=l):** ω_{kk} = 0. These K copies reproduce the standard graph Laplacian on the unfolded zeros, identical to Phase 1's scalar H_0.
- **Off-diagonal entries (k≠l):** ω_{kl} = λ_k - λ_l. These are "frequency-twisted" Laplacians testing whether zero spacings resonate at the eigenfrequencies of the prime generator.

**Validation criterion:** The K diagonal blocks must reproduce Phase 1's scalar Betti curve β₀(ε). If they don't, the Laplacian implementation has a bug.

**Output:** A K×K resonance matrix R where R_{kl} = max_ε dim ker L_{kl}(ε) at each eigenfrequency difference. This shows which prime-encoded harmonics are compatible with the zero spacings before the full non-abelian coupling is turned on.

**Note on eigenvalue degeneracies:** If A(σ) has repeated eigenvalues, some frequencies coincide, reducing the number of distinct scalar Laplacians. This is harmless for the full computation but reduces the information content of the resonance matrix. The number of distinct frequencies should be reported.

### 4.2 Phase 2b — The σ-Sweep Experiment

**The core falsifiable experiment.**

**Parameters:**
- K = 20 (fiber dimension)
- N = 1000 (number of zeta zeros)
- σ grid: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
- ε grid: 200 steps from 0.0 to 3.0
- m = 20 (number of smallest eigenvalues to compute)
- τ = 10⁻⁶ · ‖L_F‖_F (relative numerical tolerance)

**Procedure for each σ:**
1. Build generators G_p(σ) for all primes p ≤ 19
2. Compute A(σ) = Σ G_p(σ) and eigendecompose: A = V Λ V†
3. For each ε in the sweep grid (skipping ε=0, which is degenerate):
   a. Determine the edge set E(ε) = {(i,j) : |γ̃_i - γ̃_j| ≤ ε, i < j}
   b. Construct the matrix-free LinearOperator L_F(ε, σ) via `as_linear_operator(ε)`
   c. Compute the m smallest eigenvalues via LOBPCG on the LinearOperator
   d. Count eigenvalues below τ → record β₀^F(ε, σ)
4. Produce the sheaf Betti curve β₀^F(ε) for this σ

**Total computation:** 9 σ values × ~200 ε steps × LOBPCG per step = ~1,800 eigensolver calls. Each LOBPCG call runs O(iterations × |E| × K³) flops via the matrix-free matvec. At K=20, |E|≈3000, and ~500 iterations: ~12B flops per call, ~seconds on CPU. Estimated total: 1–5 hours on CPU, potentially faster with GPU-accelerated matrix multiplies.

### 4.3 Success Criteria

| Test | RH Consistent | RH Inconsistent |
|------|--------------|-----------------|
| β₀^F(ε, 0.5) plateaus at non-zero value | Non-trivial global sections exist at critical line | Kernel is trivial everywhere |
| β₀^F(ε, σ≠0.5) collapses to zero | Non-critical weightings are frustrated | Non-trivial kernel at σ ≠ 0.5 |
| σ = 0.5 is a unique maximum of max_ε β₀^F(ε, σ) | Critical line is a topological phase transition | Maximum is elsewhere or at multiple σ |
| Phase 2a zero-frequency blocks reproduce Phase 1 Betti | Internal consistency | Bug in Laplacian implementation |

**The phase transition signature:** Plot max_ε β₀^F(ε, σ) as a function of σ. If RH holds, this curve should show a sharp peak at σ = 0.5. The width and height of this peak encode the strength of the topological phase transition.

### 4.4 Self-Consistency Loop (Stretch Goal)

If a non-trivial kernel is found at σ = 0.5:

1. Extract the global section f ∈ ker(L_F), reshaped as f(v_i) ∈ Herm(K) at each zero
2. Compute the eigenvalues spec(f(v_i)) of the local Hermitian matrix at each zero
3. Test correlation between spec(f(v_i)) and the zero values {γ_i}

If the eigenvalues of the global section reproduce the zeros that defined it, the Hilbert-Pólya loop is closed: we have computationally constructed a candidate for the self-adjoint operator whose spectrum is the non-trivial zeros of the Riemann zeta function.

---

## 5. Publication Figures

### 5.1 Figure 1: Sheaf Betti Curves

Three-panel figure:
- **Panel A:** β₀^F(ε) at σ = 0.5 — the critical sheaf Betti curve, with the plateau region highlighted
- **Panel B:** Overlay of all 9 σ values showing convergence behavior; σ = 0.5 highlighted in blue, others in grey
- **Panel C:** 2D heatmap of β₀^F(ε, σ) — the phase diagram, with the σ = 0.5 ridge visible

### 5.2 Figure 2: The σ Peak

Single-panel figure: max_ε β₀^F(ε, σ) vs σ, showing the phase transition at 1/2. Error bars from bootstrap resampling of the ε grid.

### 5.3 Figure 3: Phase 2a Resonance Matrix (Supplementary)

K×K heatmap of the abelian resonance matrix R_{kl}, showing which prime-ratio frequencies the zero spacings resonate with.

---

## 6. Integration with Phase 1

### 6.1 Shared Components (No Changes)

- `atft/core/types.py` — existing types unchanged, new types added
- `atft/core/protocols.py` — no changes needed
- `atft/sources/` — all three sources reused as-is
- `atft/feature_maps/spectral_unfolding.py` — zeta unfolding reused
- `atft/topology/analytical_h0.py` — Phase 2a abelian diagnostic validates against this

### 6.2 Extension Points

- `atft/core/types.py` — add SheafBettiCurve and SheafValidationResult
- `atft/topology/sheaf_ph.py` — replace Phase 1 stub with full implementation
- `atft/visualization/plots.py` — add Phase 2 figure functions
- `atft/experiments/` — add Phase 2a and 2b experiment orchestrators

### 6.3 Precision

Phase 2 uses float64 throughout (consistent with Phase 1). The transport matrices U_{ij} are K×K complex128 (complex unitary). The matrix-free matvec performs K×K complex128 matrix multiplies per edge. L_F is never materialized as an explicit matrix — all operations go through the LinearOperator interface.

---

## 7. Hardware Constraints

### 7.1 Matrix-Free Memory Budget

The matrix-free approach (Section 3.4) eliminates the VRAM bottleneck. Memory is dominated by LOBPCG working vectors and the precomputed transport cache, both of which scale as O(N·K²) and O(|E|·K²):

| Config | Working memory | Status |
|--------|---------------|--------|
| K=20, N=1000 | ~160 MB | Trivial |
| K=30, N=1000 | ~360 MB | Trivial |
| K=20, N=5000 | ~800 MB | Comfortable |
| K=30, N=5000 | ~1.8 GB | Comfortable |
| K=50, N=10000 | ~8 GB | Feasible on 12 GB GPU |

**The constraint is now computation time, not memory.** Each LOBPCG matvec costs O(|E|·K³). At K=30 and |E|=3000, that's 81M flops per matvec. With 500 iterations, ~40B flops total per (ε, σ) point — seconds on CPU, sub-second on GPU.

### 7.2 RTX 4080 (12 GB VRAM, Ada Lovelace)

GPU acceleration is useful for:
- Batched K×K matrix multiplies in the matvec (via PyTorch)
- Potential GPU-native LOBPCG implementation
- GUE generation (reusing Phase 1's Dumitriu-Edelman source)

The 12 GB VRAM budget is no longer a constraint for the sheaf Laplacian. The entire Phase 2 pipeline fits comfortably at any reasonable (K, N) configuration.

### 7.3 CPU Fallback

All Phase 2 computations run on CPU via scipy/numpy. GPU is optional and provides speedup but is not required.

---

## 8. Risk Analysis

### 8.1 The Kernel May Be Empty

The sheaf Laplacian kernel could be trivially zero at all σ values, including σ = 0.5. This does NOT disprove RH — it would mean our truncation (K=20, N=1000) is too coarse to resolve the global sections. Mitigation: Phase 2a establishes the abelian baseline first, confirming the machinery works before interpreting null results.

### 8.2 Numerical Precision

At K=20, each edge contributes two K×K matrix multiplies to the matvec. Accumulated floating-point error across ~3000 edges and hundreds of LOBPCG iterations could pollute the smallest eigenvalues. Mitigation: the relative tolerance τ = 10⁻⁶ · ‖L_F‖_F absorbs scale-dependent noise. The Frobenius norm is estimated from edge contributions in O(|E|·K²) without assembling L_F.

### 8.3 LOBPCG Convergence

Iterative eigensolvers can fail to converge for ill-conditioned operators. Mitigation: the `solver="auto"` setting (Section 3.3) tries LOBPCG first, then falls back to `scipy.sparse.linalg.eigsh` with shift-invert mode (shift=0.0, which='LM') if LOBPCG fails to converge within 1000 iterations. Note: the eigsh `sigma` parameter (shift value) is unrelated to the spec's σ (critical line parameter).

### 8.4 Interpretation of Non-Trivial Kernel at σ ≠ 0.5

If global sections appear at unexpected σ values, this could indicate:
- A bug in the transport map construction (most likely)
- The truncation K=20 introduces artifacts
- A genuine mathematical surprise

Protocol: if unexpected results appear, validate against Phase 2a abelian baseline before drawing conclusions.
