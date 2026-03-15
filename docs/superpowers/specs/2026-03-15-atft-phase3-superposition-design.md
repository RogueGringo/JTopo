# ATFT Phase 3: Superposition & Scale Design Specification

**Date:** 2026-03-15
**Status:** Approved (Rev 1 — post spec review)
**Authors:** Blake Jones, Claude (Opus 4.6)
**Depends on:** Phase 2 Design Spec (2026-03-15-atft-phase2-sheaf-design.md)

---

## 1. Overview

### 1.1 Goal

Replace the single-prime resonant transport with a multi-prime superposition connection that encodes the Riemann explicit formula as phase interference, scale the sheaf Laplacian from N=30 to N=10,000 using sparse linear algebra, and run a definitive control test to determine whether zeta zeros produce a genuine arithmetic signal at sigma = 1/2.

### 1.2 Phase 2 Findings

Phase 2 established three transport modes (global, resonant, functional equation) and proved:

1. **Global transport is flat.** Holonomy telescopes to zero algebraically. Useless for sigma-detection.
2. **Resonant transport creates genuine holonomy** via non-commuting per-prime generators (14/14 triangles non-trivial, ||H-I|| up to 3.25).
3. **The FE generator's sigma=0.5 peak is geometric, not arithmetic.** The Frobenius-normalized FE generator is Hermitian at sigma=0.5, making transport automatically unitary. This creates maximum topological rigidity at sigma=0.5 regardless of input data. Control test confirmed: zeta zeros, random points, and GUE points all show comparable contrast (0.16-0.31).

**Root cause:** Encoding s <-> 1-s symmetry INTO the generator guarantees unitarity at sigma=0.5 for ANY point set. The signal must come from the DATA, not the CONNECTION.

### 1.3 What Phase 3 Fixes

Phase 3 introduces phase factors `e^{i*delta_gamma*log(p)}` into the generator sum. These make the generator non-Hermitian even at sigma=0.5 (complex-weighted sum of Hermitian matrices). The ONLY way unitarity can emerge at sigma=0.5 is if the specific gaps delta_gamma of the zeros create coherent destructive interference of the imaginary components — i.e., if the Riemann explicit formula is physically pulling the strings of the topology.

### 1.4 Project Trajectory

- **Phase 1 (complete):** Scalar H_0 benchmark. 92 tests.
- **Phase 2 (complete):** Sheaf-valued PH with matrix fibers. 171 tests. Proved geometric artifact.
- **Phase 3 (this spec):** Multi-prime superposition transport with vector fibers at scale. Definitive control test.
- **Phase 4 (future):** Higher cohomology (H^1 via 1-Laplacian), high-altitude zeros.

---

## 2. Mathematical Foundation

### 2.1 Multi-Prime Superposition Generator

For an oriented edge e = (i -> j) with gap delta_gamma = gamma_j - gamma_i > 0, the superposition generator is:

```
A_{ij}(sigma) = sum_{p prime, p <= K} e^{i * delta_gamma * log(p)} * B_p(sigma)
```

where the per-prime basis matrices are:

```
B_p(sigma) = log(p) * [p^{-sigma} * rho(p) + p^{-(1-sigma)} * rho(p)^T]
```

and rho(p) is the K x K truncated left-regular representation: rho(p)|n> = |pn> if pn <= K, else 0.

**Key properties:**

- `B_p(sigma)` is a real K x K matrix, identical to Phase 2's functional equation generator `G_p^FE(sigma)` before Frobenius normalization. At sigma = 1/2, it is symmetric (Hermitian): `B_p(0.5) = log(p)/sqrt(p) * (rho(p) + rho(p)^T)`. At sigma != 1/2, `B_p` is real but NOT symmetric since `p^{-sigma} != p^{-(1-sigma)}`.
- `e^{i * delta_gamma * log(p)}` is a complex scalar that depends on the edge gap AND the prime.
- `A_{ij}(sigma)` is a COMPLEX K x K matrix — a sum of real matrices with complex coefficients.
- At sigma = 0.5: each B_p is Hermitian, but the complex phases make A_{ij} non-Hermitian in general. A_{ij} is Hermitian only if `Im(sum_p e^{i*dg*log(p)} * B_p) = 0`, which requires specific arithmetic alignment between the gaps and log-prime frequencies.

**Connection to the Riemann explicit formula:**

The von Mangoldt explicit formula expresses the prime counting function as a sum over zeta zeros:

```
psi(x) = x - sum_rho x^rho / rho - log(2*pi) - ...
```

The phase factor `e^{i * delta_gamma * log(p)}` is the Fourier kernel linking zero spacings to prime frequencies. For actual zeta zeros, these phases create coherent interference patterns reflecting the deep arithmetic structure. For random points, the interference is incoherent noise.

### 2.2 Transport via Matrix Exponential

The transport matrix for edge (i, j) is:

```
U_{ij} = exp(i * A_{ij}(sigma))
```

Computed via eigendecomposition of the K x K complex matrix A_{ij}:

```
A_{ij} = P * diag(lambda_1, ..., lambda_K) * P^{-1}
U_{ij} = P * diag(exp(i*lambda_1), ..., exp(i*lambda_K)) * P^{-1}
```

Since A_{ij} is generally non-Hermitian, the eigenvalues lambda_k may be complex, and U_{ij} is NOT unitary in general. This is the critical difference from Phase 2's FE transport, which was automatically unitary at sigma = 0.5.

**Defective matrix fallback:** If A_{ij} is defective (non-diagonalizable), the eigenvector matrix P is ill-conditioned. When `cond(P) > 1e12`, fall back to `scipy.linalg.expm(1j * A_{ij})` (Pade approximant), which handles defective matrices correctly.

**Edge orientation convention:** Only edges oriented as (i -> j) with i < j are enumerated. The reverse transport U_{ji} is NOT explicitly computed. Instead, the block structure (Section 2.5) uses `U_{ij}` and `U_{ij}^dagger` directly, which is sufficient to construct the Hermitian Laplacian from a single orientation per edge.

### 2.3 Normalization Modes

Two normalization modes are supported and BOTH must be tested in the control experiment:

1. **Normalized (default):** Per-edge Frobenius normalization `A_{ij} -> A_{ij} / ||A_{ij}||_F`. Equalizes connection strength across edges. Isolates phase geometry from magnitude effects.

2. **Unnormalized:** Natural explicit-formula weights `log(p)/p^sigma`. If this mode produces a sigma=0.5 peak for zeta zeros but not controls, the signal is maximally pure — the natural mathematical weights of the primes are doing the work without artificial stabilization.

### 2.4 Vector-Valued Sheaf (C^K Fibers)

Phase 2 used K x K matrix fibers (endomorphism bundle), giving stalks F(v) = C^{K x K} and total dimension N * K^2. At K=50, N=10,000: 25 million dimensions. Infeasible for explicit sparse matrices.

Phase 3 uses K-vector fibers: F(v) = C^K. Total dimension: N * K = 500,000. This is the standard sheaf cohomology H^0(X, F) — vectors that survive parallel transport around loops.

**Mathematical equivalence of the sigma-criticality test:** The non-unitary sigma-criticality penalty acts on C^K vectors exactly as it acts on C^{K x K} matrices. When the connection is non-unitary (sigma != 0.5 under ideal conditions), vectors stretch/shrink during transport and cannot form globally consistent sections. The vector fiber captures the same phenomenon at 1/K-th the cost.

### 2.5 Sheaf Laplacian (Vector Fibers)

**Coboundary operator** delta_0: C^0(V, F) -> C^1(E, F):

For oriented edge e = (i -> j) with transport U_e = U_{ij}:

```
(delta_0 s)_e = U_e * s_i - s_j
```

**Sheaf Laplacian** L = delta_0^dagger * delta_0. The Dirichlet energy is:

```
<s, Ls> = sum_e ||U_e * s_i - s_j||^2
```

L is Hermitian positive semi-definite by construction (L = delta^dagger delta >= 0).

**Block structure** (N x N blocks of size K x K):

For each oriented edge (i -> j) with transport U_{ij}:

| Position | Block |
|----------|-------|
| (i, i) diagonal | accumulate `U_{ij}^dagger * U_{ij}` |
| (j, j) diagonal | accumulate `I_K` |
| (i, j) off-diagonal | `-U_{ij}^dagger` |
| (j, i) off-diagonal | `-U_{ij}` |

**Hermiticity check:** `L_{ji} = -U_{ij}` and `L_{ij}^dagger = (-U_{ij}^dagger)^dagger = -U_{ij} = L_{ji}`. Confirmed Hermitian.

For unitary transport (U^dagger U = I): diagonal reduces to `deg(i) * I_K`.
For non-unitary transport: diagonal is a sum of positive-definite matrices `U^dagger U`, NOT a scalar multiple of identity.

### 2.6 Edge Discovery (1D Optimized)

Zeta zeros are 1D sorted points after spectral unfolding (mean spacing = 1). The Rips complex is a banded graph:

| epsilon | Avg degree | |E| for N=10,000 |
|---------|------------|------------------|
| 1.5 | ~3 | ~15,000 |
| 2.0 | ~4 | ~20,000 |
| 3.0 | ~6 | ~30,000 |
| 5.0 | ~10 | ~50,000 |

Edge discovery uses sorted structure + binary search: O(N log N). No O(N^2) broadcasting.

---

## 3. Computational Architecture

### 3.1 Superposition Transport (transport_maps.py)

New methods added to `TransportMapBuilder`:

```python
def build_superposition_bases(self) -> NDArray:
    """Precompute B_p(sigma) for all primes. Returns (P, K, K) real array."""

def build_generator_superposition(self, delta_gamma: float,
                                   normalize: bool = True) -> NDArray:
    """Single-edge superposition generator A_{ij}(sigma). Returns (K, K) complex."""

def batch_transport_superposition(self, delta_gammas: NDArray,
                                   normalize: bool = True) -> NDArray:
    """Batch superposition transport for M edges. Returns (M, K, K) complex."""
```

**Batch computation pipeline:**

```
B_stack: (P, K, K) real           # precomputed per sigma
phases:  (M, P) complex           # e^{i * dg * log(p)} per edge per prime
A_batch: (M, K, K) complex        # einsum('ep,pij->eij', phases, B_stack)
A_norm:  (M, K, K) complex        # optional: A / ||A||_F per edge
evals:   (M, K) complex           # np.linalg.eig eigenvalues
evecs:   (M, K, K) complex        # np.linalg.eig eigenvectors
U_batch: (M, K, K) complex        # P @ diag(exp(i*lambda)) @ P_inv
```

### 3.2 SparseSheafLaplacian (new module)

```python
class SparseSheafLaplacian:
    """BSR sparse sheaf Laplacian with C^K vector fibers.

    Args:
        builder: TransportMapBuilder (provides primes, K, sigma)
        zeros: 1D array of unfolded zeta zeros
        transport_mode: "superposition" (default), "fe", or "resonant"
        normalize: whether to Frobenius-normalize the superposition generator
    """

    def build_edge_list(self, epsilon: float) -> tuple[NDArray, NDArray, NDArray]:
        """1D binary-search edge discovery. Returns (i_idx, j_idx, gaps)."""

    def build_matrix(self, epsilon: float) -> sp.bsr_matrix:
        """Assemble the N*K x N*K BSR sheaf Laplacian."""

    def smallest_eigenvalues(self, epsilon: float, k: int = 100) -> NDArray:
        """Shift-invert eigsh for bottom k eigenvalues."""

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Sum of bottom k eigenvalues (primary metric)."""
```

**BSR assembly pseudocode:**

```
1. edges = build_edge_list(epsilon)          # O(N log N)
2. U_all = builder.batch_transport_*(gaps)   # O(|E| * K^3)
3. For each edge e = (i, j):
     off_diag[(i,j)] = -U[e]^dagger          # (K, K) block
     off_diag[(j,i)] = -U[e]                  # (K, K) block
     diag[i] += U[e]^dagger @ U[e]            # accumulate
     diag[j] += I_K                           # accumulate
4. L = bsr_matrix(all_blocks, shape=(N*K, N*K))
```

### 3.3 Eigensolver Configuration

- **Method:** `scipy.sparse.linalg.eigsh` (Hermitian sparse eigensolver)
- **Mode:** Shift-invert with `sigma=1e-8`, `which='LM'` — targets eigenvalues nearest zero
- **Count:** `k=100` eigenvalues for stable spectral sum (Phase 2 used k=20, which saturated)
- **Tolerance:** `tol=1e-6` (relative)
- **Fallback:** `eigsh` with `which='SM'` if shift-invert fails (singular or ill-conditioned)

**Performance estimate (K=50, N=10K, |E|=20K):**

| Operation | Time |
|-----------|------|
| Edge discovery | < 0.1 sec |
| Superposition transport (20K edges) | ~5 sec |
| BSR assembly | ~1 sec |
| eigsh shift-invert (k=100) | ~10 sec |
| **Total per (sigma, epsilon)** | **~16 sec** |

### 3.4 Memory Budget (K=50, N=10,000, |E|=20,000)

| Component | Size |
|-----------|------|
| B_stack (15 primes x 50 x 50, float64) | 0.3 MB |
| Phase matrix (20K x 15, complex128) | 4.8 MB |
| A_batch + U_batch (20K x 50 x 50, complex128) | 1.6 GB peak |
| BSR matrix (50K blocks x 50 x 50, complex128) | 2.0 GB |
| eigsh workspace (100 Ritz vectors x 500K, complex128) | 800 MB |
| eigsh shift-invert LU factorization (banded, bandwidth ~200) | ~1.6 GB |
| **Total peak** | **~6 GB** |

Fits within 12 GB budget. A_batch can be freed after U_batch is computed. The LU factorization exploits 1D banded structure (bandwidth ~4*K=200), keeping fill-in controlled at O(N*K*b) storage. If LU memory exceeds budget, fall back to LOBPCG (no factorization needed, higher iteration count).

---

## 4. Experiment Design

### 4.1 The Definitive Control Test

The Phase 3 experiment is a direct analog of the Phase 2 control test, but with the geometric artifact removed by the superposition transport.

**Point sets:**

| Set | Description | Trials |
|-----|-------------|--------|
| Zeta | First ~9,877 Odlyzko zeros, spectrally unfolded | 1 |
| Random | Uniform on same interval, same count | 5 |
| GUE | Wigner-surmise spacing, same mean gap | 5 |

**Parameter grid:**

| Parameter | Values |
|-----------|--------|
| sigma | 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75 |
| epsilon | 1.5, 2.0, 2.5, 3.0, 4.0, 5.0 |
| normalize | True, False |
| K | 50 |
| k (eigenvalues) | 100 |

**Total computations:** 13 sigma x 6 epsilon x 11 point sets x 2 normalization modes = 1,716 Laplacian eigendecompositions.

**Estimated runtime:** Transport recomputation only when sigma or point set changes: 13 x 11 x 2 = 286 transport batches (~24 min). BSR + eigsh: 1,716 x 11 sec = ~5.2 hours. **Total: ~6 hours.**

### 4.2 Metrics

For each (sigma, epsilon, point set, normalization) tuple:

1. **Spectral sum:** `S(sigma, epsilon) = sum_{i=1}^{k} lambda_i` — sum of k smallest eigenvalues
2. **Kernel dimension:** `beta_0(sigma, epsilon) = #{lambda_i < tau}` where `tau = 1e-6 * ||L||_F_est`
3. **Symmetrized spectral sum:** `S_sym(sigma) = [S(sigma) + S(1-sigma)] / 2` — for comparing with Phase 2 results

### 4.3 Success Criteria

The experiment has three possible outcomes:

Define the **contrast ratio** as:

```
C(point_set) = [S(0.5) - S(0.25)] / S(0.5)
```

where S(sigma) is the spectral sum at a given epsilon. The **signal strength** is:

```
R = C(zeta) / mean(C(random_1..5), C(GUE_1..5))
```

**Outcome A — Strong Signal (breakthrough):**
R > 2.0 at more than 4 of 6 epsilon values for at least one normalization mode. This means the explicit-formula phase interference creates genuine topological rigidity at the critical line — a computational observation of RH as a topological phase transition.

**Outcome B — Weak Signal (promising):**
1.0 < R < 2.0 consistently (R > 1.0 at more than 4 of 6 epsilon values). The framework captures some arithmetic structure but needs refinement — possibly higher K, more zeros, or Phase 4's H^1 cohomology.

**Outcome C — No Signal (conclusive negative):**
R <= 1.0 or R > 1.0 at fewer than 3 of 6 epsilon values. The superposition transport does not create arithmetic-specific topological rigidity. This rules out H^0 sheaf cohomology with the multiplicative monoid representation as a viable RH test within the ATFT framework.

All three outcomes are scientifically valuable. The experiment is falsifiable.

---

## 5. Module Structure

### 5.1 New Files

| File | Description |
|------|-------------|
| `atft/topology/sparse_sheaf_laplacian.py` | SparseSheafLaplacian class (BSR, vector fibers, eigsh) |
| `tests/test_sparse_sheaf_laplacian.py` | Unit tests for sparse Laplacian |
| `atft/experiments/phase3_superposition_sweep.py` | Massive control test experiment |

### 5.2 Modified Files

| File | Change |
|------|--------|
| `atft/topology/transport_maps.py` | Add superposition transport methods |
| `tests/test_transport_maps.py` | Add superposition transport tests |

### 5.3 Unchanged (Phase 2 Reference)

All existing modules, tests, and experiment scripts remain untouched. The Phase 2 matrix-fiber implementation serves as a reference. 171 existing tests are unaffected.

### 5.4 Validation Strategy

1. **Small-N cross-validation:** At N=10, K=6: verify SparseSheafLaplacian eigenvalues match dense reference computation (explicit L = delta^dagger delta built manually).
2. **Hermiticity:** Verify `||L - L^dagger|| < 1e-12` for all test cases.
3. **PSD:** Verify all eigenvalues >= -1e-10 (numerical tolerance).
4. **Degenerate case:** At epsilon=0, kernel dimension = N*K (no edges, full kernel).
5. **Unitary limit:** For single-prime superposition (K=3, only prime 2), when `delta_gamma = 2*pi/log(2)` (phase factor = 1), the generator reduces to the Hermitian `B_2(0.5)`. Verify transport is unitary and diagonal block is `I_K`.
6. **Single-prime degeneration:** With K=3 (only prime 2) and `normalize=True`, superposition transport should produce the same eigenvectors as FE transport (eigenvalues may differ by a scalar factor due to different normalization denominators).

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| eigsh shift-invert fails on near-singular L | Stalls computation | Fallback to `which='SM'`; regularize shift to `sigma=1e-6` |
| Memory exceeds 12GB at K=50 | OOM crash | Free A_batch after computing U_batch; reduce K to 30 if needed |
| Superposition generator has pathological eigenvalues | Numerical instability in exp(iA) | Monitor cond(P); if cond(P) > 1e12, fall back to scipy.linalg.expm |
| 6-hour runtime too long for iteration | Slow development cycle | Start with K=20 N=1000 for fast iteration; scale up for final run |
| Phase factor coherence is weak even for zeta zeros | No signal (Outcome C) | Scientifically valid negative result; proceed to Phase 4 (H^1) |

---

## 7. Dependencies

### 7.1 Software

- Python >= 3.11
- NumPy >= 2.0 (batched np.linalg.eig on 3D arrays)
- SciPy >= 1.11 (bsr_matrix, eigsh with shift-invert)
- Existing ATFT package (Phase 1 + Phase 2)

### 7.2 Data

- `data/odlyzko_zeros.txt` — existing file with ~9,877 zeros (sufficient for N=9,877)
- No additional data downloads required for the initial experiment

### 7.3 Hardware

- CPU: any modern multi-core (NumPy parallelizes eigendecomp internally)
- RAM: >= 16 GB (6 GB peak workload + LU factorization headroom + OS)
- GPU: not required (future CuPy/JAX upgrade possible for Phase 3b)
