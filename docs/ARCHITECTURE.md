# ATFT Framework — Software Architecture (Ti V0.1)

**Project:** Adaptive Topological Field Theory (ATFT) for the Riemann Hypothesis
**Version:** Ti V0.1
**Date:** 2026-03-16

---

## 1. Package Structure

The `atft/` package is organized into seven functional layers, each with a clearly defined responsibility boundary.

```
atft/
├── core/          — Type protocols and shared data types
├── sources/       — Point cloud data sources (zeta zeros, random ensembles)
├── feature_maps/  — Spectral preprocessing transforms
├── topology/      — Sheaf Laplacian construction and eigensolver backends
├── analysis/      — Evolution curves, waypoints, statistical validation
├── visualization/ — Publication-quality plot generation
├── io/            — HDF5 persistence for diagram caching
└── experiments/   — Phase 1–3 experiment orchestrators
```

### 1.1 core/

Defines the type protocol layer that all downstream modules depend on. Contains:

- `PointCloud` — protocol for any indexed set of real vectors
- `FeatureMap` — protocol for transforms mapping a `PointCloud` to a new representation
- `TopologicalInvariant` — protocol for objects extracted from a filtered complex (Betti numbers, persistence diagrams)
- Shared numeric data types and array shape conventions

### 1.2 sources/

Three concrete data sources implementing the `PointCloud` protocol:

| Class | Description |
|---|---|
| `ZetaZerosSource` | Loads Odlyzko high-precision zeta zero tables from disk; supports arbitrary N |
| `GUESource` | Generates GUE eigenvalues via Dumitriu-Edelman tridiagonal construction (exact finite-N distribution, no rejection sampling) |
| `PoissonSource` | Generates i.i.d. Poisson-distributed spacings as a null reference |

The Dumitriu-Edelman construction was selected over naive Wigner matrix sampling because it produces the exact GUE eigenvalue distribution in O(N) space with no diagonalization overhead.

### 1.3 feature_maps/

Preprocessing transforms applied before graph construction:

| Class | Description |
|---|---|
| `SpectralUnfolding` | Maps raw zeros to unit mean spacing using the smooth zeta staircase CDF |
| `IdentityMap` | Pass-through; used for raw-spacing experiments |

**Critical implementation note:** `SpectralUnfolding` uses the analytic smooth zeta staircase CDF (the Riemann-von Mangoldt formula), not rank-based unfolding. Rank-based unfolding (dividing by empirical rank) introduces artificial correlations and must not be used for spectral statistics. See `feedback_unfolding` memory entry.

### 1.4 topology/

The computational core. Contains three eigensolver backends sharing a common interface, plus the transport map builder.

**TransportMapBuilder**
Constructs the u(K) gauge connection matrix for each edge in the Vietoris-Rips graph. Implements four transport modes:

| Mode | Description |
|---|---|
| `global` | Uniform phase rotation; zero curvature (flat connection) |
| `resonant` | Resonance-weighted phase; encodes prime harmonic structure |
| `fe` | Functional-equation mode; breaks Hermiticity off the critical line |
| `prime_weighted` | Canonical prime-weighted u(K) gauge theory (Phase 2 default) |

The connection is parameterized by `(σ, K)` where σ is the off-critical-line parameter and K is the Fourier truncation order (number of prime harmonics included).

**SheafLaplacian** (CPU, matrix-free)
Assembles the full sheaf Laplacian as a dense matrix and solves via LOBPCG. Suitable for small N (N ≤ 2000 approximately).

**SparseSheafLaplacian** (CPU, sparse)
Assembles the Laplacian in BSR (Block Sparse Row) format and solves via `scipy.sparse.linalg.eigsh`. Efficient for large N with sparse graphs (low ε).

**GPUSheafLaplacian** (GPU, CuPy)
Assembles the Laplacian as a CuPy CSR matrix and solves via the GPU eigensolver. Key implementation details:

- Duplicate diagonal block entries arising from edge-by-edge assembly are merged automatically during `tocsr()` via CuPy's COO auto-summing behavior.
- **Spectral flip trick:** The k smallest eigenvalues of L are computed by finding the k largest eigenvalues of `(λ_max · I − L)`. Lanczos-based iterative solvers (ARPACK, LOBPCG, and the CuPy equivalent) converge orders of magnitude faster on the largest eigenvalues of a matrix than on the smallest. The flip maps the smallest eigenvalue problem to a largest eigenvalue problem without altering the eigenvectors.
- Explicit CuPy memory pool cleanup is called between grid points to prevent GPU OOM accumulation during σ sweeps.

### 1.5 analysis/

Post-processing after eigenvalue extraction:

| Class | Description |
|---|---|
| `EvolutionCurveComputer` | Computes spectral sum S(σ, ε) across a σ × ε parameter grid |
| `WaypointExtractor` | Identifies peak location, turnover point, and monotonicity classification |
| `StatisticalValidator` | Computes contrast ratios and signal ratios against random controls |

### 1.6 visualization/

Generates publication-ready figures:

- Betti number evolution curves (Phase 1)
- Transport mode comparison plots (Phase 2)
- σ-sweep spectral sum curves with error bands (Phase 3)
- Fourier sharpening progression plots (Phase 3)

### 1.7 io/

HDF5-backed caching layer for persistence diagrams. Avoids redundant Vietoris-Rips computations when re-running analysis on cached point clouds.

### 1.8 experiments/

Top-level orchestrators for each experimental phase:

| Module | Role |
|---|---|
| `phase1_benchmark.py` | GUE vs zeta vs Poisson Betti curve comparison |
| `phase2_transport.py` | Transport mode construction and flat connection verification |
| `phase3_sigma_sweep.py` | Multi-prime superposition σ sweep (single machine) |
| `phase3_distributed.py` | Distributed role-based parameter partitioning |
| `aggregate_results.py` | Cross-machine result merging and analysis |

---

## 2. Data Flow Pipeline

```
Odlyzko zero table (raw imaginary parts)
        |
        v
ZetaZerosSource  — loads N zeros starting from zero index Z
        |
        v
SpectralUnfolding  — smooth zeta staircase CDF → mean spacing normalized to 1.0
        |
        v
Vietoris-Rips graph  — neighborhood graph at radius ε (ε parameter)
        |
        v
TransportMapBuilder  — assigns u(K) connection matrix U_e ∈ U(K) to each edge e
        |                (parameterized by σ, K)
        v
SheafLaplacian / SparseSheafLaplacian / GPUSheafLaplacian
        |                (backend selected by N, hardware availability)
        v
Eigenvalues λ_1 ≤ λ_2 ≤ ... ≤ λ_kN
        |
        v
Spectral sum S(σ, ε) = Σ_i f(λ_i)   — aggregation function f chosen per experiment
        |
        v
Contrast ratio = S_zeta / S_random   — normalized against Poisson/GUE control
        |
        v
Signal ratio R = max_σ(S) / S(σ=0.25)   — quantifies peak localization
```

---

## 3. GPU Architecture

### 3.1 Hybrid CPU/GPU Split

The framework uses a deliberate hybrid compute model:

| Task | Device | Rationale |
|---|---|---|
| Edge discovery (Vietoris-Rips) | CPU | Memory-efficient; graph is sparse |
| Batched matrix exponentials (transport) | CPU | NumPy `scipy.linalg.expm` batch over edges |
| Sparse matrix assembly | GPU (CuPy) | Parallelizes block outer products |
| Iterative eigensolver | GPU (CuPy) | Lanczos on GPU outperforms CPU for large sparse matrices |

### 3.2 CuPy COO Auto-Summing

During Laplacian assembly, diagonal blocks receive contributions from every incident edge. Rather than tracking these accumulations explicitly, entries are added as COO (coordinate format) triplets with repeated `(row, col)` indices. CuPy's `coo_matrix.tocsr()` automatically sums duplicate entries, yielding the correct assembled diagonal. This matches the standard finite-element assembly pattern.

### 3.3 Spectral Flip Trick

Finding the k smallest eigenvalues of a large sparse symmetric matrix L via an iterative solver is numerically slow because Lanczos convergence rate scales with the spectral gap at the target end of the spectrum. The smallest eigenvalues of L are at the dense end (near zero).

The flip computes instead the k largest eigenvalues of `M = λ_max · I − L`. The relationship `λ_i(M) = λ_max − λ_{N−i}(L)` maps the smallest eigenvalues of L to the largest eigenvalues of M. Lanczos converges on the largest eigenvalues of M in far fewer iterations. The eigenvectors are identical.

`λ_max` is estimated cheaply via a few power iterations before the main solve.

### 3.4 Memory Management

For σ-sweep grid experiments, each grid point allocates a fresh CuPy sparse matrix and eigenvector array. Between grid points, explicit `cupy.get_default_memory_pool().free_all_blocks()` calls are issued to return GPU memory to the pool. Without this, repeated allocations on an RTX 4080 (16 GB VRAM) exhaust the pool before the sweep completes.

---

## 4. Distributed Computing

### 4.1 Role-Based Parameter Partitioning

`phase3_distributed.py` partitions the (K, N, σ-grid) parameter space across machines by assigning each machine a named role:

| Role | Parameters | Hardware |
|---|---|---|
| `control-cpu` | K=20, N=9877, full σ grid | CPU workstation |
| `gpu-k50` | K=50, N=2000, full σ grid | GPU workstation (RTX 4080) |
| `gpu-k100` | K=100, N=5000, σ grid | RunPod A100 (required) |
| `gpu-k200` | K=200, N=5000, σ grid | RunPod A100 (required) |

Each machine runs independently. No inter-machine communication occurs during computation.

### 4.2 Result Format

Each machine writes results to a JSON file with schema:

```json
{
  "role": "gpu-k50",
  "K": 50,
  "N": 2000,
  "sigma_grid": [...],
  "epsilon_grid": [...],
  "spectral_sums": {"5.0": [...], "3.0": [...]},
  "random_controls": {...},
  "timestamp": "..."
}
```

### 4.3 Aggregation

`aggregate_results.py` ingests the JSON files from all roles, computes cross-K contrast ratios, fits the Fourier sharpening progression, and generates the combined publication figures.

---

## 5. Test Suite

The test suite contains 171+ tests organized to mirror the package structure:

```
tests/
├── test_core.py
├── test_sources.py
├── test_feature_maps.py
├── test_topology.py
├── test_analysis.py
├── test_visualization.py
├── test_io.py
└── test_experiments.py
```

Key testing conventions:

- All tests validate against known analytical results where available (e.g., GUE level spacing distribution moments, Poisson spacing exponential distribution).
- GPU tests (those exercising `GPUSheafLaplacian`) are decorated with a `@requires_cupy` skip marker and are silently skipped when CuPy is not installed. This allows the full CPU suite to run on machines without a GPU.
- Spectral unfolding tests verify that the output mean spacing equals 1.0 to within numerical tolerance and that no rank-based normalization path is reachable.
