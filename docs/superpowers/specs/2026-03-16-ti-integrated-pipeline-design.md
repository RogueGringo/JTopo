# Ti Integrated Pipeline: Engineering, Scaling, and H^1 Cohomology

**Date:** 2026-03-16
**Status:** Draft
**Authors:** Blake Jones, Claude (Opus 4.6)
**Depends on:** Phase 3 Superposition Design (2026-03-15-atft-phase3-superposition-design.md)

---

## 1. Overview

### 1.1 Goal

Transform the Ti V0.1 codebase from a functional research prototype into a publication-ready, falsification-driven computational physics framework capable of producing definitive evidence for or against the Riemann Hypothesis at scale K=100-500, with a new H^1 cohomology observable that measures holonomy curvature directly.

### 1.2 Motivation

The current codebase is functionally complete for K<=50 CPU validation but has critical gaps:

- **GPU backend untested:** `torch_sheaf_laplacian.py` (595 lines, zero tests) is the sole path to K=100+
- **No falsification criteria:** Results are interpreted post-hoc; no pre-committed thresholds distinguish "supports RH" from "framework artifact"
- **H^0 observable only:** The 0-Laplacian measures global section obstruction; the 1-Laplacian measuring holonomy curvature may be a sharper probe of the critical line
- **Reproducibility undocumented:** The K=20 670x discrimination ratio cannot be reproduced from a single command
- **No CI pipeline:** Code changes could silently break numerical results

### 1.3 Design Philosophy: Integrated Pipeline

Each tier produces both a code deliverable and a theoretical deliverable:

| Tier | Code | Theory |
|---|---|---|
| 1. Engineering | Test suite, CI, GPU refactor, reproducibility | Formal falsification criteria |
| 2. Scaling | K=100/200/500 experiments, scaling analysis | Formal conjectures with predictions |
| 3. H^1 Physics | Triangle complex, delta_1, 1-Laplacian | Conjecture: H^1 more sensitive than H^0 |

### 1.4 Success Criteria

The project succeeds if:

1. Every published numerical claim is reproducible from a single script
2. Falsification criteria are stated before K=100 data is collected
3. K=100 and K=200 data either confirms or refutes each conjecture with quantified confidence
4. The paper withstands scrutiny from a skeptical number theorist

The project is valuable even if RH is not supported — a rigorous negative result (sigma*(K) converges to != 0.5, or H^1 shows no peak) is a genuine contribution.

---

## 2. Current State Assessment

### 2.1 What Works

- 212 passing tests across 18 test files
- Four Laplacian backends: Phase 2 dense (`SheafLaplacian`, matrix fibers C^{KxK}), CPU sparse (`SparseSheafLaplacian`, BSR), GPU CuPy (`GPUSheafLaplacian`, CSR), GPU PyTorch (`TorchSheafLaplacian`, CSR)
- Transport modes vary by backend: Phase 2 supports global/resonant; Phase 3 backends support superposition/fe/resonant
- K=20 full sweep complete (670x discrimination, monotonic sigma profile)
- K=50 scout complete (first spectral turnover at eps=5.0)
- K=100 partial (2 data points showing eps=3.0 reversal)
- PyTorch backend enables AMD ROCm + NVIDIA CUDA
- Distributed sweep infrastructure with role-based partitioning

### 2.2 Critical Gaps

**Testing:**
- `torch_sheaf_laplacian.py`: 595 lines, zero tests, custom Lanczos implementation unvalidated
- No end-to-end regression test at K>6
- No test validating the 670x discrimination ratio
- Experiment scripts (4 files, ~900 lines) completely untested

**Architecture:**
- Edge discovery logic copy-pasted across 3 backends (~60% code duplication)
- No abstract base class enforcing consistent Laplacian interface
- Dead code: `SheafLaplacian.kernel_dimension()`, `extract_global_sections()`, singular `transport()` method

**Infrastructure:**
- No CI pipeline (GitHub Actions)
- No reproducibility script
- No configuration management (constants scattered across files)
- No CHANGELOG

**Theory:**
- No pre-committed falsification criteria
- No formal conjectures with testable predictions
- H^1 cohomology: zero foundation (no triangle enumeration, no delta_1, no 1-Laplacian)
- Hilbert-Polya gap undocumented as explicit open problem

---

## 3. Tier 1: Engineering Hardening + Falsification Framework

### 3.1 GPU Backend Consolidation

**Problem:** Three Laplacian implementations share ~60% of their logic (edge discovery, block expansion, transport dispatch) via copy-paste. Bug fixes require 3+ edits. Behavior may diverge silently.

**Solution:** Extract `BaseSheafLaplacian` abstract base class for the three Phase 3 vector-fiber backends.

**Shared logic (moves to base class):**
- `build_edge_list(zeros, epsilon)` — sorted binary search, returns (i, j, gap) arrays
- Transport mode dispatch — `if mode == "superposition": ...`
- Spectral sum / kernel dimension computation from eigenvalues
- Parameter validation (K, epsilon, sigma ranges)

**Backend-specific (stays in subclasses):**

| Backend | Matrix format | Eigensolver | GPU? |
|---|---|---|---|
| `SparseSheafLaplacian` | SciPy BSR | eigsh shift-invert | No |
| `GPUSheafLaplacian` | CuPy CSR | spectral flip + LOBPCG | NVIDIA only |
| `TorchSheafLaplacian` | PyTorch sparse CSR | spectral flip + Lanczos | NVIDIA + AMD |

**Phase 2 `SheafLaplacian` (matrix fibers):** Left untouched. It uses C^{KxK} matrix fibers (fundamentally different from the C^K vector fibers of Phase 3 backends) and serves only as an internal reference for Phase 2 validation. It cannot share a base class with the vector-fiber backends. No changes to this file.

**New file:** `atft/topology/base_sheaf_laplacian.py` (~150 lines)

**Impact on existing code:** The three Phase 3 backends shrink by ~40% each. Edge discovery becomes a single source of truth that Tier 3's `TriangleComplex` will extend.

**Correctness verification:** All 212 existing tests must pass after refactoring. Regression tolerance: eigenvalue differences > 1e-12 (absolute) or > 1e-10 (relative to eigenvalue magnitude) between pre- and post-refactoring are failures. The regression test captures golden eigenvalue snapshots before refactoring begins and compares after.

### 3.2 Test Infrastructure

**3.2.1 Torch backend test suite** (`tests/test_torch_sheaf_laplacian.py`)

| Test | What it validates |
|---|---|
| Dense equivalence at K=6, N=5 | Torch Laplacian matches brute-force dense construction |
| Cross-validation vs SparseSheafLaplacian at K=20, N=200 | Eigenvalues agree to 1e-10 relative tolerance |
| Lanczos convergence | Custom `_lanczos_largest` produces eigenvalues matching scipy eigsh |
| Spectral flip correctness | Flipped eigenvalues match direct SA eigenvalues |
| GPU fallback chain | Falls back gracefully when CUDA unavailable |
| Transport mode dispatch | All 3 Phase 3 modes (superposition, fe, resonant) produce correct-shaped output |
| VRAM cleanup | Memory pool freed between calls |

Tests that require GPU are decorated with `@pytest.mark.gpu` and skipped in CI.

**3.2.2 End-to-end regression test** (`tests/test_regression.py`)

| Test | Assertion |
|---|---|
| K=20 golden reference | S(sigma=0.5, eps=3.0, N=200) == frozen_value +/- 1e-8 |
| Discrimination ratio | S_zeta / S_random > 100 at K=20, N=500, eps=5.0 |
| Monotonicity at K=20 | S(0.25) < S(0.50) < S(0.75) at eps=5.0 |
| Superposition != FE | S_superposition != S_fe at sigma=0.3 (different transport modes give different results) |

**3.2.3 CI pipeline** (`.github/workflows/test.yml`)

- Trigger: push to master, all PRs
- Matrix: Python 3.11, 3.12
- Steps: install deps, pytest (CPU only), lint check
- Badge in README

### 3.3 Reproducibility Protocol

**New script:** `scripts/reproduce_k20.py`

- Runs the exact K=20 N=9877 experiment matching published parameters
- Sigma grid: [0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.75]
- Epsilon grid: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
- k_eig: 100, transport: superposition, normalize: True
- 5 random control trials, 5 GUE control trials
- Output: `output/reproduction_k20.csv` with columns (sigma, epsilon, S_zeta, S_random_mean, S_random_std, S_gue_mean, S_gue_std, discrimination_ratio)
- Prints PASS/FAIL: "Published ratio: 670x. Reproduced: {value}x. Tolerance: 5%. Verdict: PASS"
- Estimated runtime: ~2 hours on CPU

### 3.4 Falsification Criteria

**New document:** `docs/FALSIFICATION.md`

Pre-committed before any K=100+ data is collected. Thresholds are frozen at commit time.

**Framework falsification (the ATFT construction is flawed):**

| # | Criterion | Threshold | What it means |
|---|---|---|---|
| F1 | Peak migration failure | sigma*(K=100) > 0.60 or < 0.40 | Peak is not approaching 0.5 |
| F2 | Contrast collapse | C(K=100) < C(K=50) | Sharpening has reversed |
| F3 | Discrimination collapse | R(K=100) < 10 | Arithmetic signal has vanished |
| F4 | GUE develops peak | C_GUE(K=100) > 0.5 * C_zeta(K=100) | Signal is statistical, not arithmetic |

**RH falsification under ATFT (the hypothesis is not supported by this framework):**

| # | Criterion | Threshold | What it means |
|---|---|---|---|
| R1 | Peak converges away from 0.5 | sigma*(K) → L where |L - 0.5| > 0.02 as K → ∞ | Critical line is not special |
| R2 | Peak width does not narrow | width(K=200) >= width(K=100) | No phase transition forming |
| R3 | Scaling exponent non-positive | alpha <= 0 in C(K) ~ K^alpha fit | Framework has finite resolution |

**Positive evidence thresholds:**

| # | Criterion | Threshold | What it means |
|---|---|---|---|
| P1 | Peak migration on track | 0.45 <= sigma*(K=100) <= 0.52 | Consistent with sigma* → 0.5 |
| P2 | Contrast growing | C(K=100) > 1.5 * C(K=50) | Sharpening continues |
| P3 | Discrimination growing | R(K=100) > R(K=50) | Arithmetic signal strengthens |
| P4 | Bandwidth propagation | Turnover at eps=2.0 by K=200 | Sharpening reaches finer scales |

---

## 4. Tier 2: Scaling Experiments + Publication

### 4.1 Formal Conjectures

Stated in the paper before K=100+ data is collected:

**Conjecture 1 (Peak Migration):**
sigma*(K) = 1/2 + c / log(K) for some constant c ∈ ℝ (negative if peak approaches 0.5 from below, as suggested by K=50 data showing sigma* ≈ 0.40).

Test: Fit sigma*(K) for K in {20, 50, 100, 200}. Accept if R^2 > 0.95 for the 1/log(K) model and the extrapolated intercept (K → infinity) is within [0.495, 0.505].

**Conjecture 2 (Contrast Divergence):**
C(K) diverges as K → infinity. Specifically, C(K) ~ K^alpha for some alpha > 0.

Test: Fit C(K) to log and power-law models. Accept if the best-fit model has positive slope and R^2 > 0.90.

**Conjecture 3 (Arithmetic Discrimination):**
R(K) = S_zeta(sigma*) / S_GUE(sigma*) is monotonically increasing in K.

Test: R(K=100) > R(K=50) > R(K=20). Fail if any adjacent pair shows decrease.

**Conjecture 4 (Bandwidth Propagation):**
The minimum epsilon at which S(sigma) is non-monotonic decreases with K.

Test: Record eps_min(K) for each K. Predict: eps_min(K=200) < eps_min(K=100) < eps_min(K=50).

### 4.2 Experiment Campaign

All experiments run on RunPod MI300X (192 GB VRAM, $1.51/hr) using TorchSheafLaplacian backend.

**Fine sigma grid near critical line:**
[0.25, 0.30, 0.35, 0.40, 0.44, 0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.54, 0.56, 0.60, 0.75]

| Experiment | K | N | Epsilon grid | Controls | Est. cost |
|---|---|---|---|---|---|
| K=50 full N | 50 | 9877 | {3.0, 5.0} | 3 random, 3 GUE | ~$30 |
| K=100 full | 100 | 9877 | {2.0, 3.0, 4.0, 5.0} | 5 random, 5 GUE | ~$45 |
| K=200 full | 200 | 9877 | {2.0, 3.0, 5.0} | 3 random, 3 GUE | ~$120 |
| K=500 scout | 500 | 5000 | {3.0, 5.0} | zeta only | ~$72 |

Total estimated budget: ~$270.

**Cost basis:** Estimates assume MI300X at $1.51/hr (RunPod community pricing as of 2026-03-16). Per-grid-point timings extrapolated from K=50 GPU scout (~3 min/point on RTX 4080); K=100 and K=200 scale roughly as O(K^2) due to the K×K transport eigendecomposition. K=50 may encounter VRAM pressure at N=9877 on 80 GB A100 (estimated ~60 GB); MI300X (192 GB) provides margin. If K=200 exceeds MI300X capacity, reduce N or use gradient checkpointing.

### 4.3 Scaling Analysis Tool

**New script:** `scripts/scaling_analysis.py` (supersedes `scripts/validate_spectral_scaling.py`)

Input: JSON result files from each K experiment.
Output:
- `output/scaling/scaling_results.json` — full structured results with metadata
- `output/scaling/scaling_results.csv` — flat table for import into paper
- `output/scaling/figures/` — publication-quality plots:
  - `sigma_star_vs_K.pdf` — peak migration with 1/log(K) fit
  - `contrast_vs_K.pdf` — contrast divergence with power-law fit
  - `discrimination_vs_K.pdf` — arithmetic discrimination ratio
  - `S_sigma_overlay.pdf` — S(sigma) curves at K=20/50/100/200 overlaid
  - `bandwidth_propagation.pdf` — eps_min(K) trajectory

Features:
- Bootstrap confidence intervals (1000 resamples of control trials)
- AIC model comparison for scaling fits
- Automated falsification check against FALSIFICATION.md thresholds
- Git commit hash embedded in output metadata for exact reproducibility

### 4.4 Paper Structure

The revised paper follows the pre-registration model:

1. Introduction (existing, minor updates)
2. Mathematical Framework (existing, minor updates)
3. Computational Methods (existing + Tier 1 infrastructure)
4. **Pre-Registered Predictions** (NEW — Conjectures 1-4 with thresholds)
5. **Results** (K=20 through K=500 data)
6. **Verdicts** (NEW — each conjecture: confirmed/refuted/inconclusive, with confidence intervals)
7. Discussion (updated — honest assessment, Hilbert-Polya gap, what remains)
8. Conclusion

---

## 5. Tier 3: H^1 Sheaf Cohomology

### 5.1 Mathematical Foundation

**Current observable (H^0):**
L^(0) = delta_0^dagger delta_0, where delta_0: C^0 → C^1 maps vertex sections to edge discrepancies.
Measures: global section obstruction. "Can you find a consistent vector assignment across all vertices?"

**New observable (holonomy curvature via 1-Laplacian):**
L^(1) = δ₀ δ₀† + δ₁† δ₁, where δ₁: C^1 → C^2 maps edge sections to triangle discrepancies.
Measures: holonomy curvature — the gauge flux trapped in closed loops of the Vietoris-Rips complex.

**Key subtlety:** For a non-flat connection (which is the generic case off the critical line), δ₁ ∘ δ₀ ≠ 0, so {C^0, C^1, C^2} does not form a true cochain complex. L^(1) is still well-defined and PSD, but its kernel does not compute sheaf cohomology in the classical sense. Instead, the spectrum of L^(1) directly encodes the holonomy defect around every triangle — precisely the curvature of the gauge connection.

**Why L^(1) may be sharper than L^(0):**
L^(0) = δ₀† δ₀ detects the failure of global consistency — "can you find a consistent section across all vertices?" This can fail for many reasons (noise, finite-N effects, numerical error). L^(1) detects something more specific: the gauge curvature trapped in triangular loops. If the prime-encoded connection becomes flat (holonomy → I) precisely at σ = 0.5 for zeta zeros, L^(1) measures this directly through the holonomy defect term. L^(0) only sees it indirectly as a byproduct of global section failure.

### 5.2 Implementation

**New module: `atft/topology/triangle_complex.py`**

`TriangleComplex` class:
- Constructor: sorted zeros array + epsilon
- Produces: edge list (i, j, gap) AND triangle list (i, j, k)
- Triangle enumeration: for each edge (i, j), binary search for k > j where |zeros[k] - zeros[i]| <= epsilon AND |zeros[k] - zeros[j]| <= epsilon
- Complexity: O(|E| * mean_degree) for triangles. For N=9877 eps=5.0, mean_degree ~10, expect |T| ~ 200K-500K. For small epsilon (eps < 1.5), |T| may be zero — the code must handle this gracefully (delta_1 becomes a 0-row matrix, L^(1) reduces to δ₀ δ₀†).
- Replaces the ad-hoc `build_edge_list()` currently in each Laplacian backend (inherits from Tier 1's refactoring)

**New module: `atft/topology/coboundary_operators.py`**

`CoboundaryOperator` class:
- `build_delta_0(edges, transport_maps, K)` → sparse matrix (|E|K × NK)
  - delta_0: C^0(V) → C^1(E), so rows = edge sections, columns = vertex sections
  - (δ₀ s)_{ij} = U_{ij} s_j − s_i (same convention as existing `SheafLaplacian`)
- `build_delta_1(edges, triangles, transport_maps, K)` → sparse matrix (|T|K × |E|K)
  - delta_1: C^1(E) → C^2(T), so rows = triangle sections, columns = edge sections
  - For each ordered triangle (i < j < k): (δ₁ f)_{ijk} = U_{ij} f_{jk} − f_{ik} + f_{ij}
  - Extension maps: F_{[i,j]←[i,j,k]} = I, F_{[i,k]←[i,j,k]} = I, F_{[j,k]←[i,j,k]} = U_{ij}
- **Important:** δ₁ ∘ δ₀ = 0 if and only if the connection is flat on each triangle (U_{ij} U_{jk} = U_{ik}). For a non-flat connection, the residual (δ₁ δ₀ s)_{ijk} = (U_{ij} U_{jk} − U_{ik}) s_k measures exactly the holonomy defect around the triangle. This means L^(1) does not compute true sheaf cohomology H^1, but rather measures holonomy curvature — which is precisely the observable we want. At σ = 0.5 for zeta zeros (where the connection is nearly flat), the holonomy defect should be small; off the critical line, it should be large. This gives a more direct probe of criticality than H^0.
- **Stalk convention:** All stalks (vertex, edge, triangle) are C^K. Edge stalk F([i,j]) is identified with F(i) — the "lower-indexed" vertex stalk — so that (δ₀ s)_{ij} = U_{ij} s_j − s_i lives in F(i). This identification determines how vertex-vertex transport maps U_{ij} act on edge sections within the δ₁ formula.
- Both operators stored in CSR format (compatible with existing GPU backends)

**New module: `atft/topology/h1_sheaf_laplacian.py`**

`H1SheafLaplacian` class:
- Assembles L^(1) = delta_0 @ delta_0.H + delta_1.H @ delta_1
- Inherits eigensolver infrastructure from `BaseSheafLaplacian` (Tier 1)
- Spectral sum S^(1)(sigma, epsilon) as the new observable
- GPU-only (PyTorch backend) — CPU infeasible at scale due to |T| >> |E|

### 5.3 Scaling Estimates

For N=9877, eps=5.0, K=100:
- |E| ~ 50,000 edges
- |T| ~ 200,000-500,000 triangles (estimated from mean degree ~10)
- L^(1) dimension: |E| * K = 5,000,000
- L^(1) nnz: roughly 4 * (N*K^2 + |T|*K^2) terms from both delta operators
- **VRAM estimate (CRITICAL):** The coboundary operators use dense K×K transport blocks (not diagonal). At K=100, each block has K²=10,000 complex128 entries.
  - δ₀: 2 blocks per edge × |E| × K² × 16 bytes. For |E|=50,000: ~16 GB.
  - δ₁: 3 blocks per triangle × |T| × K² × 16 bytes. For |T|=200,000: ~96 GB.
  - L^(1) assembled: |E|K × |E|K sparse. nnz ~ O(|E|²K²/N) — likely 50-100+ GB.
  - **Conclusion:** Full-scale H^1 at N=9877, K=100, eps=5.0 likely exceeds MI300X (192 GB) capacity when all matrices coexist. Two mitigation strategies:
    - **(a) Operator-free Lanczos:** Never assemble L^(1) explicitly. Instead, compute L^(1) v = (δ₀ δ₀†) v + (δ₁† δ₁) v via sequential sparse matvecs. This requires only δ₀ (~16 GB) and δ₁ (~96 GB) in memory simultaneously, plus Lanczos workspace (~5 GB). Total ~120 GB — fits on MI300X.
    - **(b) Reduced parameters:** Start with N=2000, K=50 (~1/100th the memory) to validate the implementation, then scale up. This is the recommended first step regardless.
  - COO→CSR conversion temporarily doubles the largest matrix's footprint. Build δ₀ and δ₁ sequentially, not simultaneously.

**First experiments MUST use reduced N (N=2000) to validate before scaling.** Full-scale feasibility depends on operator-free Lanczos (strategy a).

### 5.4 Theory Deliverables

**Provable results:**
1. L^(1) is PSD (sum of two PSD terms: δ₀ δ₀† and δ₁† δ₁)
2. δ₁ ∘ δ₀ = 0 if and only if the connection is flat on every triangle (U_{ij} U_{jk} = U_{ik}). For non-flat connections, the holonomy defect (δ₁ δ₀ s)_{ijk} = (U_{ij} U_{jk} − U_{ik}) s_k directly measures the curvature 2-form.
3. L^(1) is well-defined and PSD regardless of flatness. When the connection is non-flat, L^(1) measures holonomy curvature rather than true sheaf cohomology — this is the desired observable.
**Expected behavior (not yet proven):**
4. At σ = 0.5, transport maps are unitary by construction (exp of anti-Hermitian generator). Unitarity alone does not imply flatness (U_{ij} U_{jk} = U_{ik}), but the prime-encoded connection is expected to be "most flat" at σ = 0.5 for zeta zeros, concentrating small eigenvalues of L^(1) near zero. Off the critical line, non-unitary transport increases holonomy defect, pushing eigenvalues upward. This is the core conjecture tested by Tier 3.

**Conjecture 5 (H^1 Sensitivity):**
C^(1)(K) > C^(0)(K) for K >= 50, where C^(i) is the contrast ratio from the H^i Laplacian.

Test: Run H^0 and H^1 at K=50, 100 on the same zeros with matched parameters. Compare contrast ratios. If C^(1) > C^(0) at both K values, H^1 is the superior observable.

**Falsification of Conjecture 5:**
If H^1 shows no peak at sigma=0.5 while H^0 does, the arithmetic signal is in global sections, not curvature. This is a genuine negative result with theoretical significance — it would mean the gauge connection's holonomy is not the primary carrier of arithmetic information.

---

## 6. Sequencing and Dependencies

```
TIER 1: Engineering + Falsification
  1A. BaseSheafLaplacian extraction (refactor)
  1B. Torch test suite + regression tests
  1C. CI pipeline
  1D. reproduce_k20.py
  1E. FALSIFICATION.md
       |
       | Infrastructure trusted. Predictions committed.
       v
TIER 2: Scaling + Publication
  2A. Conjectures stated in paper
  2B. scaling_analysis.py upgraded
  2C. K=50/100/200/500 campaign on MI300X
  2D. Analyze against falsification criteria
  2E. Paper revision with verdicts
       |
       | H^0 data complete. Is there more?
       v
TIER 3: H^1 Cohomology
  3A. TriangleComplex (extends Tier 1's refactored edge discovery)
  3B. CoboundaryOperator (delta_0 extraction + delta_1)
  3C. H1SheafLaplacian (assembly + GPU eigensolver)
  3D. H^1 vs H^0 comparison experiments
  3E. Conjecture 5 verdict + paper update
```

**Key dependency:** Tier 1A's `BaseSheafLaplacian` refactor extracts edge discovery into a shared module. Tier 3A's `TriangleComplex` extends this same module with triangle enumeration. Building Tier 3 without Tier 1 would require duplicating the edge discovery code a fourth time.

**Each tier is independently publishable.** Stopping after Tier 2 yields a strong paper. Tier 3 elevates it.

---

## 7. Files Created or Modified

### Tier 1

| Action | File | Purpose |
|---|---|---|
| Create | `atft/topology/base_sheaf_laplacian.py` | Shared edge discovery, transport dispatch, base class |
| Modify | `atft/topology/sparse_sheaf_laplacian.py` | Inherit from base, remove duplicated code |
| Modify | `atft/topology/gpu_sheaf_laplacian.py` | Inherit from base, remove duplicated code |
| Modify | `atft/topology/torch_sheaf_laplacian.py` | Inherit from base, remove duplicated code |
| Create | `tests/test_torch_sheaf_laplacian.py` | Full torch backend test suite |
| Create | `tests/test_regression.py` | End-to-end golden reference + discrimination tests |
| Create | `.github/workflows/test.yml` | CI pipeline |
| Create | `scripts/reproduce_k20.py` | K=20 reproduction protocol |
| Create | `docs/FALSIFICATION.md` | Pre-committed falsification criteria |

### Tier 2

| Action | File | Purpose |
|---|---|---|
| Create | `scripts/scaling_analysis.py` | Publication-quality scaling fits + figures |
| Modify | `atft/experiments/phase3_distributed.py` | Add K=500 role, fine sigma grid |
| Modify | `docs/paper/ATFT_Riemann_Hypothesis.md` | Conjectures, results, verdicts |

### Tier 3

| Action | File | Purpose |
|---|---|---|
| Create | `atft/topology/triangle_complex.py` | Edge + triangle enumeration |
| Create | `atft/topology/coboundary_operators.py` | delta_0, delta_1 sparse matrix builders |
| Create | `atft/topology/h1_sheaf_laplacian.py` | 1-Laplacian assembly + eigensolver |
| Create | `tests/test_triangle_complex.py` | Triangle enumeration tests |
| Create | `tests/test_coboundary_operators.py` | delta_0, delta_1 correctness tests |
| Create | `tests/test_h1_sheaf_laplacian.py` | H^1 Laplacian tests |
| Create | `atft/experiments/phase4_h1_sweep.py` | H^1 sigma sweep experiment |

### Unchanged

All existing Phase 1-2 code, 212 existing tests, data files, and documentation not listed above remain untouched.
