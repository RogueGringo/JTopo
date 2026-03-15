# ATFT Riemann Hypothesis: Phase 1 Design Specification

**Date:** 2026-03-15
**Status:** Approved (Rev 2 — post spec review)
**Authors:** Blake Jones, Claude (Opus 4.6)

---

## 1. Overview

### 1.1 Goal

Apply the Adaptive Topological Field Theory (ATFT) framework to the Riemann Hypothesis by treating the non-trivial zeros of the Riemann zeta function as a spectral point cloud and comparing its topological evolution signature against the Gaussian Unitary Ensemble (GUE).

### 1.2 The Core Insight

The ATFT paper (Jones, Feb 2026) demonstrates that field equations emerge as "topological waypoints" — algebraic constraints on how persistent homology evolves across filtration scales. The Hilbert-Polya conjecture posits that zeta zeros are eigenvalues of an unknown self-adjoint operator. If true, their spacing statistics must match GUE random matrix theory (the Montgomery-Odlyzko connection).

This experiment translates that conjecture into the native language of ATFT: **the zeta zeros and GUE eigenvalues must produce statistically indistinguishable waypoint signatures**.

### 1.3 Project Trajectory

- **Phase 1 (this spec):** Computational benchmark — Zeta vs GUE vs Poisson topological comparison. Validates the ATFT pipeline and produces publishable evidence.
- **Phase 2 (future):** Sheaf-valued upgrade — attach u(N) Lie algebra fibers to the Rips complex. Attempt to identify the Hilbert-Polya operator as a sheaf Laplacian.
- **Phase 3 (future):** Rigorous reformulation — prove that RH is equivalent to a specific topological waypoint constraint.

### 1.4 Success Criteria

The experiment succeeds if:

| Test | Success | Failure |
|------|---------|---------|
| Zeta beta_0 within GUE 99% band | Curves stay inside envelope | Curves deviate from band |
| Zeta G_0 within GUE 99% band | Curves stay inside envelope | Curves deviate from band |
| Zeta Mahalanobis p-value | p > 0.05 | p < 0.05 |
| Poisson beta_0 within GUE band | Must FAIL (negative control) | If passes, pipeline is broken |
| Poisson Mahalanobis p-value | Must be p near 0 (negative control) | If p > 0.05, pipeline is broken |

**Note on L^2 distances:** The L^2 functional distances (Betti and Gini) are reported as diagnostic metrics only. Pass/fail decisions are based on the confidence band containment test (Prong 1) and the Mahalanobis p-value (Prong 2). The L^2 values provide interpretability but have no explicit threshold.

**Note on multiple testing:** The five tests above are derived from the same underlying data and are highly correlated. A formal Bonferroni correction is not applied, but results should be interpreted holistically — the experiment succeeds only if all five conditions hold simultaneously.

---

## 2. Mathematical Foundation

### 2.1 Configuration Space: Zeros as Spectrum

The imaginary parts of the non-trivial zeta zeros, gamma_n, form a 1D point cloud in R^1. This mirrors the ATFT paper's treatment of the quantum harmonic oscillator spectrum (Section 6.3), where energy levels are treated as a finite metric space.

### 2.2 The GUE Connection

Freeman Dyson and Hugh Montgomery discovered that the pair correlation of zeta zeros matches the GUE eigenvalue spacing distribution. GUE exhibits "level repulsion" — eigenvalues actively avoid clustering. This produces a rigid, highly specific gap distribution that differs qualitatively from uncorrelated (Poisson) spacings.

### 2.3 Spectral Unfolding

Raw spectra have non-uniform average density. Unfolding normalizes to mean gap = 1, isolating the fluctuations (which encode universality class) from the trend (which is system-specific).

- **Zeta zeros:** N_smooth(T) = (T/(2*pi)) * ln(T/(2*pi*e)) + 7/8. Unfolded zeros: x_i = N_smooth(gamma_i).
- **GUE eigenvalues:** Rank-based unfolding via empirical CDF of each individual matrix: x_i = rank(lambda_i) / N. This is robust to finite-size effects and does not require knowing the exact normalization of the semicircle law.
- **Poisson:** Cumulative sum of Exp(1) gaps; trivially unfolded. Uses the identity feature map (bypasses SpectralUnfolding entirely).

### 2.4 The H_0 Analytical Shortcut

**Filtration convention:** We use the **diameter convention** throughout. In the Vietoris-Rips complex, an edge is added between points x_i and x_j when their distance d(x_i, x_j) <= eps. For a 1D point cloud, two adjacent points x_i and x_{i+1} merge into the same connected component at eps = g_i = x_{i+1} - x_i (the full gap, not half the gap).

For a 1D point cloud, H_0 persistent homology is analytically computable in O(N log N):

1. Sort the unfolded spectrum: x_1 < x_2 < ... < x_N
2. Compute gaps: g_i = x_{i+1} - x_i
3. H_0 persistence diagram:
   - N-1 finite features: {(0, g_i) : i = 1,...,N-1}
   - 1 immortal feature: (0, infinity) — the final connected component that never dies

No Rips complex construction or Ripser computation needed. Each finite feature's death scale is exactly the gap at which two adjacent components merge.

### 2.5 Evolution Curves

From the persistence diagram, three evolution functions are computed on a discrete epsilon grid:

- **Betti curve** beta_0(eps) = count of features alive at scale eps (where alive means birth <= eps < death). For H_0, starts at N and decreases monotonically to 1 (not 0 — the immortal feature persists at all finite scales). The immortal feature is included in the count.
- **Gini trajectory** G_0(eps) = Gini coefficient of the **full persistence lifetimes** (death - birth) of features alive at eps. Uses the full eventual lifetime, not the partial lifetime elapsed so far. Measures hierarchical organization: G=0 (uniform/disordered) to G->1 (single dominant feature).
- **Persistence curve** P_0(eps) = sum of lifetimes of features alive at eps.

**Gini edge cases:**
- Empty feature set (n=0): G = 0.0
- Single feature (n=1): G = 0.0 (by convention — a single element has no inequality)
- All lifetimes equal: G = 0.0
- All lifetimes zero: G = 0.0

### 2.6 Waypoint Signature (Adapted for H_0)

W_0(C) = (eps*, {eps_{w,i}}_{i=1}^K, {delta_0(eps_{w,i})}_{i=1}^K, G_0(eps*), dG_0/deps|_{eps*}) in R^(2K+3)

Where:
- eps* = onset scale (first eps where topology changes)
- eps_{w,i} = top-K waypoint scales (the K largest gaps in the sorted gap sequence, sorted by epsilon position)
- delta_0(eps_{w,i}) = topological derivative values at waypoints (the magnitude of the Betti number drop at that gap scale)
- G_0(eps*) = Gini coefficient at onset
- dG_0/deps|_{eps*} = Gini trajectory slope at onset

K is fixed at 2 for Phase 1, yielding vectors in R^7.

**Waypoint extraction method:** Rather than computing numerical derivatives of the Betti curve (which is a step function, making np.gradient produce noisy spikes), waypoints are extracted directly from the gap sequence: the K largest gaps correspond to the K most significant topological events (the largest drops in beta_0). This avoids numerical differentiation artifacts entirely.

### 2.7 Statistical Validation: Two-Pronged Criterion

**Prong 1 (Macroscopic — functional envelope):** Generate M=1,000 GUE matrices. Compute Betti and Gini curves for each. Build pointwise 99% confidence bands using empirical percentiles (not Gaussian assumption — the Betti curve is integer-valued and the Gini curve is bounded in [0,1], so neither follows a normal distribution at the extremes). Compute L^2 functional distance between zeta curves and ensemble mean. Check that zeta curves stay entirely within the 99% band across the full adaptive filtration.

**Prong 2 (Formal — Mahalanobis waypoint matching):** Extract R^7 waypoint signature for all 1,000 GUE matrices. Compute the multivariate mean and regularized covariance matrix (Tikhonov regularization: reg = 1e-6 * trace(cov) / dim). Extract the same R^7 signature for the zeta zeros. Compute Mahalanobis distance D_M. Report p-value under chi-square distribution with 7 degrees of freedom.

The Mahalanobis distance was chosen over Euclidean distance because it accounts for covariance between waypoint components (e.g., onset scale and first waypoint are likely correlated). Bottleneck distance was rejected because its L-infinity nature makes it hypersensitive to single outlier gaps in finite samples, penalizing the entire comparison based on microscopic fluctuations rather than the global level-repulsion signature.

---

## 3. Software Architecture

### 3.1 Design Philosophy

The code mirrors the ATFT paper's Dimensional Pathway. Each mathematical stage maps to a software module. Modules communicate through shared immutable data types and depend only on the contract layer, never on each other directly.

### 3.2 Pipeline

```
ConfigurationSource -> FeatureMap -> TopologicalOperator -> EvolutionCurves
                                                                  |
                                                                  v
                                    StatisticalValidator <- WaypointExtractor
```

### 3.3 Project Layout

```
atft/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py              # PointCloud, PersistenceDiagram, EvolutionCurve,
│   │                         #   WaypointSignature, ValidationResult
│   └── protocols.py          # ConfigurationSource, FeatureMap,
│                              #   TopologicalOperator, Cacheable
├── sources/
│   ├── __init__.py
│   ├── zeta_zeros.py          # Load/parse Odlyzko dataset
│   ├── gue.py                 # Generate GUE random matrices (GPU via PyTorch)
│   └── poisson.py             # Generate Poisson point processes
├── feature_maps/
│   ├── __init__.py
│   ├── identity.py            # Pass-through for pre-processed data
│   └── spectral_unfolding.py  # Normalize spacings to mean gap = 1
├── topology/
│   ├── __init__.py
│   ├── analytical_h0.py       # Exact H_0 for 1D point clouds (O(N log N))
│   ├── ripser_ph.py           # GPU-accelerated PH via giotto-tda (Phase 2)
│   └── sheaf_ph.py            # Sheaf-valued PH stub (Phase 2)
├── analysis/
│   ├── __init__.py
│   ├── evolution_curves.py    # Betti, Gini, Persistence curve computation
│   ├── waypoint_extractor.py  # Top-K waypoint extraction (gap-based)
│   └── statistical_tests.py   # Mahalanobis, L2 envelope, confidence bands
├── visualization/
│   ├── __init__.py
│   └── plots.py               # Three-panel publication figure
├── io/
│   ├── __init__.py
│   └── cache.py               # HDF5 serialization for intermediate results
└── experiments/
    ├── __init__.py
    └── phase1_benchmark.py     # Orchestrator: Zeta vs GUE vs Poisson
```

### 3.4 Data Flow

```
ZetaZerosSource ──► PointCloud (float64) ──► SpectralUnfolding ──► AnalyticalH0
GUESource ─────────► PointCloud (float64)──► SpectralUnfolding ──► PersistenceDiagram
PoissonSource ─────► PointCloud (float64)──► IdentityMap ────────► EvolutionCurveSet
                                                                   ──► WaypointSignature
                                                                        ──► ValidationResult
```

**Precision note:** ZetaZerosSource produces float64 PointClouds (exception to the general float32 rule). Zeta zeros gamma_n range from ~14.13 to ~9877.78 for the first 10,000 zeros. Computing gaps as differences of float32 values near 10,000 would lose precision on O(1)-sized gaps. All sources in Phase 1 produce float64 since the analytical H_0 shortcut runs on CPU (no VRAM constraint). The float32 convention applies when GPU memory matters (Phase 2 with Ripser on large point clouds).

---

## 4. Core Types

### 4.1 PointCloud

```python
@dataclass(frozen=True)
class PointCloud:
    points: NDArray[np.float64]          # shape (N, d) — float64 for Phase 1
    metadata: dict = field(default_factory=dict)
```

Immutable. N points in R^d.

### 4.2 PointCloudBatch

```python
@dataclass(frozen=True)
class PointCloudBatch:
    clouds: list[PointCloud]
```

Supports ragged batches (different N per cloud, same d). Enables GPU batch processing for the GUE ensemble without Python-loop overhead.

### 4.3 PersistenceDiagram

```python
@dataclass(frozen=True)
class PersistenceDiagram:
    diagrams: dict[int, NDArray[np.float64]]   # degree -> (n_features, 2)
    metadata: dict = field(default_factory=dict)
```

float64 for birth/death precision. The immortal H_0 feature is stored with death = np.inf. Implements Cacheable protocol (HDF5 serialization).

The `lifetimes()` method returns `np.array([], dtype=np.float64)` for empty diagrams (explicit dtype to prevent downstream concatenation errors).

### 4.4 EvolutionCurve and EvolutionCurveSet

```python
@dataclass(frozen=True)
class EvolutionCurve:
    epsilon_grid: NDArray[np.float64]    # shape (n_steps,)
    values: NDArray[np.float64]          # shape (n_steps,)
    curve_type: CurveType                # BETTI, GINI, or PERSISTENCE
    degree: int                          # homological degree k

@dataclass(frozen=True)
class EvolutionCurveSet:
    betti: dict[int, EvolutionCurve]
    gini: dict[int, EvolutionCurve]
    persistence: dict[int, EvolutionCurve]
```

### 4.5 WaypointSignature

```python
@dataclass(frozen=True)
class WaypointSignature:
    onset_scale: float
    waypoint_scales: NDArray[np.float64]         # exactly K values
    topo_derivatives: NDArray[np.float64]         # exactly K values
    gini_at_onset: float
    gini_derivative_at_onset: float

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to R^(2K+3) for Mahalanobis computation."""
```

**Critical constraint:** K is fixed across the ensemble (Top-K extraction). The WaypointExtractor always returns exactly K waypoints. If fewer than K significant gaps exist, the remaining slots are zero-padded. This guarantees all signature vectors live in the same R^(2K+3) for covariance computation.

### 4.6 ValidationResult

```python
@dataclass(frozen=True)
class ValidationResult:
    mahalanobis_distance: float
    p_value: float
    l2_distance_betti: float
    l2_distance_gini: float
    within_confidence_band: bool
    ensemble_size: int
    metadata: dict = field(default_factory=dict)
```

---

## 5. Protocols

All pipeline stages are defined by Python Protocols (structural subtyping). Phase 2 modules need only implement the same interface — no inheritance required.

```python
class ConfigurationSource(Protocol):
    def generate(self, n_points: int, **kwargs) -> PointCloud: ...
    def generate_batch(self, n_points: int, batch_size: int, **kwargs) -> PointCloudBatch: ...

class FeatureMap(Protocol):
    def transform(self, cloud: PointCloud) -> PointCloud: ...
    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch: ...

class TopologicalOperator(Protocol):
    def compute(self, cloud: PointCloud, max_degree: int = 0,
                epsilon_max: float | None = None) -> PersistenceDiagram: ...
    def compute_batch(self, batch: PointCloudBatch, max_degree: int = 0,
                      epsilon_max: float | None = None) -> list[PersistenceDiagram]: ...

class Cacheable(Protocol):
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Cacheable": ...
```

---

## 6. Phase 1 Implementations

### 6.1 Configuration Sources

**ZetaZerosSource:** Parses Odlyzko's plain-text dataset of zeta zero imaginary parts. Loads first N values as float64, reshapes to (N, 1) PointCloud.

**GUESource:** Generates N x N random Hermitian matrices. Entry distribution: each off-diagonal entry A_{ij} = (X + iY) / sqrt(2) where X, Y are i.i.d. N(0, 1); diagonal entries A_{ii} ~ N(0, 1) (real). The Hermitian matrix is H = (A + A^dagger) / (2 * sqrt(N)), yielding a semicircle law with support [-1, 1]. Eigendecomposition via torch.linalg.eigh on CUDA for batch processing. Returns eigenvalues as (N, 1) PointCloud (float64).

**PoissonSource:** Generates N i.i.d. Exp(1) gaps, cumulative sums to positions. No correlations, no level repulsion. Serves as negative control. Bypasses SpectralUnfolding — uses identity feature map directly.

### 6.2 Spectral Unfolding (Feature Map)

Normalizes raw spectra to mean gap = 1:

- **Zeta:** Uses analytic smooth staircase N_smooth(T) = (T/(2*pi)) * ln(T/(2*pi*e)) + 7/8
- **GUE:** Rank-based unfolding via empirical CDF of each individual matrix: x_i = rank(lambda_i) / N. Robust to finite-size effects and tail behavior beyond the semicircle support.
- **Poisson:** Identity transform (already unfolded by construction). Uses `identity.py` feature map, not `spectral_unfolding.py`.

### 6.3 Analytical H_0 (Topological Operator)

For 1D point clouds, H_0 persistence is exactly the sorted gap sequence (diameter convention):

1. Sort points: x_1 < x_2 < ... < x_N — O(N log N)
2. Gaps: g_i = x_{i+1} - x_i — O(N)
3. Persistence diagram:
   - N-1 finite features: {(0, g_i)}
   - 1 immortal feature: (0, infinity)

This bypass reduces computation from Ripser's sparse matrix reduction to a simple array operation. The 1,000-matrix GUE ensemble processes in seconds on CPU.

### 6.4 Evolution Curve Computer

Samples Betti, Gini, and Persistence curves on a uniform epsilon grid:

- **Epsilon grid range:** eps_min = 0, eps_max = 1.1 * max(g_i) across all finite features. This ensures the grid spans the full persistence range with a 10% margin.
- **Grid resolution:** n_steps = 1,000 (configurable).

Formulas:
- beta_0(eps) = count of (birth, death) pairs where birth <= eps < death. Includes the immortal feature (death = infinity), so beta_0 converges to 1 as eps -> max(g_i).
- G_0(eps) = Gini coefficient of full persistence lifetimes (death - birth) of features alive at eps. For the immortal feature, its lifetime is treated as eps_max (capped at the grid boundary) to avoid infinity contaminating the Gini computation.
- P_0(eps) = sum of lifetimes of features alive at eps. Same capping for the immortal feature.

**Gini coefficient formula (1-indexed):** Given sorted values v_1 <= v_2 <= ... <= v_n:

```
G = (2 * sum_{i=1}^{n} i * v_i) / (n * sum_{i=1}^{n} v_i) - (n + 1) / n
```

**NumPy implementation (0-indexed):**

```python
sorted_v = np.sort(values)
n = len(sorted_v)
index = np.arange(1, n + 1, dtype=np.float64)  # 1-indexed
G = (2 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1) / n
```

**Edge cases:** n=0 -> G=0.0; n=1 -> G=0.0; sum(v)=0 -> G=0.0.

### 6.5 Waypoint Extractor (Gap-Based, Top-K)

Rather than computing numerical derivatives of the step-function Betti curve (which produces noisy spikes from np.gradient), waypoints are extracted directly from the gap sequence:

1. From the persistence diagram, collect all finite lifetimes (deaths): these are the gaps g_i.
2. Sort gaps by magnitude (descending).
3. Take the top K gaps as the waypoint scales. These correspond to the K most significant topological merging events.
4. If fewer than K gaps exist, zero-pad the remaining slots.
5. Sort selected waypoints by epsilon position (ascending) for the signature vector.

The topological derivative at each waypoint is the magnitude of the Betti number drop at that scale — always -1 for a single merging event, but for clustered gaps at similar scales, the effective derivative (slope of beta_0 in that region) may be steeper.

**Onset scale logic (branches on degree):**
- H_0 (degree=0): eps* = min(g_i) — the smallest gap, where the first merging occurs.
- H_k (degree>0): first eps where beta_k rises above 0.

### 6.6 Statistical Validator

**Prong 1 — Functional Envelope:**
- Interpolate all ensemble curves to a common epsilon grid (the target's grid)
- Compute pointwise percentile bands (empirical, not Gaussian)
- L^2 distance = sqrt(sum((target - mean)^2 * d_eps)), reported as diagnostic only
- within_band = all(lower <= target <= upper) at every grid point

**Prong 2 — Mahalanobis Distance:**
- Stack ensemble signature vectors: (M, 2K+3) matrix
- Mean vector: mu = mean over M rows
- Regularized covariance: C_reg = C + (1e-6 * trace(C) / dim) * I
- D_M = sqrt((w - mu)^T * C_reg^{-1} * (w - mu))
- p-value = 1 - chi2.cdf(D_M^2, df=2K+3)
- Diagnostic: log the condition number of C_reg; warn if cond(C_reg) > 1e10

### 6.7 Caching

Intermediate results (PersistenceDiagram, EvolutionCurveSet) are serialized to HDF5 via h5py:

```
cache/
├── zeta_zeros/
│   ├── persistence_N10000.h5
│   └── curves_N10000.h5
├── gue_ensemble/
│   ├── persistence_M1000_N10000.h5
│   └── curves_M1000_N10000.h5
└── poisson/
    ├── persistence_M1000_N10000.h5
    └── curves_M1000_N10000.h5
```

---

## 7. Experiment Orchestration

### 7.1 Configuration

```python
Phase1Config:
    n_points: 10,000
    ensemble_size: 1,000
    k_waypoints: 2
    n_epsilon_steps: 1,000
    confidence_level: 0.99
    zeta_data_path: Path("data/odlyzko_zeros.txt")
    cache_dir: Path("cache/")
    device: "cuda"
    seed: 42              # For reproducibility of GUE and Poisson generation
```

**Reproducibility:** GUE and Poisson sources use numpy.random.Generator with SeedSequence for deterministic, parallel-safe random number generation. The seed is recorded in the experiment metadata.

### 7.2 Execution Flow

1. Generate/load data: zeta zeros (Odlyzko), 1,000 GUE matrices, 1,000 Poisson samples
2. Unfold spectra: zeta via analytic staircase, GUE via rank-based empirical CDF, Poisson via identity map
3. Compute H_0 persistence diagrams (analytical shortcut)
4. Cache persistence diagrams to HDF5
5. Compute evolution curves (Betti, Gini, Persistence)
6. Extract Top-2 waypoint signatures for all configurations
7. Fit StatisticalValidator on GUE ensemble
8. Validate zeta zeros against GUE (should PASS)
9. Validate Poisson against GUE (should FAIL — negative control)
10. Generate three-panel publication figure

### 7.3 Output: Three-Panel Figure

- **Panel A:** Betti curves beta_0(eps) — GUE 99% band (grey fill), Zeta (solid blue), Poisson (dashed red)
- **Panel B:** Gini trajectories G_0(eps) — same color scheme
- **Panel C:** PCA projection of R^7 waypoint space — GUE ensemble (grey dots), Zeta (blue star), Poisson (red cross), with 95% Mahalanobis confidence ellipse overlaid. PCA is fit on the GUE ensemble only; zeta and Poisson are projected using GUE principal components. First two principal components used for the 2D projection.
- **Footer:** Mahalanobis distances and p-values for both zeta and Poisson

---

## 8. Dependencies

```
numpy >= 1.24
scipy >= 1.11
h5py >= 3.9
matplotlib >= 3.7
torch >= 2.1          # GUE batch eigendecomposition on CUDA
giotto-tda >= 0.6     # Ripser backend (Phase 2 / H_1)
```

Phase 1 core computation has no hard dependency on GPU or Ripser — the analytical H_0 shortcut runs on CPU. GPU is used for batch GUE eigendecomposition (torch.linalg.eigh).

---

## 9. Phase 2 Extension Points

The architecture is designed so that Phase 2 (sheaf-valued persistent homology) requires only:

1. A new `SheafSpecification` type in `core/types.py` defining stalks (u(N) fibers) and restriction maps
2. A new `SheafPH` class in `topology/sheaf_ph.py` implementing the `TopologicalOperator` protocol
3. A new feature map in `feature_maps/` if needed for the sheaf construction

**Assumption:** Phase 2 sheaf-valued persistence diagrams produce scalar-valued Betti numbers (dimensions of sheaf cohomology groups), so the evolution curve and waypoint extraction layers remain unchanged. If sheaf-valued PH produces vector-valued or matrix-valued invariants, the EvolutionCurve type and WaypointExtractor will need extension. This assumption should be validated when the sheaf construction is designed.

---

## 10. Open Questions

1. **Optimal K for waypoint extraction.** K=2 is a reasonable starting point, but the true number of significant topological waypoints may differ. Consider running a sensitivity analysis on K = 1, 2, 3, 4.

2. **H_1 analysis.** The analytical H_0 shortcut is specific to 1D point clouds. When moving to higher-dimensional feature maps or H_1 computation, we need Ripser. Does H_1 carry additional discriminative information for the zeta/GUE comparison?

3. **Scaling behavior.** How do the waypoint signatures evolve as N increases from 1,000 to 100,000? Convergence of the signatures with N would strengthen the claim. Consider adding N as a sweep parameter.

4. **The sheaf construction.** What are the natural restriction maps for u(N) fibers on the Rips complex of zeta zeros? The Phase 1 empirical data should inform this choice.

5. **Connection to known results.** How does the Betti curve onset scale relate to the known pair correlation function R_2(r)? Can we derive a closed-form prediction for the GUE Betti curve from random matrix theory?

6. **Statistical power.** What is the smallest deviation from GUE that this pipeline could detect at the given N=10,000 and M=1,000? A power analysis would quantify the sensitivity of the Mahalanobis test.

---

## Appendix A: Revision History

### Rev 2 (2026-03-15) — Post Spec Review

Addressed findings from automated spec review:

**Critical fixes:**
- C1: Explicitly stated diameter convention for Rips filtration (death at g_i, not g_i/2)
- C2: Clarified beta_0 converges to 1 (not 0) due to immortal feature; defined Gini behavior for n=0, n=1; specified immortal feature lifetime capping at eps_max
- C3: Provided explicit 1-indexed Gini formula and NumPy equivalent; stated lifetimes are full persistence (death - birth), not partial

**Important fixes:**
- I1: Specified exact GUE matrix entry distribution (variance of real/imaginary parts) and resulting semicircle support [-1, 1]
- I2: Committed to rank-based empirical CDF for GUE unfolding
- I3: Specified epsilon grid range as [0, 1.1 * max(g_i)]
- I4: Clarified L^2 distance is diagnostic only, no pass/fail threshold
- I5: Explicitly stated Poisson uses identity feature map, bypasses SpectralUnfolding
- I6: Replaced numerical derivative waypoint extraction with gap-based approach (top-K largest gaps)
- I7: Specified PCA fit on GUE ensemble only, first two components, with Mahalanobis ellipse overlay

**Additional improvements:**
- Added seed parameter to Phase1Config for reproducibility
- Changed PointCloud to float64 for Phase 1 (precision on zeta zero gaps)
- Added condition number diagnostic for covariance matrix
- Added statistical power as open question
- Added Phase 2 sheaf assumption caveat
- Fixed parenthesization in zeta unfolding formula
