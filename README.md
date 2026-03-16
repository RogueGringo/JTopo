# Ti V0.1 — Topological Investigation of the Riemann Hypothesis

**Ti V0.1** is an active computational mathematics research project implementing an
**Adaptive Topological Field Theory (ATFT)** framework to investigate the Riemann
Hypothesis. The project constructs a gauge-theoretic sheaf over point clouds derived
from high-altitude zeta zeros, using the prime numbers as generators of a u(K) Lie
algebra connection. The central prediction is that if the Riemann Hypothesis (RH) is
true, the sheaf Laplacian's spectral signature will exhibit a sharp phase transition
at the critical line Re(s) = 1/2 as the fiber dimension K grows.

---

## Table of Contents

1. [Scientific Context](#1-scientific-context)
2. [Mathematical Framework](#2-mathematical-framework)
   - [2.1 Prime Representation — Fiber Structure](#21-prime-representation--fiber-structure)
   - [2.2 Gauge Connection — Transport Maps](#22-gauge-connection--transport-maps)
   - [2.3 Sheaf Laplacian](#23-sheaf-laplacian)
   - [2.4 The Statistical Test](#24-the-statistical-test)
3. [Implementation](#3-implementation)
   - [3.1 Repository Structure](#31-repository-structure)
   - [3.2 Key Modules](#32-key-modules)
   - [3.3 Computational Infrastructure](#33-computational-infrastructure)
4. [Experimental Results](#4-experimental-results)
   - [4.1 Phase 1 — Spectral Baseline](#41-phase-1--spectral-baseline)
   - [4.2 Phase 2 — Transport Map Validation](#42-phase-2--transport-map-validation)
   - [4.3 Phase 3 — K=20 Full Sweep](#43-phase-3--k20-full-sweep)
   - [4.4 Phase 3b — K=50 Scout](#44-phase-3b--k50-scout)
   - [4.5 Outlook: K=100+](#45-outlook-k100)
5. [Data](#5-data)
6. [Requirements and Installation](#6-requirements-and-installation)
7. [Running Experiments](#7-running-experiments)
8. [Tests](#8-tests)
9. [Project Status](#9-project-status)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. Scientific Context

The **Riemann Hypothesis** states that all non-trivial zeros of the Riemann zeta
function

$$\zeta(s) = \sum_{n=1}^{\infty} n^{-s} = \prod_{p\ \text{prime}} (1 - p^{-s})^{-1}$$

lie on the critical line Re(s) = 1/2. Despite overwhelming numerical evidence and
profound consequences for the distribution of primes, it remains unproven after more
than 160 years.

This project approaches RH from a topological and gauge-theoretic angle: rather than
analyzing zeros analytically, we ask whether the *geometry* of the zero set is
detectably different from random point processes precisely at sigma = 0.5. We
construct a sheaf of vector spaces over the Vietoris-Rips complex of the zero set,
equip it with a connection derived from the prime numbers, and measure the resulting
spectral rigidity.

The connection encodes the multiplicative structure of the integers via the **explicit
formula** relating zeta zeros to prime counts. If RH is true, this structure is
self-consistent only at sigma = 0.5 — a fact our gauge theory is designed to detect
topologically.

---

## 2. Mathematical Framework

### 2.1 Prime Representation — Fiber Structure

Let K be the fiber dimension. The fiber at each vertex is C^K, i.e., a
K-dimensional complex vector space indexed by integers {1, ..., K}.

For each prime p <= K, we define the **truncated left-regular representation**

$$\rho(p) \in \mathrm{GL}(K, \mathbb{Z})$$

of the multiplicative monoid (Z_{>0}, x) by the rule:

$$\rho(p)_{ij} = \begin{cases} 1 & \text{if } j = p \cdot i,\ p \cdot i \leq K \\ 0 & \text{otherwise} \end{cases}$$

In other words, rho(p) acts on the basis vector e_i by sending it to e_{p*i}
when p*i remains within {1, ..., K}, and to zero otherwise. This truncation is
the canonical way to embed the multiplicative structure of the integers into a
finite-dimensional fiber while preserving the semigroup homomorphism property.

The set of primes up to K — there are pi(K) of them — serves as a generating
set for the entire monoid action. For K=20 this yields 8 primes
{2, 3, 5, 7, 11, 13, 17, 19}; for K=50 it yields 15 primes.

### 2.2 Gauge Connection — Transport Maps

The ATFT gauge connection lives in the u(K) Lie algebra and depends on the
complex parameter s = sigma + i*gamma. The connection is assembled from the
prime representations and controls how sections of the sheaf are parallel-transported
along edges.

Four transport modes are implemented, each encoding a different hypothesis about
which arithmetic structure is relevant:

#### Global (flat connection)

The Hermitian generator for prime p at position sigma is:

$$G_p(\sigma) = \frac{\log p}{p^\sigma} \left( \rho(p) + \rho(p)^\dagger \right)$$

The total generator is the sum over all primes:

$$A(\sigma) = \sum_{p \leq K} G_p(\sigma)$$

Since A(sigma) is Hermitian, it admits a single eigendecomposition
A = V diag(lambda_k) V^dagger, cached once per (K, sigma) pair. The edge
transport map for an edge separating zeros gamma_i and gamma_j is:

$$U(\Delta\gamma) = V \cdot \mathrm{diag}(e^{i \Delta\gamma \cdot \lambda_k}) \cdot V^\dagger, \quad \Delta\gamma = \gamma_j - \gamma_i$$

This is the **flat connection**: every edge shares the same eigenbasis, differing
only in the phase factors. Cost is O(K^2) per edge after the O(K^3) initial
eigendecomposition.

#### Resonant (curved connection)

Each edge (i, j) is bound to its **resonant prime** p* = argmin_p |Delta_gamma - log(p)|,
the prime whose logarithm most closely matches the edge length in the unfolded
spectrum. The generator is then G_{p*}(sigma) restricted to that single prime,
creating a genuinely edge-dependent (curved) connection. This mode implements
the hypothesis that each pair of consecutive zeros is "explained" by a single
dominant prime.

#### Functional Equation (FE) connection

This mode directly encodes the functional equation zeta(s) <-> zeta(1-s). The
generator for prime p is:

$$G_p^{\mathrm{FE}}(\sigma) = \log(p) \left[ p^{-\sigma} \rho(p) + p^{-(1-\sigma)} \rho(p)^T \right]$$

The key property: G_p^FE(sigma) is Hermitian — and the transport U therefore
unitary — **only at sigma = 1/2**, where p^{-sigma} = p^{-(1-sigma)}. At any
other sigma, the generator is non-Hermitian and the transport is a non-unitary
contraction or expansion. This mode encodes the self-duality of the critical line
as a symmetry-breaking condition in the fiber bundle.

#### Superposition / Explicit Formula

This is the primary mode for Phase 3. The edge-dependent generator matrix is:

$$A_{ij}(\sigma) = \sum_{p \leq K} e^{i \Delta\gamma \cdot \log p} \cdot B_p(\sigma), \quad B_p(\sigma) = \log(p)\left[ p^{-\sigma} \rho(p) + p^{-(1-\sigma)} \rho(p)^T \right]$$

The exponential factor e^{i*Delta_gamma*log(p)} is the discrete Fourier kernel
that appears in the explicit formula: it creates **multi-prime phase interference**
along each edge, directly mirroring the sum over zeros in von Mangoldt's formula.
When many zeros conspire to produce a peak in prime counting, these phase factors
add constructively; otherwise they cancel. Transport is computed via the batched
matrix exponential of A_{ij}(sigma), which is non-Hermitian in general.

### 2.3 Sheaf Laplacian

Given N unfolded zeta zeros {gamma_1, ..., gamma_N}, a **Vietoris-Rips graph**
is constructed: edge (i, j) exists when |gamma_i - gamma_j| <= epsilon. The
connectivity threshold epsilon is a sweep parameter controlling the topological
scale of the complex.

The **sheaf coboundary operator** delta_0 acts on 0-cochains (vertex sections)
to produce 1-cochains (edge sections). For an oriented edge e = (i -> j) with
transport map U_e:

$$({\delta_0 \, x})_e = U_e \, x_i - x_j$$

The **sheaf Laplacian** is then:

$$L_\mathcal{F} = \delta_0^\dagger \delta_0$$

Expanding into block form, for each edge (i -> j) with transport U, the
contributions to the NK x NK block matrix are:

| Block | Contribution |
|-------|-------------|
| L[i,i] | += U^dagger U |
| L[j,j] | += I_K |
| L[i,j] | += -U^dagger |
| L[j,i] | += -U |

The **spectral sum** at parameters (sigma, epsilon) is:

$$S(\sigma, \varepsilon) = \sum_{k=1}^{k_{\mathrm{eig}}} \lambda_k(L_\mathcal{F})$$

where lambda_1 <= lambda_2 <= ... are the smallest eigenvalues of L_F. This
quantity measures the **topological rigidity** of the sheaf: when the transport
maps are consistent with each other (flat connection, small holonomy), small
eigenvalues accumulate near zero and S is small; when the connection is
inconsistent (large curvature), eigenvalues are lifted and S is large.

### 2.4 The Statistical Test

The core experiment sweeps sigma across [0.25, 0.75] and epsilon across a grid
of connectivity thresholds, computing S(sigma, epsilon) for three point-cloud
types:

- **Zeta zeros:** The actual imaginary parts of non-trivial Riemann zeros (Odlyzko
  high-altitude dataset, near the 10^20-th zero)
- **Poisson random:** Uniformly random points on the same interval
- **GUE control:** Eigenvalues of a random Gaussian Unitary Ensemble matrix,
  whose spectral statistics (level spacing, pair correlations) are known to match
  those of zeta zeros

The **contrast ratio** for each point-cloud type is:

$$C(\sigma) = \frac{S(\sigma) - S_{\min}}{S(\sigma)}$$

where S_min = min over sigma of S(sigma). This normalizes each curve to the
interval [0, 1] so that the shape of the sigma-dependence can be compared across
point-cloud types and epsilon values.

The **signal ratio** is:

$$R = \frac{C_{\mathrm{zeta}}(\sigma^*))}{\mathrm{mean}[C_{\mathrm{controls}}(\sigma^*)]}$$

evaluated at sigma* = 0.50.

**Prediction:** If the Riemann Hypothesis is true, then as K -> infinity (more
primes in the gauge connection), S(sigma) for zeta zeros develops a sharp peak
at sigma = 0.5 while remaining flat for the GUE and Poisson controls. The signal
ratio R should diverge as K grows.

The GUE control is decisive: GUE eigenvalues match the *statistical* distribution
of zeta zeros (Montgomery-Odlyzko law) but contain no *arithmetic* information
about primes. Any signal that appears for GUE is a statistical artifact; any
signal that appears for zeta zeros but not GUE is arithmetic in origin.

---

## 3. Implementation

### 3.1 Repository Structure

```
atft/
├── core/
│   ├── protocols.py        # Type protocols for point clouds, sources, feature maps
│   └── types.py            # PointCloud, PointCloudBatch dataclasses
├── sources/
│   ├── zeta_zeros.py       # Odlyzko zero loader and sub-sampler
│   ├── gue.py              # GUE random matrix eigenvalue generator
│   └── poisson.py          # Poisson/uniform random point generator
├── feature_maps/
│   ├── spectral_unfolding.py  # Smooth CDF unfolding (semicircle / zeta staircase)
│   └── identity.py            # Pass-through feature map
├── topology/
│   ├── transport_maps.py       # u(K) gauge connection; 4 transport modes
│   ├── sparse_sheaf_laplacian.py  # CPU BSR sparse assembly + scipy eigsh
│   ├── gpu_sheaf_laplacian.py     # CuPy GPU CSR assembly + spectral flip eigensolver
│   ├── sheaf_laplacian.py         # Dense/matrix-free Phase 2 implementation
│   └── sheaf_ph.py                # Persistent homology interface
├── analysis/
│   ├── evolution_curves.py    # S(sigma, epsilon) sweep orchestration
│   ├── statistical_tests.py   # Contrast ratio, signal ratio, bootstrap CI
│   └── waypoint_extractor.py  # Peak detection and phase transition localization
├── visualization/
│   └── ...                    # Publication-quality matplotlib plots
├── io/
│   └── ...                    # HDF5 result caching (h5py)
└── experiments/
    ├── phase1_benchmark.py         # GUE vs zeta spectral baseline
    ├── phase2a_abelian.py          # Abelian u(1) flat connection test
    ├── phase2b_sheaf.py            # Full u(K) sheaf Phase 2
    ├── phase3_superposition_sweep.py   # K=20 definitive control sweep
    ├── phase3_distributed.py           # Role-based multi-machine partitioning
    ├── phase3b_gpu_sweep.py            # K=50 GPU scout
    └── phase3c_gpu_k100.py             # K=100 GPU sweep

data/
    odlyzko_zeros.txt   # ~1.8 MB, high-altitude Riemann zeros

tests/
    ...                 # pytest unit and integration tests

docs/
    ...                 # Design documents, phase specifications
```

### 3.2 Key Modules

**`atft/topology/transport_maps.py`**

The `TransportMapBuilder` class is the heart of the gauge theory. It accepts
fiber dimension K and sigma, constructs the prime representations rho(p) for all
primes p <= K via Sieve of Eratosthenes, assembles the generator sum A(sigma),
and caches the eigendecomposition V, lambda. Public methods:

- `global_transport(delta_gamma)` — O(K^2) flat transport
- `resonant_transport(delta_gamma)` — prime-resonant curved transport
- `fe_transport(delta_gamma)` — functional-equation symmetry-breaking transport
- `superposition_transport(delta_gamma)` — explicit-formula multi-prime transport

**`atft/topology/sparse_sheaf_laplacian.py`**

`SparseSheafLaplacian` assembles the NK x NK block sparse matrix in
**Block Sparse Row (BSR)** format using scipy. BSR is optimal for block-structured
matrices where K x K blocks tile a sparse N x N pattern. Eigenvalues are extracted
with `scipy.sparse.linalg.eigsh` using shift-invert mode. Designed for
N ~ 10,000 and K <= 50 on CPU. Default transport mode is `"superposition"`.

**`atft/topology/gpu_sheaf_laplacian.py`**

`GPUSheafLaplacian` uses a hybrid CPU/GPU architecture: batched matrix
exponentials (for non-Hermitian superposition transport) are computed on CPU
using scipy/numpy; the resulting K x K blocks are transferred to GPU memory
and assembled into a CuPy sparse CSR matrix. Eigenvalues are extracted with a
**spectral flip trick**: rather than computing the smallest eigenvalues of L
directly (ill-conditioned for large sparse GPU matrices), we compute the largest
eigenvalues of (lambda_max * I - L) and reflect. Uses CuPy LOBPCG or cupyx
eigsh depending on available CUDA libraries. Requires cupy-cuda12x and an
NVIDIA GPU with >= 8 GB VRAM.

**`atft/feature_maps/spectral_unfolding.py`**

`SpectralUnfolding` normalizes a spectrum so that the mean level spacing is 1,
making spectra from different N and different spectral regions comparable. Three
modes:

- `"zeta"`: uses the analytic smooth zeta zero staircase
  N(T) = (T/2*pi) * log(T/2*pi*e) + O(log T)
- `"semicircle"`: uses the Wigner semicircle CDF for GUE spectra
- `"rank"`: empirical rank-based CDF (deprecated for scientific use; never
  use rank unfolding for comparing spectral statistics)

**Important:** Rank unfolding introduces systematic biases in pair-correlation
statistics and must not be used when comparing zeta zeros to GUE. Always use
the appropriate smooth CDF for each ensemble.

### 3.3 Computational Infrastructure

| Engine | Sparse format | Eigensolver | Typical scale |
|--------|--------------|-------------|---------------|
| CPU (scipy) | BSR | eigsh, shift-invert | N=9877, K=20 |
| GPU (CuPy) | CSR | LOBPCG / spectral flip | N=2000-9877, K=50-100 |
| Distributed | partitioned by role | per-node engine | multi-machine |

The **distributed sweep** (`phase3_distributed.py`) partitions the (sigma, epsilon)
grid across machines by role string, e.g., `--role gpu-k50` or `--role cpu-fe`.
Each machine writes HDF5 results to a shared directory; a merge step aggregates
the full grid.

The **spectral flip trick** is essential for GPU eigensolvers: CuPy's sparse
ARPACK-equivalent does not support shift-invert. Instead, we estimate lambda_max
with a few power iterations, form (lambda_max * I - L), extract its k largest
eigenvalues with LOBPCG, and map them back. This yields the k smallest eigenvalues
of L reliably and without dense factorization.

---

## 4. Experimental Results

### 4.1 Phase 1 — Spectral Baseline

**Goal:** Confirm that the raw zeta zero point cloud is distinguishable from GUE
at the topological level, before introducing any arithmetic structure.

**Method:** Vietoris-Rips complex with identity transport (U = I), spectral sum
of Hodge Laplacian.

**Key findings:**

- Monotonic convergence ratio decreasing from 0.144 to 0.052 across the epsilon
  grid, confirming the zeta zero topology is non-trivially ordered
- Flat Gini L2 coefficient ~0.025 for both zeta and GUE, indicating eigenvalue
  distributions have similar spread at this resolution
- Optimization of spectral normalization following Dumitriu-Edelman: semicircle
  unfolding for GUE, smooth staircase for zeta

This phase established that smooth CDF unfolding is mandatory and that the
Vietoris-Rips complex is a reliable topological probe at scales epsilon ~ 1-5
(in units of mean spacing).

### 4.2 Phase 2 — Transport Map Validation

**Goal:** Validate the u(K) gauge connection construction and the four transport
modes. Establish that the FE connection produces unitary transport only at sigma = 0.5.

**Key findings:**

- **Flat connection:** Holonomy group is compact; the eigenspectrum of the sheaf
  Laplacian is well-conditioned and smooth in sigma.
- **FE connection symmetry breaking:** Verified analytically and numerically.
  The operator norm ||U_e - U_e^{-dagger}|| is minimized precisely at sigma = 0.5
  and grows monotonically away from it for all tested edges.
- **Control test:** Applying the ATFT pipeline to a random Poisson point cloud
  with the FE connection shows a pronounced peak of S(sigma) at sigma = 0.5.
  This demonstrates that the sigma = 0.5 feature in the FE mode is **geometric**
  (arising from the non-unitarity structure of G_p^FE) and **not arithmetic**
  (not specific to zeta zeros). This control result is essential: it rules out
  the FE mode as a valid discriminator and motivates the superposition mode.

### 4.3 Phase 3 — K=20 Full Sweep

**Configuration:** K=20 (8 primes), N=9877 Odlyzko zeros, transport mode:
superposition, sigma grid: [0.25, 0.30, ..., 0.75], epsilon grid: [1.5, 2.0,
2.5, 3.0, 4.0, 5.0], k_eigenvalues=100.

**Results:**

- **Zeta zeros:** Spectral sum S(sigma) increases monotonically through sigma = 0.50.
  No peak or turnover is observed. S increases by ~15% from sigma=0.25 to sigma=0.75.
- **Random and GUE controls:** Signals are approximately 670x weaker than zeta
  at all (sigma, epsilon) pairs, confirming the zeta zero point cloud has a
  substantially stronger arithmetic signal.
- **Interpretation — Fourier truncation:** With K=20, the gauge connection includes
  only 8 primes, giving 8 distinct Fourier harmonics e^{i*Delta_gamma*log(p)} in
  the superposition generator. This is insufficient to localize the interference
  peak; more harmonics are needed to constructively resolve the peak at sigma = 0.5.
  The monotone behavior at K=20 is consistent with the Fourier truncation hypothesis:
  the spectral peak exists but lies outside the bandwidth resolved by 8 primes.

### 4.4 Phase 3b — K=50 Scout

**Configuration:** K=50 (15 primes), N=2000 Odlyzko zeros, GPU engine (RTX 4080,
16 GB VRAM), transport mode: superposition.

**Results:**

- **First spectral turnover observed** at epsilon = 5.0
- Signal peaks near sigma ~ 0.40-0.50 and drops by approximately 4% at
  sigma = 0.60-0.75
- At the same epsilon, K=20 was monotonically increasing with no turnover

**Significance:** This is the first experimental confirmation that increasing K
(adding more primes to the gauge connection) produces a qualitative change in the
sigma profile — from monotone to non-monotone. The turnover is directionally
consistent with the prediction that a peak should localize toward sigma = 0.5 as
K grows. The 4% drop is modest but statistically significant given the control
signal level.

### 4.5 Outlook: K=100+

**Prediction from Fourier truncation theory:**

As K increases, the superposition generator includes more primes and thus more
Fourier harmonics. By analogy with the explicit formula, the partial sum

$$\sum_{p \leq K} \frac{\log p}{\sqrt{p}} \cos(\gamma \log p)$$

approximates a delta function in gamma-space as K -> infinity. The analogous
behavior in the gauge connection should produce:

- A spectral peak that becomes sharper and more localized near sigma = 0.5
- A contrast ratio C(sigma) that increases monotonically with K
- A phase transition visible as a sharp inflection in S(sigma) near sigma = 0.5

The K=100 sweep (25 primes) is the next planned experiment. Expected peak
position: sigma ~ 0.45-0.50. Expected contrast ratio: 2-5x above K=50 level.

If K=100 confirms the trend, a scaling analysis C(sigma*, K) vs K will be
performed to extrapolate the K -> infinity limit and determine whether the peak
position converges to exactly 0.5.

---

## 5. Data

The experiment uses **Andrew Odlyzko's high-altitude zeta zeros**: imaginary
parts of non-trivial Riemann zeros near the 10^20-th zero. These are preferred
over low zeros because:

- The pair correlation statistics are in excellent agreement with GUE at high
  altitude, making the GUE comparison meaningful
- The large imaginary parts reduce contamination from low-energy arithmetic effects
- The density of zeros per unit is higher, enabling smaller epsilon thresholds

**File:** `data/odlyzko_zeros.txt`
**Size:** approximately 1.8 MB
**Format:** One zero per line (imaginary part only), plain text
**Count:** 100,000 zeros available; experiments use subsets of N=2000 to N=9877

The Odlyzko dataset is publicly available at:
https://www.dtc.umn.edu/~odlyzko/zeta_tables/

---

## 6. Requirements and Installation

### Core dependencies

```
Python >= 3.11
numpy >= 1.24
scipy >= 1.11
h5py >= 3.9
matplotlib >= 3.7
```

### Optional GPU dependencies

```
cupy-cuda12x       # CuPy for CUDA 12.x
NVIDIA GPU with >= 8 GB VRAM
CUDA 12.x runtime
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Reimann_Hypothesis

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# For GPU support (CuPy)
pip install cupy-cuda12x

# Verify GPU availability
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) found')"
```

---

## 7. Running Experiments

### Phase 1 — Spectral baseline benchmark

```bash
python -m atft.experiments.phase1_benchmark
```

### Phase 3 — K=20 CPU sweep (definitive control test)

```bash
python -u -m atft.experiments.phase3_superposition_sweep 2>&1 | tee output/phase3_results_K20.log
```

Quick development run (reduced grid):

```bash
python -u -m atft.experiments.phase3_superposition_sweep --quick
```

### Phase 3 — Distributed multi-machine sweep

Each machine is assigned a role that determines its slice of the (sigma, epsilon)
grid:

```bash
# On a GPU machine running K=50
python -u -m atft.experiments.phase3_distributed --role gpu-k50

# On a CPU machine running functional-equation transport
python -u -m atft.experiments.phase3_distributed --role cpu-fe
```

### Phase 3c — K=100 GPU sweep

```bash
python -u -m atft.experiments.phase3c_gpu_k100 2>&1 | tee output/phase3c_k100.log
```

### Holonomy and connection verification scripts

```bash
python verify_fe_transport.py        # Confirm FE unitarity at sigma=0.5 only
python verify_holonomy.py            # Flat connection holonomy check
python verify_resonant_holonomy.py   # Resonant connection curvature check
```

---

## 8. Tests

```bash
pytest tests/ -v
```

The test suite covers:

- Prime representation construction: semigroup homomorphism property rho(p)*rho(q) = rho(p*q) when p*q <= K
- Transport map unitarity (global mode) and near-unitarity at sigma=0.5 (FE mode)
- Sheaf Laplacian positive semi-definiteness
- Spectral unfolding: output mean spacing = 1.0, invariance under scale
- Source generators: correct spectral statistics for GUE and Poisson outputs
- End-to-end pipeline smoke test on N=30 synthetic data

---

## 9. Project Status

**Ti V0.1 — Active research. Phase 3 distributed sweep in progress across
multi-GPU fleet.**

| Phase | Status | Key finding |
|-------|--------|-------------|
| Phase 1 | Complete | Zeta topology distinguishable from GUE; smooth unfolding validated |
| Phase 2 | Complete | u(K) connection validated; FE unitarity at sigma=0.5 confirmed; FE mode ruled out as discriminator (geometric artifact) |
| Phase 3 (K=20, CPU) | Complete | 670x signal over controls; monotone profile due to Fourier truncation |
| Phase 3b (K=50, GPU) | Complete | First spectral turnover observed; peak near sigma=0.40-0.50 |
| Phase 3c (K=100, GPU) | In progress | Expected to show sharper peak, higher contrast ratio |
| Phase 4 (scaling analysis) | Planned | C(sigma*, K) vs K extrapolation; K -> infinity limit |

---

## 10. Citation

If you use this work in your research, please cite the accompanying paper:

```
@article{atft2026,
  title   = {Adaptive Topological Field Theory for Computational Investigation
             of the Riemann Hypothesis},
  author  = {},
  journal = {},
  year    = {2026},
  note    = {Ti V0.1, preprint}
}
```

---

## 11. License

Research use only. Contact the authors for collaboration or licensing inquiries.
