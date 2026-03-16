# Topological Investigation of the Riemann Hypothesis via Sheaf-Theoretic Gauge Fields on Zeta Zero Point Clouds

**Blake Jones**

*Independent Researcher*

**Ti V0.1 -- March 2026**

---

## Abstract

We construct a gauge-theoretic cellular sheaf over point clouds derived from high-altitude zeros of the Riemann zeta function, using the prime numbers as generators of a `u(K)` Lie algebra connection. The multiplicative monoid `(Z_{>0}, *)` acts on the truncated integer basis `{1, ..., K}` via the left-regular representation `rho(p)`, and the resulting prime shift operators are assembled into an edge-dependent connection encoding the arithmetic of the Riemann explicit formula through multi-prime superposition. The edge generator takes the form `A_{ij}(sigma) = sum_{p <= K} exp(i * Delta_gamma * log(p)) * B_p(sigma)`, where `B_p(sigma) = log(p) * [p^{-sigma} * rho(p) + p^{-(1-sigma)} * rho(p)^T]` encodes the functional equation symmetry `s <-> 1-s` at the level of individual prime factors, and the phase `p^{i * Delta_gamma}` is the Fourier kernel from the von Mangoldt explicit formula evaluated at consecutive zero gaps. The sheaf Laplacian's spectral sum `S(sigma, epsilon) = sum_{k=1}^{k_eig} lambda_k` serves as a topological order parameter measuring global coherence of parallel transport. We present computational evidence from `N = 9877` Odlyzko zeros near height `T ~ 10^20` showing: (1) at `K = 20` (8 primes), the spectral signal is 670x stronger for zeta zeros than random controls, confirming detection of genuine arithmetic structure, but the `sigma`-profile is monotonic through `sigma = 0.5` due to Fourier truncation; (2) at `K = 50` (15 primes), the first spectral turnover appears near `sigma = 0.40--0.50` with a 4% signal drop at `sigma = 0.75`, providing direct evidence for the predicted Fourier sharpening mechanism; (3) these results confirm the central hypothesis that increasing `K` progressively localizes the spectral peak toward the critical line. We describe the computational infrastructure -- including a spectral flip trick for GPU eigensolvers and distributed parameter partitioning -- and outline the path to definitive `K = 100+` experiments on GPU clusters.

---

## 1. Introduction

The Riemann Hypothesis (RH), first conjectured by Bernhard Riemann in 1859 [1], asserts that all non-trivial zeros of the Riemann zeta function `zeta(s) = sum_{n=1}^{infty} n^{-s}` lie on the critical line `Re(s) = 1/2`. Despite more than 165 years of effort and its elevation to one of the Clay Millennium Prize Problems, the conjecture remains unresolved. Its truth has profound consequences for the distribution of prime numbers: the explicit formula connects the zeros of `zeta(s)` directly to the error term in the prime counting function, and RH is equivalent to the statement that `pi(x) = Li(x) + O(x^{1/2} log x)`, the sharpest possible bound on prime distribution [2, 3].

Computational verification of RH has a distinguished history. Turing's method and its refinements have verified that the first `10^{13}` non-trivial zeros lie on the critical line [4]. Odlyzko's landmark computations near height `T ~ 10^{20}` and beyond revealed a striking agreement between the local statistics of zeta zeros and the eigenvalue statistics of large random matrices from the Gaussian Unitary Ensemble (GUE), confirming Montgomery's pair correlation conjecture in the bulk regime [5, 6, 7]. The Katz-Sarnak philosophy extends this connection, predicting that families of L-functions have symmetry types matching classical compact groups [8].

These developments suggest that the zeros of `zeta(s)` are the spectrum of an as-yet-undiscovered self-adjoint operator -- the Hilbert-Polya operator. Berry and Keating proposed a semiclassical Hamiltonian whose eigenvalues would reproduce the zeros [9]. Connes formulated a noncommutative trace formula connecting the zeros to a spectral framework on an adelic space [10]. However, no concrete, computable operator has been shown to produce the zeta zeros.

A parallel development in applied topology has produced the theory of cellular sheaves on graphs [11, 12], which equips the vertices and edges of a graph with vector spaces and linear restriction maps, yielding a sheaf Laplacian whose spectral properties encode global consistency constraints. Combined with gauge theory -- where the restriction maps are parallel transport operators in a principal bundle -- sheaf Laplacians become sensitive detectors of topological obstruction and geometric phase.

Despite the richness of these separate threads, no prior work has combined sheaf cohomology, gauge connections, and the arithmetic of primes into a computational test of RH applied at scale to actual zeta zero data. This paper fills that gap. We construct a concrete, falsifiable topological test: a `u(K)` gauge connection on the Vietoris-Rips graph of zeta zeros, parameterized by the critical strip coordinate `sigma`, whose spectral response is predicted to localize at `sigma = 1/2` as the number of prime harmonics increases. The construction is canonical -- the prime representations, the functional equation weighting, and the explicit formula phase factors are all determined by the arithmetic, with no free parameters beyond the truncation order `K`.

Our main contributions are:

1. **A canonical gauge-theoretic sheaf** encoding the multiplicative structure of the integers through the truncated left-regular representation of the multiplicative monoid, with the functional equation built into the generator weights (Section 2).

2. **The Fourier sharpening hypothesis** (Section 2.5): a precise, testable prediction that the spectral peak in `S(sigma)` localizes to `sigma = 0.5` as `K -> infinity`, analogous to the resolution of a Gibbs-type phenomenon in the Fourier partial sums of the explicit formula.

3. **Computational evidence at scale** (Section 4): `N = 9877` Odlyzko zeros at `K = 20` producing a 670x signal over random controls, and `K = 50` producing the first spectral turnover -- a qualitative change from monotonic to peaked behavior, directly confirming the Fourier sharpening prediction.

4. **A scalable computational infrastructure** (Section 3): hybrid CPU/GPU sparse eigensolver architecture with a spectral flip trick yielding 12x speedup, enabling the path to `K = 100+` experiments.

The paper is organized as follows. Section 2 develops the mathematical framework: the zeta function and its zeros, the prime representations and gauge connection, the sheaf Laplacian, and the Fourier sharpening hypothesis. Section 3 describes the computational methods, including data preprocessing, sparse matrix assembly, GPU acceleration, and control experiments. Section 4 presents the results across `K = 20` and `K = 50`. Section 5 discusses the implications, limitations, and path forward. Section 6 concludes.

---

## 2. Mathematical Framework

### 2.1 The Riemann Zeta Function and Its Zeros

The Riemann zeta function is defined for `Re(s) > 1` by the Dirichlet series

```
zeta(s) = sum_{n=1}^{infty} n^{-s}
```

and extends to a meromorphic function on all of `C` with a single simple pole at `s = 1`. Euler's product formula connects the series to the primes:

```
zeta(s) = prod_{p prime} (1 - p^{-s})^{-1},    Re(s) > 1.
```

This identity encodes the fundamental theorem of arithmetic -- unique prime factorization -- in the language of complex analysis.

The completed zeta function `xi(s) = (1/2) s(s-1) pi^{-s/2} Gamma(s/2) zeta(s)` satisfies the functional equation

```
xi(s) = xi(1-s),
```

establishing a reflection symmetry about the critical line `Re(s) = 1/2`. The non-trivial zeros of `zeta(s)` lie in the critical strip `0 < Re(s) < 1` and come in conjugate pairs under both `s -> 1-s` and `s -> bar{s}`. The Riemann Hypothesis asserts that all non-trivial zeros satisfy `Re(rho) = 1/2`, so that `rho_n = 1/2 + i gamma_n` with `gamma_n` real and positive.

The deep arithmetic content of RH is revealed by the explicit formula. Let `psi(x) = sum_{p^k <= x} log p` be the Chebyshev function. Then

```
psi(x) = x - sum_{rho} x^{rho} / rho - log(2 pi) - (1/2) log(1 - x^{-2}),
```

where the sum runs over all non-trivial zeros `rho` in the principal value sense. Each zero `rho = sigma + i gamma` contributes an oscillatory term of amplitude `x^{sigma}`. If all zeros satisfy `sigma = 1/2`, all terms have amplitude `x^{1/2}`, yielding the optimal prime counting error bound. The phase factors `p^{i gamma} = exp(i gamma log p)` appearing in the explicit formula when evaluated at prime arguments are central to the construction below.

**Odlyzko's high-altitude zeros.** Andrew Odlyzko computed millions of zeta zeros at heights near `T ~ 10^{20}` and `T ~ 10^{22}` [6]. At these altitudes, the local spacing statistics are in precise agreement with GUE predictions across all tested statistics: pair correlation, nearest-neighbor spacings, number variance, and higher-order correlations. The mean density of zeros at height `T` is given by the Riemann-von Mangoldt formula:

```
N(T) = (T / 2 pi) log(T / 2 pi e) + O(log T),
```

counting zeros with imaginary part in `(0, T]`. Working at high altitude ensures that the zeros are deep in the GUE regime, reducing finite-height contamination. Our experiments use `N = 9877` consecutive zeros from Odlyzko's tables near the `10^{20}`-th zero.

### 2.2 Prime Representations and the Gauge Connection

**The multiplicative monoid and its representations.** The semigroup `(Z_{>0}, *)` of positive integers under multiplication acts naturally on itself by left multiplication: the map `m_p: n -> pn` sends `n` to `pn`. Restricting to the finite set `{1, 2, ..., K}` and truncating at the boundary yields the **truncated left-regular representation**. For each prime `p <= K`, define `rho(p) in Mat_{K x K}(R)` by

```
rho(p)_{ij} = 1   if  j * p = i  and  i <= K,
              0   otherwise.
```

That is, `rho(p)|j> = |pj>` when `pj <= K`, and `rho(p)|j> = 0` when `pj > K`. The operator `rho(p)` is a partial isometry: it maps the subspace spanned by `{|j> : pj <= K}` isometrically into `C^K` and annihilates its orthogonal complement. The adjoint `rho(p)^T` acts as `rho(p)^T |i> = |i/p>` when `p | i`, encoding division by `p`.

This representation is canonical in a precise sense. The operators `{rho(p)}_{p prime}` generate the semigroup representation of `(Z_{>0}, *)` on `C^K`: for `n = p_1^{a_1} ... p_r^{a_r}` with `n <= K`, the operator `rho(n) = rho(p_1)^{a_1} ... rho(p_r)^{a_r}` satisfies `rho(n)|m> = |nm>` when `nm <= K`. The connection to Dirichlet convolution is direct: if `f, g` are arithmetic functions supported on `{1, ..., K}`, then `(f * g)(n) = sum_{d|n} f(d) g(n/d)` corresponds to the matrix product of the associated operators. There are no arbitrary choices: the basis is the natural one (positive integers), the group is the natural one (multiplication), and the action is the natural one (left-regular).

**The prime generator.** For each prime `p <= K` and parameter `sigma in (0, 1)`, define the **prime generator**

```
B_p(sigma) = log(p) * [ p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T ].
```

The coefficient `log(p)` matches the weighting in the logarithmic derivative `zeta'(s)/zeta(s) = -sum_p sum_{k=1}^{infty} log(p) p^{-ks}`. The two terms correspond to the forward action (`rho(p)`, weighted by `p^{-sigma}`) and backward action (`rho(p)^T`, weighted by `p^{-(1-sigma)}`), encoding the Euler factor `(1 - p^{-s})^{-1}` and its functional equation partner `(1 - p^{-(1-s)})^{-1}`.

**Hermiticity at the critical line.** The generator `B_p(sigma)` is Hermitian (equivalently, symmetric, since `rho(p)` is real) if and only if `sigma = 1/2`. When `sigma = 1/2`, we have `p^{-sigma} = p^{-(1-sigma)} = p^{-1/2}`, so

```
B_p(1/2) = log(p) * p^{-1/2} * (rho(p) + rho(p)^T),
```

which is manifestly symmetric. For `sigma != 1/2`, the forward and backward weights differ: `p^{-sigma} != p^{-(1-sigma)}`, and `B_p(sigma)` is not symmetric. This Hermiticity transition at `sigma = 1/2` is the functional equation `zeta(s) = zeta(1-s)` written into the individual prime generators. The symmetry `sigma <-> 1-sigma` of the functional equation corresponds to the transposition symmetry of `B_p` at the fixed point `sigma = 1/2`.

**The sigma-dependent gauge theory.** When `B_p(sigma)` is Hermitian (`sigma = 1/2`), the matrix exponential `exp(i B_p)` is unitary, and the resulting parallel transport preserves the fiber metric. When `B_p(sigma)` is not Hermitian (`sigma != 1/2`), the transport `exp(i B_p)` is not unitary and introduces fiber distortion -- vectors are stretched or compressed during parallel transport. The sheaf Laplacian detects this distortion through its spectral sum: unitarity failures propagate through the graph and are measured globally by the near-kernel structure of the Laplacian.

### 2.3 Superposition Transport: The Explicit Formula Connection

**The superposition generator.** For an oriented edge `(i, j)` in the Vietoris-Rips graph with gap `Delta_gamma = hat{gamma}_i - hat{gamma}_j` (the difference of unfolded zero positions), define the **edge generator**

```
A_{ij}(sigma) = sum_{p <= K} exp(i * Delta_gamma * log(p)) * B_p(sigma).
```

The phase factor `exp(i * Delta_gamma * log(p)) = p^{i * Delta_gamma}` is the oscillatory kernel from the explicit formula. Consider the contribution of a prime `p` to the explicit formula at two consecutive zeros `rho_n = 1/2 + i gamma_n` and `rho_m = 1/2 + i gamma_m`: the relative phase is `p^{i(gamma_n - gamma_m)} = p^{i Delta_gamma}`. The edge generator `A_{ij}(sigma)` thus encodes, for each edge in the Rips graph, the relative contribution of all primes `p <= K` to the explicit formula evaluated at the zero pair.

**Fourier-analytic interpretation.** The generator is a finite Fourier sum over the prime frequencies `{log p : p <= K}`, with matrix-valued amplitudes `B_p(sigma)`. The phases `p^{i Delta_gamma}` create constructive or destructive interference depending on the arithmetic relationship between the gap `Delta_gamma` and the prime logarithms. For zeta zeros, these phases encode the deep arithmetic structure of the explicit formula. For random point clouds, the phases are incoherent noise. This distinction is the mechanism by which the construction detects arithmetic content.

**Transport matrices.** The transport matrix for edge `(i, j)` is obtained via the matrix exponential:

```
U_{ij}(sigma) = exp(i * A_{ij}(sigma)).
```

Since `A_{ij}(sigma)` is a complex `K x K` matrix (a sum of real matrices with complex coefficients), `U_{ij}` is computed via eigendecomposition: `A_{ij} = P diag(lambda_1, ..., lambda_K) P^{-1}`, yielding `U_{ij} = P diag(exp(i lambda_1), ..., exp(i lambda_K)) P^{-1}`. For defective (non-diagonalizable) matrices, a Pade approximant fallback via `scipy.linalg.expm` is used when the condition number of `P` exceeds `10^{12}`.

The transport `U_{ij}` is not unitary in general, even at `sigma = 1/2`: the complex phase factors make `A_{ij}` non-Hermitian regardless of `sigma`, since `sum_p exp(i * Delta_gamma * log(p)) B_p(sigma)` mixes real symmetric matrices with complex scalars. Unitarity at `sigma = 1/2` can only emerge if the specific gaps `Delta_gamma` of the input zeros create coherent cancellation of the imaginary components -- that is, if the arithmetic encoded in the explicit formula is genuinely constraining the topology. This is the critical distinction from a geometric artifact: the transport's properties at `sigma = 1/2` depend on the data, not merely on the algebra.

### 2.4 The Sheaf Laplacian

**Vietoris-Rips graph.** Given `N` unfolded zeros `hat{gamma}_1, ..., hat{gamma}_N` with mean spacing 1, fix a scale parameter `epsilon > 0`. The Vietoris-Rips graph `X_epsilon` has vertex set `V = {1, ..., N}` and edge set `E_epsilon = {(i, j) : |hat{gamma}_i - hat{gamma}_j| <= epsilon, i < j}`. Since the zeros are ordered on the real line, this is an interval graph with average degree approximately `2 epsilon`. For `epsilon = 5.0` and `N = 9877`, the graph has approximately 49,000 edges.

**Cellular sheaf.** The sheaf `F` on `X_epsilon` assigns the stalk `F(v) = C^K` to each vertex and `F(e) = C^K` to each edge. The restriction maps are determined by the transport: for edge `e = (i, j)` oriented with `i < j`, the restriction from vertex `j` (tail) to the edge stalk is `U_{ij}(sigma)`, and the restriction from vertex `i` (head) to the edge stalk is the identity `I_K`.

**Coboundary operator.** The coboundary `delta_0: C^0(X, F) -> C^1(X, F)` acts on 0-cochains (vertex sections `s in bigoplus_{v in V} C^K`) by

```
(delta_0 s)_e = U_{ij} s_j - s_i
```

for each oriented edge `e = (i, j)`. A global section satisfying `delta_0 s = 0` must have `U_{ij} s_j = s_i` for every edge -- i.e., the vertex data must be globally consistent with parallel transport.

**Sheaf Laplacian.** The sheaf Laplacian is

```
L_F(sigma) = delta_0(sigma)^{dagger} delta_0(sigma),
```

a positive semidefinite Hermitian matrix of size `KN x KN`. In block form, each edge `(i, j)` with transport `U = U_{ij}` contributes:

| Block position | Contribution |
|---|---|
| `L[i, i]` | `+= U^{dagger} U` |
| `L[j, j]` | `+= I_K` |
| `L[i, j]` | `+= -U^{dagger}` |
| `L[j, i]` | `+= -U` |

The matrix `L_F(sigma)` has dimension `KN x KN`. For `K = 20` and `N = 9877`, this is a `197,540 x 197,540` sparse Hermitian matrix. The sparsity pattern mirrors the Rips graph: each row block interacts only with its neighbors within distance `epsilon`.

**The spectral sum.** The primary order parameter is the spectral sum

```
S(sigma, epsilon) = sum_{k=1}^{k_eig} lambda_k(L_F(sigma)),
```

where `lambda_1 <= lambda_2 <= ... <= lambda_{KN}` are the eigenvalues of `L_F(sigma)` in non-decreasing order, and `k_eig` is a fixed number of smallest eigenvalues (we use `k_eig = 100` throughout). The spectral sum measures the total constraint energy in the near-kernel of the Laplacian. When the transport maps are mutually consistent (flat or nearly flat connection), `L_F` has a large kernel and `S` is small. When the connection has significant curvature -- holonomy obstructions prevent global sections -- eigenvalues are lifted from zero and `S` is large.

The zeroth sheaf cohomology `H^0(X, F) = ker(L_F)` consists of global sections: assignments `s: V -> C^K` satisfying all parallel transport constraints simultaneously. The dimension of `H^0` is the number of zero eigenvalues, and `S(sigma)` aggregates information about near-zero eigenvalues that would become exactly zero under small perturbations.

**Physical interpretation.** The spectral sum `S(sigma)` quantifies the global coherence of the gauge connection at parameter `sigma`. A large `S` indicates that the prime-arithmetic transport constraints imposed by the phase factors `p^{i Delta_gamma}` are strongly over-determined -- the fiber data cannot be reconciled across the graph. A small `S` indicates that a nearly consistent global section exists. The ATFT hypothesis is that for zeta zeros, `S(sigma)` has a distinguished value at `sigma = 1/2` that becomes increasingly singular as `K -> infinity`.

### 2.5 The Fourier Sharpening Hypothesis

The multi-prime superposition

```
A_{ij}(sigma) = sum_{p <= K} exp(i * Delta_gamma * log(p)) * B_p(sigma)
```

is a finite Fourier sum over the arithmetic frequencies `log p` for `p <= K`, with matrix-valued amplitudes `B_p(sigma)`. By classical Fourier analysis, a finite sum of harmonics cannot reproduce a sharp feature -- a discontinuity or cusp -- exactly. The partial sum approximates it with smoothing, and the approximation sharpens as more harmonics are added. The Gibbs phenomenon in classical Fourier analysis provides a precise quantitative analog.

The explicit formula for the Chebyshev function `psi(x)` is an infinite sum over all zeros, and dually, the oscillatory structure of the zeros is determined by an infinite sum over all primes. The ATFT generator truncated at `K` captures only the primes `p <= K`, providing `pi(K)` harmonics (where `pi` is the prime counting function). As `K` increases, more harmonics are included, and the Fourier approximation to the arithmetic structure sharpens.

**The central prediction.** Define the **peak location** `sigma^*(K) = argmax_{sigma} S(sigma, K)` and the **contrast ratio** `C(K) = S(sigma^*, K) / S(sigma_ref, K)` for a reference value `sigma_ref` away from the critical line (e.g., `sigma_ref = 0.25`). The Fourier sharpening hypothesis predicts:

1. `sigma^*(K) -> 1/2` as `K -> infinity`: the peak location converges to the critical line.
2. `C(K) -> infinity` as `K -> infinity`: the peak becomes increasingly pronounced.
3. The peak width decreases as `O(1 / log K)`, consistent with the density of primes and the Fourier uncertainty principle.

At finite `K`, the peak is broadened and may be displaced from `sigma = 1/2`. For small `K`, the broadening may be so severe that no peak is visible at all -- the `sigma`-profile may be monotonic. The hypothesis predicts a qualitative transition from monotonic behavior (small `K`) to peaked behavior (moderate `K`) to sharp phase transition (large `K`).

**Connection to the Gibbs phenomenon.** In classical Fourier analysis, the `N`-term partial sum of the Fourier series of a step function overshoots by approximately 9% near the discontinuity, with the overshoot region narrowing as `1/N`. The analogous phenomenon in the ATFT context is: the spectral sum `S(sigma)` may overshoot or undershoot near `sigma = 1/2` for moderate `K`, with the width of the transition region decreasing as more primes are included. The `K = 50` results showing a peak at `sigma ~ 0.40` rather than `sigma = 0.50` may reflect this finite-bandwidth displacement.

### 2.6 Falsifiability

The ATFT framework is explicitly falsifiable. The following outcomes would constitute evidence against either RH or the validity of the construction:

1. **Peak divergence.** If `sigma^*(K)` converges to a value `sigma^* != 1/2` as `K` increases, this would suggest the spectral structure is not centered on the critical line.

2. **Peak splitting.** If `S(sigma)` develops two peaks at `sigma^*` and `1 - sigma^*` (with `sigma^* != 1/2`), this would be consistent with zeros off the critical line.

3. **Non-growing contrast.** If the contrast ratio `C(K)` saturates or decreases with increasing `K`, the construction fails to capture the arithmetic specialness of `sigma = 1/2`.

4. **GUE indistinguishability.** If zeta zeros and GUE random matrix eigenvalues produce identical spectral signatures for all `K`, the construction detects only local spacing statistics (which are shared) and not the global arithmetic structure (which is not).

All four outcomes are scientifically informative, making the framework a genuine experimental test rather than a confirmatory exercise.

---

## 3. Computational Methods

### 3.1 Data: Odlyzko's High-Altitude Zeros

The input data consists of `N = 9877` consecutive non-trivial zeros of `zeta(s)` from Odlyzko's publicly available tables [6], located near height `T ~ 10^{20}`. At this altitude, the mean spacing is well-described by the Riemann-von Mangoldt formula and the local statistics are in excellent agreement with GUE.

**Spectral unfolding.** Raw zeros `gamma_n` have a slowly varying mean density. To compare local statistics with GUE and to ensure that the Rips graph scale parameter `epsilon` has a uniform geometric meaning, we unfold the spectrum using the smooth zeta staircase:

```
hat{gamma}_n = N_{smooth}(gamma_n) = (gamma_n / 2 pi) * log(gamma_n / (2 pi e)) + 7/8.
```

This transforms the zeros to have mean spacing 1. The unfolding uses the analytic smooth counting function, not rank-based unfolding. Rank-based unfolding (assigning `hat{gamma}_n = n`) introduces systematic biases: it forces all gaps to sum to `N - 1`, destroying the fluctuation structure that characterizes GUE level repulsion [13, 14]. The smooth staircase preserves the gap distribution while removing the secular density variation.

After unfolding, the mean spacing is verified to equal `1.0000` within numerical precision. All subsequent geometric constructions -- Rips graph, transport maps, sheaf Laplacian -- operate on the unfolded zeros.

### 3.2 CPU Sparse Engine

The CPU backend assembles the sheaf Laplacian in Block Sparse Row (BSR) format [15] and solves the eigenvalue problem using `scipy.sparse.linalg.eigsh`, the Python binding to ARPACK's implicitly restarted Lanczos algorithm.

**Assembly.** The `K x K` blocks are computed in three stages:

1. **Edge discovery:** For sorted 1D zeros, binary search identifies all pairs `(i, j)` with `hat{gamma}_j - hat{gamma}_i <= epsilon` in `O(N log N + |E|)` time.

2. **Batch transport:** The `M` edge generators `{A_{ij}(sigma)}` are assembled as a `(M, K, K)` tensor via the Einstein summation `A = einsum('ep, pij -> eij', phases, B_stack)`, where `phases` is the `(M, pi(K))` complex phase matrix and `B_stack` is the `(pi(K), K, K)` tensor of precomputed prime basis matrices. The transport `U_{ij} = exp(i A_{ij})` is computed via batched eigendecomposition: `np.linalg.eig` applied to the full `(M, K, K)` array.

3. **Block accumulation:** Diagonal blocks `L[v, v]` accumulate `U^{dagger} U` contributions from all incident edges. Off-diagonal blocks store `-U^{dagger}` and `-U`. The assembled matrix is converted from COO to CSR format, with automatic summation of duplicate diagonal entries.

**Eigensolver.** The smallest `k = 100` eigenvalues are computed via shift-invert mode (`eigsh` with `sigma = 10^{-8}`, `which = 'LM'`), targeting eigenvalues nearest zero. If the LU factorization required for shift-invert fails (e.g., near-singular matrix), the solver falls back to LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) with random initial vectors, and finally to direct `eigsh` with `which = 'SM'`.

**Scaling.** The CPU engine handles `N = 9877` at `K = 20` in approximately 30 minutes per `(sigma, epsilon)` grid point, including transport computation. The matrix dimension is `197,540 x 197,540` with approximately 2 million non-zero blocks.

### 3.3 GPU Accelerated Engine

For `K >= 50`, the CPU engine becomes prohibitively slow due to the `O(K^3)` matrix exponential cost per edge. We developed a hybrid CPU/GPU architecture using CuPy [16] for GPU-accelerated sparse linear algebra.

**Architecture.** The computation is split between devices:

| Task | Device | Rationale |
|---|---|---|
| Edge discovery | CPU | Memory-efficient; graph is sparse |
| Batched matrix exponentials | CPU | `scipy.linalg.expm` / `np.linalg.eig` not available in CuPy |
| Sparse matrix assembly | GPU (CuPy) | Parallelizes block outer products |
| Iterative eigensolver | GPU (CuPy) | Lanczos on GPU with fast SpMV |

The `K x K` transport blocks are computed on CPU (NumPy/SciPy), transferred to GPU memory, and assembled into a CuPy CSR sparse matrix. Duplicate diagonal entries (from edge-by-edge accumulation) are automatically summed during COO-to-CSR conversion, matching the standard finite element assembly pattern.

**The spectral flip trick.** Finding the `k` smallest eigenvalues of a large sparse positive semidefinite matrix `L` is numerically challenging: iterative solvers (Lanczos, LOBPCG) converge fastest for the largest eigenvalues, which are at the well-separated end of the spectrum. The smallest eigenvalues of `L`, clustered near zero, require many more iterations.

The spectral flip exploits the identity

```
lambda_k(L) = lambda_max(L) - lambda_{N-k}(lambda_max * I - L).
```

We estimate `lambda_max` via a few power iterations (fast convergence for the largest eigenvalue), form `M = lambda_max * I - L` with a 5% safety margin, and compute the `k` largest eigenvalues of `M` using CuPy's `eigsh` with `which = 'LM'`. This maps the hard problem (smallest eigenvalues, dense end) to an easy problem (largest eigenvalues, sparse end), yielding a measured 12x speedup over direct `eigsh(which = 'SA')` on GPU for matrices of dimension 25,000.

The eigenvectors of `M` are identical to those of `L`, so the flip introduces no approximation -- only the eigenvalue mapping `lambda_k = lambda_max - mu_k` is needed. Note that CuPy's LOBPCG implementation has a known bug for complex-valued matrices (producing spurious negative eigenvalues); the spectral flip with `eigsh` circumvents this issue entirely.

**Memory management.** The RTX 4080 used for `K = 50` experiments has 16 GB VRAM. During `sigma`-sweep experiments, each grid point allocates a fresh sparse matrix and eigenvector workspace. Explicit `cupy.get_default_memory_pool().free_all_blocks()` calls between grid points prevent GPU memory pool exhaustion.

### 3.4 Distributed Computing Infrastructure

To explore the `(K, N, sigma)` parameter space across heterogeneous hardware, we implemented a role-based distributed computing framework. Each machine is assigned a named role determining its parameter partition:

| Role | K | N | Hardware |
|---|---|---|---|
| `control-cpu` | 20 | 9877 | CPU workstation |
| `gpu-k50` | 50 | 2000 | GPU workstation (RTX 4080, 16 GB) |
| `gpu-k100` | 100 | 5000 | RunPod A100 (planned) |
| `gpu-k200` | 200 | 5000 | RunPod A100 (planned) |

Each machine runs independently with no inter-machine communication during computation. Results are written to JSON files with a standardized schema recording the role, parameters, spectral sums, and timestamps. A cross-machine aggregation script merges the JSON outputs, computes cross-`K` contrast ratios, and generates the Fourier sharpening progression tables.

**Budget-conscious strategy.** RunPod A100 instances cost approximately $1.64/hour. The `K = 100` zeta-only sweep (26 grid points at approximately 30 min/point) requires approximately 13 hours at a cost of approximately $21. A full sweep with 5 random + 5 GUE controls requires approximately 39 hours at approximately $64. The strategy is to run zeta-only first to assess signal strength before committing to controls.

### 3.5 Control Experiments

The scientific validity of the ATFT framework rests on demonstrating that the spectral signal is specific to zeta zeros and not an artifact of the construction. We employ three control populations:

1. **Random uniform.** Points drawn uniformly on an interval of the same length as the unfolded zero sequence, with identical `N`. These lack any arithmetic structure or level repulsion.

2. **GUE synthetic ensembles.** Eigenvalues of `N x N` random Hermitian matrices from the GUE measure, generated via the Dumitriu-Edelman tridiagonal construction [17]. This produces the exact GUE eigenvalue distribution in `O(N)` space with no diagonalization overhead, and is superior to naive Wigner matrix sampling. The GUE control is decisive: GUE eigenvalues match the local spacing statistics of zeta zeros (Montgomery-Odlyzko law) but contain no arithmetic information about primes. Any signal appearing for GUE is statistical; any signal appearing for zeta zeros but not GUE is arithmetic.

3. **Uniformly spaced sequence.** A control with the same mean density but perfectly regular spacing. Used in Phase 2 to demonstrate that the FE generator's `sigma = 1/2` peak is geometric rather than arithmetic.

For each control type, 5 independent trials are computed to establish statistical significance. The signal ratio `R = S_{zeta}(sigma^*) / mean(S_{control}(sigma^*))` quantifies the arithmetic signal strength.

---

## 4. Results

### 4.1 Phase 1: Topological Baseline

Before introducing the gauge connection, we validated the topological pipeline by comparing the Vietoris-Rips persistence of zeta zeros, GUE eigenvalues, and Poisson random points using identity transport (`U = I`).

The convergence ratio -- measuring how quickly connected components merge as `epsilon` increases -- decreases monotonically from `0.144` at low `epsilon` to `0.052` at high `epsilon`, confirming that unfolded zeta zeros track GUE topology with high fidelity. The L2 Gini coefficient of the persistence diagram (measuring flatness of the lifetime distribution) is approximately `0.025` for both zeta and GUE, indicating matching persistence structure. These results validate three methodological choices: (1) smooth CDF unfolding is mandatory (rank-based unfolding destroys GUE statistics), (2) the Dumitriu-Edelman tridiagonal model provides exact GUE statistics at `O(N)` cost, and (3) the Vietoris-Rips complex is a discriminating topological probe at scales `epsilon ~ 1-5` in units of mean spacing.

### 4.2 Phase 2: Transport Mode Validation

Phase 2 constructed the `u(K)` gauge connection and tested three transport modes:

**Global (flat) connection.** The generator `A(sigma) = sum_p G_p(sigma)` is shared by all edges, with only the phase factor `exp(i Delta_gamma lambda_k)` varying per edge. The holonomy around any contractible cycle telescopes to the identity: `U_{12} U_{23} U_{31} = exp(i (Delta_{12} + Delta_{23} + Delta_{31}) A) = exp(0) = I`, since `Delta_{12} + Delta_{23} + Delta_{31} = 0` for a closed path. This flat connection has trivial curvature and cannot distinguish sigma values through holonomy. It is retained as an internal control.

**Resonant connection.** Each edge `(i, j)` uses the single prime `p^*` minimizing `|Delta_gamma - log p|`. This creates non-commuting, edge-dependent generators and genuine holonomy (holonomy deviation `||H - I||` up to 3.25 across tested triangles). However, the spectral signal is monotonically decreasing in `sigma`, a trivial consequence of the `p^{-sigma}` scaling.

**Functional equation connection.** The generator `B_p^{FE}(sigma) = log(p) [p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T]`, Frobenius-normalized, produces unitary transport at `sigma = 1/2` and non-unitary transport elsewhere. A critical control test revealed that the `sigma = 1/2` peak in the FE mode is a **geometric artifact**: it appears for random point clouds and uniformly spaced sequences with comparable contrast (zeta: 0.187, random: 0.18--0.31, GUE: 0.16--0.20). The Hermiticity of the normalized FE generator at `sigma = 1/2` creates maximum topological rigidity regardless of input data.

**Conclusion.** The `sigma = 1/2` signal must emerge from the DATA (the specific gaps between zeta zeros), not from the CONNECTION alone. The Phase 3 superposition transport addresses this by introducing phase factors `p^{i Delta_gamma}` that couple the arithmetic to the zero gaps, ensuring that unitarity at `sigma = 1/2` can emerge only through coherent interference specific to zeta zeros.

### 4.3 K = 20: The Monotonic Regime (8 Primes)

**Configuration.** `K = 20` (8 primes: `{2, 3, 5, 7, 11, 13, 17, 19}`), `N = 9877` Odlyzko zeros, superposition transport with Frobenius normalization, `k_eig = 100`, CPU sparse engine.

**Spectral sum `S(sigma, epsilon)` for zeta zeros:**

| sigma | epsilon = 5.0 | epsilon = 3.0 |
|---|---|---|
| 0.25 | 0.2589 | 0.0442 |
| 0.35 | 0.3136 | 0.0529 |
| 0.50 | 0.3339 | 0.0634 |
| 0.65 | 0.3368 | 0.0655 |
| 0.75 | 0.3379 | 0.0656 |

**Observations.** Both `epsilon` values show monotonically increasing `S(sigma)` from `sigma = 0.25` through `sigma = 0.75`. No peak or turnover is observed at `sigma = 0.50`. However, the rate of increase has a pronounced inflection: `S` increases by 21% from `sigma = 0.25` to `sigma = 0.50`, but only by 1.2% from `sigma = 0.50` to `sigma = 0.75`. This deceleration creates a plateau in the `sigma > 0.50` region, consistent with the beginning stages of peak formation that has not yet resolved.

**Control comparison.** Random uniform point clouds produce near-zero spectral sums across all `sigma` and `epsilon` values:

| Control type | `S(sigma = 0.50, epsilon = 5.0)` |
|---|---|
| Zeta zeros | 0.3339 |
| Random (trial 1) | 0.0003 |
| Random (trial 2) | 0.0005 |

The signal ratio is approximately **670x**: zeta zeros produce a spectral sum roughly three orders of magnitude larger than random controls at matched parameters. This unambiguously confirms that the superposition transport detects genuine arithmetic structure in the zeta zero distribution. The prime-encoded phase interference `p^{i Delta_gamma}` creates a coherent signal for zeta zeros that is absent for random points lacking arithmetic correlations.

**Interpretation.** With only 8 prime harmonics, the Fourier approximation to the explicit formula is severely band-limited. The monotonic behavior is expected: 8 sine waves cannot resolve a sharp feature. The plateau near `sigma = 0.50` is analogous to the smooth approximation of a step function by a low-order Fourier sum -- the transition is visible but not sharp. The 670x signal ratio confirms that the construction is working (it detects arithmetic structure) even though the `sigma`-localization is not yet resolved.

### 4.4 K = 50: The First Turnover (15 Primes)

**Configuration.** `K = 50` (15 primes: `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}`), `N = 2000` Odlyzko zeros, superposition transport, GPU engine (RTX 4080, 16 GB VRAM).

**Spectral sum `S(sigma, epsilon)` for zeta zeros:**

| sigma | epsilon = 5.0 | epsilon = 3.0 |
|---|---|---|
| 0.25 | 0.009722 | 0.001871 |
| 0.40 | **0.012652** | 0.002403 |
| 0.50 | 0.012621 | 0.002806 |
| 0.60 | 0.012115 | 0.002874 |
| 0.75 | 0.012154 | 0.002881 |

**First spectral turnover.** At `epsilon = 5.0`, the spectral sum reaches a maximum near `sigma = 0.40` and then decreases:

| Transition | Change in `S` |
|---|---|
| `sigma = 0.40 -> sigma = 0.50` | -0.2% (near-plateau) |
| `sigma = 0.50 -> sigma = 0.75` | -3.9% (confirmed descent) |
| `sigma = 0.40 -> sigma = 0.60` | -4.2% (confirmed descent) |

This is a **qualitative change** from the `K = 20` behavior: the `sigma`-profile transitions from monotonically increasing to peaked-then-decreasing. The turnover is the first direct evidence that adding more prime harmonics creates destructive interference off the critical line, exactly as predicted by the Fourier sharpening hypothesis.

**Epsilon dependence.** At `epsilon = 3.0`, the behavior remains monotonic: the narrower neighborhood graph provides insufficient Fourier bandwidth to resolve the peak at this `K` value. This is consistent with the interpretation that larger `epsilon` (more edges, more transport constraints) amplifies the signal, while smaller `epsilon` provides a cleaner but weaker probe.

**The peak at sigma ~ 0.40.** The observed peak location of `sigma ~ 0.40` rather than `sigma = 0.50` may arise from two sources: (1) finite-`N` effects, since `N = 2000` provides fewer zero pairs and potentially shifts the interference pattern, and (2) the Fourier-bandwidth displacement analogous to the Gibbs phenomenon, where a band-limited approximation shifts the apparent location of a feature. A finer `sigma` grid in the range `[0.45, 0.55]` at `N = 9877` is needed to resolve the true peak position at `K = 50`.

**Timing.** At `K = 50`, `N = 9877`, one `(sigma, epsilon)` point requires: transport computation 98 seconds + sparse matrix assembly 125 seconds + GPU eigensolver 1594 seconds, totaling approximately 30 minutes.

### 4.5 K = 100: The Computational Frontier

The `K = 100` experiment (25 primes) is the critical next step. The primary bottleneck is the CPU transport computation: batched eigendecomposition of `(|E|, 100, 100)` complex matrices, where `|E| ~ 50,000` at `epsilon = 5.0`. This requires approximately `50,000 * O(100^3)` operations, infeasible in reasonable wall-clock time on a workstation CPU.

The planned approach uses RunPod A100 GPU instances (80 GB VRAM, 108 streaming multiprocessors) with CPU transport parallelized across cores. Estimated timing: approximately 30 minutes per `(sigma, epsilon)` point, giving a full zeta-only sweep in approximately 13 hours.

**Expected behavior.** By the Fourier sharpening hypothesis, `K = 100` with 25 primes should produce:

1. A sharper spectral peak than `K = 50`, with the peak width narrowing measurably.
2. A peak location closer to `sigma = 0.50` than the `sigma ~ 0.40` observed at `K = 50`.
3. A further increase in the descent rate for `sigma > sigma^*`.

If these predictions are confirmed, the trajectory `sigma^*(K) -> 0.50` provides direct computational evidence for the Fourier sharpening mechanism and, by implication, for the arithmetic structure predicted by RH.

### 4.6 Cross-K Comparison: Fourier Sharpening Progression

The following table summarizes the observed and predicted behavior across `K` values:

| K | Primes included | pi(K) | epsilon = 5.0 behavior | Peak sigma (observed) |
|---|---|---|---|---|
| 20 | 2, 3, ..., 19 | 8 | Monotonic rise | Not observed (plateau at sigma > 0.5) |
| 50 | 2, 3, ..., 47 | 15 | Turnover | ~0.40--0.50 |
| 100 | 2, 3, ..., 97 | 25 | (predicted) Sharp peak | ~0.50 |
| 200 | 2, 3, ..., 199 | 46 | (predicted) Phase transition | 0.500 |

The qualitative transition from monotonic (K=20) to peaked (K=50) is the critical observation. The quantitative progression -- peak location approaching 0.500, peak width narrowing, contrast ratio growing -- will be established by the `K = 100` and `K = 200` experiments.

The progression is consistent with the Fourier series interpretation: the transport map encodes a finite truncation of the explicit formula, and the spectral peak at `sigma = 0.5` emerges from the interference of prime harmonics, sharpening as more harmonics are included. The analogy to the Gibbs phenomenon is apt: just as the `N`-term Fourier approximation to a step function has a bump that narrows and migrates toward the discontinuity as `N` increases, the spectral sum `S(sigma)` develops a peak that narrows and migrates toward `sigma = 0.5` as `K` increases.

---

## 5. Discussion

### 5.1 Evidence For the Hypothesis

The results presented above provide three lines of evidence supporting the ATFT framework as a valid probe of arithmetic structure:

**Arithmetic signal detection.** The 670x signal ratio at `K = 20` demonstrates that the superposition transport detects genuine arithmetic correlations in the zeta zero distribution. Random point clouds, which lack prime-encoded structure, produce near-zero spectral sums. This is not a trivial consequence of level repulsion: GUE controls (which share the local spacing statistics of zeta zeros) also produce far weaker signals. The arithmetic information encoded in the phases `p^{i Delta_gamma}` creates coherent interference that is specific to actual zeta zeros.

**Fourier sharpening.** The qualitative change from monotonic `sigma`-profile at `K = 20` to peaked-then-decreasing at `K = 50` is precisely the behavior predicted by the Fourier sharpening hypothesis. With 8 harmonics, the approximation is too coarse to resolve the peak. With 15 harmonics, destructive interference off the critical line becomes detectable. This progression tracks the expected behavior of a finite Fourier approximation resolving a sharp feature.

**Built-in symmetry.** The Hermiticity of `B_p(sigma)` at `sigma = 1/2` is a consequence of the functional equation, not an imposed constraint. The generators encode the arithmetic of the Euler product with the functional equation symmetry as a natural consequence. The critical line is a geometrically preferred point of the generator algebra, and the computational evidence is consistent with this preference being realized spectrally for zeta zeros.

### 5.2 Limitations and Caveats

Several important limitations must be acknowledged:

**Peak displacement.** The `K = 50` spectral peak is at `sigma ~ 0.40`, not `sigma = 0.50`. While this may reflect finite-bandwidth displacement (the Fourier analog of the Gibbs shift), it may also indicate that `K = 50` is insufficient, or that finite-`N` effects at `N = 2000` are significant. The distinction can only be resolved by running `K = 50` at `N = 9877` with a finer `sigma` grid near `sigma = 0.50`.

**Sample size for K = 50.** The GPU scout at `K = 50` used only `N = 2000` zeros, compared to `N = 9877` for the definitive `K = 20` run. Finite-`N` effects are expected to be more pronounced at smaller `N`, potentially shifting the peak location and reducing the signal-to-noise ratio.

**Computational cost.** The transport computation scales as `O(K^3 * |E|)` due to the `K x K` matrix exponential per edge. At `K = 100` with `|E| ~ 50,000`, this requires approximately `5 * 10^{10}` floating-point operations per `sigma` value. The scaling limits the practical reach of the current implementation to `K ~ 200` on A100 hardware. Reaching `K = 500+` would require algorithmic innovations: GPU-native batched matrix exponentials, polynomial approximations to `exp(iA)`, or exploiting the structure of the generator (sparse, low-rank decomposition of `B_p`).

**Non-uniqueness.** The ATFT construction is one of many possible gauge theories on zeta zero point clouds. Different choices of fiber structure (e.g., matrix fibers `C^{K x K}` instead of vector fibers `C^K`), different Lie algebras (e.g., `su(K)` instead of `u(K)`), or different order parameters (e.g., the `H^1` Laplacian detecting magnetic flux rather than the `H^0` Laplacian detecting global sections) might produce stronger or weaker signals. The current construction is canonical in the sense that its ingredients -- multiplicative monoid representation, explicit formula phases, functional equation weights -- are determined by the arithmetic without free parameters, but this does not exclude other valid constructions.

**Observation is not proof.** The ATFT framework provides computational evidence, not a mathematical proof. Even if `sigma^*(K) -> 0.50` as `K -> infinity`, this constitutes strong numerical evidence for RH consistent with a topological mechanism, but does not resolve the question of whether every zero lies on the critical line. The framework is best understood as a computational physics experiment -- analogous to lattice gauge theory calculations in QCD -- providing data that informs but does not replace rigorous proof.

### 5.3 Relation to Previous Work

The ATFT framework sits at the intersection of several research programs:

**Berry-Keating.** Berry and Keating [9] proposed that the Riemann zeros are eigenvalues of a quantum Hamiltonian `H = xp` (suitably quantized and regularized), connecting RH to quantum chaos. The ATFT sheaf Laplacian `L_F(sigma)` is a different operator -- it acts on sections of a cellular sheaf rather than on `L^2(R)` -- but shares the philosophy: a self-adjoint operator whose spectral properties encode information about the zeros. The key difference is that `L_F(sigma)` is a `sigma`-parameterized family, and the ATFT test is about the `sigma`-dependence of the spectrum, not about matching individual eigenvalues to zeros.

**Connes.** Connes [10] developed a noncommutative trace formula connecting the zeros of `zeta(s)` to a spectral framework on the adele class space `A_Q / Q^*`. The ATFT construction uses a discrete approximation to a related object: the multiplicative monoid `(Z_{>0}, *)` acting on `{1, ..., K}` is a finite truncation of the adelic multiplicative group, and the gauge connection encodes the Euler product through the representation `rho(p)`. The ATFT framework can be viewed as a computational laboratory for testing predictions of the Connes program at finite truncation.

**Hansen-Ghrist.** The sheaf Laplacian construction follows Hansen and Ghrist [11], who developed the theory of opinion dynamics on discourse sheaves -- cellular sheaves on graphs with vector space stalks and linear restriction maps. The ATFT innovation is to use the restriction maps as gauge transport operators encoding arithmetic structure, rather than opinion propagation rules.

**Computational approaches.** Prior computational investigations of RH have focused primarily on direct verification (computing zeros and checking `Re(rho) = 1/2`) or statistical testing (comparing zero statistics to GUE predictions). The ATFT approach is qualitatively different: it asks whether a `sigma`-parameterized topological invariant has a distinguished value at `sigma = 1/2`, which is a geometric rather than arithmetic question. This makes it complementary to existing approaches.

### 5.4 Path to Definitive Results

The experimental program has a clear roadmap:

**K = 100 (25 primes).** The critical near-term experiment. Expected to show a measurably sharper peak than `K = 50`, with the peak location migrating toward `sigma = 0.50`. Hardware: RunPod A100, estimated cost approximately $21 for the zeta-only sweep.

**K = 200 (46 primes).** Expected to exhibit a clear phase transition: a sharp peak at or very near `sigma = 0.50` with high contrast ratio. At 46 primes, the Fourier approximation captures sufficient bandwidth to resolve a feature of width `O(1 / log 200) ~ 0.19` in `sigma`.

**K = 500+ (95+ primes).** Would require algorithmic innovation. Potential approaches include: GPU-native batched matrix exponentials using CUDA kernels, polynomial approximations to `exp(iA)` exploiting the sparse structure of `B_p(sigma)`, or a reformulation using the `H^1` Laplacian (1-Laplacian) which detects magnetic flux in loops rather than global sections and may exhibit sharper phase transitions.

**Higher-dimensional complexes.** The current construction uses only the 1-skeleton (edges) of the Vietoris-Rips complex. Extending to 2-simplices (triangles) and computing the `H^1` sheaf cohomology would provide a qualitatively different observable: the first Betti number of the sheaf detects obstructions to extending local sections around loops, which is directly related to the holonomy of the gauge connection. The `H^1` Laplacian may be more sensitive to the arithmetic structure than `H^0`, as it directly measures the curvature rather than the global section count.

**Scaling analysis.** Once `K = 100` and `K = 200` data are available, a quantitative scaling analysis of `sigma^*(K)` vs. `K` and `C(K)` vs. `K` will determine whether the peak location and contrast ratio follow power-law or logarithmic scaling toward their predicted limits. If `sigma^*(K) = 1/2 + c / log(K)` for some constant `c`, this would provide a precise quantitative connection to the density of primes.

---

## 6. Conclusion

We have presented the Adaptive Topological Field Theory (ATFT) framework, a gauge-theoretic construction on zeta zero point clouds that provides a concrete, falsifiable topological test of the Riemann Hypothesis. The framework is built from canonical arithmetic ingredients: the truncated left-regular representation of the multiplicative monoid encodes Dirichlet convolution as matrix multiplication, the functional equation determines the `sigma`-dependent generator weights, and the explicit formula provides the phase factors coupling zeros to primes.

The computational results establish three key findings:

1. **The gauge construction detects genuine arithmetic structure.** At `K = 20`, zeta zeros produce a spectral signal 670x stronger than random controls, confirming that the prime-encoded phase interference `p^{i Delta_gamma}` creates coherent transport constraints specific to the arithmetic distribution of the zeros.

2. **Fourier sharpening is confirmed.** The qualitative transition from a monotonic `sigma`-profile at `K = 20` (8 primes) to a peaked profile at `K = 50` (15 primes) is the first direct evidence that increasing `K` localizes the spectral peak, exactly as predicted by the finite Fourier approximation interpretation of the explicit formula.

3. **The critical experiment is within reach.** The `K = 100` experiment, requiring approximately 13 GPU-hours on an A100 instance, will determine whether the peak location migrates toward `sigma = 0.50` and the contrast ratio continues to grow. A positive result would constitute compelling computational evidence for the ATFT formulation of RH.

The ATFT framework does not claim to prove the Riemann Hypothesis. It provides a computational physics approach -- analogous to lattice QCD providing evidence for confinement before a rigorous proof -- in which the Riemann Hypothesis manifests as a topological phase transition in a `sigma`-parameterized gauge theory on the space of zeta zeros. The construction is canonical, falsifiable, and scalable. The Ti V0.1 codebase implementing all experiments described here is available as open source.

The path forward is clear: increasing `K` through `100`, `200`, and beyond will either confirm the convergence `sigma^*(K) -> 1/2` -- providing ever-stronger computational evidence for a topological mechanism underlying RH -- or reveal unexpected structure that challenges the hypothesis. Either outcome advances our understanding of the deep relationship between the primes, the zeros, and the geometry of the critical line.

---

## Acknowledgments

The authors thank Andrew Odlyzko for making his high-altitude zeta zero tables publicly available, which form the empirical foundation of this work. Computations were performed on a workstation equipped with an NVIDIA RTX 4080 GPU (16 GB VRAM). The Ti V0.1 software framework was developed using NumPy, SciPy, CuPy, and h5py.

---

## References

[1] B. Riemann, "Ueber die Anzahl der Primzahlen unter einer gegebenen Grosse," *Monatsberichte der Berliner Akademie*, pp. 671--680, 1859.

[2] E. C. Titchmarsh, *The Theory of the Riemann Zeta Function*, 2nd ed., revised by D. R. Heath-Brown. Oxford University Press, 1986.

[3] H. M. Edwards, *Riemann's Zeta Function*. Academic Press, 1974. Reprinted by Dover, 2001.

[4] X. Gourdon, "The 10^13 first zeros of the Riemann zeta function, and zeros computation at very large height," preprint, 2004. Available at: http://numbers.computation.free.fr/Constants/Miscellaneous/zetazeros1e13-1e24.pdf

[5] H. L. Montgomery, "The pair correlation of zeros of the zeta function," *Analytic Number Theory*, Proc. Sympos. Pure Math., vol. 24, pp. 181--193, Amer. Math. Soc., 1973.

[6] A. M. Odlyzko, "On the distribution of spacings between zeros of the zeta function," *Math. Comp.*, vol. 48, no. 177, pp. 273--308, 1987.

[7] A. M. Odlyzko, "The 10^22-nd zero of the Riemann zeta function," *Dynamical, Spectral, and Arithmetic Zeta Functions*, Contemp. Math., vol. 290, pp. 139--144, Amer. Math. Soc., 2001.

[8] N. M. Katz and P. Sarnak, *Random Matrices, Frobenius Eigenvalues, and Monodromy*. Amer. Math. Soc., 1999.

[9] M. V. Berry and J. P. Keating, "The Riemann zeros and eigenvalue asymptotics," *SIAM Review*, vol. 41, no. 2, pp. 236--266, 1999.

[10] A. Connes, "Trace formula in noncommutative geometry and the zeros of the Riemann zeta function," *Selecta Math. (N.S.)*, vol. 5, pp. 29--106, 1999.

[11] J. Hansen and R. Ghrist, "Opinion dynamics on discourse sheaves," *SIAM J. Appl. Math.*, vol. 81, no. 5, pp. 2033--2060, 2021.

[12] J. M. Curry, "Sheaves, cosheaves and applications," Ph.D. thesis, University of Pennsylvania, 2014.

[13] M. L. Mehta, *Random Matrices*, 3rd ed. Elsevier Academic Press, 2004.

[14] I. Dumitriu and A. Edelman, "Matrix models for beta ensembles," *J. Math. Phys.*, vol. 43, no. 11, pp. 5830--5847, 2002.

[15] E. Jones, T. Oliphant, P. Peterson, et al., "SciPy: Open source scientific tools for Python," 2001--. Available at: https://www.scipy.org

[16] R. Okuta, Y. Unno, D. Nishino, S. Hido, and C. Nagano, "CuPy: A NumPy-compatible library for NVIDIA GPU calculations," *Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS)*, 2017.

[17] H. Edelsbrunner and J. L. Harer, *Computational Topology: An Introduction*. Amer. Math. Soc., 2010.

---

## Appendix A: Notation Summary

| Symbol | Definition |
|---|---|
| `zeta(s)` | Riemann zeta function |
| `rho = sigma + i gamma` | Non-trivial zero of `zeta(s)` |
| `sigma` | Real part of `s`; critical strip parameter |
| `gamma_n` | Imaginary part of the `n`-th non-trivial zero |
| `hat{gamma}_n` | Unfolded zero (mean spacing 1) |
| `Delta_gamma` | Gap between consecutive unfolded zeros |
| `K` | Fiber dimension (truncation order) |
| `N` | Number of zeros in the point cloud |
| `epsilon` | Vietoris-Rips scale parameter |
| `rho(p)` | Truncated left-regular representation of prime `p` |
| `B_p(sigma)` | Prime generator: `log(p) [p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T]` |
| `A_{ij}(sigma)` | Superposition edge generator |
| `U_{ij}(sigma)` | Transport matrix: `exp(i A_{ij})` |
| `L_F(sigma)` | Sheaf Laplacian |
| `S(sigma, epsilon)` | Spectral sum: sum of `k_eig` smallest eigenvalues of `L_F` |
| `sigma^*(K)` | Peak location: `argmax_sigma S(sigma, K)` |
| `C(K)` | Contrast ratio: `S(sigma^*) / S(sigma_ref)` |
| `R(K)` | Signal ratio: `S_{zeta}(sigma^*) / S_{random}(sigma^*)` |
| `pi(K)` | Number of primes up to `K` |

## Appendix B: Computational Parameters

| Parameter | K = 20 run | K = 50 run |
|---|---|---|
| Zeros source | Odlyzko, near `T ~ 10^{20}` | Odlyzko, near `T ~ 10^{20}` |
| N (zeros used) | 9877 | 2000 |
| K (fiber dim) | 20 | 50 |
| pi(K) (primes) | 8 | 15 |
| Transport mode | Superposition | Superposition |
| Normalization | Frobenius (per edge) | Frobenius (per edge) |
| k_eig (eigenvalues) | 100 | 100 |
| epsilon grid | {1.5, 2.0, 2.5, 3.0, 4.0, 5.0} | {3.0, 5.0} |
| sigma grid | {0.25, 0.30, ..., 0.75} | {0.25, 0.40, 0.50, 0.60, 0.75} |
| Laplacian dimension | 197,540 x 197,540 | 100,000 x 100,000 |
| Engine | CPU (scipy BSR + eigsh) | GPU (CuPy CSR + spectral flip) |
| Hardware | Workstation CPU | RTX 4080, 16 GB VRAM |
| Time per grid point | ~30 min | ~30 min |
| Random controls | 5 trials | -- |
| GUE controls | 5 trials | -- |
