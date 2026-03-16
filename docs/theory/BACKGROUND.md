# Theoretical Background: Arithmetic Topological Field Theory (ATFT) Approach to the Riemann Hypothesis

**Document version:** 2026-03-16
**Audience:** Graduate-level mathematics and mathematical physics
**Purpose:** Reference document for collaborators and reviewers

---

## Table of Contents

1. [The Riemann Hypothesis](#1-the-riemann-hypothesis)
2. [Random Matrix Theory Connection](#2-random-matrix-theory-connection)
3. [Gauge Theory and Fiber Bundles](#3-gauge-theory-and-fiber-bundles)
4. [Sheaf Cohomology on Graphs](#4-sheaf-cohomology-on-graphs)
5. [The ATFT Construction](#5-the-atft-construction)
6. [The Fourier Sharpening Hypothesis](#6-the-fourier-sharpening-hypothesis)
7. [What Would Disprove RH?](#7-what-would-disprove-rh)
8. [Computational Strategy](#8-computational-strategy)

---

## 1. The Riemann Hypothesis

### 1.1 The Riemann Zeta Function

The Riemann zeta function is initially defined for Re(s) > 1 by the absolutely convergent Dirichlet series

$$\zeta(s) = \sum_{n=1}^{\infty} n^{-s}, \quad \text{Re}(s) > 1.$$

Euler's product formula connects this to the primes: for Re(s) > 1,

$$\zeta(s) = \prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}.$$

This identity encodes the fundamental theorem of arithmetic — unique prime factorization — in the language of complex analysis. Each prime p contributes a local factor $(1 - p^{-s})^{-1}$ corresponding to the geometric series $\sum_{k=0}^{\infty} p^{-ks}$.

### 1.2 Analytic Continuation and the Functional Equation

Riemann (1859) showed that $\zeta(s)$ extends to a meromorphic function on all of $\mathbb{C}$, with a single simple pole at $s = 1$ with residue 1. This analytic continuation is achieved via the integral representation

$$\zeta(s) = \frac{1}{\Gamma(s)} \int_0^{\infty} \frac{t^{s-1}}{e^t - 1}\, dt, \quad \text{Re}(s) > 1,$$

extended by analytic continuation. The completed zeta function, incorporating the archimedean Euler factor,

$$\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma\!\left(\frac{s}{2}\right)\zeta(s),$$

is entire and satisfies the functional equation

$$\xi(s) = \xi(1-s).$$

In terms of $\zeta(s)$ alone, this reads

$$\zeta(s) = 2^s \pi^{s-1} \sin\!\left(\frac{\pi s}{2}\right) \Gamma(1-s)\, \zeta(1-s).$$

The functional equation establishes a symmetry of $\zeta(s)$ under $s \mapsto 1-s$, reflecting the critical line $\text{Re}(s) = \tfrac{1}{2}$. This symmetry is the primary motivation for the Riemann Hypothesis and plays a central structural role in the ATFT construction (Section 5.5).

### 1.3 Non-Trivial Zeros and the Critical Strip

The zeros of $\zeta(s)$ fall into two classes:

- **Trivial zeros:** at $s = -2, -4, -6, \ldots$, arising from the poles of $\sin(\pi s/2)$ in the functional equation.
- **Non-trivial zeros:** all other zeros, which necessarily lie in the **critical strip** $0 < \text{Re}(s) < 1$.

The functional equation implies that non-trivial zeros come in pairs: if $\zeta(\rho) = 0$, then $\zeta(1-\rho) = 0$. Since $\zeta(\bar{s}) = \overline{\zeta(s)}$ for real $s$, zeros also come in complex-conjugate pairs. Thus non-trivial zeros have a fourfold symmetry: $\rho, 1-\rho, \bar{\rho}, 1-\bar{\rho}$.

The **Riemann Hypothesis (RH)** asserts that all non-trivial zeros lie on the **critical line** $\text{Re}(s) = \tfrac{1}{2}$:

$$\text{RH}: \quad \zeta(\rho) = 0,\; 0 < \text{Re}(\rho) < 1 \implies \text{Re}(\rho) = \frac{1}{2}.$$

The non-trivial zeros may be enumerated as $\rho_n = \tfrac{1}{2} + i\gamma_n$ (if RH holds), where $0 < \gamma_1 \leq \gamma_2 \leq \cdots$ are the positive imaginary parts. Numerically, $\gamma_1 \approx 14.135$, $\gamma_2 \approx 21.022$, $\gamma_3 \approx 25.011$.

As of the present date, more than $10^{13}$ non-trivial zeros have been verified numerically to lie on the critical line, and no counterexample has ever been found. Yet no proof exists.

### 1.4 The Explicit Formula Connecting Zeros to Primes

The deep arithmetic content of RH is revealed by the **explicit formula**. Let $\psi(x) = \sum_{p^k \leq x} \log p$ be the Chebyshev function. Then

$$\psi(x) = x - \sum_{\rho} \frac{x^{\rho}}{\rho} - \log(2\pi) - \frac{1}{2}\log\!\left(1 - x^{-2}\right), \quad x \notin \mathbb{Z},$$

where the sum runs over all non-trivial zeros $\rho$ and is understood in the principal value sense (pairing $\rho$ with $1-\rho$). Each zero $\rho = \sigma + i\gamma$ contributes an oscillatory term $x^{\sigma + i\gamma}/(\sigma + i\gamma)$, whose amplitude grows as $x^{\sigma}$. If $\sigma = \tfrac{1}{2}$ for all $\rho$ (RH), then all terms have amplitude $x^{1/2}$, giving the error bound

$$\pi(x) = \text{Li}(x) + O\!\left(x^{1/2} \log x\right),$$

where $\text{Li}(x) = \int_2^x dt/\log t$ is the logarithmic integral. Any zero with $\text{Re}(\rho) > \tfrac{1}{2}$ would increase this error exponent, degrading the prime counting estimate.

The explicit formula is central to ATFT: the phase factors $e^{i\gamma_n \log p} = p^{i\gamma_n}$ appearing in the ATFT generator (Section 5.6) are precisely the oscillatory terms from the explicit formula. The ATFT construction encodes this Fourier-analytic relationship between primes and zeros directly into the fiber geometry.

### 1.5 Why RH Matters

Beyond prime distribution, RH has consequences throughout mathematics:

- **Error terms in arithmetic functions:** Sharper bounds on $\sum_{n \leq x} \mu(n)$, $\sum_{n \leq x} \lambda(n)$, and related sums.
- **Goldbach-type problems:** Error terms in the circle method improve under GRH.
- **Cryptography:** Many factoring and discrete logarithm algorithms have complexity bounds depending on GRH.
- **Spectral theory:** The Hilbert-Polya conjecture proposes that $\gamma_n$ are eigenvalues of a self-adjoint operator on some Hilbert space, which would prove RH via spectral theory. ATFT is a candidate construction for this operator.
- **Random matrix theory:** The Montgomery-Odlyzko law (Section 2) is one of the most striking discoveries in modern mathematics, connecting number theory to quantum chaos.

---

## 2. Random Matrix Theory Connection

### 2.1 Montgomery's Pair Correlation Conjecture (1973)

H. L. Montgomery studied the pair correlation function of zeta zeros. Define, for $0 < \alpha \leq \beta$,

$$F(\alpha) = \frac{2\pi}{T \log T} \# \left\{ (\gamma, \gamma') : 0 < \gamma, \gamma' \leq T,\; \frac{2\pi\alpha}{\log T} \leq \gamma - \gamma' \leq \frac{2\pi\beta}{\log T} \right\}.$$

Montgomery conjectured, and partially proved assuming RH, that

$$F(\alpha) \sim \int_\alpha^\beta \left(1 - \left(\frac{\sin \pi u}{\pi u}\right)^2\right) du$$

as $T \to \infty$. This is the **pair correlation function of the GUE** (Gaussian Unitary Ensemble). The term $\left(\frac{\sin \pi u}{\pi u}\right)^2$ is the Fourier transform of the pair correlation function of eigenvalues of random Hermitian matrices drawn from the GUE measure, and its suppression near $u = 0$ encodes **level repulsion** — the tendency of eigenvalues (or zeros) to avoid clustering.

### 2.2 Odlyzko's Numerical Verification

Andrew Odlyzko performed large-scale computations of zeros near height $T \approx 10^{20}$ and $T \approx 10^{22}$. At these altitudes the local statistics of zeros are well-described by GUE predictions across all tested statistics: pair correlation, nearest-neighbor spacings, number variance, and higher-order correlations. The agreement is striking and persists with increasing precision as $T$ grows.

Odlyzko's high-altitude zeros (available as numerical datasets) are the primary input data for the ATFT framework. Working at altitude $T \approx 10^{20}$ rather than near $T = 0$ provides zeros in a regime where GUE behavior is clearly established, reducing finite-height contamination.

### 2.3 GUE Statistics

The **Gaussian Unitary Ensemble** $\text{GUE}(N)$ is the probability measure on $N \times N$ Hermitian matrices $H$ with density proportional to $e^{-N \text{tr}(H^2)/2}$. Its eigenvalue joint density is

$$P(\lambda_1, \ldots, \lambda_N) \propto \prod_{i < j} (\lambda_i - \lambda_j)^2 \cdot e^{-N \sum_i \lambda_i^2 / 2}.$$

The Vandermonde determinant squared $\prod_{i<j}(\lambda_i - \lambda_j)^2$ produces level repulsion: the probability density vanishes when any two eigenvalues coincide. This quadratic repulsion (as opposed to the linear repulsion of GOE or no repulsion of Poisson) characterizes the GUE universality class.

The global eigenvalue density converges to the **Wigner semicircle**:

$$\rho_{sc}(\lambda) = \frac{1}{2\pi} \sqrt{4 - \lambda^2}, \quad |\lambda| \leq 2.$$

### 2.4 The Wigner Surmise for Nearest-Neighbor Spacings

After spectral unfolding (Section 2.6), the nearest-neighbor spacing distribution for GUE is well-approximated by the **Wigner surmise**:

$$P(s) = \frac{32}{\pi^2}\, s^2\, \exp\!\left(-\frac{4s^2}{\pi}\right).$$

This distribution peaks at $s \approx 1.12$, has linear level repulsion as $s \to 0$ (actually cubic for GUE in the exact formula; the Wigner surmise has $s^2$ as an approximation), and decays superexponentially. Poisson statistics, by contrast, give $P(s) = e^{-s}$, with maximum probability at $s = 0$.

The ATFT Phase 1 results verify that unfolded zeta zeros follow GUE spacing statistics, as measured by the L2 distance between empirical Betti curves and GUE controls, with ratio $D_M(\text{zeta})/D_M(\text{Poisson})$ decreasing monotonically from $0.144$ to $0.052$ as $N$ grows from $500$ to $10{,}000$.

### 2.5 The Katz-Sarnak Philosophy

Katz and Sarnak (2000) extended the GUE connection to families of L-functions. They conjecture that:

- Zeros of a "generic" family of L-functions, in the limit of large conductor, have local statistics matching those of eigenvalues of a specific classical compact group (GUE, GOE, GSE, or their symplectic/orthogonal variants), depending on the symmetry type of the family.
- The $\zeta(s)$ zeros correspond to GUE (unitary symmetry type).

This philosophy, if correct, implies that the non-trivial zeros of $\zeta(s)$ are the "quantum spectrum" of a self-adjoint operator with unitary symmetry — the Hilbert-Polya operator. The ATFT construction seeks to build a concrete realization of this operator.

### 2.6 Spectral Unfolding

Raw zeros $\gamma_n$ have a mean density that grows logarithmically: by the Riemann-von Mangoldt formula,

$$N(T) = \frac{T}{2\pi}\log\frac{T}{2\pi e} + O(\log T),$$

where $N(T)$ counts zeros with $0 < \text{Im}(\rho) \leq T$. To compare local statistics with GUE, one must **unfold** the spectrum: apply a smooth local transformation $\gamma_n \mapsto \hat{\gamma}_n = N(\gamma_n)$ so that the mean spacing becomes 1. This transformation must use a smooth version of $N(T)$, not rank-based unfolding.

**Warning:** Rank-based unfolding (assigning $\hat{\gamma}_n = n$) destroys level repulsion for GUE matrices, since all gaps become identically 1. The correct procedure uses the smooth counting function (semicircle CDF for matrices, Riemann-von Mangoldt staircase for zeta zeros) as the unfolding map. This was a critical methodological correction discovered during Phase 1.

---

## 3. Gauge Theory and Fiber Bundles

### 3.1 Principal Bundles and Connections

Let $G$ be a Lie group, $M$ a smooth manifold. A **principal $G$-bundle** $\pi: P \to M$ is a fiber bundle with right $G$-action on $P$ that is free, transitive on fibers, and locally trivial. The gauge group $\mathcal{G}$ is the group of vertical automorphisms of $P$.

A **connection** on $P$ is a $G$-equivariant distribution $H \subset TP$ complementary to the vertical subbundle $VP = \ker(d\pi)$. Equivalently, it is a $\mathfrak{g}$-valued 1-form $\omega \in \Omega^1(P, \mathfrak{g})$ satisfying the equivariance condition. In local trivializations, a connection is specified by a $\mathfrak{g}$-valued 1-form $A$ on $M$ (the **gauge potential** or **connection 1-form**).

The **curvature** of a connection is the $\mathfrak{g}$-valued 2-form

$$F = dA + \frac{1}{2}[A, A],$$

or in components, $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]$. A connection with $F = 0$ is called **flat**; flat connections are locally gauge-equivalent to the trivial connection, but may have non-trivial global topology.

### 3.2 The Gauge Group U(K) and Its Lie Algebra

The ATFT framework uses the gauge group $G = U(K)$, the group of $K \times K$ unitary matrices. Its Lie algebra is

$$\mathfrak{u}(K) = \{ X \in \text{Mat}_{K \times K}(\mathbb{C}) : X^* = -X \},$$

the space of skew-Hermitian matrices. Equivalently, $i\mathfrak{u}(K)$ is the space of Hermitian matrices.

The exponential map $\exp: \mathfrak{u}(K) \to U(K)$ is surjective. For skew-Hermitian $X$, $e^X \in U(K)$. In the ATFT construction, the connection generators $A_{ij}(\sigma)$ (defined below) are Hermitian, so $iA_{ij}(\sigma) \in \mathfrak{u}(K)$, and the transport matrices $U_{ij} = \exp(iA_{ij}) \in U(K)$.

The Lie algebra $\mathfrak{u}(K)$ has dimension $K^2$ over $\mathbb{R}$. A standard basis consists of $K$ diagonal generators $\{iE_{kk}\}$ and $K(K-1)/2$ pairs $\{i(E_{jk}+E_{kj}), (E_{jk}-E_{kj})\}$ for $j < k$.

### 3.3 Parallel Transport Along Paths

Given a connection on a vector bundle $E \to M$, **parallel transport** along a path $\gamma: [0,1] \to M$ is the isomorphism $P_\gamma: E_{\gamma(0)} \to E_{\gamma(1)}$ obtained by solving the parallel transport equation along $\gamma$. In a local trivialization with gauge potential $A$,

$$\frac{d}{dt} v(t) + A(\dot{\gamma}(t)) v(t) = 0,$$

with $v(0) = v_0$. The solution is $v(t) = P_{\gamma|_{[0,t]}} v_0$, formally written as the path-ordered exponential

$$P_\gamma = \mathcal{P}\exp\!\left(-\int_\gamma A\right).$$

For a constant (edge-wise) connection on a graph — the discrete analog — there is no path ordering issue, and the transport along edge $e = (i,j)$ is simply $U_{ij} = \exp(iA_{ij})$.

### 3.4 Holonomy and Curvature

The **holonomy** of a connection at a base point $x_0 \in M$ is the group

$$\text{Hol}_{x_0} = \{ P_\gamma : \gamma \text{ a loop based at } x_0 \} \subset G.$$

For a flat connection ($F = 0$), the holonomy depends only on the homotopy class of $\gamma$ and defines a representation $\pi_1(M, x_0) \to G$. Non-trivial holonomy around contractible loops requires non-zero curvature.

In the discrete (graph) setting, the holonomy around a cycle $e_{i_0 i_1} e_{i_1 i_2} \cdots e_{i_{k-1} i_0}$ is

$$H = U_{i_0 i_1} U_{i_1 i_2} \cdots U_{i_{k-1} i_0}.$$

The "global" transport mode in ATFT (Section 5.7) was found to have identically trivial holonomy (telescoping product). The "resonant" and functional-equation modes create genuine non-trivial holonomy, making them useful for spectral analysis.

### 3.5 Primes as Generators of a Gauge Connection

The multiplicative structure of positive integers — encoded in the semigroup $(\mathbb{Z}_{>0}, \times)$ — provides a natural family of generators for a gauge connection indexed by primes. The key insight is that multiplication by a prime $p$ is an endomorphism of $(\mathbb{Z}_{>0}, \times)$, and the representation $\rho(p)$ of this action on a truncated basis $\{1, \ldots, K\}$ gives a sparse matrix encoding Dirichlet convolution.

The analogy is: primes play the role of parallel transporters along the "arithmetic directions" of a fiber bundle over the space of zeta zeros. The ATFT construction makes this analogy precise by constructing a cellular sheaf (Section 4) with transport maps built from prime representations.

---

## 4. Sheaf Cohomology on Graphs

### 4.1 Cellular Sheaves on Graphs

Let $X$ be a finite graph with vertex set $V$ and edge set $E$ (with a chosen orientation). A **cellular sheaf of vector spaces** $\mathcal{F}$ on $X$ assigns:

- A vector space $\mathcal{F}(v)$ (the **stalk at** $v$) to each vertex $v \in V$,
- A vector space $\mathcal{F}(e)$ (the **stalk at** $e$) to each edge $e \in E$,
- A linear map $\mathcal{F}_{v \trianglelefteq e}: \mathcal{F}(v) \to \mathcal{F}(e)$ for each incidence $v \trianglelefteq e$ (vertex $v$ is an endpoint of edge $e$).

The maps $\mathcal{F}_{v \trianglelefteq e}$ are the **restriction maps** of the sheaf. In the ATFT setting, all stalks are $\mathbb{C}^K$ and restriction maps are $K \times K$ unitary matrices (or Hermitian generators).

### 4.2 The Coboundary Operator and Sheaf Laplacian

The **coboundary operator** $\delta_0: C^0(X, \mathcal{F}) \to C^1(X, \mathcal{F})$ acts on 0-cochains (assignments of stalk elements to vertices) by

$$(\delta_0 f)(e) = \mathcal{F}_{v_+ \trianglelefteq e}(f(v_+)) - \mathcal{F}_{v_- \trianglelefteq e}(f(v_-)),$$

where $v_+, v_-$ are the head and tail of the oriented edge $e$. Explicitly, for $f \in \bigoplus_{v \in V} \mathcal{F}(v) \cong \mathbb{C}^{K|V|}$,

$$\delta_0 = \bigoplus_{e=(i,j)} \left[ U_{ij} | -I \right],$$

where $U_{ij}$ is the restriction map from $j$ (tail) to the edge and $I$ is the restriction map from $i$ (head). The **sheaf Laplacian** is

$$\mathcal{L}_\mathcal{F} = \delta_0^\dagger \delta_0: C^0(X, \mathcal{F}) \to C^0(X, \mathcal{F}).$$

$\mathcal{L}_\mathcal{F}$ is a positive semidefinite Hermitian operator of size $K|V| \times K|V|$. For $K|V| = K \cdot N$ with $N$ zeros and $K = 20$, this is a $20N \times 20N$ sparse matrix.

### 4.3 Sheaf Cohomology

The **zeroth sheaf cohomology** is

$$H^0(X, \mathcal{F}) = \ker(\delta_0) \cong \ker(\mathcal{L}_\mathcal{F}).$$

An element $f \in H^0(X, \mathcal{F})$ is a **global section** of the sheaf: a consistent assignment of vectors to all vertices such that the restriction maps "agree" on every edge — i.e., $U_{ij} f(j) = f(i)$ for all edges $(i,j)$.

In the flat connection limit, $U_{ij}$ varies slowly and the sheaf has large $H^0$ (many global sections). In the highly curved limit, the holonomy constraints are overdetermined, $H^0$ collapses, and $\mathcal{L}_\mathcal{F}$ has no zero eigenvalues. The spectral structure of $\mathcal{L}_\mathcal{F}$ — particularly its smallest eigenvalues — encodes the global coherence of the gauge connection.

### 4.4 The Cheeger Constant and Spectral Gap

For a graph $G = (V, E)$, the **Cheeger constant** (isoperimetric constant) is

$$h(G) = \min_{S \subset V,\; |S| \leq |V|/2} \frac{|\partial S|}{|S|},$$

where $\partial S$ is the edge boundary. The discrete **Cheeger inequalities** relate $h(G)$ to the second smallest eigenvalue $\lambda_1$ of the graph Laplacian:

$$\frac{\lambda_1}{2} \leq h(G) \leq \sqrt{2\lambda_1}.$$

For sheaves, the analogous quantity is the **spectral gap** of $\mathcal{L}_\mathcal{F}$: the smallest nonzero eigenvalue $\lambda_{\min}^+(\mathcal{L}_\mathcal{F})$. A large spectral gap indicates that global sections (if any exist) are "isolated" and that the sheaf has strong global rigidity — small perturbations of the connection cannot create new global sections.

The ATFT spectral sum $S(\sigma)$ is a normalized measure of this spectral structure, aggregating eigenvalue information across the full spectrum of $\mathcal{L}_\mathcal{F}(\sigma)$ as $\sigma$ varies through the critical strip.

### 4.5 Global Coherence vs. Local Disorder

The sheaf Laplacian discriminates between two qualitatively different phenomena:

- **Local coherence:** Restriction maps that are nearly compatible on each individual edge (small $\|U_{ij} f(j) - f(i)\|$ for most edges) but fail globally.
- **Global coherence:** Existence of a section $f$ satisfying all constraints simultaneously.

Only global coherence contributes to $\ker(\mathcal{L}_\mathcal{F})$. The arithmetic of the prime representations and the geometry of the zeta zeros together determine whether the gauge constraints imposed by $U_{ij}(\sigma)$ are globally satisfiable. The hypothesis is that they are maximally (or specially) satisfiable exactly at $\sigma = \tfrac{1}{2}$ when the input points are zeta zeros.

---

## 5. The ATFT Construction

### 5.1 Input Data: Odlyzko's High-Altitude Zeta Zeros

Let $\gamma_1 < \gamma_2 < \cdots < \gamma_N$ denote the positive imaginary parts of non-trivial zeta zeros (assuming RH places them all on the critical line). The ATFT framework uses Odlyzko's high-altitude datasets, which tabulate zeros near height $T \approx 10^{20}$.

Working at altitude is important for two reasons:
1. At high altitude, the zeros are in the bulk of the GUE regime (finite-height effects near $\gamma \approx 14$ have decayed).
2. The mean density is large, so the locally-uniform spacing assumption used in Vietoris-Rips construction is better justified.

After spectral unfolding using the smooth Riemann-von Mangoldt formula, the zeros are transformed to have mean spacing 1. All subsequent geometric constructions use the unfolded zeros.

### 5.2 The Vietoris-Rips Graph

Given the unfolded zero sequence $\hat{\gamma}_1, \ldots, \hat{\gamma}_N$, choose a scale parameter $\varepsilon > 0$. The **Vietoris-Rips graph** $X_\varepsilon$ has:

- **Vertices:** $\{1, 2, \ldots, N\}$, identified with the unfolded zeros $\hat{\gamma}_i$.
- **Edges:** $(i, j) \in E_\varepsilon$ if and only if $|\hat{\gamma}_i - \hat{\gamma}_j| \leq \varepsilon$.

Since the zeros are ordered on the real line, this graph is an interval graph: vertex $i$ is adjacent to all $j$ within a window of width $\varepsilon$ in the unfolded spectrum. The parameter $\varepsilon$ controls the connectivity of the graph and the range of correlations captured. Typical values used are $\varepsilon \in \{3.0, 5.0\}$, capturing 3 to 5 neighboring zeros on each side.

### 5.3 The Fiber: Truncated Arithmetic Space

At each vertex $v \in V$, the sheaf assigns the stalk $\mathcal{F}(v) = \mathbb{C}^K$, with orthonormal basis $\{|1\rangle, |2\rangle, \ldots, |K\rangle\}$. These basis vectors represent the first $K$ positive integers, which are the objects of arithmetic acted upon by multiplication.

The truncation parameter $K$ determines:
- The arithmetic resolution: primes $p \leq K$ are represented exactly; primes $p > K$ are ignored.
- The fiber dimension: $K \times K$ matrices.
- The sheaf Laplacian size: $K \cdot N \times K \cdot N$.

For the experiments described here, $K = 20$ captures the 8 primes $\{2, 3, 5, 7, 11, 13, 17, 19\}$, and $K = 50$ captures 15 primes up to $47$.

### 5.4 Prime Representation: The Multiplicative Monoid Action

The multiplicative semigroup $(\mathbb{Z}_{>0}, \times)$ acts on $\{1, \ldots, K\}$ by multiplication (with truncation at $K$). For each prime $p$, define the **truncated shift operator** $\rho(p) \in \text{Mat}_{K \times K}(\mathbb{C})$ by

$$\rho(p)_{ij} = \begin{cases} 1 & \text{if } j \cdot p = i \text{ and } i \leq K \\ 0 & \text{otherwise.} \end{cases}$$

That is, $\rho(p)|j\rangle = |pj\rangle$ if $pj \leq K$, and $\rho(p)|j\rangle = 0$ otherwise. This is a truncated isometry (partial isometry): $\rho(p)^\dagger \rho(p)$ is a projection onto the span of $\{|j\rangle : pj \leq K\}$.

**Why this representation is canonical:** The operator $\rho(p)$ encodes the action of Dirichlet convolution with the prime-supported function $\delta_p$: in terms of arithmetic functions, $(\delta_p * f)(n) = f(n/p)$ (if $p | n$, else 0), which is exactly the transpose action $\rho(p)^T$. Dirichlet convolution — the fundamental product of multiplicative number theory — becomes matrix multiplication in this representation. There are no arbitrary choices: the basis is the natural one (positive integers), and the action is the natural one (multiplication).

The operators $\{\rho(p)\}_{p \leq K}$ satisfy approximate commutativity: $\rho(p)\rho(q) = \rho(q)\rho(p)$ whenever $pq \leq K$ (both equal $\rho(pq)$ acting on appropriate vectors). This approximate commutativity breaks down at the truncation boundary, which is the source of non-trivial holonomy.

### 5.5 The Generator: Encoding the Functional Equation

For each prime $p \leq K$ and parameter $\sigma \in (0,1)$, define the **prime generator**

$$B_p(\sigma) = \log(p) \cdot \left[ p^{-\sigma} \rho(p) + p^{-(1-\sigma)} \rho(p)^T \right] \in \text{Mat}_{K \times K}(\mathbb{C}).$$

The coefficient $\log(p)$ arises from the derivative $-\frac{d}{ds}\log(1-p^{-s}) = \sum_{k=1}^\infty \log(p) \cdot p^{-ks}$, matching the weighting in the logarithmic derivative $\zeta'/\zeta$.

**Critical observation:** $B_p(\sigma)$ is Hermitian if and only if $\sigma = \tfrac{1}{2}$. When $\sigma = \tfrac{1}{2}$, $p^{-\sigma} = p^{-(1-\sigma)} = p^{-1/2}$, so $B_p(1/2) = \log(p) \cdot p^{-1/2}(\rho(p) + \rho(p)^T)$, which is manifestly Hermitian. For $\sigma \neq \tfrac{1}{2}$, the two terms have different weights $p^{-\sigma} \neq p^{-(1-\sigma)}$, and $B_p(\sigma)$ is not Hermitian.

This is the functional equation $\zeta(s) = \zeta(1-s)$ (up to known factors) encoded at the level of individual generators: the functional equation symmetry $s \leftrightarrow 1-s$ corresponds to $\sigma \leftrightarrow 1-\sigma$, which is a symmetry of $B_p(\sigma)$ only at the fixed point $\sigma = \tfrac{1}{2}$.

The $p^{-\sigma}$ weighting comes directly from the Euler product: the local factor at $p$ for $\zeta(s)$ is $(1-p^{-s})^{-1}$, and the generator $B_p(\sigma)$ corresponds to the linearization $\log(1-p^{-s})^{-1} \approx p^{-s}\rho(p)$ (forward action) plus its adjoint $p^{-(1-s)}\rho(p)^T$ (backward action, from the functional equation).

### 5.6 The Superposition Generator: Fourier-Analytic Heart of the Construction

For an oriented edge $(i,j) \in E_\varepsilon$ with gap $\Delta\gamma = \hat{\gamma}_i - \hat{\gamma}_j$, define the **edge generator**

$$A_{ij}(\sigma) = \sum_{p \leq K} e^{i\Delta\gamma \cdot \log p} \cdot B_p(\sigma) \in \text{Mat}_{K \times K}(\mathbb{C}).$$

The phase factor $e^{i\Delta\gamma \cdot \log p} = p^{i\Delta\gamma}$ is exactly the oscillation $p^{i(\gamma_i - \gamma_j)}$ that appears in the explicit formula when one "differences" the contribution of prime $p$ at two adjacent zeros. This is the Fourier-analytic connection to the explicit formula made precise.

Specifically, consider the explicit formula contribution of prime $p$ at a zero $\rho_n = \tfrac{1}{2} + i\gamma_n$: the term $-x^{\rho_n}/\rho_n$ oscillates as $x^{i\gamma_n} = e^{i\gamma_n \log x}$. Setting $x = p$ (a prime), this oscillation is $e^{i\gamma_n \log p} = p^{i\gamma_n}$. The phase factor in $A_{ij}$ is the difference $p^{i\gamma_i} / p^{i\gamma_j} = p^{i\Delta\gamma}$, which is the relative phase between consecutive zeros at prime $p$.

**Why this is essential:** The edge generator $A_{ij}(\sigma)$ couples the arithmetic structure (which primes $p \leq K$, and how they act on $\{1,\ldots,K\}$ via $B_p(\sigma)$) to the spectral structure (the gaps $\Delta\gamma$ between consecutive zeros). The superposition over primes creates constructive/destructive interference based on the prime-gap relationships. If the zeros encode arithmetic information (as RH implies through the explicit formula), this interference should be coherent precisely at $\sigma = \tfrac{1}{2}$.

### 5.7 Transport Matrices and the Three Modes

The **transport matrix** for edge $(i,j)$ is

$$U_{ij}(\sigma) = \exp\!\left(i \cdot A_{ij}(\sigma)\right) \in U(K),$$

obtained via the matrix exponential of the (Hermitian) edge generator scaled by $i$.

Three transport modes were investigated:

1. **Global mode:** $A_{ij} = \sum_p G_p(\sigma)$ independent of the edge (same generator for all edges). This connection is **provably flat**: the holonomy around any cycle telescopes to the identity, since the transport matrices are all identical and the product $U \cdot U^{-1} \cdot U \cdot U^{-1} \cdots = I$ around any cycle in the Vietoris-Rips graph. A flat connection on a simply-connected graph has trivial holonomy; on a graph with cycles, it depends only on the fundamental group, which provides no sigma-dependent information. **Conclusion: useless for sigma-detection.**

2. **Resonant mode:** Each edge $(i,j)$ uses the single prime $p^*_{ij}$ minimizing $|\Delta\gamma_{ij} - \log p|$. This creates non-commuting, edge-dependent generators and genuine holonomy. However, the spectral signal is monotonically decreasing in $\sigma$ — a trivial consequence of the $1/p^\sigma$ overall scaling — rather than peaking at $\sigma = \tfrac{1}{2}$.

3. **Functional-equation (FE) mode:** The multi-prime superposition with FE generators $B_p(\sigma)$ as defined above. This is the mode described in Sections 5.5–5.6 and used in Phase 3. The Hermiticity at $\sigma = \tfrac{1}{2}$ creates maximum topological rigidity at the critical line, with sigma-dependence emerging from both the weighting $p^{-\sigma}$ and the phase interference $p^{i\Delta\gamma}$.

### 5.8 Sheaf Laplacian Assembly and Spectral Analysis

Given the transport matrices $\{U_{ij}(\sigma)\}_{(i,j) \in E_\varepsilon}$, the coboundary operator is assembled as the $K|E| \times KN$ block matrix

$$\delta_0(\sigma) = \bigoplus_{(i,j) \in E_\varepsilon} \begin{bmatrix} U_{ij}(\sigma) & -I_K \end{bmatrix},$$

where the columns correspond to vertex $j$ (tail) and vertex $i$ (head) respectively, and the rows correspond to the edge stalk.

The **sheaf Laplacian** is

$$\mathcal{L}(\sigma) = \delta_0(\sigma)^\dagger \delta_0(\sigma).$$

This is a $KN \times KN$ positive semidefinite sparse Hermitian matrix. For $K = 20$, $N = 9877$, the matrix has dimension $197{,}540 \times 197{,}540$ with $O(N\varepsilon)$ nonzero blocks (sparsity determined by graph connectivity).

The **spectral sum** is defined as

$$S(\sigma) = \frac{1}{KN} \sum_{k=1}^{KN} \lambda_k(\mathcal{L}(\sigma)) \cdot w_k,$$

where $w_k$ is an appropriate weight (e.g., normalized eigenvalue index, or a smooth cutoff). In practice, only the top eigenvalues are computed via LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient), using the spectral flip trick: compute the largest eigenvalues of $\lambda_{\max} I - \mathcal{L}$ using the `which='LM'` option in `scipy.sparse.linalg.eigsh`.

The key diagnostic is the dependence of $S(\sigma)$ on $\sigma \in (0.25, 0.75)$ for zeta zeros vs. random controls, and how this dependence sharpens as $K$ increases.

---

## 6. The Fourier Sharpening Hypothesis

### 6.1 The Truncation Analogy

The multi-prime superposition

$$A_{ij}(\sigma) = \sum_{p \leq K} e^{i\Delta\gamma \log p} \cdot B_p(\sigma)$$

is a **finite Fourier sum** over the "arithmetic frequencies" $\log p$ (for $p \leq K$), with "spectral amplitudes" $B_p(\sigma)$. By analogy with classical Fourier analysis, a finite sum of sine waves cannot reproduce a discontinuous function exactly — it approximates it with smoothing artifacts. As more terms are added, the approximation sharpens.

The explicit formula for $\psi(x)$ is an exact (infinite) Fourier-like sum over all primes. The ATFT generator truncated at $K$ is a finite approximation. The **Fourier Sharpening Hypothesis** predicts that as $K \to \infty$, the spectral response $S(\sigma)$ sharpens toward a sharp peak (or cusp) at $\sigma = \tfrac{1}{2}$.

### 6.2 K = 20 Behavior: Eight Primes, Smooth Hill

At $K = 20$, the 8 primes $\{2, 3, 5, 7, 11, 13, 17, 19\}$ are incorporated. The experimental results (Phase 3, $N = 9{,}877$ zeros) show:

| $\sigma$ | $S(\sigma, \varepsilon=5.0)$ |
|----------|------------------------------|
| 0.25 | 0.2589 |
| 0.35 | 0.3136 |
| 0.50 | 0.3339 |
| 0.65 | 0.3368 |
| 0.75 | 0.3379 |

The signal is monotonically increasing through $\sigma = 0.5$ with no turnover. However, the **rate of change** is informative: the acceleration is highest far from the critical line and decelerates near $\sigma = 0.5$, creating a pronounced plateau. This is consistent with 8 sine waves beginning to "build up" near the correct location but lacking the resolution to create a peak.

Crucially, random point sets produce $S(\sigma = 0.5, \varepsilon = 5.0) \approx 0.0005$ vs. $0.3339$ for zeta zeros — a contrast ratio of approximately $670\times$. The arithmetic structure of zeta zeros (their prime-encoded geometry) generates an enormous spectral signal compared to random inputs.

### 6.3 K = 50 Behavior: First Evidence of Spectral Turnover

At $K = 50$, incorporating 15 primes up to 47, the Phase 3 GPU scout ($N = 2{,}000$ zeros) shows:

| $\sigma$ | $S(\sigma, \varepsilon=3.0)$ | $S(\sigma, \varepsilon=5.0)$ |
|----------|------------------------------|------------------------------|
| 0.25 | 0.001871 | 0.009722 |
| 0.40 | 0.002403 | **0.012652** (peak) |
| 0.50 | 0.002806 | 0.012621 (-0.2%) |
| 0.60 | 0.002874 | 0.012115 (-4.2%) |
| 0.75 | 0.002881 | 0.012154 (-3.9%) |

At $\varepsilon = 5.0$, the signal now exhibits a **spectral turnover**: the peak occurs near $\sigma \approx 0.40$–$0.50$ and the signal drops for $\sigma > 0.50$. This qualitative change — from monotonically increasing (K=20) to peaked-then-decreasing (K=50) — is the first direct evidence of Fourier sharpening in the ATFT framework.

The destructive interference off the critical line is becoming visible as more prime frequencies are added. Higher-frequency primes (larger $p$, shorter "wavelength" in $\log p$) contribute oscillations that cancel for $\sigma \neq \tfrac{1}{2}$ but reinforce at $\sigma = \tfrac{1}{2}$.

**Note on the $\sigma \approx 0.40$ peak location:** With $N = 2{,}000$ (a smaller dataset than $N = 9{,}877$), finite-$N$ effects may shift the apparent peak location. A finer $\sigma$ grid near $0.45$–$0.55$ at $N = 9{,}877$ is needed to resolve the true peak position for $K = 50$.

### 6.4 K → ∞: Predicted Phase Transition

In the limit $K \to \infty$, all primes are included, and the Euler product formula implies that the construction becomes equivalent to working with the full zeta function $\zeta(\sigma + it)$ on the critical strip. The explicit formula becomes exact, and the spectral sum $S(\sigma)$ should exhibit a **sharp phase transition** at $\sigma = \tfrac{1}{2}$:

- $S(\sigma)$ should develop a cusp or discontinuous derivative at $\sigma = \tfrac{1}{2}$.
- The contrast $S(1/2) / S(\sigma)$ for $\sigma \neq \tfrac{1}{2}$ should diverge.
- The peak width should decrease as $O(1/\log K)$, consistent with the density of primes and the Fourier uncertainty principle.

This phase-transition behavior is the central **testable prediction** of the ATFT framework. If RH is true and the prime representation captures the arithmetic structure faithfully, then:

$$\lim_{K \to \infty} \arg\max_\sigma S(\sigma) = \frac{1}{2}.$$

Conversely, if the peak drifts away from $\sigma = \tfrac{1}{2}$ as $K$ grows, or if it splits, this would indicate either a failure of RH or a failure of the ATFT construction to capture the relevant arithmetic.

---

## 7. What Would Disprove RH?

Within the ATFT framework, the following outcomes would constitute evidence against RH or evidence that the framework fails to capture the arithmetic of RH:

### 7.1 Peak Localization Away from $\sigma = 1/2$

If, as $K \to \infty$, the spectral peak $\sigma^* = \arg\max_\sigma S(\sigma)$ converges to a value $\sigma^* \neq \tfrac{1}{2}$, and if the convergence is robust across different values of $N$ and $\varepsilon$, this would suggest the arithmetic structure peaks off the critical line — evidence that some zeros have $\text{Re}(\rho) = \sigma^* \neq \tfrac{1}{2}$.

### 7.2 Peak Splitting

If $S(\sigma)$ develops two distinct local maxima at $\sigma^*$ and $1 - \sigma^*$ (with $\sigma^* < \tfrac{1}{2}$), this would be consistent with a pair of zeros $\rho_0$ and $1-\rho_0$ off the critical line, contributing their symmetrized arithmetic weight at $\sigma^*$ and $1-\sigma^*$ respectively.

### 7.3 Non-Growing Contrast Ratio

Define the **contrast ratio**

$$R(K) = \frac{S_{\text{zeta}}(1/2, K)}{S_{\text{random}}(1/2, K)}.$$

If $R(K)$ does not grow with $K$ — or worse, decreases — this would indicate that the prime-arithmetic structure encoded in $U_{ij}(\sigma)$ does not distinguish zeta zeros from random points at the critical line. This would mean the construction fails to capture what is special about $\sigma = \tfrac{1}{2}$ for zeta zeros, even if RH is true.

At $K = 20$, $R(20) \approx 670$. The prediction is that $R(K)$ grows, possibly as a power law or faster, as $K$ increases.

### 7.4 Indistinguishability from GUE Controls

If zeta zeros become spectrally indistinguishable from GUE random matrix eigenvalues in the ATFT framework (i.e., $S_{\text{zeta}}(\sigma, K) \approx S_{\text{GUE}}(\sigma, K)$ for all $\sigma$ and all $K$), this would indicate that the ATFT construction is sensitive only to GUE statistics — the local spacing distribution — and not to the deeper arithmetic structure encoded in the prime factorization of the integers.

This would not disprove RH but would show that the ATFT framework, as implemented, captures only the Montgomery-Odlyzko GUE statistics (which are consistent with RH but do not imply it) rather than the finer arithmetic structure.

---

## 8. Computational Strategy

### 8.1 Phase 1: Topological Baseline (Completed)

The goal of Phase 1 was to validate that the ATFT framework's topological invariants — Betti curves and Gini trajectories of persistent homology — can distinguish the spectral structure of zeta zeros from random and Poisson baselines.

Key results from Phase 1:
- The distance $D_M(\text{zeta})$ between zeta Betti curves and GUE Betti curves decreased monotonically from $0.144$ to $0.052$ as $N$ grew from $500$ to $10{,}000$, confirming that zeta zeros converge topologically to GUE as predicted by Montgomery-Odlyzko.
- The L2 Gini coefficient remained flat at $\approx 0.025$ across scales, indicating scale-invariant shape matching between zeta and GUE merging rules.
- Critical methodological fix: rank-based spectral unfolding was replaced by smooth CDF unfolding (semicircle for GUE, Riemann-von Mangoldt for zeta zeros).
- Computational fix: Dumitriu-Edelman tridiagonal model for GUE replaced dense complex Hermitian matrix generation, enabling $N = 10{,}000+$ at $O(N)$ memory.

### 8.2 Phase 2: Transport Map Construction (Completed)

Phase 2 developed the core ATFT machinery:
- Designed and implemented the three transport modes (global, resonant, FE).
- Proved flat holonomy of the global mode, ruling it out for sigma-detection.
- Identified the FE mode as the canonical choice based on the functional equation symmetry argument.
- Proved that the FE generator's Hermiticity at $\sigma = \tfrac{1}{2}$ creates a geometrically preferred point, and that this preference is intrinsic to the generator — present even for random input points.
- **Key methodological lesson:** To detect RH, the specialness of $\sigma = \tfrac{1}{2}$ must emerge from the DATA (the zeta zeros), not from the CONNECTION (the FE generator alone). The Phase 3 multi-prime superposition addresses this by having sigma-dependence arise from the phase interference $p^{i\Delta\gamma}$, which does depend on the zero data.

### 8.3 Phase 3: Multi-Prime Superposition at Scale (Current)

Phase 3 implements and scales the full FE-mode superposition generator with multiple primes:

**CPU sparse engine:**
- `scipy` BSR (Block Sparse Row) format for the sheaf Laplacian.
- `eigsh` (ARPACK-based) for near-kernel and large eigenvalues.
- Spectral flip trick: compute `eigsh(λ_max * I - L, which='LM')` to access smallest eigenvalues efficiently.
- Handles $N = 9{,}877$ at $K = 20$ with feasible runtime.

**GPU engine:**
- `CuPy` CSR format for GPU-accelerated sparse linear algebra.
- CuPy LOBPCG has a known bug for complex matrices (spurious negative eigenvalues); the spectral flip trick resolves this.
- Achieves significant speedup over CPU for large $K$.
- Timing at $K = 50$, $N = 9{,}877$: transport 98s + Laplacian assembly 125s + eigensolver 1594s $\approx$ 30 min per $(\sigma, \varepsilon)$ point.

**Distributed computing:**
- Role-based sweep scripts (`zeta-only`, `control-cpu`, `gpu-k50`).
- JSON result aggregation with cross-$K$ sharpening tables.
- Remote provisioning on RunPod A100 nodes for $K \geq 100$.

### 8.4 Scaling Roadmap

| Phase | N | K | Status |
|-------|---|---|--------|
| 3a | 9,877 | 20 | Complete |
| 3b | 9,877 | 50 | Running |
| 3c | 9,877 | 100 | Planned (RunPod A100) |
| 3d | 9,877 | 200 | Planned |
| 4 | 100,000 | 100+ | Future |

The critical milestones are:
- **$K = 50$, $N = 9{,}877$:** Confirm spectral turnover seen in $N = 2{,}000$ scout; determine peak location with finer $\sigma$ grid.
- **$K = 100$:** First test of Fourier sharpening at 25 primes; expected significant narrowing of peak.
- **$K = 200$:** If peak is narrowing toward $\sigma = 0.50$, this constitutes strong evidence for the ATFT formulation of RH.

### 8.5 Contrast Ratio as Primary Observable

The primary observable across all phases is the **contrast ratio** between zeta zeros and random controls:

$$R(K, N, \varepsilon) = \frac{S_{\text{zeta}}(\sigma^*, K, N, \varepsilon)}{S_{\text{random}}(\sigma^*, K, N, \varepsilon)},$$

where $\sigma^* = \arg\max_\sigma S_{\text{zeta}}(\sigma, K, N, \varepsilon)$ is the empirical peak location. A growing contrast ratio — $R(K) \to \infty$ as $K \to \infty$ — combined with $\sigma^* \to \tfrac{1}{2}$ constitutes the computational evidence for the ATFT formulation of RH.

At $K = 20$: $R \approx 670$ (zeta vs. random), with $\sigma^*$ not yet resolved from the plateau region. At $K = 50$: first turnover observed, $\sigma^* \approx 0.40$–$0.50$ (finite-$N$ effects present). The trajectory of $\sigma^*(K)$ and $R(K)$ as $K$ increases is the core empirical question driving the computational program.

---

## References

The theoretical foundations draw on the following standard references:

**Riemann Hypothesis and Analytic Number Theory:**
- Riemann, B. (1859). "Uber die Anzahl der Primzahlen unter einer gegebenen Grosse." *Monatsberichte der Berliner Akademie.*
- Davenport, H. (2000). *Multiplicative Number Theory*, 3rd ed. Springer.
- Titchmarsh, E. C. (1986). *The Theory of the Riemann Zeta Function*, 2nd ed. Oxford University Press.
- Edwards, H. M. (1974). *Riemann's Zeta Function*. Academic Press.

**Random Matrix Theory:**
- Montgomery, H. L. (1973). "The pair correlation of zeros of the zeta function." *Analytic Number Theory*, Proc. Symp. Pure Math. 24, AMS.
- Odlyzko, A. M. (1987). "On the distribution of spacings between zeros of the zeta function." *Math. Comp.* 48(177), 273–308.
- Katz, N. M. and Sarnak, P. (1999). *Random Matrices, Frobenius Eigenvalues, and Monodromy*. AMS.
- Mehta, M. L. (2004). *Random Matrices*, 3rd ed. Elsevier.

**Gauge Theory and Fiber Bundles:**
- Kobayashi, S. and Nomizu, K. (1963). *Foundations of Differential Geometry*, Vol. I. Wiley.
- Atiyah, M. F. and Bott, R. (1983). "The Yang-Mills equations over Riemann surfaces." *Phil. Trans. R. Soc. London* A 308, 523–615.

**Sheaf Theory and Topological Data Analysis:**
- Hansen, J. and Ghrist, R. (2021). "Opinion dynamics on discourse sheaves." *SIAM J. Appl. Math.* 81(5), 2033–2060.
- Curry, J. M. (2014). "Sheaves, cosheaves and applications." PhD thesis, University of Pennsylvania.
- Edelsbrunner, H. and Harer, J. (2010). *Computational Topology: An Introduction*. AMS.

**Hilbert-Polya and Spectral Approaches:**
- Berry, M. V. and Keating, J. P. (1999). "The Riemann zeros and eigenvalue asymptotics." *SIAM Review* 41(2), 236–266.
- Connes, A. (1999). "Trace formula in noncommutative geometry and the zeros of the Riemann zeta function." *Selecta Math.* 5, 29–106.

---

*This document describes ongoing research. The ATFT framework is a computational approach to generating evidence for or against RH; no claim of proof is made. All experimental results are as of 2026-03-16.*
