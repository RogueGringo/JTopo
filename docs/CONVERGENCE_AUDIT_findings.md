# Convergence Audit — Findings

**Status:** ζ resolved; GUE/even/random rows + β₀ comparison in progress.
**Method discipline:** IG-PRIMON pre-registration (see `CONVERGENCE_AUDIT_prereg.md`).
**Machine:** RTX 5070 (12 GB) + i9-9900K + 32 GB. The original project hardware.

---

## TL;DR

The published spectral sums — and therefore the **21.5% premium**, **"16σ"**,
**"670×"**, the **S(σ) hierarchy**, and the **"β₀ = 0"** claim — were produced by
an eigensolver that **does not converge** on this operator. The headline numbers
are artifacts of Krylov truncation, not properties of the zeta zeros.

| Quantity | Published | Converged (this audit) | Status |
|---|---|---|---|
| S(ζ), K=100 | 12.480 | **0.0034** | artifact (~3,700× inflated) |
| smallest eigenvalue, K=100 | 0.002 (reported) | **5.5e-12** (genuine kernel) | artifact |
| β₀ (kernel dim) | "0 at all points" | **≥ 30** (≥2 machine-zero) | inverted |
| 21.5% / 16σ / 670× premium | headline | re-derivation in progress | magnitude refuted |

## Root cause

`smallest_eigenvalues` in the GPU backends (`torch_sheaf_laplacian`,
`matfree_sheaf_laplacian`, `hybrid_sheaf_laplacian`) finds the smallest
eigenvalues via a **single-shot hand-rolled Lanczos** (`_lanczos_largest`,
spectral-flip) with a fixed Krylov dimension

    m = min(max(2k+20, k+50), dim) = 70    (for k_eig = 20)

independent of its `max_iter` argument. The sheaf Laplacian has a **dense
near-kernel** (≥30 eigenvalues clustered at ~0). A 70-vector Krylov space cannot
resolve a cluster of that size; instead of returning ~30 values near 0, it
returns a **sparse sampling spread across the cluster's range**. That sampling,
summed, gives ~12; the true 20-smallest sum is ~0.003.

This is a textbook singular-perturbation / boundary-layer failure: the near-kernel
is a thin singular region at the bottom of the spectrum, and the coarse Krylov
space smoothed over it instead of resolving it at its own scale.

## Evidence (receipts)

**1. The literal published code reproduces the artifact AND exposes it.**
`TorchSheafLaplacian.smallest_eigenvalues` at K=100 returns:

    S = 12.318
    "eigs" = [0.002 0.013 0.032 0.061 0.098 0.146 0.204 0.268 0.344 0.427
              0.522 0.623 0.731 0.85 0.971 1.104 1.244 1.397 1.561 1.721]

A smooth ramp to λ_max(flip)=1.721 — not the bottom of the spectrum.

**2. Robust solvers (ARPACK shift-invert AND spectral-flip with IRAM restarts)
agree on the true spectrum.** K=100, ζ:

    smallest 5 eigenvalues = [5.5e-12, 6.7e-12, 2.8e-9, 7.3e-7, 1.4e-6]
    S(20 smallest)         = 0.0034

**3. Residual proof — the ≈0 eigenvalues are genuine.** For the smallest pairs,
‖L x − λ x‖ / ‖x‖:

    [5.1e-12, 4.9e-12, 5.9e-12, 1.0e-4, 7.6e-12]

Four of five at machine precision: these are real eigenvectors of real zero
eigenvalues. (The lone 1e-4 is the expected ambiguity of labelling individual
vectors inside a *degenerate* kernel cluster — corroboration, not a counter.)

**4. Same matrix, both solvers.** The CPU `SparseSheafLaplacian` (nnz=59,830,100)
and the GPU `TorchSheafLaplacian` (nnz=59,830,090) build the same operator (the
transport math is byte-for-byte the same code path). The discrepancy is solver,
not matrix.

## What this does and does not touch

- **Refuted (magnitude):** "21.5% tighter than GUE", "16σ", "670×", "S(ζ)=11.784",
  the Tier-1..4 S-value hierarchy as numbers, "β₀ = 0 / no topological structure".
- **Open (re-derivation in progress):** whether any *small* ζ-vs-GUE premium
  survives a converged solver (Test C), and whether the arithmetic signature lives
  in **β₀(ζ) vs β₀(GUE)** — the topologically-protected (Čech–de Rham) invariant
  that the artifact had zeroed out. This is the inversion: the value, if any, is in
  the kernel, not the eigenvalue sum.
- **Untouched:** the frozen `FALSIFICATION.md` thresholds (separate object); the
  engineering (matrix-free Padé, speedups); the explicit refusal to claim an RH
  proof.

## Test C — converged premium and the kernel (β₀) inversion

With the robust solver, K=100, k_eig=20, all six sources:

| source | edges | S (converged) | near-kernel (<1e-3) | exact β₀ (<1e-8) |
|---|---|---|---|---|
| **zeta** | 2492 | **0.0034** | **≥30** (censored at window) | **3** |
| gue_2000 | 2717 | 0.0078 | 19 | 1 |
| gue_2001 | 2738 | 0.0076 | 19 | 1 |
| gue_2002 | 2742 | 0.0065 | 20 | 1 |
| even | 2994 | 0.0081 | 21 | 2 |
| random | 2963 | 0.0099 | 16 | 0 |

**Two findings:**

1. **The ζ-tighter *direction* survives convergence.** S(ζ)=0.0034 is below every
   control (GUE ~0.007, even 0.008, random 0.010). The *magnitude* is at the
   O(0.003) near-kernel scale — nothing like the artifactual 21.5%; do not quote
   a "%" premium here, it is a ratio of tiny kernel-dominated numbers.

2. **The signal is carried by the kernel, not the eigenvalue sum.** ζ is tighter
   *because it has more flat sections* — a larger β₀. This is the topological
   **inversion**: the value lives in the kernel (Čech–de Rham-protected), which is
   exactly the quantity the 70-vector Lanczos zeroed out when it claimed "β₀=0".

**Confound — NOT yet broken.** Kernel dimension anti-correlates with edge count
(more edges = more constraints = smaller kernel), and ζ is the *sparsest* graph
in the set. A 5-point control fit predicts ζ's near-kernel ≈ 19.7; actual ≥30
(surplus +10.3) — but that rests on **extrapolating a noisy, near-flat fit below
the controls' edge range**. Suggestive, not established. The clean test
(`edge_matched_kernel.py`): re-measure every control at the ε giving ζ's edge
count (~2492), so all graphs have identical connectivity, then compare kernels.

### Edge-matched confound test (resolves it)

Re-measuring every source at the ε giving ζ's edge count (~2492), with one
consistent solver tolerance (tol=1e-4 for all — Test C had inadvertently given
ζ a *tighter* tolerance, which alone inflated ζ's near-kernel from 24 to 30):

| source | edges | near-kernel (<1e-3) |
|---|---|---|
| **zeta** | 2492 | **24** |
| gue_2000 | 2492 | 24 |
| gue_2001 | 2492 | 20 |
| gue_2002 | 2492 | 20 |
| random | 2491 | 14 |
| even | 2994* | 17 (*uniform gaps make edge count a step function — can't hit 2492) |

**Result:** at matched edges, ζ (24) sits **tied with the top GUE draw**, inside
the GUE band (20–24). The earlier "+10 surplus" was two stacked confounds —
inconsistent solver tolerance + ζ's sparser graph. It is gone.

The kernel is **not** purely edge-count, though: at equal edges it still splits
by spacing statistics — **repulsive** (ζ, GUE: 20–24) vs **Poisson** (random:
14). But level repulsion is *exactly* what Montgomery–Odlyzko says ζ and GUE
share. So on both quantities the kernel responds to (edge density + repulsion),
ζ and GUE are matched and do not separate. The only separation is random
(no repulsion) — the instrument detecting Poisson-ness, not arithmetic.

**Verdict (matched-edge β₀ test):** the sheaf-Laplacian *near-kernel* is a
detector of **local spacing statistics** (edge density + level repulsion), which
ζ and GUE share by Montgomery–Odlyzko. At matched edges with the prime connection,
ζ does **not** separate from GUE. On this clean test, **no arithmetic signal.**

### Open thread — scrambled-connection flicker (not yet resolved)

A follow-up replaced the prime frequencies {log p} with random frequencies of
matched range, keeping the prime generators, at **fixed ε=3** (so points/edges
unchanged within each variant). Two scramble seeds, K=100:

| variant | ζ near | GUE near (2 draws) | ζ-surplus |
|---|---|---|---|
| PRIME | 24 | 20, 20 | **+4.0** |
| SCRAMBLE-1 | 19 | 22, 21 | −2.5 |
| SCRAMBLE-2 | 21 | 22, 19 | +0.5 |

The script auto-printed **"VERDICT: ARITHMETIC"** (PRIME +4.0 vs SCRAMBLE mean
−1.0). **That label is not trustworthy, for two decisive reasons:**

1. **The two scramble seeds disagree in sign** (−2.5 vs +0.5) — a 3-unit swing
   from changing only the RNG seed. The metric's noise floor (±3) is as large as
   the claimed effect. GUE-to-GUE scatter within a variant is the same size
   (e.g. SCRAMBLE-2 GUE: 22 vs 19). You cannot resolve a +4-vs−1 difference at
   this noise level with one ζ realization and two seeds.
2. **The PRIME +4 baseline is the edge confound, not arithmetic.** At ε=3 GUE has
   more edges than ζ → smaller kernel → 20; the edge-matched test showed GUE
   rises to 24 at ζ's edge count. So +4 is mostly "ζ is sparser here," unrelated
   to primes. The auto-verdict compares an edge-inflated baseline to a noisy mean.

Two *real* facts do survive (correcting the "purely graph-determined" claim
above): scrambling the connection — edges untouched — **does** move the soft
(<1e-3) modes, and ζ's own prime kernel (24) sits above its own scrambles (19,21).
But "ζ likes its prime connection" at a ±3 floor, edge-confounded, is a **whiff,
not a result.**

**The arbiter** is `prime_specificity.py`: scramble **at matched edges**, 3 seeds,
tol=1e-6, smooth S. It measures ζ's prime-vs-random gap and GUE's prime-vs-random
gap *with edges held equal*. If ζ's gap robustly exceeds GUE's, there is a genuine
prime-specific effect and the M–O null above is incomplete. If not, the flicker
was edge confound + noise. **Pending — and the auto-"ARITHMETIC" is being treated
as un-earned until it survives this.**

## Reproduce

    # CPU venv (numpy+scipy only): the converged truth
    python scripts/diagnostics/converged_premium.py 100

    # GPU venv (torch): the literal published path, side-by-side with ARPACK
    python scripts/diagnostics/gpu_reproduce.py 100 cpu

Receipt: `output/convergence_audit/converged_premium_K100.json`.
