# Convergence Audit — Pre-Registration

**Frozen:** before reading the K=100 isolation verdict.
**Author:** Blake Jones + Claude (Opus 4.8).
**Provenance of method:** mirrors the IG-PRIMON pre-registration discipline
(`ig-primon-t1`: `CBP_WOLFCAMP_STRUCTURAL_EDGE_prereg`) — register the falsifier
first, report whatever the data says, accept the null if that is where it lands.
"Structure-real ≠ structure-pays" → here, **"S-premium-reproduces ≠ S-premium-is-real."**

---

## 1. What triggered this audit

The "arithmetic premium" S(GUE) − S(ζ) (headline: 21.5% at K=200) is the sum of
the `k_eig = 20` smallest eigenvalues of the sheaf Laplacian L = δ₀†δ₀. The
published values were produced by `TorchSheafLaplacian.smallest_eigenvalues`,
which finds the smallest eigenvalues via a **single-shot hand-rolled Lanczos**
(`_lanczos_largest`, spectral-flip) using

    m = min(max(2k+20, k+50), dim) = 70 Krylov vectors   (for k=20)

regardless of its `max_iter` argument.

Independent CPU recomputation of the SAME quantity, same config (K=100, N=1000,
σ=0.5, ε=3.0, k=20), with two robust solvers — ARPACK shift-invert AND ARPACK
spectral-flip — both return **S(ζ) ≈ 0.01**, not the published **12.48**. The 20
smallest eigenvalues form a clean ascending near-kernel sequence
(1.6e-10 → 1.6e-3); none reach the O(0.6) magnitude required to sum to ~12.

A 70-vector Krylov space resolving ≥20 **clustered near-kernel** modes is the
textbook regime where Lanczos under-converges and **overestimates** the smallest
eigenvalues. That is the suspect.

## 2. Pre-registered falsifier (frozen before the verdict)

**Test A — same-matrix solver isolation (K=100).** Build one L. Run three
solvers on it: (a) ARPACK restarted spectral-flip, (b) a faithful port of the
GPU's 70-vector hand-Lanczos, (c) ARPACK over-resolved (k=60, ncv=160).

- If **(b) ≈ 12** while **(a) ≈ (c) ≈ 0** on the identical matrix → the published
  premium is a **Krylov-truncation artifact** of the 70-vector Lanczos. PROCEED to Test C.
- If all three agree → no artifact; the discrepancy is elsewhere (investigate the
  matrix build). The audit does not implicate the published value.

**Test B — reproduce on original hardware (RTX 5070).** Run the *actual*
`TorchSheafLaplacian` path on the 5070 (the GPU named in the README). Pull the
GPU-built matrix to scipy and run ARPACK on it.

- If GPU = 12.48 and ARPACK-on-GPU-matrix ≈ 0 → confirms Test A on real hardware,
  on the GPU's own matrix. Artifact established end-to-end.

**Test C — does any real premium survive proper convergence?** With a robust,
converged solver, compute S(ζ) vs S(GUE) (and Even, Random) at K=100 (and K=200
if feasible), with enough precision to compare.

- If converged S(ζ) ≈ S(GUE), both ≈ 0, statistically indistinguishable →
  **premium is pure artifact. Null accepted. Headline corrected** (the 21.5% /
  16σ / "tighter than GUE" family of claims retracted or rewritten).
- If converged S(ζ) < S(GUE) survives at meaningful separation → **premium is
  real, but the metric was mislabeled**: it is a "sum of the 20 smallest
  eigenvalues" claim that actually held only for the 70-vector-Lanczos Ritz
  functional. Relabel and re-state with the converged numbers; do not retract.

## 3. Commitments

1. Report the verdict regardless of which branch it lands on.
2. No edits to README / index.html / app.js / CLAIMS_LEDGER quantitative claims
   until Tests A–C are resolved.
3. Ship a computational receipt (`output/convergence_audit/*.json`): every
   solver's eigenvalue array, config, wall-time, and machine, so the correction
   is as auditable as the original claim.
4. The frozen FALSIFICATION.md (F1–P4 pre-registration) is **not** modified —
   this audit concerns the control-battery spectral-sum metric, a separate object.
