# Ti V0.1

```
 ████████╗██╗    ██╗   ██╗ ██████╗    ██╗
 ╚══██╔══╝██║    ██║   ██║██╔═████╗  ███║
    ██║   ██║    ██║   ██║██║██╔██║  ╚██║
    ██║   ██║    ╚██╗ ██╔╝████╔╝██║   ██║
    ██║   ██║     ╚████╔╝ ╚██████╔╝██╗██║
    ╚═╝   ╚═╝      ╚═══╝   ╚═════╝ ╚═╝╚═╝
```

> Topological Investigation of the Riemann Hypothesis.
> One GPU. 46 primes. 21.5%.

---

> ## ⚠️ CORRECTION IN PROGRESS (2026-06) — read before the claims below
>
> An independent convergence audit found that the headline numbers in this
> README — **21.5% premium, "16σ", "670×", the S(σ) hierarchy, and "β₀ = 0"** —
> were produced by an eigensolver that **does not converge** on this operator.
>
> The GPU `smallest_eigenvalues` uses a fixed 70-vector Lanczos that cannot
> resolve the sheaf Laplacian's dense **near-kernel**. It reports a spread of
> Ritz values (summing to ~12) instead of the true smallest eigenvalues (~0).
>
> | Quantity (K=100) | Published | Converged (robust solver) |
> |---|---|---|
> | S(ζ) | 12.480 | **0.0034** |
> | smallest eigenvalue | 0.002 | **5.5e-12** (genuine, residual 1e-12) |
> | β₀ (kernel dim) | "0" | **≥ 30** |
>
> The premium *magnitude* is refuted. Whether any *small* real ζ-vs-GUE signal
> survives a converged solver — and whether it lives in **β₀** (the kernel), not
> the eigenvalue sum — is being re-derived. Full audit:
> [`docs/CONVERGENCE_AUDIT_findings.md`](docs/CONVERGENCE_AUDIT_findings.md).
> **Treat every quantitative claim below as under revision.**

---

## Primes Aren't a List. They're a Field.

Everyone who's looked at primes long enough arrives at the same place: the gaps oscillate, the patterns recur at scale, the structure is multi-dimensional. Some see standing waves. Some see eigenvalue repulsion. We see **transport coherence** — how well the primes' internal grammar carries information across the zeros of the zeta function.

We built an interpreter for that coherence. A sheaf Laplacian with a gauge connection woven from prime arithmetic — the same phase factors that appear in the explicit formula connecting primes to zeros. Then we fed it four different point clouds and asked: **who threads the zeros most coherently?**

The answer, at every scale tested: the zeta zeros. By 21.5% over random matrices that share their local statistics. By 7.3% over mathematically perfect spacing. Sixteen standard deviations from the ensemble mean. Stable from 25 primes to 78.

Then we tried to kill it.

Built the most ordered thing mathematics can produce — evenly spaced points, zero randomness. The primes still won. Built 10 independent random matrix ensembles using the proper Dumitriu-Edelman model. The primes fell 16σ below. Checked if it was just edge density in the graph. It wasn't — 15.3% per-edge premium survives normalization.

The primes carry something that order alone doesn't. Something that statistics alone doesn't. The interpreter says coherence. The data says 21.5%. This repo is the measurement.

## The Numbers

> **⚠️ This table is RETRACTED — every S value is a 70-vector-Lanczos artifact.**
> The converged values (robust ARPACK, K=100) are ~3,700× smaller and tell a
> different story. See [`docs/CONVERGENCE_AUDIT_findings.md`](docs/CONVERGENCE_AUDIT_findings.md).

| Source | Published S (RETRACTED) | Converged S (K=100) |
|--------|-------|--------|
| Zeta zeros | ~~11.784~~ | **0.0034–0.0057** |
| Even spacing | ~~12.713~~ | 0.0081 |
| GUE (D-E) | ~~14.970~~ | 0.0065–0.0078 |
| Poisson random | ~~22.087~~ | 0.0099 |

~~**Arithmetic premium over GUE: 21.3%.**~~ The ~12 premium was **~99.97% artifact**.
Under a converged solver the residual ζ-tighter effect is **edge-count** (ζ's level
repulsion → fewer Rips edges → bigger sheaf kernel β₀ → lower S) — and GUE shares
level repulsion. Re-measured at **matched edge count and matched solver tolerance**,
ζ's kernel (≈24) sits *within* the GUE draws' own scatter (20–24). **No arithmetic
signal beyond local statistics survives.** Even the published hierarchy
~~S(ζ) < S(Even) < S(GUE) < S(Random)~~ does not hold converged (even-spaced lands
looser than GUE). (Final controls + a scrambled-connection test pending.)

## How We Got Here

### Chapter One: Eight Primes Walked Into a Manifold

K=20. Eight primes. N=9,877 Odlyzko zeros near the 10²⁰-th zero. The spectral sum went monotonically up through σ = 0.5. No peak. Eight Fourier harmonics isn't enough bandwidth to resolve the signal. Like trying to see a face with eight pixels.

The 670× signal over random controls was encouraging. But the peak wasn't there. Not enough primes.

### Chapter Two: The Turnover

K=50. Fifteen primes. First spectral turnover at ε = 5.0. The summit appeared near σ ≈ 0.40-0.50 and S dropped 4% on the far side. Fifteen harmonics resolved what eight couldn't. The critical line was pulling.

K=100. Twenty-five primes. Signal reversal confirmed at ε = 3.0 — the narrower bandwidth that was still monotonic at K=50. Fourier sharpening. Each new prime brought the peak closer to home.

### Chapter Three: K=200 on a Desktop GPU

N=1000 zeros, 46 primes, RTX 5070 (12 GB VRAM). Three tranches across 12 hours. Crashed once (VRAM). Added batched edge assembly with scipy coalesce. Crashed again (CPU RAM). Added incremental list release. Third time: it ran.

The premium peaked at **σ = 0.500** exactly. Not σ = 0.52. Not σ = 0.48. The critical line.

### Chapter Four: The Surgeon

A three-agent validation committee attacked every claim:

**The Statistician** found pseudoreplication in our p-values and an epsilon confound where our K=100 comparison mixed ε=3.0 and ε=5.0 data. Both fixed. The corrected K=100 premium curve is flat (range: 0.34%) — K=200 is genuinely sharper.

**The Physicist** confirmed the signal behaves like a physical phenomenon: consistent across scales, sharpens with resolution, shows functional equation symmetry. Predicted K=400 premium ≈ 27.7%. Actual K=400 result: **21.6%** (Wigner surmise GUE). The premium converges, not diverges — it's a constant, not a trend.

**The Adversary** proposed the kill shot: *"The spectral sum just measures how many Rips edges you have. Any ordered set will show lower S."* We built evenly-spaced points and ran them. The Adversary was wrong — but partially right about why. We caught a GUE unfolding bug that mapped every realization to evenly-spaced points (zero variance — useless). Fixed it. Ran 10 proper Dumitriu-Edelman GUE realizations with spacing-preserving rescale. Z-score: −16.06.

Then we edge-normalized everything. Even after controlling for the sparser Rips graph from zeta's level repulsion, the per-edge premium holds at 15.3%. The transport matrices carry arithmetic structure that geometry alone cannot explain.

## The Math

The mathematical name for what we're doing: constructing a **sheaf Laplacian with a u(K) gauge connection** over the **Vietoris-Rips complex** of spectrally unfolded zeta zeros, using prime representations as generators.

**In English:** we string the zeros across a graph, connect them with threads made from prime arithmetic, and measure how well the threads agree with each other. Where they agree most is where the fabric fits best.

### Core objects

```
Fiber:        ℂᴷ at each vertex (K = dimension, one slot per integer 1..K)
Prime rep:    ρ(p)|n⟩ = |pn⟩ if pn ≤ K, else 0
Generator:    Bₚ(σ) = log(p) [p⁻σ ρ(p) + p⁻⁽¹⁻σ⁾ ρ(p)ᵀ]
Transport:    Aᵢⱼ(σ) = Σₚ exp(iΔγ·log p) · Bₚ(σ)
Coboundary:   (δ₀x)ₑ = Uₑ xᵢ − xⱼ
Laplacian:    L_𝓕 = δ₀†δ₀
Observable:   S(σ) = Σₖ λₖ(L_𝓕)  — lower S = tighter fabric
```

The exponential factor `exp(iΔγ·log p)` is the explicit formula's Fourier kernel — the same phase factor that connects prime counting to zeta zeros. When many primes constructively interfere at a particular σ, transport becomes coherent and S drops. At σ = 0.500, the interference is maximally constructive for zeta zeros. Not for any control.

### What we're NOT claiming

This is not a proof of the Riemann Hypothesis. ~~The sheaf Laplacian kernel dimension β₀ᶠ = 0 at all points tested~~ **[RETRACTED — exactly inverted.** The 70-vector Lanczos reported kernel modes as ~0.002, so it concluded β₀=0. Converged, **β₀ ≥ 24** with residual-certified zero-eigenvectors. The topological content the framework claimed was absent is in fact its *dominant* feature — and it is **graph-determined** (β₀ = NK − rank δ₀, set by edge count), carrying no arithmetic signal once edges are matched.**]** What the framework actually has is a near-kernel-dominated spectral quantity that, once converged and edge-matched, **does not distinguish ζ from GUE** beyond their shared local statistics. The original (refuted) claim list followed:

1. Distinguishes zeta zeros from all tested controls (16σ from GUE)
2. Peaks at the critical line (σ = 0.500)
3. Sharpens with increasing K (Fourier sharpening)
4. Survives edge normalization (15.3% per-edge premium)
5. Cannot be reproduced by geometric order alone (evenly-spaced control)

Whether this converges to a genuine phase transition as K → ∞ is the open question. K=400 is running now.

## Where the Real Arithmetic Signal Would Be — A Roadmap

The convergence audit established what this instrument *can't* see, and — usefully —
exactly where the signal it was hunting actually lives. The sheaf-kernel observable
detects only **local spacing statistics**, which Montgomery–Odlyzko proves ζ and GUE
*share*. Quantified (`scripts/diagnostics/spacing_delta.py`): **D_KL(ζ ‖ GUE) ≈ 0.006
nats** at N=20,000 zeros — *below* the 0.014-nat finite-sample floor. The arithmetic is
real but **submerged in this projection.** Whoever wants the genuine signal should
change the projection, not tighten this one. Where it lives:

1. **Number variance Σ²(L)** — variance of the count of (unfolded) zeros in an interval
   of length L. GUE: Σ²(L) ≈ (1/π²)(ln 2πL + γ + 1), growing logarithmically forever.
   The zeros track this for small L but **saturate** at large L (Berry 1988), at a scale
   set by the *lowest prime* (~log 2). **This is the cleanest imprint of the primes** —
   it concentrates the diffuse spacing-delta into one feature that *grows* with L instead
   of sitting at the noise floor.
2. **Spectral form factor K(τ)** — Fourier dual of pair correlation. GUE: ramp K(τ)=τ
   (τ<1), plateau =1. The zeros show the GUE ramp **plus** arithmetic oscillations at
   small τ from off-diagonal prime pairs (Berry–Keating semiclassical theory: diagonal
   term → GUE, prime pairs → the correction).
3. **Pair-correlation deviations** — Montgomery's R₂(r) = 1 − (sin πr/πr)² matches GUE,
   but the O(1/log) corrections carry von Mangoldt / prime content (Rudnick–Sarnak
   n-level results and their range limits).

**Sizing it honestly:** the signal is ~0.006 nats in the *spacing* projection. To
resolve it you must either (a) push the finite-sample floor below it — N ≳ 10⁵–10⁶
zeros (Odlyzko's tables reach the 10²²-nd zero, so the data exists) — or, better,
(b) use Σ²(L) or K(τ), which *concentrate* the arithmetic into a feature that grows
with the observable rather than staying diffuse. The right move is (b): **a projection
matched to where the primes are, not a finer sweep of where they aren't.**

**Sheaf analog (optional):** a complement-observable in *this* framework would build
transport coherence over **long-range** zero pairs (large Δγ, where `exp(iΔγ·log p)`
oscillates fastest) rather than short-range Rips edges — projecting onto long-range
rigidity instead of the local kernel. Whether that recovers the number-variance signal
is an open, honest experiment, and the only one this repo's machinery is natively suited
to attempt.

## Project Status

| Phase | Status | Finding |
|-------|--------|---------|
| Phase 1 | Done | Zeta topology distinguishable from GUE; smooth unfolding validated |
| Phase 2 | Done | FE connection unitary at σ=½; FE mode ruled out (geometric artifact) |
| Phase 3 K=20 | Done | 670× signal; monotone (8 primes insufficient) |
| Phase 3b K=50 | Done | First turnover at ε=5.0; peak near σ≈0.40-0.50 |
| Phase 3c K=100 | Done | Signal reversal confirmed; flat premium curve at ε=3.0 |
| Phase 3d K=200 | Done | Premium 21.5% at σ=0.500; three-tier hierarchy universal |
| Phase 3e Controls | Done | Even-spaced, 10 GUE realizations, edge-normalized. ON_SHELL. |
| Phase 3f K=400 | Done | Matrix-free engine. S(ζ)=11.440, S(GUE)=14.590 (Wigner). Premium=21.6% — converging. |
| Phase 4 | Planned | K → ∞ extrapolation |

## Quick Start

```bash
git clone https://github.com/RogueGringo/JTopo.git
cd JTopo
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib torch h5py

# Run the K=200 analysis (uses existing results)
python atft/analysis/k200_full_analysis.py

# Run the control battery
python -m atft.experiments.phase3e_test2_rerun

# Run a new K=200 sweep (requires GPU, ~12 hours)
python -u -m atft.experiments.phase3d_torch_k200 --tranche ALL 2>&1 | tee output/k200.log
```

## Hardware

Everything runs on local hardware. No cloud. No RunPod. No external compute.

| Machine | Role |
|---------|------|
| Threadripper 7960X | Development, CPU sweeps |
| RTX 5070 (12 GB) | Primary GPU — all K=200/K=400 sweeps |

## Data

**Odlyzko zeta zeros** — high-altitude imaginary parts near the 10²⁰-th zero. Publicly available at the [University of Minnesota DTC](https://www.dtc.umn.edu/~odlyzko/zeta_tables/).

## Tests

```bash
pytest tests/ -v  # 299 passing
```

## Falsification

All thresholds were frozen before data collection. If the hierarchy inverts at K=400, or if the GUE ensemble shifts to overlap with zeta, the corresponding claim is withdrawn. The surgical verdicts documenting what we got wrong along the way are in `output/SURGICAL_VERDICT_*.md`.

## Citation

```
@article{jones2026ti,
  title   = {Topological Investigation of the Riemann Hypothesis via
             Sheaf-Theoretic Gauge Fields on Zeta Zero Point Clouds},
  author  = {Jones, B. Aaron},
  year    = {2026},
  note    = {Ti V0.1 — Independent research},
  url     = {https://github.com/RogueGringo/JTopo}
}
```

## License

Research use only. Contact the author for collaboration or licensing.

---

*Built by B. Aaron Jones. One GPU, 46 primes, and the habit of looking before you leap.*

*The manifold has a heartbeat and the primes are its pulse.*
