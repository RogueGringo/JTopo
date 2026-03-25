# P2: Sheaf Laplacian Kernel Scaling — Validation Result

## Prediction
> On-shell configurations have dim ker(L_F) > 0 at the critical filtration scale.

## Reframed Test
> Does λ₁ approach zero faster for Zeta than GUE? (Continuous crossover vs binary kernel jump)

## Verdict: FAIL (specific prediction) / PARTIAL PASS (premium exists but rate is equal)

## Results

### K Sweep (σ=0.5, ε=3.0)

| K | λ₁(Zeta) | λ₁(GUE) | λ₁(Random) | Zeta/GUE Ratio |
|---|----------|---------|-----------|---------------|
| 50 | 0.002562 | 0.003267 | 0.001383 | 0.784 |
| 100 | 0.002602 | 0.003239 | 0.001362 | 0.803 |
| 200 | 0.002050 | 0.002599 | 0.001054 | 0.789 |
| 400 | 0.001802 | 0.002257 | — (OOM) | 0.798 |

**Power law fits:**
- Zeta: λ₁ = 0.0056 × K^(-0.187), R² = 0.876
- GUE: λ₁ = 0.0073 × K^(-0.192), R² = 0.911
- Random: K=400 OOM'd (subprocess), fit invalid

**Key finding:** α(Zeta) = -0.187 vs α(GUE) = -0.192. GUE approaches zero SLIGHTLY faster. The prediction is FAIL — Zeta does NOT approach zero fastest.

### Eigenvalue Ratio Uniformity (K=200)

| Index | λᵢ(Zeta)/λᵢ(GUE) |
|-------|-----------------|
| 1 | 0.789 |
| 2 | 0.803 |
| 3 | 0.795 |
| 4 | 0.791 |
| 5 | 0.784 |

**CV = 0.8%.** The ~21% premium is uniform across all five lowest eigenvalues. This is a structural feature, not concentrated in one mode.

### Epsilon Sweep (K=200, σ=0.5)

| ε | λ₁(Zeta) | λ₁(GUE) | S(Zeta) | S(GUE) | λ₁ Premium | S Premium |
|---|----------|---------|---------|--------|-----------|----------|
| 1.5 | 0.000147 | 0.000155 | 7.407 | 9.675 | 5.0% | 23.4% |
| 2.0 | 0.000877 | 0.000995 | 8.800 | 11.941 | 11.8% | 26.3% |
| 3.0 | 0.002050 | 0.002599 | 11.784 | 15.004 | 21.1% | 21.5% |
| 4.0 | 0.002648 | 0.003344 | 14.711 | 18.605 | 20.8% | 20.9% |

**Hierarchy S(Zeta) < S(GUE) holds at ALL epsilon values.** The premium grows from 5% (fine scale) to 21% (coarse scale) then plateaus.

## Interpretation

The paper predicts ker(L_F) > 0 — a discontinuous phase transition where the kernel dimension jumps. The data shows something different: a **continuous premium** where every Zeta eigenvalue is ~21% lower than the corresponding GUE eigenvalue, with the premium itself depending on the topological scale ε.

This is not what the paper predicted. But it may be more interesting: a continuous, scale-dependent, structurally uniform arithmetic premium across the entire low-lying spectrum. The premium is not a threshold effect — it's a persistent spectral offset.

The scaling exponents are nearly equal (−0.187 vs −0.192), meaning both Zeta and GUE eigenvalues approach zero at the same rate. The difference is a multiplicative constant, not a rate. The premium is a SHIFT, not an ACCELERATION.

## What Survived
- ✓ Hierarchy λ₁(Zeta) < λ₁(GUE) at all ε tested
- ✓ Premium uniform across eigenvalue indices (CV=0.8%)
- ✓ Premium is scale-dependent (5% → 21% as ε grows)
- ✓ Spectral sum hierarchy S(Z) < S(G) < S(R) holds at all ε

## What Failed
- ✗ α(Zeta) is NOT less than α(GUE) — similar decay rates
- ✗ No evidence of ker(L_F) > 0 at any K tested
- ✗ K=400 Random OOM'd (subprocess isolation failed for this source)

## Assets
- `assets/validation/p2_lambda1_scaling.png` — λ₁ vs K log-log plot
- `assets/validation/p2_eigenvalue_ratio.png` — Ratio uniformity
- `assets/validation/p2_epsilon_sweep.png` — Hierarchy across ε
- `output/atft_validation/p2_kernel_scaling.json` — Raw results

## Reproduce
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python atft/experiments/p2_kernel_scaling.py
```
