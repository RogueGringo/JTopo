# P5b: BPST Instanton Discrimination — Validation Result

## Prediction
> Parity-complete feature map distinguishes Q=+1 from Q=-1 via mirror-image persistence signatures.

## Verdict: PARTIAL FAIL (vacuum/instanton: PASS, Q=+1 vs Q=-1: FAIL)

## Results

| Comparison | KS Statistic | p-value | Discriminated? |
|-----------|-------------|---------|---------------|
| Q=0 (vacuum) vs Q=+1 (instanton) | **1.000** | 2.96e-299 | **YES** |
| |Q|=1 vs |Q|=2 (multi-instanton) | **0.457** | 2.43e-47 | **YES** |
| Q=+1 vs Q=-1 (instanton vs anti) | 0.066 | 0.225 | **NO** |

## Root Cause of Q=+1 vs Q=-1 Failure

The topological charge density q_μν = ½ Im Tr P_μν is **identically zero** for all Q values in the naive BPST discretization. The continuum instanton produces real-valued plaquettes on the lattice at this spacing — Im(Tr P) vanishes numerically.

This means the parity-odd component of the feature map (which should distinguish Q=+1 from Q=-1 by sign flip) carries no signal. The s_μν (action density) component correctly discriminates vacuum from instanton and |Q|=1 from |Q|=2, but cannot distinguish the sign of Q.

## What Would Fix It
- **Lattice cooling:** Start from random configuration, cool toward the instanton minimum. This produces lattice-native instantons with non-zero q_μν.
- **Finer lattice:** Smaller lattice spacing a → q_μν becomes non-zero as discretization effects diminish.
- **Improved discretization:** Use the overlap Dirac operator for topological charge measurement instead of the clover plaquette.

## What Survived
- Vacuum vs instanton: perfect discrimination (KS=1.0)
- Multi-instanton structure: strong discrimination (KS=0.457)
- The s_μν component of the feature map works correctly
- Gini coefficients differ: Q=0 (G=0.000), Q=+1 (G=0.566), Q=-1 (G=0.724), Q=+2 (G=0.583)

## Assets
- `assets/validation/p5b_instanton_barcodes.png`
- `output/atft_validation/p5b_instanton.json`

## Reproduce
```bash
.venv/bin/python atft/experiments/p5b_instanton_validation.py
```
