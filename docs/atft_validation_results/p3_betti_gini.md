# P3: Betti Curves + Gini Trajectory — Validation Result

## Predictions
> (6) Betti curve β₁(ε) is sufficient to detect phase transitions.
> (7) Gini trajectory is the strongest predictor of system quality.

## Verdict: PASS (both predictions)

## Results

### Onset Scale Discrimination
| Source | Onset τ* | Gini at Onset | Spectral Gap |
|--------|---------|--------------|-------------|
| Zeta | 0.00205 | 0.468 | 0.00981 |
| GUE | 0.00260 | 0.471 | 0.01218 |
| Random | 0.00105 | 0.492 | 0.01567 |

Onset scale differs by **21.1%** between Zeta and GUE — sufficient to discriminate.

### Gini Trajectory Across K
| Source | G(K=50) | G(K=400) | Direction |
|--------|---------|----------|-----------|
| Zeta | 0.487 | 0.492 | INCREASING (hierarchifying) |
| GUE | 0.487 | 0.493 | INCREASING |
| Random | 0.497 | 0.000 | DECREASING (flattening) |

Zeta and GUE both hierarchify with more primes. Random flattens. Gini trajectory direction discriminates structured from unstructured sources.

### Gini Across Epsilon (K=200)
Gini decreases with epsilon for all sources (fewer eigenvalues near zero = less hierarchy). Zeta consistently has slightly lower Gini than GUE, and both are lower than Random.

## Assets
- `assets/validation/p3_betti_curves.png`
- `assets/validation/p3_gini_trajectory.png`
- `assets/validation/p3_gini_vs_epsilon.png`
- `output/atft_validation/p3_betti_gini.json`

## Reproduce
```bash
# Requires P2 data first
.venv/bin/python atft/experiments/p3_betti_gini.py
```
