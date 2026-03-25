# P1: Quantum Harmonic Oscillator — Validation Result

## Prediction
> The longest-lived H₀ features correspond to the largest energy gaps — the most physically significant features of the spectrum.

## Verdict: PASS

## Method
- Generated 2D anisotropic QHO spectrum: V(x,y) = 2k·k_ratio·x² + ½k·y²
- First 225 energy levels (n_max=15, unique after degeneracy collapse)
- H₀ persistence on R¹ point cloud (sorted gaps = merge events)
- Spearman rank correlation between gap sizes and bar lengths
- Swept anisotropy ratio k_x/k_y from 0.5 to 4.0

## Results

| k_ratio | Levels | Spearman ρ | Verdict |
|---------|--------|-----------|---------|
| 0.5 (isotropic) | 29 | 1.000 | PASS |
| 1.0 | 225 | 1.000 | PASS |
| 1.5 | 225 | 1.000 | PASS |
| 2.0 | 43 | 1.000 | PASS |
| 2.5 | 225 | 1.000 | PASS |
| 3.0 | 225 | 1.000 | PASS |
| 3.5 | 225 | 1.000 | PASS |
| 4.0 | 225 | 1.000 | PASS |

## Interpretation

Perfect correspondence at all anisotropy ratios. This is mathematically expected: in R¹, H₀ persistence bars ARE the gaps between consecutive sorted points. The test confirms correct implementation of the persistence pipeline, not a non-trivial prediction.

The paper's claim becomes non-trivial when applied to higher-dimensional point clouds where the relationship between metric gaps and topological features is no longer one-to-one. The QHO serves as a sanity check — the simplest case where the prediction is tautological.

## Assets
- `assets/validation/p1_barcode_aniso_k2.png` — Barcode for anisotropic k=2.0
- `assets/validation/p1_barcode_iso_k05.png` — Barcode for isotropic k=0.5
- `assets/validation/p1_anisotropy_sweep.png` — Sweep results chart
- `output/atft_validation/p1_qho_analysis.json` — Raw results

## Reproduce
```bash
.venv/bin/python atft/experiments/p1_qho_validation.py
```
