# ATFT Falsification Criteria

**Frozen:** 2026-03-17, commit time.
**Status:** Pre-committed before any K=100+ data is collected.
**Authors:** Blake Jones, Claude (Opus 4.6)

These thresholds are frozen at commit time and MUST NOT be modified after
K=100 data collection begins. The integrity of the experimental design
depends on pre-registration of success and failure criteria.

---

## 1. Definitions

| Symbol | Definition |
|--------|-----------|
| sigma*(K) | The sigma value at which S(sigma, epsilon) is maximized for fiber dimension K |
| C(K) | Contrast ratio: (S(sigma*) - S_min) / S(sigma*), measuring peak sharpness |
| R(K) | Discrimination ratio: S_zeta(sigma*) / S_random(sigma*), measuring arithmetic signal |
| C_GUE(K) | Contrast ratio computed on GUE control point cloud |
| width(K) | Full width at half maximum of the S(sigma) peak |

---

## 2. Framework Falsification

**Question:** Is the ATFT construction itself flawed?

If any of these criteria are triggered, the gauge-theoretic sheaf framework
does not reliably detect arithmetic structure, regardless of whether RH is
true. The framework should be abandoned or fundamentally redesigned.

| # | Criterion | Threshold | Interpretation |
|---|-----------|-----------|----------------|
| F1 | Peak migration failure | sigma*(K=100) > 0.60 or sigma*(K=100) < 0.40 | The spectral peak is not approaching the critical line. The gauge connection does not encode sigma=0.5 as special. |
| F2 | Contrast collapse | C(K=100) < C(K=50) | Adding more primes to the connection makes the signal weaker, not stronger. The Fourier sharpening hypothesis is wrong. |
| F3 | Discrimination collapse | R(K=100) < 10 | The arithmetic signal (zeta vs random) has vanished at higher K. The 670x discrimination at K=20 was a finite-size artifact. |
| F4 | GUE develops peak | C_GUE(K=100) > 0.5 * C_zeta(K=100) | The GUE control (which has matching spectral statistics but no arithmetic content) shows a comparable peak. The signal is statistical, not arithmetic. |

**Action if triggered:** Report as a negative result. Document which criterion
failed and at what values. A rigorous negative result is a genuine contribution.

---

## 3. RH Falsification Under ATFT

**Question:** Does ATFT provide evidence against the Riemann Hypothesis?

These criteria assume the framework is valid (Section 2 criteria not triggered)
but the data does not support RH.

| # | Criterion | Threshold | Interpretation |
|---|-----------|-----------|----------------|
| R1 | Peak converges away from 0.5 | sigma*(K) converges to L where \|L - 0.5\| > 0.02 as K grows | The critical line is not the unique spectral minimum. The arithmetic structure encoded by primes points to a different sigma value. |
| R2 | Peak width does not narrow | width(K=200) >= width(K=100) | No phase transition is forming. The peak is not sharpening into a wall at sigma=0.5 as K grows. |
| R3 | Scaling exponent non-positive | alpha <= 0 in C(K) ~ K^alpha power-law fit | The contrast does not diverge with K. The framework has finite spectral resolution and cannot probe the K -> infinity limit. |

**Action if triggered:** Report as "ATFT does not support RH" with quantified
confidence intervals. This does NOT prove RH is false — only that this
particular framework cannot distinguish sigma=0.5 from nearby values.

---

## 4. Positive Evidence Thresholds

**Question:** Does the data support the Riemann Hypothesis under ATFT?

These criteria define what constitutes positive evidence. Meeting all four
at K=100 would be a strong (but not conclusive) result worthy of publication.

| # | Criterion | Threshold | Interpretation |
|---|-----------|-----------|----------------|
| P1 | Peak migration on track | 0.45 <= sigma*(K=100) <= 0.52 | The peak position is consistent with convergence toward sigma=0.5. The slight asymmetry (wider range below 0.5) accounts for the K=50 observation of sigma* ~ 0.40-0.50. |
| P2 | Contrast growing | C(K=100) > 1.5 * C(K=50) | The spectral peak is sharper at K=100 than K=50, confirming the Fourier sharpening hypothesis. The 1.5x threshold is conservative. |
| P3 | Discrimination growing | R(K=100) > R(K=50) | The arithmetic signal (zeta vs random controls) strengthens with more primes. |
| P4 | Bandwidth propagation | Spectral turnover observed at eps=2.0 by K=200 | The sharpening phenomenon, first seen at eps=5.0 (K=50), propagates to finer topological scales. This is the hallmark of a genuine phase transition. |

**Strength of evidence:**
- P1 alone: weak (peak could still drift)
- P1 + P2: moderate (sharpening confirmed)
- P1 + P2 + P3: strong (arithmetic origin confirmed)
- P1 + P2 + P3 + P4: very strong (phase transition forming)

---

## 5. Existing Data Summary

For reference, the existing experimental results that inform these thresholds:

| K | Primes | eps=5.0 Behavior | eps=3.0 Behavior | sigma* | R (discrimination) |
|---|--------|-----------------|-----------------|--------|-------------------|
| 20 | 8 | Monotonic rise | Monotonic rise | Not observed | 670x |
| 50 | 15 | First turnover | Monotonic rise | ~0.40-0.50 | (pending full N) |
| 100 | 25 | (2 pts only) | Signal reversal | (pending) | (pending) |

---

## 6. Protocol

1. This document is committed to version control BEFORE K=100 data collection.
2. The thresholds in Sections 2-4 are FROZEN and must not be modified after
   this commit, even if preliminary results suggest adjustments.
3. Each criterion will be evaluated with bootstrap confidence intervals
   (1000 resamples of control trials) to quantify uncertainty.
4. Results will be reported honestly regardless of whether they support or
   refute the hypothesis. Both positive and negative results are publishable.
5. The git commit hash of this document will be cited in the paper to
   establish pre-registration.
