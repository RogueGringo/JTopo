# ATFT Validation Program — From Scratch

> Seven predictions. Seven tests. Zero inherited claims. Every number traced to code that runs.

**Date:** 2026-03-24
**Paper:** `docs/framework_theories/adaptive_topological_field_theory.pdf`
**Standard:** V&V per ASME V&V 10-2019 (adapted for computational mathematics)

---

## Scope

The ATFT paper makes 7 testable predictions. This program tests all 7 independently, documents every phase, and reports results objectively. If a prediction fails, we report that. If it passes, we report the conditions under which it passes and the conditions under which it might not.

No prediction is assumed true. No result from previous sessions is reused without re-execution.

---

## Execution Order (tractability → impact)

| Phase | Prediction | Why This Order | Estimated Effort |
|-------|-----------|---------------|-----------------|
| **P1** | 5: Quantum Harmonic Oscillator | Simplest point cloud. Pure math. No GPU needed. | 2 hours |
| **P2** | 4: ker(L_F) scaling analysis | We have the engine. Need systematic sweep. | 3 hours GPU |
| **P3** | 6+7: Betti curves + Gini trajectory | Computed from P2 data. Analysis, not new compute. | 2 hours |
| **P4** | 3: LLM hidden state analysis | Needs model inference. Local GPU. 4 small models. | 6 hours |
| **P5** | 1+2: SU(2) lattice gauge + instantons | New physics code. Heat bath. Feature map. Full pipeline. | 12 hours |

**Total estimated: ~25 hours of work + GPU time**

---

## P1: Quantum Harmonic Oscillator (Prediction 5)

### Paper claim
> Apply the adaptive operator to the point cloud of energy levels in R¹. The onset scales at which mergers occur encode the gap structure. The longest-lived H₀ features correspond to the largest energy gaps.

### Test protocol
1. **Generate spectrum:** V(x,y) = 2kx² + ½ky² (anisotropic, k=1). E(nₓ,nᵧ) = ℏ√(k/m)(2nₓ + nᵧ + 3/2). First 100 energy levels.
2. **Build point cloud:** Treat energy levels as points in R¹.
3. **Run H₀ persistence:** Standard Vietoris-Rips on R¹ (equivalent to sorting gaps and merging nearest neighbors).
4. **Extract barcode:** Each bar (birth, death) = one connected component's lifetime.
5. **Compare:** Do the longest bars correspond to the largest spectral gaps?
6. **Vary anisotropy:** Sweep k_ratio = k_x/k_y from 1.0 to 4.0. Does the persistence diagram track the degeneracy structure?

### Verification criteria
- [ ] Barcode bars ranked by length match gaps ranked by size (Spearman ρ > 0.9)
- [ ] Isotropic case (k_x = k_y) produces degenerate levels → distinct persistence signature
- [ ] Anisotropic case lifts degeneracy → persistence signature changes measurably

### Output
- `output/atft_validation/p1_qho_barcode.json` — persistence diagrams
- `output/atft_validation/p1_qho_analysis.json` — gap-bar correlation
- `output/atft_validation/p1_qho_figures/` — barcode plots, gap comparison

---

## P2: Sheaf Laplacian Kernel Scaling (Prediction 4)

### Paper claim
> On-shell SU(2) configurations have dim ker(L_F) > 0 at the critical filtration scale ε_c.

### What we know already
Lambda_1 scales as K^(-0.27) for zeta zeros. ker = 0 at all points tested. The transition appears to be a crossover, not a discontinuity.

### Test protocol (fresh, systematic)
1. **Sweep K:** K = 50, 100, 200, 400. All at sigma=0.5, eps=3.0. Zeta + GUE + Random.
2. **Record lambda_1 through lambda_10** for each (K, source) pair.
3. **Fit power law:** lambda_i = C_i * K^(alpha_i) for each eigenvalue index i.
4. **Compare alpha across sources:** Does zeta's alpha differ from GUE's? From Random's?
5. **Epsilon sweep at K=200:** eps = 1.0, 2.0, 3.0, 4.0, 5.0 (matrix-free for large eps if needed).
6. **Threshold analysis:** At what tau does ker > 0 first appear for each source?

### Verification criteria
- [ ] Power law alpha_1(Zeta) < alpha_1(GUE) < alpha_1(Random) (zeta approaches zero fastest)
- [ ] Alpha is stable across eigenvalue indices (the premium is uniform)
- [ ] Epsilon sweep shows consistent hierarchy at all scales (not just eps=3.0)

### Output
- `output/atft_validation/p2_kernel_scaling.json` — eigenvalues per (K, source, eps)
- `output/atft_validation/p2_powerlaw_fits.json` — alpha, C per source
- `output/atft_validation/p2_figures/` — scaling plots, epsilon sweep

---

## P3: Betti Curves + Gini Trajectory (Predictions 6+7)

### Paper claims
> (6) The Betti curve β₁(ε) is sufficient to detect phase transitions.
> (7) Gini trajectory (how G evolves across ε) is the strongest predictor of system quality.

### Test protocol
1. **Compute Betti curves β₀(ε)** across full epsilon range [0, ε_max] for K=200 Zeta, GUE, Random, EvenSpaced.
2. **Compute Gini coefficient G(ε)** of the eigenvalue distribution at each epsilon.
3. **Extract waypoint signatures:** onset scale ε*, max topological derivative δ₁, Gini at onset G(ε*), Gini derivative dG/dε|ε*.
4. **Compare waypoint signatures across sources:** Do they discriminate Zeta from controls?
5. **Test shape-over-count:** Is the Gini trajectory more discriminative than the raw Betti number?

### Verification criteria
- [ ] Waypoint signatures differ significantly between Zeta and GUE
- [ ] Gini trajectory slope is positive (hierarchifying) for Zeta, flat for Random
- [ ] Shape (Gini) discriminates better than count (Betti) — quantified by Fisher discriminant ratio

### Output
- `output/atft_validation/p3_betti_curves.json`
- `output/atft_validation/p3_gini_trajectories.json`
- `output/atft_validation/p3_waypoint_signatures.json`
- `output/atft_validation/p3_figures/`

---

## P4: LLM Hidden State Analysis (Prediction 3)

### Paper claims
> (3) Phase transition in hidden states is universal across architectures (r > 0.9).
> (4) Gini trajectory correlates with reasoning accuracy.
> Cited r = 0.935 across four models.

### Test protocol (from scratch — no inherited data)
1. **Select 4 models:** LFM-1.2B (if available locally), SmolLM2-1.7B, Qwen2.5-1.5B, TinyLlama-1.1B.
2. **Generate prompts:** 50 structured prompts at varying complexity (τ = token count: 10, 50, 100, 200, 500).
3. **Extract hidden states:** Layer-by-layer activations for each prompt. Save as numpy arrays.
4. **Build point clouds:** PCA-reduce each layer's hidden states to d=50 dimensions.
5. **Run adaptive operator:** For each (model, layer, prompt), compute Betti curve β₁(ε) and Gini trajectory.
6. **Detect phase transition:** Find onset scale ε*(τ) as a function of prompt complexity τ.
7. **Cross-model correlation:** Compute Pearson r of onset scale vectors across model pairs.
8. **Gini-accuracy correlation:** If ground truth is available, correlate Gini trajectory slope with output quality.

### Verification criteria
- [ ] β₁ shows discontinuous jump as prompt complexity τ crosses a threshold
- [ ] Cross-model correlation r > 0.9 for onset scale vectors
- [ ] Gini trajectory slope sign predicts output quality direction

### Hardware requirement
- RTX 5070 12GB — sufficient for 1-2B models with 4-bit quantization
- Models downloaded to local storage (no cloud inference)

### Output
- `output/atft_validation/p4_llm_hidden_states/` — per-model activation data
- `output/atft_validation/p4_phase_transitions.json`
- `output/atft_validation/p4_cross_model_correlation.json`
- `output/atft_validation/p4_figures/`

---

## P5: SU(2) Lattice Gauge Theory (Predictions 1+2)

### Paper claims
> (1) Onset scale ε*(β) exhibits discontinuous jump at β_c ≈ 2.30 on 16³×4 lattice.
> (2) Parity-complete feature map distinguishes Q=+1 from Q=-1 instantons.

### Test protocol

#### P5a: Confinement-Deconfinement (Prediction 1)
1. **Implement SU(2) heat bath:** Generate thermalized lattice configurations on 16³×4 lattice.
2. **Sweep coupling β:** β = 1.0, 1.5, 2.0, 2.1, 2.2, 2.25, 2.30, 2.35, 2.4, 2.5, 3.0, 4.0.
3. **Generate N_configs = 100 configurations per β** (after 1000 thermalization sweeps, save every 10th).
4. **Apply parity-complete feature map:** φ(x) = (s_μν(x), q_μν(x)) ∈ R¹² per lattice site.
5. **Build point cloud:** All lattice sites' feature vectors form the point cloud.
6. **Run adaptive operator:** Compute β₁(ε; β) for each coupling.
7. **Extract onset scale ε*(β).**
8. **Test:** Does ε*(β) show discontinuity at β ≈ 2.30?

#### P5b: Instanton Discrimination (Prediction 2)
1. **Generate BPST instanton configurations** on 12⁴ lattice at Q = 0, +1, -1, +2.
2. **Apply parity-complete feature map** (including q_μν = ½ ImTr P_μν).
3. **Compute adaptive Betti curves** for each Q value.
4. **Test:** Does Q=+1 produce different persistence signature from Q=-1?
5. **Test:** Does |Q|=1 produce different normalized Betti curve from |Q|=2?

### Verification criteria
- [ ] ε*(β) shows measurable discontinuity or steep transition at β ∈ [2.2, 2.4]
- [ ] Gini trajectory transitions from flat (confined) to hierarchical (deconfined) at β_c
- [ ] Q=+1 and Q=-1 produce mirror-image persistence diagrams
- [ ] ε*(|Q|) shifts systematically with |Q|

### Implementation notes
- SU(2) heat bath: standard Creutz algorithm, well-documented
- Feature map: Construction 2.3 from paper — gauge-invariant by Proposition 2.4
- This is the most code-intensive phase — new physics module needed

### Output
- `atft/lattice/` — new module for lattice gauge theory
- `output/atft_validation/p5_lattice_configs/` — generated configurations
- `output/atft_validation/p5_confinement.json` — onset scales per beta
- `output/atft_validation/p5_instanton.json` — persistence signatures per Q
- `output/atft_validation/p5_figures/`

---

## Documentation Standard

Every test produces:
1. **Phase log** — timestamped record of what was run, with what parameters, on what hardware
2. **Raw data** — JSON artifacts with all computed values
3. **Analysis** — statistical tests, correlations, comparisons
4. **Verdict** — PASS / FAIL / INCONCLUSIVE with specific criteria cited
5. **Figures** — publication-quality plots for each result

No result is stated without a source file. No source file is cited without a reproduction command.

---

## Success Criteria

| Prediction | PASS if | FAIL if |
|-----------|---------|---------|
| 1. SU(2) confinement | ε*(β) discontinuity at β ∈ [2.2, 2.4] | ε*(β) is smooth through β_c |
| 2. Instanton discrimination | Q=±1 persistence signatures are mirror-distinct | Indistinguishable signatures |
| 3. LLM cross-model correlation | r > 0.9 across ≥3 model pairs | r < 0.7 for any pair |
| 4. ker(L_F) scaling | Zeta alpha < GUE alpha (approaches zero faster) | Zeta alpha ≥ GUE alpha |
| 5. QHO gap-bar correspondence | Spearman ρ > 0.9 between gap size and bar length | ρ < 0.7 |
| 6. Betti curve discrimination | Waypoint signatures differ (p < 0.01) between sources | No significant difference |
| 7. Gini trajectory prediction | Gini slope sign predicts quality | No correlation |

---

*This program tests the framework the way the framework tests configurations: look at what's there, compute the topology, check if the waypoint constraints are satisfied. If they are, the theory is on-shell. If they're not, we say so.*
