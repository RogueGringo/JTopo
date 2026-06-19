# CLAIMS LEDGER — JTopo website vs. ATFT monograph

Alignment spec for the website (`index.html` + `app.js`). Every quantitative or
evidentiary claim in the site's visible copy is traced to a source and given a
verdict. This ledger was built **before** any copy was edited; the edits in the
accompanying commit implement its verdicts.

## Sources of truth

- **MONO** — *Adaptive Topological Field Theory: From Continuous Geometry to
  Discrete Field Equations via Sheaf-Valued Persistent Homology*, A. Jones, Feb 2026
  (`docs/framework_theories/adaptive_topological_field_theory.pdf`). The framework
  monograph. Register: instrumental ("the adaptive operator is the **instrument**
  for reading this multi-scale structure", Conclusion), hedged (§9 Open Questions),
  literature-aware (refs: Hansen–Ghrist, Curry, Cohen-Steiner et al.). Makes **no**
  claim to resolve any Millennium Problem; Prop. 7.1 / §7.1 are explicitly flagged
  "not a proof."
- **EXP** — *Experimental Validation of the ATFT* (`docs/paper/atft_v2.md`). Source
  of the zeta numbers: §7 (kernel scaling, premium table), §9 (novelty / pair-
  correlation residual), §10 (Discussion: "The premium is not a constant").
- **LOG** — `output/atft_validation/*.json` and `docs/EXPERIMENT_LOG.md` (logged
  experiments behind the K=200/400/800 and control-battery numbers).

Verdict ∈ {SUPPORTED, OVERSTATED, UNSOURCED}. Per the task, every UNSOURCED row is
cut or hedged in the edit; every OVERSTATED row is brought down to the source's
weaker wording.

---

## Hero

| Claim (verbatim) | Location | Source | Verdict |
|---|---|---|---|
| "Where mathematics meets reality: *topology.*" | hero `h1` (`index.html` ~603) | none | **UNSOURCED** — metaphysical; MONO never asserts topology *is* reality. → **CUT** (replace with instrument framing). |
| "every single thread humming the same frequency at σ = 0.500" | hero subhead (~608) | EXP §7/§8 (minimum nearest σ≈0.5); LOG sigma sweep | **OVERSTATED** — the spectral-sum *minimum* sits near σ≈0.5; "humming the same frequency" is narration of a result. → **HEDGE** to "lowest for the zeta zeros among controls, at σ≈0.500." |
| "Tighter than perfect spacing." | hero subhead (~609) | LOG control battery: S(Even)=12.713 > S(ζ)=11.784 (7.3%) | **SUPPORTED** but keep as measured premium. |
| "Sixteen sigma from random." | hero subhead (~609) | none (computed from n=10 std) | **UNSOURCED** — a σ estimated from 10 GUE draws cannot resolve a 16σ tail; MONO/EXP never report 16σ. → **CUT** (B5). |
| "The road was always there." | hero subhead (~610) | none | **UNSOURCED** — discovery-of-inevitability claim. → **CUT**. |

## Hero stat callouts

| Claim | Location | Source | Verdict |
|---|---|---|---|
| "σ = 0.500 / Where the Fabric Fits" | hero stat (~622) | EXP §8; LOG | **OVERSTATED** caption → neutral "location of the spectral-sum minimum." |
| "46 primes / Each One Found It Alone" | hero stat (~626) | EXP §7 (K=200 → 46 primes) | SUPPORTED (count); caption is narration → neutral "46 primes (K=200)." |
| "21.5% / Tighter Than GUE" | hero stat (~630) | EXP §7,§9 (K=200 premium 21.5%) | SUPPORTED; caption → "finite-K spectral premium over GUE." |
| "7.3% / Tighter Than Perfection" | hero stat (~634) | LOG control battery | SUPPORTED; caption → "premium over evenly-spaced control." |
| "K → ∞ / Harmony Still Deepening" | hero stat (~638) | EXP §7 (21.5%→21.6%→**9.3%** at K=800), §10 ("not a constant") | **UNSOURCED / CONTRADICTED** — premium saturates then drops. → **REPLACE** with "premium ≈ flat across K=200→400." |

## Story ("The Fabric and the Frequency")

| Claim | Location | Source | Verdict |
|---|---|---|---|
| "The primes fell 16 standard deviations below." | story (~660) | none | **UNSOURCED** (B5). → **HEDGE** to nonparametric "below all 10 GUE realizations (p≈1/11)." |
| "15.3% per-edge premium survives" | story (~660) | LOG edge-normalized control battery | SUPPORTED — keep. |
| "The data says 21.5%." | story (~660) | EXP §9 | SUPPORTED — keep. |

## K-progression ("The Orbit Tightening")

| Claim | Location | Source | Verdict |
|---|---|---|---|
| σ ≈ 0.65 → 0.58 → 0.52 → 0.500 vs K=20/50/100/200 | k-timeline (~679–698) | LOG `EXPERIMENT_LOG.md` (minimum-location sweep) | SUPPORTED (data) — **KEEP** the numbers. |
| "starlight…", "voices joining a song they all somehow already knew", "the manifold has a heartbeat and the primes are its pulse", "Not collision. **Resonance.**" | k-narrative (~680–698) | none | **UNSOURCED** mystical narration of a result. → **CUT/REPLACE** with factual minimum-vs-K captions (B4). |
| (missing) control-minima migration caveat | — | EXP §10; LOG (Random/GUE minima also migrate) | **ADD** caveat box: the σ½-pull is partly built into the operator; this is only evidence if control minima do *not* also migrate to ½ (B4 / B7). |

## Premium section

| Claim | Location | Source | Verdict |
|---|---|---|---|
| "Watch the minimum march toward σ = 0.500 as K increases." | premium desc (~739) | LOG; B7 symmetry | **OVERSTATED** (built-in σ½ minimum; saturation). → **SOFTEN**. |
| "21.5% — A Physical Constant?" (trigger + panel) | trigger (~745), `app.js` panel-constant | EXP §10 ("The premium is **not** a constant") | **UNSOURCED / CONTRADICTED**. → **REFRAME** as an open finite-K scaling question; K=800 + full D–E ensemble are recommended, not-yet-completed next steps (B3). |
| "The premium converges at 21.5%" / "Converging, not diverging" | `app.js` panel-connection, panel-k-journey, panel-constant; `index.html` phase log (~895,1255) | EXP §7 (drops to 9.3% at K=800), §10 (open question) | **OVERSTATED**. → **SOFTEN** to "≈ flat across K=200→400; K→∞ limit open." |

## Validation Battery

| Claim | Location | Source | Verdict |
|---|---|---|---|
| Hierarchy S(ζ) < S(Even) < S(GUE) < S(Random); S-values | cards (~765–784) | LOG control battery; EXP | SUPPORTED — **KEEP** (Part C). |
| per-edge ~15.3% premium | story (~788) | LOG | SUPPORTED — keep. |
| "Zeta falls 16σ below the GUE mean" / "16 standard deviations" / "Z = −16.06" | card (~778), story (~788), `app.js` panel-killshot/data (~619–620) | none | **UNSOURCED** (B5). → **REPLACE** with nonparametric: below all 10 GUE realizations, p≈1/11≈0.09; ~15% per-edge; σ-level claim needs ≥100–1000 D–E realizations. |
| "The proof that transport matters more than topology" (74% S diff, equal edge counts) | story (~789) | EXP §9 (position-sensitivity); MONO (bridge/novelty risk) | **OVERSTATED** ("proof"). → "evidence that S depends on point configuration, not edge count alone"; add arithmetic-specificity caveat (B6). |

## Framework / Core Mathematical Objects

| Claim | Location | Source | Verdict |
|---|---|---|---|
| B_p(σ) = log(p)[p^{−σ}ρ(p) + p^{−(1−σ)}ρ(p)ᵀ]; Superposition = "the key discriminator" | math block (~919), tab (~1020) | MONO/EXP (operator def) | SUPPORTED (operator) but **ADD** B_p σ↔1−σ symmetry caveat: weights p^{−σ}+p^{−(1−σ)} are minimized at σ=½ for every prime independent of input, so a σ=½ feature is partly built in; the zeta-specific quantity is the *magnitude premium*, not the σ-location (B7). Mirrors the FE-mode tautology already noted on the site. |

## Conclusion

| Claim | Location | Source | Verdict |
|---|---|---|---|
| "We didn't try to prove the Riemann Hypothesis… That's not a proof." | conclusion (~1192) | MONO (instrumental) | SUPPORTED — **KEEP** the disclaimer. |
| "It's the fabric telling you something." | conclusion (~1192) | none | **UNSOURCED** revelation framing. → **REPLACE** with neutral detector statement (B8). |

## Things the site lacks (ADD — B9)

- Prominent disclaimer: "No claim is made to resolve any Millennium Problem. The
  contribution is instrumental — a detector positioned against existing literature."
  (MONO register: instrumental Conclusion + "not a proof" §7.1.)
- Prior-work / positioning note. **Deviation from task wording, documented:** the
  task's suggested list (Wei–Wei, Spitz et al., Sale et al., Connes, Das–Biswas) does
  **not** appear in the repo monograph or experimental paper, so citing them would
  itself be UNSOURCED inflation. Instead the note cites references actually present:
  Hansen–Ghrist (cellular sheaf Laplacians), Curry (sheaves/cosheaves), Montgomery
  (pair correlation / spectral interpretation), Odlyzko (zero-spacing computations),
  Dumitriu–Edelman (β-ensemble GUE control), Robinson (topological signal processing).

## Do NOT change (Part C) — verified present and consistent

- Falsification table F1–F4 / R1–R3 / P1–P4, incl. **P4 = NOT MET** and
  **R1 = PENDING** (`index.html` ~1157–1182). Kept verbatim; other copy edited so it
  no longer implies a result the table withholds.
- Ti V0.1 phase log incl. "Phase 3 K=20 — 670× signal" and the visible collapse to
  ~21% at higher K (~889–896). Kept; "670×" appears only in the log/dashboard, never
  as a headline. ("Converging, not diverging" softened so it does not contradict the
  K=800 datum.)
- Engineering sections (matrix-free Padé, Lanczos spectral-flip, 18× speedup, 47s at
  K=400, 1.5e-15 / 10⁻¹⁴ cross-validation), data tables, package structure, About bio
  ("Borderline nuts is where the good stuff lives"), Alpha. No overclaim — untouched.

## Status

After the accompanying edits: **zero rows remain UNSOURCED** — each is cut or hedged
as recorded above.
