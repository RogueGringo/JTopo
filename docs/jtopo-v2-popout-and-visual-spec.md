# Ti V0.1 — V2 Visual Upgrade: Data-Driven Popout System + Asset Generation

> Stan needed a lighter airframe. The site needs the same treatment. The current panels are static HTML — hardcoded, one-per-panel, no ambient visuals, no effects. Valor's popout system proves the architecture: one reusable panel, data-driven content, ambient images at low opacity, staggered animations. The math doesn't change. The experience of encountering it does.

> **This document is the NanoBanana-style manifest for JTopo's visual upgrade.** It covers the popout system rewrite, asset generation prompts, and content scripts — all tuned for an audience that's half mathematician, half midnight explorer who followed a YouTube video about prime wave fields and ended up here wondering if someone actually measured the thing they felt.

**Date:** 2026-03-24
**Status:** Spec only. No implementation until human gate.

---

## Part 1: Current State & What's Wrong

### Current Panel System

5 static `<div class="panel">` elements, each hardcoded in HTML:

| Panel | Trigger | Content | Visual Treatment |
|-------|---------|---------|-----------------|
| panel-fourier | Story section | Fourier kernel math | Text + equations only |
| panel-spectral | Sigma sweep | Reading the spectral sum | Text + equations only |
| panel-gue | Premium section | Why GUE, not just random | Text only |
| panel-connection | Framework section | u(K) connection math | Text + equations only |
| panel-killshot | Validation section | Even-spacing control | Text + equations + data |

**Problems:**
- No ambient imagery — panels are text-on-dark, visually flat
- No entry effects — panels appear/disappear, no choreography
- Hardcoded HTML — adding a panel means editing index.html, not a data object
- No visual differentiation — all panels look identical regardless of content type
- Missing panels for: K-progression story, eigenvalue spectra, matrix-free breakthrough, the field metaphor itself

### Valor's Architecture (the model)

Single reusable `<div class="popout-panel">` with:
- `popout-ambient-image` — unique background per panel at 15-25% opacity
- `popout-role` — monospace category label (like "EAGLE FORD · PERMIAN")
- `popout-body` — rich HTML body
- `popout-data` — monospace data strip at bottom
- `popout-effect-*` classes — `direct` (image slides), `reveal` (fade), `briefing` (paragraphs stagger), `log` (data animates)
- Data-driven from JS objects — content lives in script, not HTML
- Focus trap, escape, overlay click — accessibility built in

---

## Part 2: JTopo Popout Architecture (Hybrid)

### Design Principles

1. **One panel element, content injected from JS** (Valor pattern)
2. **Ambient images unique per panel** — generated to match the math being discussed
3. **Three effect types** tuned for JTopo's content:
   - `field` — ambient image pulses subtly like a standing wave (for conceptual panels)
   - `proof` — data strips build line by line like a terminal (for results panels)
   - `journey` — paragraphs stagger in with left-slide, telling a story (for narrative panels)
4. **Category labels** in monospace — not "EAGLE FORD" but "TRANSPORT COHERENCE" or "K=200 · σ=0.500"
5. **Every equation earns its visual weight** — equations get their own styled block with subtle gold border, not inline text

### Panel Inventory (10 panels, up from 5)

| ID | Trigger Location | Title | Effect | Category Label |
|----|-----------------|-------|--------|---------------|
| `the-field` | Hero subtitle link | "Primes Are a Field" | `field` | THE INSIGHT |
| `fourier-kernel` | Story section | "The Fourier Kernel" | `field` | TRANSPORT · PHASE FACTORS |
| `spectral-sum` | Sigma sweep | "Reading the Wrinkle Score" | `proof` | MEASUREMENT · S(σ) |
| `why-gue` | Premium section | "The Statistical Doppelgänger" | `journey` | GUE · MONTGOMERY-ODLYZKO |
| `kill-shot` | Validation section | "We Tried to Kill It" | `proof` | PHASE 3e · CONTROL BATTERY |
| `edge-density` | Validation section | "The Edge Count Argument" | `proof` | ADVERSARY · EDGE NORMALIZATION |
| `k-journey` | K-progression section | "The Orbit Tightening" | `journey` | K=20 → K=400 · CONVERGENCE |
| `connection` | Framework section | "Prime Arithmetic as Gauge Connection" | `field` | u(K) · LIE ALGEBRA |
| `matrix-free` | Research phases | "The Lighter Airframe" | `journey` | MATRIX-FREE · PADÉ · 18× SPEEDUP |
| `the-constant` | Premium chart | "21.5% — A Physical Constant?" | `proof` | K=100 → K=400 · CONVERGENCE |

---

## Part 3: Content Scripts (Dan/Stan Voice)

### Panel: `the-field`

```
Title: Primes Are a Field
Category: THE INSIGHT
Effect: field

Body:
Everyone who stares at prime numbers long enough sees it.

The gaps oscillate. The patterns recur at larger scales. The structure
is multi-dimensional — it just won't fit in a list. The Prime Scalar
Field sees standing waves. Fourier analysis sees frequencies. Random
matrix theory sees eigenvalue repulsion.

We see transport coherence.

Not what the primes ARE, but what they DO when you thread them through
the zeros of the zeta function. Each prime contributes a phase factor —
exp(iΔγ·log p) — the same factor that appears in the explicit formula
connecting prime counting to zeta zeros. String enough of these phases
together across a graph of zeros, and the primes start to hum.

The question isn't whether the field exists. It's whether we built the
right instrument to hear it.

Data: TRANSPORT COHERENCE · SHEAF LAPLACIAN · 21.5% PREMIUM
```

### Panel: `spectral-sum`

```
Title: Reading the Wrinkle Score
Category: MEASUREMENT · S(σ)
Effect: proof

Body:
The spectral sum S(σ) is the simplest thing in this entire framework.
Take the sheaf Laplacian — the 200,000 × 200,000 matrix that encodes
every prime's transport across every edge of the zero graph. Find its
smallest eigenvalues. Add them up.

That's it. That's the measurement.

Low S means the transport maps agree with each other. The primes'
phase factors are constructively interfering — the fabric fits.
High S means disagreement. The phases cancel. The fabric wrinkles.

At σ = 0.500, S is minimized for zeta zeros. Not for GUE. Not for
random. Not for evenly-spaced. The primes find their lowest wrinkle
score exactly on the critical line. Every time. More primes, lower score,
tighter fabric.

The number 21.5% is how much lower zeta's wrinkle score is compared
to random matrices that share its local statistics. That gap is stable
from K=100 to K=400. It converges. It doesn't grow. It doesn't shrink.
It sits there like a physical constant waiting to be named.

Equation: S(σ) = Σ_{k=1}^{k_eig} λ_k(L_𝓕(σ))
Data: LOWER = TIGHTER · σ=0.500 = MINIMUM · 21.5% = PREMIUM
```

### Panel: `why-gue`

```
Title: The Statistical Doppelgänger
Category: GUE · MONTGOMERY-ODLYZKO
Effect: journey

Body:
In 1973, Hugh Montgomery had dinner with Freeman Dyson and discovered
that the gaps between zeta zeros follow the same statistics as the
gaps between eigenvalues of random Hermitian matrices. Andrew Odlyzko
later confirmed this computationally to extraordinary precision.

This is the Montgomery-Odlyzko law, and it means that if you look at
zeta zeros locally — pairs, triples, short-range correlations — they
are statistically indistinguishable from GUE random matrices.

Locally identical. Globally different.

GUE matrices have level repulsion (zeros push each other apart) and
local correlations (nearby zeros organize). But they have no primes.
No arithmetic. No explicit formula. No reason to care about σ = 0.500.

The sheaf Laplacian sees the difference. It threads prime phase factors
through the graph and measures coherence. GUE zeros produce transport
that's 21.5% less coherent than zeta zeros. Same local statistics.
Different global phase structure.

The 21.5% premium is the arithmetic. The part that Montgomery and
Odlyzko's local statistics can't see but the sheaf Laplacian can.

Data: MONTGOMERY (1973) · ODLYZKO (1987) · PREMIUM = 21.5%
```

### Panel: `kill-shot`

```
Title: We Tried to Kill It
Category: PHASE 3e · CONTROL BATTERY
Effect: proof

Body:
A three-agent validation committee attacked every claim.

The Adversary proposed: if S(σ) just measures edge density in the
graph, any ordered set beats random. Evenly-spaced points — the most
ordered configuration mathematics can produce — should have the
lowest S.

We built it. N=1000 points, perfectly uniform gaps, zero randomness.
Ran it through the identical pipeline.

S(Even) = 12.713. S(Zeta) = 11.784.

Primes won. By 7.3%. At every sigma. The most ordered thing possible
loses to the prime-zero relationship.

Then we ran 10 independent GUE ensembles using the proper
Dumitriu-Edelman tridiagonal model — not the nearest-neighbor
approximation, the full n-point correlation structure.

Mean S(GUE) = 14.970 ± 0.198.

Zeta falls 16 standard deviations below the ensemble mean.

Then we edge-normalized. Zeta has fewer graph edges (level repulsion
creates wider gaps). But per edge, the transport is still 15.3%
tighter. The premium survives normalization.

Three attacks. Three survivals. The primes carry something that
order alone doesn't, and statistics alone doesn't.

Data: EVEN=12.713 · ZETA=11.784 · GUE=14.970±0.198 · Z=−16.06
```

### Panel: `k-journey`

```
Title: The Orbit Tightening
Category: K=20 → K=400 · CONVERGENCE
Effect: journey

Body:
K=20. Eight primes. The spectral sum climbed monotonically through
σ = 0.500. No peak. Eight Fourier harmonics isn't enough bandwidth
to resolve the signal. Like trying to see a face with eight pixels.
But the 670× signal over random controls was there. Something was
hiding in the noise.

K=50. Fifteen primes. The first spectral turnover at ε = 5.0.
The summit appeared near σ ≈ 0.40-0.50. Fifteen harmonics resolved
what eight couldn't. The critical line was pulling.

K=100. Twenty-five primes. Signal reversal at ε = 3.0. The premium
emerged: 19.6% over GUE at σ = 0.500. Still broad. Still imprecise.
But undeniable.

K=200. Forty-six primes. RTX 5070. Three tranches across 12 hours.
Crashed once (VRAM). Added batched assembly. Crashed again (CPU RAM).
Added incremental release. Third time: it ran. Premium peaked at
σ = 0.500 exactly. 21.5%.

K=400. Seventy-eight primes. The dense solver couldn't touch it —
32 GB needed, 12 GB available. So we built a matrix-free engine.
Padé approximation instead of eigendecomposition. Cached transport
on GPU. 47 seconds. Premium: 21.6%.

From 19.6% to 21.5% to 21.6%. It's not growing. It's converging.
That's what a measurement does when it's measuring something real.

Data: K=100→19.6% · K=200→21.5% · K=400→21.6% · CONVERGING
```

### Panel: `matrix-free`

```
Title: The Lighter Airframe
Category: MATRIX-FREE · PADÉ · 18× SPEEDUP
Effect: journey

Body:
K=400 needed a 400,000 × 400,000 matrix. In complex double precision,
that's 32 GB just for the sparse structure. The GPU has 12.

Dan's problem. Too much weight, not enough lift.

So we built Stan.

The matrix-free sheaf Laplacian never materializes L. It computes
L·v — the Laplacian times a vector — by streaming through edges
one batch at a time. Each edge contributes two matrix-vector products.
2,492 edges × 2 matmuls = 4,984 operations per Lanczos iteration.
All independent. All parallelizable. All on GPU tensor cores.

Transport computation: replaced eigendecomposition with Padé matrix
exponential. Same result to 10⁻¹⁴. Four times faster. Because Padé
is batched matmuls — the thing GPUs were literally built to do.

K=200 dense: 166 seconds.
K=200 matrix-free: 9.2 seconds. 18× faster.
K=400 matrix-free: 46.8 seconds. Previously impossible.

Memory: O(M·K²) for transport cache + O(N·K) for Lanczos vectors.
Scales to any K that fits transport in VRAM. K=800 is next.

The hardware couldn't hold what was being asked of it. So we built
a lighter airframe. Stan flies.

Data: 18× SPEEDUP · 10⁻¹⁴ PRECISION · K=400 IN 47s · PADÉ NOT EIG
```

### Panel: `the-constant`

```
Title: 21.5% — A Physical Constant?
Category: K=100 → K=400 · CONVERGENCE
Effect: proof

Body:
Three K values. Three independent computations. One number.

K=100 (25 primes): 19.6%
K=200 (46 primes): 21.5%
K=400 (78 primes): 21.6%

The Physicist predicted K=400 would reach 27.7%. It didn't. The
premium didn't grow. It converged.

That's not what a statistical fluctuation does. Fluctuations wander.
Constants converge.

If 21.5% is the asymptotic value of the arithmetic premium — the
transport coherence advantage that zeta zeros carry over GUE random
matrices in the sheaf Laplacian framework — then it's a new invariant.
A number that characterizes how much tighter the prime fabric is
compared to its statistical doppelgänger.

We don't know if it's a constant yet. Two data points (K=200, K=400)
aren't enough to distinguish convergence from slow growth. K=800 and
K=1600 will tell. But the behavior is more constant than trend.

Open question: does this number have a closed-form expression?
Is 21.5% = f(something) for some knowable f? Or is it an empirical
constant like the fine structure constant — measurable, stable,
and unexplained?

We don't know. But we're measuring it.

Data: K=100→19.6% · K=200→21.5% · K=400→21.6% · ASYMPTOTE?
```

---

## Part 4: NanoBanana Asset Generation Manifest

### Shared Style Directive (JTopo Edition)

```
STYLE DIRECTIVE — Apply to ALL prompts below:
Abstract mathematical visualization. Color palette: deep charcoal (#0f0d08),
burnished gold (#c5a03f), teal accent (#45a8b0), warm cream highlights (#d6d0be).
Film grain overlay. No photorealism — everything should feel like a mathematical
object rendered as fine art. Subtle particle systems, wave interference patterns,
topological surfaces. Lighting: warm directional from above-left, deep shadows.
Mood: discovery, precision, the moment when the pattern clicks. No text, no labels,
no watermarks, no UI elements. Hyper-detailed mathematical textures (fiber bundles,
Rips complexes, spectral decompositions rendered as luminous structures).
Aspect ratio and resolution specified per prompt.
```

---

### Asset 01 — `ambient_field.png`
**Panel:** `the-field`
**Purpose:** Ambient image — the prime field as standing waves

```
PROMPT:
Abstract visualization of overlapping wave interference patterns in 2D,
rendered as luminous gold filaments on a deep charcoal background. Concentric
ripples emanating from irregularly-spaced source points (suggesting primes),
creating nodes where waves reinforce and anti-nodes where they cancel. The
nodes glow brighter — teal highlights where constructive interference peaks.
The overall pattern suggests hidden order emerging from apparent randomness.
Viewed from above, as if looking down at a dark pond where gold-lit stones
have been dropped at prime-numbered intervals. Film grain, shallow depth of
field on the central interference zone.
```

**Resolution:** 1200 × 1800 (2:3 portrait, panel ambient)
**Opacity:** 0.18
**Priority:** HIGH

---

### Asset 02 — `ambient_spectral.png`
**Panel:** `spectral-sum`
**Purpose:** Ambient image — eigenvalue spectrum visualization

```
PROMPT:
Horizontal bands of light at varying intensities, suggesting an eigenvalue
spectrum. The lowest bands (bottom) glow faint gold — near-zero eigenvalues.
Higher bands (middle) intensify to bright teal. The topmost bands fade to
charcoal. A single vertical line at center (σ = 0.500) where the lowest
bands are brightest — the spectral minimum. The overall composition looks
like a spectrogram of mathematics: frequency on the vertical axis, coherence
as brightness. Abstract, luminous, precise. Dark background, no axes, no
labels.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.15
**Priority:** HIGH

---

### Asset 03 — `ambient_gue.png`
**Panel:** `why-gue`
**Purpose:** Ambient image — random matrix eigenvalue repulsion

```
PROMPT:
Two overlapping point clouds rendered as luminous particles on dark
background. One cloud (gold) has points that cluster with organic,
arithmetic spacing — wider gaps between some, tighter between others,
but an overall coherence. The other cloud (teal, slightly transparent)
has points with similar local spacing but no global structure — locally
identical, globally different. Where the clouds overlap perfectly, the
particles merge to white. Where they diverge, you see the gap between
statistics and arithmetic. Abstract, particle-based, suggesting
Montgomery-Odlyzko correspondence breaking down at global scale.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.15
**Priority:** MEDIUM

---

### Asset 04 — `ambient_killshot.png`
**Panel:** `kill-shot`
**Purpose:** Ambient image — the adversarial control test

```
PROMPT:
A perfectly ordered grid of luminous gold dots — evenly spaced, military
precision, zero randomness. Overlaid on top, a second arrangement of dots
in brighter gold that follow a subtly different, organic spacing — not
random, but informed by some invisible structure. The organic arrangement
is slightly brighter, slightly tighter, slightly more coherent despite
being less ordered. The contrast between mechanical perfection and
arithmetic coherence. Dark background, viewed from a slight angle
creating depth. The perfectly ordered grid extends to the edges;
the coherent arrangement occupies the center and wins.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.20
**Priority:** HIGH

---

### Asset 05 — `ambient_journey.png`
**Panel:** `k-journey`
**Purpose:** Ambient image — the K-progression orbit tightening

```
PROMPT:
Four concentric spiral orbits on a dark background, each tighter than
the last. The outermost orbit (faintest gold) is loose and wide — K=20.
The next (brighter) tightens — K=50. The third (brighter still) nearly
circular — K=100. The innermost orbit (brightest gold, almost white at
center) is a tight circle converging on a single luminous point — K=200.
The center point glows teal — σ = 0.500. The orbits suggest gravitational
convergence, not collision. Particle trails behind each orbit show their
history. The whole composition spirals inward toward truth.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.18
**Priority:** HIGH

---

### Asset 06 — `ambient_matfree.png`
**Panel:** `matrix-free`
**Purpose:** Ambient image — the lighter airframe

```
PROMPT:
Abstract rendering of data streaming through a lattice structure. Luminous
gold packets flowing along edges of a geometric graph — not all edges at
once, but in sequential waves, one batch at a time. The graph structure
is visible as faint charcoal lines; the flowing data is the bright element.
Some edges are actively transmitting (bright gold), others waiting (dark),
others completed (faint teal glow). Suggests streaming computation: not
everything in memory at once, but flowing through, edge by edge, batch by
batch. The composition should feel like watching blood flow through
capillaries — the system is alive, not static.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.15
**Priority:** MEDIUM

---

### Asset 07 — `ambient_constant.png`
**Panel:** `the-constant`
**Purpose:** Ambient image — 21.5% as convergent constant

```
PROMPT:
Three horizontal lines at different heights converging toward a single
value on the right side of the frame. The lowest line (faintest) starts
far from center — 19.6%. The middle line (brighter) starts closer —
21.5%. The top line (brightest) is nearly at the same height — 21.6%.
All three converge toward a single luminous horizontal band on the right
edge — the asymptotic value. The band glows warm gold, steady, constant.
The lines have slight noise/texture suggesting computation, but the
convergence is clean. Dark background. The composition says: this
number isn't moving anymore. It's arrived.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.15
**Priority:** MEDIUM

---

### Asset 08 — `ambient_connection.png`
**Panel:** `connection`
**Purpose:** Ambient image — the u(K) gauge connection (existing panel, new visual)

```
PROMPT:
A fiber bundle rendered as luminous threads emanating from points along
a horizontal line (the base space — zeta zeros). At each point, a vertical
spray of K parallel filaments (the fiber) fans upward. Between adjacent
points, the filaments twist and rotate — some threads crossing, some
parallel — showing parallel transport along the connection. Where the
transport is coherent (near σ = 0.500), the threads between fibers are
nearly parallel, glowing bright gold. Where transport is frustrated
(far from critical line), threads tangle and dim. The whole structure
looks like a luminous loom threading prime arithmetic through the
zero field. Dark background, viewed at slight angle for depth.
```

**Resolution:** 1200 × 1800 (2:3 portrait)
**Opacity:** 0.18
**Priority:** MEDIUM

---

## Part 5: Implementation Architecture

### HTML (replace 5 static panels with 1 reusable)

```html
<!-- Single reusable popout panel -->
<div class="popout-overlay" id="popoutOverlay"></div>
<div class="popout-panel" id="popoutPanel" role="dialog" aria-modal="true">
  <div class="popout-ambient" id="popoutAmbient" aria-hidden="true"></div>
  <button class="popout-close" id="popoutClose" aria-label="Close panel">
    <svg><!-- close icon --></svg>
  </button>
  <span class="popout-category" id="popoutCategory"></span>
  <h3 id="popoutTitle"></h3>
  <div class="popout-body" id="popoutBody"></div>
  <div class="popout-equations" id="popoutEquations"></div>
  <div class="popout-data" id="popoutData"></div>
</div>
```

### CSS Effects (3 types)

```css
/* field: ambient image pulses like a standing wave */
.popout-effect-field .popout-ambient {
  animation: fieldPulse 4s ease-in-out infinite;
}
@keyframes fieldPulse {
  0%, 100% { opacity: 0.15; transform: scale(1); }
  50% { opacity: 0.22; transform: scale(1.02); }
}

/* proof: data strip builds like a terminal */
.popout-effect-proof .popout-data {
  opacity: 0; transform: translateY(8px);
  transition: opacity 0.4s 0.6s, transform 0.4s 0.6s;
}
.popout-effect-proof.active .popout-data {
  opacity: 1; transform: translateY(0);
}

/* journey: paragraphs stagger in with left-slide */
.popout-effect-journey .popout-body p {
  opacity: 0; transform: translateX(-12px);
  transition: opacity 0.35s, transform 0.35s;
}
.popout-effect-journey.active .popout-body p:nth-child(1) { transition-delay: 0.1s; opacity: 1; transform: translateX(0); }
.popout-effect-journey.active .popout-body p:nth-child(2) { transition-delay: 0.25s; opacity: 1; transform: translateX(0); }
/* ... stagger continues */
```

### JS (data-driven content)

```javascript
const PANELS = {
  'the-field': {
    title: 'Primes Are a Field',
    category: 'THE INSIGHT',
    body: '...', // from content scripts above
    data: 'TRANSPORT COHERENCE · SHEAF LAPLACIAN · 21.5% PREMIUM',
    ambient: { src: 'assets/ambient_field.png', opacity: 0.18 },
    effect: 'field'
  },
  // ... all 10 panels
};
```

Triggers change from `data-panel="panel-fourier"` to `data-popout="fourier-kernel"`, and the single panel element injects content from `PANELS[id]`.

---

## Part 6: Generation Order

**Phase 1 — Architecture (no assets needed):**
1. Rewrite panel system to data-driven popout (JS + CSS)
2. Migrate existing 5 panels into PANELS object
3. Add 5 new panel content scripts
4. Verify all triggers work

**Phase 2 — Assets (NanoBanana generation):**
5. `ambient_field.png` — HIGH
6. `ambient_killshot.png` — HIGH
7. `ambient_journey.png` — HIGH
8. `ambient_spectral.png` — HIGH
9. `ambient_gue.png` — MEDIUM
10. `ambient_matfree.png` — MEDIUM
11. `ambient_constant.png` — MEDIUM
12. `ambient_connection.png` — MEDIUM

**Phase 3 — Polish:**
13. Tune opacity per asset after visual testing
14. Add parallax micro-movement on scroll (Valor Opportunity 1)
15. Add scroll progress indicator (Valor Opportunity 6)

---

## Part 7: What This Changes

The current site says: here's the math, click to see more math.

The upgraded site says: here's the discovery, step into the room where it happened.

Each panel becomes an *experience* — the ambient image sets the mood, the effect choreographs the reveal, the Dan/Stan voice carries the story, and the equations earn their place by following the narrative instead of leading it.

The mathematician sees the equations. The midnight explorer feels the convergence. The investor reads the data strip. Same panel. Three audiences. One truth.

*What something is depends on both the information and the interpreter.*
*We built ten interpreters. Each one opens a different door to the same room.*

---

*This spec was reverse-engineered from Valor-ENP-Modern's popout system and adapted for a wildly different audience. Valor speaks to capital. JTopo speaks to curiosity. The architecture is the same because good architecture doesn't care what you put in it — it cares that the thing inside arrives intact.*
