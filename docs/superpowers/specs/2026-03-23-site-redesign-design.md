# Ti V0.1 Site Redesign — Design Spec

**Date:** 2026-03-23
**Status:** Draft
**Sub-project:** Results & Framing + Visual Upgrade

---

## 1. Goals

Redesign the existing `index.html` single-page site to:

1. Present the K=200 headline finding (arithmetic premium peaks at exactly σ=0.500)
2. Speak to three audience layers simultaneously without patronizing any of them
3. Add interactive data visualization so visitors can feel the data, not just read tables
4. Adopt the Chopper Stan narrative voice — unity, resonance, primes-as-travelers — layered over the existing technical framework
5. Upgrade visual components toward Valor-ENP-Modern class (particles, glass morphism, gold gradients, scroll reveals)
6. Maintain zero external JS dependencies (pure Canvas API + CSS)

## 2. Audience Architecture

Three layers, all served by the same content through progressive depth:

### Layer 1 — The Curious (no math background)
- Reads narrative. Understands through metaphor: primes as travelers finding the same road, fabric woven from irreducible numbers, resonance at the critical line.
- Gets the *why this matters* without parsing equations.
- Served by: hero copy, narrated K-progression, "logical sidestep" callout.

### Layer 2 — The Practitioner (CS/ML/data science)
- Reads technical vernacular. Understands the pipeline: point cloud → graph → sheaf → Laplacian → eigenvalues → sweep.
- Gets the *how it works* through framework tabs, interactive sigma sweep, comparison tables, code snippets.
- Served by: interactive data viz, framework section, infrastructure section.

### Layer 3 — The Mathematician (algebraic topology, analytic number theory)
- Reads formal framework. Understands u(K) connections, Vietoris-Rips filtration, sheaf cohomology, the explicit formula Fourier kernel.
- Gets the *what this implies* for the Fourier sharpening conjecture.
- Served by: deep-dive pop-out panels with full notation.

### Depth Mechanism
Each major section has a surface (Layer 1+2 visible by default) and an optional depth panel (Layer 3) that slides in from the right as a frosted glass overlay. The original content remains visible (slightly blurred) on the left so context is preserved.

## 3. Voice & Tone

### The Chopper Stan Pattern (adapted for Ti)
The voice is **unity, not conflict**. Primes are travelers finding home, not rebels fighting structure. The humor comes from the *inevitability* — forty-six independent entities agreeing on something without instructions.

Key metaphor system:
- **Fabric** = the sheaf Laplacian's global structure
- **Threads** = individual prime transport maps
- **Weave** = the superposition of all prime contributions
- **Wrinkles** = spectral sum (high S = loose fabric, low S = tight)
- **Tuning** = varying sigma across the critical strip
- **Resonance** = the minimum at σ=0.500 (where the fabric fits best)

### Voice examples

Hero: *"Forty-six primes found the same road without a map."*

K-progression narration: *"At K=20, eight primes drifted into the manifold the way starlight arrives — each from its own origin, each carrying only what it was born with."*

Logical sidestep: *"A prime number is the universe's smallest act of completeness. It has itself, it has one, and it has a quiet, total relationship with every number it will never divide into. That relationship is not absence — it's structure."*

Stats labels: "WHERE THE FABRIC FITS" / "EACH ONE FOUND IT ALONE" / "THE HARMONY IS STILL DEEPENING"

## 4. Hero Section

### Content
- **Title:** "Forty-six primes found the same road without a map."
- **Subtitle:** "Each one traveling its own geodesic through the zero field — no instructions, no shared itinerary — and every single one of them ended up humming the same frequency at σ = 0.500. Not because they had to. Because the road was always there."
- **CTAs:** "Explore the Data" (scrolls to sigma sweep) / "Read the Paper" / GitHub pill

### Stats Strip
| Value | Label |
|-------|-------|
| `σ = 0.500` | WHERE THE FABRIC FITS |
| `46 primes` | EACH ONE FOUND IT ALONE |
| `21.5%` | HOW MUCH MORE THE PRIMES RESONATE THAN NOISE |
| `K → ∞` | THE HARMONY IS STILL DEEPENING |

**Metric definition:** The 21.5% figure is `(1 - S_zeta/S_GUE) × 100` at σ=0.500, K=200, eps=3.0. Zeta spectral sum (11.784) is 21.5% lower than GUE spectral sum (15.004), meaning the prime fabric is 21.5% tighter than the statistical baseline at the critical line.

**Stats rendering:** These stats are static text, not animated counters. The existing `animateCounter` in `app.js` handles integers only; these values (decimal, percentage, symbol) should render as pre-set text with a fade-in on scroll.

### Particle Animation (Canvas API)
- Gold particles (one per prime at K=200) drift in from screen edges like embers
- Each traces a gentle curved path (bezier, not straight)
- Paths gradually braid — tributaries finding the river
- At full animation: particles form a luminous filament along a central axis (σ=0.500)
- The filament breathes softly (subtle opacity pulse)
- Performance: requestAnimationFrame, limit to 46 particles, reduce to 15 on mobile
- Fallback: if Canvas unavailable, show the existing hero background image (`./assets/researcher-doberman-lab.jpg`) with a dark gradient overlay matching `linear-gradient(135deg, oklch(from var(--color-bg) l c h / 0.9), oklch(from var(--color-primary) l c h / 0.15))`

## 5. The Story Section — "What We Did"

Three short paragraphs in Chopper Stan voice:
1. What primes are (the universe's smallest complete things, each irreducible, each encoding structure)
2. What we did (wove them into a fabric over zeta zeros and asked where the wrinkles smooth out)
3. What we found (σ=0.500, every time, more primes = tighter fabric)

Pop-out panel: **"The Fourier Kernel"** — the explicit formula `e^{i·Δγ·log(p)}`, how primes encode phase interference, why more primes = sharper resolution. Full notation for Layer 3.

## 6. K-Progression Section — "The Orbit Tightening"

### Narrated Timeline
Vertical timeline component (Valor-style animated line with gold waypoint dots):

```
K=20  (8 primes)   → σ ≈ 0.65
K=50  (15 primes)  → σ ≈ 0.58
K=100 (25 primes)  → σ ≈ 0.52
K=200 (46 primes)  → σ = 0.500  ← critical line
```

Each waypoint includes:
- Narrated text (the "starlight arrives" / "voices joining a song" / "invited not pulled" / "the manifold has a heartbeat" progression)
- A gold marker on a horizontal sigma number line showing the peak position
- Waypoint dot: green for completed runs, gold-glow for current frontier (K=200)

### Behavior
- Auto-plays on scroll-into-view (IntersectionObserver)
- Each waypoint animates in sequence with a stagger (~400ms)
- Sigma markers animate sliding left toward 0.500
- Pauses on hover for inspection
- K=200 waypoint pulses on completion

### Implementation
CSS transforms + IntersectionObserver. No Canvas needed here — DOM animation is sufficient for 4 waypoints.

## 7. Interactive Data Visualization

### 7a. Sigma Sweep — "Tune the Fabric Yourself"

A horizontal slider from σ=0.25 to σ=0.75. As the user drags:

**Spectral curve chart (Canvas API):**
- K=100: Three lines — Zeta (gold), GUE (teal), Random (muted gray). Full sigma range 0.25–0.75.
- K=200: Two lines — Zeta (gold), GUE (teal). No Random data yet (T3 pending). Sigma range 0.44–0.56 only (T1 critical zone; T2 pending).
- K=20, K=50: Data to be extracted from earlier experiment runs or interpolated from K=100 trends. If unavailable at launch, these K toggles show "data pending" state.
- When K=200 is selected and the slider is outside 0.44–0.56, the chart area outside the data range is drawn with a subtle dashed line and a "data pending — T2 tranche" label in `--color-text-faint`.
- Smooth interpolation between data points within available range
- Hover tooltip (frosted glass) shows exact values

**Fabric coherence visualization (Canvas API, above chart):**
- Abstract particle field representing fabric tightness
- At σ far from 0.5: particles spread, connections faint, threads drifting
- At σ = 0.500: particles tight, connections bright, fabric coheres
- Subtle pulse at the minimum

**K toggle:**
- Four buttons: K=20, K=50, K=100, K=200
- Switching K re-draws the curves with a smooth transition
- Watch the zeta curve sharpen and the minimum approach 0.500

**Data source:**
- Baked into an inline JS object (~4KB estimated: K=100 has 30 points × 3 sources, K=200 has 5 points × 2 sources, plus K=20/K=50 if available). No fetch calls, no external files at runtime.
- Extracted from `output/phase3c_torch_k100_results.json` (K=100, all 3 sources, full sigma range) and `output/phase3d_torch_k200_results.json` (K=200, Zeta+GUE only, critical zone only)
- K=20 and K=50 data: extract from earlier Phase 3/3b experiment logs if JSON exists, otherwise defer these K toggles to a future update

Pop-out panel: **"Reading the Spectral Sum"** — what S(σ) measures physically, Betti-0 interpretation, why minimum = maximum consistency.

### 7b. Arithmetic Premium Chart — "How Much More Than Noise"

**Multi-K line chart (Canvas API):**
- Four curves layered: K=20 (faintest), K=50, K=100, K=200 (bright gold)
- Y-axis: zeta/GUE ratio
- X-axis: sigma
- Minimum of each curve marked with a dot + sigma label
- Dots visually march toward 0.500 as K increases
- Hover tooltip shows exact ratio values

Pop-out panel: **"Why GUE, Not Just Random?"** — Random matrix theory connection, Montgomery-Odlyzko law, what the arithmetic premium isolates.

### 7c. Data Tables (existing pattern, refreshed)

The existing `data-table` component updated with K=100 and K=200 results. Sortable by sigma. Highlight cells at σ=0.500.

### Shared Visual Treatment (Valor DNA)
- Dark surface cards: `--color-surface` / `--color-border` (existing tokens)
- Glass morphism on tooltips: `backdrop-filter: blur(18px)`
- Gold gradient on zeta/prime data: `--color-primary` → `--color-primary-hover`
- Teal accent on GUE data: `--color-teal`
- Muted gray on random/noise baselines
- JetBrains Mono for all axis labels and data values
- Scroll-reveal entrance animations (existing pattern)
- No Chart.js, no D3 — pure Canvas API

## 8. Deep-Dive Pop-Out Panels

### Component Specification

**Trigger:** Teal-colored `+ See the math` link, positioned at the bottom-right of the parent section.

**Panel:**
- Slides from right edge: `transform: translateX(100%) → translateX(0)`
- `max-width: 480px`, full viewport height, fixed position
- Background: `var(--color-surface)` at 92% opacity
- `backdrop-filter: blur(24px) saturate(1.3)`
- Left border: 3px solid `var(--color-teal)`
- Padding: `--space-8`
- Close button (×) top-right
- Closes on: close button, click-outside, Escape key
- **Mobile (viewport < 640px):** Panel becomes a full-screen overlay (no max-width constraint). Close button enlarged to 48px tap target. No blur on background — full opacity panel replaces the view. Swipe-right to dismiss (optional progressive enhancement).

**Content styling:**
- Section headers: Zodiak font, `--text-lg`
- Body text: Satoshi, `--text-sm`, `--color-text-muted`
- Equations: JetBrains Mono, `--text-sm`, `--color-teal`
- Equation blocks: existing `.eq-block` component

**When panel is open:**
- Main content gets `filter: blur(2px); opacity: 0.7`
- Scroll lock on body (panel scrolls independently)
- Transition: 280ms cubic-bezier(0.16, 1, 0.3, 1)

### Panel Inventory

| ID | Section | Title | Content Summary |
|----|---------|-------|-----------------|
| `panel-fourier` | The Story | The Fourier Kernel | Explicit formula, `e^{i·Δγ·log(p)}`, phase interference, resolution vs K |
| `panel-spectral` | Sigma Sweep | Reading the Spectral Sum | S(σ) physical meaning, Betti-0, consistency interpretation |
| `panel-gue` | Arithmetic Premium | Why GUE? | RMT, Montgomery-Odlyzko, what the premium isolates |
| `panel-connection` | Framework | u(K) Connection | Lie algebra representation, transport map construction, superposition mode |

## 9. Overall Page Flow

```
 1. NAV ─────────────────────────────────────────────────
    Frosted sticky header (existing, keep as-is)

 2. HERO ────────────────────────────────────────────────
    "Forty-six primes found the same road without a map"
    ├── Particle braiding animation (Canvas)
    ├── Stats strip (4 items)
    └── CTAs: Explore / Paper / GitHub

 3. THE STORY ───────────────────────────────────────────
    "What We Did" — 3 paragraphs, Chopper Stan voice
    ├── Fabric metaphor, the question, the finding
    └── Pop-out: The Fourier Kernel

 4. K-PROGRESSION ───────────────────────────────────────
    "The Orbit Tightening"
    ├── Animated vertical timeline (K=20 → K=200)
    ├── Narrated waypoints + sigma markers
    └── Auto-play on scroll, gold glow on K=200

 5. SIGMA SWEEP ─────────────────────────────────────────
    "Tune the Fabric Yourself"
    ├── Interactive slider (σ=0.25 to 0.75)
    ├── Fabric coherence viz + spectral curve
    ├── K toggle (20/50/100/200)
    └── Pop-out: Reading the Spectral Sum

 6. ARITHMETIC PREMIUM ──────────────────────────────────
    "How Much More Than Noise"
    ├── Multi-K line chart (zeta/GUE ratio)
    ├── Peak migration dots → 0.500
    └── Pop-out: Why GUE?

 7. FRAMEWORK ───────────────────────────────────────────
    "How the Fabric Is Woven" (existing tabs, refreshed)
    ├── Tab: The Points (zeros + Vietoris-Rips)
    ├── Tab: The Threads (transport maps + primes)
    ├── Tab: The Fabric (sheaf Laplacian)
    ├── Tab: The Measurement (eigenvalues + sweep)
    └── Pop-out: u(K) Connection

 8. FALSIFICATION ───────────────────────────────────────
    "How This Could Be Wrong"
    ├── Pre-registered criteria cards (F1-F4, R1-R3, P1-P4)
    ├── Status badges (pass/partial/pending)
    └── Logical sidestep callout (frosted card)

 9. INFRASTRUCTURE ──────────────────────────────────────
    "The Hardware"
    ├── RTX 5070 / batched assembly / 12GB constraint
    ├── File tree (existing, refreshed)
    └── "Stan flies because he fits"

    ─── RESEARCH / PERSONAL BOUNDARY ───────────────────

10. ABOUT ───────────────────────────────────────────────
    B. Jones bio + portrait (existing, moved down)
    └── Alpha card (compact, within About)

11. PROJECTS ────────────────────────────────────────────
    Project grid (existing, keep as-is)

12. BLOG / NOTES ────────────────────────────────────────
    Article cards (existing, keep as-is)

13. FOOTER ──────────────────────────────────────────────
    GitHub / Paper / License / Contact / Citation-BibTeX
    "The harmony is still deepening"
```

## 10. Disposition of Existing Personal Sections

The current site includes personal/portfolio sections beyond the research content. Their fate in the redesign:

| Existing Section | Decision | Rationale |
|-----------------|----------|-----------|
| About (B. Jones bio, portrait) | **Keep, move below research** | Establishes credibility. Place after Falsification (§8), before Infrastructure. |
| Alpha (the Doberman) | **Keep, compact** | Part of the site's identity. Reduce to a single card within About, not a full section. |
| Projects grid (Chopper Stan, mpd-overwatch, etc.) | **Keep, move below About** | Shows broader work. Keep as-is but after the research content. |
| Blog/Notes (article cards) | **Keep, move to bottom** | Useful but secondary to the research narrative. Place above Footer. |
| Contact | **Keep in Footer** | Merge contact info into an expanded Footer. |
| Citation/BibTeX | **Keep, move into Falsification or Footer** | Academic credibility. BibTeX block fits in either location. |

The research narrative (§2–§8) becomes the top half of the page. Personal/portfolio content follows naturally below. Nothing is deleted — the page just reorders to lead with the headline finding.

## 11. What Changes vs. Current Site (Summary)

| Section | Change Type | Notes |
|---------|------------|-------|
| Nav | Keep | Existing frosted sticky header works |
| Hero | Rewrite | New copy, particle animation, updated stats |
| The Story (§3) | New | Chopper Stan narrative intro |
| K-Progression (§4) | New | Animated timeline, narrated waypoints |
| Sigma Sweep (§5) | New | Interactive slider + Canvas charts |
| Arithmetic Premium (§6) | New | Multi-K comparison chart |
| Framework (§7) | Refresh + Rename | Tabs renamed: "Fiber Structure" → "The Points", "Gauge Connection" → "The Threads", "Sheaf Laplacian" → "The Fabric", "Statistical Test" → "The Measurement". Content updated with K=100/200 language. |
| Results | Replace | Old phase cards → integrated into §5 and §6 |
| Falsification (§8) | New | Pre-registered criteria with honest status |
| Infrastructure (§9) | Refresh | Update file tree, add hardware specifics |
| Footer | Refresh | Add paper link, update closing line |
| Pop-out panels | New | Reusable component, 4 panels initially |

## 12. Technical Constraints

- **Single file architecture:** Everything in `index.html` + `app.js` (existing pattern)
- **Zero external JS:** No Chart.js, no D3, no React. Canvas API for charts, CSS for animations.
- **Existing design tokens:** Use the full `--color-*`, `--text-*`, `--space-*`, `--radius-*` system already in the CSS
- **Existing fonts:** Zodiak (display), Satoshi (body), JetBrains Mono (code/data) — already loaded
- **Mobile:** Particle count reduced (46 → 15), charts resize via canvas scaling, timeline stacks naturally
- **Performance:** All animations via requestAnimationFrame or CSS transforms (GPU-composited). No layout thrashing.
- **Accessibility:** Canvas charts get `aria-label` descriptions. Pop-out panels trap focus. Reduced-motion media query disables animations (existing pattern).
- **Data:** Experimental results baked into inline JS (~4KB). No fetch calls, no external data files needed at runtime.
- **Canvas colors:** CSS custom properties (`--color-primary`, `--color-teal`, etc.) are not natively readable by Canvas API. At app init, resolve all chart colors via `getComputedStyle(document.documentElement).getPropertyValue('--color-primary')` and cache in a `CHART_COLORS` object. This also handles theme switching — re-resolve on theme toggle. Hex fallbacks for the critical colors: gold `#d08a28` (dark) / `#a05c00` (light), teal `#45a8b0` (dark) / `#00848c` (light), gray `#544f3e` (dark) / `#bab4a6` (light).
- **Theme toggle:** All new components (Canvas charts, pop-out panels, particle animation) must work in both light and dark modes. Canvas redraws on theme change using the re-resolved `CHART_COLORS`. Pop-out panel opacity/blur values are the same in both themes (the surface color token handles the difference). Particle animation uses `--color-primary` resolved at draw time.
- **RunPod references:** The existing site mentions "RunPod A100/MI300X" in the Phase 4 research timeline. Remove these references — all compute is local (RTX 5070). Replace with the actual hardware posture.

## 13. Out of Scope

- Multi-page architecture (stays single-page)
- Build system or bundler
- Server-side rendering
- User accounts or persistence
- The academic paper (separate sub-project)
- K=200 T2/T3 results (will update data when available, but site doesn't block on them)
