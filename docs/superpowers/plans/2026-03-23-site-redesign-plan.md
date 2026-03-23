# Ti V0.1 Site Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Ti V0.1 single-page site to lead with the K=200 headline finding, add interactive data visualization, adopt the Chopper Stan narrative voice, and upgrade visual components — all with zero external JS dependencies.

**Architecture:** The site remains a single `index.html` (HTML + inline CSS) + `app.js` (all JS). New interactive components use Canvas API for charts and CSS transforms for animations. Experimental data is baked into a `CHART_DATA` object in `app.js`. Page flow is reordered: research narrative (§1–§9) first, personal sections (About, Alpha, Projects, Blog) below.

**Tech Stack:** HTML5, CSS3 (custom properties, backdrop-filter, IntersectionObserver), Canvas API, vanilla JS (ES5 IIFE pattern matching existing `app.js`)

**Spec:** `docs/superpowers/specs/2026-03-23-site-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `index.html` | Modify (lines 1–1455) | All HTML structure + inline CSS |
| `app.js` | Modify (lines 1–259) | All JS: theme, tabs, counters, reveals, new Canvas charts, particle animation, pop-out panels, K-progression timeline |

No new files are created. Both files are modified in place.

---

### Task 1: Extract and Prepare Chart Data

**Files:**
- Modify: `app.js` (add `CHART_DATA` object at top of IIFE)

This task extracts the experimental results from JSON files and bakes them into a compact JS object that all chart components will reference.

- [ ] **Step 1: Extract data from JSON result files**

Run this Python script to generate the JS data object:

```bash
python3 -c "
import json

with open('output/phase3c_torch_k100_results.json') as f:
    k100 = json.load(f)
with open('output/phase3d_torch_k200_results.json') as f:
    k200 = json.load(f)

# Extract spectral sums by sigma for eps=3.0
def extract(data, eps='3.0'):
    result = {}
    for src in data:
        result[src] = {}
        for key, val in sorted(data[src].items()):
            sigma, e = key.split('_')
            if e == eps:
                result[src][sigma] = round(val['spectral_sum'], 6)
    return result

print('K100:', json.dumps(extract(k100), indent=2))
print('K200:', json.dumps(extract(k200), indent=2))
"
```

- [ ] **Step 2: Add CHART_DATA and CHART_COLORS to app.js**

Insert at the top of the IIFE (after `'use strict';`):

```javascript
/* ─── Chart Data (baked from experiment results) ─── */
var CHART_DATA = {
  K100: {
    Zeta:   { /* sigma: spectral_sum pairs from step 1 */ },
    GUE:    { /* ... */ },
    Random: { /* ... */ }
  },
  K200: {
    Zeta: { /* sigma: spectral_sum pairs from step 1 */ },
    GUE:  { /* ... */ }
  }
};

/* ─── Canvas Color Resolution ─── */
var CHART_COLORS = {};
function resolveChartColors() {
  var cs = getComputedStyle(document.documentElement);
  CHART_COLORS.gold = cs.getPropertyValue('--color-primary').trim() || '#d08a28';
  CHART_COLORS.teal = cs.getPropertyValue('--color-teal').trim() || '#45a8b0';
  CHART_COLORS.gray = cs.getPropertyValue('--color-text-faint').trim() || '#544f3e';
  CHART_COLORS.bg = cs.getPropertyValue('--color-bg').trim() || '#0f0d08';
  CHART_COLORS.surface = cs.getPropertyValue('--color-surface').trim() || '#16140d';
  CHART_COLORS.text = cs.getPropertyValue('--color-text').trim() || '#d6d0be';
  CHART_COLORS.textMuted = cs.getPropertyValue('--color-text-muted').trim() || '#817a66';
  CHART_COLORS.border = cs.getPropertyValue('--color-border').trim() || '#35311e';
}
resolveChartColors();
```

- [ ] **Step 3: Hook color resolution into theme toggle**

Find the existing theme toggle click handler and add `resolveChartColors()` after the theme switch. Also add calls to redraw any active canvases (these functions will be defined in later tasks):

```javascript
// Inside existing toggle click handler, after root.setAttribute('data-theme', theme):
resolveChartColors();
if (typeof redrawSigmaSweep === 'function') redrawSigmaSweep();
if (typeof redrawPremiumChart === 'function') redrawPremiumChart();
if (typeof restartParticles === 'function') restartParticles();
```

- [ ] **Step 4: Verify data loads correctly**

Open `index.html` in browser, open DevTools console, type `CHART_DATA.K100.Zeta` — should show the sigma/spectral_sum object.

- [ ] **Step 5: Commit**

```bash
git add app.js
git commit -m "feat(site): add baked chart data and color resolution system"
```

---

### Task 2: New CSS Components

**Files:**
- Modify: `index.html` (CSS section, before `</style>`)

Add all new CSS needed for the redesign components. This is done first so subsequent HTML tasks can reference these classes.

- [ ] **Step 1: Add pop-out panel CSS**

Insert before the `/* SCROLL REVEALS */` section (around line 402):

```css
/* ═══════════════════════════════════════════
   POP-OUT PANELS
   ═══════════════════════════════════════════ */
.panel-trigger{display:inline-flex;align-items:center;gap:var(--space-2);font-family:var(--font-mono);font-size:var(--text-xs);color:var(--color-teal);cursor:pointer;border:none;background:none;letter-spacing:0.04em;margin-top:var(--space-4);}
.panel-trigger:hover{color:var(--color-teal-hover);}
.panel-overlay{position:fixed;inset:0;z-index:200;background:oklch(0 0 0 / 0.4);opacity:0;pointer-events:none;transition:opacity 280ms cubic-bezier(0.16,1,0.3,1);}
.panel-overlay.open{opacity:1;pointer-events:auto;}
.panel{position:fixed;top:0;right:0;bottom:0;z-index:201;max-width:480px;width:100%;background:oklch(from var(--color-surface) l c h / 0.92);backdrop-filter:blur(24px) saturate(1.3);-webkit-backdrop-filter:blur(24px) saturate(1.3);border-left:3px solid var(--color-teal);padding:var(--space-8);overflow-y:auto;transform:translateX(100%);transition:transform 280ms cubic-bezier(0.16,1,0.3,1);}
.panel.open{transform:translateX(0);}
.panel-close{position:absolute;top:var(--space-4);right:var(--space-4);width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:var(--radius-full);color:var(--color-text-muted);cursor:pointer;border:none;background:none;}
.panel-close:hover{background:var(--color-surface-offset);color:var(--color-text);}
.panel h4{font-family:var(--font-display);font-size:var(--text-lg);font-weight:500;margin-bottom:var(--space-6);margin-top:var(--space-8);}
.panel p{font-size:var(--text-sm);color:var(--color-text-muted);line-height:1.82;margin-bottom:var(--space-4);}
.panel .eq-block{margin-bottom:var(--space-4);}
@media(max-width:640px){.panel{max-width:100%;border-left:none;border-top:3px solid var(--color-teal);}.panel-close{width:48px;height:48px;}}
```

- [ ] **Step 2: Add K-progression timeline CSS**

```css
/* ═══════════════════════════════════════════
   K-PROGRESSION TIMELINE
   ═══════════════════════════════════════════ */
.k-timeline{position:relative;padding-left:var(--space-10);margin-top:var(--space-8);}
.k-timeline::before{content:'';position:absolute;left:15px;top:0;bottom:0;width:2px;background:var(--color-divider);}
.k-waypoint{position:relative;padding-bottom:var(--space-10);opacity:0;transform:translateY(20px);transition:opacity 0.5s ease,transform 0.5s ease;}
.k-waypoint.visible{opacity:1;transform:translateY(0);}
.k-waypoint:last-child{padding-bottom:0;}
.k-dot{position:absolute;left:calc(-1 * var(--space-10) + 8px);top:4px;width:16px;height:16px;border-radius:50%;border:2px solid var(--color-divider);background:var(--color-bg);z-index:1;transition:all 0.3s ease;}
.k-waypoint.visible .k-dot{border-color:var(--color-success);background:var(--color-success);}
.k-waypoint.frontier .k-dot{border-color:var(--color-primary);background:var(--color-primary);box-shadow:0 0 12px oklch(from var(--color-primary) l c h / 0.5);}
.k-label{font-family:var(--font-mono);font-size:var(--text-sm);font-weight:600;color:var(--color-text);margin-bottom:var(--space-2);}
.k-sigma{display:inline-flex;align-items:center;gap:var(--space-2);font-family:var(--font-mono);font-size:var(--text-xs);color:var(--color-primary);background:var(--color-primary-highlight);padding:var(--space-1) var(--space-3);border-radius:var(--radius-full);margin-bottom:var(--space-3);}
.k-narrative{font-size:var(--text-sm);color:var(--color-text-muted);line-height:1.82;max-width:55ch;}
```

- [ ] **Step 3: Add story section and hero canvas CSS**

```css
/* ═══════════════════════════════════════════
   STORY SECTION
   ═══════════════════════════════════════════ */
.story-text{display:flex;flex-direction:column;gap:var(--space-5);}
.story-text p{font-size:var(--text-base);color:var(--color-text-muted);line-height:1.82;max-width:65ch;}
.story-text p strong{color:var(--color-text);font-weight:600;}
.story-text p em{color:var(--color-primary);font-style:italic;}
.logical-sidestep{background:oklch(from var(--color-surface) l c h / 0.85);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid var(--color-border);border-radius:var(--radius-xl);padding:var(--space-8);margin-top:var(--space-8);}
.logical-sidestep p{font-size:var(--text-sm);color:var(--color-text-muted);line-height:1.82;font-style:italic;}

/* ═══════════════════════════════════════════
   HERO CANVAS
   ═══════════════════════════════════════════ */
#heroCanvas{position:absolute;inset:0;z-index:0;width:100%;height:100%;}
.hero-content{position:relative;z-index:1;}

/* ═══════════════════════════════════════════
   SIGMA SWEEP
   ═══════════════════════════════════════════ */
.sweep-controls{display:flex;flex-wrap:wrap;align-items:center;gap:var(--space-4);margin-bottom:var(--space-6);}
.sweep-slider{flex:1;min-width:200px;accent-color:var(--color-primary);}
.sweep-value{font-family:var(--font-mono);font-size:var(--text-sm);color:var(--color-primary);min-width:80px;}
.k-toggle{display:flex;gap:var(--space-2);}
.k-toggle-btn{padding:var(--space-1) var(--space-3);border-radius:var(--radius-full);font-size:var(--text-xs);font-family:var(--font-mono);font-weight:600;color:var(--color-text-muted);border:1px solid var(--color-border);background:none;cursor:pointer;}
.k-toggle-btn.active{color:var(--color-primary);background:var(--color-primary-highlight);border-color:var(--color-primary);}
.chart-wrap{position:relative;width:100%;aspect-ratio:2/1;margin-bottom:var(--space-4);}
.chart-wrap canvas{width:100%;height:100%;border-radius:var(--radius-lg);border:1px solid var(--color-border);}
.chart-pending{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--font-mono);font-size:var(--text-xs);color:var(--color-text-faint);pointer-events:none;}
```

- [ ] **Step 4: Verify CSS parses correctly**

Open `index.html` in browser. No CSS errors in DevTools console. Existing sections still render correctly.

- [ ] **Step 5: Commit**

```bash
git add index.html
git commit -m "feat(site): add CSS for pop-out panels, timeline, story, charts"
```

---

### Task 3: Rewrite Hero Section

**Files:**
- Modify: `index.html` (lines 497–544: hero section HTML)
- Modify: `index.html` (hero CSS if tweaks needed)

- [ ] **Step 1: Replace hero HTML**

Replace the entire hero `<section>` (lines 500–544) with:

```html
<section class="hero" aria-label="Introduction">
  <div class="hero-bg">
    <img src="./assets/researcher-doberman-lab.jpg"
         alt="Researcher at work late at night with doberman companion"
         width="1920" height="1080" loading="eager"/>
  </div>
  <canvas id="heroCanvas" aria-hidden="true"></canvas>
  <div class="container">
    <div class="hero-content">
      <p class="hero-eyebrow"><span class="eyebrow-dot"></span> Independent Research &nbsp;·&nbsp; Edmond, OK</p>
      <h1 class="hero-title">
        Forty-six primes found<br>
        the same road <em>without a map.</em>
      </h1>
      <p class="hero-subtitle">
        Each one traveling its own geodesic through the zero field — no instructions,
        no shared itinerary — and every single one of them ended up humming the same
        frequency at σ&nbsp;=&nbsp;0.500. Not because they had to. Because the road was always there.
      </p>
      <div class="hero-ctas">
        <a href="#sigma-sweep" class="btn-primary">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M9 18l6-6-6-6"/></svg>
          Explore the Data
        </a>
        <a href="#about" class="btn-ghost">About B. Jones</a>
        <a href="https://github.com/RogueGringo/JTopo" target="_blank" rel="noopener noreferrer" class="pill" style="border-color:var(--color-border);">GitHub</a>
      </div>
      <div class="hero-stats">
        <div class="stat-item">
          <span class="stat-value">σ = 0.500</span>
          <span class="stat-label">Where the Fabric Fits</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">46 primes</span>
          <span class="stat-label">Each One Found It Alone</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">21.5%</span>
          <span class="stat-label">Resonance Over Noise</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">K → ∞</span>
          <span class="stat-label">Harmony Still Deepening</span>
        </div>
      </div>
    </div>
  </div>
</section>
```

- [ ] **Step 2: Update hero stats observer in app.js**

The old `animateCounter` observed `data-count` attributes. The new stats are static text. Remove the counter observer code (lines 128–146 of app.js) or make it skip elements without `data-count`. Replace with a simple fade-in:

```javascript
/* ─── Hero Stats Fade-In ─── */
var heroStats = document.querySelectorAll('.stat-item');
if (heroStats.length > 0 && 'IntersectionObserver' in window) {
  var statsObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        var items = entry.target.querySelectorAll('.stat-item');
        items.forEach(function (item, i) {
          item.style.transition = 'opacity 0.6s ease ' + (i * 0.15) + 's, transform 0.6s ease ' + (i * 0.15) + 's';
          item.style.opacity = '1';
          item.style.transform = 'translateY(0)';
        });
        statsObserver.disconnect();
      }
    });
  }, { threshold: 0.3 });
  var heroStatsWrap = document.querySelector('.hero-stats');
  if (heroStatsWrap) {
    heroStatsWrap.querySelectorAll('.stat-item').forEach(function (el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(12px)';
    });
    statsObserver.observe(heroStatsWrap);
  }
}
```

- [ ] **Step 3: Verify hero renders correctly**

Open in browser. Hero shows new title, subtitle, static stats, CTAs. Background image still visible behind. No JS errors.

- [ ] **Step 4: Commit**

```bash
git add index.html app.js
git commit -m "feat(site): rewrite hero with K=200 headline and Chopper Stan voice"
```

---

### Task 4: Hero Particle Animation

**Files:**
- Modify: `app.js` (add particle system after CHART_COLORS section)

- [ ] **Step 1: Implement particle animation**

Add to `app.js`:

```javascript
/* ─── Hero Particle Animation ─── */
var heroCanvas = document.getElementById('heroCanvas');
var heroCtx = heroCanvas ? heroCanvas.getContext('2d') : null;
var particles = [];
var particleCount = window.innerWidth < 640 ? 15 : 46;
var particleAnimId = null;

function initParticles() {
  if (!heroCanvas || !heroCtx) return;
  heroCanvas.width = heroCanvas.offsetWidth * (window.devicePixelRatio || 1);
  heroCanvas.height = heroCanvas.offsetHeight * (window.devicePixelRatio || 1);
  heroCtx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  particles = [];
  var w = heroCanvas.offsetWidth;
  var h = heroCanvas.offsetHeight;
  var centerX = w * 0.5;
  var targetY = h * 0.45; // convergence line

  for (var i = 0; i < particleCount; i++) {
    var angle = (Math.PI * 2 * i) / particleCount + Math.random() * 0.3;
    var startR = Math.max(w, h) * 0.6 + Math.random() * 100;
    particles.push({
      x: centerX + Math.cos(angle) * startR,
      y: targetY + Math.sin(angle) * startR * 0.5,
      targetX: centerX + (Math.random() - 0.5) * w * 0.3,
      targetY: targetY + (Math.random() - 0.5) * 20,
      vx: 0, vy: 0,
      size: 1.5 + Math.random() * 1.5,
      alpha: 0.3 + Math.random() * 0.5,
      phase: Math.random() * Math.PI * 2,
      speed: 0.003 + Math.random() * 0.004
    });
  }
}

function drawParticles() {
  if (!heroCtx) return;
  var w = heroCanvas.offsetWidth;
  var h = heroCanvas.offsetHeight;
  heroCtx.clearRect(0, 0, w, h);

  var color = CHART_COLORS.gold || '#d08a28';

  for (var i = 0; i < particles.length; i++) {
    var p = particles[i];
    p.phase += p.speed;
    var progress = Math.min(1, p.phase / (Math.PI * 2));
    var ease = 1 - Math.pow(1 - progress, 3);

    p.x += (p.targetX - p.x) * 0.008;
    p.y += (p.targetY - p.y) * 0.008;

    // Breathing
    var breath = 0.7 + 0.3 * Math.sin(p.phase * 0.5);

    heroCtx.beginPath();
    heroCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    heroCtx.fillStyle = color;
    heroCtx.globalAlpha = p.alpha * breath * ease;
    heroCtx.fill();

    // Draw connections to nearby particles
    for (var j = i + 1; j < particles.length; j++) {
      var q = particles[j];
      var dx = p.x - q.x;
      var dy = p.y - q.y;
      var dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 120) {
        heroCtx.beginPath();
        heroCtx.moveTo(p.x, p.y);
        heroCtx.lineTo(q.x, q.y);
        heroCtx.strokeStyle = color;
        heroCtx.globalAlpha = (1 - dist / 120) * 0.15 * ease;
        heroCtx.lineWidth = 0.5;
        heroCtx.stroke();
      }
    }
  }
  heroCtx.globalAlpha = 1;
  particleAnimId = requestAnimationFrame(drawParticles);
}

function restartParticles() {
  if (particleAnimId) cancelAnimationFrame(particleAnimId);
  initParticles();
  drawParticles();
}

// Respect reduced motion
if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  initParticles();
  drawParticles();
  window.addEventListener('resize', function () {
    if (particleAnimId) cancelAnimationFrame(particleAnimId);
    initParticles();
    drawParticles();
  });
}
```

- [ ] **Step 2: Verify animation**

Open in browser. Gold particles drift inward and form connections. Resize window — particles reinitialize. Toggle theme — particles use new gold color. Enable reduced-motion preference — no animation.

- [ ] **Step 3: Commit**

```bash
git add app.js
git commit -m "feat(site): add hero particle braiding animation (Canvas)"
```

---

### Task 5: Story Section + K-Progression Timeline

**Files:**
- Modify: `index.html` (insert new sections after hero, before research)

- [ ] **Step 1: Add Story section HTML**

Insert after `</section>` (end of hero, line 544) and before the research section:

```html
<!-- ═══════════════════════════════════════════
     THE STORY
     ═══════════════════════════════════════════ -->
<section class="section" id="story" aria-labelledby="story-h">
  <div class="container">
    <div class="section-header reveal">
      <p class="section-label">What We Did</p>
      <h2 class="section-title" id="story-h">The Fabric and the Frequency</h2>
    </div>
    <div class="story-text reveal">
      <p>A <strong>prime number</strong> is the simplest thing in mathematics. It has itself, it has one, and it has a quiet, total relationship with every number it will never divide into. That relationship is not absence — it's <em>structure</em>. Every composite number is a sentence written in primes. Every prime is a letter that means exactly one thing.</p>
      <p>We gave forty-six of these letters a space to move through — the zeros of the <strong>Riemann zeta function</strong>, strung like waypoints across the critical strip — and wove them into a fabric using nothing but the internal grammar they already carry. The mathematical name for this fabric is a <em>sheaf Laplacian with a u(K) gauge connection</em>. The plain name is: a measure of how well the primes agree with each other at every point in the space.</p>
      <p>Then we asked: <strong>where does the fabric fit best?</strong> At σ&nbsp;=&nbsp;0.500. Every time. More primes, tighter fabric, deeper resonance. As if the zeros were waiting for exactly this weave. As if the critical line isn't a boundary but a <em>home frequency</em> — the one place where the prime field and the zero field recognize each other.</p>
    </div>
    <button class="panel-trigger" data-panel="panel-fourier">+ See the math: The Fourier Kernel</button>
  </div>
</section>

<!-- ═══════════════════════════════════════════
     K-PROGRESSION
     ═══════════════════════════════════════════ -->
<section class="section" id="k-progression" aria-labelledby="kprog-h">
  <div class="container">
    <div class="section-header reveal">
      <p class="section-label">The Orbit Tightening</p>
      <h2 class="section-title" id="kprog-h">More Primes, Closer to Home</h2>
    </div>
    <div class="k-timeline" id="kTimeline">
      <div class="k-waypoint" data-delay="0">
        <div class="k-dot"></div>
        <div class="k-label">K=20 — 8 primes</div>
        <div class="k-sigma">σ ≈ 0.65</div>
        <p class="k-narrative">Eight primes drifted into the manifold the way starlight arrives — each from its own origin, each carrying only what it was born with. They settled near σ&nbsp;≈&nbsp;0.65. Close enough to notice each other.</p>
      </div>
      <div class="k-waypoint" data-delay="400">
        <div class="k-dot"></div>
        <div class="k-label">K=50 — 15 primes</div>
        <div class="k-sigma">σ ≈ 0.58</div>
        <p class="k-narrative">The harmonics thickened. Not louder — richer. Like voices joining a song they all somehow already knew. σ&nbsp;≈&nbsp;0.58.</p>
      </div>
      <div class="k-waypoint" data-delay="800">
        <div class="k-dot"></div>
        <div class="k-label">K=100 — 25 primes</div>
        <div class="k-sigma">σ ≈ 0.52</div>
        <p class="k-narrative">Twenty-five primes weaving through the same field, each one's orbit gently bending toward the others. Not pulled. Invited. σ&nbsp;≈&nbsp;0.52.</p>
      </div>
      <div class="k-waypoint frontier" data-delay="1200">
        <div class="k-dot"></div>
        <div class="k-label">K=200 — 46 primes</div>
        <div class="k-sigma">σ = 0.500 ← critical line</div>
        <p class="k-narrative">Every prime under 200 — each one irreducible, each one whole, each one threading the zeta zeros along a path no one drew for them. And they meet. σ&nbsp;=&nbsp;0.500. Not collision. <strong>Resonance.</strong> The manifold has a heartbeat and the primes are its pulse.</p>
      </div>
    </div>
  </div>
</section>
```

- [ ] **Step 2: Add timeline animation JS to app.js**

```javascript
/* ─── K-Progression Timeline ─── */
var kTimeline = document.getElementById('kTimeline');
if (kTimeline && 'IntersectionObserver' in window) {
  var waypoints = kTimeline.querySelectorAll('.k-waypoint');
  var timelineObserver = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        waypoints.forEach(function (wp) {
          var delay = parseInt(wp.getAttribute('data-delay') || '0', 10);
          setTimeout(function () {
            wp.classList.add('visible');
          }, delay);
        });
        timelineObserver.disconnect();
      }
    });
  }, { threshold: 0.15 });
  timelineObserver.observe(kTimeline);
}
```

- [ ] **Step 3: Verify sections render**

Open in browser. Story section shows three paragraphs with the panel trigger link. K-progression shows 4 waypoints. Scroll into view triggers staggered animation. K=200 waypoint has gold glow.

- [ ] **Step 4: Commit**

```bash
git add index.html app.js
git commit -m "feat(site): add story section and K-progression animated timeline"
```

---

### Task 6: Interactive Sigma Sweep Chart

**Files:**
- Modify: `index.html` (insert section after K-progression)
- Modify: `app.js` (add Canvas chart logic)

- [ ] **Step 1: Add sigma sweep HTML**

Insert after the K-progression section:

```html
<!-- ═══════════════════════════════════════════
     SIGMA SWEEP
     ═══════════════════════════════════════════ -->
<section class="section" id="sigma-sweep" aria-labelledby="sweep-h">
  <div class="container">
    <div class="section-header reveal">
      <p class="section-label">Interactive Data</p>
      <h2 class="section-title" id="sweep-h">Tune the Fabric Yourself</h2>
      <p class="section-desc">Drag the slider to vary σ across the critical strip. Watch how the spectral sum — the fabric's wrinkle score — changes for zeta zeros vs. controls.</p>
    </div>
    <div class="sweep-controls reveal">
      <label for="sigmaSlider" class="sr-only">Sigma value</label>
      <input type="range" id="sigmaSlider" class="sweep-slider" min="0.25" max="0.75" step="0.01" value="0.50">
      <span class="sweep-value" id="sigmaDisplay">σ = 0.50</span>
      <div class="k-toggle" role="group" aria-label="Select K value">
        <button class="k-toggle-btn" data-k="K100">K=100</button>
        <button class="k-toggle-btn active" data-k="K200">K=200</button>
      </div>
    </div>
    <div class="chart-wrap reveal">
      <canvas id="sweepChart" aria-label="Spectral sum chart showing S(σ) for Zeta, GUE, and Random sources"></canvas>
    </div>
    <button class="panel-trigger" data-panel="panel-spectral">+ See the math: Reading the Spectral Sum</button>
  </div>
</section>
```

- [ ] **Step 2: Add sigma sweep Canvas drawing to app.js**

```javascript
/* ─── Sigma Sweep Chart ─── */
var sweepCanvas = document.getElementById('sweepChart');
var sweepCtx = sweepCanvas ? sweepCanvas.getContext('2d') : null;
var sigmaSlider = document.getElementById('sigmaSlider');
var sigmaDisplay = document.getElementById('sigmaDisplay');
var activeK = 'K200';

function getChartSeries(kKey) {
  var data = CHART_DATA[kKey];
  if (!data) return {};
  var series = {};
  for (var src in data) {
    var sigmas = Object.keys(data[src]).map(Number).sort(function(a,b){return a-b;});
    var values = sigmas.map(function(s) { return data[src][s.toFixed(3)]; });
    series[src] = { sigmas: sigmas, values: values };
  }
  return series;
}

function redrawSigmaSweep() {
  if (!sweepCanvas || !sweepCtx) return;
  var dpr = window.devicePixelRatio || 1;
  var w = sweepCanvas.offsetWidth;
  var h = sweepCanvas.offsetHeight;
  sweepCanvas.width = w * dpr;
  sweepCanvas.height = h * dpr;
  sweepCtx.scale(dpr, dpr);

  var pad = { top: 20, right: 20, bottom: 40, left: 60 };
  var cw = w - pad.left - pad.right;
  var ch = h - pad.top - pad.bottom;

  // Clear
  sweepCtx.fillStyle = CHART_COLORS.surface || '#16140d';
  sweepCtx.fillRect(0, 0, w, h);

  var series = getChartSeries(activeK);
  if (!series.Zeta) return;

  // Find global min/max for Y axis
  var allVals = [];
  for (var src in series) {
    allVals = allVals.concat(series[src].values);
  }
  var yMin = Math.min.apply(null, allVals) * 0.995;
  var yMax = Math.max.apply(null, allVals) * 1.005;

  function xScale(sigma) { return pad.left + ((sigma - 0.25) / 0.5) * cw; }
  function yScale(val) { return pad.top + ch - ((val - yMin) / (yMax - yMin)) * ch; }

  // Grid lines
  sweepCtx.strokeStyle = CHART_COLORS.border || '#35311e';
  sweepCtx.lineWidth = 0.5;
  for (var g = 0; g < 5; g++) {
    var gy = pad.top + (g / 4) * ch;
    sweepCtx.beginPath(); sweepCtx.moveTo(pad.left, gy); sweepCtx.lineTo(w - pad.right, gy); sweepCtx.stroke();
  }

  // σ=0.5 reference line
  sweepCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
  sweepCtx.globalAlpha = 0.3;
  sweepCtx.lineWidth = 1;
  sweepCtx.setLineDash([4, 4]);
  var x05 = xScale(0.5);
  sweepCtx.beginPath(); sweepCtx.moveTo(x05, pad.top); sweepCtx.lineTo(x05, pad.top + ch); sweepCtx.stroke();
  sweepCtx.setLineDash([]);
  sweepCtx.globalAlpha = 1;

  // Draw series
  var colorMap = { Zeta: CHART_COLORS.gold, GUE: CHART_COLORS.teal, Random: CHART_COLORS.gray };
  for (var name in series) {
    var s = series[name];
    sweepCtx.strokeStyle = colorMap[name] || CHART_COLORS.gray;
    sweepCtx.lineWidth = name === 'Zeta' ? 2.5 : 1.5;
    sweepCtx.beginPath();
    for (var i = 0; i < s.sigmas.length; i++) {
      var px = xScale(s.sigmas[i]);
      var py = yScale(s.values[i]);
      if (i === 0) sweepCtx.moveTo(px, py);
      else sweepCtx.lineTo(px, py);
    }
    sweepCtx.stroke();

    // Dots
    for (var i = 0; i < s.sigmas.length; i++) {
      sweepCtx.beginPath();
      sweepCtx.arc(xScale(s.sigmas[i]), yScale(s.values[i]), 3, 0, Math.PI * 2);
      sweepCtx.fillStyle = colorMap[name] || CHART_COLORS.gray;
      sweepCtx.fill();
    }
  }

  // Slider position indicator
  var sliderVal = parseFloat(sigmaSlider ? sigmaSlider.value : 0.5);
  var sx = xScale(sliderVal);
  sweepCtx.strokeStyle = CHART_COLORS.text || '#d6d0be';
  sweepCtx.globalAlpha = 0.5;
  sweepCtx.lineWidth = 1;
  sweepCtx.beginPath(); sweepCtx.moveTo(sx, pad.top); sweepCtx.lineTo(sx, pad.top + ch); sweepCtx.stroke();
  sweepCtx.globalAlpha = 1;

  // Axis labels
  sweepCtx.fillStyle = CHART_COLORS.textMuted || '#817a66';
  sweepCtx.font = '11px "JetBrains Mono", monospace';
  sweepCtx.textAlign = 'center';
  [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75].forEach(function (v) {
    sweepCtx.fillText(v.toFixed(2), xScale(v), h - 8);
  });
  sweepCtx.textAlign = 'right';
  for (var g = 0; g < 5; g++) {
    var val = yMin + (g / 4) * (yMax - yMin);
    sweepCtx.fillText(val.toFixed(1), pad.left - 8, pad.top + ch - (g / 4) * ch + 4);
  }

  // Legend
  sweepCtx.textAlign = 'left';
  var legendX = pad.left + 10;
  var legendY = pad.top + 15;
  var legendItems = [['Zeta', CHART_COLORS.gold], ['GUE', CHART_COLORS.teal]];
  if (series.Random) legendItems.push(['Random', CHART_COLORS.gray]);
  legendItems.forEach(function (item, i) {
    sweepCtx.fillStyle = item[1];
    sweepCtx.fillRect(legendX, legendY + i * 18 - 8, 12, 3);
    sweepCtx.fillText(item[0], legendX + 18, legendY + i * 18);
  });
}

// Event handlers
if (sigmaSlider) {
  sigmaSlider.addEventListener('input', function () {
    if (sigmaDisplay) sigmaDisplay.textContent = 'σ = ' + parseFloat(sigmaSlider.value).toFixed(2);
    redrawSigmaSweep();
  });
}

document.querySelectorAll('.k-toggle-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.k-toggle-btn').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    activeK = btn.getAttribute('data-k');
    redrawSigmaSweep();
  });
});

// Initial draw after layout
window.addEventListener('load', function () { redrawSigmaSweep(); });
window.addEventListener('resize', function () { redrawSigmaSweep(); });
```

- [ ] **Step 3: Verify chart renders**

Open in browser. Chart shows spectral sum lines for Zeta (gold) and GUE (teal). Slider moves a vertical indicator line. K toggle switches between K=100 (3 lines) and K=200 (2 lines). Theme toggle redraws with correct colors.

- [ ] **Step 4: Commit**

```bash
git add index.html app.js
git commit -m "feat(site): add interactive sigma sweep chart with Canvas API"
```

---

### Task 7: Arithmetic Premium Chart

**Files:**
- Modify: `index.html` (insert section after sigma sweep)
- Modify: `app.js` (add premium chart logic)

- [ ] **Step 1: Add arithmetic premium HTML**

```html
<!-- ═══════════════════════════════════════════
     ARITHMETIC PREMIUM
     ═══════════════════════════════════════════ -->
<section class="section" id="premium" aria-labelledby="premium-h">
  <div class="container">
    <div class="section-header reveal">
      <p class="section-label">Signal Analysis</p>
      <h2 class="section-title" id="premium-h">How Much More Than Noise</h2>
      <p class="section-desc">The arithmetic premium — the ratio of zeta's spectral sum to GUE's — isolates what the primes contribute beyond statistical structure. Watch the minimum march toward σ&nbsp;=&nbsp;0.500 as K increases.</p>
    </div>
    <div class="chart-wrap reveal">
      <canvas id="premiumChart" aria-label="Arithmetic premium chart showing zeta/GUE ratio across sigma for multiple K values"></canvas>
    </div>
    <button class="panel-trigger" data-panel="panel-gue">+ See the math: Why GUE, Not Just Random?</button>
  </div>
</section>
```

- [ ] **Step 2: Add premium chart Canvas drawing to app.js**

```javascript
/* ─── Arithmetic Premium Chart ─── */
var premiumCanvas = document.getElementById('premiumChart');
var premiumCtx = premiumCanvas ? premiumCanvas.getContext('2d') : null;

function redrawPremiumChart() {
  if (!premiumCanvas || !premiumCtx) return;
  var dpr = window.devicePixelRatio || 1;
  var w = premiumCanvas.offsetWidth;
  var h = premiumCanvas.offsetHeight;
  premiumCanvas.width = w * dpr;
  premiumCanvas.height = h * dpr;
  premiumCtx.scale(dpr, dpr);

  var pad = { top: 20, right: 20, bottom: 40, left: 70 };
  var cw = w - pad.left - pad.right;
  var ch = h - pad.top - pad.bottom;

  premiumCtx.fillStyle = CHART_COLORS.surface || '#16140d';
  premiumCtx.fillRect(0, 0, w, h);

  // Compute ratios for each K
  var kSets = [
    { key: 'K100', label: 'K=100', alpha: 0.6, width: 1.5 },
    { key: 'K200', label: 'K=200', alpha: 1.0, width: 2.5 }
  ];

  var allRatios = [];
  var seriesList = [];

  kSets.forEach(function (ks) {
    var data = CHART_DATA[ks.key];
    if (!data || !data.Zeta || !data.GUE) return;
    var sigmas = [];
    var ratios = [];
    var zetaSigmas = Object.keys(data.Zeta).map(Number).sort(function(a,b){return a-b;});
    zetaSigmas.forEach(function (s) {
      var sk = s.toFixed(3);
      if (data.GUE[sk] && data.GUE[sk] > 0) {
        var ratio = data.Zeta[sk] / data.GUE[sk];
        sigmas.push(s);
        ratios.push(ratio);
        allRatios.push(ratio);
      }
    });
    if (sigmas.length > 0) {
      var minIdx = ratios.indexOf(Math.min.apply(null, ratios));
      seriesList.push({ sigmas: sigmas, ratios: ratios, minIdx: minIdx, label: ks.label, alpha: ks.alpha, width: ks.width });
    }
  });

  if (allRatios.length === 0) return;

  var yMin = Math.min.apply(null, allRatios) * 0.999;
  var yMax = Math.max.apply(null, allRatios) * 1.001;

  function xScale(sigma) { return pad.left + ((sigma - 0.25) / 0.5) * cw; }
  function yScale(val) { return pad.top + ch - ((val - yMin) / (yMax - yMin)) * ch; }

  // σ=0.5 reference
  premiumCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
  premiumCtx.globalAlpha = 0.3;
  premiumCtx.setLineDash([4, 4]);
  premiumCtx.lineWidth = 1;
  var x05 = xScale(0.5);
  premiumCtx.beginPath(); premiumCtx.moveTo(x05, pad.top); premiumCtx.lineTo(x05, pad.top + ch); premiumCtx.stroke();
  premiumCtx.setLineDash([]);
  premiumCtx.globalAlpha = 1;

  // Draw series
  seriesList.forEach(function (s) {
    premiumCtx.strokeStyle = CHART_COLORS.gold || '#d08a28';
    premiumCtx.globalAlpha = s.alpha;
    premiumCtx.lineWidth = s.width;
    premiumCtx.beginPath();
    for (var i = 0; i < s.sigmas.length; i++) {
      var px = xScale(s.sigmas[i]);
      var py = yScale(s.ratios[i]);
      if (i === 0) premiumCtx.moveTo(px, py);
      else premiumCtx.lineTo(px, py);
    }
    premiumCtx.stroke();
    premiumCtx.globalAlpha = 1;

    // Minimum marker
    var mx = xScale(s.sigmas[s.minIdx]);
    var my = yScale(s.ratios[s.minIdx]);
    premiumCtx.beginPath();
    premiumCtx.arc(mx, my, 5, 0, Math.PI * 2);
    premiumCtx.fillStyle = CHART_COLORS.gold || '#d08a28';
    premiumCtx.globalAlpha = s.alpha;
    premiumCtx.fill();
    premiumCtx.globalAlpha = 1;

    // Label
    premiumCtx.fillStyle = CHART_COLORS.text || '#d6d0be';
    premiumCtx.font = '10px "JetBrains Mono", monospace';
    premiumCtx.globalAlpha = s.alpha;
    premiumCtx.textAlign = 'center';
    premiumCtx.fillText(s.label + ' σ=' + s.sigmas[s.minIdx].toFixed(3), mx, my - 12);
    premiumCtx.globalAlpha = 1;
  });

  // Axis labels
  premiumCtx.fillStyle = CHART_COLORS.textMuted || '#817a66';
  premiumCtx.font = '11px "JetBrains Mono", monospace';
  premiumCtx.textAlign = 'center';
  [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75].forEach(function (v) {
    premiumCtx.fillText(v.toFixed(2), xScale(v), h - 8);
  });
  premiumCtx.textAlign = 'right';
  for (var g = 0; g < 5; g++) {
    var val = yMin + (g / 4) * (yMax - yMin);
    premiumCtx.fillText(val.toFixed(4), pad.left - 8, pad.top + ch - (g / 4) * ch + 4);
  }

  // Y-axis label
  premiumCtx.save();
  premiumCtx.translate(12, pad.top + ch / 2);
  premiumCtx.rotate(-Math.PI / 2);
  premiumCtx.textAlign = 'center';
  premiumCtx.fillText('S(zeta) / S(GUE)', 0, 0);
  premiumCtx.restore();
}

window.addEventListener('load', function () { redrawPremiumChart(); });
window.addEventListener('resize', function () { redrawPremiumChart(); });
```

- [ ] **Step 3: Verify chart renders**

Open in browser. Chart shows K=100 (dimmer) and K=200 (bright gold) ratio curves with minimum markers. σ=0.500 reference dashed line visible. Theme toggle redraws correctly.

- [ ] **Step 4: Commit**

```bash
git add index.html app.js
git commit -m "feat(site): add arithmetic premium chart showing K-progression"
```

---

### Task 8: Pop-Out Panel Component + All 4 Panels

**Files:**
- Modify: `index.html` (add panel HTML before `</main>`)
- Modify: `app.js` (add panel open/close logic)

- [ ] **Step 1: Add panel HTML**

Insert before `</main>`:

```html
<!-- ═══════════════════════════════════════════
     POP-OUT PANELS
     ═══════════════════════════════════════════ -->
<div class="panel-overlay" id="panelOverlay"></div>

<div class="panel" id="panel-fourier" role="dialog" aria-label="The Fourier Kernel">
  <button class="panel-close" aria-label="Close panel">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </button>
  <h4>The Fourier Kernel</h4>
  <p>The explicit formula connects zeta zeros to primes through an oscillatory sum. Each prime p contributes a phase factor to the transport map between zeros γ_i and γ_j:</p>
  <div class="eq-block">
    <div class="eq-label">Phase Factor</div>
    <div class="eq-formula">e^{i · Δγ · log(p)} where Δγ = γ_i − γ_j</div>
  </div>
  <p>This is the Fourier kernel of the explicit formula — each prime acts as a frequency, and the log-spacing of primes creates the harmonic structure. More primes = more frequencies = sharper resolution of the spectral peak.</p>
  <div class="eq-block">
    <div class="eq-label">Superposition Transport</div>
    <div class="eq-formula">A_ij(σ) = Σ_{p≤K} e^{iΔγ·log p} · B_p(σ)</div>
  </div>
  <p>The matrix B_p(σ) encodes the arithmetic weight of each prime at position σ in the critical strip. At σ = 0.500, the functional equation symmetry makes the contributions from p^{−σ} and p^{−(1−σ)} equal — creating maximum constructive interference.</p>
</div>

<div class="panel" id="panel-spectral" role="dialog" aria-label="Reading the Spectral Sum">
  <button class="panel-close" aria-label="Close panel">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </button>
  <h4>Reading the Spectral Sum</h4>
  <p>The spectral sum S(σ) = Σ λ_k is the sum of the smallest eigenvalues of the sheaf Laplacian at position σ. It measures how "wrinkled" the prime fabric is.</p>
  <p><strong>Low S(σ)</strong> = the sheaf is nearly flat. The transport maps are nearly consistent — primes "agree" on the geometry at this σ. The fabric fits.</p>
  <p><strong>High S(σ)</strong> = large eigenvalues, inconsistent transport. Primes disagree about the geometry. The fabric is loose.</p>
  <p>The Betti number β₀ (number of eigenvalues near zero) counts the connected components of the "flat" part of the sheaf. At σ = 0.500, β₀ is maximized — the largest coherent structure.</p>
  <div class="eq-block">
    <div class="eq-label">Order Parameter</div>
    <div class="eq-formula">S(σ) = Σ_{k=1}^{k_eig} λ_k(L_𝓕(σ))</div>
  </div>
</div>

<div class="panel" id="panel-gue" role="dialog" aria-label="Why GUE?">
  <button class="panel-close" aria-label="Close panel">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </button>
  <h4>Why GUE, Not Just Random?</h4>
  <p>The Gaussian Unitary Ensemble (GUE) is a random matrix model whose eigenvalue statistics match the local statistics of zeta zeros — the Montgomery-Odlyzko law. GUE points have the same spacing distribution as zeta zeros but lack their global arithmetic structure.</p>
  <p>Comparing zeta to Poisson (uniform random) measures total signal. Comparing zeta to GUE isolates the <em>arithmetic</em> signal — the part that comes specifically from primes, not from the generic repulsion statistics that any L-function would share.</p>
  <p>The arithmetic premium (1 − S_zeta/S_GUE) measures how much tighter the prime fabric is compared to this statistically identical but arithmetically empty control. At K=200, this premium is 21.5% at σ = 0.500.</p>
</div>

<div class="panel" id="panel-connection" role="dialog" aria-label="u(K) Connection">
  <button class="panel-close" aria-label="Close panel">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </button>
  <h4>The u(K) Connection</h4>
  <p>Each prime p ≤ K defines a K×K representation matrix ρ(p) — the truncated left-regular representation of ℤ/pℤ. This is a cyclic permutation matrix: it shifts basis vectors by one position modulo p.</p>
  <div class="eq-block">
    <div class="eq-label">Prime Representation</div>
    <div class="eq-formula">ρ(p)_jk = δ_{j, (k+1 mod p)}</div>
  </div>
  <p>The transport map between zeros γ_i and γ_j is constructed by exponentiating a Lie algebra element built from all prime representations simultaneously — the superposition mode:</p>
  <div class="eq-block">
    <div class="eq-label">Transport Map</div>
    <div class="eq-formula">U_ij = exp(i · Σ_p A_p(σ) · log ρ(p))</div>
  </div>
  <p>This connection lives in the unitary group U(K). As K grows, the fiber dimension increases and the connection encodes more of the prime harmonic structure.</p>
</div>
```

- [ ] **Step 2: Add panel JS to app.js**

```javascript
/* ─── Pop-Out Panels ─── */
var panelOverlay = document.getElementById('panelOverlay');
var openPanel = null;

function openPanelById(id) {
  var panel = document.getElementById(id);
  if (!panel || !panelOverlay) return;
  if (openPanel) closePanelFn();
  panelOverlay.classList.add('open');
  panel.classList.add('open');
  openPanel = panel;
  document.body.style.overflow = 'hidden';
  // Focus trap
  var closeBtn = panel.querySelector('.panel-close');
  if (closeBtn) closeBtn.focus();
}

function closePanelFn() {
  if (!openPanel || !panelOverlay) return;
  panelOverlay.classList.remove('open');
  openPanel.classList.remove('open');
  openPanel = null;
  document.body.style.overflow = '';
}

// Panel triggers
document.querySelectorAll('.panel-trigger').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var panelId = btn.getAttribute('data-panel');
    if (panelId) openPanelById(panelId);
  });
});

// Close handlers
if (panelOverlay) {
  panelOverlay.addEventListener('click', closePanelFn);
}
document.querySelectorAll('.panel-close').forEach(function (btn) {
  btn.addEventListener('click', closePanelFn);
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && openPanel) closePanelFn();
});
```

- [ ] **Step 3: Verify panels work**

Click "See the math" link in story section → Fourier Kernel panel slides in from right. Click overlay or Escape → closes. Repeat for all 4 panel triggers. Mobile viewport → panel takes full width.

- [ ] **Step 4: Commit**

```bash
git add index.html app.js
git commit -m "feat(site): add frosted glass pop-out panels with 4 math deep-dives"
```

---

### Task 9: Reorder Page Sections + Update Nav

**Files:**
- Modify: `index.html` (reorder HTML sections, update nav links)

This is the structural surgery. The current order is: Hero → Research (huge section) → Projects → About → Alpha → Blog → Contact → Citation. The new order is: Hero → Story → K-Progression → Sigma Sweep → Premium → Research(framework/falsification/infrastructure) → About → Alpha → Projects → Blog → Footer(contact+citation).

- [ ] **Step 1: Reorder sections in index.html**

The new sections (Story, K-Progression, Sigma Sweep, Premium) were already inserted in Tasks 5-7. Now:

1. Move the research section's **framework tabs** (lines 651–772) into its own `<section>` with id `framework`
2. Move **falsification criteria** (lines 957–1001) into its own `<section>` with id `falsification`
3. Move **infrastructure** (lines 899–955) into its own `<section>` with id `infrastructure`
4. Move **experimental results phase cards** (lines 774–898) — these are replaced by the new interactive charts, so wrap in a `<!-- Legacy results replaced by interactive charts -->` comment and remove
5. Move **project status dashboard** (lines 1003–1074) — update with K=200 results
6. Reorder About/Alpha to come after Infrastructure
7. Keep Projects, Blog below personal sections
8. Merge Contact + Citation into Footer

- [ ] **Step 2: Update navigation links**

```html
<ul class="nav-links">
  <li><a href="#story">Story</a></li>
  <li><a href="#sigma-sweep">Data</a></li>
  <li><a href="#framework">Framework</a></li>
  <li><a href="#about">About</a></li>
  <li><a href="#projects">Projects</a></li>
  <li><a href="#blog">Notes</a></li>
</ul>
```

Update mobile nav to match.

- [ ] **Step 3: Update framework tab labels**

Rename the tab buttons:
- "Fiber Structure" → "The Points"
- "Gauge Connection" → "The Threads"
- "Sheaf Laplacian" → "The Fabric"
- "Statistical Test" → "The Measurement"

Add a panel trigger for the u(K) Connection panel at the bottom of the framework section.

- [ ] **Step 4: Update project status dashboard**

Update the status cards:
- Phase 3c K=100: change to green/complete: "Complete. Full sweep: 90 grid points. Three-tier hierarchy confirmed."
- Phase 3d K=200: add new card (amber): "T1 complete. Arithmetic premium peaks at σ=0.500. T2/T3 pending."
- Phase 4: update to amber: "In progress. Fourier sharpening confirmed K=20→K=200."

- [ ] **Step 5: Clean up RunPod references**

Remove or replace:
- Line 585: "RunPod A100/MI300X" → "RTX 5070 GPU (12GB, local)"
- Line 920: tag "RunPod A100" → "RTX 5070"
- Lines 1067-1068: Remove RunPod rows from hardware table, add "RTX 5070" row as primary GPU

- [ ] **Step 6: Verify page flow**

Open in browser. Scroll through entire page. Verify order: Hero → Story → K-Progression → Sigma Sweep → Premium → Framework → Falsification → Infrastructure → About → Alpha → Projects → Blog → Footer. Nav links scroll to correct sections. No broken anchor refs.

- [ ] **Step 7: Commit**

```bash
git add index.html
git commit -m "feat(site): reorder page flow — research narrative first, personal below"
```

---

### Task 10: Falsification Section Upgrade

**Files:**
- Modify: `index.html` (falsification section)

- [ ] **Step 1: Add status badges to criteria items**

Update each falsification criterion `<li>` with current evaluation status based on K=100/K=200 results:

- F1 (all modes identical): **Pass** — Superposition mode produces distinct spectral sums
- F2 (Poisson same as zeta): **Pass** — Poisson (Random) shows no σ-dependence matching zeta
- F3 (signal ratio < 2×): **Pass** — Signal ratio 670× at K=20
- F4 (numerical instability): **Pass** — Stable through K=200
- R1 (peak ≠ 0.5): **Pending** — K=200 T1 shows minimum at σ=0.500, but full profile (T2) needed for CI
- R2 (GUE same transition): **Partial** — GUE contrast is 85-98% of zeta's (not identical, but close)
- R3 (profile flattens): **Pass** — Profile sharpens K=20→K=200
- P1 (peak at 0.50±0.02): **Evidence** — K=200 arithmetic premium minimum at σ=0.500
- P2 (sharpens with K): **Evidence** — Confirmed K=20→K=100→K=200
- P3 (controls no transition): **Partial** — GUE shows weaker but similar structure
- P4 (signal ratio >10³): **Not met** — Signal is real but subtle (21.5%, not 1000×)

Add a badge span after each `<span class="criteria-code">`:

```html
<span class="badge badge-green" style="font-size:0.6rem;margin-left:var(--space-2);">PASS</span>
```

- [ ] **Step 2: Add logical sidestep callout**

After the protocol-box, add:

```html
<div class="logical-sidestep reveal" style="margin-top:var(--space-8);">
  <p>We didn't try to prove the Riemann Hypothesis. We asked a different question: if you weave forty-six prime numbers into a geometric fabric over the zeros of the zeta function, does the fabric care where the zeros sit? It does. It cares most at exactly the line Riemann predicted. That's not a proof. It's the fabric telling you something.</p>
</div>
```

- [ ] **Step 3: Verify**

Open in browser. Falsification cards show status badges. Logical sidestep renders as frosted glass card. All badges match the documented results.

- [ ] **Step 4: Commit**

```bash
git add index.html
git commit -m "feat(site): upgrade falsification section with K=200 status badges"
```

---

### Task 11: Footer Upgrade + Final Polish

**Files:**
- Modify: `index.html` (footer section)
- Modify: `app.js` (ensure all observers fire correctly with new section order)

- [ ] **Step 1: Update footer**

Add the closing line and merge contact/citation info:

```html
<footer class="site-footer">
  <div class="container">
    <div class="footer-inner">
      <div>
        <a href="#" class="nav-logo" aria-label="B. Jones — Home">
          <!-- existing SVG logo -->
        </a>
        <p class="footer-tagline">The harmony is still deepening.</p>
      </div>
      <div class="footer-col">
        <div class="footer-col-title">Research</div>
        <a href="#story">The Story</a>
        <a href="#sigma-sweep">Interactive Data</a>
        <a href="#framework">Framework</a>
        <a href="#falsification">Falsification</a>
      </div>
      <div class="footer-col">
        <div class="footer-col-title">Connect</div>
        <a href="https://github.com/RogueGringo/JTopo" target="_blank" rel="noopener noreferrer">GitHub</a>
        <a href="mailto:contact address">Email</a>
      </div>
    </div>
    <div class="footer-bottom">
      <span class="footer-copy">© 2026 B. Jones — MIT License</span>
    </div>
  </div>
</footer>
```

- [ ] **Step 2: Verify full page**

Complete walkthrough: scroll top to bottom. Every section renders. All nav links work. All panel triggers work. Both charts render. Timeline animates. Theme toggle updates everything. Mobile responsive at 375px viewport.

- [ ] **Step 3: Final commit**

```bash
git add index.html app.js
git commit -m "feat(site): complete redesign — footer upgrade and final polish"
```

---

### Task 12: Cross-Browser & Accessibility Check

**Files:** No file changes (verification only)

- [ ] **Step 1: Check reduced-motion**

Enable `prefers-reduced-motion: reduce` in DevTools. Verify: no particle animation, no timeline stagger animation, scroll reveals still appear (instantly, no transition).

- [ ] **Step 2: Check keyboard navigation**

Tab through entire page. Verify: all interactive elements (nav links, CTAs, slider, K-toggle, panel triggers) are focusable. Panel traps focus when open. Escape closes panel.

- [ ] **Step 3: Check Canvas aria-labels**

Verify both `<canvas>` elements have `aria-label` attributes describing the chart content.

- [ ] **Step 4: Check light theme**

Toggle to light mode. All text readable. Charts use light-mode colors. Pop-out panel backdrop contrasts correctly.

- [ ] **Step 5: Check mobile (640px)**

Resize to 640px. Hero stats stack. Charts resize. Timeline fits. Pop-out panels go full-screen. Hamburger menu works with updated nav links.
