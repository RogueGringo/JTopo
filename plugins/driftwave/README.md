# driftwave

Adaptive Topological Field Theory operationalized as a Claude Code plugin.

## What This Actually Does (Plain English)

Most brainstorming and analysis plugins work like a checklist: generate ideas, rank them, pick one, go. Driftwave works differently — it treats your problem as a **shape** and asks whether that shape is getting more coherent or falling apart as you work on it.

The plugin enforces a four-layer pipeline (L0 → L1 → L2 → L3) where each layer acts as a gate. You can't skip ahead. Ideas and artifacts enter raw at L0, get clustered by persistence at L1, get monitored for structural coherence at L2, and get validated for global consistency at L3. If something degrades at any point, the pipeline sends you back down rather than letting you push through with a flawed structure.

The key metric is the **Gini trajectory** — not "how many ideas do you have?" but "are your ideas becoming more hierarchically organized or more scattered?" A design with 3 deeply connected sections beats one with 12 loosely related sections. Shape over count.

### Based on the Mathematical Framework in

- *Adaptive Topological Field Theory* (Jones, 2026)
- *A Unified Topological Framework for System Abstraction via Reverse Engineering*
- *Computational Topology and the Riemann Hypothesis*

## When to Use Driftwave vs. Other Plugins

### Use driftwave when:

- **The problem has hidden structure** you need to discover, not just enumerate. Architecture decisions, system design, complex refactors where the "right" decomposition isn't obvious.
- **You keep going in circles.** The Gini watchdog and routing signals (ASCEND/REPROBE/HOLD/SPLIT) are designed to detect and break circular reasoning by forcing you to either commit to a direction or acknowledge you need more data.
- **Quality of decomposition matters more than speed of ideation.** Driftwave is deliberately slower — it won't let you skip to implementation until the structure converges.
- **Cross-system or cross-domain work.** The `/boundary-mode` skill and sheaf-valued synthesis (L3) are designed for problems that span multiple systems or conceptual domains where local consistency doesn't guarantee global consistency.

### Use /superpowers:brainstorming (or similar) when:

- **You need volume and speed.** Traditional brainstorming plugins are optimized for rapid idea generation and ranking. If you already know the problem structure and just need creative options, they're faster.
- **The problem is well-scoped.** If you already know what you're building and just need to explore implementation approaches, the full L0-L3 pipeline is overhead.
- **You want a lighter touch.** Driftwave's axioms are strict by design (no averaging, no layer skipping, hard implementation gates). Sometimes you just want a quick creative session.

### The honest comparison:

| | driftwave | Traditional brainstorming plugins |
|---|---|---|
| **Speed** | Slower (by design) | Faster |
| **When it shines** | Ambiguous, multi-dimensional problems | Well-scoped creative tasks |
| **Failure mode** | Over-rigorous on simple problems | Under-rigorous on complex ones |
| **Learning curve** | Steep (topological vocabulary) | Gentle |
| **Output** | Validated structure with convergence proof | Ranked idea list |
| **Philosophy** | "Don't build until you know the shape" | "Generate, evaluate, decide" |

They're not competitors — they solve different problems. Use brainstorming plugins for ideation velocity. Use driftwave when you suspect the problem is more complex than it looks and you need structural rigor to avoid building the wrong thing.

## Five Axioms

These are non-negotiable constraints enforced throughout the pipeline:

1. **NO_AVERAGING** — Raw probes never averaged before filtration. The noise floor contains signal.
2. **UPWARD_FLOW** — L0 → L1 → L2 → L3, no layer skipping. Each gate must pass.
3. **WAYPOINT_ROUTING** — Routing decisions (ASCEND, REPROBE, HOLD, SPLIT) are topological phase transitions, not timers.
4. **SHAPE_OVER_COUNT** — Gini trajectory dominates raw feature count. Validated at r=0.935 across four LLM architectures.
5. **ADAPTIVE_SCALE** — Epsilon parameters derived from data geometry, never user preference. The number of clusters emerges from persistence.

## Skills

| Skill | Layer | What It Does |
|-------|-------|--------------|
| `/dw-map` | L0 | Ingests raw artifacts as a point cloud. Entropy gate rejects zero-variance inputs — if your data is too uniform, it asks for more before proceeding. |
| `/dw-filter` | L1 | Builds a persistence barcode from the point cloud. Long-lived clusters become viable approaches; short-lived ones are noise (YAGNI). The number of approaches is determined by data geometry, not preference. |
| `/dw-ascend` | L2/L3 | L2: Detects structural loops and monitors Gini trajectory. Routes based on whether structure is hierarchifying (+) or flattening (-). L3: Validates global consistency via sheaf Laplacian — are all the pieces actually compatible? |
| `/wavefront` | ALL | Full pipeline orchestrator. Runs L0→L1→L2→L3 in sequence, enforcing all five axioms at every transition. Use this when you want the complete treatment. |
| `/topological-brainstorm` | ALL | The brainstorming-specific pipeline. Same L0-L3 progression but tuned for ideation: idea-spaces as point clouds, approaches as persistent clusters, design coherence as loops. Hard gate: no implementation until the structure converges. |
| `/boundary-mode` | L3 | For deep human-machine collaboration at the highest abstraction level. Activates when you signal you want sheaf-valued output rather than scalar summaries. |

### Pipeline Flow

```
L0: /dw-map          Ingest raw artifacts → entropy gate
      │
      ▼ (pass)
L1: /dw-filter        H₀ persistence → identify viable clusters
      │
      ▼ (clusters found)
L2: /dw-ascend        H₁ loops + Gini monitoring → routing
      │                  ├─ slope > +0.01 → ASCEND
      │                  ├─ slope < -0.01 → REPROBE ↩
      │                  ├─ |slope| < 0.01 → HOLD
      │                  └─ waypoints > 3  → SPLIT
      ▼ (ascend)
L3: /dw-ascend --sheaf  Sheaf Laplacian → global consistency
      │                  ├─ ker(L_F) converges → ON-SHELL ✓
      │                  └─ obstruction persists → surface to human
      ▼
    Waypoint gate → implement or iterate
```

## Agents

| Agent | Purpose |
|-------|---------|
| `gini-watchdog` | Background monitor during L2/L3 work. Continuously evaluates whether your output's structural hierarchy is improving or degrading, and recommends routing (ASCEND/REPROBE/HOLD/SPLIT). |

## CLI Tool

The plugin includes `scripts/topo.sh` for applying the pipeline to the project itself:

```bash
topo scan          # L0: Raw artifact inventory
topo cluster       # L1: Identify persistent work clusters from git history
topo synthesize    # L2: Validate documentation structure and skill configs
topo validate      # L3: Sheaf consistency check (image refs, JSON schemas, axiom docs)
topo figure-it-out # Full L0→L1→L2→L3 pipeline
topo serve         # Build and serve the docs-site
```

## Install

```bash
claude plugin marketplace add gh:RogueGringo/JTopo
claude plugin install driftwave
```

## The Core Principle

The quality of a process is not determined by its state at any single phase, but by the trajectory of its topological evolution across all phases. Shape over count. Trajectory over snapshot.
