---
description: "Full pipeline — runs L0→L1→L2→L3 sequentially with automatic routing. The `wavefront` skill as a single invocation."
arguments: "[task description]"
---

# /driftwave:run

Execute the full driftwave pipeline from ingestion to verdict.

## What to do

This is the operational form of the `wavefront` skill. Run all four layers sequentially:

1. **L0 — Ingest**: Scan the codebase, produce RawCloud
   - `mkdir -p /tmp/dw-artifacts`
   - Dispatch dw-ingest agent OR run topo.sh scan
   - Check entropy gate

2. **L1 — Filter**: Run persistence, identify clusters
   - Execute compute_persistence.py on the RawCloud
   - Dispatch dw-cluster agent for labeling
   - Check routing: REPROBE → back to L0, SPLIT → decompose, ASCEND → continue

3. **L2 — Synthesize**: Build design from clusters
   - Dispatch dw-synthesize agent (sonnet) with FilteredTopology + source files
   - Monitor Gini trajectory
   - Check loops: iterate if open (max 3), REPROBE → back to L1, ASCEND → continue

4. **L3 — Review**: Sheaf consistency check
   - Dispatch dw-review agent (opus) with SynthesisMap
   - ON_SHELL → report success, implementation gate open
   - OFF_SHELL → report obstructions, route back to L2 (max 3 iterations)

5. **Meta**: Append this session to meta-persistence
   - Update `/tmp/dw-artifacts/meta.json`
   - Compute meta-persistence

If a task description argument is provided, use it as context for the L2 synthesis (what are we building/analyzing?).

## Progress reporting

After each layer, report:
```
✓ L0: 47 files scanned, entropy=1.23
✓ L1: 3 clusters, routing=ASCEND
  → Topology Engine (12 files)
  → Experiment Pipeline (8 files)
  → Visualization (5 files)
◐ L2: synthesizing...
```

## Axiom enforcement

All five axioms are enforced structurally:
- NO_AVERAGING: RawCloud has no summary field
- UPWARD_FLOW: Each agent only accepts its layer's input type
- WAYPOINT_ROUTING: Routing on ASCEND/REPROBE/SPLIT, not on timers
- SHAPE_OVER_COUNT: L2 monitors Gini trajectory, not section count
- ADAPTIVE_SCALE: Persistence threshold from data geometry, compute from GPU state
