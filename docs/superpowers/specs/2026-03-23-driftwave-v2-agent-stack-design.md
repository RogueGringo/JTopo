# Driftwave V2 — Agent Stack Architecture

**Date:** 2026-03-23
**Status:** Draft
**Origin:** L2 synthesis from topological-brainstorm session

---

## 1. Problem

Driftwave V1 has 6 skills (text prompts) and 1 agent (Gini Watchdog). The mathematical language (H₀ persistence, sheaf Laplacian, Gini trajectory) reads as decoration — users cannot verify whether the topology is actually shaping output. The interaction model is text-in/text-out with no visual feedback, no verifiable computation, and no structural enforcement of the pipeline.

## 2. Solution

Replace the prompt-based pipeline with a **4-agent stack** where each layer is a specialized agent producing typed artifacts. The topology becomes the routing protocol — not metaphor but structure. Artifacts are JSON packets with enforced schemas. Agents are dispatched per-layer with model tiers matched to compute complexity. A local LLM on the RTX 5070 handles the fast/cheap layers. API models handle reasoning and judgment.

## 3. Architecture

### 3.1 Compute Stack

| Layer | Agent | Compute | Model | Latency | Cost |
|-------|-------|---------|-------|---------|------|
| L0 | dw-ingest | Local RTX 5070 | Llama 3.2 3B (4-bit) | ~2s | Free |
| L1 | dw-cluster | Local RTX 5070 (Python) + Local LLM | PyTorch persistence + Llama 3B | ~10s | Free |
| L2 | dw-synthesize | Anthropic API | Sonnet | ~30s | API |
| L3 | dw-review | Anthropic API | Opus | ~60s | API |
| Viz | Dashboard | Local browser | Three.js | Instant | Free |
| Compute | Persistence | Local RTX 5070 | PyTorch (ATFT code) | ~5s | Free |

**VRAM budget:** Llama 3.2 3B at 4-bit ≈ 2GB. PyTorch persistence computation ≈ 1-2GB. Total ≈ 4GB, leaving 8GB headroom on the 12GB RTX 5070. No conflict with ATFT experiments if run separately.

### 3.2 Artifact Type System

Each inter-agent handoff is a typed JSON artifact. The type system enforces UPWARD_FLOW structurally — an L1 agent literally cannot receive a raw file list, only a RawCloud artifact.

```
L0 → RawCloud {
  files: [{path, content_hash, language, size_bytes}],
  git_state: {branch, recent_commits: [{hash, message, files_changed}], dirty_files},
  docs: [{path, staleness_days, type: "spec"|"plan"|"log"|"theory"}],
  memory: [{path, type, description}],
  entropy: float  // variance across artifacts. <0.1 = REJECT (no differentiation)
}

L1 → FilteredTopology {
  clusters: [{
    id: int,
    label: string,
    members: [file_paths],
    bar_length: float,  // persistence — longer = more real
    centroid_description: string
  }],
  barcode: [{birth: float, death: float, dimension: int}],
  noise: [file_paths],  // filtered out — short bars
  distances: [[float]],  // pairwise distance matrix (for viz)
  routing: "ASCEND" | "REPROBE" | "SPLIT",
  routing_reason: string
}

L2 → SynthesisMap {
  sections: [{
    title: string,
    content: string,
    source_cluster: int,
    coherence_score: float,  // 0-1
    gini_slope: float  // positive = hierarchifying
  }],
  loops: [{
    feature: string,
    sections: [int, int],  // which sections create the loop
    status: "OPEN" | "CLOSED"
  }],
  trajectory: [float],  // gini values over time
  routing: "ASCEND" | "REPROBE" | "SPLIT"
}

L3 → SheavedVerdict {
  sections: [{...L2.sections, compatibility: "CONSISTENT" | "INCOMPATIBLE"}],
  kernel_dim: int,  // ker(L_F) — number of globally consistent sections
  obstructions: [{
    section_a: int,
    section_b: int,
    incompatibility: string  // what contradicts
  }],
  verdict: "ON_SHELL" | "OFF_SHELL",
  verdict_reason: string
}
```

### 3.3 Agent Specifications

**L0 Agent — `dw-ingest`**
- **Role:** Raw artifact scanner. No interpretation, no summary.
- **Model:** Local Llama 3.2 3B (4-bit quantized) via `llama.cpp` or `vllm`
- **System prompt guardrail:**
  ```
  You are dw-ingest. You ONLY output RawCloud JSON artifacts.
  You scan the provided file listing and classify each file by:
  language, size, staleness (days since last git modification).
  You compute entropy as the standard deviation of file sizes
  normalized by mean.
  Output ONLY valid JSON matching the RawCloud schema.
  Any text outside the JSON block is an error.
  ```
- **Tools:** File listing provided as input (from controller's Glob/Grep)
- **Output validation:** JSON schema check. Reject and retry if malformed.
- **Fallback:** If local LLM unavailable, controller builds RawCloud directly from `topo.sh scan` output (already produces similar data)

**L1 Agent — `dw-cluster`**
- **Role:** Persistent clustering on the artifact space.
- **Model:** Hybrid — Python for persistence computation, local LLM for cluster labeling
- **Process:**
  1. Receive RawCloud artifact
  2. Compute pairwise semantic distances (file content similarity via edit-distance or TF-IDF)
  3. Run Vietoris-Rips filtration → H₀ persistence barcode (Python, using ATFT code patterns)
  4. Identify clusters from long bars
  5. Local LLM labels each cluster with a human-readable description
  6. Route: all bars short → REPROBE, clear clusters → ASCEND, >3 clusters → SPLIT
- **Compute:** `scripts/compute_persistence.py` (new file, uses scipy/numpy)
- **Output validation:** JSON schema check on FilteredTopology

**L2 Agent — `dw-synthesize`**
- **Role:** Design synthesis from filtered clusters.
- **Model:** Anthropic API — Sonnet (needs real writing ability)
- **Dispatched as:** Claude Code subagent with FilteredTopology artifact as context
- **System prompt:** Includes the artifact type definitions and Gini monitoring instructions
- **Process:**
  1. Receive FilteredTopology artifact + relevant source files (cluster members)
  2. Write one design section per cluster
  3. Monitor Gini trajectory (are dominant sections emerging?)
  4. Detect H₁ loops (section A references concept from section B's cluster → loop)
  5. Route: negative Gini slope → REPROBE, open loops → iterate, clean → ASCEND
- **Key constraint:** Each section maps to exactly one cluster. Cross-cluster references are flagged as loops.

**L3 Agent — `dw-review`**
- **Role:** Global consistency check (sheaf review).
- **Model:** Anthropic API — Opus (needs judgment + global view)
- **Dispatched as:** Claude Code subagent with SynthesisMap artifact
- **Process:**
  1. Receive SynthesisMap with all sections
  2. Check pairwise compatibility of all sections (do interfaces agree?)
  3. Compute kernel dimension (how many sections compose globally)
  4. Identify obstructions (contradictions between sections)
  5. Verdict: ON_SHELL (all consistent) or OFF_SHELL (obstruction found)
- **Key property:** This is the ONLY agent that sees all sections simultaneously. L2 sees them one at a time.

### 3.4 Routing Protocol

```
User invokes /wavefront or /topological-brainstorm
  │
  Controller (main session):
  │
  ├─→ Dispatch L0 (local LLM or topo.sh fallback)
  │     → RawCloud artifact saved to /tmp/dw-artifacts/raw.json
  │     │
  │     ├─ entropy < 0.1? → Ask user for more input
  │     └─ entropy OK → continue
  │
  ├─→ Run compute_persistence.py on RawCloud
  │     → Distance matrix + barcode
  │
  ├─→ Dispatch L1 (local LLM for labeling)
  │     Input: RawCloud + barcode
  │     → FilteredTopology artifact saved to /tmp/dw-artifacts/filtered.json
  │     │
  │     ├─ REPROBE → back to L0
  │     ├─ SPLIT → fork sub-pipelines
  │     └─ ASCEND → continue
  │
  ├─→ Dispatch L2 (sonnet subagent)
  │     Input: FilteredTopology + source files
  │     → SynthesisMap artifact saved to /tmp/dw-artifacts/synthesis.json
  │     │
  │     ├─ REPROBE → back to L1
  │     └─ ASCEND → continue
  │
  └─→ Dispatch L3 (opus subagent)
        Input: SynthesisMap + spec docs
        → SheavedVerdict artifact saved to /tmp/dw-artifacts/verdict.json
        │
        ├─ OFF_SHELL → report obstruction → human decides
        └─ ON_SHELL → W(I) ∈ W_phys → implementation gate open
```

### 3.5 Local LLM Infrastructure

**Runtime:** `llama-cpp-python` with CUDA backend (already have PyTorch+CUDA on the RTX 5070)

**Model:** Llama 3.2 3B Instruct (Q4_K_M quantized, ~2GB VRAM)
- Download: `huggingface-cli download TheBloke/Llama-3.2-3B-Instruct-GGUF`
- Or distill a purpose-built model via Chopper Stan (the project's own distillation pipeline)

**Server:** Persistent local inference server on localhost:8080
- Start: `python -m llama_cpp.server --model models/llama-3.2-3b-q4.gguf --n_gpu_layers -1 --port 8080`
- OpenAI-compatible API — agents call it the same way they'd call any LLM

**Guardrails:**
- System prompt constrains output to JSON-only
- Output parsed by JSON schema validator before acceptance
- Max tokens capped (256 for L0, 512 for L1 labeling)
- Temperature 0.0 for deterministic classification
- If local model produces invalid output 3 times → fallback to `topo.sh` deterministic path

**Chopper Stan recursion:** The driftwave plugin can eventually use Chopper Stan's distillation pipeline to create a purpose-built "dw-ingest" model — a student model trained specifically on the RawCloud classification task. The topology of the teacher's understanding, transferred to a model that fits on the GPU alongside the experiments. Stan builds Stan.

### 3.6 Three.js Visualization Dashboard

**What it shows:**
1. **Force-directed graph** — nodes = artifacts, edges = similarity above threshold, colored by cluster assignment. Updates live as L1 produces FilteredTopology.
2. **Persistence barcode** — horizontal bars showing birth/death of each topological feature. Long bars = real structure, short bars = noise. Color-coded by dimension (H₀ blue, H₁ orange).
3. **Pipeline progress** — four-stage indicator (L0→L1→L2→L3) with current status, routing decisions, and artifact sizes.
4. **Gini trajectory** — time series chart showing hierarchy evolution during L2 synthesis.

**Implementation:**
- Served by `topo.sh serve` (already exists, uses Vite)
- Reads artifact JSON files from `/tmp/dw-artifacts/`
- Polls for changes every 2 seconds (or uses filesystem watcher)
- Three.js for the force-directed graph (ForceGraph3D library or custom)
- Canvas API for barcode and Gini charts (same pattern as the Ti site charts)
- No build step required — Vite hot-reloads

**When it opens:**
- `/wavefront` command auto-opens the dashboard in the default browser
- Dashboard persists across pipeline stages, updating as each artifact is produced
- User can see their idea-space being filtered in real time

### 3.7 ATFT Bridge — Real Persistence Computation

**New file:** `scripts/compute_persistence.py`

```python
"""Compute persistent homology on artifact distance matrices.

Same mathematics as atft/topology/ but applied to code/idea spaces
instead of zeta zero point clouds.
"""
import json
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_h0_persistence(D):
    """Union-Find H0 persistence on distance matrix."""
    n = D.shape[0]
    parent = list(range(n))
    rank = [0] * n
    births = {i: 0.0 for i in range(n)}
    barcodes = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, eps):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        barcodes.append({"birth": births[rb], "death": eps, "dimension": 0})
        del births[rb]

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((D[i,j], i, j))
    edges.sort()

    for eps, i, j in edges:
        union(i, j, eps)

    for comp in births:
        barcodes.append({"birth": births[comp], "death": float('inf'), "dimension": 0})

    return barcodes

def main():
    raw = json.load(sys.stdin)
    # Build distance matrix from file features
    files = raw.get("files", [])
    n = len(files)
    if n < 2:
        print(json.dumps({"barcode": [], "distances": []}))
        return

    # Feature vectors: [size_normalized, staleness_normalized, language_hash]
    sizes = np.array([f.get("size_bytes", 0) for f in files], dtype=float)
    stale = np.array([f.get("staleness_days", 0) for f in files], dtype=float)

    if sizes.max() > 0: sizes /= sizes.max()
    if stale.max() > 0: stale /= stale.max()

    features = np.column_stack([sizes, stale])
    D = squareform(pdist(features, metric='euclidean'))

    barcodes = compute_h0_persistence(D)

    print(json.dumps({
        "barcode": barcodes,
        "distances": D.tolist()
    }))

if __name__ == "__main__":
    main()
```

This is a lightweight persistence computation using the same Union-Find algorithm that underlies Ripser. For idea-spaces (tens to hundreds of points), it runs in milliseconds. For larger artifact sets, the ATFT's batched GPU path can be invoked.

## 4. What Changes in the Plugin

| Component | V1 (Current) | V2 (Proposed) |
|-----------|-------------|--------------|
| Skills | 6 text prompts that instruct Claude | 6 skills that instruct the CONTROLLER to dispatch agents |
| Agents | 1 (Gini Watchdog) | 5 (4 layer agents + Gini Watchdog) |
| Computation | None (metaphor only) | Real persistence via `compute_persistence.py` |
| Visualization | None | Three.js dashboard with live topology |
| Local LLM | None | Llama 3.2 3B on RTX 5070 for L0/L1 |
| Artifact protocol | None (free text) | Typed JSON with schema validation |
| `topo.sh` | 508-line scan/validate script | Refactored as L0 fallback + artifact generator |
| Axiom enforcement | By instruction | By type system (structural) |

## 5. New Files

```
driftwave/
├── agents/
│   ├── gini-watchdog.md      (existing)
│   ├── dw-ingest.md          (NEW — L0 agent spec)
│   ├── dw-cluster.md         (NEW — L1 agent spec)
│   ├── dw-synthesize.md      (NEW — L2 agent spec)
│   └── dw-review.md          (NEW — L3 agent spec)
├── scripts/
│   ├── topo.sh               (existing, refactored)
│   ├── compute_persistence.py (NEW — ATFT bridge)
│   └── start_local_llm.sh    (NEW — llama.cpp server launcher)
├── schemas/
│   ├── raw_cloud.json        (NEW — L0 artifact schema)
│   ├── filtered_topology.json (NEW — L1 artifact schema)
│   ├── synthesis_map.json    (NEW — L2 artifact schema)
│   └── sheaved_verdict.json  (NEW — L3 artifact schema)
├── docs-site/
│   └── src/
│       ├── components/
│       │   ├── ForceGraph.jsx (NEW — Three.js topology viz)
│       │   ├── Barcode.jsx    (NEW — persistence barcode viz)
│       │   ├── GiniChart.jsx  (NEW — trajectory chart)
│       │   └── Pipeline.jsx   (existing, upgraded with live status)
│       └── ...
└── models/                    (NEW — local LLM weights, .gitignored)
    └── .gitkeep
```

## 6. Implementation Order

1. **Artifact schemas** — define the JSON schemas first (the algebra)
2. **compute_persistence.py** — the ATFT bridge (the computation)
3. **4 agent specs** — dw-ingest, dw-cluster, dw-synthesize, dw-review
4. **Refactor /wavefront skill** — dispatch agents instead of prompting
5. **Local LLM setup** — llama.cpp server + start script
6. **Three.js dashboard** — force graph + barcode + pipeline + gini
7. **Integration test** — run full pipeline on the JTopo codebase itself

## 7. Out of Scope

- Custom-distilled orchestration model via Chopper Stan (future — after V2 is stable)
- Multi-GPU orchestration (single RTX 5070 for now)
- Persistent artifact storage across sessions (artifacts are ephemeral in /tmp/)
- Remote collaboration (single-user, local machine)
