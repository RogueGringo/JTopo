# Topological Brainstorming: Dimensionalizing the Superpowers Workflow via Adaptive Topological Field Theory and the Driftwave Pipeline

## Executive Summary

This archive constructs a formal synthesis between the procedural brainstorming workflow of [obra/superpowers](https://github.com/obra/superpowers) (a structured ideation-to-implementation skill for Claude Code) and two mathematical frameworks authored by Aaron Jones: the **Adaptive Topological Field Theory (ATFT)** and the **driftwave plugin** (an operative realization of ATFT for agentic systems). The result is a *dimensionalized* brainstorming methodology in which each phase of creative ideation is not merely a sequential step but a topological operation on idea-spaces — with measurable invariants, quality gates derived from persistent homology, and formal routing decisions based on Gini trajectories and Betti curve dynamics.

The core claim is structural: the superpowers:brainstorming workflow already implicitly performs topological filtration on a design space. By making this explicit through the ATFT formalism and operationalizing it through the driftwave abstraction layers (L0 → L1 → L2 → L3), the brainstorming process acquires mathematical rigor, principled quality gates, and the capacity for adaptive self-correction that mirrors how field equations emerge from topological waypoints.

---

## The Three Source Frameworks

### Superpowers: Brainstorming

The [superpowers:brainstorming skill](https://github.com/obra/superpowers) implements a nine-step creative workflow designed to transform vague ideas into validated designs before any implementation occurs ([Superpowers blog, 2025](https://blog.fsck.com/2025/10/09/superpowers/)). Its step sequence is:

1. **Explore project context** — examine files, docs, recent commits
2. **Offer visual companion** — assess whether visual modalities are needed
3. **Ask clarifying questions** — one at a time, probe purpose/constraints/success criteria
4. **Propose 2-3 approaches** — with trade-offs and a recommendation
5. **Present design** — incrementally, section by section, with approval gates
6. **Write design doc** — persist the validated design as a spec
7. **Spec review loop** — dispatch subagent reviewer, iterate until approved (max 3)
8. **User reviews written spec** — human gate before implementation
9. **Transition to implementation** — invoke the writing-plans skill

Key principles include the HARD-GATE (no implementation before design approval), YAGNI (aggressive scope reduction), and the anti-pattern "This Is Too Simple To Need A Design" (even trivial tasks pass through the pipeline).

### Adaptive Topological Field Theory (ATFT)

The ATFT paper constructs a mathematical framework in which field equations emerge as geometric constraints on topological evolution under adaptive filtration. Its central object is the **Adaptive Topological Operator** \(\gimel^{(\mathrm{adp})}_{\mathrm{PH}}\), which assigns to each field configuration a family of persistence diagrams parameterized by the filtration scale \(\varepsilon\). The framework rests on three pillars:

- **Adaptive filtration**: \(\varepsilon\) is a scale variable, not a static threshold. The fundamental observable is the trajectory of topological evolution across scales.
- **The Čech–de Rham bridge**: The isomorphism between smooth de Rham cohomology and discrete Čech cohomology guarantees that persistent homology on Rips complexes computes the same cohomological invariants as continuous differential geometry ([Bott & Tu, 1982](https://link.springer.com/book/10.1007/978-1-4757-3951-0)).
- **Sheaf-valued homology**: Lie algebra fibers attached to the Rips complex via cellular sheaves upgrade output from scalar Betti numbers to algebraic-valued sections, resolving the category mismatch between TDA and gauge field theory ([Hansen & Ghrist, 2019](https://link.springer.com/article/10.1007/s41468-019-00038-7)).

The **waypoint principle** states: field equations are topological waypoints — critical scales at which the topology of a configuration undergoes qualitative change. The **Gini trajectory** (how the Gini coefficient of persistence lifetime distributions evolves across \(\varepsilon\)) is shown to be the strongest predictor of system quality, validated at \(r = 0.935\) cross-model correlation across four LLM architectures.

### Driftwave Plugin

The driftwave plugin operationalizes ATFT as a Claude Code CLI pipeline organized into four abstraction layers:

| Layer | Command | Operation | Topological Content |
|-------|---------|-----------|-------------------|
| L0 | `/dw-map` | Ingest raw artifacts → point cloud | Raw variance structure as topological signal |
| L1 | `/dw-filter` | Vietoris-Rips filtration → H₀ persistent clustering | Module routing via persistence bar lifetime |
| L2/L3 | `/dw-ascend` | H₁ topology + Gini-slope quality gate | Sheaf-valued synthesis with Lie-algebra structure |
| ALL | `@wavefront` | Full pipeline orchestrator | Enforces all axioms across layers |

Five axioms govern the pipeline:
1. **NO_AVERAGING** — raw probes never averaged before filtration
2. **UPWARD_FLOW** — L0 → L1 → L2 → L3, no layer skipping
3. **WAYPOINT_ROUTING** — routing decisions are topological phase transitions
4. **SHAPE_OVER_COUNT** — Gini trajectory dominates raw Betti number
5. **ADAPTIVE_SCALE** — \(\varepsilon_{\max}\) always derived from data geometry

---

## The Isomorphism: Brainstorming Steps as Topological Layers

The synthesis maps each superpowers:brainstorming phase onto the driftwave abstraction hierarchy, with the ATFT providing the mathematical semantics. The brainstorming "idea space" is treated as a point cloud in a metric space where distance encodes semantic dissimilarity between concepts, requirements, and design options.

### L0 — Raw Ingestion (Steps 1-3: Context → Questions)

**Superpowers mapping**: Steps 1-3 (explore project context, offer visual companion, ask clarifying questions) correspond to the L0 `/dw-map` operation. Raw artifacts are ingested without premature synthesis.

**ATFT formalization**: Let \(\mathcal{I}\) be the "idea configuration" — the totality of project context, user intent, codebase state, and constraints. The feature map \(\varphi : \mathcal{I} \to (X, d)\) produces a finite metric space where:
- Each point \(x_i \in X\) is a discrete datum: a file, a commit, a stated requirement, an answer to a clarifying question
- The distance \(d(x_i, x_j)\) encodes semantic/functional dissimilarity

**The NO_AVERAGING axiom is critical here.** The superpowers skill explicitly asks one question at a time, never bundling. This mirrors the driftwave axiom that raw probes must never be aggregated before filtration — the variance structure across individual answers *is* the topological signal. Premature summarization (averaging) destroys the fine structure that subsequent filtration needs.

**Entropy gate**: The driftwave `entropy_gate.py` hook blocks zero-variance inputs. In the brainstorming context, this translates to: if the user's answers contain no differentiation (all requirements identical, no constraints, no trade-offs), the process should REPROBE — ask more probing questions rather than proceeding with a flat input space.

**Output**: A point cloud \(\{x_i\}_{i=1}^{N}\) in idea-space with pairwise distance matrix, preserving full variance structure. No interpretation yet.

### L1 — Persistent Clustering and Module Routing (Step 4: Propose Approaches)

**Superpowers mapping**: Step 4 (propose 2-3 approaches with trade-offs and recommendation) is the L1 `/dw-filter` operation. The raw idea-cloud is now filtered to identify stable clusters — the distinct viable approaches.

**ATFT formalization**: Construct the Vietoris-Rips complex \(\mathrm{Rips}_\varepsilon(\varphi(\mathcal{I}))\) and compute H₀ persistence. The H₀ barcode reveals:

| H₀ Bar Lifetime | Interpretation | Action |
|-----------------|---------------|--------|
| \(> \text{threshold}\) | Stable conceptual cluster — a viable approach | Promote to L1 module (present as approach option) |
| \(\leq \text{threshold}\) | Topological noise — a tangential or unstable idea | Discard (YAGNI) |
| All bars short | No clear clustering — the idea space is too diffuse | REPROBE — return to L0, ask more questions |

The superpowers instruction to "propose 2-3 approaches" corresponds to identifying 2-3 high-persistence H₀ components — connected clusters of requirements and possibilities that survive filtration across scales. The recommendation corresponds to selecting the cluster with the longest bar (most persistent, most robust to perturbation).

**Adaptive scale**: The \(\varepsilon_{\max}\) is set as the 95th percentile of pairwise distances in the idea-space, not a fixed threshold. This means the number of approaches is determined by the data's own geometry — sometimes 2 approaches, sometimes 3, determined by where the natural gaps fall.

**Routing decision**: If the H₀ persistence analysis reveals all bars are short (no clear approaches emerge), the REPROBE signal fires — just as the superpowers process can "go back and clarify if something doesn't make sense." The brainstorming process descends back to L0.

### L2 — Topological Synthesis (Steps 5-6: Design → Document)

**Superpowers mapping**: Steps 5-6 (present design section by section with approval, write design doc) correspond to `/dw-ascend` at the L2 level. The selected approach is now synthesized into a coherent design through H₁ topology.

**ATFT formalization**: After L1 selects a module (a chosen approach), the H₁ persistent homology detects loops — closed paths in the design space that represent internal consistency constraints, circular dependencies, or coherence structures. The design sections correspond to examining these loops:

- Architecture → data flow → error handling → testing forms a loop that must close (the architecture must support the error handling which must feed the testing which validates the architecture)
- Each "section" in the brainstorming skill maps to examining a different H₁ feature

The **incremental validation** principle (present section by section, get approval) corresponds to the Gini routing table:

| Gini Slope | Decision | Brainstorming Translation |
|-----------|----------|--------------------------|
| \(> +0.01\) | ASCEND | Design is hierarchifying — proceed to next section |
| \(< -0.01\) | REPROBE | Design is degrading — revisit earlier assumptions |
| Within \(\pm 0.01\) | HOLD | Design is stable — maintain current level, await new input |
| Waypoints \(> 3\) | SPLIT | Dimensional branch — the design needs to fork into sub-specs |

The superpowers skill explicitly handles the SPLIT case: "if the request describes multiple independent subsystems, flag this immediately... help the user decompose into sub-projects." This is precisely the driftwave SPLIT routing when waypoints exceed 3 — the topological complexity indicates the idea-space has branched into multiple independent dimensional branches that need separate treatment.

**Gini watchdog**: The `gini_watchdog.py` post-hook monitors for negative Gini slope. In brainstorming terms: if successive design sections are making the overall design *less* coherent (the hierarchy is flattening, not sharpening), the process triggers REPROBE. This is the formal version of the superpowers principle "Be ready to go back and clarify."

### L3 — Sheaf-Valued Synthesis (Steps 7-8: Review Loop → Human Gate)

**Superpowers mapping**: Steps 7-8 (spec review loop with subagent, user reviews spec) correspond to `/dw-ascend` at the L3 level — sheaf-valued synthesis with Lie-algebra structure.

**ATFT formalization**: At L3, the output is no longer scalar (a "good/bad" assessment) but sheaf-valued — it carries algebraic structure. The spec review loop is precisely this: the subagent reviewer doesn't just check "is this design correct?" but evaluates the design as a section of a sheaf — checking whether local design decisions (at each component) are globally consistent under the restriction maps (interfaces between components).

The **sheaf Laplacian kernel** \(\ker(L_\mathcal{F})\) encodes field equation solutions. In the brainstorming context: a design spec lives in \(\ker(L_\mathcal{F})\) if and only if every component's local specification is compatible with every other component's specification under the interface constraints. The review loop iterates until the spec converges to a global section — a globally consistent design.

The max-3-iterations bound on the review loop corresponds to the ATFT's recognition that if convergence hasn't occurred after sufficient iterations, the obstruction may be fundamental — "surface to human for guidance" is the escape from a topological obstruction that the automated process cannot resolve.

### Transition: The Waypoint Principle as Implementation Gate

**Superpowers mapping**: Step 9 (invoke writing-plans) is the terminal state — the topological waypoint at which the brainstorming trajectory undergoes a qualitative phase transition from ideation to implementation.

**ATFT formalization**: The waypoint principle states that field equations are constraints selecting which topological evolution trajectories correspond to physical configurations. Analogously, the accumulated waypoints across L0→L3 define the "field equation" of the design:

\[
\mathcal{W}(\mathcal{I}) = \left(\varepsilon^*, \{\varepsilon_{w,i}\}_{i=1}^{n_w}, \{\delta_1(\varepsilon_{w,i})\}_{i=1}^{n_w}, \mathcal{G}_1(\varepsilon^*), \left.\frac{d\mathcal{G}_1}{d\varepsilon}\right|_{\varepsilon^*}\right) \in \mathbb{R}^{2n_w+3}
\]

This waypoint signature captures: the onset scale (when did coherent design first emerge?), the critical transitions (where did the design qualitatively change during iteration?), and the Gini trajectory (is the design hierarchifying or flattening?). The implementation gate fires when \(\mathcal{W}(\mathcal{I}) \in \mathcal{W}_{\text{phys}}\) — the waypoint signature satisfies the constraints that define a viable design.

---

## The Dimensional Reduction of Creative Work

The ATFT's dimensional reduction pathway maps directly onto how brainstorming compresses infinite possibility into executable specification:

| ATFT Stage | Dimension | Brainstorming Analogue |
|-----------|-----------|----------------------|
| Configuration space | \(\infty\)-dimensional | All possible designs, architectures, implementations |
| Feature space | \(d\)-dimensional | The structured representation after context exploration |
| Simplicial complex | Combinatorial | The Rips complex of ideas — which concepts connect at what scale |
| Sheaf cohomology | \(\mathfrak{g}\)-valued | The algebraically-typed review of global consistency |
| Persistence diagrams | 2D per feature | Each design choice as a birth-death pair in idea-space |
| Waypoint signature | \((2n_w+3)\)-D | The finite-dimensional design spec |
| Scalar invariants | 1D each | Go/no-go decisions, quality scores |

The **Two-Dimensional Sufficiency Proposition** (ATFT Proposition 5.1) has a direct brainstorming analogue: the Betti curve of the idea-space — how many independent conceptual loops survive at each scale of scrutiny — is sufficient to discriminate viable designs from unviable ones, provided the "feature map" captures the right dimensions of the problem (analogous to the parity-complete requirement).

---

## Axioms of Topological Brainstorming

Synthesizing the axioms from driftwave with the principles from superpowers:brainstorming:

### Axiom 1: NO_AVERAGING (Preserve Variance)
Never aggregate, summarize, or flatten raw requirements before the filtration phase. Individual user responses, codebase signals, and constraint statements are the topological raw material. Premature consensus destroys the persistence signal that reveals which requirements are truly robust versus topological noise.

**Violation detector**: If someone attempts to "summarize the requirements so far" before proposing approaches, this triggers `[DW-AXIOM-VIOLATION]` — a category error equivalent to averaging probes before filtration.

### Axiom 2: UPWARD_FLOW (No Layer Skipping)
L0 → L1 → L2 → L3. No implementation without design. No design without approaches. No approaches without raw context. This is the HARD-GATE from superpowers, elevated to a topological invariant. The Čech–de Rham bridge guarantees that information faithfully transfers between layers — but only in the correct direction.

### Axiom 3: WAYPOINT_ROUTING (Phase Transition Gates)
Every routing decision between phases is a topological phase transition, not a timer or a checklist. The transition from "asking questions" to "proposing approaches" fires when the H₀ persistence diagram exhibits sufficient clustering — not after a fixed number of questions. The transition from "design" to "implementation" fires when the Gini trajectory indicates hierarchification — not after a fixed number of design sections.

### Axiom 4: SHAPE_OVER_COUNT (Quality Over Quantity)
The Gini trajectory of the design's topological evolution matters more than the count of features, requirements, or design sections. A design with 3 deeply coherent sections (positive Gini slope, hierarchifying) outperforms a design with 12 scattered sections (negative Gini slope, flattening). This is validated by ATFT Section 6.4 showing that positive Gini trajectory (+0.025/layer) outperforms negative trajectory (−0.007/layer) across LLM architectures.

### Axiom 5: ADAPTIVE_SCALE (Data-Driven Thresholds)
The scope of brainstorming — how many questions to ask, how many approaches to propose, how many design sections to write — is determined by the data's own geometry (95th percentile of pairwise distances), never by a fixed number. The superpowers skill's injunction to "scale each section to its complexity" is the qualitative version of this axiom.

---

## Operationalization: The Wavefront Agent for Brainstorming

The driftwave `@wavefront` agent orchestrates the full pipeline. Adapted for brainstorming:

```
Topological Brainstorming Pipeline (@wavefront-brainstorm)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. /dw-map [idea_artifacts]          ← Ingest: context, files, user intent
   ├─ entropy_gate check             ← Reject zero-variance inputs
   └─ Output: idea point cloud

2. /dw-filter [--eps-percentile 95]  ← H₀ clustering of approaches
   ├─ If ALL BARS SHORT → REPROBE (return to step 1, ask more questions)
   ├─ Long bars → viable approaches (present as 2-3 options)
   └─ Output: module routing (selected approach)

3. /dw-ascend [--degree 1]           ← H₁ loops: design coherence
   ├─ Gini > +0.01 → ASCEND (proceed through design sections)
   ├─ Gini < -0.01 → REPROBE (design degrading, revisit)
   ├─ |Gini| < 0.01 → HOLD (stable, await new input)
   ├─ waypoints > 3 → SPLIT (decompose into sub-specs)
   └─ gini_watchdog monitors post each section

4. /dw-ascend [--degree 1 --sheaf su2]  ← L3: Sheaf-valued review
   ├─ Subagent review: check global consistency of spec sections
   ├─ Iterate until ker(L_F) convergence (max 3 iterations)
   └─ Surface to human if obstruction persists

5. Waypoint gate                     ← Phase transition: ideation → implementation
   ├─ W(I) ∈ W_phys?
   ├─ YES → Invoke writing-plans
   └─ NO → REPROBE to appropriate layer
```

---

## Connections to the Literature

The synthesis of structured brainstorming with topological methods connects to several research threads:

**Design space exploration**: [Suh et al. (2024)](https://dl.acm.org/doi/10.1145/3613904.3642400) propose Luminate, a system for structured generation and exploration of design spaces with LLMs. Their framework argues that current interaction paradigms "guide users towards rapid convergence on a limited set of ideas" — precisely the premature averaging that Axiom 1 forbids. The topological brainstorming framework provides formal machinery for maintaining divergence (high H₀ count at low \(\varepsilon\)) before convergent filtration.

**Divergent and convergent thinking**: [Wen et al. (2025)](https://arxiv.org/abs/2512.18388) explicitly scaffold brainstorming into divergent (exploration) and convergent (refinement) phases based on Wallas's model of creativity. The driftwave pipeline provides a mathematical formalization: divergence is the L0 ingestion phase (maximize point cloud spread), convergence is the L1 filtration (H₀ clustering reduces to viable approaches). The phase transition between them is a topological waypoint.

**TDA beyond persistent homology**: [Su et al. (2025)](https://link.springer.com/10.1007/s10462-025-11462-w) provide a comprehensive review of TDA methods beyond basic persistent homology, including persistent topological Laplacians, sheaf theory, and Mayer topology. The ATFT's sheaf-valued persistent homology directly extends these methods, and the brainstorming application demonstrates their applicability to non-physical domains.

**Categorical persistence**: [Odetallah et al. (2025)](https://etamaths.com/index.php/ijaa/article/view/4605) develop categorical foundations for persistent homology that establish functorial relationships between classical invariants and persistent counterparts. The brainstorming isomorphism proposed here — mapping procedural creative steps to functorial operations on idea-spaces — is a concrete instantiation of this categorical perspective.

**Semantic creativity metrics**: [Georgiev & Georgiev (2025)](https://arxiv.org/html/2501.11090v1) develop dynamic semantic networks for exploring creative thinking, using information-theoretic measures to predict idea success. The Gini trajectory from ATFT provides an alternative topological measure: positive Gini slope (hierarchification) predicts design quality just as it predicts LLM reasoning accuracy.

---

## Open Questions

1. **Computability of idea-space distances**: The formalization assumes a metric on idea-space. In practice, this requires embedding requirements, constraints, and design options into a vector space (e.g., via LLM embeddings). Does the choice of embedding preserve the topological features that matter for design quality? The ATFT's bottleneck stability theorem ([Cohen-Steiner et al., 2007](https://link.springer.com/article/10.1007/s00454-006-1276-5)) guarantees small perturbations in the embedding produce small perturbations in the persistence diagram — but "small" may not be small enough.

2. **Gini trajectory validation for creative tasks**: The Gini trajectory's predictive power is validated for LLM reasoning (\(r = 0.935\)) and SU(2) gauge theory. Does it transfer to design quality assessment? The hypothesis is that a brainstorming process with positive Gini trajectory (increasing hierarchification of design components) produces better implementations than one with negative trajectory — this is testable.

3. **The SPLIT routing threshold**: Driftwave triggers SPLIT when waypoints exceed 3. In brainstorming, this determines when a project should be decomposed into sub-specs. Is 3 the right threshold for creative work, or does the optimal split point depend on the domain?

4. **Sheaf structure for design review**: The L3 sheaf-valued review is the most abstract element of the synthesis. Making this concrete requires defining: what is the Lie algebra fiber over each design component? What are the restriction maps between components? The most natural candidate is the interface contract between modules — the types, protocols, and invariants that must be preserved when crossing component boundaries.

5. **Inverse problem**: Given a waypoint signature of a successful design, can we reconstruct the brainstorming process that produced it? This is the design-space analogue of the ATFT's inversion problem (Section 9, Open Question 1).

---

## Conclusion

The superpowers:brainstorming workflow is, at its core, a filtration process. It ingests a diffuse cloud of possibilities (L0), clusters them into viable approaches via persistent features (L1), synthesizes coherent designs by detecting and resolving topological loops (L2), validates global consistency through sheaf-valued review (L3), and transitions to implementation at a topological waypoint.

The ATFT provides the mathematical language to make this precise. The driftwave plugin provides the operative machinery to execute it. Together, they transform brainstorming from a procedural checklist into a dimensionalized process with topological invariants, adaptive quality gates, and principled routing decisions.

The deepest insight mirrors the ATFT's central claim: the quality of a brainstorming process is not determined by its state at any single phase, but by the trajectory of its topological evolution across all phases. Shape over count. Trajectory over snapshot. The wavefront of creative ideation, like the wavefront of physical field equations, is written in the native language of topology.
