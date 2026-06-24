---
description: "The core process. Three subagents: WHAT IS, WHAT'S ASKED, THE GAP. Then route, execute, log, verify."
arguments: "<request>"
---

# /driftwave:directive

The fundamental operation. Everything else in driftwave is a specialization of this.

## Process

You receive a request. You do this:

### Step 1: Launch three subagents IN PARALLEL

**Subagent A — WHAT IS** (Explore agent, quick)
```
Scan the current state relevant to this request:
- What files exist that this touches?
- What's the git state? Any uncommitted work?
- What docs/specs/plans exist?
- What's running? (GPU processes, servers, background tasks)
- What did we do last? (recent commits, memory files)
Report facts only. No interpretation. No recommendations.
```

**Subagent B — WHAT'S ASKED** (haiku, fast)
```
Parse this request into:
- INTENT: What does the user want to happen? (one sentence)
- SCOPE: What files/systems does this touch?
- SUCCESS: How will we know it worked? (concrete check)
- CONSTRAINTS: What must NOT change?
```

**Subagent C — THE GAP** (sonnet, after A and B return)
```
Given WHAT IS and WHAT'S ASKED:
- What specifically needs to change?
- What's the smallest set of actions to close the gap?
- What order? What depends on what?
- What could go wrong?
Return a numbered action list. Nothing else.
```

### Step 2: Present the gap to the user

Show them:
```
WHAT IS:     [2-3 line summary from Subagent A]
WHAT'S ASKED: [intent + success criteria from Subagent B]
THE GAP:     [numbered action list from Subagent C]

Proceed? (y/n/adjust)
```

### Step 3: Execute

For each action in the gap list:
1. Do the action (write code, edit file, run command)
2. Log what happened (one line, append to `/tmp/dw-artifacts/directive.log`)
3. Check if it worked (run the success criteria from Step 1B)
4. If it failed → stop, report, ask

### Step 4: Verify

Run the SUCCESS check from Subagent B. Report:
```
✓ Done. [what changed]
  Verified: [how we know it worked]
  Logged: /tmp/dw-artifacts/directive.log
```

## That's it.

No schemas required. No typed artifacts required. No L0/L1/L2/L3 layer formalism required. Just:

1. **Look** at what's there
2. **Parse** what's asked
3. **Compute** the gap
4. **Execute** the actions
5. **Log** what happened
6. **Verify** it worked

The driftwave axioms emerge naturally from this process:
- NO_AVERAGING: Subagent A reports facts, not summaries
- UPWARD_FLOW: You look before you act (A before C, C before execute)
- WAYPOINT_ROUTING: You stop if verification fails (route on reality, not assumption)
- SHAPE_OVER_COUNT: The gap list is ordered by what matters, not by what's easy
- ADAPTIVE_SCALE: The scope of scanning matches the scope of the request

## When to use the full pipeline instead

If the directive reveals complexity:
- Gap list has >5 actions → consider `/driftwave:run` (full L0→L3 with persistence)
- Multiple independent workstreams → consider parallel subagent dispatch
- Design decisions needed → consider the `topological-brainstorm` skill
- Need to check cross-session patterns → consider `/driftwave:meta`

The directive is the atomic unit. Everything else is the directive applied at scale.
