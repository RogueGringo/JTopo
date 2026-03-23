# driftwave

You know that thing where you start coding, get halfway through, realize you missed something, undo half your work, start over, and end up worse than where you began?

Driftwave is the habit that stops that from happening.

## What It Does

Before you touch anything, it looks. Three questions, answered in parallel:

| Question | What It Checks |
|----------|---------------|
| **What's there?** | Your files, git state, what's running, what changed recently |
| **What's asked?** | What you actually want — not what you said, what you *meant* |
| **What's the gap?** | The specific actions to get from here to there |

Then it shows you the gap. You say yes. It does the work. It writes down what happened. It checks if it worked.

That's it. That's the whole thing.

```
/driftwave:directive add user authentication
```

```
WHAT IS:     Express app, 12 routes, no auth middleware, users table exists
WHAT'S ASKED: Add login/logout with session management
THE GAP:
  1. Add bcrypt + express-session deps
  2. Create auth middleware
  3. Add login/logout routes
  4. Protect existing routes
  5. Test login flow

Proceed? (y)
```

## Why It Works

Not because it's smart. Because it **looks before it acts.**

Most mistakes aren't intelligence failures. They're attention failures. You didn't check what was already there. You didn't scope what you were actually trying to do. You started coding before you understood the gap.

Driftwave forces the pause. Three subagents check three things simultaneously — takes seconds — and now you have a map instead of a guess.

## The Loop

Every action follows the same cycle:

```
  LOOK ──→ PARSE ──→ GAP ──→ DO ──→ LOG ──→ CHECK
    ↑                                          │
    └──────────── next cycle reads the log ────┘
```

The log is the key. Without it, each cycle starts from scratch. With it, each round knows what the last round did, what worked, what didn't. Mistakes don't repeat because the log remembers them.

Over time, patterns emerge. Ideas that keep coming back session after session — those are real. Ideas that appeared once and disappeared — that was noise. The log tells you which is which.

## Commands

**The basics:**

| Command | Plain English |
|---------|--------------|
| `/driftwave:directive` | "Here's what I want. Figure out the gap and do it." |
| `/driftwave:status` | "What's going on right now?" |

**When things get complex:**

| Command | Plain English |
|---------|--------------|
| `/driftwave:run` | "This is big. Run the full analysis pipeline." |
| `/driftwave:dashboard` | "Show me the topology visually." |
| `/driftwave:meta` | "What patterns keep showing up across sessions?" |

**Manual layer control (power users):**

| Command | What Layer |
|---------|-----------|
| `/driftwave:ingest` | Scan (what's there) |
| `/driftwave:filter` | Cluster (what's related) |
| `/driftwave:synthesize` | Design (what to build) |
| `/driftwave:review` | Check (does it all fit together) |

## Install

```bash
claude plugin add gh:RogueGringo/driftwave-v0
```

That's it. Next time you open Claude Code:

```
/driftwave:directive <what you want>
```

## The Deeper Layer

Under the hood, driftwave uses the same mathematics that finds structure in prime numbers. Not metaphorically — literally. The clustering algorithm is persistent homology. The quality check is sheaf consistency. The log analysis is a persistence barcode.

You don't need to know any of that. The commands work whether you understand the math or not. But if you're curious:

- [VISION.md](VISION.md) — the geometric theory of why this works
- [PROTOCOL.md](PROTOCOL.md) — the full engineering spec

The short version: when you look at what's there, parse what's asked, and compute the gap — you're doing topology on your project. You're finding the persistent structure (what's real) and filtering the noise (what's not). The math just makes it precise.

## Five Habits

Driftwave has five principles. They sound fancy but they're common sense:

1. **Don't summarize too early.** Get the facts first. Details matter.
2. **Look before you act.** Scan before you plan. Plan before you build.
3. **Stop when something's wrong.** Don't push through a failing check.
4. **Focus on what matters most.** Not what's easiest. What's most important.
5. **Match your effort to the problem.** Small ask, small scan. Big ask, full pipeline.

## What This Is Really About

The gap between "AI can do this" and "AI reliably does this" isn't intelligence. It's process.

Today's AI models can reason, write code, analyze data, design systems. What they can't do — without help — is consistently look before they act, log what they did, and verify it worked.

Driftwave is that help. Not new AI. New habits for existing AI.

The tools are ready. The models are ready. The process was missing. Now it's not.

---

*Built by Aaron Jones. Powered by algebraic topology and the habit of looking before you leap.*

*All work is non-published private minus the public repos associated with company and name. All rights reserved.*
