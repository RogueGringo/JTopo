# Changelog

All notable changes to the **driftwave** plugin are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-24

First packaged, installable release.

### Added
- Plugin manifest metadata: `version`, `license`, `homepage`, `repository`, `keywords`.
- `$id` identifiers on all artifact schemas (L0–L3 + meta-persistence).
- `CHANGELOG.md`, `.coderabbit.yaml`, and `.gitattributes` (LF for shell scripts).
- Marketplace + standalone install instructions in `README.md`.

### Fixed
- **Portability:** Python scripts now use `#!/usr/bin/env python3` instead of a
  hardcoded developer virtualenv path (`/home/wb1/.../.venv/bin/python`).
- **Portability:** `topo.sh` derives the Claude memory directory from the project
  path rather than a hardcoded machine-specific slug; override with `DW_MEMORY_DIR`.
- **Portability:** shell scripts normalized to LF so the SessionStart hook runs on
  Linux/macOS (CRLF previously broke the shebang).
- `gini-watchdog` agent given valid YAML frontmatter (`name`, `description`,
  `model`, `tools`).
- Corrected install instructions — the previous `gh:RogueGringo/driftwave-v0`
  reference did not resolve to this plugin.
- `run.md` / `directive.md` refer to `wavefront` and `topological-brainstorm` as
  skills rather than non-existent commands.
- `meta.md` validates its output against `meta_persistence.json` before reporting.
- Aligned author email between plugin and marketplace manifests.

[0.1.0]: https://github.com/RogueGringo/JTopo/releases/tag/driftwave-v0.1.0
