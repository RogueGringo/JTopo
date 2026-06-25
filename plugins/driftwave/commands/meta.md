---
description: "Compute meta-persistence across accumulated sessions. The barcode of barcodes — what ideas keep coming back?"
---

# /driftwave:meta

Run meta-persistence analysis across session history.

## What to do

1. Check if `/tmp/dw-artifacts/meta.json` exists
   - If not, create it with the current session's artifacts as the first entry
   - If yes, append the current session's artifacts as a new entry

2. Run meta-persistence computation:
   ```bash
   cat /tmp/dw-artifacts/meta.json | ${CLAUDE_PLUGIN_ROOT}/scripts/compute_meta_persistence.py > /tmp/dw-artifacts/meta_updated.json
   mv /tmp/dw-artifacts/meta_updated.json /tmp/dw-artifacts/meta.json
   ```

3. Validate the result: confirm `/tmp/dw-artifacts/meta.json` conforms to
   `${CLAUDE_PLUGIN_ROOT}/schemas/meta_persistence.json` (required keys present,
   types correct) before reporting.

4. Report the convergence signature:
   - **Dominant clusters**: ideas/patterns appearing in >50% of sessions
   - **Gini meta-trajectory**: is the project hierarchifying over time?
   - **Sheaf consistency rate**: fraction of sessions where L3 said ON_SHELL
   - **Meta-barcode**: which features have the longest bars across sessions?

## What this tells you

- Long bars in the meta-barcode = architectural decisions that keep proving right
- Short bars = ideas that were tried once and didn't persist
- Rising Gini meta-trajectory = the project is converging on its structure
- High sheaf consistency rate = the architecture is stable
