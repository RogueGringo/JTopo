# Distributed Fire-and-Forget KPM Engine — Design Spec

**Date:** 2026-03-18
**Status:** Approved
**Scope:** `atft/distributed/` module + `synthesizer/` and `node/` packages

## Problem

The KPM pipeline is bottlenecked by single-machine memory limits. At N=9877, K=100, the COO-to-CSR conversion of a 987,700 x 987,700 matrix causes OOM on 12GB GPUs. The project needs to distribute the computation across multiple machines while maintaining mathematical exactness of the Chebyshev moments.

## Solution

A fire-and-forget distributed architecture using static ghost zones, SSH dispatch, and raw probe trace payloads. The Primary (synthesizer) partitions the graph, dispatches work via SSH CLI, and collects results via SCP. Workers run independently with no real-time synchronization. The boundary error from static ghosts (O(h/N) ~ 0.2%) is negligible compared to Hutchinson variance (~10%).

## Design

### 1. Package Structure

Two installable packages inside the JTopo repo:

```
JTopo/
├── atft/                          # existing science code
│   └── distributed/               # NEW — shared protocol definitions
│       ├── __init__.py
│       └── protocol.py            # PartitionSpec, ContributeResult dataclasses
├── synthesizer/                   # NEW — Primary coordinator package
│   ├── pyproject.toml
│   └── src/
│       └── jtopo_synthesizer/
│           ├── __init__.py
│           ├── scanner.py         # DeviceScanner — reads nodes.yaml, SSH probes
│           ├── benchmarker.py     # mini-bench on each node, throughput model
│           ├── planner.py         # JobPlanner — pre-flight gate, VRAM clamp
│           ├── dispatcher.py      # SSH dispatch + SCP result retrieval
│           ├── validator.py       # ResultValidator — moment QC
│           ├── merger.py          # dimension-weighted trace merge
│           └── log.py             # ContributionLog — append-only provenance
├── node/                          # NEW — Worker package
│   ├── pyproject.toml
│   └── src/
│       └── jtopo_node/
│           ├── __init__.py
│           ├── worker.py          # CLI entry point for ephemeral worker
│           ├── compute.py         # KPM moment computation kernel (self-contained)
│           └── auth.py            # password validation on install/connect
```

Install:
- Primary: `pip install -e ./synthesizer`
- Worker: `pip install git+https://<PAT>@github.com/RogueGringo/JTopo.git#subdirectory=node`

The `node` package contains a self-contained KPM compute kernel with zero dependency on `atft/`. The `synthesizer` depends on `atft/` for graph partitioning and result assembly. They communicate via JSON/NPZ files, not Python imports.

### 2. Job Protocol

**PartitionSpec (Synthesizer → Node via CLI args):**

Zero-payload dispatch. The worker has the zeros dataset locally. The Synthesizer passes only indices and parameters:

```bash
ssh user@host "cd ~/JTopo && \
  python -m jtopo_node.worker \
    --start-idx 5000 --end-idx 10000 \
    --ghost-left 12 --ghost-right 0 \
    --K 100 --sigma 0.50 --epsilon 3.0 \
    --degree 300 --num-vectors 100 \
    --zeros-path data/odlyzko_zeros.txt \
    --output /tmp/jtopo_result.npz"
```

**ContributeResult (Node → Synthesizer via SCP):**

Saved as `.npz` file containing:

```python
{
    "worker_id": str,
    "partition": (start_idx, end_idx),   # owned range (excludes ghosts)
    "dim_local": int,                     # owned vertices * K
    "lam_max_local": float,
    "raw_traces": float64[D+1, nv],      # per-probe Hutchinson traces (NOT averaged)
    "device_type": str,
    "compute_time_s": float,
    "checksum": str,                      # SHA-256 of raw_traces
}
```

The raw traces preserve full statistical information. The Primary computes moments AND variance from them.

### 3. Synthesizer Pipeline

Six components executed in sequence:

**DeviceScanner** — Reads `~/.jtopo/nodes.yaml`:
```yaml
nodes:
  - id: local-4080
    host: localhost
    device: cuda
    vram_mb: 12000
  - id: ubuntu-5070
    host: 192.168.1.50
    ssh_user: blake
    ssh_key: ~/.ssh/id_rsa
    device: cuda
    vram_mb: 12000
```
For remote nodes, SSHs in and verifies `jtopo-node` is installed and GPU accessible.

**Benchmarker** — Dispatches a tiny job (N=200, K=target, D=50) to each node. Measures `steps_per_second`. Builds throughput model for runtime prediction (+-15% at 90% confidence after calibration).

**JobPlanner** — Pre-flight validation gate:
- Estimates total time per node from benchmark data
- Estimates VRAM per node
- Computes optimal partition split proportional to throughput
- VRAM clamping: enforces 85% hard ceiling (12GB * 0.85 = 10.2GB usable)
- If throughput-optimal split pushes either node over VRAM limit, clamps toward 50/50
- Time limit enforcement: <1hr auto, 1-8hr user confirmation, >24hr rejected with optimization suggestions
- Prints pre-flight summary before committing

**Dispatcher** — For each node:
- Local: runs KPM directly in-process, stores raw per-probe traces (shape [D+1, nv])
- Remote: SSH-launches `jtopo_node.worker` with CLI args, waits for exit, SCPs result `.npz` back
- Both local and remote produce identical `.npz` result files

**ResultValidator** — Checks each ContributeResult:
- mu_0 = mean(raw_traces[0, :]) approximately equals 1.0 (within Hutchinson variance)
- All moments are real (dtype check)
- Moments decay: |mu_D| < |mu_0|
- SHA-256 checksum matches
- If validation fails: log failure, optionally re-dispatch

**ContributionLog** — Append-only record at `output/contributions/`:
```
output/contributions/
├── 2026-03-18-kpm-001-local-4080.npz
├── 2026-03-18-kpm-001-ubuntu-5070.npz
└── 2026-03-18-kpm-001-merged.json
```

### 4. Merge Math

**Dimension-weighted merge for asymmetric partitions:**

Each worker normalizes its traces by `dim_local`. The Primary reconstructs global traces via:

```python
traces_global = (dim_A * traces_A + dim_B * traces_B) / dim_global
mu_global = traces_global.mean(axis=1)      # shape [D+1]
mu_variance = traces_global.var(axis=1)     # shape [D+1] — proof-grade statistics
```

This correctly handles asymmetric partitions from throughput-based load balancing and ghost zone overlap.

**Ghost zone computation:**

Given sorted zeros and partition boundary at index `split`:
- Rank 0 right ghosts: zeros[split : split + h] where h = count of zeros within epsilon of zeros[split-1]
- Rank 1 left ghosts: zeros[split - h : split] where h = count of zeros within epsilon of zeros[split]
- Ghost vertices participate in L_csr assembly but are EXCLUDED from Hutchinson traces
- This ensures each eigenvalue is counted exactly once across all partitions

**Worker KPM loop (modified from KPMSheafLaplacian):**

```python
# In compute.py — the self-contained worker kernel
# Key difference from single-node: stores per-probe traces, not averages

raw_traces = np.empty((D + 1, num_vectors), dtype=np.float64)

for k in range(D + 1):
    # Hutchinson trace per probe vector (DO NOT average)
    per_vec = torch.real(torch.sum(Z_owned.conj() * T_k_owned, dim=0))
    raw_traces[k] = (per_vec.cpu().numpy()) / dim_owned
    # ... Chebyshev recurrence continues
```

`Z_owned` and `T_k_owned` exclude ghost rows — only owned vertices contribute to traces.

### 5. Worker CLI Interface

The `jtopo_node.worker` module is a standalone CLI entry point:

```
python -m jtopo_node.worker \
  --start-idx START --end-idx END \
  --ghost-left H_LEFT --ghost-right H_RIGHT \
  --K K --sigma SIGMA --epsilon EPS \
  --degree D --num-vectors NV \
  --zeros-path PATH \
  --output OUTPUT.npz
```

Execution:
1. Load zeros from local `--zeros-path`
2. Slice: owned = zeros[start:end], with ghost extensions
3. Build TransportMapBuilder(K=K, sigma=sigma)
4. Build local L_csr from owned + ghost zeros
5. Run Chebyshev recurrence, store raw traces (excluding ghost contributions)
6. Save ContributeResult as `.npz`
7. Exit

## Files Created

- `atft/distributed/__init__.py`
- `atft/distributed/protocol.py` — PartitionSpec, ContributeResult dataclasses
- `synthesizer/pyproject.toml`
- `synthesizer/src/jtopo_synthesizer/__init__.py`
- `synthesizer/src/jtopo_synthesizer/scanner.py`
- `synthesizer/src/jtopo_synthesizer/benchmarker.py`
- `synthesizer/src/jtopo_synthesizer/planner.py`
- `synthesizer/src/jtopo_synthesizer/dispatcher.py`
- `synthesizer/src/jtopo_synthesizer/validator.py`
- `synthesizer/src/jtopo_synthesizer/merger.py`
- `synthesizer/src/jtopo_synthesizer/log.py`
- `node/pyproject.toml`
- `node/src/jtopo_node/__init__.py`
- `node/src/jtopo_node/worker.py`
- `node/src/jtopo_node/compute.py`
- `node/src/jtopo_node/auth.py`
- `tests/test_distributed.py`

## Files Modified

- None. The distributed module is entirely additive.

## Files NOT Modified

- `atft/topology/kpm_sheaf_laplacian.py` — single-node KPM unchanged
- `atft/topology/torch_sheaf_laplacian.py` — parent class unchanged
- `docs/FALSIFICATION.md` — frozen
- `docs/FALSIFICATION_IDOS.md` — frozen

## Testing

**Unit tests (in `tests/test_distributed.py`):**
- Ghost zone computation: verify correct halo depth for known zero configurations
- Dimension-weighted merge: verify mu_global = weighted sum of mu_A, mu_B
- Variance preservation: verify merged variance matches concatenated probe statistics
- VRAM clamping: verify planner clamps partition when estimate exceeds limit
- ResultValidator: verify acceptance of valid results, rejection of corrupted results

**Integration test:**
- Run a local-only 2-partition test (both partitions on localhost) on a small problem (N=200, K=6)
- Verify merged moments match single-node KPMSheafLaplacian moments within Hutchinson variance

**End-to-end (requires 2 machines):**
- Dispatch real job to Ubuntu server
- Verify round-trip: dispatch → compute → SCP → validate → merge → log

## Risks

- **SSH reliability:** Long-running SSH sessions may drop. Mitigated: worker saves results to disk; Primary can re-SCP without re-computing.
- **Clock skew:** Compute time measurements may differ between machines. Mitigated: each worker measures its own wall-clock; Primary logs both.
- **Static ghost error:** O(h/N) ~ 0.2% boundary perturbation. Mitigated: negligible vs 10% Hutchinson variance. Can be measured by comparing distributed vs single-node results on small problems.
- **Zeros dataset version mismatch:** Worker and Primary must have identical `odlyzko_zeros.txt`. Mitigated: checksum verification at dispatch time.
