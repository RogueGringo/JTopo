# ATFT Phase 3 — Multi-Machine Deployment Guide

## Hardware Fleet

| Machine | GPU | VRAM | RAM | Role | Est. Time | Status |
|---------|-----|------|-----|------|-----------|--------|
| RTX 4080 Laptop | RTX 4080 Laptop | 12 GB | 16 GB | K=20 CPU sweep (current) | ~18h total | RUNNING |
| GTX 1060 Laptop | GTX 1060 | 3 GB | 16 GB | K=20 controls (CPU) | ~12h | READY |
| RTX 5070 Desktop | RTX 5070 | 16 GB | ? | K=50 full sweep (GPU) | ~8h | READY |
| RunPod A100 | A100 | 80 GB | 200 GB | K=100 + K=200 (GPU) | ~12h | READY |
| Qualcomm NPU | N/A | N/A | 16 GB | Aggregation + viz only | minutes | STANDBY |

## Compute Budget (RunPod)

A100 80GB pricing (community cloud): ~$1.64/hr

| Role | Grid Points | Est. per Point | Total Time | Cost |
|------|-------------|----------------|------------|------|
| gpu-k100 | 13σ × 2ε = 26 zeta + 26 rand + 26 GUE = 78 | ~30 min | ~39 hrs | ~$64 |
| gpu-k200 | 5σ × 2ε = 10 zeta only | ~60 min | ~10 hrs | ~$16 |
| **Sequential run** | | | **~49 hrs** | **~$80** |

**$50 budget strategy:** Run K=100 zeta-only first (26 points, ~13 hrs, ~$21). If results show peak sharpening, run K=100 controls (52 points, ~$42). Skip K=200 scout unless budget allows.

Revised ROLES config for $50 budget:
```python
# In phase3_distributed.py, modify gpu-k100 to skip controls:
"gpu-k100": {
    ...
    "n_random": 0,  # Skip controls to save budget
    "n_gue": 0,     # Run controls on cheaper hardware if needed
}
```

## Step-by-Step Deployment

### Machine 1: GTX 1060 Laptop (CPU controls)

This machine runs K=20 random + GUE controls, freeing the 4080 laptop.

```bash
# 1. Transfer project
scp atft_phase3.tar.gz user@1060-laptop:~/
ssh user@1060-laptop

# 2. Extract and install
cd ~ && tar xzf atft_phase3.tar.gz
cd atft && pip install -e ".[dev]"

# 3. Run K=20 controls (CPU only, ~12 hours)
mkdir -p output
python -u -m atft.experiments.phase3_distributed \
    --role control-cpu --trials 1,2,3,4,5 \
    2>&1 | tee output/k20_controls_1060.log

# 4. When done, grab the JSON
scp output/phase3_control-cpu_K20.json user@main-machine:~/atft/output/
```

### Machine 2: RTX 5070 Desktop (K=50 GPU)

```bash
# 1. Transfer
scp atft_phase3.tar.gz user@5070-desktop:~/
ssh user@5070-desktop

# 2. Extract, install, add CuPy
cd ~ && tar xzf atft_phase3.tar.gz
cd atft && pip install -e ".[dev]"
pip install cupy-cuda12x

# 3. Verify GPU
python -c "import cupy as cp; print(cp.cuda.Device(0).mem_info)"

# 4. Run K=50 full sweep (~8 hours)
mkdir -p output
python -u -m atft.experiments.phase3_distributed \
    --role gpu-k50 \
    2>&1 | tee output/k50_sweep_5070.log

# 5. Collect results
scp output/phase3_gpu-k50_K50.json user@main-machine:~/atft/output/
```

### Machine 3: RunPod A100 (K=100 definitive)

```bash
# 1. Create pod: A100 80GB, PyTorch 2.x template, ~$1.64/hr community
# 2. Connect via SSH or web terminal

# 3. Upload and setup
scp atft_phase3.tar.gz root@<POD_IP>:/workspace/
ssh root@<POD_IP>
cd /workspace && bash runpod_setup.sh

# 4. Run K=100 zeta-only first (budget-conscious, ~13 hrs, ~$21)
python -u -m atft.experiments.phase3_distributed \
    --role gpu-k100 --zeta-only \
    2>&1 | tee output/k100_sweep_a100.log

# 5. If budget allows and K=100 shows sharpening, run K=200 scout:
python -u -m atft.experiments.phase3_distributed \
    --role gpu-k200 \
    2>&1 | tee output/k200_scout_a100.log

# 6. Download results BEFORE terminating pod!
# From local machine:
scp root@<POD_IP>:/workspace/atft/output/phase3_*.json output/
```

### Machine 4: Qualcomm NPU Laptop (Analysis Hub)

The Qualcomm machine has no useful GPU for CUDA, but works great as the
aggregation and analysis station.

```bash
# 1. Install project (CPU only)
pip install -e ".[dev]"

# 2. Collect all JSONs into output/
# (scp from each machine, or use a shared drive)

# 3. Run aggregation
python scripts/aggregate_results.py output/phase3_*.json

# 4. Generate publication-quality plots (future)
# python scripts/plot_sharpening.py output/phase3_*.json
```

## Data Flow

```
GTX 1060 ─── K=20 controls ──────┐
                                   │
RTX 4080 ─── K=20 zeta (running) ─┤
                                   │──→ Qualcomm ──→ aggregate_results.py
RTX 5070 ─── K=50 full sweep ─────┤       │
                                   │       └──→ Cross-K sharpening table
RunPod A100 ─ K=100 + K=200 ──────┘              + contrast ratios
                                                  + peak σ detection
```

## What We're Looking For

The Fourier truncation hypothesis predicts:

| K | Primes | Expected Behavior |
|---|--------|-------------------|
| 20 | 8 | Monotonic rise through σ=0.50 (CONFIRMED) |
| 50 | 15 | Spectral turnover near σ=0.40-0.50 (CONFIRMED at N=2000) |
| 100 | 25 | Sharp peak localizing toward σ=0.50 |
| 200 | 46 | Phase transition — wall at σ=0.50 |

**Success criteria:** The cross-K summary table shows:
1. Peak σ converges toward 0.500 as K increases
2. Contrast ratio C increases with K
3. R(zeta/controls) >> 1 at all K values

## Timing Strategy

Run in this order to maximize information per dollar:

1. **Now:** Let K=20 CPU sweep finish on the 4080 (already running, free)
2. **Parallel:** Start K=20 controls on the 1060 (free compute)
3. **Parallel:** Start K=50 sweep on the 5070 (free compute)
4. **After K=50 results:** If turnover confirmed at N=9877, launch RunPod
5. **RunPod:** K=100 zeta-only first. Assess. Then K=100 controls or K=200.
