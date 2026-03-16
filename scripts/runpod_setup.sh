#!/bin/bash
# RunPod A100 Setup Script for ATFT Phase 3
# ==========================================
# Run this on a fresh RunPod instance (PyTorch template recommended).
#
# Usage:
#   # 1. Start an A100 80GB pod (~$1.64/hr community, ~$2.49/hr secure)
#   # 2. Upload this script + the project tarball:
#   scp atft_phase3.tar.gz scripts/runpod_setup.sh root@<POD_IP>:/workspace/
#   # 3. SSH in and run:
#   ssh root@<POD_IP>
#   cd /workspace && bash runpod_setup.sh
#
# Expected setup time: ~3 minutes

set -euo pipefail

echo "============================================"
echo "  ATFT Phase 3 — RunPod A100 Setup"
echo "============================================"

# 1. Check GPU
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 2. Extract project
if [ -f /workspace/atft_phase3.tar.gz ]; then
    echo "--- Extracting project ---"
    cd /workspace
    tar xzf atft_phase3.tar.gz
    cd atft
else
    echo "ERROR: /workspace/atft_phase3.tar.gz not found."
    echo "Create it locally with:"
    echo "  cd C:/Claude/Reimann_Hypothesis"
    echo "  tar czf atft_phase3.tar.gz atft/ data/ pyproject.toml scripts/"
    exit 1
fi

# 3. Install dependencies
echo "--- Installing dependencies ---"
pip install -e ".[dev]" 2>&1 | tail -5
pip install cupy-cuda12x 2>&1 | tail -3

# 4. Verify CuPy sees the GPU
echo ""
echo "--- CuPy Verification ---"
python -c "
import cupy as cp
dev = cp.cuda.Device(0)
mem = dev.mem_info
print(f'  CuPy OK: {dev.name}, {mem[0]/1e9:.1f} GB free / {mem[1]/1e9:.1f} GB total')
# Quick matmul test
a = cp.random.randn(1000, 1000, dtype=cp.float64)
b = a @ a.T
print(f'  Matmul test: {float(b[0,0]):.4f} (nonzero = working)')
"

# 5. Verify zeta zeros load
echo ""
echo "--- Data Verification ---"
python -c "
from atft.sources.zeta_zeros import ZetaZerosSource
source = ZetaZerosSource('data/odlyzko_zeros.txt')
cloud = source.generate(9877)
print(f'  Zeros loaded: N={len(cloud.points)}')
print(f'  Range: [{cloud.points[0,0]:.2f}, {cloud.points[-1,0]:.2f}]')
"

# 6. Quick smoke test — K=20, N=100, single sigma/epsilon
echo ""
echo "--- Smoke Test (K=20, N=100) ---"
python -c "
import numpy as np
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.gpu_sheaf_laplacian import GPUSheafLaplacian

source = ZetaZerosSource('data/odlyzko_zeros.txt')
cloud = source.generate(100)
zeros = SpectralUnfolding(method='zeta').transform(cloud).points[:, 0]

builder = TransportMapBuilder(K=20, sigma=0.50)
lap = GPUSheafLaplacian(builder, zeros, transport_mode='superposition')
eigs = lap.smallest_eigenvalues(3.0, k=5)
print(f'  GPU eigenvalues: {eigs[:5]}')
print(f'  Spectral sum: {np.sum(eigs):.6f}')
print()
print('  SMOKE TEST PASSED — Ready for production sweep!')
"

echo ""
echo "============================================"
echo "  Setup complete. Run sweeps with:"
echo ""
echo "  # K=100 definitive sweep (~4-8 hours):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 2>&1 | tee output/k100_sweep_a100.log"
echo ""
echo "  # K=200 scout (~2-4 hours):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k200 2>&1 | tee output/k200_scout_a100.log"
echo ""
echo "  # Both sequentially:"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 2>&1 | tee output/k100_sweep_a100.log && \\"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k200 2>&1 | tee output/k200_scout_a100.log"
echo "============================================"
