#!/bin/bash
# RunPod A100 Setup Script for Ti V0.1
# ======================================
# One-liner deploy on a fresh RunPod instance:
#
#   curl -sL https://raw.githubusercontent.com/RogueGringo/Ti_V0.1/master/scripts/runpod_setup.sh | bash
#
# Or manually:
#   1. Start an A100 80GB pod (~$1.64/hr community), PyTorch template
#   2. SSH in:  ssh root@<POD_IP>
#   3. Run:     bash <(curl -sL https://raw.githubusercontent.com/RogueGringo/Ti_V0.1/master/scripts/runpod_setup.sh)
#
# Expected setup time: ~3 minutes

set -euo pipefail

echo "============================================"
echo "  Ti V0.1 — RunPod A100 Setup"
echo "============================================"

# 1. GPU check
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 2. Clone repo
echo "--- Cloning Ti V0.1 ---"
cd /workspace
if [ -d "Ti_V0.1" ]; then
    echo "  Repo already exists, pulling latest..."
    cd Ti_V0.1 && git pull
else
    git clone https://github.com/RogueGringo/Ti_V0.1.git
    cd Ti_V0.1
fi
echo ""

# 3. Install dependencies
echo "--- Installing dependencies ---"
pip install -e ".[dev]" 2>&1 | tail -5
pip install cupy-cuda12x 2>&1 | tail -3

# 4. Verify CuPy + GPU
echo ""
echo "--- CuPy Verification ---"
python -c "
import cupy as cp
dev = cp.cuda.Device(0)
mem = dev.mem_info
print(f'  CuPy OK: {mem[0]/1e9:.1f} GB free / {mem[1]/1e9:.1f} GB total')
a = cp.random.randn(1000, 1000, dtype=cp.float64)
b = a @ a.T
print(f'  Matmul test: {float(b[0,0]):.4f} (nonzero = working)')
"

# 5. Verify data
echo ""
echo "--- Data Verification ---"
python -c "
from atft.sources.zeta_zeros import ZetaZerosSource
source = ZetaZerosSource('data/odlyzko_zeros.txt')
cloud = source.generate(9877)
print(f'  Zeros loaded: N={len(cloud.points)}')
print(f'  Range: [{cloud.points[0,0]:.2f}, {cloud.points[-1,0]:.2f}]')
"

# 6. Smoke test — K=20, N=100, one point
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
print('  SMOKE TEST PASSED')
"

# 7. Create output dir
mkdir -p output

echo ""
echo "============================================"
echo "  Setup complete. Ready to run:"
echo ""
echo "  # K=100 zeta-only (budget-conscious, ~13h, ~\$21):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 --zeta-only \\"
echo "      2>&1 | tee output/k100_zeta_a100.log"
echo ""
echo "  # K=100 full sweep with controls (~39h, ~\$64):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k100 \\"
echo "      2>&1 | tee output/k100_sweep_a100.log"
echo ""
echo "  # K=200 scout (~10h, ~\$16):"
echo "  python -u -m atft.experiments.phase3_distributed \\"
echo "      --role gpu-k200 \\"
echo "      2>&1 | tee output/k200_scout_a100.log"
echo ""
echo "  # When done, download results:"
echo "  # scp root@<POD_IP>:/workspace/Ti_V0.1/output/*.json ."
echo "============================================"
