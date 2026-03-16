#!/bin/bash
# Pack ATFT project for transfer to remote machines.
#
# Creates a minimal tarball with only what's needed for Phase 3 sweeps.
# Output: atft_phase3.tar.gz (~2MB)
#
# Usage:
#   bash scripts/pack_for_remote.sh
#   scp atft_phase3.tar.gz root@<RUNPOD_IP>:/workspace/
#   # or for 1060/5070 machines:
#   scp atft_phase3.tar.gz user@<IP>:~/atft/

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Packing ATFT Phase 3 from: $PROJECT_ROOT"

tar czf atft_phase3.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='output/*.log' \
    --exclude='output/*.png' \
    --exclude='*.egg-info' \
    atft/ \
    data/odlyzko_zeros.txt \
    pyproject.toml \
    scripts/ \
    tests/

SIZE=$(du -h atft_phase3.tar.gz | cut -f1)
echo "Created: atft_phase3.tar.gz ($SIZE)"
echo ""
echo "Transfer commands:"
echo "  RunPod:  scp atft_phase3.tar.gz root@<POD_IP>:/workspace/"
echo "  Desktop: scp atft_phase3.tar.gz user@<5070_IP>:~/atft/"
echo "  Laptop:  scp atft_phase3.tar.gz user@<1060_IP>:~/atft/"
