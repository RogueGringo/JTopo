#!/usr/bin/env python
"""Reproduce the published GPU spectral sum on the RTX 5070, then cross-check
the GPU's OWN matrix with a robust CPU solver.

Three numbers at K=100 (known published S_zeta=12.480):
  1. TorchSheafLaplacian.smallest_eigenvalues  -> the published pipeline value
  2. Pull the torch-built L to scipy, run ARPACK spectral-flip -> robust value
     on the IDENTICAL matrix the GPU built.
  3. Print both eigenvalue spectra.

If (1)=12.48 and (2)=0.01 on the same GPU-built matrix, the published premium
is a Krylov-truncation artifact of the 70-vector hand-Lanczos, full stop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N, EPS, SIGMA, KEIG = 1000, 3.0, 0.5, 20


def torch_csr_to_scipy(L_csr):
    """Convert a torch sparse CSR (complex128) to scipy CSR on CPU."""
    L = L_csr.cpu()
    crow = L.crow_indices().numpy()
    col = L.col_indices().numpy()
    val = L.values().numpy()
    dim = L.shape[0]
    return sp.csr_matrix((val, col, crow), shape=(dim, dim))


def arpack_flip(L, k):
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    dim = L.shape[0]
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=k, which="LA", tol=1e-6, return_eigenvectors=False)
    return np.sort(np.maximum((lam_max - mu).real, 0.0))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    print(f"K={K}, N={N}, eps={EPS}, sigma={SIGMA}, k_eig={KEIG}", flush=True)

    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]

    dev = sys.argv[2] if len(sys.argv) > 2 else None
    builder = TransportMapBuilder(K=K, sigma=SIGMA)
    lap = TorchSheafLaplacian(builder, zeta, transport_mode="superposition", device=dev)
    print(f"running literal TorchSheafLaplacian on device={lap.device}", flush=True)

    # 1. Published pipeline: GPU smallest_eigenvalues (70-vector hand-Lanczos)
    t0 = time.time()
    eigs_gpu = lap.smallest_eigenvalues(EPS, k=KEIG)
    print(f"(1) GPU TorchSheafLaplacian  S={eigs_gpu.sum():.4f}  "
          f"max_eig={eigs_gpu[-1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"    eigs={np.array2string(eigs_gpu, precision=3, max_line_width=130)}",
          flush=True)

    # 2. Same matrix, robust ARPACK
    t0 = time.time()
    L_csr = lap.build_matrix(EPS)
    L = torch_csr_to_scipy(L_csr)
    L = (L + L.getH()) * 0.5
    print(f"    pulled GPU matrix -> scipy: dim={L.shape[0]} nnz={L.nnz}", flush=True)
    eigs_cpu = arpack_flip(L, KEIG)
    print(f"(2) ARPACK on GPU's matrix   S={eigs_cpu.sum():.4f}  "
          f"max_eig={eigs_cpu[-1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"    eigs={np.array2string(eigs_cpu, precision=3, max_line_width=130)}",
          flush=True)

    verdict = ("ARTIFACT: GPU 70-vector Lanczos overestimates; true S~0"
               if eigs_gpu.sum() > 1.0 and eigs_cpu.sum() < 1.0
               else "matrices/solvers agree — investigate further")
    print(f"\nVERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
