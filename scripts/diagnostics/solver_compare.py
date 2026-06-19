#!/usr/bin/env python
"""Decisive solver comparison at K=100.

Same matrix L (built once via the sparse backend), three eigensolvers:
  A. shift-invert eigsh(sigma=1e-8, which='LM')  -- the current sparse method
  B. spectral-flip eigsh on (lam_max*I - L), which='LA'  -- the GPU strategy
  C. dense eigvalsh on a SMALL K (sanity ground truth at K where dense fits)

Known GPU value: S_zeta(K=100) = 12.480, S_zeta(K=200) = 11.784.
If B reproduces ~12.48 and A gives ~0, the sparse shift-invert is the bug.
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

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N = 1000
EPS = 3.0
SIGMA = 0.5
KEIG = 20


def smallest_via_shift_invert(L, k):
    eigs, _ = eigsh(L, k=k, sigma=1e-8, which="LM", tol=1e-6)
    return np.sort(np.maximum(eigs.real, 0.0))


def smallest_via_flip(L, k):
    # lam_max via Lanczos (largest algebraic), no factorization
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    dim = L.shape[0]
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=k, which="LA", tol=1e-4, return_eigenvectors=False)
    eigs = np.sort(np.maximum((lam_max - mu).real, 0.0))
    return eigs


def main():
    source = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(source.generate(N)).points[:, 0]

    for K, known in ((100, 12.480), (200, 11.784)):
        print("=" * 60)
        print(f"  K={K}  (known GPU S_zeta = {known})")
        builder = TransportMapBuilder(K=K, sigma=SIGMA)
        lap = SparseSheafLaplacian(builder, zeta, transport_mode="superposition")
        t0 = time.time()
        L = lap.build_matrix(EPS)
        L = (L + L.getH()) * 0.5  # enforce exact Hermitian for ARPACK
        print(f"  built L: dim={L.shape[0]}, nnz={L.nnz}, build={time.time()-t0:.1f}s")

        t0 = time.time()
        try:
            eA = smallest_via_shift_invert(L, KEIG)
            print(f"  A shift-invert: S={eA.sum():.6f}  "
                  f"min={eA[0]:.2e} max={eA[-1]:.2e}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  A shift-invert FAILED: {e}")

        t0 = time.time()
        try:
            eB = smallest_via_flip(L, KEIG)
            n_kernel = int(np.sum(eB < 1e-6 * np.sqrt(max(eB.sum(), 1e-30))))
            print(f"  B spectral-flip: S={eB.sum():.6f}  "
                  f"min={eB[0]:.2e} max={eB[-1]:.2e}  kernel~{n_kernel}  "
                  f"({time.time()-t0:.1f}s)")
            print(f"  B eigs: {np.array2string(eB, precision=4, max_line_width=120)}")
        except Exception as e:
            print(f"  B spectral-flip FAILED: {e}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
