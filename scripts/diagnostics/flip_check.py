#!/usr/bin/env python
"""Spectral-flip CPU solver vs known GPU values (flip-only, fast, unbuffered).

Run with `python -u`. Skips the broken shift-invert entirely.
Known GPU: S_zeta(K=100)=12.480, S_zeta(K=200)=11.784.
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

N, EPS, SIGMA, KEIG = 1000, 3.0, 0.5, 20


def smallest_via_flip(L, k):
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    dim = L.shape[0]
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=k, which="LA", tol=1e-4, return_eigenvectors=False)
    return np.sort(np.maximum((lam_max - mu).real, 0.0))


def main():
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    for K, known in ((100, 12.480), (200, 11.784)):
        builder = TransportMapBuilder(K=K, sigma=SIGMA)
        lap = SparseSheafLaplacian(builder, zeta, transport_mode="superposition")
        t0 = time.time(); L = lap.build_matrix(EPS); tb = time.time() - t0
        L = (L + L.getH()) * 0.5
        t0 = time.time(); e = smallest_via_flip(L, KEIG); ts = time.time() - t0
        print(f"K={K}: S_flip={e.sum():.4f}  known={known}  "
              f"diff={abs(e.sum()-known):.3f}  build={tb:.0f}s solve={ts:.0f}s",
              flush=True)
        print(f"   eigs={np.array2string(e, precision=3, max_line_width=140)}",
              flush=True)


if __name__ == "__main__":
    main()
