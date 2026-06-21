#!/usr/bin/env python
"""Lowest-level solver test: against EXACT dense ground truth.

At small dim, np.linalg.eigvalsh IS the truth — no iterative solver can claim
otherwise. We build the real sheaf Laplacian at a few small (N,K), then compare
the sum of the 20 smallest eigenvalues from:

  DENSE   : np.linalg.eigvalsh(L.toarray())        -> exact ground truth
  ARPACK  : spectral-flip eigsh(which='LA')         -> the converged solver
  HAND-70 : faithful port of the GPU 70-vector Lanczos (_lanczos_largest)

This pins the convergence bug at unit scale: when the near-kernel cluster grows
past what 70 Krylov vectors resolve, HAND-70 over-estimates while DENSE/ARPACK
agree. Scale-sweep shows exactly where it breaks.
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

EPS, SIGMA, KEIG = 3.0, 0.5, 20


def hand_lanczos_smallest(L, k):
    """Faithful port of torch _lanczos_largest spectral-flip: m=70 Krylov, k=20."""
    dim = L.shape[0]
    matvec = lambda v: L @ v
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05

    def lanczos_largest(mv, kk):
        m = min(max(2 * kk + 20, kk + 50), dim)   # = 70 for kk=20
        rng = np.random.default_rng(42)
        v = rng.standard_normal(dim).astype(np.complex128); v /= np.linalg.norm(v)
        V = np.zeros((m + 1, dim), dtype=np.complex128)
        al = np.zeros(m); be = np.zeros(m); V[0] = v
        for j in range(m):
            w = mv(V[j])
            if j > 0:
                w = w - be[j - 1] * V[j - 1]
            a = np.real(np.vdot(V[j], w)); al[j] = a; w = w - a * V[j]
            for _ in range(2):
                w = w - V[:j + 1].T @ (V[:j + 1].conj() @ w)
            b = np.linalg.norm(w).real
            if b < 1e-14:
                m = j + 1; al = al[:m]; be = be[:m]; break
            be[j] = b
            if j + 1 < m:
                V[j + 1] = w / b
        T = np.diag(al[:m])
        if m > 1:
            T += np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        return np.sort(np.linalg.eigvalsh(T).real)[-kk:][::-1]

    mu = lanczos_largest(lambda v: lam_max * v - matvec(v), k)
    return np.sort(np.maximum((lam_max - mu).real, 0.0))


def arpack_smallest(L, k):
    dim = L.shape[0]
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=k, which="LA", tol=1e-9, return_eigenvectors=False)
    return np.sort(np.maximum((lam_max - mu).real, 0.0))


def main():
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    print(f"{'N':>4} {'K':>3} {'dim':>6} {'kernel<1e-6':>11} "
          f"{'DENSE':>9} {'ARPACK':>9} {'HAND-70':>9} {'HAND/DENSE':>10}")
    for (Nn, K) in [(150, 20), (200, 20), (300, 20), (200, 25), (400, 20)]:
        zeta = SpectralUnfolding(method="zeta").transform(src.generate(Nn)).points[:, 0]
        builder = TransportMapBuilder(K=K, sigma=SIGMA)
        lap = SparseSheafLaplacian(builder, zeta, transport_mode="superposition")
        L = lap.build_matrix(EPS); L = (L + L.getH()) * 0.5
        dim = L.shape[0]
        t0 = time.time()
        dense_all = np.sort(np.linalg.eigvalsh(L.toarray()).real)
        dense = float(np.maximum(dense_all[:KEIG], 0).sum())
        kern = int(np.sum(dense_all < 1e-6))
        arp = float(arpack_smallest(L, KEIG).sum())
        hand = float(hand_lanczos_smallest(L, KEIG).sum())
        ratio = hand / dense if dense > 1e-12 else float("inf")
        print(f"{Nn:>4} {K:>3} {dim:>6} {kern:>11} {dense:>9.4f} {arp:>9.4f} "
              f"{hand:>9.4f} {ratio:>9.1f}x   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
