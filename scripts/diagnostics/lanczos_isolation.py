#!/usr/bin/env python
"""Decisive isolation: same matrix L, three eigensolvers.

Settles whether the published S_zeta(K=100)=12.480 is a real eigenvalue sum or
a non-converged hand-Lanczos artifact. We build the SAME sparse L once, then:

  (a) ARPACK spectral-flip  (robust, IRAM restarts)            -> reference
  (b) PORTED hand-Lanczos   (faithful copy of the GPU's _lanczos_largest:
      70-vector single-shot Krylov, k=20, full reorth)         -> mimics GPU
  (c) ARPACK k=60 + large ncv  (over-resolve the near-kernel)  -> cross-check

If (b) ~ 12 while (a)/(c) ~ 0, the published premium is a Krylov-truncation
artifact of the 70-vector Lanczos, not a property of the operator.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, aslinearoperator

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N, EPS, SIGMA, KEIG = 1000, 3.0, 0.5, 20


def lanczos_largest_numpy(matvec, dim, k, tol=1e-4):
    """Faithful numpy port of torch_sheaf_laplacian._lanczos_largest.

    m = min(max(2k+20, k+50), dim) Krylov vectors, full reorthogonalization,
    return k largest Ritz values. This is EXACTLY the GPU algorithm.
    """
    m = min(max(2 * k + 20, k + 50), dim)
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.complex128)
    v /= np.linalg.norm(v)
    V = np.zeros((m + 1, dim), dtype=np.complex128)
    alpha = np.zeros(m); beta = np.zeros(m)
    V[0] = v
    for j in range(m):
        w = matvec(V[j])
        if j > 0:
            w = w - beta[j - 1] * V[j - 1]
        a_j = np.real(np.vdot(V[j], w)); alpha[j] = a_j
        w = w - a_j * V[j]
        for _ in range(2):  # full reorth, 2 passes
            coeffs = V[:j + 1].conj() @ w
            w = w - V[:j + 1].T @ coeffs
        b_j = np.linalg.norm(w).real
        if b_j < 1e-14:
            m = j + 1; alpha = alpha[:m]; beta = beta[:m]; break
        beta[j] = b_j
        if j + 1 < m:
            V[j + 1] = w / b_j
    T = np.diag(alpha[:m])
    if m > 1:
        T += np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
    ritz = np.sort(np.linalg.eigvalsh(T).real)
    return ritz[-min(k, len(ritz)):][::-1].copy()


def flip_sum(matvec, dim, k, lam_max):
    def mv_flip(v):
        return lam_max * v - matvec(v)
    mu = lanczos_largest_numpy(mv_flip, dim, k)
    eigs = np.sort(np.maximum((lam_max - mu).real, 0.0))
    return eigs


def main():
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"K={K}, N={N}, eps={EPS}, sigma={SIGMA}, k_eig={KEIG}", flush=True)

    builder = TransportMapBuilder(K=K, sigma=SIGMA)
    lap = SparseSheafLaplacian(builder, zeta, transport_mode="superposition")
    t0 = time.time(); L = lap.build_matrix(EPS); L = (L + L.getH()) * 0.5
    dim = L.shape[0]
    print(f"built L dim={dim} nnz={L.nnz} ({time.time()-t0:.0f}s)", flush=True)

    matvec = lambda v: L @ v

    # lam_max (shared)
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    print(f"lam_max ~ {lam_max:.4f}", flush=True)

    # (a) ARPACK spectral-flip reference
    t0 = time.time()
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu_a = eigsh(flip, k=KEIG, which="LA", tol=1e-6, return_eigenvectors=False)
    eigs_a = np.sort(np.maximum((lam_max - mu_a).real, 0.0))
    print(f"(a) ARPACK flip      S={eigs_a.sum():.4f}  "
          f"max_eig={eigs_a[-1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)

    # (b) ported hand-Lanczos (mimics GPU, 70 vectors)
    t0 = time.time()
    eigs_b = flip_sum(matvec, dim, KEIG, lam_max)
    print(f"(b) hand-Lanczos(70) S={eigs_b.sum():.4f}  "
          f"max_eig={eigs_b[-1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)

    # (c) ARPACK over-resolved: k=60, big ncv
    t0 = time.time()
    mu_c = eigsh(flip, k=60, which="LA", ncv=160, tol=1e-6,
                 return_eigenvectors=False)
    eigs_c = np.sort(np.maximum((lam_max - mu_c).real, 0.0))
    print(f"(c) ARPACK k=60      S20={eigs_c[:20].sum():.4f}  "
          f"S60={eigs_c.sum():.4f}  eig20={eigs_c[19]:.3e} "
          f"eig60={eigs_c[-1]:.3e}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"   smallest 60 eigs:\n{np.array2string(eigs_c, precision=3, max_line_width=130)}",
          flush=True)


if __name__ == "__main__":
    main()
