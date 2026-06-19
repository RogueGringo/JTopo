#!/usr/bin/env python
"""Edge-matched confound breaker for the beta0 (kernel) inversion.

The converged audit shows zeta is tighter via a bigger near-kernel, but zeta is
also the SPARSEST graph (fewest Rips edges), and kernel anti-correlates with
edges. To separate arithmetic from graph-sparsity: re-measure every source at
the epsilon where IT has the same edge count as zeta (~2492), so all six graphs
have identical connectivity. Then compare kernels at matched edges.

  zeta still biggest kernel at matched edges -> arithmetic (confound broken).
  controls' kernels rise to ~zeta            -> it was graph sparsity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N, SIGMA, KEIG = 1000, 0.5, 30
TARGET_EDGES = 2492  # zeta's edge count at eps=3.0


def gue_points(z_min, z_max, seed):
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(N)
    dof = 2.0 * np.arange(N - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    e = np.sort(eigvalsh_tridiagonal(diag, sub) / np.sqrt(2.0 * N))
    s = np.diff(e); sc = s * ((z_max - z_min) / (N - 1) / s.mean())
    pts = np.zeros(N); pts[0] = z_min; pts[1:] = z_min + np.cumsum(sc)
    return pts


def n_edges(pts, eps):
    return int(np.sum(pdist(pts.reshape(-1, 1)) <= eps))


def eps_for_edges(pts, target):
    """Binary search epsilon so edge count ~ target."""
    lo, hi = 0.01, 20.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_edges(pts, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def kernel_at(pts, eps, K):
    builder = TransportMapBuilder(K=K, sigma=SIGMA)
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    L = lap.build_matrix(eps); L = (L + L.getH()) * 0.5
    dim = L.shape[0]
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=KEIG, which="LA", tol=1e-4, return_eigenvectors=False)
    eigs = np.sort(np.maximum((lam_max - mu).real, 0.0))
    return eigs, float(eigs[:20].sum())


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = _ROOT / "output" / "convergence_audit"
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    sources = {
        "zeta": zeta,
        "gue_2000": gue_points(z_min, z_max, 2000),
        "gue_2001": gue_points(z_min, z_max, 2001),
        "gue_2002": gue_points(z_min, z_max, 2002),
        "even": np.linspace(z_min, z_max, N),
        "random": np.sort(np.random.default_rng(42).uniform(z_min, z_max, N)),
    }
    print(f"EDGE-MATCHED KERNEL (target edges={TARGET_EDGES}) K={K}", flush=True)
    print(f"{'source':<10} {'eps':>6} {'edges':>6} {'exactβ0':>8} "
          f"{'near<1e-3':>10} {'S':>9} {'t(s)':>6}", flush=True)
    res = {}
    for name, pts in sources.items():
        t0 = time.time()
        eps = 3.0 if name == "zeta" else eps_for_edges(pts, TARGET_EDGES)
        ne = n_edges(pts, eps)
        eigs, S = kernel_at(pts, eps, K)
        b0 = int(np.sum(eigs < 1e-8)); near = int(np.sum(eigs < 1e-3))
        res[name] = {"eps": eps, "edges": ne, "exact_b0": b0,
                     "near_1e-3": near, "S": S, "eigs30": eigs.tolist()}
        print(f"{name:<10} {eps:>6.3f} {ne:>6} {b0:>8} {near:>10} {S:>9.4f} "
              f"{time.time()-t0:>6.0f}", flush=True)
        (out / f"edge_matched_K{K}.json").write_text(json.dumps(res, indent=2))

    zk = res["zeta"]["near_1e-3"]
    ctrl = [res[k]["near_1e-3"] for k in res if k != "zeta"]
    print(f"\n  AT MATCHED EDGES (~{TARGET_EDGES}): zeta near-kernel={zk}  "
          f"controls={ctrl}", flush=True)
    verdict = ("ARITHMETIC: zeta kernel still exceeds all controls at matched edges"
               if zk > max(ctrl) + 2 else
               "GRAPH SPARSITY: controls match zeta once edges are equalized")
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
