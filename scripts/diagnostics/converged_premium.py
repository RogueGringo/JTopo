#!/usr/bin/env python
"""Test C + residual proof: the converged spectral sum, and whether any
zeta-vs-GUE premium survives proper eigensolver convergence.

Per the convergence-audit pre-registration. For each source we build L and
compute the k_eig=20 smallest eigenvalues with a ROBUST solver (ARPACK
spectral-flip, IRAM restarts). For zeta we also return eigenvectors and report
the residual ||L x - lambda x|| / ||x|| of the smallest pairs — if those are
~1e-6, the near-zero eigenvalues are genuine and the published 70-vector-Lanczos
sum (~12) is the artifact.

Writes a receipt to output/convergence_audit/converged_premium_K{K}.json.
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N, EPS, SIGMA, KEIG = 1000, 3.0, 0.5, 20


def gue_points(z_min, z_max, seed):
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(N)
    dof = 2.0 * np.arange(N - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    e = np.sort(eigvalsh_tridiagonal(diag, sub) / np.sqrt(2.0 * N))
    sp_ = np.diff(e); scaled = sp_ * ((z_max - z_min) / (N - 1) / sp_.mean())
    pts = np.zeros(N); pts[0] = z_min; pts[1:] = z_min + np.cumsum(scaled)
    return pts


def build_L(pts, K):
    builder = TransportMapBuilder(K=K, sigma=SIGMA)
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    L = lap.build_matrix(EPS)
    return (L + L.getH()) * 0.5


def converged_smallest(L, k, want_vecs=False):
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    dim = L.shape[0]
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    if want_vecs:
        mu, X = eigsh(flip, k=k, which="LA", tol=1e-5)
        eigs = lam_max - mu
        order = np.argsort(eigs.real)
        return np.maximum(eigs.real[order], 0.0), X[:, order], lam_max
    mu = eigsh(flip, k=k, which="LA", tol=1e-4, return_eigenvectors=False)
    return np.sort(np.maximum((lam_max - mu).real, 0.0)), None, lam_max


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = _ROOT / "output" / "convergence_audit"
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"K": K, "N": N, "eps": EPS, "sigma": SIGMA, "keig": KEIG,
               "machine": "i9-9900K / 32GB / win", "sources": {}}

    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())

    # zeta first (residual proof early), then GUE (premium verdict early),
    # then even/random as secondary controls.
    sources = {
        "zeta": zeta,
        "gue_2000": gue_points(z_min, z_max, 2000),
        "gue_2001": gue_points(z_min, z_max, 2001),
        "gue_2002": gue_points(z_min, z_max, 2002),
        "even": np.linspace(z_min, z_max, N),
        "random": np.sort(np.random.default_rng(42).uniform(z_min, z_max, N)),
    }

    print(f"CONVERGED SPECTRAL SUM (ARPACK) -- K={K}, k_eig={KEIG}", flush=True)
    print(f"{'source':<10} {'S_converged':>12} {'lam_max':>9} {'n<1e-3':>7} {'t(s)':>6}",
          flush=True)
    for name, pts in sources.items():
        t0 = time.time()
        L = build_L(pts, K)
        want = (name == "zeta")
        eigs, X, lam_max = converged_smallest(L, max(KEIG, 30), want_vecs=want)
        S = float(eigs[:KEIG].sum())
        n_kernel = int(np.sum(eigs < 1e-3))
        rec = {"S": S, "lam_max": lam_max, "n_below_1e-3": n_kernel,
               "eigs30": eigs[:30].tolist(), "time_s": time.time() - t0}
        if want and X is not None:
            # residual proof on the 5 smallest genuine pairs
            res = []
            for c in range(5):
                x = X[:, c]; lam = eigs[c]
                r = np.linalg.norm(L @ x - lam * x) / np.linalg.norm(x)
                res.append(float(r))
            rec["smallest5_eigs"] = eigs[:5].tolist()
            rec["smallest5_residuals"] = res
            print(f"  [zeta residual proof] smallest 5 eigs={np.array2string(eigs[:5], precision=2)}",
                  flush=True)
            print(f"  [zeta residual proof] ||Lx-lam*x||/||x|| ={['%.1e' % r for r in res]}",
                  flush=True)
        receipt["sources"][name] = rec
        print(f"{name:<10} {S:>12.4f} {lam_max:>9.3f} {n_kernel:>7} {rec['time_s']:>6.0f}",
              flush=True)
        (out / f"converged_premium_K{K}.json").write_text(json.dumps(receipt, indent=2))

    # premium analysis
    sz = receipt["sources"]["zeta"]["S"]
    gue = [receipt["sources"][k]["S"] for k in ("gue_2000", "gue_2001", "gue_2002")]
    print(f"\nCONVERGED PREMIUM CHECK:", flush=True)
    print(f"  S(zeta)={sz:.4f}  S(GUE)={np.mean(gue):.4f}±{np.std(gue):.4f}  "
          f"S(even)={receipt['sources']['even']['S']:.4f}  "
          f"S(random)={receipt['sources']['random']['S']:.4f}", flush=True)
    if np.mean(gue) > 0:
        prem = (1 - sz / np.mean(gue)) * 100
        print(f"  converged premium (GUE-zeta)/GUE = {prem:+.1f}%  "
              f"[published claim: +21.5%]", flush=True)
    receipt["converged_premium_pct"] = (
        float((1 - sz / np.mean(gue)) * 100) if np.mean(gue) > 0 else None)
    (out / f"converged_premium_K{K}.json").write_text(json.dumps(receipt, indent=2))
    print(f"\nreceipt -> {out / f'converged_premium_K{K}.json'}", flush=True)


if __name__ == "__main__":
    main()
