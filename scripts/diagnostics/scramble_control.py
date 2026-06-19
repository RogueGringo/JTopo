#!/usr/bin/env python
"""Scrambled-connection control — does the kernel surplus need the PRIMES?

The load-bearing claim is that the phase exp(i*dgamma*log p) — the explicit-
formula Fourier kernel — makes the PRIME frequencies resonate specifically with
zeta-zero gaps. Test it: keep the prime generators B_p, but replace the
frequencies {log p} with RANDOM frequencies of matched range. Recompute the
near-kernel for zeta and GUE under PRIME vs SCRAMBLED connections.

  PRIME: zeta surplus, SCRAMBLED: surplus gone  -> ARITHMETIC (primes essential).
  surplus survives scrambling                   -> it's zeta's spacing geometry,
                                                   "arithmetic" is the wrong word.

K=100, eps=3.0, sigma=0.5, k_eig=20. Averages scrambled over a few seeds.
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
    s = np.diff(e); sc = s * ((z_max - z_min) / (N - 1) / s.mean())
    pts = np.zeros(N); pts[0] = z_min; pts[1:] = z_min + np.cumsum(sc)
    return pts


def make_builder(K, scramble_seed=None):
    """Prime connection, or frequency-scrambled if scramble_seed is given."""
    b = TransportMapBuilder(K=K, sigma=SIGMA)
    b.build_superposition_bases()  # builds B_p AND sets b._log_primes = [log p]
    if scramble_seed is not None:
        lp = np.array([np.log(p) for p in b.primes])
        rng = np.random.default_rng(scramble_seed)
        # random frequencies over the SAME range as the prime log-frequencies,
        # keeping the prime generators B_p fixed (scramble only the resonance)
        b._log_primes = rng.uniform(lp.min(), lp.max(), size=len(b.primes))
    return b


def near_kernel(pts, builder):
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    L = lap.build_matrix(EPS); L = (L + L.getH()) * 0.5
    dim = L.shape[0]
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=max(KEIG, 30), which="LA", tol=1e-4,
              return_eigenvectors=False)
    eigs = np.sort(np.maximum((lam_max - mu).real, 0.0))
    return int(np.sum(eigs < 1e-3)), int(np.sum(eigs < 1e-8)), float(eigs[:20].sum())


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = _ROOT / "output" / "convergence_audit"
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    pts = {"zeta": zeta,
           "gue_2000": gue_points(z_min, z_max, 2000),
           "gue_2001": gue_points(z_min, z_max, 2001)}

    variants = [("PRIME", None), ("SCRAMBLE-1", 101), ("SCRAMBLE-2", 202)]
    print(f"SCRAMBLED-CONNECTION CONTROL  K={K}  (near-kernel <1e-3)", flush=True)
    print(f"{'variant':<11} {'source':<10} {'near':>5} {'exactβ0':>8} {'S':>9} {'t':>5}",
          flush=True)
    res = {}
    for vname, seed in variants:
        res[vname] = {}
        for name, p in pts.items():
            t0 = time.time()
            builder = make_builder(K, scramble_seed=seed)
            near, b0, S = near_kernel(p, builder)
            res[vname][name] = {"near": near, "exact_b0": b0, "S": S}
            print(f"{vname:<11} {name:<10} {near:>5} {b0:>8} {S:>9.4f} "
                  f"{time.time()-t0:>5.0f}", flush=True)
            (out / f"scramble_control_K{K}.json").write_text(json.dumps(res, indent=2))

    # surplus = zeta near-kernel minus mean GUE near-kernel, per variant
    print("\n  zeta near-kernel SURPLUS over mean(GUE), by variant:", flush=True)
    for vname, _ in variants:
        zk = res[vname]["zeta"]["near"]
        g = np.mean([res[vname][k]["near"] for k in res[vname] if k.startswith("gue")])
        print(f"    {vname:<11} surplus = {zk} - {g:.1f} = {zk - g:+.1f}", flush=True)
    prime_surplus = res["PRIME"]["zeta"]["near"] - np.mean(
        [res["PRIME"][k]["near"] for k in res["PRIME"] if k.startswith("gue")])
    scr_surplus = np.mean([
        res[v]["zeta"]["near"] - np.mean([res[v][k]["near"] for k in res[v] if k.startswith("gue")])
        for v in ("SCRAMBLE-1", "SCRAMBLE-2")])
    verdict = ("ARITHMETIC: surplus needs the prime frequencies (dies when scrambled)"
               if prime_surplus - scr_surplus > 3 else
               "GEOMETRY: surplus survives scrambling -> not prime-specific")
    print(f"\n  PRIME surplus {prime_surplus:+.1f} vs SCRAMBLED surplus {scr_surplus:+.1f}",
          flush=True)
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
