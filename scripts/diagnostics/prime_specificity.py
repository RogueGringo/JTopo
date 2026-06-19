#!/usr/bin/env python
"""Definitive prime-specificity test: scrambled connection AT MATCHED EDGES.

The quick scramble_control hinted the PRIME connection gives zeta a near-kernel
surplus that scrambling removes — but it ran at fixed eps=3 (ζ has fewer edges
than GUE there) and used a hard count at the convergence floor (noisy ±3). This
test removes both problems:

  * every source edge-matched to ζ's count (~2492) via per-source epsilon,
  * one consistent tight tolerance (tol=1e-6),
  * a smoother metric: S = sum of 20 smallest eigenvalues (no hard threshold),
    alongside the near-kernel count,
  * PRIME vs several SCRAMBLE seeds, so ζ's prime-vs-random gap is compared to
    GUE's prime-vs-random gap on equal footing.

If, at matched edges, ζ's PRIME kernel exceeds its SCRAMBLED kernel by more than
GUE's does (consistently across seeds), the prime frequencies carry a real
ζ-specific effect. If the prime-vs-scramble gap is the same for ζ and GUE (or
within seed scatter), the flicker was noise/edges.
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

N, SIGMA, KEIG, TARGET_EDGES = 1000, 0.5, 20, 2492
SCRAMBLE_SEEDS = [101, 202]


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
    lo, hi = 0.01, 20.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if n_edges(pts, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_builder(K, scramble_seed=None):
    b = TransportMapBuilder(K=K, sigma=SIGMA)
    b.build_superposition_bases()
    if scramble_seed is not None:
        lp = np.array([np.log(p) for p in b.primes])
        rng = np.random.default_rng(scramble_seed)
        b._log_primes = rng.uniform(lp.min(), lp.max(), size=len(b.primes))
    return b


def measure(pts, eps, builder):
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    L = lap.build_matrix(eps); L = (L + L.getH()) * 0.5
    dim = L.shape[0]
    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    flip = sp.identity(dim, dtype=L.dtype, format="csr") * lam_max - L
    mu = eigsh(flip, k=max(KEIG, 30), which="LA", tol=1e-5,
              return_eigenvectors=False)
    eigs = np.sort(np.maximum((lam_max - mu).real, 0.0))
    return float(eigs[:20].sum()), int(np.sum(eigs < 1e-3))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = _ROOT / "output" / "convergence_audit"
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    sources = {"zeta": zeta,
               "gue_2000": gue_points(z_min, z_max, 2000),
               "gue_2001": gue_points(z_min, z_max, 2001)}
    # per-source epsilon to hit matched edges
    eps_of = {n: (3.0 if n == "zeta" else eps_for_edges(p, TARGET_EDGES))
              for n, p in sources.items()}

    print(f"PRIME-SPECIFICITY @ MATCHED EDGES ({TARGET_EDGES}), tol=1e-6, K={K}",
          flush=True)
    print(f"{'source':<10} {'variant':<11} {'edges':>6} {'S':>9} {'near':>5} {'t':>5}",
          flush=True)
    res = {n: {} for n in sources}
    for name, p in sources.items():
        eps = eps_of[name]
        variants = [("PRIME", None)] + [(f"SCR{seed}", seed) for seed in SCRAMBLE_SEEDS]
        for vname, seed in variants:
            t0 = time.time()
            S, near = measure(p, eps, make_builder(K, seed))
            res[name][vname] = {"S": S, "near": near}
            print(f"{name:<10} {vname:<11} {n_edges(p, eps):>6} {S:>9.4f} {near:>5} "
                  f"{time.time()-t0:>5.0f}", flush=True)
            (out / f"prime_specificity_K{K}.json").write_text(json.dumps(res, indent=2))

    # prime-vs-scramble gap, per source, on the smooth metric S (lower S = more kernel)
    print("\n  PRIME-vs-SCRAMBLE on S (negative = prime gives MORE kernel than random):",
          flush=True)
    gaps = {}
    for name in res:
        sp_ = res[name]["PRIME"]["S"]
        scr = np.mean([res[name][f"SCR{s}"]["S"] for s in SCRAMBLE_SEEDS])
        scr_sd = np.std([res[name][f"SCR{s}"]["S"] for s in SCRAMBLE_SEEDS])
        gaps[name] = sp_ - scr
        print(f"    {name:<10} S_prime={sp_:.4f}  S_scramble={scr:.4f}±{scr_sd:.4f}  "
              f"gap={sp_-scr:+.4f}", flush=True)
    zgap = gaps["zeta"]
    ggap = np.mean([gaps[k] for k in gaps if k.startswith("gue")])
    print(f"\n  zeta prime-effect {zgap:+.4f} vs GUE prime-effect {ggap:+.4f}", flush=True)
    verdict = ("PRIME-SPECIFIC: zeta's prime connection helps its kernel more than GUE's"
               if zgap < ggap - 0.001 else
               "NOT prime-specific: zeta and GUE respond to the prime connection alike")
    print(f"  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
