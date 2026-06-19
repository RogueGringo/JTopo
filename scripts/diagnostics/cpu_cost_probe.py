#!/usr/bin/env python
"""CPU cost probe for the GUE-ensemble power-up.

Validates that the CPU SparseSheafLaplacian reproduces the known GPU
spectral sums (S_zeta(K=200) ~ 11.784) and measures wall-time per
realization across K so the powered ensemble can be sized honestly.

No torch / no GPU. numpy + scipy only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.spatial.distance import pdist

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N = 1000
EPSILON = 3.0
K_EIG = 20
SIGMA = 0.5


def gue_points(n, z_min, z_max, seed):
    """Dumitriu-Edelman GUE eigenvalues, spacing-preserving unfold to [z_min, z_max]."""
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(n)
    dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    eigs = np.sort(eigvalsh_tridiagonal(diag, sub) / np.sqrt(2.0 * n))
    spacings = np.diff(eigs)
    target_mean = (z_max - z_min) / (n - 1)
    scaled = spacings * (target_mean / spacings.mean())
    pts = np.zeros(n)
    pts[0] = z_min
    pts[1:] = z_min + np.cumsum(scaled)
    return pts


def spectral_sum(pts, K):
    builder = TransportMapBuilder(K=K, sigma=SIGMA)
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    eigs = lap.smallest_eigenvalues(EPSILON, k=K_EIG)
    return float(np.sum(eigs))


def main():
    source = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(source.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    print(f"zeta: N={N} range=[{z_min:.2f},{z_max:.2f}]  edges@eps3="
          f"{int(np.sum(pdist(zeta.reshape(-1,1))<=EPSILON))}")
    print(f"{'K':>5} | {'S_zeta':>10} {'t_zeta(s)':>10} | {'S_gue':>10} {'t_gue(s)':>10}")
    print("-" * 56)
    for K in (20, 50, 100, 200):
        t0 = time.time(); sz = spectral_sum(zeta, K); tz = time.time() - t0
        g = gue_points(N, z_min, z_max, 2000)
        t0 = time.time(); sg = spectral_sum(g, K); tg = time.time() - t0
        print(f"{K:>5} | {sz:>10.4f} {tz:>10.1f} | {sg:>10.4f} {tg:>10.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
