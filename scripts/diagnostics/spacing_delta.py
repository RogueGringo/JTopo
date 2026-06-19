#!/usr/bin/env python
"""'The delta is the knowledge' — measured. Information = divergence from the void.

Neither the raw zeta spacings nor the GUE void carries knowledge alone. The
KNOWLEDGE is D_KL(zeta || GUE) — how far the real zeros deviate from the
maximum-entropy (Wigner/GUE) baseline. This is fast (histograms, no eigensolves).

Outputs:
  * nearest-neighbour spacing distributions: zeta vs GUE (Wigner surmise) vs Poisson
  * the DELTA curve (zeta_hist - GUE_hist) — the knowledge, made visible
  * D_KL(zeta || GUE), D_KL(zeta || Poisson) in nats (the information content)
  * higher moments where the arithmetic actually lives
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource

N = 20000  # zeros to use


def gue_spacings(n, seed=0):
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(n)
    dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    e = np.sort(eigvalsh_tridiagonal(diag, sub)) / np.sqrt(2.0 * n)  # -> [-1,1]
    e = e[np.abs(e) < 0.98]                          # drop semicircle edges
    # unfold by the integrated semicircle density u(x) = (n/pi)(x*sqrt(1-x^2)+asin(x))
    u = (n / np.pi) * (e * np.sqrt(1 - e**2) + np.arcsin(e))
    s = np.diff(u); return s / s.mean()


def wigner_pdf(s):
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def poisson_pdf(s):
    return np.exp(-s)


def kl(p, q, dx):
    m = (p > 1e-12) & (q > 1e-12)
    return float(np.sum(p[m] * np.log(p[m] / q[m])) * dx)


def main():
    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    z = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z = np.sort(z)
    sz = np.diff(z); sz = sz / sz.mean()           # unfolded zeta spacings
    sg = gue_spacings(N)                            # D-E GUE void spacings

    bins = np.linspace(0, 4, 81); dx = bins[1] - bins[0]
    mids = 0.5 * (bins[:-1] + bins[1:])
    hz, _ = np.histogram(sz, bins=bins, density=True)
    hg, _ = np.histogram(sg, bins=bins, density=True)
    wig = wigner_pdf(mids); poi = poisson_pdf(mids)

    print(f"'THE DELTA IS THE KNOWLEDGE'  (N={N} unfolded zeta zeros)\n")
    print("  Information content (nats) = divergence from the baseline:")
    print(f"    D_KL(zeta || GUE-Wigner void) = {kl(hz, wig, dx):.5f}")
    print(f"    D_KL(zeta || GUE-sample void) = {kl(hz, hg, dx):.5f}")
    print(f"    D_KL(zeta || Poisson void)    = {kl(hz, poi, dx):.5f}")
    print(f"    D_KL(GUE-sample || Wigner)    = {kl(hg, wig, dx):.5f}  (finite-sample floor)")

    print("\n  The DELTA curve  (zeta_density - GUE-Wigner)  — the knowledge, by s:")
    print(f"  {'s':>5} {'zeta':>7} {'GUE':>7} {'DELTA':>8}")
    for i in range(0, len(mids), 8):
        print(f"  {mids[i]:>5.2f} {hz[i]:>7.4f} {wig[i]:>7.4f} {hz[i]-wig[i]:>+8.4f}")

    print("\n  Higher moments (where arithmetic deviations live):")
    for name, s in (("zeta", sz), ("GUE", sg)):
        print(f"    {name:<5} mean={s.mean():.4f} var={s.var():.4f} "
              f"skew={((s-s.mean())**3).mean()/s.std()**3:.4f} "
              f"P(s<0.1)={np.mean(s<0.1):.5f}  (level-repulsion hole)")
    dvar = abs(sz.var() - sg.var())
    print(f"\n    delta(variance) zeta-vs-GUE = {dvar:.5f}  "
          f"<- the knowledge is here, tiny, not in either side alone")


if __name__ == "__main__":
    main()
