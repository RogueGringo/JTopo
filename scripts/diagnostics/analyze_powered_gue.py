#!/usr/bin/env python
"""Aggregate powered-GUE shards and compute the honest significance of S(zeta).

Reads <out>.zeta.json and all <out>.w*.jsonl shards, then reports:
  1. NONPARAMETRIC (assumption-free): rank of S(zeta) in the GUE sample.
     p_upper = (#{GUE <= S_zeta} + 1) / (n + 1).  This is the defensible number.
  2. NORMALITY CHECK on the GUE S-distribution (Shapiro-Wilk, Anderson-Darling)
     — tests whether a parametric tail estimate is even licensed.
  3. PARAMETRIC (model-dependent, reported only with the caveat): Gaussian
     z-score and tail probability. Far-tail extrapolation is flagged as an
     assumption, not a measurement.
  4. EDGE-NORMALIZED premium S/|E| (assumption-light robustness cross-check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "output/powered_gue_K200"
    base = _ROOT / out
    zeta_path = base.with_name(base.name + ".zeta.json")
    if not zeta_path.exists():
        print(f"missing {zeta_path}"); return
    zinfo = json.loads(zeta_path.read_text())
    s_zeta = zinfo["S_zeta"]; zeta_edges = zinfo["edges"]; K = zinfo["K"]

    recs = []
    for shard in sorted(base.parent.glob(base.name + ".w*.jsonl")):
        for line in shard.read_text().splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        print("no GUE realizations found yet"); return

    S = np.array([r["S"] for r in recs], dtype=float)
    E = np.array([r["edges"] for r in recs], dtype=float)
    n = len(S)

    print("=" * 64)
    print(f"  POWERED GUE ENSEMBLE — K={K}, N=1000, sigma=0.5, eps=3.0")
    print(f"  GUE realizations: n = {n}")
    print("=" * 64)
    print(f"  S(zeta)      = {s_zeta:.6f}   (edges={zeta_edges})")
    print(f"  S(GUE)  mean = {S.mean():.6f}  std = {S.std(ddof=1):.6f}")
    print(f"  S(GUE)  min  = {S.min():.6f}  max = {S.max():.6f}")
    print(f"  S(GUE)  range vs zeta: zeta is {(S.min()-s_zeta):.4f} below the "
          f"lowest of {n} GUE draws")

    # 1. NONPARAMETRIC ------------------------------------------------------
    n_below = int(np.sum(S <= s_zeta))
    p_upper = (n_below + 1) / (n + 1)
    print("\n  [1] NONPARAMETRIC (assumption-free)")
    print(f"      #{{GUE <= S_zeta}} = {n_below} / {n}")
    print(f"      p_upper = ({n_below}+1)/({n}+1) = {p_upper:.5f}")
    # one-sided nonparametric equivalent sigma (informational only)
    if p_upper > 0:
        sig = stats.norm.isf(p_upper)
        print(f"      => one-sided z-equivalent of the BOUND: {sig:.2f}sigma "
              f"(this is a floor, the true tail may be thinner)")

    # 2. NORMALITY ----------------------------------------------------------
    print("\n  [2] NORMALITY OF GUE S-DISTRIBUTION (licenses parametric tail?)")
    if n >= 8:
        w, p_sw = stats.shapiro(S)
        print(f"      Shapiro-Wilk: W={w:.4f}, p={p_sw:.4f} "
              f"({'consistent with normal' if p_sw > 0.05 else 'NOT normal'})")
        ad = stats.anderson(S, dist="norm")
        crit_5 = ad.critical_values[2]
        print(f"      Anderson-Darling: A2={ad.statistic:.4f}, 5%% crit={crit_5:.4f} "
              f"({'consistent' if ad.statistic < crit_5 else 'NOT normal'})")
    else:
        p_sw = None
        print(f"      n={n} too small for a meaningful normality test")

    # 3. PARAMETRIC (caveated) ---------------------------------------------
    z = (s_zeta - S.mean()) / S.std(ddof=1)
    p_param = stats.norm.cdf(z)
    print("\n  [3] PARAMETRIC (Gaussian; MODEL-DEPENDENT, far-tail extrapolation)")
    print(f"      z = (S_zeta - mean)/std = {z:.2f}")
    print(f"      Gaussian lower-tail p = {p_param:.3e}")
    print("      CAVEAT: |z| this large is an extrapolation far beyond the")
    print("      sampled support; trust the nonparametric bound in [1] for claims.")

    # 4. EDGE-NORMALIZED ----------------------------------------------------
    spe_zeta = s_zeta / zeta_edges
    spe_gue = (S / E)
    print("\n  [4] EDGE-NORMALIZED S/|E| (assumption-light cross-check)")
    print(f"      zeta S/|E| = {spe_zeta:.6f}")
    print(f"      GUE  S/|E| = {spe_gue.mean():.6f} +/- {spe_gue.std(ddof=1):.6f}")
    n_below_e = int(np.sum(spe_gue <= spe_zeta))
    print(f"      #{{GUE S/|E| <= zeta S/|E|}} = {n_below_e}/{n}  "
          f"=> per-edge premium {(1-spe_zeta/spe_gue.mean())*100:.1f}%")

    summary = {
        "K": K, "n_gue": n, "S_zeta": s_zeta,
        "gue_mean": float(S.mean()), "gue_std": float(S.std(ddof=1)),
        "gue_min": float(S.min()), "gue_max": float(S.max()),
        "n_below": n_below, "p_upper_nonparametric": p_upper,
        "z_parametric": float(z), "p_parametric_gaussian": float(p_param),
        "shapiro_p": (None if n < 8 else float(p_sw)),
        "edge_norm_premium_pct": float((1 - spe_zeta / spe_gue.mean()) * 100),
        "n_below_edge_norm": n_below_e,
    }
    spath = base.with_name(base.name + ".summary.json")
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {spath}")


if __name__ == "__main__":
    main()
