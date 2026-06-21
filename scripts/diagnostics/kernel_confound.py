#!/usr/bin/env python
"""Edge-count confound test for the β₀ (kernel) inversion.

The converged audit shows ζ is tighter (lower S) because it has a bigger
near-kernel. But kernel dimension anti-correlates with edge count (more edges =
more constraints = smaller kernel), and ζ has the fewest Rips edges. So the
question: does ζ's bigger kernel just track its sparser graph, or does ζ sit
ABOVE the edge-count trend (a genuine arithmetic surplus of flat sections)?

Reads the receipt (output/convergence_audit/converged_premium_K100.json),
recomputes each source's edge count at eps=3, extracts exact β₀ (machine-zero
modes) and the near-kernel count from the saved eigenvalues, and reports
β₀ vs edges. ζ above the line = arithmetic; ζ on the line = graph artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.spatial.distance import pdist

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource

N, EPS = 1000, 3.0
EXACT_KERNEL_TOL = 1e-8   # below this = machine-zero mode


def gue_points(z_min, z_max, seed):
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(N)
    dof = 2.0 * np.arange(N - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    e = np.sort(eigvalsh_tridiagonal(diag, sub) / np.sqrt(2.0 * N))
    s = np.diff(e); sc = s * ((z_max - z_min) / (N - 1) / s.mean())
    pts = np.zeros(N); pts[0] = z_min; pts[1:] = z_min + np.cumsum(sc)
    return pts


def edges(pts):
    return int(np.sum(pdist(pts.reshape(-1, 1)) <= EPS))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    receipt_path = _ROOT / "output" / "convergence_audit" / f"converged_premium_K{K}.json"
    R = json.loads(receipt_path.read_text())

    src = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(src.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    pts = {
        "zeta": zeta,
        "gue_2000": gue_points(z_min, z_max, 2000),
        "gue_2001": gue_points(z_min, z_max, 2001),
        "gue_2002": gue_points(z_min, z_max, 2002),
        "even": np.linspace(z_min, z_max, N),
        "random": np.sort(np.random.default_rng(42).uniform(z_min, z_max, N)),
    }

    print(f"{'source':<10} {'edges':>6} {'exactβ0':>8} {'near(<1e-3)':>11} "
          f"{'S':>9} {'gap λ_{β0+1}':>12}")
    rows = []
    for name in pts:
        if name not in R["sources"]:
            continue
        rec = R["sources"][name]
        eigs = np.array(rec["eigs30"])
        b0 = int(np.sum(eigs < EXACT_KERNEL_TOL))
        near = rec["n_below_1e-3"]
        gap = float(eigs[b0]) if b0 < len(eigs) else float("nan")
        ne = edges(pts[name])
        rows.append((name, ne, b0, near, rec["S"], gap))
        print(f"{name:<10} {ne:>6} {b0:>8} {near:>11} {rec['S']:>9.4f} {gap:>12.2e}")

    # Edge-count confound: is near-kernel explained by edges across CONTROLS,
    # and does zeta deviate from the controls' trend?
    ctrl = [r for r in rows if r[0] != "zeta"]
    z = [r for r in rows if r[0] == "zeta"]
    if len(ctrl) >= 3 and z:
        ce = np.array([r[1] for r in ctrl], float)
        cn = np.array([min(r[3], 30) for r in ctrl], float)  # near-kernel (censored at window)
        # linear fit near-kernel ~ a*edges + b on controls only
        A = np.c_[ce, np.ones_like(ce)]
        coef, *_ = np.linalg.lstsq(A, cn, rcond=None)
        pred_z = coef[0] * z[0][1] + coef[1]
        actual_z = min(z[0][3], 30)
        print(f"\n  CONTROLS near-kernel ~ {coef[0]:.4f}*edges + {coef[1]:.1f}")
        print(f"  zeta edges={z[0][1]} -> predicted near-kernel={pred_z:.1f}, "
              f"actual={actual_z}  (surplus {actual_z - pred_z:+.1f})")
        print(f"  zeta exact-β0={z[0][2]} vs controls exact-β0="
              f"{[r[2] for r in ctrl]}")
        verdict = ("ARITHMETIC SURPLUS: zeta kernel exceeds edge-count prediction"
                   if actual_z - pred_z > 3 else
                   "EDGE-COUNT EXPLAINS IT: zeta kernel on the controls' trend")
        print(f"\n  VERDICT: {verdict}")
        R["confound"] = {"controls_fit": coef.tolist(), "zeta_pred": pred_z,
                         "zeta_actual": actual_z, "verdict": verdict}
        receipt_path.write_text(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
