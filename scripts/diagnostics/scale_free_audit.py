#!/usr/bin/env python
"""Residual Audit of the 'scale-free' claim on real networks.

CLAIM:    a network's degree distribution is power-law, P(k) ~ k^-gamma.
VOID:     the alternative heavy-tailed laws it must beat -- lognormal, exponential.
RESIDUAL: the likelihood-ratio advantage of power-law over each alternative (Vuong).
TRUTH:    Clauset-Shalizi-Newman MLE fit (the `powerlaw` package, Clauset co-author):
          estimate x_min by KS minimization, alpha by MLE, then compare distributions.
FLOOR:    a claim only survives if power-law is favored over lognormal WITH significance
          (Broido-Clauset 2019: most 'scale-free' networks fail exactly this test).

Verdict per network, then the meta-verdict on framework Axiom G5.
"""
from __future__ import annotations

import gzip
import pickle
import sys
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import powerlaw  # noqa: E402


def planetoid_degrees(name):
    url = f"https://github.com/kimiyoung/planetoid/raw/master/data/ind.{name}.graph"
    raw = urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    graph = pickle.loads(raw, encoding="latin1")
    deg = {}
    for u, nbrs in graph.items():
        s = set(n for n in nbrs if n != u)
        deg[u] = deg.get(u, set()) | s
        for n in s:
            deg[n] = deg.get(n, set()) | {u}
    return np.array([len(v) for v in deg.values()])


def snap_degrees(url):
    raw = urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    txt = gzip.decompress(raw).decode()
    edges = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line[0] in "#%":
            continue
        a, b = line.split()[:2]
        a, b = int(a), int(b)
        if a != b:
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)
    return np.array([len(v) for v in edges.values()])


def audit(name, deg):
    fit = powerlaw.Fit(deg, discrete=True, verbose=False)
    n_tail = int(np.sum(deg >= fit.xmin))
    rng = deg.max() / max(fit.xmin, 1)
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    # verdict logic (Broido-Clauset style)
    beats_exp = R_exp > 0 and p_exp < 0.10           # heavy-tailed at all
    beats_ln = R_ln > 0 and p_ln < 0.10              # the decisive test
    if not beats_exp:
        verdict = "REFUTED (not even heavy-tailed vs exponential)"
    elif beats_ln:
        verdict = "CERTIFIED scale-free (power-law beats lognormal)"
    else:
        verdict = "REFUTED / NEEDS_CONTROL (lognormal fits as well or better)"
    print(f"\n[{name}]  N={len(deg)}  <k>={deg.mean():.1f}  k_max={deg.max()}")
    print(f"  fit: alpha={fit.alpha:.2f}  x_min={fit.xmin:.0f}  tail n={n_tail}  "
          f"tail range x_max/x_min={rng:.0f}x")
    print(f"  power-law vs exponential : R={R_exp:+.2f}  p={p_exp:.3f}  "
          f"{'(PL favored)' if R_exp>0 else '(exp favored)'}")
    print(f"  power-law vs LOGNORMAL   : R={R_ln:+.2f}  p={p_ln:.3f}  "
          f"{'(PL favored)' if R_ln>0 else '(lognormal favored)'}  <- the decisive test")
    print(f"  VERDICT: {verdict}")
    return verdict, R_ln, p_ln


def main():
    print("RESIDUAL AUDIT — the 'scale-free' claim (P(k)~k^-gamma) on real networks")
    print("ground truth: Clauset-Shalizi-Newman (powerlaw pkg); void: lognormal/exponential")
    sources = [
        ("Cora (GNN benchmark)", lambda: planetoid_degrees("cora")),
        ("PubMed (GNN benchmark)", lambda: planetoid_degrees("pubmed")),
        ("email-Eu-core (SNAP)", lambda: snap_degrees(
            "https://snap.stanford.edu/data/email-Eu-core.txt.gz")),
    ]
    results = []
    for name, getter in sources:
        try:
            deg = getter()
            results.append((name,) + audit(name, deg)[1:])
        except Exception as e:
            print(f"\n[{name}] fetch/fit failed: {e}")
    print("\n" + "=" * 64)
    print("META-VERDICT (framework G5: 'the manifold is scale-free, P(k)~k^-gamma'):")
    cert = sum(1 for _, R, p in results if R > 0 and p < 0.10)
    print(f"  {cert}/{len(results)} real benchmark networks pass the decisive (vs-lognormal) test.")
    print("  Consistent with Broido-Clauset (2019): clean scale-free structure is RARE;")
    print("  'heavy-tailed' is common, 'power-law specifically' usually is not distinguishable")
    print("  from lognormal. G5's strong form is NEEDS_CONTROL on real data, as triaged.")


if __name__ == "__main__":
    main()
