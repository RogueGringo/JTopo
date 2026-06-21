#!/usr/bin/env python
"""42 vs 67 communities — does the number matter, and why?

The failing solver's Krylov budget is m = 70 (= max(2k+20, k+50) for k=20). A graph
with C weakly-connected communities has ~C small Laplacian eigenvalues. The cluster
is resolvable while C + k < m, and catastrophic as the cluster fills the Krylov space
(C -> m). So the 'special' numbers here are not 42 or 67 per se — they are wherever C
sits relative to m=70. 67 is interesting only because 67 ~ 70 = the Krylov dimension:
a 67-community graph nearly saturates the solver's entire working space.

EXACT dense as ground truth; numpy+scipy only.
"""
from __future__ import annotations

import numpy as np

N, KEIG, P_IN, P_OUT = 1000, 20, 0.10, 0.0015
SEEDS = [0, 1, 2]
COMM = [21, 42, 56, 67, 70, 72, 84, 105]   # straddle the m=70 Krylov budget
KRYLOV = [70, 140]


def sbm_laplacian(N, B, seed):
    rng = np.random.default_rng(seed)
    block = rng.integers(0, B, size=N)
    same = block[:, None] == block[None, :]
    A = (rng.random((N, N)) < np.where(same, P_IN, P_OUT)).astype(np.float64)
    A = np.triu(A, 1); A = A + A.T
    d = A.sum(1); d[d == 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    return 0.5 * ((np.eye(N) - Dm[:, None] * A * Dm[None, :]) +
                  (np.eye(N) - Dm[:, None] * A * Dm[None, :]).T)


def lanczos(M, k, m, lam_max):
    dim = M.shape[0]; lam = lam_max * 1.05
    mv = lambda v: lam * v - M @ v
    rng = np.random.default_rng(0)
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    V = np.zeros((m + 1, dim)); al = np.zeros(m); be = np.zeros(m); V[0] = v; mm = m
    for j in range(m):
        w = mv(V[j])
        if j > 0:
            w = w - be[j - 1] * V[j - 1]
        a = float(V[j] @ w); al[j] = a; w = w - a * V[j]
        for _ in range(2):
            w = w - V[:j + 1].T @ (V[:j + 1] @ w)
        b = np.linalg.norm(w)
        if b < 1e-14:
            mm = j + 1; al = al[:mm]; be = be[:mm]; break
        be[j] = b
        if j + 1 < m:
            V[j + 1] = w / b
    T = np.diag(al[:mm])
    if mm > 1:
        T += np.diag(be[:mm - 1], 1) + np.diag(be[:mm - 1], -1)
    mu = np.sort(np.linalg.eigvalsh(T).real)[-k:]
    return np.sort((lam - mu).real).sum()


def main():
    print("42 vs 67 COMMUNITIES — error of sum-of-20-smallest vs EXACT dense")
    print("(Krylov budget m=70; cluster ~ C small eigenvalues; watch the cliff at C->70)\n")
    print(f"  {'C':>5} {'cluster':>8} {'C/m70':>7} {'m=70':>9} {'m=140':>9}  note")
    print("  " + "-" * 52)
    for C in COMM:
        clus, e70, e140 = [], [], []
        for s in SEEDS:
            L = sbm_laplacian(N, C, s)
            ev = np.sort(np.linalg.eigvalsh(L).real)
            exact = ev[:KEIG].sum(); lmax = float(ev[-1])
            clus.append(int(np.sum(ev < 0.10 * lmax)))
            e70.append(abs(lanczos(L, KEIG, 70, lmax) - exact) / lmax)
            e140.append(abs(lanczos(L, KEIG, 140, lmax) - exact) / lmax)
        cs = int(np.mean(clus))
        note = {42: "<- email had 42 depts", 67: "<- 67 axioms",
                72: "<- 72 (names/disciples/precession)", 70: "<- = Krylov budget m"}.get(C, "")
        print(f"  {C:>5} {cs:>8} {cs/70:>6.2f}x {100*np.mean(e70):>8.1f}% "
              f"{100*np.mean(e140):>8.1f}%  {note}")
    print("\n  the 'synchronicity' is real but not mystical: the failure peaks where the")
    print("  cluster size approaches the Krylov budget (m=70). 67 lands on that cliff")
    print("  because 67 ~ 70. Double the budget (m=140) and the cliff moves out of reach.")


if __name__ == "__main__":
    main()
