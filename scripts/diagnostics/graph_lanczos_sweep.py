#!/usr/bin/env python
"""Harden the transfer: WHEN does fixed-Krylov Lanczos fail on graph-Laplacian
small-eigenvalue estimation, and does more Krylov fix it?

Spectral clustering / GNN spectral filters / graph signal processing read the
smallest eigenvalues of a (normalized) graph Laplacian. A graph with C weakly
connected communities has a dense cluster of ~C small eigenvalues. We sweep the
community count C (= cluster size) against the Krylov dimension m, with EXACT
dense eigvalsh as ground truth, averaged over seeds.

Claim under test: the relative error in the sum of the k=20 smallest eigenvalues
  - grows as the cluster size C approaches/exceeds the Krylov budget, and
  - is FIXED by enlarging m (or by restarts).
numpy+scipy only; no torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

N, KEIG = 1000, 20
SEEDS = [0, 1, 2]
COMMUNITIES = [2, 4, 8, 16, 32, 64, 128]
KRYLOV = [70, 140, 280]


def sbm_laplacian(N, B, p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    block = rng.integers(0, B, size=N)
    same = block[:, None] == block[None, :]
    pr = np.where(same, p_in, p_out)
    A = (rng.random((N, N)) < pr).astype(np.float64)
    A = np.triu(A, 1); A = A + A.T
    d = A.sum(1); d[d == 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(N) - (Dm[:, None] * A * Dm[None, :])
    return 0.5 * (L + L.T)


def lanczos_smallest(M, k, m, lam_max):
    """Spectral-flip Lanczos with Krylov dim m (m=70 reproduces the GPU default)."""
    dim = M.shape[0]
    lam = lam_max * 1.05
    mv = lambda v: lam * v - M @ v
    rng = np.random.default_rng(0)
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    V = np.zeros((m + 1, dim)); al = np.zeros(m); be = np.zeros(m); V[0] = v
    mm = m
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
    return np.sort((lam - mu).real)


def main():
    print(f"GRAPH-LAPLACIAN LANCZOS FAILURE REGIME  (N={N}, k={KEIG}, seeds={len(SEEDS)})")
    print("rel.err of sum-of-20-smallest vs EXACT dense, by community count C x Krylov m\n")
    header = f"{'C(comm)':>8} {'cluster':>8} " + " ".join(f"m={m:<5}" for m in KRYLOV)
    print(header); print("-" * len(header))
    rows = {}
    for B in COMMUNITIES:
        errs = {m: [] for m in KRYLOV}
        csize = []
        for s in SEEDS:
            L = sbm_laplacian(N, B, p_in=0.10, p_out=0.0015, seed=s)
            e = np.sort(np.linalg.eigvalsh(L).real)
            exact = e[:KEIG].sum()
            lam_max = float(e[-1])
            # cluster size = # eigenvalues below the gap to the bulk (bottom 10% range)
            csize.append(int(np.sum(e < 0.10 * lam_max)))
            for m in KRYLOV:
                hand = lanczos_smallest(L, KEIG, m, lam_max).sum()
                errs[m].append(abs(hand - exact) / abs(exact))
        cs = int(np.mean(csize))
        cells = " ".join(f"{100*np.mean(errs[m]):>6.1f}%" for m in KRYLOV)
        print(f"{B:>8} {cs:>8} {cells}")
        rows[B] = (cs, {m: float(np.mean(errs[m])) for m in KRYLOV})

    print("\nREAD:")
    print("  - down a column (fixed m=70): error grows as the community/cluster count rises")
    print("    -> the dense small-eigenvalue cluster outgrows the fixed Krylov budget.")
    print("  - across a row (fixed C): error shrinks as Krylov m grows")
    print("    -> the fix. Enough Krylov vectors to span the cluster => converges.")
    print("  ML impact: spectral clustering / GNN spectral filters / graph signal")
    print("  processing read exactly these smallest eigenvalues; a fixed-iteration")
    print("  Lanczos/SLQ mis-reads the community structure when it is rich.")


if __name__ == "__main__":
    main()
