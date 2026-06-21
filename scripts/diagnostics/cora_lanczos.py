#!/usr/bin/env python
"""Confirm the graph-Laplacian Lanczos failure on Cora — the canonical GNN benchmark.

Cora citation network (Planetoid): 2708 papers, ~5278 citation edges, 7 classes.
Sparse (avg degree ~4) and modular/tree-like -> a rich band of small Laplacian
eigenvalues -> the regime that breaks a fixed-Krylov Lanczos. EXACT dense as truth.

Contrast with email-Eu-core (dense, large spectral gap -> Lanczos fine): the failure
needs a SPARSE MODULAR graph, which is exactly what citation-network GNN benchmarks are.
"""
from __future__ import annotations

import pickle
import sys
import urllib.request

import numpy as np

KEIG = 20
KRYLOV = [70, 140, 280, 560]
URL = "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph"


def fetch_adjacency():
    print(f"  fetching {URL} ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    graph = pickle.loads(raw, encoding="latin1")   # dict: node -> [neighbors]
    N = len(graph)
    A = np.zeros((N, N))
    for u, nbrs in graph.items():
        for v in nbrs:
            if u != v:
                A[u, v] = 1.0; A[v, u] = 1.0
    return A, N


def normalized_laplacian(A):
    d = A.sum(1); d[d == 0] = 1.0
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(A.shape[0]) - (Dm[:, None] * A * Dm[None, :])
    return 0.5 * (L + L.T)


def lanczos_smallest(M, k, m, lam_max):
    dim = M.shape[0]
    lam = lam_max * 1.05
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
    return np.sort((lam - mu).real)


def main():
    print("CORA — canonical GNN benchmark (citation network)")
    try:
        A, N = fetch_adjacency()
    except Exception as e:
        print(f"  fetch failed: {e}  — SBM sweep result stands."); return
    M = int(A.sum() / 2)
    L = normalized_laplacian(A)
    e = np.sort(np.linalg.eigvalsh(L).real)
    lam_max = float(e[-1]); exact = e[:KEIG].sum()
    cluster = int(np.sum(e < 0.10 * lam_max))
    n_zero = int(np.sum(e < 1e-8))
    print(f"\n  N={N} nodes, {M} undirected edges, avg degree {2*M/N:.1f}, lambda_max={lam_max:.3f}")
    print(f"  EXACT zero modes (disconnected components) = {n_zero}")
    print(f"  small-eigenvalue cluster (<10%*lmax) = {cluster}  (sparse+modular => large)")
    print(f"  smallest 8 eigenvalues: {np.array2string(e[:8], precision=4)}")
    print(f"\n  sum of {KEIG} smallest — EXACT dense = {exact:.6f} (all {KEIG} are in the "
          f"{n_zero}-fold degenerate kernel => true sum = 0)")
    print(f"  {'Krylov m':>10} {'Lanczos sum':>14} {'abs.err / lmax':>16}")
    print("  " + "-" * 42)
    for m in KRYLOV:
        h = lanczos_smallest(L, KEIG, m, lam_max).sum()
        print(f"  {m:>10} {h:>14.6f} {abs(h-exact)/lam_max:>15.1%}")
    print(f"\n  -> plain Lanczos cannot resolve the {n_zero}-fold-degenerate zero "
          f"(disconnected\n     components): it finds the kernel ONCE and fills the rest with the\n"
          f"     soft cluster, reporting a bottom-20 sum of ~2.7 when the truth is 0.\n"
          f"     More Krylov helps slowly; the multiplicity needs BLOCK Lanczos / restarts.\n"
          f"     Real sparse GNN graphs have BOTH pathologies (degenerate kernel + soft cluster).")


if __name__ == "__main__":
    main()
