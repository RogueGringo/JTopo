"""Regression test for the 2026-06 convergence audit.

The published spectral sums were produced by a fixed 70-vector Lanczos that does
NOT converge on the sheaf Laplacian's low-lying spectrum — it over-estimated the
sum of the 20 smallest eigenvalues by 77-162x at small scale (and ~3700x at
K=100). Every prior eigensolver test ran at dim <= ~42 (the dense branch), so the
iterative path was never checked against ground truth at a scale where it breaks.

These tests run at dim > 500 (iterative territory) and pin the converged value
against an EXACT dense eigendecomposition — the one reference no iterative solver
can dispute. If a future change reintroduces an under-converged solver, the
spectral sum jumps from ~0.1 to ~14 and these tests fail loudly.

They build a dense matrix at dim ~3000-4000 and run a full dense eigendecomposition,
so they are marked ``slow`` — exclude from the fast suite with ``-m "not slow"``.

See docs/CONVERGENCE_AUDIT_findings.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

EPS, SIGMA, KEIG = 3.0, 0.5, 20
DATA = "data/odlyzko_zeros.txt"


def _build(n, K):
    """Build the (Hermitian-symmetrized) sheaf Laplacian for N=n zeros, fiber dim K."""
    zeros = SpectralUnfolding(method="zeta").transform(
        ZetaZerosSource(DATA).generate(n)).points[:, 0]
    lap = SparseSheafLaplacian(TransportMapBuilder(K=K, sigma=SIGMA), zeros,
                               transport_mode="superposition")
    L = lap.build_matrix(EPS)
    return lap, (L + L.getH()) * 0.5


def _dense_spectral_sum(L, keig):
    eigs = np.sort(np.linalg.eigvalsh(L.toarray()).real)
    return float(np.maximum(eigs[:keig], 0.0).sum())


@pytest.mark.slow
@pytest.mark.parametrize("n,K", [(150, 20), (200, 20)])
def test_smallest_eigenvalues_match_exact_dense(n, K):
    """SparseSheafLaplacian must match exact dense at dim>500 (the iterative path)."""
    lap, L = _build(n, K)
    assert L.shape[0] > 500                       # genuinely the iterative branch
    dense = _dense_spectral_sum(L, KEIG)
    got = float(np.sum(lap.smallest_eigenvalues(EPS, k=KEIG)))
    # exact ground truth is O(0.1); converged solver must land on it, not ~14
    assert dense < 1.0, f"sanity: dense sum should be small, got {dense}"
    assert abs(got - dense) < 0.05, (
        f"spectral sum {got:.4f} != exact dense {dense:.4f} "
        f"(ratio {got/max(dense,1e-9):.1f}x) — eigensolver under-converged?")


@pytest.mark.slow
def test_under_converged_solver_would_be_caught():
    """Document the failure mode: a 70-vector spectral-flip Lanczos over-estimates.

    This is the exact algorithm the GPU backends use; here we reproduce it on the
    same matrix and assert it is FAR from the truth — so the guard above is known
    to bite. (Not a solver we ship; a tripwire for the bug.)
    """
    from scipy.sparse.linalg import eigsh

    lap, L = _build(150, 20)
    dim = L.shape[0]
    dense = _dense_spectral_sum(L, KEIG)

    lam_max = float(eigsh(L, k=1, which="LA", tol=1e-3,
                          return_eigenvectors=False)[0]) * 1.05
    matvec = lambda v: lam_max * v - (L @ v)
    m = min(max(2 * KEIG + 20, KEIG + 50), dim)   # = 70, the GPU's Krylov dim
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.complex128); v /= np.linalg.norm(v)
    V = np.zeros((m + 1, dim), dtype=np.complex128)
    al = np.zeros(m); be = np.zeros(m); V[0] = v
    for j in range(m):
        w = matvec(V[j])
        if j > 0:
            w = w - be[j - 1] * V[j - 1]
        a = np.real(np.vdot(V[j], w)); al[j] = a; w = w - a * V[j]
        for _ in range(2):
            w = w - V[:j + 1].T @ (V[:j + 1].conj() @ w)
        b = np.linalg.norm(w).real
        if b < 1e-14:
            m = j + 1; al = al[:m]; be = be[:m]; break
        be[j] = b
        if j + 1 < m:
            V[j + 1] = w / b
    T = np.diag(al[:m])
    if m > 1:
        T += np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
    mu = np.sort(np.linalg.eigvalsh(T).real)[-KEIG:]
    hand = float(np.sort(np.maximum((lam_max - mu).real, 0.0)).sum())

    # the bug: ~14 vs true ~0.19 — a >50x over-estimate
    assert hand > 10 * dense, (
        f"expected the 70-vector Lanczos to grossly over-estimate; "
        f"hand={hand:.3f} dense={dense:.3f}")
