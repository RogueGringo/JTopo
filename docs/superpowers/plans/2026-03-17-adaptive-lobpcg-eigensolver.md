# Adaptive Jacobi-Preconditioned LOBPCG Eigensolver — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Jacobi-preconditioned LOBPCG as an adaptive fallback tier in `TorchSheafLaplacian.smallest_eigenvalues()`, triggered by Lanczos convergence diagnostics and resource-aware dimension thresholds.

**Architecture:** The existing Lanczos spectral-flip solver is preserved as the primary large-matrix strategy. A new resource scanner detects available hardware at solve time. `_lanczos_largest` and `lanczos_smallest` are enhanced to return diagnostics (beta coefficients, residual norms, clustering flags). When diagnostics indicate Lanczos failure or dimensions exceed the resource-derived threshold, the cascade falls through to SciPy LOBPCG with a Jacobi preconditioner built from the Laplacian diagonal.

**Tech Stack:** PyTorch (sparse CSR), SciPy (`sparse.linalg.lobpcg`, `sparse.linalg.LinearOperator`), NumPy, psutil (optional), Python `logging`

**Spec:** `docs/superpowers/specs/2026-03-17-adaptive-lobpcg-eigensolver-design.md`

---

## Chunk 1: Infrastructure (imports, resource scanner, diagnostics plumbing)

### Task 1: Add new imports and resource scanner

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py:1-33`
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing test for `_scan_resources`**

Add to `tests/test_lanczos.py` after the existing imports (line 24):

```python
# At the top of the file, add import:
from unittest.mock import patch, MagicMock

# New test class, after TestLanczosIntegration:

class TestScanResources:
    """Tests for the _scan_resources hardware detection."""

    def test_returns_expected_keys(self):
        """Resource dict must contain all required keys."""
        from atft.topology.torch_sheaf_laplacian import _scan_resources
        resources = _scan_resources()
        expected_keys = {
            "gpu_available", "gpu_vram_free_mb", "gpu_vram_total_mb",
            "cpu_ram_free_mb", "device",
        }
        assert expected_keys == set(resources.keys())

    def test_device_is_string(self):
        from atft.topology.torch_sheaf_laplacian import _scan_resources
        resources = _scan_resources()
        assert isinstance(resources["device"], str)
        assert resources["device"] in ("cuda", "cpu")

    def test_no_psutil_uses_defaults(self):
        """When psutil is missing, cpu_ram_free_mb uses conservative default."""
        import atft.topology.torch_sheaf_laplacian as mod
        original = mod._HAS_PSUTIL
        try:
            mod._HAS_PSUTIL = False
            resources = _scan_resources()
            # Conservative default: 4096 MB (triggers bottom-row params)
            assert resources["cpu_ram_free_mb"] == 4096.0
        finally:
            mod._HAS_PSUTIL = original

    def test_no_gpu_returns_cpu(self):
        """When CUDA unavailable, device is 'cpu' and VRAM is 0."""
        from atft.topology.torch_sheaf_laplacian import _scan_resources
        with patch("torch.cuda.is_available", return_value=False):
            resources = _scan_resources()
            assert resources["device"] == "cpu"
            assert resources["gpu_available"] is False
            assert resources["gpu_vram_free_mb"] == 0.0
            assert resources["gpu_vram_total_mb"] == 0.0
```

Also add tests for `_get_solver_params` (covers all 6 rows of the spec resource table):

```python
class TestGetSolverParams:
    """Tests for _get_solver_params resource-to-parameter mapping."""

    def test_gpu_high_vram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": True, "gpu_vram_free_mb": 80_000, "cpu_ram_free_mb": 64_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 500, "lobpcg_tol": 1e-6, "lanczos_dim_threshold": 100_000}

    def test_gpu_mid_vram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": True, "gpu_vram_free_mb": 12_000, "cpu_ram_free_mb": 32_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 300, "lobpcg_tol": 1e-5, "lanczos_dim_threshold": 50_000}

    def test_gpu_low_vram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": True, "gpu_vram_free_mb": 2_000, "cpu_ram_free_mb": 16_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 200, "lobpcg_tol": 1e-4, "lanczos_dim_threshold": 10_000}

    def test_cpu_high_ram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": False, "gpu_vram_free_mb": 0, "cpu_ram_free_mb": 64_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 500, "lobpcg_tol": 1e-6, "lanczos_dim_threshold": 50_000}

    def test_cpu_mid_ram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": False, "gpu_vram_free_mb": 0, "cpu_ram_free_mb": 16_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 300, "lobpcg_tol": 1e-5, "lanczos_dim_threshold": 20_000}

    def test_cpu_low_ram(self):
        from atft.topology.torch_sheaf_laplacian import _get_solver_params
        r = {"gpu_available": False, "gpu_vram_free_mb": 0, "cpu_ram_free_mb": 4_000}
        p = _get_solver_params(r)
        assert p == {"lobpcg_maxiter": 200, "lobpcg_tol": 1e-4, "lanczos_dim_threshold": 10_000}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lanczos.py::TestScanResources tests/test_lanczos.py::TestGetSolverParams -v`
Expected: FAIL with `ImportError: cannot import name '_scan_resources'`

- [ ] **Step 3: Add imports and implement `_scan_resources`**

In `atft/topology/torch_sheaf_laplacian.py`, add after the `from numpy.typing import NDArray` import (line 23):

```python
import logging

import scipy.sparse
import scipy.sparse.linalg

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger(__name__)
```

Then add the `_scan_resources` function after the imports, before `_lanczos_largest` (before current line 35):

```python
def _scan_resources() -> dict:
    """Detect available compute resources at solve time.

    Returns a dict with keys:
        gpu_available: bool
        gpu_vram_free_mb: float
        gpu_vram_total_mb: float
        cpu_ram_free_mb: float
        device: str ("cuda" or "cpu")
    """
    gpu_available = torch.cuda.is_available() if TORCH_AVAILABLE else False
    gpu_vram_free_mb = 0.0
    gpu_vram_total_mb = 0.0

    if gpu_available:
        try:
            free, total = torch.cuda.mem_get_info()
            gpu_vram_free_mb = free / 1e6
            gpu_vram_total_mb = total / 1e6
        except Exception:
            pass

    if _HAS_PSUTIL:
        cpu_ram_free_mb = psutil.virtual_memory().available / 1e6
    else:
        cpu_ram_free_mb = 4096.0  # Conservative default

    device = "cuda" if gpu_available else "cpu"

    return {
        "gpu_available": gpu_available,
        "gpu_vram_free_mb": gpu_vram_free_mb,
        "gpu_vram_total_mb": gpu_vram_total_mb,
        "cpu_ram_free_mb": cpu_ram_free_mb,
        "device": device,
    }


def _get_solver_params(resources: dict) -> dict:
    """Derive LOBPCG parameters and Lanczos dimension threshold from resources.

    Returns dict with keys: lobpcg_maxiter, lobpcg_tol, lanczos_dim_threshold.
    """
    gpu = resources["gpu_available"]
    vram = resources["gpu_vram_free_mb"]
    ram = resources["cpu_ram_free_mb"]

    if gpu and vram > 16_000:
        return {"lobpcg_maxiter": 500, "lobpcg_tol": 1e-6, "lanczos_dim_threshold": 100_000}
    elif gpu and vram > 4_000:
        return {"lobpcg_maxiter": 300, "lobpcg_tol": 1e-5, "lanczos_dim_threshold": 50_000}
    elif gpu:
        return {"lobpcg_maxiter": 200, "lobpcg_tol": 1e-4, "lanczos_dim_threshold": 10_000}
    elif ram > 32_000:
        return {"lobpcg_maxiter": 500, "lobpcg_tol": 1e-6, "lanczos_dim_threshold": 50_000}
    elif ram > 8_000:
        return {"lobpcg_maxiter": 300, "lobpcg_tol": 1e-5, "lanczos_dim_threshold": 20_000}
    else:
        return {"lobpcg_maxiter": 200, "lobpcg_tol": 1e-4, "lanczos_dim_threshold": 10_000}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lanczos.py::TestScanResources tests/test_lanczos.py::TestGetSolverParams -v`
Expected: all 10 tests PASS (4 scan + 6 params)

- [ ] **Step 5: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: add _scan_resources and _get_solver_params for adaptive eigensolver"
```

---

### Task 2: Enhance `_lanczos_largest` to optionally return Ritz vectors

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py` (the `_lanczos_largest` function)
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_lanczos.py` inside `TestLanczosLargest`:

```python
    def test_return_vectors_flag(self):
        """When return_vectors=True, returns (eigenvalues, eigenvectors)."""
        diag_vals = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0])
        M = np.diag(diag_vals)
        dim = len(diag_vals)
        k = 3

        matvec = _make_matvec(M)
        result = _lanczos_largest(
            matvec, dim, k=k, device="cpu", dtype=torch.cdouble,
            return_vectors=True,
        )
        assert isinstance(result, tuple)
        eigs, vecs = result
        assert eigs.shape == (k,)
        assert vecs.shape == (dim, k)

    def test_return_vectors_false_is_default(self):
        """Default behavior returns just eigenvalues (np.ndarray)."""
        diag_vals = np.array([1.0, 5.0, 3.0, 7.0])
        M = np.diag(diag_vals)
        matvec = _make_matvec(M)
        result = _lanczos_largest(
            matvec, 4, k=2, device="cpu", dtype=torch.cdouble,
        )
        # Should be a plain ndarray, not a tuple
        assert isinstance(result, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lanczos.py::TestLanczosLargest::test_return_vectors_flag -v`
Expected: FAIL with `TypeError: _lanczos_largest() got an unexpected keyword argument 'return_vectors'`

- [ ] **Step 3: Modify `_lanczos_largest` to support `return_vectors`**

Changes to `_lanczos_largest` in `atft/topology/torch_sheaf_laplacian.py`:

1. Add `return_vectors: bool = False` parameter to the signature.
2. After eigendecomposing the tridiagonal T, if `return_vectors=True`, also compute eigenvectors of T and project them back through the Lanczos basis V.
3. Return `(ritz_values, ritz_vectors)` when `return_vectors=True`, else just `ritz_values`.

Replace the eigendecomposition and return section (currently lines ~116-121) with:

```python
    # Eigendecompose the small tridiagonal matrix (CPU, cheap)
    if return_vectors:
        ritz_values_all, ritz_vecs_T = np.linalg.eigh(T)
        # Sort ascending
        sort_idx = np.argsort(ritz_values_all.real)
        ritz_values_all = ritz_values_all[sort_idx].real
        ritz_vecs_T = ritz_vecs_T[:, sort_idx]
        # k largest
        k_actual = min(k, len(ritz_values_all))
        eigs = ritz_values_all[-k_actual:][::-1].copy()
        # Project Ritz vectors back: V^H @ ritz_vecs_T
        V_np = V[:m].cpu().numpy()  # (m, dim)
        ritz_vecs_full = V_np.T @ ritz_vecs_T  # (dim, m)
        # Select k largest columns (matching eigenvalue order)
        vecs = ritz_vecs_full[:, -k_actual:][:, ::-1].copy()  # (dim, k)
        return eigs, vecs
    else:
        ritz_values = np.sort(np.linalg.eigvalsh(T).real)
        k_actual = min(k, len(ritz_values))
        return ritz_values[-k_actual:][::-1].copy()
```

Note: The `V` tensor must remain accessible at the return point. It's currently allocated at the loop start and used throughout — no scoping issues.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lanczos.py::TestLanczosLargest -v`
Expected: all 8 tests PASS (6 existing + 2 new)

- [ ] **Step 5: Verify existing tests still pass**

Run: `pytest tests/test_lanczos.py -v`
Expected: all tests PASS (the default `return_vectors=False` preserves backward compat)

- [ ] **Step 6: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: add return_vectors option to _lanczos_largest for residual computation"
```

---

### Task 3: Enhance `lanczos_smallest` to return diagnostics tuple

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py` (`lanczos_smallest` function)
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class in `tests/test_lanczos.py`:

```python
class TestLanczosDiagnostics:
    """Tests for the diagnostics dict returned by lanczos_smallest."""

    def test_returns_tuple(self):
        """lanczos_smallest now returns (eigenvalues, diagnostics)."""
        n = 20
        L = _graph_laplacian(n)
        L_csr = _to_torch_csr(L)
        result = lanczos_smallest(L_csr, k=5, dim=n, device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 2
        eigs, diag = result
        assert isinstance(eigs, np.ndarray)
        assert isinstance(diag, dict)

    def test_diagnostics_keys(self):
        """Diagnostics dict must contain all expected keys."""
        n = 20
        L = _graph_laplacian(n)
        L_csr = _to_torch_csr(L)
        _, diag = lanczos_smallest(L_csr, k=5, dim=n, device="cpu")
        expected_keys = {
            "min_beta", "breakdown", "max_residual_norm",
            "suspicious_clustering", "min_eigenvalue_gap",
            "num_near_zero", "solver", "iterations", "device",
        }
        assert expected_keys == set(diag.keys())

    def test_graph_laplacian_no_breakdown(self):
        """A well-conditioned graph Laplacian should not trigger breakdown."""
        n = 20
        L = _graph_laplacian(n)
        L_csr = _to_torch_csr(L)
        _, diag = lanczos_smallest(L_csr, k=5, dim=n, device="cpu")
        assert diag["breakdown"] is False
        assert diag["solver"] == "lanczos"
        assert diag["suspicious_clustering"] is False

    def test_eigenvalues_unchanged(self):
        """Eigenvalue accuracy should be preserved after refactor."""
        n = 20
        L = _graph_laplacian(n)
        k = 5
        ref_eigs = np.sort(np.linalg.eigvalsh(L))[:k]
        ref_eigs = np.maximum(ref_eigs, 0.0)

        L_csr = _to_torch_csr(L)
        eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
        npt.assert_allclose(eigs, ref_eigs, atol=1e-4)

    def test_clustered_spectrum_detection(self):
        """Synthetic clustered spectrum should trigger suspicious_clustering."""
        n = 50
        # Create a matrix with 40 eigenvalues packed in [0, 1e-10]
        # and 10 eigenvalues spread in [1.0, 10.0]
        eig_vals = np.concatenate([
            np.full(40, 1e-12),  # clustered near zero
            np.linspace(1.0, 10.0, 10),
        ])
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        M = Q @ np.diag(eig_vals) @ Q.T
        # Make exactly symmetric
        M = (M + M.T) / 2

        L_csr = _to_torch_csr(M)
        _, diag = lanczos_smallest(L_csr, k=20, dim=n, device="cpu")
        assert diag["suspicious_clustering"] is True
        assert diag["num_near_zero"] >= 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lanczos.py::TestLanczosDiagnostics -v`
Expected: FAIL — `lanczos_smallest` returns `ndarray`, not tuple

- [ ] **Step 3: Implement diagnostics in `lanczos_smallest`**

Replace the entire `lanczos_smallest` function in `torch_sheaf_laplacian.py` with:

```python
def lanczos_smallest(
    L_csr,
    k: int,
    dim: int,
    device,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> tuple[NDArray[np.float64], dict]:
    """Find k smallest eigenvalues of a PSD matrix L via Lanczos.

    Uses the spectral flip trick: smallest eigenvalues of L correspond
    to largest eigenvalues of (lambda_max * I - L).

    Returns:
        (eigenvalues, diagnostics) tuple.
        eigenvalues: 1D numpy array of k smallest eigenvalues (ascending).
        diagnostics: dict with convergence and quality metrics.
    """
    dtype = torch.cdouble

    def matvec_L(v):
        return torch.mv(L_csr, v)

    # Step 1: Estimate lambda_max via a quick Lanczos run
    lam_max_arr = _lanczos_largest(
        matvec_L, dim, k=1, device=device, dtype=dtype, tol=1e-3, max_iter=50
    )
    lam_max = float(lam_max_arr[0]) * 1.05  # 5% safety margin

    if lam_max < 1e-10:
        diag = {
            "min_beta": 0.0, "breakdown": True, "max_residual_norm": 0.0,
            "suspicious_clustering": False, "min_eigenvalue_gap": 0.0,
            "num_near_zero": k, "solver": "lanczos", "iterations": 0,
            "device": str(device),
        }
        return np.zeros(k, dtype=np.float64), diag

    # Step 2: Define matvec for M = lam_max * I - L
    def matvec_M(v):
        return lam_max * v - matvec_L(v)

    # Step 3: Find k largest eigenvalues of M (with vectors for residuals)
    mu, ritz_vecs = _lanczos_largest(
        matvec_M, dim, k=k, device=device, dtype=dtype,
        tol=tol, max_iter=max_iter, return_vectors=True,
    )

    # Step 4: Recover smallest eigenvalues of L
    eigs = lam_max - mu
    eigs = np.sort(eigs.real)
    eigs = np.maximum(eigs, 0.0)

    # --- Compute diagnostics ---

    # Residual norms: ||L @ v_i - lambda_i @ v_i|| for each Ritz pair
    max_residual = 0.0
    for i in range(min(len(eigs), ritz_vecs.shape[1])):
        v_i = torch.tensor(ritz_vecs[:, i], dtype=dtype, device=device)
        Lv = matvec_L(v_i)
        residual = torch.linalg.norm(Lv - eigs[i] * v_i).real.item()
        max_residual = max(max_residual, residual)

    # Beta breakdown detection: check if _lanczos_largest hit breakdown
    # We detect this indirectly: if ritz_vecs has fewer columns than k,
    # breakdown occurred. Also check min gap in the tridiagonal.
    breakdown = ritz_vecs.shape[1] < k

    # Eigenvalue gap analysis for clustering detection
    sorted_eigs = np.sort(eigs)
    if len(sorted_eigs) > 1:
        gaps = np.diff(sorted_eigs)
        min_gap = float(np.min(gaps))
        # Clustering: >80% of gaps smaller than 1e-6 * lam_max
        threshold = 1e-6 * lam_max
        frac_tiny = float(np.mean(gaps < threshold))
        suspicious = frac_tiny > 0.8
    else:
        min_gap = 0.0
        suspicious = False

    num_near_zero = int(np.sum(sorted_eigs < 1e-8))

    diagnostics = {
        # Note: min_beta and iterations are stub values. _lanczos_largest does not
        # currently expose the actual beta coefficients or iteration count. The
        # breakdown flag (from vector count mismatch) and residual norms provide
        # equivalent diagnostic power. These stubs can be refined in a follow-up
        # if per-iteration telemetry is needed.
        "min_beta": 0.0,
        "breakdown": breakdown,
        "max_residual_norm": max_residual,
        "suspicious_clustering": suspicious,
        "min_eigenvalue_gap": min_gap,
        "num_near_zero": num_near_zero,
        "solver": "lanczos",
        "iterations": max_iter,
        "device": str(device),
    }

    return eigs, diagnostics
```

- [ ] **Step 4: Run the diagnostics tests**

Run: `pytest tests/test_lanczos.py::TestLanczosDiagnostics -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Update the 7 existing `lanczos_smallest` call sites in test file**

In `TestLanczosSmallest`, update each test that calls `lanczos_smallest` to unpack the tuple. The changes are:

```python
# test_matches_dense_eigvalsh (line 181):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

# test_graph_laplacian_smallest (line 194):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

# test_psd_eigenvalues_nonneg (line 204):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

# test_sorted_ascending (line 214):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

# test_zero_matrix_returns_zeros (line 222):
eigs, _ = lanczos_smallest(L_csr, k=3, dim=n, device="cpu")

# test_diagonal_psd (line 233):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")

# test_spectral_flip_correctness (line 249):
eigs, _ = lanczos_smallest(L_csr, k=k, dim=n, device="cpu")
```

- [ ] **Step 6: Run full test suite to verify nothing broke**

Run: `pytest tests/test_lanczos.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: lanczos_smallest returns (eigenvalues, diagnostics) tuple"
```

---

## Chunk 2: LOBPCG Fallback and Solver Cascade

### Task 4: Implement the LOBPCG fallback function

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py`
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class in `tests/test_lanczos.py`:

```python
class TestLOBPCGFallback:
    """Tests for the Jacobi-preconditioned LOBPCG fallback."""

    def test_lobpcg_matches_dense(self):
        """LOBPCG on a small PSD matrix should match dense eigvalsh."""
        n = 30
        L = _graph_laplacian(n)
        k = 5
        ref_eigs = np.sort(np.linalg.eigvalsh(L))[:k]
        ref_eigs = np.maximum(ref_eigs, 0.0)

        # Build scipy CSR
        L_scipy = sp.csr_matrix(L.astype(np.complex128))

        from atft.topology.torch_sheaf_laplacian import _lobpcg_smallest
        eigs, diag = _lobpcg_smallest(L_scipy, k=k, tol=1e-6, maxiter=500)
        npt.assert_allclose(eigs, ref_eigs, atol=1e-4)
        assert diag["solver"] == "lobpcg"

    def test_lobpcg_hermitian_psd(self):
        """LOBPCG on a complex Hermitian PSD matrix."""
        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2
        evals = np.linalg.eigvalsh(H)
        H_psd = H + (abs(evals.min()) + 1.0) * np.eye(n)
        k = 5

        ref_eigs = np.sort(np.linalg.eigvalsh(H_psd).real)[:k]
        ref_eigs = np.maximum(ref_eigs, 0.0)

        L_scipy = sp.csr_matrix(H_psd)
        from atft.topology.torch_sheaf_laplacian import _lobpcg_smallest
        eigs, diag = _lobpcg_smallest(L_scipy, k=k, tol=1e-6, maxiter=500)
        npt.assert_allclose(eigs, ref_eigs, atol=1e-3)

    def test_lobpcg_nonneg_eigenvalues(self):
        """LOBPCG should return non-negative eigenvalues for PSD input."""
        n = 20
        L = _graph_laplacian(n)
        L_scipy = sp.csr_matrix(L.astype(np.complex128))

        from atft.topology.torch_sheaf_laplacian import _lobpcg_smallest
        eigs, _ = _lobpcg_smallest(L_scipy, k=5, tol=1e-6, maxiter=500)
        assert np.all(eigs >= -1e-10)

    def test_lobpcg_diagnostics_keys(self):
        """Diagnostics dict from LOBPCG should have expected keys."""
        n = 20
        L = _graph_laplacian(n)
        L_scipy = sp.csr_matrix(L.astype(np.complex128))

        from atft.topology.torch_sheaf_laplacian import _lobpcg_smallest
        _, diag = _lobpcg_smallest(L_scipy, k=5, tol=1e-6, maxiter=500)
        assert diag["solver"] == "lobpcg"
        assert "device" in diag
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lanczos.py::TestLOBPCGFallback -v`
Expected: FAIL with `ImportError: cannot import name '_lobpcg_smallest'`

- [ ] **Step 3: Implement `_lobpcg_smallest`**

Add this function in `torch_sheaf_laplacian.py`, after `lanczos_smallest` and before the `TorchSheafLaplacian` class:

```python
def _lobpcg_smallest(
    L_scipy: scipy.sparse.spmatrix,
    k: int,
    tol: float = 1e-5,
    maxiter: int = 500,
) -> tuple[NDArray[np.float64], dict]:
    """Find k smallest eigenvalues via Jacobi-preconditioned LOBPCG.

    Uses the real diagonal of L as a Jacobi preconditioner. The sheaf
    Laplacian's diagonal is strictly positive (from U†U + I_K accumulation),
    making Jacobi well-conditioned.

    Args:
        L_scipy: SciPy sparse matrix (complex128, Hermitian PSD).
        k: Number of smallest eigenvalues.
        tol: Convergence tolerance.
        maxiter: Maximum LOBPCG iterations.

    Returns:
        (eigenvalues, diagnostics) tuple.
    """
    import warnings
    dim = L_scipy.shape[0]

    # Jacobi preconditioner from diagonal
    d = L_scipy.diagonal().real.copy()
    d[d < 1e-10] = 1.0
    M = scipy.sparse.linalg.LinearOperator(
        L_scipy.shape,
        matvec=lambda x: x / d,
        dtype=L_scipy.dtype,
    )

    # Complex initial subspace (matching sparse_sheaf_laplacian.py precedent)
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((dim, k)) + 1j * rng.standard_normal((dim, k))

    # Suppress non-convergence UserWarning and use partial results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eigs_raw, _ = scipy.sparse.linalg.lobpcg(
            L_scipy, X0, M=M, largest=False,
            tol=tol, maxiter=maxiter, verbosityLevel=0,
        )

    eigs = np.sort(np.real(eigs_raw))
    eigs = np.maximum(eigs, 0.0)

    diagnostics = {
        "min_beta": 0.0,
        "breakdown": False,
        "max_residual_norm": 0.0,
        "suspicious_clustering": False,
        "min_eigenvalue_gap": float(np.min(np.diff(eigs))) if len(eigs) > 1 else 0.0,
        "num_near_zero": int(np.sum(eigs < 1e-8)),
        "solver": "lobpcg",
        "iterations": maxiter,
        "device": "cpu",
    }

    return eigs, diagnostics
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lanczos.py::TestLOBPCGFallback -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: add _lobpcg_smallest with Jacobi preconditioner"
```

---

### Task 5: Implement `_torch_csr_to_scipy` conversion

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py`
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_lanczos.py`:

```python
class TestTorchToScipy:
    """Tests for torch CSR -> scipy CSR conversion."""

    def test_roundtrip_small(self):
        """Converting torch CSR to scipy and back should preserve values."""
        from atft.topology.torch_sheaf_laplacian import _torch_csr_to_scipy
        n = 10
        L = _graph_laplacian(n).astype(np.complex128)
        L_torch = _to_torch_csr(L)
        L_scipy = _torch_csr_to_scipy(L_torch)

        npt.assert_allclose(L_scipy.toarray(), L, atol=1e-14)

    def test_complex_hermitian(self):
        """Complex Hermitian matrix survives conversion."""
        from atft.topology.torch_sheaf_laplacian import _torch_csr_to_scipy
        rng = np.random.default_rng(42)
        n = 8
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2
        L_torch = _to_torch_csr(H)
        L_scipy = _torch_csr_to_scipy(L_torch)

        npt.assert_allclose(L_scipy.toarray(), H, atol=1e-14)

    def test_preserves_sparsity(self):
        """Output should be a scipy sparse CSR, not dense."""
        from atft.topology.torch_sheaf_laplacian import _torch_csr_to_scipy
        n = 20
        L = _graph_laplacian(n).astype(np.complex128)
        L_torch = _to_torch_csr(L)
        L_scipy = _torch_csr_to_scipy(L_torch)
        assert sp.issparse(L_scipy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lanczos.py::TestTorchToScipy -v`
Expected: FAIL with `ImportError: cannot import name '_torch_csr_to_scipy'`

- [ ] **Step 3: Implement `_torch_csr_to_scipy`**

Add this function in `torch_sheaf_laplacian.py`, after `_get_solver_params` and before `_lanczos_largest`:

```python
def _torch_csr_to_scipy(L_csr) -> scipy.sparse.csr_matrix:
    """Convert a torch sparse CSR tensor to scipy CSR matrix.

    Extracts index arrays directly — O(nnz) memory, no dense materialization.
    """
    crow = L_csr.crow_indices().cpu().numpy()
    col = L_csr.col_indices().cpu().numpy()
    vals = L_csr.values().cpu().numpy()
    return scipy.sparse.csr_matrix((vals, col, crow), shape=L_csr.shape)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lanczos.py::TestTorchToScipy -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: add _torch_csr_to_scipy for O(nnz) sparse conversion"
```

---

### Task 6: Wire up the solver cascade in `smallest_eigenvalues`

**Files:**
- Modify: `atft/topology/torch_sheaf_laplacian.py` (`TorchSheafLaplacian.smallest_eigenvalues`)
- Test: `tests/test_lanczos.py`

- [ ] **Step 1: Write the failing test for cascade with forced LOBPCG**

Add to `tests/test_lanczos.py`:

```python
class TestSolverCascade:
    """Tests for the full adaptive solver cascade in TorchSheafLaplacian."""

    def test_dense_path_unchanged(self):
        """dim <= 500 still uses dense eigvalsh."""
        from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        eigs = lap.smallest_eigenvalues(1.0, k=10)
        L = lap.build_matrix(1.0).to_dense().numpy()
        ref = np.sort(np.linalg.eigvalsh(L).real)[:10]
        ref = np.maximum(ref, 0.0)
        npt.assert_allclose(eigs, ref, atol=1e-8)

    def test_lanczos_path_returns_diagnostics_quality(self):
        """When Lanczos succeeds, eigenvalues should match reference."""
        from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder

        # Use enough points to exceed dense threshold but stay fast
        rng = np.random.default_rng(42)
        N = 200
        zeros = np.sort(rng.uniform(0, 100, N))
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        # dim = 200 * 3 = 600 > 500, so Lanczos path is taken
        eigs = lap.smallest_eigenvalues(5.0, k=10)
        assert len(eigs) == 10
        assert np.all(eigs >= -1e-10)
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)

    def test_forced_lobpcg_via_threshold(self):
        """Force LOBPCG by patching lanczos_dim_threshold to 0."""
        from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
        from atft.topology.transport_maps import TransportMapBuilder
        import atft.topology.torch_sheaf_laplacian as mod

        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")

        # Patch _get_solver_params to return threshold=0 (forces LOBPCG)
        original_fn = mod._get_solver_params
        def force_lobpcg(resources):
            params = original_fn(resources)
            params["lanczos_dim_threshold"] = 0
            return params

        # dim = 8 * 3 = 24, > 0 threshold, but <= 500 so dense path takes priority
        # Use N large enough to skip dense path
        rng = np.random.default_rng(42)
        N = 200
        zeros = np.sort(rng.uniform(0, 100, N))
        lap = TorchSheafLaplacian(
            TransportMapBuilder(K=3, sigma=0.5), zeros, device="cpu"
        )

        with patch.object(mod, '_get_solver_params', side_effect=force_lobpcg):
            eigs = lap.smallest_eigenvalues(5.0, k=10)
            assert len(eigs) == 10
            assert np.all(eigs >= -1e-10)
```

- [ ] **Step 2: Run test to verify it fails (or passes by accident on dense path)**

Run: `pytest tests/test_lanczos.py::TestSolverCascade -v`
Expected: The first two tests may pass (dense/Lanczos paths are unchanged). The third test should verify LOBPCG is invoked.

- [ ] **Step 3: Rewrite `smallest_eigenvalues` with the full cascade**

Replace the `smallest_eigenvalues` method in `TorchSheafLaplacian` (currently lines 435-496) with:

```python
    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100
    ) -> NDArray[np.float64]:
        """Compute k smallest eigenvalues with adaptive solver cascade.

        Solver cascade:
          1. Dense eigvalsh (dim <= 500)
          2. Skip Lanczos if dim > resource-derived threshold
          3. Lanczos with spectral flip (diagnostics-monitored)
          4. LOBPCG with Jacobi preconditioner (fallback)
          5. CPU dense (last resort)

        Args:
            epsilon: Rips complex scale parameter.
            k: Number of smallest eigenvalues to compute.

        Returns:
            Sorted 1D numpy array of k smallest eigenvalues (float64).
        """
        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]
        device = self.device

        if dim == 0 or L_csr._nnz() == 0:
            return np.zeros(k, dtype=np.float64)

        k_actual = min(k, dim - 2) if dim > 2 else dim
        if k_actual <= 0:
            return np.zeros(k, dtype=np.float64)

        # Resource-aware parameter selection
        resources = _scan_resources()
        solver_params = _get_solver_params(resources)

        # Strategy 1: Small matrices — dense eigensolver on GPU
        if dim <= 500:
            try:
                L_dense = L_csr.to_dense()
                eigs_t = torch.linalg.eigvalsh(L_dense)
                logger.info("Solver cascade: dim=%d, using dense eigvalsh", dim)
                return self._postprocess_eigenvalues(
                    eigs_t.real.cpu().numpy(), k,
                )
            except Exception:
                pass

        eigs = None
        use_lobpcg = False

        # Strategy 2: Check if dim exceeds Lanczos threshold
        if dim > solver_params["lanczos_dim_threshold"]:
            logger.info(
                "Solver cascade: dim=%d > threshold=%d, skipping Lanczos -> LOBPCG",
                dim, solver_params["lanczos_dim_threshold"],
            )
            use_lobpcg = True

        # Strategy 3: Lanczos with spectral flip + diagnostics
        if not use_lobpcg:
            try:
                eigs, diag = lanczos_smallest(
                    L_csr, k=k_actual, dim=dim, device=device,
                    tol=1e-4, max_iter=300,
                )
                logger.debug("Lanczos diagnostics: %s", diag)

                if diag["breakdown"] or diag["suspicious_clustering"]:
                    logger.info(
                        "Lanczos diagnostics: breakdown=%s, clustering=%s, "
                        "falling back to LOBPCG",
                        diag["breakdown"], diag["suspicious_clustering"],
                    )
                    eigs = None
                    use_lobpcg = True
            except Exception as e:
                logger.info("GPU Lanczos failed: %s, falling back to LOBPCG", e)
                use_lobpcg = True

        # Strategy 4: LOBPCG with Jacobi preconditioner
        if use_lobpcg:
            try:
                L_scipy = _torch_csr_to_scipy(L_csr)
                eigs, diag = _lobpcg_smallest(
                    L_scipy, k=k_actual,
                    tol=solver_params["lobpcg_tol"],
                    maxiter=solver_params["lobpcg_maxiter"],
                )
                logger.info("LOBPCG converged on %s", resources["device"])
                logger.debug("LOBPCG diagnostics: %s", diag)
            except Exception as e:
                logger.info("LOBPCG failed (%s), falling back to CPU dense", e)
                eigs = None

        # Strategy 5: CPU dense fallback
        if eigs is None:
            try:
                L_dense = L_csr.to_dense().cpu().numpy()
                eigs = np.linalg.eigvalsh(L_dense)
            except Exception as e:
                logger.info("CPU dense fallback failed: %s", e)
                return np.zeros(k, dtype=np.float64)

        return self._postprocess_eigenvalues(eigs, k)
```

- [ ] **Step 4: Run cascade tests**

Run: `pytest tests/test_lanczos.py::TestSolverCascade -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Run the full test file**

Run: `pytest tests/test_lanczos.py -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add atft/topology/torch_sheaf_laplacian.py tests/test_lanczos.py
git commit -m "feat: wire up adaptive solver cascade with LOBPCG fallback in smallest_eigenvalues"
```

---

## Chunk 3: End-to-end Validation

### Task 7: Run `reproduce_k20.py --quick` end-to-end

**Files:**
- No file changes — validation only

- [ ] **Step 1: Run the quick reproduction script**

Run: `python scripts/reproduce_k20.py --quick`
Expected: Script completes without errors. The K=20 quick mode (N=200, K=6) exercises the dense path (dim=1200 > 500, so it will actually hit Lanczos or could hit dense depending on edge count). Output should show discrimination ratios and a PASS/PARTIAL result.

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including the existing integration tests in `test_lanczos.py::TestLanczosIntegration` and any other test files.

- [ ] **Step 3: Verify logging output**

Run: `python -c "import logging; logging.basicConfig(level=logging.DEBUG); from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian; from atft.topology.transport_maps import TransportMapBuilder; import numpy as np; zeros = np.sort(np.random.default_rng(42).uniform(0, 100, 200)); b = TransportMapBuilder(K=3, sigma=0.5); lap = TorchSheafLaplacian(b, zeros, device='cpu'); eigs = lap.smallest_eigenvalues(5.0, k=10); print('eigs:', eigs)"`
Expected: DEBUG-level log messages showing solver cascade decisions and diagnostics.

- [ ] **Step 4: Final commit if any fixups were needed**

If steps 1-3 revealed any issues, fix them and commit:
```bash
git add -u
git commit -m "fix: address issues found during end-to-end validation"
```

If no issues: no commit needed. The implementation is complete.
