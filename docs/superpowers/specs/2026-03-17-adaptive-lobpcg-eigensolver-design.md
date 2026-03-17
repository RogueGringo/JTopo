# Adaptive Jacobi-Preconditioned LOBPCG Eigensolver

**Date:** 2026-03-17
**Status:** Approved
**Scope:** `atft/topology/torch_sheaf_laplacian.py`

## Problem

The GPU Lanczos spectral-flip solver in `TorchSheafLaplacian.smallest_eigenvalues()` fails to resolve tightly clustered near-zero eigenvalues at high dimensions (K=100, dim~200,000). On RunPod (Blackwell GPU), this produced an inverted discrimination ratio (0.5x) and a false peak at sigma=0.25 — invariant subspace breakdown.

Meanwhile, SciPy's shift-invert `eigsh` requires an LU factorization that exceeds available RAM at these dimensions (300+ GB dense for a 200,000x200,000 matrix).

## Solution

Add Jacobi-preconditioned LOBPCG as a fallback tier inside `TorchSheafLaplacian.smallest_eigenvalues()`, triggered adaptively by Lanczos convergence diagnostics and a resource-aware dimension threshold. The existing Lanczos path is preserved for cases where it works well.

## Design

### 1. Diagnostics Data Structure

`lanczos_smallest` changes from returning `np.ndarray` to `Tuple[np.ndarray, dict]`.

```python
diagnostics = {
    # Lanczos health
    "min_beta": float,              # smallest beta coefficient seen
    "breakdown": bool,              # True if beta < 1e-14 at any step
    "max_residual_norm": float,     # max ||L*v - lambda*v|| across Ritz pairs

    # Eigenvalue quality
    "suspicious_clustering": bool,  # True if gap stats suggest unresolved near-zero cluster
    "min_eigenvalue_gap": float,    # smallest gap between consecutive sorted eigenvalues
    "num_near_zero": int,           # count of eigenvalues < 1e-8

    # Metadata
    "solver": str,                  # "lanczos" | "lobpcg" | "dense"
    "iterations": int,
    "device": str,                  # "cuda:0" | "cpu"
}
```

**Clustering detection heuristic:** After sorting the k eigenvalues, compute gaps between consecutive values. If more than 80% of the eigenvalues are within `1e-6 * lam_max_estimate` of each other, flag `suspicious_clustering = True`. Here `lam_max_estimate` is the estimated largest eigenvalue of L from the spectral flip step (i.e., the full spectral range, not the max of the returned smallest eigenvalues). This makes the threshold relative to the full spectral width, which is the correct scale for judging whether near-zero eigenvalues are genuinely resolved or just numerical noise.

**Residual norm computation:** `_lanczos_largest` is modified to optionally return Ritz vectors (the Lanczos basis V projected through the tridiagonal eigenvectors). When called from `lanczos_smallest`, the Ritz vectors are used to compute residual norms `||L*v_i - lambda_i*v_i||` for each Ritz pair. `_lanczos_largest` gains an optional `return_vectors=False` parameter to avoid the cost when vectors aren't needed (e.g., the initial lambda_max estimation call).

### 2. Resource Scanner

A private function `_scan_resources()` in `torch_sheaf_laplacian.py` that detects available hardware at solve time.

```python
def _scan_resources() -> dict:
    resources = {
        "gpu_available": bool,
        "gpu_vram_free_mb": float,
        "gpu_vram_total_mb": float,
        "cpu_ram_free_mb": float,
        "device": str,
    }
    return resources
```

Resource-driven parameter selection:

| Resource condition       | LOBPCG maxiter | LOBPCG tol | Lanczos dim threshold |
|--------------------------|----------------|------------|----------------------|
| GPU + VRAM > 16 GB       | 500            | 1e-6       | 100,000              |
| GPU + VRAM 4-16 GB       | 300            | 1e-5       | 50,000               |
| GPU + VRAM < 4 GB        | 200            | 1e-4       | 10,000               |
| CPU + RAM > 32 GB        | 500            | 1e-6       | 50,000               |
| CPU + RAM 8-32 GB        | 300            | 1e-5       | 20,000               |
| CPU + RAM < 8 GB         | 200            | 1e-4       | 10,000               |

`psutil` is an optional dependency, imported via `try/except ImportError` with a module-level flag `_HAS_PSUTIL`. If not installed, conservative defaults (bottom row) are used. GPU with VRAM < 4 GB is treated equivalently to CPU-constrained — the GPU is still used for transport and assembly, but the eigensolver parameters are conservative.

### 3. Solver Cascade

The new 5-level cascade in `smallest_eigenvalues(epsilon, k)`:

```
1. build_matrix(epsilon) -> L_csr
2. resources = _scan_resources()
3. dim <= 500          -> Dense eigvalsh (unchanged)
4. dim > lanczos_threshold -> Skip to LOBPCG (step 6)
5. Try Lanczos:
   eigs, diag = lanczos_smallest(L_csr, k, dim, device)
   if diag["breakdown"] or diag["suspicious_clustering"]:
       fall through to step 6
   else:
       return self._postprocess_eigenvalues(eigs, k)
6. LOBPCG fallback:
   a. Convert torch CSR -> scipy CSR (O(nnz), no dense materialization)
   b. Jacobi preconditioner from diagonal
   c. Seeded random initial subspace X0
   d. scipy.sparse.linalg.lobpcg(L_scipy, X0, M=M, largest=False, ...)
   e. return self._postprocess_eigenvalues(eigs_raw, k)
7. CPU dense fallback (last resort, unchanged)
```

### 3a. Torch-to-SciPy Conversion (O(nnz))

For large matrices, extract CSR index arrays directly from the torch tensor:

```python
crow = L_csr.crow_indices().cpu().numpy()
col  = L_csr.col_indices().cpu().numpy()
vals = L_csr.values().cpu().numpy()
L_scipy = scipy.sparse.csr_matrix((vals, col, crow), shape=(dim, dim))
```

This avoids materializing a dense dim x dim matrix. For dim=200,000 with ~10M non-zeros, this uses ~200 MB vs 300+ GB dense.

### 3b. Jacobi Preconditioner

```python
d = L_scipy.diagonal().real
d[d < 1e-10] = 1.0  # clamp near-zero to avoid division by zero
M = scipy.sparse.linalg.LinearOperator(
    L_scipy.shape,
    matvec=lambda x: x / d  # x is 1D (dim,) from LinearOperator contract
)
```

Note: SciPy's `LinearOperator.matvec` always receives a 1D vector `(dim,)`. The `lobpcg` implementation calls the preconditioner column-by-column internally. Using `x / d` (both 1D) is correct; `x / d[:, np.newaxis]` would broadcast incorrectly to shape `(dim, dim)`.

The BSR assembly guarantees a strictly positive diagonal (each vertex accumulates U^dagger U from incident edges plus I_K contributions), so the Jacobi preconditioner is well-defined and effective.

### 3c. Reproducible Initial Subspace

```python
rng = np.random.default_rng(42)
X0 = rng.standard_normal((dim, k)) + 1j * rng.standard_normal((dim, k))
```

Uses the modern NumPy RNG API (matching `sparse_sheaf_laplacian.py` precedent). The initial subspace is complex-valued because the sheaf Laplacian is complex Hermitian — a real-valued X0 would restrict the search to a real subspace and miss complex eigenvector directions, degrading convergence. Fixed seed ensures deterministic results across runs, required by the F1-F4 falsification criteria.

## New Imports in `torch_sheaf_laplacian.py`

```python
import scipy.sparse
import scipy.sparse.linalg
import logging

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger(__name__)
```

`scipy` is already a project dependency (used by `sparse_sheaf_laplacian.py`). No new entries in `pyproject.toml`.

## Files Modified

- `atft/topology/torch_sheaf_laplacian.py` — Primary changes:
  - Add imports (`scipy.sparse`, `scipy.sparse.linalg`, `psutil` optional, `logging`)
  - Add `_scan_resources()` private function
  - Modify `_lanczos_largest()` to accept `return_vectors=False` parameter
  - Modify `lanczos_smallest()` return type to `Tuple[np.ndarray, dict]`
  - Add clustering detection and residual norm logic inside `lanczos_smallest()`
  - Add LOBPCG fallback tier in `smallest_eigenvalues()`
  - Add cascade logging (`logger.info` for solver decisions, `logger.debug` for diagnostics)
  - Update the `lanczos_smallest` call site to unpack diagnostics

- `tests/test_lanczos.py` — Update all 7 call sites to unpack the new `(eigenvalues, diagnostics)` return type. Add tests for:
  - Diagnostics dict structure validation
  - Clustering detection on a synthetically clustered spectrum
  - LOBPCG fallback path (force via mock diagnostics or dim threshold override)

## Files NOT Modified

- `base_sheaf_laplacian.py` — No changes needed; `self._postprocess_eigenvalues` already handles the output
- `sparse_sheaf_laplacian.py` — Already has its own LOBPCG fallback
- `heat_kernel_laplacian.py` — Separate approach (trace estimation), unrelated
- `reproduce_k20.py` — Uses `spectral_sum()` which calls `smallest_eigenvalues()` internally; no script changes needed

## Testing

**End-to-end:**
- `scripts/reproduce_k20.py --quick` validates the full pipeline (N=200, K=6, exercises dense path)
- A medium run (N=2000, K=20) exercises the Lanczos path and verifies diagnostics are returned

**Unit tests (in `tests/test_lanczos.py`):**
- Verify `lanczos_smallest` returns `(np.ndarray, dict)` with all expected diagnostics keys
- Verify clustering detection: construct a matrix with synthetically clustered eigenvalues, confirm `suspicious_clustering=True`
- Verify LOBPCG fallback: use a small matrix where we can compare LOBPCG output against dense `eigvalsh` ground truth (eigenvalue diff < 1e-10)
- Force-trigger LOBPCG by setting lanczos dim threshold to 0 via a test-only parameter override
- Verify `_scan_resources()` returns valid dict structure (mock `psutil` and `torch.cuda` for both present/absent cases)

**Existing test updates:**
- All 7 existing `lanczos_smallest` call sites in `test_lanczos.py` updated to unpack `(eigs, diag)` tuple

## Risks

- **LOBPCG convergence on ill-conditioned L:** Jacobi preconditioning handles the diagonal dominance, but extremely ill-conditioned transport matrices could still cause slow convergence. Mitigated by the resource-driven maxiter/tol tuning.
- **SciPy LOBPCG memory:** For dim=200,000 with k=100, LOBPCG needs ~200MB for the search vectors plus the sparse matrix. Well within even 8 GB RAM budgets.
- **Complex-valued matrices:** The sheaf Laplacian is Hermitian but complex. `scipy.sparse.linalg.lobpcg` supports complex matrices. Verified in SciPy docs.
- **LOBPCG failure:** If LOBPCG itself raises an exception or fails to converge within `maxiter`, the cascade falls through to step 7 (CPU dense fallback), matching the existing exception-handling pattern in `smallest_eigenvalues`. SciPy's `lobpcg` emits a `UserWarning` on non-convergence — this is caught and logged, then the partial result is used if the residuals are acceptable, otherwise fall through to dense.

## Observability

Cascade decisions are logged via Python `logging` at `INFO` level:
- `"Solver cascade: dim={dim}, using dense eigvalsh"`
- `"Solver cascade: dim={dim} > threshold={threshold}, skipping Lanczos -> LOBPCG"`
- `"Lanczos diagnostics: breakdown={breakdown}, clustering={clustering}, falling back to LOBPCG"`
- `"LOBPCG converged in {n} iterations on {device}"`
- `"LOBPCG failed ({reason}), falling back to CPU dense"`

Diagnostics dict is logged at `DEBUG` level for full post-mortem analysis.
