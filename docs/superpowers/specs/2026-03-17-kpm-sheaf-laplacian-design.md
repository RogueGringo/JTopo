# KPM Sheaf Laplacian — Design Spec

**Date:** 2026-03-17
**Status:** Approved
**Scope:** New `KPMSheafLaplacian` class + refactor of shared SLQ primitives + updated falsification criteria

## Problem

The discrete eigenvalue approach (Lanczos, LOBPCG) suffers from invariant subspace breakdown on clustered near-zero eigenvalues at high K. As K→∞, the spectrum becomes continuous — extracting individual eigenvalues is mathematically ill-posed in the thermodynamic limit. The project needs a macroscopic spectral observable aligned with the continuous density of states, not discrete eigenvalue extraction.

## Solution

Implement the Kernel Polynomial Method (KPM) with Jackson damping to reconstruct the continuous density of states ρ(λ) from Stochastic Lanczos Quadrature. The GPU computes raw Chebyshev moments μₙ = (1/dim) Tr(Tₙ(L̃)) via the existing Chebyshev recurrence + Hutchinson trace machinery. Jackson damping is applied dynamically at reconstruction time, preserving the raw moments for analytic scaling analysis.

The primary observable shifts from the eigenvalue-based spectral sum to the Integrated Density of States (IDOS) — a macroscopic, scale-invariant topological measure of the near-kernel that is robust against spectral clustering and scales to K=500+.

## Design

### 1. Class Hierarchy — Hoist SLQ Primitives

Two SLQ primitives are hoisted to `TorchSheafLaplacian`:

- `_power_iteration_lam_max`: Exists as a method on `HeatKernelSheafLaplacian` (lines 197-220). Moved directly — no code change needed.
- `_rademacher_probes`: Currently 7 lines of inline code inside `HeatKernelSheafLaplacian.heat_trace()` (lines 156-164). Extracted into a new method with a `seed` parameter (default 42, matching the existing hardcoded seed). `heat_trace()` is updated to call `self._rademacher_probes(dim, num_vectors, seed=42)` to preserve exact reproducibility.

```python
# New methods in TorchSheafLaplacian:

def _power_iteration_lam_max(self, L_csr, dim: int, n_iter: int = 30) -> float:
    """Estimate largest eigenvalue via power iteration. O(n_iter * nnz).
    Returns lam_max with 5% safety margin."""

def _rademacher_probes(self, dim: int, num_vectors: int, seed: int = 42) -> torch.Tensor:
    """Generate Rademacher probe matrix Z in {-1, +1}^{dim x num_vectors}.
    Returns complex128 tensor on self.device."""
```

`HeatKernelSheafLaplacian` is updated to call `self._power_iteration_lam_max()` and `self._rademacher_probes()` instead of its inline copies. Pure refactor — no behavior change.

Resulting hierarchy:

```
BaseSheafLaplacian (ABC)
├── SparseSheafLaplacian (SciPy CPU)
├── TorchSheafLaplacian (GPU, Lanczos, SLQ primitives)
│   ├── HeatKernelSheafLaplacian (Tr(e^{-tL}), existing)
│   └── KPMSheafLaplacian (raw moments, IDOS, ρ(λ))  ← NEW
```

### 2. KPMSheafLaplacian — Moment Computation

```python
class KPMSheafLaplacian(TorchSheafLaplacian):
    """KPM-based sheaf Laplacian — density of states via Chebyshev moments.

    Computes raw Chebyshev moments μₙ = (1/dim) Tr(Tₙ(L̃)) on GPU,
    then reconstructs ρ(λ), IDOS, and spectral density at zero using
    Jackson-damped KPM expansion on CPU.

    The raw moments are path-ordered holonomies — the exact analytic
    objects needed to bridge discrete numerics to continuous spectral
    bounds in the thermodynamic limit K→∞.

    Args:
        builder: TransportMapBuilder instance.
        zeros: 1D array of unfolded zero positions.
        transport_mode: "superposition" (default), "fe", or "resonant".
        device: Torch device (None for auto-detection).
        num_vectors: Number of Rademacher probe vectors (default 30).
        degree: Chebyshev polynomial degree (default 300).
    """

    def __init__(self, builder, zeros, transport_mode="superposition",
                 device=None, num_vectors=30, degree=300):
        super().__init__(builder, zeros, transport_mode, device)
        self._kpm_num_vectors = num_vectors
        self._kpm_degree = degree
        self._moments = None  # raw, undamped Chebyshev moments
        self._lam_max = None
        self._dim = None
```

**`compute_moments(epsilon)` — the GPU workhorse:**

```
1. L_csr = self.build_matrix(epsilon)
2. lam_max = self._power_iteration_lam_max(L_csr, dim)
3. L_norm = (2/lam_max) * L - I  (maps spectrum to [-1, 1])
4. Z = self._rademacher_probes(dim, num_vectors)
5. Chebyshev recurrence with per-step Hutchinson trace:
     T_prev = Z                      # T_0
     T_curr = L_norm @ Z             # T_1
     mu[0] = (1/dim) * real(mean_j(z_j† T_0 z_j))
     mu[1] = (1/dim) * real(mean_j(z_j† T_1 z_j))
     For k = 2..D:
       T_next = 2 * L_norm @ T_curr - T_prev
       mu[k] = (1/dim) * real(mean_j(z_j† T_k z_j))
       T_prev = T_curr
       T_curr = T_next
6. Store: self._moments = mu (D+1 real floats)
         self._lam_max = lam_max
         self._dim = dim
```

**Reality enforcement:** The Hutchinson trace uses `torch.real()` to discard floating-point imaginary noise from complex Hermitian matrix operations. The sheaf Laplacian is Hermitian, so all spectral moments are strictly real.

**Cost:** O(D * nnz * num_vectors) SpMV operations — identical to the heat kernel. For D=300, nnz=10M, num_vectors=30: ~30-90 seconds on A100.

**Memory:** D+1 floats for moments (~2.4 KB at D=300). Probe matrix Z and Chebyshev iterates T_prev/T_curr have the same memory footprint as the existing heat kernel.

### 3. KPM Reconstructors

All reconstruction methods operate on stored raw moments with Jackson damping applied dynamically.

**Jackson damping coefficients:**

```python
def _jackson_coefficients(self, D: int) -> np.ndarray:
    """Jackson kernel damping factors g_n for n=0..D.

    Ensures strict positivity of reconstructed ρ(λ) and
    uniform convergence O(1/D). No tunable parameters.

    Convention: D = polynomial degree, N_Jackson = D+1 in the
    denominator. Follows Weisse et al., Rev. Mod. Phys. 78 (2006),
    Eq. 71 with N → D. Verify: g[0] = 1.0 exactly.
    """
    n = np.arange(D + 1, dtype=np.float64)
    Dp1 = D + 1
    g = ((Dp1 - n) * np.cos(np.pi * n / Dp1)
         + np.sin(np.pi * n / Dp1) / np.tan(np.pi / Dp1)) / Dp1
    return g
```

**3a. `density_of_states(lambda_grid)` — full spectral measure:**

Reconstructs ρ(λ) via the KPM expansion:

ρ(x) = (1 / (π√(1 - x²))) * [g₀μ₀ + 2 Σₙ₌₁ᴰ gₙμₙ Tₙ(x)]

where x = 2λ/λ_max - 1 maps λ ∈ [0, λ_max] to x ∈ [-1, 1].

The 1/(π√(1-x²)) factor is the KPM reconstruction denominator — it cancels the Chebyshev orthogonality weight from the moment definition.

The scale factor dx/dλ = 2/λ_max ensures ∫ρ(λ)dλ = 1.

Grid points are clipped to (-1+δ, 1-δ) with δ=1e-10 to avoid the boundary singularities of the Chebyshev basis.

**3b. `idos(cutoff)` — integrated density of states:**

Computes ∫₀^cutoff ρ(λ)dλ via numerical integration (trapezoidal rule) of the KPM-reconstructed density on a fine internal grid (1000 points).

Returns a fraction (normalized by dim). Multiply by `self._dim` for absolute eigenvalue count.

Positivity is enforced via `np.maximum(rho, 0.0)` — Jackson guarantees this analytically, but floating-point noise is clamped.

**3c. `spectral_density_at_zero()` — topological invariant:**

Returns `self.idos(cutoff=π * λ_max / D)` — the integrated density up to the fundamental KPM resolution limit Δλ = πλ_max/D.

This avoids the fatal trap of evaluating ρ pointwise at the Chebyshev boundary singularity x=-1 (which maps to λ=0). The 1/√(1-x²) denominator diverges at x=±1, producing artificial blowup that destroys the measurement. The integrated measure over [0, Δλ] is the correct scale-invariant topological observable.

In the thermodynamic limit (K→∞), this observable → 0 for σ≠0.5 is the topological obstruction that forbids off-line zeros.

**3d. `spectral_sum(epsilon, k)` — backward-compatible interface:**

```python
def spectral_sum(self, epsilon: float, k: int = 100) -> float:
    self.compute_moments(epsilon)
    # Integrate lambda * rho(lambda) over [0, cutoff] to approximate
    # the eigenvalue sum (not eigenvalue count). This gives the same
    # quantity as sum(smallest_eigenvalues) — a continuous energy value.
    cutoff = self._lam_max * 0.01  # bottom 1% of spectrum
    lambda_grid = np.linspace(1e-12, cutoff, 1000)
    rho = self.density_of_states(lambda_grid)
    rho = np.maximum(rho, 0.0)
    # Weight by lambda to get eigenvalue sum, not count
    return float(self._dim * np.trapz(lambda_grid * rho, lambda_grid))
```

Note: The `k` parameter is accepted for API compatibility with `BaseSheafLaplacian.spectral_sum()` but is ignored — KPM does not compute individual eigenvalues. This integrates λ·ρ(λ) (the eigenvalue-weighted density) to approximate Σλᵢ, making the output numerically comparable to the eigenvalue-based `spectral_sum` from other backends.

**3e. `smallest_eigenvalues()` — not available:**

Raises `NotImplementedError` with guidance to use `compute_moments()` + `idos()` instead. Same pattern as `HeatKernelSheafLaplacian`.

**3f. Reconstruction guard:**

All reconstruction methods (`density_of_states`, `idos`, `spectral_density_at_zero`) raise `RuntimeError("Call compute_moments(epsilon) first")` if `self._moments is None`. The `spectral_sum` method auto-calls `compute_moments`, so it always works directly.

### 4. Updated Falsification Criteria

The current `FALSIFICATION.md` is frozen per pre-registration protocol (commit 2b3f023). It is NOT modified. Instead, a new document `docs/FALSIFICATION_IDOS.md` is created to define the KPM-era criteria. It explicitly references the original as its predecessor and notes that the original eigenvalue-based criteria remain valid for K≤50 validation runs.

The new criteria track the thermodynamic scaling of the continuous spectral measure.

#### Positive Evidence (P1-P4):

| ID | Criterion | Observable | Pass |
|----|-----------|-----------|------|
| P1 | Near-kernel concentration | IDOS(Δλ, σ=0.5) | Remains finite and grows with K |
| P2 | Off-line collapse | IDOS(Δλ, σ≠0.5) | Monotonically decreases toward 0 as K increases |
| P3 | Contrast divergence | IDOS(σ=0.5) / IDOS(σ≠0.5) | Ratio grows unboundedly with K |
| P4 | Moment scaling | Raw μₙ(σ=0.5) vs μₙ(σ≠0.5) | Decay rate of μₙ is faster off-line, slower on-line |

Where Δλ = πλ_max/D is the KPM resolution limit.

#### Falsification Criteria (F1-F4):

| ID | Criterion | Falsifies | Trigger |
|----|-----------|-----------|---------|
| F1 | Persistent off-line density | ATFT mechanism | IDOS(Δλ, σ≠0.5) does NOT decrease with K |
| F2 | Contrast saturation | RH prediction | IDOS ratio plateaus at finite value as K→∞ |
| F3 | Symmetric collapse | Selectivity | IDOS collapses to 0 at σ=0.5 as well as off-line |
| F4 | GUE artifact | Arithmetic origin | GUE IDOS contrast ratio within 2x of zeta IDOS contrast ratio at same K |

#### Backward compatibility:
The existing K=20 discrimination ratio (670x) can be recomputed as an IDOS contrast ratio for validation. The new criteria subsume the old.

## Files Modified

- `atft/topology/torch_sheaf_laplacian.py` — Hoist `_power_iteration_lam_max` and `_rademacher_probes` from HeatKernelSheafLaplacian
- `atft/topology/heat_kernel_laplacian.py` — Replace inline lambda_max and probe code with calls to hoisted parent methods
- `atft/topology/__init__.py` — Add `KPMSheafLaplacian` to public exports (if exports exist)

## Files Created

- `atft/topology/kpm_sheaf_laplacian.py` — New `KPMSheafLaplacian` class (~250 lines)
- `tests/test_kpm.py` — Unit tests for moments, Jackson damping, reconstructors, IDOS
- `docs/FALSIFICATION_IDOS.md` — KPM-era falsification criteria (successor to frozen FALSIFICATION.md)

## Files NOT Modified

- `base_sheaf_laplacian.py` — No changes
- `sparse_sheaf_laplacian.py` — Separate backend, unrelated
- `transport_maps.py` — Transport unchanged
- `reproduce_k20.py` — Uses spectral_sum() which KPMSheafLaplacian provides
- `docs/FALSIFICATION.md` — Frozen per pre-registration protocol. NOT modified. Superseded by FALSIFICATION_IDOS.md for K≥100 runs

## Testing

**Unit tests (in `tests/test_kpm.py`):**
- Jackson coefficients: verify g_0=1, sum(g_n)>0, known values at small D
- Moment computation: small diagonal matrix with known eigenvalues, verify μₙ matches Tₙ(eigenvalues) analytically
- density_of_states: reconstruct ρ(λ) for a matrix with known spectrum, verify peaks at correct locations
- idos: integrate ρ(λ) for a matrix with 3 eigenvalues below cutoff, verify IDOS ≈ 3/dim
- spectral_density_at_zero: verify returns a finite positive value (not singularity blowup)
- spectral_sum backward compatibility: verify KPM spectral_sum produces finite, reasonable values

**Refactor validation:**
- Verify HeatKernelSheafLaplacian produces identical results after hoisting (heat_trace values unchanged)

**End-to-end:**
- `scripts/reproduce_k20.py --quick` with KPMSheafLaplacian swapped in

## Risks

- **Jackson resolution limit:** At D=300 with λ_max=100, Δλ ≈ 1.05. This may be too coarse to resolve fine structure in the near-kernel. Mitigated by allowing D up to 1000 on high-memory systems.
- **Hutchinson variance:** 30 probe vectors give ~18% relative error per moment. Mitigated by increasing num_vectors for production runs.
- **Positivity enforcement:** Jackson kernel guarantees analytical positivity, but floating-point accumulation may produce tiny negative values. Clamped via np.maximum(rho, 0.0).
