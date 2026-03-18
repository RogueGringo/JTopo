"""GPU-native Heat Kernel Trace Estimator for sheaf Laplacians.

Replaces the eigenvalue-based spectral sum with a heat kernel trace
estimate Tr(e^{-tL}) computed entirely on GPU using:

  1. Chebyshev polynomial expansion of e^{-tL} (sparse matvecs only)
  2. Hutchinson stochastic trace estimator (Rademacher probes)

This avoids both:
  - scipy eigsh LU factorization (CPU, blows memory at dim>200k)
  - Lanczos spectral flip (fails to resolve clustered near-zero eigs)

Cost: O(degree * nnz * num_vectors) sparse matvecs, all on GPU.
For dim=200k, nnz~10M, degree=60, num_vectors=30: ~10-30 seconds
on an A100/Blackwell.

Requires: torch (with CUDA or ROCm support for GPU acceleration).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ive as scaled_bessel_iv

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian


class HeatKernelSheafLaplacian(TorchSheafLaplacian):
    """GPU-native sheaf Laplacian using heat kernel trace estimation.

    Instead of computing discrete eigenvalues, estimates Tr(e^{-tL})
    via Chebyshev expansion + Hutchinson trace estimator. This quantity
    is a smooth spectral observable that detects the same phase
    transition at sigma=0.5 as the eigenvalue sum.

    The spectral_sum method returns dim - Tr(e^{-tL}), which is
    approximately t * sum(lambda_i) for small t and has the same
    directional semantics as the eigenvalue-based spectral sum
    (peaks where curvature is largest).

    Inherits build_matrix and gpu_transport from TorchSheafLaplacian.

    Args:
        builder: TransportMapBuilder instance.
        zeros: 1D array of unfolded zero positions.
        transport_mode: "superposition" (default), "fe", or "resonant".
        device: Torch device (None for auto-detection).
        t: Heat diffusion time. Larger t emphasizes smaller eigenvalues.
            Default 1.0 balances sensitivity vs compute cost. At K=100
            with lam_max~100, t=1.0 gives alpha=50 and ~95 Chebyshev
            iterations. t=10.0 would require ~770 iterations.
        num_vectors: Number of Rademacher probe vectors for Hutchinson
            estimator. More vectors = lower variance. Default 30 gives
            ~18% relative error on the trace.
        degree: Chebyshev polynomial degree. If None, auto-selected
            based on t and lambda_max. Capped at 300 to keep compute
            bounded. Increase t for more spectral resolution rather
            than increasing degree beyond 300.
    """

    def __init__(
        self,
        builder,
        zeros,
        transport_mode: str = "superposition",
        device=None,
        t: float = 1.0,
        num_vectors: int = 30,
        degree: int | None = None,
    ):
        super().__init__(builder, zeros, transport_mode, device)
        self._heat_t = t
        self._heat_num_vectors = num_vectors
        self._heat_degree = degree

    def heat_trace(
        self,
        epsilon: float,
        t: float | None = None,
        num_vectors: int | None = None,
        degree: int | None = None,
    ) -> float:
        """Estimate Tr(e^{-tL}) using Chebyshev expansion + Hutchinson.

        Algorithm:
          1. Build L as torch sparse CSR on GPU
          2. Estimate lambda_max via power iteration
          3. Normalize L to [-1, 1] via L_norm = (2/lam_max)L - I
          4. Compute Chebyshev coefficients for e^{-alpha*x} using
             exponentially-scaled Bessel functions (no overflow)
          5. Generate Rademacher probe matrix Z in {-1, +1}
          6. Chebyshev recurrence: accumulate c_k * T_k(L_norm) @ Z
             using only sparse-dense matmuls (L_csr @ Z)
          7. Hutchinson estimate: Tr ≈ mean(diag(Z^H @ result))

        Args:
            epsilon: Rips complex scale parameter.
            t: Heat diffusion time. Defaults to self._heat_t.
            num_vectors: Probe count. Defaults to self._heat_num_vectors.
            degree: Polynomial degree. Defaults to auto or self._heat_degree.

        Returns:
            Estimated Tr(e^{-tL}) as a float. Larger values indicate
            more eigenvalues near zero (stronger topological signal).
        """
        if t is None:
            t = self._heat_t
        if num_vectors is None:
            num_vectors = self._heat_num_vectors
        if degree is None:
            degree = self._heat_degree

        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]
        device = self.device
        dtype = torch.cdouble

        if dim == 0 or L_csr._nnz() == 0:
            return float(dim)

        # 1. Estimate lambda_max via power iteration
        lam_max = self._power_iteration_lam_max(L_csr, dim)
        if lam_max < 1e-10:
            return float(dim)  # zero matrix

        # 2. Compute alpha and auto-select degree if needed
        alpha = t * lam_max / 2.0

        if degree is None:
            degree = max(int(1.5 * alpha) + 20, 50)
            degree = min(degree, 300)

        # 3. Chebyshev coefficients using exponentially-scaled Bessel
        #    ive(k, alpha) = I_k(alpha) * e^{-alpha}
        #    This absorbs the e^{-alpha} prefactor to avoid overflow.
        #
        #    e^{-tL} v = sum_k scaled_c[k] * T_k(L_norm) v
        #    where scaled_c[0] = ive(0, alpha)
        #          scaled_c[k] = 2 * (-1)^k * ive(k, alpha)  for k >= 1
        ks = np.arange(degree + 1)
        ive_vals = scaled_bessel_iv(ks, alpha)
        scaled_coeffs = np.empty(degree + 1)
        scaled_coeffs[0] = ive_vals[0]
        scaled_coeffs[1:] = 2.0 * ((-1.0) ** ks[1:]) * ive_vals[1:]

        coeffs_gpu = torch.tensor(
            scaled_coeffs, dtype=torch.double, device=device,
        )

        # 4. Rademacher probe matrix Z in {-1, +1}^{dim x num_vectors}
        Z = self._rademacher_probes(dim, num_vectors, seed=42)

        # 5. Chebyshev recurrence: T_k(L_norm) @ Z
        #    L_norm = (2/lam_max)L - I maps spectrum from [0, lam_max] to [-1, 1]
        #    L_norm @ V = (2/lam_max)(L @ V) - V
        scale = 2.0 / lam_max

        def L_norm_mm(V):
            return scale * (L_csr @ V) - V

        T_prev = Z.clone()           # T_0 @ Z = Z
        T_curr = L_norm_mm(Z)        # T_1 @ Z = L_norm @ Z

        # Accumulate: result = sum_k coeffs[k] * T_k(L_norm) @ Z
        result = coeffs_gpu[0] * T_prev + coeffs_gpu[1] * T_curr

        for k in range(2, degree + 1):
            T_next = 2.0 * L_norm_mm(T_curr) - T_prev
            result = result + coeffs_gpu[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        # 6. Hutchinson trace estimate
        #    Tr(e^{-tL}) ≈ mean_j( z_j^H @ (e^{-tL} z_j) )
        per_vector_traces = torch.real(torch.sum(Z.conj() * result, dim=0))
        trace_estimate = float(per_vector_traces.mean().cpu())

        # GPU cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return trace_estimate

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100,
    ) -> NDArray[np.float64]:
        """Not available -- use heat_trace() instead.

        The heat kernel approach estimates Tr(e^{-tL}) rather than
        individual eigenvalues. Use spectral_sum() for a compatible
        proxy, or heat_trace() for the raw observable.
        """
        raise NotImplementedError(
            "HeatKernelSheafLaplacian does not compute individual "
            "eigenvalues. Use heat_trace(epsilon) for the raw heat "
            "kernel trace, or spectral_sum(epsilon) for a compatible "
            "proxy that works with sigma sweep scripts."
        )

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Spectral sum proxy via heat kernel trace.

        Returns dim - Tr(e^{-tL}) = sum_i (1 - e^{-t*lambda_i}).

        For small eigenvalues: (1 - e^{-t*lam}) ≈ t*lam (proportional).
        For large eigenvalues: (1 - e^{-t*lam}) ≈ 1 (saturated).

        This has the same directional semantics as the eigenvalue sum:
        larger values indicate more curvature (less consistent sections).
        Peak location, contrast ratio, and discrimination ratio are
        preserved relative to the eigenvalue-based spectral sum.
        """
        ht = self.heat_trace(epsilon)
        dim = self._N * self._K
        return float(dim) - ht
