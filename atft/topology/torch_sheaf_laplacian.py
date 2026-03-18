"""PyTorch-based Sheaf Laplacian -- CUDA and ROCm GPU support.

Drop-in replacement for GPUSheafLaplacian (CuPy-based, CUDA-only).
By using PyTorch as the GPU backend, this module works identically on:
  - NVIDIA GPUs via CUDA
  - AMD GPUs via ROCm (which exposes the same torch.cuda API)
  - CPU fallback when no GPU is available

Architecture (hybrid CPU/GPU):
  - CPU: edge discovery via base class (fast, saves VRAM)
  - GPU: transport computation (batched eig), sparse assembly, Lanczos eigensolver

The key performance win over the CuPy version is GPU-accelerated transport:
the batch_transport_superposition bottleneck (np.linalg.eig on (M, K, K)
complex matrices) moves entirely to GPU via torch.linalg.eig, eliminating
the 80+ minute CPU wall for K=100.

Requires: torch (with CUDA or ROCm support for GPU acceleration).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.base_sheaf_laplacian import BaseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def _lanczos_largest(
    matvec_fn,
    dim: int,
    k: int,
    device,
    dtype,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> NDArray[np.float64]:
    """Find k largest eigenvalues of a Hermitian matrix via Lanczos iteration.

    Uses full reorthogonalization to prevent ghost eigenvalues.

    Args:
        matvec_fn: Callable that computes M @ v for a given vector v.
        dim: Dimension of the matrix.
        k: Number of largest eigenvalues to find.
        device: Torch device for computation.
        dtype: Torch dtype (must be complex, e.g. torch.cdouble).
        tol: Convergence tolerance on eigenvalue residuals.
        max_iter: Maximum number of Lanczos iterations.

    Returns:
        1D numpy array of k largest eigenvalues (real, descending order).
    """
    # Lanczos iteration count: enough to capture k eigenvalues with margin
    m = min(max(2 * k + 20, k + 50), dim)

    # Initialize random starting vector
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    v = torch.randn(dim, dtype=torch.double, device=device, generator=rng)
    v = v.to(dtype)
    v = v / torch.linalg.norm(v)

    # Storage for Lanczos vectors and tridiagonal coefficients
    V = torch.zeros(m + 1, dim, dtype=dtype, device=device)
    alpha = torch.zeros(m, dtype=torch.double, device=device)
    beta = torch.zeros(m, dtype=torch.double, device=device)

    V[0] = v

    for j in range(m):
        # Matrix-vector product
        w = matvec_fn(V[j])

        # Subtract previous component
        if j > 0:
            w = w - beta[j - 1] * V[j - 1]

        # Compute diagonal element
        a_j = torch.real(torch.dot(V[j].conj(), w))
        alpha[j] = a_j
        w = w - a_j * V[j]

        # Full reorthogonalization against all previous Lanczos vectors
        # This is essential to prevent ghost eigenvalues in finite precision
        for reorth_pass in range(2):
            coeffs = torch.mv(V[:j + 1].conj(), w)
            w = w - torch.mv(V[:j + 1].T, coeffs)

        b_j = torch.linalg.norm(w).real
        if b_j < 1e-14:
            # Lanczos breakdown: invariant subspace found
            m = j + 1
            alpha = alpha[:m]
            beta = beta[:m]
            break

        beta[j] = b_j
        if j + 1 < m:
            V[j + 1] = w / b_j

    # Build tridiagonal matrix T on CPU and eigendecompose
    alpha_cpu = alpha[:m].cpu().numpy()
    beta_cpu = beta[:m - 1].cpu().numpy() if m > 1 else np.array([])

    T = np.diag(alpha_cpu)
    if len(beta_cpu) > 0:
        T += np.diag(beta_cpu, 1) + np.diag(beta_cpu, -1)

    # Eigendecompose the small tridiagonal matrix (CPU, cheap)
    ritz_values = np.sort(np.linalg.eigvalsh(T).real)

    # Return k largest
    k_actual = min(k, len(ritz_values))
    return ritz_values[-k_actual:][::-1].copy()


def lanczos_smallest(
    L_csr,
    k: int,
    dim: int,
    device,
    tol: float = 1e-6,
    max_iter: int = 300,
) -> NDArray[np.float64]:
    """Find k smallest eigenvalues of a PSD matrix L via Lanczos.

    Uses the spectral flip trick: the smallest eigenvalues of L correspond
    to the largest eigenvalues of (lambda_max * I - L). Lanczos converges
    much faster for largest eigenvalues than smallest.

    Algorithm:
      1. Estimate lambda_max via a short Lanczos run (largest eigenvalue
         converges in very few iterations).
      2. Form M = lambda_max * I - L (flips the spectrum).
      3. Run full Lanczos on M to find k largest eigenvalues mu.
      4. Return lambda_max - mu (these are the k smallest of L).

    Args:
        L_csr: Torch sparse CSR tensor (complex128, on device).
        k: Number of smallest eigenvalues to find.
        dim: Matrix dimension (L is dim x dim).
        device: Torch device.
        tol: Convergence tolerance.
        max_iter: Maximum Lanczos iterations.

    Returns:
        1D numpy array of k smallest eigenvalues (real, ascending order).
    """
    dtype = torch.cdouble

    def matvec_L(v):
        """Sparse matrix-vector product: L @ v."""
        return torch.mv(L_csr, v)

    # Step 1: Estimate lambda_max via a quick Lanczos run
    lam_max_arr = _lanczos_largest(
        matvec_L, dim, k=1, device=device, dtype=dtype, tol=1e-3, max_iter=50
    )
    lam_max = float(lam_max_arr[0]) * 1.05  # 5% safety margin

    # Ensure lam_max is positive (L is PSD)
    if lam_max < 1e-10:
        return np.zeros(k, dtype=np.float64)

    # Step 2: Define matvec for M = lam_max * I - L
    def matvec_M(v):
        """Compute (lambda_max * I - L) @ v."""
        return lam_max * v - matvec_L(v)

    # Step 3: Find k largest eigenvalues of M
    mu = _lanczos_largest(
        matvec_M, dim, k=k, device=device, dtype=dtype, tol=tol, max_iter=max_iter
    )

    # Step 4: Recover smallest eigenvalues of L
    eigs = lam_max - mu
    eigs = np.sort(eigs.real)
    eigs = np.maximum(eigs, 0.0)  # Clamp tiny negatives from numerical noise
    return eigs


class TorchSheafLaplacian(BaseSheafLaplacian):
    """PyTorch-based sheaf Laplacian -- works on CUDA and ROCm.

    Drop-in replacement for GPUSheafLaplacian. Auto-detects GPU backend.
    In PyTorch, AMD ROCm GPUs appear as torch.cuda.is_available() == True
    with torch.cuda.get_device_name() returning the AMD GPU name. The API
    is identical between NVIDIA CUDA and AMD ROCm.

    Advantages over CuPy version:
      - Works on AMD GPUs (ROCm) in addition to NVIDIA (CUDA)
      - GPU-accelerated transport computation (batched eig on GPU)
      - No CuPy dependency

    The class exposes the same interface as GPUSheafLaplacian:
      - __init__(builder, zeros, transport_mode="superposition", device=None)
      - build_matrix(epsilon) -> torch sparse CSR tensor
      - smallest_eigenvalues(epsilon, k=100) -> numpy array
      - spectral_sum(epsilon, k=100) -> float

    Args:
        builder: TransportMapBuilder instance providing K, sigma, primes,
            and batch transport methods.
        zeros: 1D array of unfolded zero positions (imaginary parts).
        transport_mode: "superposition" (default), "fe", or "resonant".
        device: Torch device string ("cuda", "cpu", etc.), or None for
            auto-detection. When None, uses CUDA if available (covers
            both NVIDIA and ROCm), otherwise falls back to CPU.
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64] | np.ndarray,
        transport_mode: str = "superposition",
        device=None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TorchSheafLaplacian. "
                "Install with: pip install torch  "
                "(for ROCm: pip install torch --index-url "
                "https://download.pytorch.org/whl/rocm6.0)"
            )

        super().__init__(builder, zeros, transport_mode)

        # Auto-detect device: CUDA covers both NVIDIA and AMD ROCm
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

    def _power_iteration_lam_max(
        self, L_csr, dim: int, n_iter: int = 30,
    ) -> float:
        """Estimate largest eigenvalue via power iteration.

        Universal SLQ primitive: required by any Chebyshev-based trace
        estimator to normalize the spectrum to [-1, 1].

        Returns lam_max with 5% safety margin.
        """
        device = self.device
        dtype = torch.cdouble

        rng = torch.Generator(device=device)
        rng.manual_seed(123)
        v = torch.randn(
            dim, dtype=torch.double, device=device, generator=rng,
        ).to(dtype)
        v = v / torch.linalg.norm(v)

        lam = torch.tensor(0.0, device=device)
        for _ in range(n_iter):
            w = torch.mv(L_csr, v)
            lam = torch.real(torch.dot(v.conj(), w))
            norm_w = torch.linalg.norm(w).real
            if norm_w < 1e-14:
                return 0.0
            v = w / norm_w

        return float(lam.cpu()) * 1.05  # 5% safety margin

    def _rademacher_probes(
        self, dim: int, num_vectors: int, seed: int = 42,
    ) -> torch.Tensor:
        """Generate Rademacher probe matrix Z in {-1, +1}^{dim x num_vectors}.

        Universal SLQ primitive: provides stochastic trace estimation
        via Hutchinson's method. Returns complex128 tensor on self.device.
        """
        device = self.device
        dtype = torch.cdouble
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        Z = (
            torch.randint(
                0, 2, (dim, num_vectors),
                device=device, dtype=torch.double, generator=rng,
            ) * 2 - 1
        ).to(dtype)
        return Z

    def gpu_transport(
        self, gaps: NDArray[np.float64]
    ) -> torch.Tensor:
        """GPU-accelerated transport via batched eigendecomposition.

        This is the key performance win over the CuPy version: the
        batch_transport_superposition bottleneck (np.linalg.eig on
        (M, K, K) complex matrices at K=100) moves entirely to GPU.

        Algorithm:
          1. Get (P, K, K) superposition bases from builder (CPU, cached)
          2. Move bases to GPU as torch tensors
          3. Compute phase matrix: phases = exp(i * gaps * log_primes)
          4. Einsum for generator batch: A = einsum('ep,pij->eij', phases, bases)
          5. Frobenius-normalize each generator
          6. GPU batched eigendecomposition: eigenvals, P = torch.linalg.eig(A)
          7. GPU batched inverse: P_inv = torch.linalg.inv(P)
          8. Compute U = P @ diag(exp(i * eigenvals)) @ P_inv on GPU
          9. Fallback to torch.matrix_exp(1j * A) for defective matrices

        Args:
            gaps: (M,) numpy array of gap values.

        Returns:
            (M, K, K) complex128 tensor on GPU -- the transport matrices.
        """
        M = len(gaps)
        K = self._K
        device = self.device
        dtype = torch.cdouble

        if M == 0:
            return torch.empty(0, K, K, dtype=dtype, device=device)

        # Step 1: Get bases from builder (CPU, cached after first call)
        bases_np = self._builder.build_superposition_bases()  # (P, K, K)
        P_count = bases_np.shape[0]

        if P_count == 0:
            return torch.eye(
                K, dtype=dtype, device=device,
            ).unsqueeze(0).expand(M, K, K).clone()

        # Ensure log_primes is available on the builder
        if self._builder._log_primes is None:
            self._builder._log_primes = np.array(
                [np.log(p) for p in self._builder.primes]
            )
        log_primes_np = self._builder._log_primes  # (P,)

        # Step 2: Move to GPU
        bases_gpu = torch.tensor(bases_np, dtype=dtype, device=device)
        gaps_gpu = torch.tensor(gaps, dtype=torch.double, device=device)
        log_primes_gpu = torch.tensor(
            log_primes_np, dtype=torch.double, device=device,
        )

        # Step 3: Phase matrix: (M, P) complex
        phases = torch.exp(
            1j * gaps_gpu[:, None] * log_primes_gpu[None, :]
        )

        # Step 4: Generator batch via einsum: (M, K, K) complex
        A_batch = torch.einsum('ep,pij->eij', phases, bases_gpu)

        # Step 5: Frobenius normalization per generator
        norms = torch.linalg.norm(A_batch.reshape(M, -1), dim=1)
        mask = norms > 0
        A_batch[mask] /= norms[mask, None, None]

        # Step 6: GPU batched eigendecomposition
        eigenvals, P_mat = torch.linalg.eig(A_batch)

        # Step 7: GPU batched inverse
        P_inv = torch.linalg.inv(P_mat)

        # Step 8: Compute U = P @ diag(exp(i * eigenvals)) @ P_inv
        exp_eigenvals = torch.exp(1j * eigenvals)
        result = torch.einsum('mik,mk,mkj->mij', P_mat, exp_eigenvals, P_inv)

        # Step 9: Check for defective matrices via condition number estimate
        P_norms = torch.linalg.norm(P_mat.reshape(M, -1), dim=1)
        P_inv_norms = torch.linalg.norm(P_inv.reshape(M, -1), dim=1)
        cond_est = P_norms * P_inv_norms
        defective = cond_est > 1e12

        n_defective = int(defective.sum().item())
        if n_defective > 0:
            defective_indices = torch.where(defective)[0]
            for idx in defective_indices:
                result[idx] = torch.matrix_exp(1j * A_batch[idx])

        return result

    def build_matrix(self, epsilon: float):
        """Assemble the N*K x N*K Laplacian as a torch sparse CSR tensor.

        Block structure per edge (i -> j) with transport U:
          L[i,j] = -U^dagger   (off-diagonal K x K block)
          L[j,i] = -U          (off-diagonal K x K block)
          L[i,i] += U^dagger U (diagonal K x K block)
          L[j,j] += I_K        (diagonal K x K block)

        Duplicate diagonal entries from different edges are automatically
        summed via torch.sparse_coo_tensor coalescing.

        Returns:
            Torch sparse CSR tensor (complex128) on self.device.
        """
        i_idx_cpu, j_idx_cpu, gaps = self.build_edge_list(epsilon)
        M = len(gaps)
        K = self._K
        dim = self._N * K
        device = self.device
        dtype = torch.cdouble

        if M == 0:
            crow = torch.zeros(dim + 1, dtype=torch.int64, device=device)
            col = torch.empty(0, dtype=torch.int64, device=device)
            vals = torch.empty(0, dtype=dtype, device=device)
            return torch.sparse_csr_tensor(crow, col, vals, size=(dim, dim))

        # 1. Compute transport matrices
        if self._transport_mode == "superposition":
            U_all = self.gpu_transport(gaps)  # (M, K, K) on GPU
        else:
            U_all_np = self._compute_transport(gaps)
            U_all = torch.tensor(U_all_np, dtype=dtype, device=device)

        U_dagger = U_all.conj().transpose(1, 2)  # (M, K, K)

        # 2. Move indices to GPU
        i_idx = torch.tensor(i_idx_cpu, dtype=torch.int64, device=device)
        j_idx = torch.tensor(j_idx_cpu, dtype=torch.int64, device=device)

        # 3. Block expansion -- build COO indices for all K x K blocks
        k_range = torch.arange(K, device=device)
        r_offset, c_offset = torch.meshgrid(k_range, k_range, indexing='ij')
        r_off = r_offset.unsqueeze(0)
        c_off = c_offset.unsqueeze(0)

        row_base_i = i_idx * K
        col_base_j = j_idx * K
        row_base_j = j_idx * K
        col_base_i = i_idx * K

        # (i, j) off-diagonal blocks: -U_dagger
        row_ij = row_base_i[:, None, None] + r_off
        col_ij = col_base_j[:, None, None] + c_off
        data_ij = -U_dagger

        # (j, i) off-diagonal blocks: -U
        row_ji = row_base_j[:, None, None] + r_off
        col_ji = col_base_i[:, None, None] + c_off
        data_ji = -U_all

        # (i, i) diagonal blocks: U_dagger @ U
        row_ii = row_base_i[:, None, None] + r_off
        col_ii = col_base_i[:, None, None] + c_off
        data_ii = torch.bmm(U_dagger, U_all)

        # (j, j) diagonal blocks: I_K
        row_jj = row_base_j[:, None, None] + r_off
        col_jj = col_base_j[:, None, None] + c_off
        I_K = torch.eye(K, dtype=dtype, device=device)
        data_jj = I_K.unsqueeze(0).expand(M, K, K).clone()

        # 4. Flatten all blocks into COO format
        all_rows = torch.cat([
            row_ij.reshape(-1), row_ji.reshape(-1),
            row_ii.reshape(-1), row_jj.reshape(-1),
        ])
        all_cols = torch.cat([
            col_ij.reshape(-1), col_ji.reshape(-1),
            col_ii.reshape(-1), col_jj.reshape(-1),
        ])
        all_data = torch.cat([
            data_ij.reshape(-1), data_ji.reshape(-1),
            data_ii.reshape(-1), data_jj.reshape(-1),
        ])

        # 5. Build COO tensor -- coalesce() sums duplicate diagonal entries
        indices = torch.stack([all_rows, all_cols])
        L_coo = torch.sparse_coo_tensor(
            indices, all_data, size=(dim, dim), dtype=dtype, device=device
        ).coalesce()

        # 6. Convert to CSR for efficient SpMV in Lanczos
        L_csr = L_coo.to_sparse_csr()

        return L_csr

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100
    ) -> NDArray[np.float64]:
        """Compute k smallest eigenvalues on GPU.

        Solver strategy:
          1. Small matrices (dim <= 500): dense eigvalsh on GPU
          2. Large matrices: Lanczos with spectral flip trick

        Falls back to CPU dense eigensolver as last resort.

        Args:
            epsilon: Rips complex scale parameter.
            k: Number of smallest eigenvalues to compute.

        Returns:
            Sorted 1D numpy array of k smallest eigenvalues (float64).
            Matches the GPUSheafLaplacian return format exactly.
        """
        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]
        device = self.device

        if dim == 0 or L_csr._nnz() == 0:
            return np.zeros(k, dtype=np.float64)

        k_actual = min(k, dim - 2) if dim > 2 else dim
        if k_actual <= 0:
            return np.zeros(k, dtype=np.float64)

        # Strategy 1: Small matrices -- dense eigensolver on GPU
        if dim <= 500:
            try:
                L_dense = L_csr.to_dense()
                eigs_t = torch.linalg.eigvalsh(L_dense)
                return self._postprocess_eigenvalues(
                    eigs_t.real.cpu().numpy(), k,
                )
            except Exception:
                pass

        eigs = None

        # Strategy 2: Lanczos with spectral flip trick on GPU
        try:
            eigs = lanczos_smallest(
                L_csr, k=k_actual, dim=dim, device=device,
                tol=1e-4, max_iter=300,
            )
        except Exception as e:
            print(f"GPU Lanczos failed: {e}")

        # Strategy 3: CPU dense fallback
        if eigs is None:
            try:
                L_dense = L_csr.to_dense().cpu().numpy()
                eigs = np.linalg.eigvalsh(L_dense)
            except Exception as e:
                print(f"CPU dense fallback failed: {e}")
                return np.zeros(k, dtype=np.float64)

        return self._postprocess_eigenvalues(eigs, k)

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Compute the sum of the smallest k eigenvalues.

        This is the primary metric: total constraint energy in the
        near-kernel of the sheaf Laplacian. Lower values indicate
        more globally consistent sections (stronger topological signal).
        """
        eigs = self.smallest_eigenvalues(epsilon, k)
        result = float(np.sum(eigs))

        # Clean up GPU memory between grid points
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result
