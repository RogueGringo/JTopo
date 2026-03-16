"""GPU-Accelerated Sheaf Laplacian for Massive Scale ATFT.

Hybrid CPU/GPU architecture:
  - CPU: edge discovery, batched matrix exponentials (transport)
  - GPU: sparse CSR assembly in VRAM, eigensolver (eigsh or LOBPCG)

CuPy's COO constructor automatically sums duplicate entries during tocsr(),
so overlapping diagonal blocks merge natively in hardware.

Requires: cupy, cupyx, nvidia-cuda-runtime-cu12, nvidia-cusolver-cu12, etc.
"""
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cp_splinalg
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class GPUSheafLaplacian:
    """GPU-accelerated sparse sheaf Laplacian using CuPy.

    Uses CPU for the batched non-Hermitian matrix exponentials (transport),
    and GPU for the massive sparse CSR assembly and LOBPCG eigensolver.
    """

    def __init__(self, builder, zeros, transport_mode="superposition"):
        if not GPU_AVAILABLE:
            raise ImportError("CuPy is required for GPUSheafLaplacian. Run: pip install cupy-cuda12x")

        self.builder = builder
        self.zeros = np.array(zeros)  # Keep on CPU for edge discovery
        self.N = len(zeros)
        self.K = builder.K
        self.transport_mode = transport_mode

    def build_edge_list(self, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1D edge discovery on CPU (fast and doesn't waste VRAM).

        Returns (i_idx, j_idx, gaps) with i < j convention and positive gaps.
        For N > 5000, uses binary search (O(N log N)) to avoid O(N²) memory.
        """
        zeros = self.zeros
        N = self.N

        if epsilon <= 0 or N < 2:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        if N <= 5000:
            # Pairwise approach — fine for moderate N
            diff = zeros[None, :] - zeros[:, None]  # diff[i,j] = zeros[j] - zeros[i]
            mask = (diff > 0) & (diff <= epsilon)
            i_idx, j_idx = np.where(mask)
            gaps = zeros[j_idx] - zeros[i_idx]
        else:
            # Binary search — O(N log N + |E|) for large N
            i_parts, j_parts = [], []
            for i in range(N - 1):
                j_right = int(np.searchsorted(zeros, zeros[i] + epsilon, side='right'))
                j_right = min(j_right, N)
                if i + 1 < j_right:
                    js = np.arange(i + 1, j_right, dtype=np.int64)
                    i_parts.append(np.full(len(js), i, dtype=np.int64))
                    j_parts.append(js)
            if i_parts:
                i_idx = np.concatenate(i_parts)
                j_idx = np.concatenate(j_parts)
            else:
                i_idx = np.array([], dtype=np.int64)
                j_idx = np.array([], dtype=np.int64)
            gaps = zeros[j_idx] - zeros[i_idx] if len(i_idx) > 0 else np.array([], dtype=np.float64)

        return i_idx, j_idx, gaps

    def build_matrix(self, epsilon: float) -> csp.csr_matrix:
        """Assembles the N*K x N*K Laplacian directly in VRAM."""
        i_idx_cpu, j_idx_cpu, gaps = self.build_edge_list(epsilon)
        M = len(gaps)

        if M == 0:
            return csp.csr_matrix((self.N * self.K, self.N * self.K), dtype=cp.complex128)

        # 1. CPU computes the transport matrices
        if self.transport_mode == "superposition":
            U_all_cpu = self.builder.batch_transport_superposition(gaps)
        elif self.transport_mode == "fe":
            U_all_cpu = self.builder.batch_transport_fe(gaps)
        else:
            U_all_cpu = self.builder.batch_transport_resonant(gaps)

        # 2. Move to GPU
        i_idx = cp.array(i_idx_cpu)
        j_idx = cp.array(j_idx_cpu)
        U_all = cp.array(U_all_cpu)
        U_dagger = cp.conj(cp.transpose(U_all, axes=(0, 2, 1)))

        # 3. Block Expansion (Broadcasting)
        # Base coordinates for blocks
        row_base_i = i_idx * self.K
        col_base_j = j_idx * self.K
        row_base_j = j_idx * self.K
        col_base_i = i_idx * self.K

        # Meshgrid offsets for KxK blocks
        k_range = cp.arange(self.K)
        r_offset, c_offset = cp.meshgrid(k_range, k_range, indexing='ij')
        r_off_broad = r_offset[None, :, :]
        c_off_broad = c_offset[None, :, :]

        # (i, j) Off-diagonal blocks: -U_dagger
        row_ij = row_base_i[:, None, None] + r_off_broad
        col_ij = col_base_j[:, None, None] + c_off_broad
        data_ij = -U_dagger

        # (j, i) Off-diagonal blocks: -U
        row_ji = row_base_j[:, None, None] + r_off_broad
        col_ji = col_base_i[:, None, None] + c_off_broad
        data_ji = -U_all

        # (i, i) Diagonal blocks: U_dagger @ U
        row_ii = row_base_i[:, None, None] + r_off_broad
        col_ii = col_base_i[:, None, None] + c_off_broad
        data_ii = cp.matmul(U_dagger, U_all)

        # (j, j) Diagonal blocks: Identity
        row_jj = row_base_j[:, None, None] + r_off_broad
        col_jj = col_base_j[:, None, None] + c_off_broad
        I_K = cp.eye(self.K, dtype=cp.complex128)
        data_jj = cp.broadcast_to(I_K[None, :, :], (M, self.K, self.K)).copy()

        # Flatten everything
        all_rows = cp.concatenate([row_ij.ravel(), row_ji.ravel(), row_ii.ravel(), row_jj.ravel()])
        all_cols = cp.concatenate([col_ij.ravel(), col_ji.ravel(), col_ii.ravel(), col_jj.ravel()])
        all_data = cp.concatenate([data_ij.ravel(), data_ji.ravel(), data_ii.ravel(), data_jj.ravel()])

        # Construct COO and let CuPy automatically sum duplicate diagonal entries during tocsr()
        L_csr = csp.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.N * self.K, self.N * self.K)
        ).tocsr()

        return L_csr

    def smallest_eigenvalues(self, epsilon: float, k: int = 100) -> np.ndarray:
        """Compute k smallest eigenvalues on GPU.

        Uses the spectral flip trick: finding k smallest eigenvalues of L is
        equivalent to finding k largest eigenvalues of (lambda_max*I - L) and
        subtracting from lambda_max. Lanczos converges MUCH faster for largest
        eigenvalues than smallest.

        Solver chain:
          1. Spectral flip: eigsh(lambda_max*I - L, which='LM') — fast convergence
          2. Direct eigsh(which='SA') — fallback
          3. CPU dense — last resort
        """
        L_csr = self.build_matrix(epsilon)
        dim = L_csr.shape[0]

        if dim == 0 or L_csr.nnz == 0:
            return np.zeros(k)

        k_actual = min(k, dim - 2) if dim > 2 else dim
        if k_actual <= 0:
            return np.zeros(k)

        # Small matrices: dense on CPU
        if dim <= 500:
            L_np = cp.asnumpy(L_csr.toarray())
            eigs = np.sort(np.linalg.eigvalsh(L_np).real)
            eigs = np.maximum(eigs[:k], 0.0)
            if len(eigs) < k:
                eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
            return eigs

        eigs = None

        # Strategy 1: Spectral flip — find k smallest via k largest of (lam_max*I - L)
        try:
            # Step 1: Find lambda_max (largest eigenvalue, converges in ~10 iterations)
            lam_max_arr, _ = cp_splinalg.eigsh(L_csr, k=1, which='LM', tol=1e-3)
            lam_max = float(cp.asnumpy(lam_max_arr)[0]) * 1.05  # 5% safety margin

            # Step 2: Form M = lam_max * I - L (flips the spectrum)
            I_sparse = csp.eye(dim, dtype=cp.complex128, format='csr')
            M = lam_max * I_sparse - L_csr

            # Step 3: Find k largest eigenvalues of M (fast Lanczos convergence)
            mu_arr, _ = cp_splinalg.eigsh(M, k=k_actual, which='LM', tol=1e-4)
            mu = cp.asnumpy(mu_arr).real

            # Step 4: Recover smallest eigenvalues of L
            eigs = np.sort(lam_max - mu)
            eigs = np.maximum(eigs, 0.0)
        except Exception as e:
            print(f"GPU spectral flip failed: {e}")

        # Strategy 2: Direct eigsh(SA) — slower but simpler
        if eigs is None:
            try:
                eigs_raw, _ = cp_splinalg.eigsh(L_csr, k=k_actual, which='SA', tol=1e-5)
                eigs = np.sort(cp.asnumpy(eigs_raw).real)
                eigs = np.maximum(eigs, 0.0)
            except Exception as e:
                print(f"GPU eigsh(SA) failed: {e}")

        # Strategy 3: CPU dense fallback
        if eigs is None:
            L_np = cp.asnumpy(L_csr.toarray())
            eigs = np.sort(np.linalg.eigvalsh(L_np).real)
            eigs = np.maximum(eigs[:k], 0.0)

        if len(eigs) < k:
            eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
        return eigs[:k]

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Computes the sum of the smallest k eigenvalues."""
        eigs = self.smallest_eigenvalues(epsilon, k)
        return float(np.sum(eigs))
