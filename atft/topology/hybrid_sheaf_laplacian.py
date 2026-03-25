"""Hybrid Sheaf Laplacian — CPU transport + GPU Lanczos.

Solves the K=800 OOM problem:
  - MatFreeSheafLaplacian tries to cache all transport on GPU.
    At K=800: 2492 edges × 800×800 × 16 bytes = 25.5 GB > 12 GB RTX 5070.
  - HybridSheafLaplacian moves transport to CPU RAM (64 GB available),
    transfers one small batch to GPU per Lanczos matvec iteration.

Memory profile at K=800, N=1000, eps=3.0:
  - Transport (CPU RAM):   M × K² × 16 ≈ 2492 × 640000 × 16 ≈ 25.5 GB
  - Lanczos vectors (GPU): dim × 16 = 800000 × 16 ≈ 12.8 MB (tiny)
  - Per-matvec GPU peak:   batch_size × K² × 16 (configurable)

The key insight: scipy.linalg.expm handles any matrix size on CPU,
and each batch transfer (CPU numpy → GPU torch) is only a few hundred MB.
Each matvec streams through edge batches; all heavy linear algebra stays
on the GPU tensor cores via torch.bmm.

Transport computation at K=800 is slow (~10-30 min for 2492 edges):
scipy.linalg.expm on 800×800 complex matrices, called per-edge.
This is expected and acceptable. Progress is printed every 100 edges.
"""
from __future__ import annotations

import gc
import time

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from atft.topology.base_sheaf_laplacian import BaseSheafLaplacian
from atft.topology.torch_sheaf_laplacian import _lanczos_largest
from atft.topology.transport_maps import TransportMapBuilder


class HybridSheafLaplacian(BaseSheafLaplacian):
    """CPU transport + GPU Lanczos. Scales to any K that fits in system RAM.

    Drop-in replacement for MatFreeSheafLaplacian at large K where GPU VRAM
    is insufficient to cache all transport matrices.

    Transport computation uses scipy.linalg.expm (Padé approximation, CPU).
    Transport is stored in CPU RAM as a numpy (M, K, K) complex128 array.
    The Lanczos eigensolver runs on GPU; per matvec, edge batches are
    transferred CPU→GPU on-demand, then freed.

    Args:
        builder: TransportMapBuilder providing K, sigma, primes, and bases.
        zeros: 1D array of unfolded zero positions.
        transport_mode: "superposition" (default), "fe", or "resonant".
        device: Torch device for Lanczos. None = auto-detect (CUDA > CPU).
        matvec_batch_size: Number of edges per GPU batch during matvec.
            0 = auto (128 MB budget per batch).
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64] | np.ndarray,
        transport_mode: str = "superposition",
        device=None,
        matvec_batch_size: int = 0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HybridSheafLaplacian")

        super().__init__(builder, zeros, transport_mode)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._matvec_batch_size_hint = matvec_batch_size

        # Filled by _prepare()
        self._cached_epsilon: float | None = None
        self._U_cpu: np.ndarray | None = None   # (M, K, K) complex128, CPU
        self._i_idx_cpu: np.ndarray | None = None
        self._j_idx_cpu: np.ndarray | None = None
        self._M: int = 0
        self._dim: int = 0
        self._matvec_batch_size: int = 0

    # ------------------------------------------------------------------
    # Transport computation (CPU, scipy.linalg.expm)
    # ------------------------------------------------------------------

    def _compute_transport_cpu(
        self,
        gaps: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Compute all transport matrices on CPU via scipy.linalg.expm.

        For each edge e with gap gaps[e]:
          1. Build superposition generator A_e = Σ_p exp(i*gap*log(p)) * B_p
          2. Frobenius-normalize A_e
          3. U_e = expm(1j * A_e)

        This is a per-edge loop — no batching, just scipy.linalg.expm.
        At K=800 each call takes ~1-3 seconds; print progress every 100 edges.

        Returns:
            (M, K, K) complex128 numpy array stored in CPU RAM.
        """
        M = len(gaps)
        K = self._K

        if M == 0:
            return np.empty((0, K, K), dtype=np.complex128)

        if self._transport_mode != "superposition":
            # For non-superposition modes, delegate to the base class CPU path
            return self._compute_transport(gaps)

        # Get superposition bases (P, K, K) and log-primes (P,)
        bases = self._builder.build_superposition_bases()  # (P, K, K)
        P_count = bases.shape[0]

        if P_count == 0:
            return np.tile(np.eye(K, dtype=np.complex128), (M, 1, 1))

        if self._builder._log_primes is None:
            self._builder._log_primes = np.array(
                [np.log(p) for p in self._builder.primes]
            )
        log_primes = self._builder._log_primes  # (P,)

        U_cpu = np.empty((M, K, K), dtype=np.complex128)

        t_start = time.time()
        for e in range(M):
            # Phase vector: (P,)
            phases = np.exp(1j * gaps[e] * log_primes)
            # Superposition generator: (K, K)
            A = np.einsum('p,pij->ij', phases, bases)
            # Frobenius normalize
            norm = np.linalg.norm(A)
            if norm > 0:
                A = A / norm
            # Matrix exponential via Padé approximation
            U_cpu[e] = expm(1j * A)

            if (e + 1) % 100 == 0 or e == M - 1:
                elapsed = time.time() - t_start
                rate = (e + 1) / elapsed if elapsed > 0 else float('inf')
                eta = (M - e - 1) / rate if rate > 0 else 0
                mem_gb = U_cpu.nbytes / 1e9
                print(
                    f"  [HybridLap] Transport {e+1}/{M} "
                    f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
                    f"{mem_gb:.2f} GB CPU RAM)"
                )
                import sys
                sys.stdout.flush()

        return U_cpu

    # ------------------------------------------------------------------
    # Preparation (build edge list, compute all transport on CPU)
    # ------------------------------------------------------------------

    def _prepare(self, epsilon: float) -> None:
        """Build edge list and compute all transport on CPU.

        Transport is stored in self._U_cpu as numpy (M, K, K) complex128.
        This may be slow at large K — that is expected and acceptable.
        """
        if self._cached_epsilon == epsilon:
            return

        t0 = time.time()
        K = self._K
        N = self._N
        device = self.device

        # Edge discovery (CPU, fast)
        i_idx_np, j_idx_np, gaps = self.build_edge_list(epsilon)
        M = len(gaps)
        self._M = M
        self._dim = N * K
        self._i_idx_cpu = i_idx_np
        self._j_idx_cpu = j_idx_np

        if M == 0:
            self._U_cpu = np.empty((0, K, K), dtype=np.complex128)
            self._cached_epsilon = epsilon
            return

        # Determine matvec batch size: target ~128 MB per batch transfer
        if self._matvec_batch_size_hint > 0:
            self._matvec_batch_size = self._matvec_batch_size_hint
        else:
            bytes_per_edge = K * K * 16  # one complex128 K×K matrix
            target_bytes = 128 * 1024 * 1024  # 128 MB
            self._matvec_batch_size = max(1, target_bytes // bytes_per_edge)
            # Cap at M (no need for more)
            self._matvec_batch_size = min(self._matvec_batch_size, M)

        print(
            f"  [HybridLap] K={K}, N={N}, M={M} edges, dim={N*K}. "
            f"matvec_batch={self._matvec_batch_size}. "
            f"Computing transport on CPU (scipy.linalg.expm)..."
        )
        import sys
        sys.stdout.flush()

        # Compute ALL transport on CPU — the slow step at large K
        self._U_cpu = self._compute_transport_cpu(gaps)

        ram_gb = self._U_cpu.nbytes / 1e9
        elapsed = time.time() - t0
        print(
            f"  [HybridLap] Transport done: {ram_gb:.2f} GB in CPU RAM. "
            f"Total prep: {elapsed:.1f}s"
        )
        sys.stdout.flush()

        self._cached_epsilon = epsilon

    # ------------------------------------------------------------------
    # Matrix-free matvec (GPU, batch CPU→GPU transfer per batch)
    # ------------------------------------------------------------------

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute L·v: stream edge batches from CPU RAM to GPU.

        For each batch of edges:
          1. Transfer U_batch from CPU numpy → GPU torch  (fast, ~128 MB)
          2. Gather v[i], v[j] from v on GPU
          3. bmm-based Laplacian contributions
          4. Accumulate into result via index_add_
          5. Free GPU batch tensors immediately

        Args:
            v: (dim,) complex128 tensor on self.device.

        Returns:
            (dim,) complex128 tensor — L·v.
        """
        K = self._K
        M = self._M
        device = self.device
        dtype = torch.cdouble

        if M == 0:
            return torch.zeros_like(v)

        result = torch.zeros_like(v)
        v_blocks = v.view(-1, K)           # (N, K)
        result_blocks = result.view(-1, K)  # (N, K)

        batch_size = self._matvec_batch_size

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)

            # CPU→GPU transfer for this batch
            U_batch_np = self._U_cpu[start:end]  # (b, K, K) numpy
            U_batch = torch.tensor(
                U_batch_np, dtype=dtype, device=device
            )

            # Edge indices for this batch
            i_batch = torch.from_numpy(
                self._i_idx_cpu[start:end].astype(np.int64)
            ).to(device=device)
            j_batch = torch.from_numpy(
                self._j_idx_cpu[start:end].astype(np.int64)
            ).to(device=device)

            # Compute U†
            Uh = U_batch.conj().transpose(-2, -1).contiguous()

            # Gather fiber vectors: (b, K, 1)
            vi = v_blocks[i_batch].unsqueeze(-1)
            vj = v_blocks[j_batch].unsqueeze(-1)

            # Batched matmuls
            Uvi   = torch.bmm(U_batch, vi).squeeze(-1)   # (b, K)
            Uhvj  = torch.bmm(Uh, vj).squeeze(-1)         # (b, K)
            UhUvi = torch.bmm(Uh, torch.bmm(U_batch, vi)).squeeze(-1)  # (b, K)

            # Accumulate Laplacian contributions:
            #   result[i] += U†U v[i] - U† v[j]
            #   result[j] += v[j]     - U   v[i]
            result_blocks.index_add_(0, i_batch, UhUvi - Uhvj)
            result_blocks.index_add_(0, j_batch, v_blocks[j_batch] - Uvi)

            # Free GPU memory for this batch
            del U_batch, Uh, i_batch, j_batch
            del vi, vj, Uvi, Uhvj, UhUvi
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Abstract method stubs
    # ------------------------------------------------------------------

    def build_matrix(self, epsilon: float):
        """Not used — matrix-free. Returns None."""
        return None

    # ------------------------------------------------------------------
    # Eigensolver (spectral flip Lanczos, same as MatFreeSheafLaplacian)
    # ------------------------------------------------------------------

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 20,
    ) -> NDArray[np.float64]:
        """Compute k smallest eigenvalues via matrix-free Lanczos on GPU.

        Uses the spectral flip trick: find the k largest eigenvalues of
        (lambda_max · I - L) and map back to smallest of L.

        Transport lives in CPU RAM; Lanczos vectors live on GPU.
        Each matvec streams edge batches CPU→GPU.

        Args:
            epsilon: Rips complex scale parameter.
            k: Number of smallest eigenvalues to compute.

        Returns:
            Sorted 1D float64 numpy array of length k.
        """
        self._prepare(epsilon)

        dim = self._dim
        device = self.device
        dtype = torch.cdouble

        if dim == 0 or self._M == 0:
            return np.zeros(k, dtype=np.float64)

        k_actual = min(k, dim - 2) if dim > 2 else dim
        if k_actual <= 0:
            return np.zeros(k, dtype=np.float64)

        # Step 1: Estimate lambda_max via a quick Lanczos run (k=1)
        print(f"  [HybridLap] Estimating lambda_max (quick Lanczos)...")
        import sys
        sys.stdout.flush()

        lam_max_arr = _lanczos_largest(
            self.matvec, dim, k=1, device=device, dtype=dtype,
            tol=1e-3, max_iter=50,
        )
        lam_max = float(lam_max_arr[0]) * 1.05  # 5% safety margin

        if lam_max < 1e-10:
            return np.zeros(k, dtype=np.float64)

        print(f"  [HybridLap] lambda_max={lam_max:.4f}. Running full Lanczos...")
        sys.stdout.flush()

        # Step 2: Flipped matvec: (lam_max · I - L) · v
        def matvec_flipped(v):
            return lam_max * v - self.matvec(v)

        # Step 3: Find k_actual largest of flipped matrix
        mu = _lanczos_largest(
            matvec_flipped, dim, k=k_actual, device=device, dtype=dtype,
            tol=1e-4, max_iter=300,
        )

        # Step 4: Recover smallest of L
        eigs = lam_max - mu
        eigs = np.sort(eigs.real)
        eigs = np.maximum(eigs, 0.0)

        return self._postprocess_eigenvalues(eigs, k)

    def spectral_sum(self, epsilon: float, k: int = 20) -> float:
        """Sum of the k smallest eigenvalues (primary metric).

        Triggers transport computation on first call for a given epsilon.
        """
        eigs = self.smallest_eigenvalues(epsilon, k)
        result = float(np.sum(eigs))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return result
