"""Tests for KPM infrastructure and KPMSheafLaplacian."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not installed"
)


def _to_torch_csr(dense_np, device="cpu"):
    """Convert a dense numpy matrix to a torch sparse CSR tensor."""
    csr = sp.csr_matrix(dense_np.astype(np.complex128))
    return torch.sparse_csr_tensor(
        torch.tensor(csr.indptr, dtype=torch.int64, device=device),
        torch.tensor(csr.indices, dtype=torch.int64, device=device),
        torch.tensor(csr.data, dtype=torch.cdouble, device=device),
        size=csr.shape,
    )


def _graph_laplacian(n):
    """Path graph Laplacian on n vertices."""
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        L[i, i] += 1
        L[i + 1, i + 1] += 1
        L[i, i + 1] = -1
        L[i + 1, i] = -1
    return L


class TestPowerIterationHoisted:
    def test_method_exists_on_parent(self):
        assert hasattr(TorchSheafLaplacian, '_power_iteration_lam_max')

    def test_lam_max_diagonal(self):
        from atft.topology.transport_maps import TransportMapBuilder
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0, 2.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        diag = np.diag([1.0, 5.0, 9.0]).astype(np.complex128)
        L_csr = _to_torch_csr(diag)
        lam_max = lap._power_iteration_lam_max(L_csr, 3)
        assert 9.0 <= lam_max <= 10.0

class TestRademacherProbesHoisted:
    def test_method_exists_on_parent(self):
        assert hasattr(TorchSheafLaplacian, '_rademacher_probes')

    def test_shape_and_values(self):
        from atft.topology.transport_maps import TransportMapBuilder
        builder = TransportMapBuilder(K=1, sigma=0.5)
        zeros = np.array([0.0, 1.0])
        lap = TorchSheafLaplacian(builder, zeros, device="cpu")
        Z = lap._rademacher_probes(10, 5)
        assert Z.shape == (10, 5)
        real_parts = Z.real.numpy()
        assert set(np.unique(real_parts)).issubset({-1.0, 1.0})
