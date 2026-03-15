"""Three-panel publication figure for Phase 1 results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from atft.core.types import (
    EvolutionCurveSet,
    ValidationResult,
    WaypointSignature,
)


def plot_phase1_results(
    zeta_curves: EvolutionCurveSet,
    gue_curves: list[EvolutionCurveSet],
    poisson_curves: list[EvolutionCurveSet],
    zeta_sig: WaypointSignature,
    gue_sigs: list[WaypointSignature],
    poisson_sig: WaypointSignature,
    zeta_result: ValidationResult,
    poisson_result: ValidationResult,
    save_path: Path | None = None,
    degree: int = 0,
) -> plt.Figure:
    """Generate the three-panel publication figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Betti curves
    ax = axes[0]
    _plot_envelope(ax, gue_curves, "betti", degree)
    zb = zeta_curves.betti[degree]
    ax.plot(zb.epsilon_grid, zb.values, "b-", linewidth=2, label="Zeta zeros")
    pb = poisson_curves[0].betti[degree]
    ax.plot(pb.epsilon_grid, pb.values, "r--", linewidth=1.5, label="Poisson")
    ax.set_xlabel("ε (filtration scale)")
    ax.set_ylabel("β₀(ε)")
    ax.set_title("Panel A: Betti Curves")
    ax.legend()

    # Panel B: Gini trajectories
    ax = axes[1]
    _plot_envelope(ax, gue_curves, "gini", degree)
    zg = zeta_curves.gini[degree]
    ax.plot(zg.epsilon_grid, zg.values, "b-", linewidth=2, label="Zeta zeros")
    pg = poisson_curves[0].gini[degree]
    ax.plot(pg.epsilon_grid, pg.values, "r--", linewidth=1.5, label="Poisson")
    ax.set_xlabel("ε (filtration scale)")
    ax.set_ylabel("G₀(ε)")
    ax.set_title("Panel B: Gini Trajectories")
    ax.legend()

    # Panel C: PCA of waypoint space
    ax = axes[2]
    gue_vecs = np.array([s.as_vector() for s in gue_sigs])

    # Simple 2D projection using first two principal components
    mean = np.mean(gue_vecs, axis=0)
    centered = gue_vecs - mean
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 eigenvectors (sorted ascending, so last two)
    pc = eigenvectors[:, -2:][:, ::-1]

    gue_proj = centered @ pc
    zeta_centered = zeta_sig.as_vector() - mean
    zeta_proj = zeta_centered @ pc
    poisson_centered = poisson_sig.as_vector() - mean
    poisson_proj = poisson_centered @ pc

    ax.scatter(gue_proj[:, 0], gue_proj[:, 1], c="grey", alpha=0.3, s=10, label="GUE")
    ax.scatter(zeta_proj[0], zeta_proj[1], c="blue", marker="*", s=200, label="Zeta", zorder=5)
    ax.scatter(poisson_proj[0], poisson_proj[1], c="red", marker="x", s=150, label="Poisson", zorder=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Panel C: Waypoint Space (PCA)")
    ax.legend()

    fig.suptitle(
        f"Zeta: D_M={zeta_result.mahalanobis_distance:.2f} (p={zeta_result.p_value:.4f})  |  "
        f"Poisson: D_M={poisson_result.mahalanobis_distance:.2f} (p={poisson_result.p_value:.4f})",
        y=0.02, fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def _plot_envelope(ax, ensemble_curves, attr, degree, label="99% band"):
    all_vals = []
    ref_eps = getattr(ensemble_curves[0], attr)[degree].epsilon_grid
    for c in ensemble_curves:
        curve = getattr(c, attr)[degree]
        vals = np.interp(ref_eps, curve.epsilon_grid, curve.values)
        all_vals.append(vals)
    all_vals = np.array(all_vals)
    lower = np.percentile(all_vals, 0.5, axis=0)
    upper = np.percentile(all_vals, 99.5, axis=0)
    mean = np.mean(all_vals, axis=0)
    ax.fill_between(ref_eps, lower, upper, color="grey", alpha=0.3, label=label)
    ax.plot(ref_eps, mean, "k-", linewidth=0.5, alpha=0.5)
