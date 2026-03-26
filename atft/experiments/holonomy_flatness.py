#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""Holonomy flatness test: σ=½ as the unique flat connection surface.

Theory
------
The functional-equation generators are:
  B_p(σ) = log(p) [p^{-σ} ρ(p) + p^{-(1-σ)} ρ(p)^T]

These are Hermitian iff p^{-σ} = p^{-(1-σ)}, which holds exactly at σ=½.
Hermitian generators → transport U is unitary (U†U = I).
Non-Hermitian generators → U is NOT unitary → non-unitarity defect ||U†U - I||_F > 0.

Two flatness metrics are computed:
  1. Non-unitarity defect per edge: ||U†U - I||_F  (0 at σ=½ by theory)
  2. Holonomy defect per triangle:  ||H - I||_F where H = U_ij U_jk U_ki

The non-unitarity defect has a provably sharp minimum at σ=½ (by the
Hermitian condition on B_p). The holonomy defect captures additional
curvature from non-commutativity of parallel transport around loops.

Transport mode: "fe" (functional-equation), per-prime resonant assignment.
  - Per-prime eigendecompositions computed once per σ (O(n_primes × K³)).
  - Edge transport: O(K²).  100 triangles × 11 sigma points: ~10-20s total.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.base_sheaf_laplacian import BaseSheafLaplacian

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/atft_validation")
FIG_DIR    = Path("assets/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── JTopo colour palette ───────────────────────────────────────────────────────
COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "bg":    "#0f0d08",
    "text":  "#d6d0be",
    "muted": "#817a66",
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.facecolor":   COLORS["bg"],
    "figure.facecolor": COLORS["bg"],
    "text.color":       COLORS["text"],
    "axes.labelcolor":  COLORS["text"],
    "xtick.color":      COLORS["muted"],
    "ytick.color":      COLORS["muted"],
    "axes.edgecolor":   COLORS["muted"],
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.dpi":      200,
})

# ── Experiment parameters ─────────────────────────────────────────────────────
K             = 50  # Reduced for speed; flatness is about generator Hermiticity, not fiber dim
N             = 1000
EPSILON       = 3.0
SIGMA_GRID    = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
MAX_TRIANGLES = 100   # subsampled; fe transport is O(K²) per edge after setup
RNG_SEED      = 42


# ── Minimal stub to get edge list without a full backend ──────────────────────

class _EdgeOnlyLaplacian(BaseSheafLaplacian):
    def build_matrix(self, epsilon: float):  # type: ignore[override]
        raise NotImplementedError

    def smallest_eigenvalues(self, epsilon: float, k: int = 100) -> np.ndarray:
        raise NotImplementedError


# ── Triangle finding ──────────────────────────────────────────────────────────

def find_triangles(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    max_triangles: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Find up to max_triangles triangles (a<b<c) in an edge list (i<j)."""
    edge_set: set[tuple[int, int]] = set(zip(i_idx.tolist(), j_idx.tolist()))
    neighbors: dict[int, list[int]] = defaultdict(list)
    for u, v in edge_set:
        neighbors[u].append(v)

    triangles: list[tuple[int, int, int]] = []
    vertices = list(neighbors.keys())
    rng.shuffle(vertices)

    for a in vertices:
        if len(triangles) >= max_triangles:
            break
        for b in neighbors[a]:
            if b <= a:
                continue
            for c in neighbors[b]:
                if c <= b:
                    continue
                if (a, c) in edge_set:
                    triangles.append((a, b, c))
                    if len(triangles) >= max_triangles:
                        break
            if len(triangles) >= max_triangles:
                break

    return triangles


# ── Holonomy and non-unitarity computation (fe mode) ─────────────────────────

def compute_flatness_metrics(
    zeros: np.ndarray,
    triangles: list[tuple[int, int, int]],
    sigma: float,
) -> dict:
    """Return flatness metrics over edges and triangles at this sigma.

    Non-unitarity defect per edge: ||U^dagger U - I||_F
      Uses transport_fe (resonant-prime functional-equation transport):
        G_p^FE(sigma) = log(p) [p^{-sigma} rho(p) + p^{-(1-sigma)} rho(p)^T]
      At sigma=1/2: G_p^FE is real symmetric (Hermitian) -> U is unitary ->
        defect = 0 exactly.
      At sigma!=1/2: G_p^FE is NOT Hermitian -> U is NOT unitary ->
        defect > 0, growing symmetrically away from 1/2.

    Holonomy defect per triangle: ||H - I||_F,  H = U_ij U_jk U_ki.

    Per-prime eigendecompositions are cached by TransportMapBuilder
    (computed once on the first _ensure_fe_decomps call).
    Transport_fe then uses P diag(exp(i lambda dg)) P_inv which is O(K^2).
    """
    builder = TransportMapBuilder(K=K, sigma=sigma)
    builder._ensure_fe_decomps()     # precompute per-prime eigendecompositions
    builder._ensure_prime_decomps()  # needed for resonant prime lookup

    I_K = np.eye(K, dtype=np.complex128)

    # Collect unique forward edges across all triangles
    seen_edges: dict[tuple[int, int], float] = {}
    for vi, vj, vk in triangles:
        for u, v in ((vi, vj), (vj, vk), (vi, vk)):
            if (u, v) not in seen_edges:
                seen_edges[(u, v)] = float(zeros[v] - zeros[u])

    # Non-unitarity per unique forward edge
    unitarity_defects: list[float] = []
    for gap in seen_edges.values():
        U = builder.transport_fe(gap)
        UtU = U.conj().T @ U
        unitarity_defects.append(float(np.linalg.norm(UtU - I_K, ord="fro")))

    # Holonomy per triangle (uses cached fe decomps for all three legs)
    holonomy_defects = np.empty(len(triangles))
    for t, (vi, vj, vk) in enumerate(triangles):
        U_ij = builder.transport_fe(zeros[vj] - zeros[vi])
        U_jk = builder.transport_fe(zeros[vk] - zeros[vj])
        U_ki = builder.transport_fe(zeros[vi] - zeros[vk])   # reversed gap
        H = U_ij @ U_jk @ U_ki
        holonomy_defects[t] = float(np.linalg.norm(H - I_K, ord="fro"))

    return {
        "mean_unitarity_defect": float(np.mean(unitarity_defects)),
        "max_unitarity_defect":  float(np.max(unitarity_defects)),
        "mean_holonomy_defect":  float(np.mean(holonomy_defects)),
        "max_holonomy_defect":   float(np.max(holonomy_defects)),
        "n_edges":               len(seen_edges),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print("=" * 65)
    print("  ATFT Holonomy Flatness Test")
    print(f"  K={K}, N={N}, epsilon={EPSILON}")
    print(f"  sigma grid: {SIGMA_GRID}")
    print(f"  Max triangles: {MAX_TRIANGLES}")
    print("  Transport: fe (functional-equation, non-Hermitian at sigma!=0.5)")
    print("=" * 65)

    # 1. Load and unfold zeta zeros
    print("\n[1] Loading zeros...")
    src   = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = src.generate(N)
    zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    zeros = np.sort(zeros)
    print(f"    {len(zeros)} zeros, range [{zeros[0]:.3f}, {zeros[-1]:.3f}]")

    # 2. Build Rips edge list
    print(f"\n[2] Building Rips complex at epsilon={EPSILON}...")
    dummy   = TransportMapBuilder(K=K, sigma=0.5)
    stub    = _EdgeOnlyLaplacian(dummy, zeros, transport_mode="fe")
    i_idx, j_idx, _gaps = stub.build_edge_list(EPSILON)
    print(f"    {len(i_idx)} edges found.")

    # 3. Find triangles
    print(f"\n[3] Finding triangles (cap at {MAX_TRIANGLES})...")
    rng       = np.random.default_rng(RNG_SEED)
    triangles = find_triangles(i_idx, j_idx, MAX_TRIANGLES, rng)
    print(f"    Using {len(triangles)} triangles.")

    if not triangles:
        print("    ERROR: No triangles found. Increase epsilon.")
        sys.exit(1)

    # 4. Sweep sigma — capture both non-unitarity and holonomy metrics
    print("\n[4] Computing flatness metrics across sigma...")
    mean_unitarity: list[float] = []
    max_unitarity:  list[float] = []
    mean_holonomy:  list[float] = []
    max_holonomy:   list[float] = []
    per_sigma:      list[dict]  = []

    for sigma in SIGMA_GRID:
        t0      = time.time()
        metrics = compute_flatness_metrics(zeros, triangles, sigma)
        elapsed = time.time() - t0

        mean_unitarity.append(metrics["mean_unitarity_defect"])
        max_unitarity.append(metrics["max_unitarity_defect"])
        mean_holonomy.append(metrics["mean_holonomy_defect"])
        max_holonomy.append(metrics["max_holonomy_defect"])
        per_sigma.append({
            "sigma":                 sigma,
            "mean_unitarity_defect": metrics["mean_unitarity_defect"],
            "max_unitarity_defect":  metrics["max_unitarity_defect"],
            "mean_holonomy_defect":  metrics["mean_holonomy_defect"],
            "max_holonomy_defect":   metrics["max_holonomy_defect"],
            "n_edges":               metrics["n_edges"],
            "n_triangles":           len(triangles),
            "elapsed_s":             round(elapsed, 2),
        })
        print(
            f"    sigma={sigma:.2f}"
            f"  unitarity={metrics['mean_unitarity_defect']:.6f}"
            f"  holonomy={metrics['mean_holonomy_defect']:.6f}"
            f"  ({elapsed:.1f}s)"
        )
        sys.stdout.flush()

    min_u_idx      = int(np.argmin(mean_unitarity))
    min_h_idx      = int(np.argmin(mean_holonomy))
    sigma_at_min_u = SIGMA_GRID[min_u_idx]
    sigma_at_min_h = SIGMA_GRID[min_h_idx]
    print(
        f"\n    => Non-unitarity minimum: {mean_unitarity[min_u_idx]:.6f}"
        f" at sigma={sigma_at_min_u:.2f}"
    )
    print(
        f"    => Holonomy minimum:      {mean_holonomy[min_h_idx]:.6f}"
        f" at sigma={sigma_at_min_h:.2f}"
    )

    # 5. Plot — two panels
    print("\n[5] Plotting...")
    sigmas = np.array(SIGMA_GRID)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS["bg"])

    for ax in axes:
        ax.set_facecolor(COLORS["bg"])
        ax.grid(True, color=COLORS["muted"], alpha=0.25, ls=":")
        ax.tick_params(colors=COLORS["muted"])
        ax.axvline(
            0.5, color=COLORS["gold"], lw=1.8, ls="--", alpha=0.85, zorder=4,
            label="sigma=1/2 (critical line)",
        )

    # Left: non-unitarity (theoretically minimal at sigma=0.5)
    ax0 = axes[0]
    ax0.plot(
        sigmas, mean_unitarity,
        color=COLORS["teal"], lw=2.5, marker="o", markersize=7, zorder=3,
        label="Mean ||U^dagger U - I||",
    )
    ax0.fill_between(
        sigmas, mean_unitarity, max_unitarity,
        color=COLORS["teal"], alpha=0.12, zorder=1,
    )
    ax0.plot(
        sigmas, max_unitarity,
        color=COLORS["muted"], lw=1.2, ls="--", zorder=2,
        label="Max ||U^dagger U - I||",
    )
    ax0.scatter(
        [sigma_at_min_u], [mean_unitarity[min_u_idx]],
        color=COLORS["gold"], s=120, zorder=6,
        label=f"Min at sigma={sigma_at_min_u:.2f}",
    )
    ax0.set_xlabel("sigma", color=COLORS["text"])
    ax0.set_ylabel("Non-unitarity ||U^dagger U - I||_F", color=COLORS["text"])
    ax0.set_title(
        "Transport non-unitarity\n(Hermitian iff sigma=1/2  ->  min at 1/2)",
        color=COLORS["text"], pad=8,
    )
    ax0.legend(
        facecolor=COLORS["bg"], edgecolor=COLORS["muted"],
        labelcolor=COLORS["text"], fontsize=8.5,
    )

    # Right: holonomy defect
    ax1 = axes[1]
    ax1.plot(
        sigmas, mean_holonomy,
        color=COLORS["red"], lw=2.5, marker="s", markersize=7, zorder=3,
        label="Mean ||H - I||_F",
    )
    ax1.fill_between(
        sigmas, mean_holonomy, max_holonomy,
        color=COLORS["red"], alpha=0.12, zorder=1,
    )
    ax1.plot(
        sigmas, max_holonomy,
        color=COLORS["muted"], lw=1.2, ls="--", zorder=2,
        label="Max ||H - I||_F",
    )
    ax1.scatter(
        [sigma_at_min_h], [mean_holonomy[min_h_idx]],
        color=COLORS["gold"], s=120, zorder=6,
        label=f"Min at sigma={sigma_at_min_h:.2f}",
    )
    ax1.set_xlabel("sigma", color=COLORS["text"])
    ax1.set_ylabel("Holonomy ||H - I||_F", color=COLORS["text"])
    ax1.set_title(
        f"Triangle holonomy  H = U_ij U_jk U_ki\n"
        f"N={N}, K={K}, eps={EPSILON}, {len(triangles)} triangles",
        color=COLORS["text"], pad=8,
    )
    ax1.legend(
        facecolor=COLORS["bg"], edgecolor=COLORS["muted"],
        labelcolor=COLORS["text"], fontsize=8.5,
    )

    fig.suptitle(
        "ATFT gauge connection: flatness vs sigma (fe transport)",
        color=COLORS["text"], fontsize=13, y=1.01,
    )
    fig.tight_layout()

    fig_path = FIG_DIR / "holonomy_flatness.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"    Figure saved to {fig_path}")

    # 6. Save JSON
    print("\n[6] Saving results...")
    total_elapsed = time.time() - t_start
    output = {
        "experiment":             "holonomy_flatness",
        "K":                      K,
        "N":                      N,
        "epsilon":                EPSILON,
        "n_triangles":            len(triangles),
        "transport_mode":         "fe",
        "sigma_grid":             SIGMA_GRID,
        "mean_unitarity_defects": mean_unitarity,
        "max_unitarity_defects":  max_unitarity,
        "mean_holonomy_defects":  mean_holonomy,
        "max_holonomy_defects":   max_holonomy,
        "sigma_at_min_unitarity": sigma_at_min_u,
        "sigma_at_min_holonomy":  sigma_at_min_h,
        "min_mean_unitarity":     mean_unitarity[min_u_idx],
        "min_mean_holonomy":      mean_holonomy[min_h_idx],
        "total_elapsed_s":        round(total_elapsed, 1),
        "per_sigma":              per_sigma,
    }
    json_path = OUTPUT_DIR / "holonomy_flatness.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Results saved to {json_path}")

    # Summary
    print("\n" + "=" * 65)
    print("  HOLONOMY FLATNESS SUMMARY")
    print("=" * 65)
    print(f"  {'sigma':>6}  {'unitarity':>12}  {'holonomy':>12}")
    for s, u, h in zip(SIGMA_GRID, mean_unitarity, mean_holonomy):
        um = " <--U" if s == sigma_at_min_u else "     "
        hm = " <--H" if s == sigma_at_min_h else "     "
        print(f"  {s:>6.2f}  {u:>12.6f}{um}  {h:>12.6f}{hm}")
    print()
    print(
        f"  Non-unitarity minimum at sigma={sigma_at_min_u:.2f}"
        f"  (theory: sigma=0.50)"
    )
    print(f"  Holonomy minimum      at sigma={sigma_at_min_h:.2f}")
    if 0.4 <= sigma_at_min_u <= 0.6:
        verdict = "PASS -- non-unitarity min at or near sigma=0.5 (within +-0.1)"
    else:
        verdict = (
            f"NOTE -- non-unitarity min at sigma={sigma_at_min_u:.2f}, "
            "not 0.5"
        )
    print(f"  Verdict: {verdict}")
    print(f"  Total runtime: {total_elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
