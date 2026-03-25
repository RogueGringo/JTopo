#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""K=800 Scaling Experiment — ATFT Validation Task 2.

Computes the arithmetic premium at K=800 using HybridSheafLaplacian
(CPU transport + GPU Lanczos) to bypass the VRAM limit that OOMs
MatFreeSheafLaplacian.

Known results from previous sweeps:
  K=100: S_zeta=12.480, S_gue=15.527, premium=19.6%
  K=200: S_zeta=11.784, S_gue=15.004, premium=21.5%
  K=400: S_zeta=11.440, S_gue=14.590, premium=21.6%
  K=800: this script computes it

Settings:
  N=1000, sigma=0.5, eps=3.0, k_eig=20

WARNING: Transport computation at K=800 takes 10-30 minutes.
         The Lanczos eigensolver adds another ~10 minutes.
         Total wall time estimate: 45-60 minutes per source (Zeta + GUE).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.hybrid_sheaf_laplacian import HybridSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K = 800
N = 1000
K_EIG = 20
EPSILON = 3.0
SIGMA = 0.5

OUTPUT_DIR = Path(_PROJECT_ROOT / "output" / "atft_validation")
FIGURE_DIR = Path(_PROJECT_ROOT / "assets" / "validation")
RESULTS_PATH = OUTPUT_DIR / "k800_results.json"
FIGURE_PATH = FIGURE_DIR / "k800_scaling.png"

# Known results from prior sweeps (K=100, 200, 400)
KNOWN_SCALING = [
    {"K": 100,  "S_zeta": 12.480, "S_gue": 15.527},
    {"K": 200,  "S_zeta": 11.784, "S_gue": 15.004},
    {"K": 400,  "S_zeta": 11.440, "S_gue": 14.590},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram_status() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free, total = torch.cuda.mem_get_info()
        return f"{alloc:.2f}GB alloc, {free/1e9:.1f}GB free / {total/1e9:.1f}GB total"
    return "CPU mode"


def run_hybrid(zeros: np.ndarray, k_fiber: int, sigma: float,
               epsilon: float, k_eig: int, label: str) -> dict | None:
    """Run HybridSheafLaplacian and return result dict."""
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=k_fiber, sigma=sigma)
        lap = HybridSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(epsilon, k=k_eig)
        s = float(np.sum(eigs))
        tau = 1e-6 * np.sqrt(s) if s > 0 else 1e-10
        beta_0 = int(np.sum(eigs < tau))
        elapsed = time.time() - t0
        print(
            f"  [{label}] K={k_fiber}, sigma={sigma:.3f}: "
            f"S={s:.6f}, b0={beta_0}, elapsed={elapsed:.1f}s"
        )
        print(f"  [{label}] VRAM: {vram_status()}")
        sys.stdout.flush()
        return {
            "K": k_fiber,
            "sigma": sigma,
            "epsilon": epsilon,
            "spectral_sum": s,
            "kernel_dim": beta_0,
            "eigs_top5": eigs[:5].tolist(),
            "time_s": elapsed,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  [{label}] FAILED after {elapsed:.1f}s: {exc}")
        sys.stdout.flush()
        return None


def arithmetic_premium(s_zeta: float, s_gue: float) -> float:
    """Arithmetic premium = (S_gue - S_zeta) / S_gue * 100 %."""
    if s_gue <= 0:
        return 0.0
    return (1.0 - s_zeta / s_gue) * 100.0


def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results → {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Scaling plot (4-point: K=100, 200, 400, 800)
# ---------------------------------------------------------------------------

def make_scaling_plot(k_values, s_zeta_values, s_gue_values,
                      premiums, save_path: Path) -> None:
    """4-panel scaling figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ATFT K-Scaling: Arithmetic Premium vs Fiber Dimension",
                 fontsize=13, fontweight="bold")

    # Panel 1: Spectral sums vs K
    ax = axes[0]
    ax.plot(k_values, s_zeta_values, "b-o", linewidth=2,
            markersize=7, label="Zeta zeros")
    ax.plot(k_values, s_gue_values, "r--s", linewidth=2,
            markersize=7, label="GUE")
    ax.set_xlabel("Fiber dimension K", fontsize=11)
    ax.set_ylabel("Spectral sum S(K)", fontsize=11)
    ax.set_title("Spectral Sums vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])

    # Panel 2: Premium vs K
    ax = axes[1]
    ax.plot(k_values, premiums, "-^", linewidth=2, markersize=8,
            color="darkorange")
    ax.axhline(y=premiums[-1] if len(premiums) > 0 else 0,
               color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Fiber dimension K", fontsize=11)
    ax.set_ylabel("Arithmetic premium (%)", fontsize=11)
    ax.set_title("Arithmetic Premium = (S_GUE - S_zeta) / S_GUE")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    for i, (kv, pv) in enumerate(zip(k_values, premiums)):
        ax.annotate(f"{pv:.1f}%", (kv, pv),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=9, color="darkorange")

    # Panel 3: Ratio S_zeta / S_gue
    ratios = [sz / sg for sz, sg in zip(s_zeta_values, s_gue_values)
              if sg > 0]
    ax = axes[2]
    ax.plot(k_values[:len(ratios)], ratios, "m-D", linewidth=2, markersize=7)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1,
               label="GUE baseline (ratio=1)")
    ax.set_xlabel("Fiber dimension K", fontsize=11)
    ax.set_ylabel("S_zeta / S_GUE", fontsize=11)
    ax.set_title("Spectral Ratio (< 1 = arithmetic signal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    for i, (kv, rv) in enumerate(zip(k_values[:len(ratios)], ratios)):
        ax.annotate(f"{rv:.3f}", (kv, rv),
                    textcoords="offset points", xytext=(4, -14),
                    fontsize=9, color="purple")

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    primes = [p for p in range(2, K + 1)
              if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]

    print("=" * 72)
    print(f"  ATFT VALIDATION TASK 2: K=800 SCALING (HybridSheafLaplacian)")
    print(f"  K={K}, N={N}, sigma={SIGMA}, eps={EPSILON}, k_eig={K_EIG}")
    print(f"  Primes: {primes[:4]}...{primes[-3:]} ({len(primes)} total)")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {vram_status()}")
    else:
        print("  GPU: not available, running on CPU")
    print(f"  NOTE: CPU transport at K=800 will take ~10-30 minutes per source.")
    print("=" * 72)
    sys.stdout.flush()

    # ---- Load zeros ----
    data_path = _PROJECT_ROOT / "data" / "odlyzko_zeros.txt"
    source = ZetaZerosSource(str(data_path))
    cloud = source.generate(N)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    print(f"  Zeta zeros: N={len(zeta_zeros)}, range=[{zeta_zeros.min():.3f}, {zeta_zeros.max():.3f}]")
    sys.stdout.flush()

    # ---- Generate GUE zeros (matched mean spacing) ----
    mean_spacing = float(np.mean(np.diff(np.sort(zeta_zeros))))
    gue_source = GUESource(seed=42)
    gue_cloud = gue_source.generate(N)
    gue_eigs = SpectralUnfolding(method="semicircle").transform(gue_cloud).points[:, 0]
    # Scale GUE to match zeta mean spacing and anchor at same start
    gue_sorted = np.sort(gue_eigs)
    gue_gaps = np.diff(gue_sorted)
    gue_mean_gap = float(np.mean(gue_gaps))
    scale = mean_spacing / gue_mean_gap if gue_mean_gap > 0 else 1.0
    gue_zeros = gue_sorted * scale
    gue_zeros = gue_zeros - gue_zeros[0] + zeta_zeros[0]
    print(f"  GUE zeros: N={len(gue_zeros)}, mean_spacing={float(np.mean(np.diff(gue_zeros))):.4f}")
    sys.stdout.flush()

    # ---- Load or run K=800 ----
    results = load_results()

    def get_or_run(source_label: str, pts: np.ndarray) -> dict | None:
        key = source_label
        if key in results and results[key].get("K") == K:
            cached = results[key]
            print(f"  [{source_label}] CACHED: S={cached['spectral_sum']:.6f}")
            return cached
        print(f"\n  [{source_label}] Running K=800 hybrid Laplacian...")
        sys.stdout.flush()
        r = run_hybrid(pts, K, SIGMA, EPSILON, K_EIG, source_label)
        if r is not None:
            results[key] = r
            save_results(results)
        return r

    t_total = time.time()

    print(f"\n{'='*72}")
    print(f"  [ZETA] K=800")
    print(f"{'='*72}")
    r_zeta = get_or_run("Zeta_K800", zeta_zeros)

    print(f"\n{'='*72}")
    print(f"  [GUE] K=800")
    print(f"{'='*72}")
    r_gue = get_or_run("GUE_K800", gue_zeros)

    total_elapsed = time.time() - t_total

    # ---- Build scaling table ----
    print(f"\n{'='*72}")
    print(f"  K-SCALING TABLE (sigma={SIGMA}, eps={EPSILON})")
    print(f"{'='*72}")
    print(f"  {'K':>6}  {'S_zeta':>10}  {'S_gue':>10}  {'Premium':>10}  {'Ratio':>8}")
    print(f"  {'-'*55}")

    scaling_rows = list(KNOWN_SCALING)
    if r_zeta is not None and r_gue is not None:
        scaling_rows.append({
            "K": K,
            "S_zeta": r_zeta["spectral_sum"],
            "S_gue": r_gue["spectral_sum"],
        })

    for row in scaling_rows:
        sz = row["S_zeta"]
        sg = row["S_gue"]
        prem = arithmetic_premium(sz, sg)
        ratio = sz / sg if sg > 0 else float("nan")
        print(f"  {row['K']:>6}  {sz:>10.3f}  {sg:>10.3f}  {prem:>9.1f}%  {ratio:>8.4f}")

    print(f"\n  Total elapsed: {total_elapsed/3600:.2f}h")

    # ---- Generate the scaling figure ----
    k_vals_plot = [row["K"] for row in scaling_rows]
    s_z_plot = [row["S_zeta"] for row in scaling_rows]
    s_g_plot = [row["S_gue"] for row in scaling_rows]
    prem_plot = [arithmetic_premium(sz, sg)
                 for sz, sg in zip(s_z_plot, s_g_plot)]

    make_scaling_plot(k_vals_plot, s_z_plot, s_g_plot, prem_plot, FIGURE_PATH)

    # ---- Save complete results ----
    full_results = {
        "scaling_table": scaling_rows,
        "K800_zeta": r_zeta,
        "K800_gue": r_gue,
        "config": {
            "K": K, "N": N, "sigma": SIGMA, "epsilon": EPSILON, "k_eig": K_EIG,
        },
        "total_elapsed_s": total_elapsed,
    }
    save_results(full_results)

    # ---- Summary ----
    print(f"\n{'='*72}")
    if r_zeta is not None and r_gue is not None:
        sz = r_zeta["spectral_sum"]
        sg = r_gue["spectral_sum"]
        prem = arithmetic_premium(sz, sg)
        print(f"  K=800 RESULT: S_zeta={sz:.6f}, S_gue={sg:.6f}")
        print(f"  Arithmetic premium at K=800: {prem:.2f}%")
    elif r_zeta is not None:
        print(f"  K=800 Zeta DONE: S={r_zeta['spectral_sum']:.6f}")
        print(f"  GUE run incomplete — premium cannot be computed.")
    else:
        print("  K=800 run incomplete.")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  Figure:  {FIGURE_PATH}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
