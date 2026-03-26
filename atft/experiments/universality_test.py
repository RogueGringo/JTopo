#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""ATFT Universality Test — Perturbation Robustness of the Arithmetic Premium.

Question: Is the ATFT spectral sum minimum at sigma=1/2 robust to perturbation
of the input zero positions, or fragile (dependent on exact arithmetic positions)?

Two tests:
  1. PERTURBATION SWEEP: Add Gaussian noise at 5 levels (0.001 to 0.5 × mean
     spacing) and measure S at sigma=0.5. If S rises sharply at tiny noise,
     the premium is fragile — the exact zero positions carry the arithmetic
     signal. If S is flat until large noise, the premium comes from aggregate
     spacing statistics (GUE-like geometry), not precise positions.

  2. STRETCH TEST: Multiply all gaps by a constant factor c in {0.5, 0.8, 1.0,
     1.2, 2.0}. Stretching preserves the ordering and relative gap structure
     but changes the absolute scale relative to the prime log-frequencies that
     drive the transport maps. Tests whether the premium is scale-invariant or
     tuned to the specific scale of unfolded zeros (mean gap = 1).

Methodology:
  K=200, N=1000, eps=3.0, sigma=0.5, transport_mode="superposition"
  MatFreeSheafLaplacian (same as main K=200 sweep, S_baseline ≈ 11.784)
  Each perturbed configuration uses a fresh builder + laplacian instance.
  Seed is fixed for reproducibility.

Output:
  Figure: assets/validation/universality_perturbation.png
  Data:   output/atft_validation/universality_test.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Resolve project root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

# ── Experiment parameters ──────────────────────────────────────────────────
K         = 200
N         = 1000
EPSILON   = 3.0
SIGMA     = 0.5
K_EIG     = 20
SEED      = 42

# Noise levels as fraction of mean spacing
NOISE_FRACS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]

# Stretch factors (multiplicative gap scaling)
STRETCH_FACTORS = [0.5, 0.8, 1.0, 1.2, 2.0]

# ── Output paths ───────────────────────────────────────────────────────────
FIG_DIR = ROOT / "assets" / "validation"
OUT_DIR = ROOT / "output" / "atft_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PATH = FIG_DIR / "universality_perturbation.png"
JSON_PATH = OUT_DIR / "universality_test.json"

# ── JTopo plot style ───────────────────────────────────────────────────────
COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "green": "#5cb85c",
    "purple":"#9b59b6",
    "bg":    "#0f0d08",
    "text":  "#d6d0be",
    "muted": "#817a66",
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "axes.facecolor":    COLORS["bg"],
    "figure.facecolor":  COLORS["bg"],
    "text.color":        COLORS["text"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["text"],
    "ytick.color":       COLORS["text"],
    "axes.edgecolor":    COLORS["muted"],
    "axes.grid":         True,
    "grid.color":        COLORS["muted"],
    "grid.alpha":        0.25,
    "legend.facecolor":  "#1a1810",
    "legend.edgecolor":  COLORS["muted"],
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": COLORS["bg"],
})


# ── Helpers ────────────────────────────────────────────────────────────────

def gpu_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_status() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free  = torch.cuda.mem_get_info()[0] / 1e9
        total = torch.cuda.mem_get_info()[1] / 1e9
        return f"{alloc:.2f}GB alloc, {free:.1f}GB free/{total:.1f}GB"
    return "CPU"


def compute_S(zeros: np.ndarray, label: str) -> dict:
    """Run MatFreeSheafLaplacian at (K=200, sigma=0.5, eps=3.0) and return result dict."""
    t0 = time.time()
    try:
        builder = TransportMapBuilder(K=K, sigma=SIGMA)
        lap = MatFreeSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(EPSILON, k=K_EIG)
        s = float(np.sum(eigs))
        elapsed = time.time() - t0
        print(f"  [{label:<32s}]  S = {s:.6f}  ({elapsed:.1f}s)  [{vram_status()}]")
        sys.stdout.flush()
        return {
            "label":        label,
            "S":            s,
            "eigs_top5":    eigs[:5].tolist(),
            "time_s":       elapsed,
            "n_points":     len(zeros),
            "ok":           True,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  [{label:<32s}]  FAILED: {exc}  ({elapsed:.1f}s)")
        sys.stdout.flush()
        return {"label": label, "S": None, "ok": False, "error": str(exc)}
    finally:
        gpu_cleanup()


def load_zeta_zeros() -> np.ndarray:
    data_path = ROOT / "data" / "odlyzko_zeros.txt"
    source = ZetaZerosSource(str(data_path))
    cloud  = source.generate(N)
    return SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]


# ── Main experiment ────────────────────────────────────────────────────────

def main() -> None:
    banner = "=" * 72
    print(banner)
    print("  ATFT Universality Test — Perturbation Robustness of the Arithmetic Premium")
    print(f"  K={K}, N={N}, eps={EPSILON}, sigma={SIGMA}, transport=superposition")
    device = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Backend: PyTorch {torch.__version__}, {device}")
    print(banner)

    rng = np.random.default_rng(SEED)

    # ── Load baseline zeros ────────────────────────────────────────────
    print("\n  Loading 1000 Odlyzko zeros and applying zeta spectral unfolding...")
    zeta_zeros   = load_zeta_zeros()
    sorted_zeros = np.sort(zeta_zeros)
    spacings     = np.diff(sorted_zeros)
    mean_spacing = float(spacings.mean())
    z_min, z_max = float(sorted_zeros[0]), float(sorted_zeros[-1])
    print(f"  Range: [{z_min:.3f}, {z_max:.3f}],  mean_spacing = {mean_spacing:.6f}")

    results: dict = {
        "params": {
            "K": K, "N": N, "epsilon": EPSILON, "sigma": SIGMA,
            "k_eig": K_EIG, "seed": SEED, "mean_spacing": mean_spacing,
        },
        "perturbation": [],
        "stretch": [],
    }

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: PERTURBATION SWEEP
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{banner}")
    print("  TEST 1: PERTURBATION SWEEP")
    print(f"  Noise levels (as fraction of mean spacing={mean_spacing:.4f}):")
    print(f"  {NOISE_FRACS}")
    print(banner)

    perturb_results: list[dict] = []

    for frac in NOISE_FRACS:
        sigma_noise = frac * mean_spacing
        if frac == 0.0:
            pts   = sorted_zeros.copy()
            label = "baseline (no noise)"
        else:
            noise = rng.normal(0.0, sigma_noise, size=N)
            pts   = np.sort(sorted_zeros + noise)
            label = f"noise_frac={frac:.3f} (sigma_abs={sigma_noise:.4f})"

        rec = compute_S(pts, label)
        rec["noise_frac"]       = frac
        rec["sigma_noise_abs"]  = sigma_noise
        perturb_results.append(rec)

    results["perturbation"] = perturb_results

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: STRETCH TEST
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{banner}")
    print("  TEST 2: STRETCH TEST")
    print(f"  Gap scale factors: {STRETCH_FACTORS}")
    print(banner)

    stretch_results: list[dict] = []

    for factor in STRETCH_FACTORS:
        # Reconstruct a stretched point set: keep z_min fixed, scale gaps
        gaps_stretched = spacings * factor
        pts = np.concatenate([[sorted_zeros[0]], sorted_zeros[0] + np.cumsum(gaps_stretched)])
        # pts is already sorted by construction
        label = f"stretch={factor:.2f}x"
        rec = compute_S(pts, label)
        rec["stretch_factor"]   = factor
        rec["new_mean_spacing"] = float(gaps_stretched.mean())
        stretch_results.append(rec)

    results["stretch"] = stretch_results

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS & VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{banner}")
    print("  ANALYSIS")
    print(banner)

    baseline_S = next(
        (r["S"] for r in perturb_results if r["noise_frac"] == 0.0 and r["ok"]),
        None,
    )
    print(f"\n  Baseline S (no noise): {baseline_S:.6f}" if baseline_S else "  Baseline S: FAILED")

    # Perturbation: find noise level that causes >10% relative increase
    print("\n  Perturbation results (relative change vs baseline):")
    print(f"  {'noise_frac':>12}  {'S':>12}  {'rel_change%':>12}  {'status':>12}")
    fragility_threshold = 0.10   # 10% rise counts as "fragile"
    fragility_frac = None
    for r in perturb_results:
        if not r["ok"] or baseline_S is None:
            print(f"  {r['noise_frac']:>12.4f}  {'FAILED':>12}")
            continue
        rel = (r["S"] - baseline_S) / baseline_S * 100 if baseline_S > 0 else 0.0
        r["rel_change_pct"] = rel
        status = "FRAGILE" if rel > fragility_threshold * 100 else "robust"
        if fragility_frac is None and rel > fragility_threshold * 100:
            fragility_frac = r["noise_frac"]
        print(f"  {r['noise_frac']:>12.4f}  {r['S']:>12.6f}  {rel:>+11.2f}%  {status:>12}")

    # Stretch: highlight how S depends on scale factor
    print("\n  Stretch results:")
    print(f"  {'factor':>8}  {'S':>12}  {'rel_change%':>14}")
    stretch_baseline_S = next(
        (r["S"] for r in stretch_results if r["stretch_factor"] == 1.0 and r["ok"]),
        baseline_S,
    )
    for r in stretch_results:
        if not r["ok"]:
            print(f"  {r['stretch_factor']:>8.2f}  {'FAILED':>12}")
            continue
        rel = (r["S"] - stretch_baseline_S) / stretch_baseline_S * 100 if stretch_baseline_S and stretch_baseline_S > 0 else 0.0
        r["rel_change_pct"] = rel
        print(f"  {r['stretch_factor']:>8.2f}  {r['S']:>12.6f}  {rel:>+13.2f}%")

    # Verdict
    print(f"\n{banner}")
    print("  VERDICT")
    print(banner)

    if fragility_frac is not None:
        results["verdict"] = "FRAGILE"
        results["fragility_frac"] = fragility_frac
        print(f"  FRAGILE — S rises >10% at noise_frac = {fragility_frac}")
        print(f"  The arithmetic premium depends on exact zero positions,")
        print(f"  not just the aggregate spacing statistics.")
        print(f"  This implies the zero VALUES (not just GUE-like spacings) carry")
        print(f"  arithmetic information detectable by the sheaf Laplacian.")
    else:
        results["verdict"] = "ROBUST"
        results["fragility_frac"] = None
        print(f"  ROBUST — S does not rise >10% at any tested noise level.")
        print(f"  The arithmetic premium survives perturbation up to {NOISE_FRACS[-1]:.0%}")
        print(f"  of the mean spacing. The premium comes from aggregate gap")
        print(f"  statistics (GUE-like geometry), not precise zero positions.")

    results["baseline_S"] = baseline_S

    # ── Save JSON ──────────────────────────────────────────────────────
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {JSON_PATH}")

    # ══════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "ATFT Universality Test — Perturbation Robustness of the Arithmetic Premium\n"
        rf"K={K}, N={N}, $\varepsilon$={EPSILON}, $\sigma$={SIGMA}",
        color=COLORS["text"], fontsize=13, y=1.02,
    )

    # ── Panel 1: Perturbation sweep ────────────────────────────────────
    ax = axes[0]
    ok_perturb = [r for r in perturb_results if r["ok"] and r["S"] is not None]
    if ok_perturb:
        xp = np.array([r["noise_frac"] for r in ok_perturb])
        yp = np.array([r["S"]          for r in ok_perturb])

        ax.plot(xp, yp, color=COLORS["gold"], linewidth=2.0, marker="o",
                markersize=7, zorder=4, label="S (perturbed zeros)")

        if baseline_S is not None:
            ax.axhline(baseline_S, color=COLORS["teal"], linewidth=1.5,
                       linestyle="--", label=f"Baseline S = {baseline_S:.3f}")
            # 10% fragility band
            ax.axhline(baseline_S * 1.10, color=COLORS["red"], linewidth=1.0,
                       linestyle=":", alpha=0.7, label="10% rise threshold")

        if fragility_frac is not None:
            ax.axvline(fragility_frac, color=COLORS["red"], linewidth=1.5,
                       linestyle="-.", alpha=0.8,
                       label=f"Fragility onset: {fragility_frac:.3f}")

        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlabel("Noise fraction (× mean spacing)")
        ax.set_ylabel("Spectral sum  S = Σ λᵢ")
        ax.set_title("Test 1: Perturbation Sweep")
        ax.legend(fontsize=10)

        # Annotate each point
        for r in ok_perturb:
            lbl = f"{r['noise_frac']:.3f}" if r["noise_frac"] > 0 else "0"
            ax.annotate(
                lbl, (r["noise_frac"], r["S"]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=8, color=COLORS["text"],
            )

    ax.set_facecolor(COLORS["bg"])

    # ── Panel 2: Stretch test ──────────────────────────────────────────
    ax2 = axes[1]
    ok_stretch = [r for r in stretch_results if r["ok"] and r["S"] is not None]
    if ok_stretch:
        xs = np.array([r["stretch_factor"] for r in ok_stretch])
        ys = np.array([r["S"]              for r in ok_stretch])

        ax2.plot(xs, ys, color=COLORS["purple"], linewidth=2.0, marker="s",
                 markersize=7, zorder=4, label="S (stretched gaps)")

        if stretch_baseline_S is not None:
            ax2.axhline(stretch_baseline_S, color=COLORS["teal"], linewidth=1.5,
                        linestyle="--", label=f"Unscaled S = {stretch_baseline_S:.3f}")

        ax2.axvline(1.0, color=COLORS["muted"], linewidth=1.0, linestyle=":",
                    alpha=0.6, label="Original scale")

        ax2.set_xlabel("Gap stretch factor")
        ax2.set_ylabel("Spectral sum  S = Σ λᵢ")
        ax2.set_title("Test 2: Gap Stretch Test")
        ax2.legend(fontsize=10)

        for r in ok_stretch:
            ax2.annotate(
                f"{r['stretch_factor']:.1f}×",
                (r["stretch_factor"], r["S"]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=8, color=COLORS["text"],
            )

    ax2.set_facecolor(COLORS["bg"])

    plt.tight_layout()

    # Verdict annotation below the title
    verdict_txt = results["verdict"]
    verdict_color = COLORS["red"] if verdict_txt == "FRAGILE" else COLORS["green"]
    fig.text(
        0.5, -0.03,
        f"Verdict: {verdict_txt}",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color=verdict_color,
    )

    plt.savefig(FIG_PATH)
    print(f"  Figure saved to {FIG_PATH}")
    print(banner)


if __name__ == "__main__":
    main()
