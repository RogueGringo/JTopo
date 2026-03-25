#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""ATFT Validation — Task 1: Novelty Test for the Sheaf Laplacian Spectral Sum.

Question: Is the 21.5% arithmetic premium S(Zeta)/S(GUE) predictable from
pair correlations alone?  If yes, the sheaf Laplacian is redundant.  If no,
it captures novel structure beyond standard random-matrix statistics.

Ground truth (K=200, sigma=0.5, eps=3.0):
  S(Zeta)  = 11.784
  S(GUE)   = 15.004
  S(Random)= 22.087
  Arithmetic premium = 21.5%

Method
------
1. Load 1000 Riemann zeta zeros (Odlyzko), apply spectral unfolding.
2. Generate GUE and Poisson controls matching the phase3d experiment.
3. Compute r₂(s), p(s), Σ²(L) for each source.
4. Predict S(Zeta) from each statistic using the reference S(GUE).
5. Measure residual |S_predicted − S_actual| / S_actual.
6. VERDICT: all predictors > 5% residual → NEW INVARIANT, else REDUNDANT.

Usage:
  .venv/bin/python -u atft/experiments/novelty_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Resolve project root regardless of cwd ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atft.analysis.pair_correlation import (
    correlation_energy,
    nearest_neighbour_distribution,
    number_variance,
    pair_correlation_function,
    predict_S_from_r2,
)
from atft.experiments.phase3d_torch_k200 import generate_gue_points
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource

# ── Ground-truth spectral sums from the K=200 sweep ────────────────────────
S_ZETA_ACTUAL = 11.784
S_GUE_ACTUAL = 15.004
S_RANDOM_ACTUAL = 22.087

# ── Plot style (JTopo palette) ──────────────────────────────────────────────
COLORS = {
    "gold":  "#c5a03f",
    "teal":  "#45a8b0",
    "red":   "#e94560",
    "bg":    "#0f0d08",
    "text":  "#d6d0be",
    "muted": "#817a66",
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "axes.facecolor":     COLORS["bg"],
    "figure.facecolor":   COLORS["bg"],
    "text.color":         COLORS["text"],
    "axes.labelcolor":    COLORS["text"],
    "xtick.color":        COLORS["text"],
    "ytick.color":        COLORS["text"],
    "axes.edgecolor":     COLORS["muted"],
    "axes.grid":          True,
    "grid.color":         COLORS["muted"],
    "grid.alpha":         0.25,
    "legend.facecolor":   "#1a1810",
    "legend.edgecolor":   COLORS["muted"],
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  COLORS["bg"],
})

# ── Output paths ────────────────────────────────────────────────────────────
FIG_DIR = ROOT / "assets" / "validation"
OUT_DIR = ROOT / "output" / "atft_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_zeta_zeros(n: int = 1000) -> np.ndarray:
    """Load and spectrally unfold the first n Odlyzko zeros."""
    data_path = ROOT / "data" / "odlyzko_zeros.txt"
    source = ZetaZerosSource(str(data_path))
    cloud = source.generate(n)
    unfolded = SpectralUnfolding(method="zeta").transform(cloud)
    return unfolded.points[:, 0]


def build_controls(zeta_zeros: np.ndarray, seed: int = 42):
    """Generate GUE and Poisson controls matching phase3d_torch_k200."""
    rng = np.random.default_rng(seed)
    mean_sp = float(np.mean(np.diff(np.sort(zeta_zeros))))
    gue_pts = generate_gue_points(
        len(zeta_zeros), mean_sp, float(zeta_zeros.min()), rng
    )
    rand_pts = np.sort(
        rng.uniform(zeta_zeros.min(), zeta_zeros.max(), len(zeta_zeros))
    )
    return gue_pts, rand_pts


def predict_from_nn_spacing(
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    S_reference: float,
    n_bins: int = 100,
    s_max: float = 4.0,
) -> float:
    """Predict S_source via nearest-neighbour spacing energy.

    Uses ∫|p(s) − p_poisson(s)|² ds as surrogate for pair-correlation energy.
    Poisson baseline: p_poisson(s) = e^{-s}.
    """
    s_c, p_src = nearest_neighbour_distribution(pts_source, n_bins, s_max)
    _, p_ref = nearest_neighbour_distribution(pts_reference, n_bins, s_max)
    ds = s_c[1] - s_c[0]
    p_poisson = np.exp(-s_c)
    E_src = float(np.sum((p_src - p_poisson) ** 2) * ds)
    E_ref = float(np.sum((p_ref - p_poisson) ** 2) * ds)
    if E_ref == 0.0:
        return S_reference
    return S_reference * (E_src / E_ref)


def predict_from_number_variance(
    pts_source: np.ndarray,
    pts_reference: np.ndarray,
    S_reference: float,
) -> float:
    """Predict S_source via integrated number-variance.

    Uses ∫Σ²(L) dL as the surrogate energy.
    """
    L_vals = np.linspace(0.5, 8.0, 30)
    _, sv2_src = number_variance(pts_source, L_vals)
    _, sv2_ref = number_variance(pts_reference, L_vals)

    # drop NaN entries (windows wider than the spectrum)
    valid = ~(np.isnan(sv2_src) | np.isnan(sv2_ref))
    if valid.sum() < 3:
        return S_reference

    dL = L_vals[1] - L_vals[0]
    E_src = float(np.sum(sv2_src[valid]) * dL)
    E_ref = float(np.sum(sv2_ref[valid]) * dL)
    if E_ref == 0.0:
        return S_reference
    return S_reference * (E_src / E_ref)


def residual(S_predicted: float, S_actual: float) -> float:
    return abs(S_predicted - S_actual) / S_actual


# ── Figure 1: r₂(s) comparison ──────────────────────────────────────────────

def plot_r2_comparison(
    s_c_z: np.ndarray, r2_z: np.ndarray,
    s_c_g: np.ndarray, r2_g: np.ndarray,
    s_c_r: np.ndarray, r2_r: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(s_c_z, r2_z, color=COLORS["gold"],  lw=2.0, label="Zeta zeros")
    ax.plot(s_c_g, r2_g, color=COLORS["teal"],  lw=2.0, label="GUE")
    ax.plot(s_c_r, r2_r, color=COLORS["red"],   lw=1.5, label="Poisson random", ls="--")
    ax.axhline(1.0, color=COLORS["muted"], lw=1.0, ls=":", label="Poisson baseline (r₂=1)")

    ax.set_xlabel("s  (in units of mean spacing)")
    ax.set_ylabel("r₂(s)  (pair correlation function)")
    ax.set_title("Pair Correlation Function r₂(s) — Zeta vs GUE vs Poisson")
    ax.set_xlim(0, 4)
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: prediction residual bar chart ─────────────────────────────────

def plot_residual_bars(
    predictor_names: list[str],
    residuals: list[float],
    out_path: Path,
) -> None:
    threshold = 0.05  # 5% — boundary between REDUNDANT and NEW INVARIANT

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(predictor_names))
    bar_colors = [
        COLORS["red"] if r < threshold else COLORS["gold"]
        for r in residuals
    ]
    bars = ax.bar(x, [r * 100 for r in residuals],
                  color=bar_colors, width=0.55, edgecolor=COLORS["muted"], linewidth=1.2)

    ax.axhline(threshold * 100, color=COLORS["teal"], lw=1.5, ls="--",
               label=f"5% threshold (REDUNDANT if below)")

    for bar, r in zip(bars, residuals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{r*100:.1f}%",
            ha="center", va="bottom", fontsize=11,
            color=COLORS["text"], fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(predictor_names, fontsize=11)
    ax.set_ylabel("Prediction residual (%)")
    ax.set_title(
        "ATFT Novelty Test — Prediction Residuals\n"
        "Gold = residual > 5% (S is novel)   |   Red = residual < 5% (S is redundant)"
    )
    ax.set_ylim(0, max(r * 100 for r in residuals) * 1.25)
    ax.legend(loc="upper right")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main experiment ──────────────────────────────────────────────────────────

def main() -> dict:
    print("=" * 70)
    print("  ATFT VALIDATION — Task 1: Novelty Test")
    print("  Question: Is S detectable from pair correlations alone?")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading zeta zeros and generating controls...")
    zeta_zeros = load_zeta_zeros(1000)
    gue_pts, rand_pts = build_controls(zeta_zeros, seed=42)

    print(f"  Zeta zeros: N={len(zeta_zeros)}, "
          f"range=[{zeta_zeros.min():.3f}, {zeta_zeros.max():.3f}]")
    print(f"  GUE:        N={len(gue_pts)}")
    print(f"  Random:     N={len(rand_pts)}")

    # ── 2. Compute r₂(s) ─────────────────────────────────────────────────
    print("\n[2] Computing pair correlation functions r₂(s)...")
    N_BINS = 100
    S_MAX = 4.0

    s_c_z, r2_z = pair_correlation_function(zeta_zeros, n_bins=N_BINS, s_max=S_MAX)
    s_c_g, r2_g = pair_correlation_function(gue_pts,    n_bins=N_BINS, s_max=S_MAX)
    s_c_r, r2_r = pair_correlation_function(rand_pts,   n_bins=N_BINS, s_max=S_MAX)
    ds = s_c_z[1] - s_c_z[0]

    # ── 3. Correlation energies ───────────────────────────────────────────
    print("\n[3] Computing correlation energies E = ∫|r₂(s)−1|² ds ...")
    E_zeta   = correlation_energy(r2_z, ds)
    E_gue    = correlation_energy(r2_g, ds)
    E_random = correlation_energy(r2_r, ds)

    print(f"  E(Zeta)   = {E_zeta:.6f}")
    print(f"  E(GUE)    = {E_gue:.6f}")
    print(f"  E(Random) = {E_random:.6f}")
    print(f"  Ratio E_zeta/E_gue = {E_zeta/E_gue:.4f}")

    # ── 4. Predictor 1: S via pair-correlation energy ─────────────────────
    print("\n[4] Predictor 1 — S predicted from r₂(s) correlation energy...")
    S_pred_r2 = predict_S_from_r2(r2_z, r2_g, S_GUE_ACTUAL, ds)
    res_r2 = residual(S_pred_r2, S_ZETA_ACTUAL)
    print(f"  S_predicted (r₂ energy)  = {S_pred_r2:.4f}   "
          f"actual = {S_ZETA_ACTUAL}   residual = {res_r2*100:.1f}%")

    # ── 5. Predictor 2: S via nearest-neighbour spacing distribution ───────
    print("\n[5] Predictor 2 — S predicted from p(s) (NN spacings)...")
    S_pred_nn = predict_from_nn_spacing(
        zeta_zeros, gue_pts, S_GUE_ACTUAL, n_bins=N_BINS, s_max=S_MAX
    )
    res_nn = residual(S_pred_nn, S_ZETA_ACTUAL)
    print(f"  S_predicted (p(s) energy) = {S_pred_nn:.4f}   "
          f"actual = {S_ZETA_ACTUAL}   residual = {res_nn*100:.1f}%")

    # ── 6. Predictor 3: S via number variance ─────────────────────────────
    print("\n[6] Predictor 3 — S predicted from Σ²(L) (number variance)...")
    S_pred_nv = predict_from_number_variance(zeta_zeros, gue_pts, S_GUE_ACTUAL)
    res_nv = residual(S_pred_nv, S_ZETA_ACTUAL)
    print(f"  S_predicted (Σ²(L))       = {S_pred_nv:.4f}   "
          f"actual = {S_ZETA_ACTUAL}   residual = {res_nv*100:.1f}%")

    # ── 7. VERDICT ────────────────────────────────────────────────────────
    THRESHOLD = 0.05  # 5%
    all_above = all(r > THRESHOLD for r in [res_r2, res_nn, res_nv])
    verdict = "NEW INVARIANT" if all_above else "REDUNDANT"
    min_residual = min(res_r2, res_nn, res_nv)

    print("\n" + "=" * 70)
    print(f"  VERDICT: {verdict}")
    print(f"  Min residual across all predictors: {min_residual*100:.1f}%")
    print(f"  Threshold: 5% — if ANY predictor achieves < 5%, verdict = REDUNDANT")
    print("=" * 70)

    # ── 8. Figures ────────────────────────────────────────────────────────
    print("\n[7] Generating figures...")
    plot_r2_comparison(
        s_c_z, r2_z,
        s_c_g, r2_g,
        s_c_r, r2_r,
        FIG_DIR / "novelty_r2_comparison.png",
    )

    predictor_names = [
        "r₂(s)\nenergy",
        "p(s) NN\nspacing",
        "Σ²(L)\nnumber var.",
    ]
    plot_residual_bars(
        predictor_names,
        [res_r2, res_nn, res_nv],
        FIG_DIR / "novelty_prediction_residual.png",
    )

    # ── 9. Save JSON results ──────────────────────────────────────────────
    results = {
        "meta": {
            "task": "ATFT Validation Task 1 — Novelty Test",
            "question": "Is S predictable from pair correlations alone?",
            "ground_truth": {
                "S_zeta_actual":  S_ZETA_ACTUAL,
                "S_gue_actual":   S_GUE_ACTUAL,
                "S_random_actual": S_RANDOM_ACTUAL,
                "arithmetic_premium_pct": round(
                    (1 - S_ZETA_ACTUAL / S_GUE_ACTUAL) * 100, 2
                ),
            },
        },
        "correlation_energies": {
            "E_zeta":   E_zeta,
            "E_gue":    E_gue,
            "E_random": E_random,
            "ratio_zeta_gue": E_zeta / E_gue,
        },
        "predictors": {
            "r2_correlation_energy": {
                "S_predicted": S_pred_r2,
                "residual_pct": res_r2 * 100,
                "above_threshold": res_r2 > THRESHOLD,
            },
            "nn_spacing_distribution": {
                "S_predicted": S_pred_nn,
                "residual_pct": res_nn * 100,
                "above_threshold": res_nn > THRESHOLD,
            },
            "number_variance": {
                "S_predicted": S_pred_nv,
                "residual_pct": res_nv * 100,
                "above_threshold": res_nv > THRESHOLD,
            },
        },
        "verdict": {
            "result":       verdict,
            "threshold_pct": THRESHOLD * 100,
            "min_residual_pct": min_residual * 100,
            "all_above_threshold": all_above,
            "interpretation": (
                "The sheaf Laplacian spectral sum S detects structure in the Riemann zeta zero "
                "distribution that is NOT captured by any of the three pair-correlation-based predictors "
                "(all-pair r₂(s), nearest-neighbour p(s), number variance Σ²(L)). "
                "The arithmetic premium of ~21.5% at sigma=0.5 is a genuinely new invariant."
            ) if all_above else (
                "At least one pair-correlation predictor achieves < 5% residual. "
                "The sheaf Laplacian spectral sum may be largely redundant with standard "
                "random-matrix statistics for this dataset."
            ),
        },
        "figures": {
            "r2_comparison": str(FIG_DIR / "novelty_r2_comparison.png"),
            "residual_bar":  str(FIG_DIR / "novelty_prediction_residual.png"),
        },
    }

    out_path = OUT_DIR / "novelty_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ── 10. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  S(Zeta) actual = {S_ZETA_ACTUAL}")
    print(f"  Arithmetic premium vs GUE = "
          f"{(1 - S_ZETA_ACTUAL/S_GUE_ACTUAL)*100:.1f}%")
    print()
    print(f"  {'Predictor':<30} {'S_predicted':>12} {'Residual':>10} {'Novel?':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*8}")
    for name, S_p, r in [
        ("r₂(s) correlation energy",   S_pred_r2, res_r2),
        ("p(s) NN spacing energy",      S_pred_nn, res_nn),
        ("Σ²(L) number variance",       S_pred_nv, res_nv),
    ]:
        flag = "YES" if r > THRESHOLD else "NO"
        print(f"  {name:<30} {S_p:>12.4f} {r*100:>9.1f}% {flag:>8}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  (min residual = {min_residual*100:.1f}%, threshold = 5%)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
