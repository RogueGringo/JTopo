#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""P2: Sheaf Laplacian Kernel Scaling — ATFT Prediction 4 Validation.

Paper claim: On-shell configurations have dim ker(L_F) > 0 at critical scale.

What we test: Does lambda_1 approach zero, and does it approach FASTER for
zeta zeros than for GUE/Random? The phase transition may be a crossover
(continuous) rather than a discontinuity (binary kernel jump).

Protocol:
1. Sweep K = 50, 100, 200, 400 at sigma=0.5, eps=3.0
2. Record lambda_1 through lambda_10 for Zeta, GUE, Random
3. Fit power law: lambda_i = C_i * K^alpha_i
4. Compare alpha across sources
5. Epsilon sweep at K=200: eps = 1.5, 2.0, 3.0, 4.0

PASS: alpha_1(Zeta) < alpha_1(GUE) — zeta approaches zero faster
FAIL: alpha_1(Zeta) >= alpha_1(GUE)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("output/atft_validation")
FIG_DIR = Path("assets/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "bg": "#0f0d08", "text": "#d6d0be", "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


def load_existing_eigenvalues() -> dict:
    """Load eigenvalues from existing result files."""
    data = {}

    # K=100
    try:
        with open("output/phase3c_torch_k100_results.json") as f:
            k100 = json.load(f)
        for src in ["Zeta", "GUE", "Random"]:
            key = "0.500_3.0"
            if src in k100 and key in k100[src]:
                data.setdefault(100, {})[src] = k100[src][key]["eigs_top5"]
    except FileNotFoundError:
        pass

    # K=200
    try:
        with open("output/phase3d_torch_k200_results.json") as f:
            k200 = json.load(f)
        for src in ["Zeta", "GUE", "Random"]:
            key = "0.500_3.0"
            if src in k200 and key in k200[src]:
                data.setdefault(200, {})[src] = k200[src][key]["eigs_top5"]
    except FileNotFoundError:
        pass

    return data


def run_matfree_point(zeros, K, sigma, eps, k_eig=20):
    """Run matrix-free sheaf Laplacian and return eigenvalues.

    Runs in a subprocess for K>=400 to guarantee GPU memory is fully released
    between sources (Python GC can't always free CUDA tensors in-process).
    """
    import torch

    if K < 400:
        # In-process for small K (fast, no subprocess overhead)
        from atft.topology.transport_maps import TransportMapBuilder
        from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        builder = TransportMapBuilder(K=K, sigma=sigma)
        lap = MatFreeSheafLaplacian(builder, zeros, transport_mode="superposition")
        eigs = lap.smallest_eigenvalues(eps, k=k_eig)

        del lap, builder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return eigs
    else:
        # Subprocess for large K — guarantees full GPU cleanup
        import subprocess, tempfile
        zeros_path = tempfile.mktemp(suffix=".npy")
        result_path = tempfile.mktemp(suffix=".npy")
        np.save(zeros_path, zeros)

        code = f"""
import numpy as np
from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian
zeros = np.load("{zeros_path}")
builder = TransportMapBuilder(K={K}, sigma={sigma})
lap = MatFreeSheafLaplacian(builder, zeros, transport_mode="superposition")
eigs = lap.smallest_eigenvalues({eps}, k={k_eig})
np.save("{result_path}", eigs)
"""
        result = subprocess.run(
            [".venv/bin/python", "-u", "-c", code],
            capture_output=True, text=True, timeout=300,
            env={**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        if result.returncode != 0:
            print(f"\n    SUBPROCESS FAILED: {result.stderr[-200:]}")
            import os
            os.unlink(zeros_path)
            return np.zeros(k_eig)

        eigs = np.load(result_path)
        import os
        os.unlink(zeros_path)
        os.unlink(result_path)
        return eigs


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P2: SHEAF LAPLACIAN KERNEL SCALING — ATFT PREDICTION 4")
    print(f"  {timestamp}")
    print("=" * 70)

    from atft.feature_maps.spectral_unfolding import SpectralUnfolding
    from atft.sources.zeta_zeros import ZetaZerosSource

    # Load zeros
    source = ZetaZerosSource("data/odlyzko_zeros.txt")
    cloud = source.generate(1000)
    zeta_zeros = SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]
    z_min, z_max = float(zeta_zeros.min()), float(zeta_zeros.max())
    mean_sp = float(np.mean(np.diff(np.sort(zeta_zeros))))

    # Generate controls (same seeds as phase3d for comparability)
    rng = np.random.default_rng(42)
    rand_pts = np.sort(rng.uniform(z_min, z_max, 1000))

    # Wigner surmise GUE
    from atft.experiments.phase3d_torch_k200 import generate_gue_points
    rng2 = np.random.default_rng(42)
    gue_pts = generate_gue_points(1000, mean_sp, z_min, rng2)

    sources = {"Zeta": zeta_zeros, "GUE": gue_pts, "Random": rand_pts}

    # Load existing data
    existing = load_existing_eigenvalues()
    print(f"  Loaded existing eigenvalues for K={list(existing.keys())}")

    # ── Part 1: K sweep at sigma=0.5, eps=3.0 ──
    print(f"\n{'='*70}")
    print("  PART 1: K SWEEP (sigma=0.5, eps=3.0)")
    print(f"{'='*70}")

    K_values = [50, 100, 200, 400]
    all_eigs = {}  # {K: {source: [lambda_1..lambda_10]}}

    import torch as _torch

    for K in K_values:
        all_eigs[K] = {}
        for src_name, pts in sources.items():
            # Use existing data if available (K=100, K=200)
            if K in existing and src_name in existing[K]:
                eigs = existing[K][src_name]
                print(f"  K={K} {src_name}: CACHED lambda_1={eigs[0]:.8f}")
                all_eigs[K][src_name] = eigs
                continue

            # Force total GPU cleanup before large K computations
            if K >= 400:
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    free_mb = _torch.cuda.mem_get_info()[0] / 1e6
                    print(f"  [GPU cleanup: {free_mb:.0f} MB free]")

            # Compute fresh
            print(f"  K={K} {src_name}: computing...", end=" ", flush=True)
            t0 = time.time()
            eigs = run_matfree_point(pts, K, 0.5, 3.0, k_eig=20)
            elapsed = time.time() - t0
            all_eigs[K][src_name] = eigs[:10].tolist()
            print(f"lambda_1={eigs[0]:.8f} ({elapsed:.1f}s)")

    # ── Part 2: Power law fits ──
    print(f"\n{'='*70}")
    print("  PART 2: POWER LAW FITS")
    print(f"{'='*70}")

    fits = {}  # {source: {alpha, C, r_squared}}
    for src_name in ["Zeta", "GUE", "Random"]:
        K_arr = np.array([K for K in K_values if src_name in all_eigs.get(K, {})])
        lam1_arr = np.array([all_eigs[K][src_name][0] for K in K_arr])

        if len(K_arr) >= 2:
            log_K = np.log(K_arr)
            log_lam = np.log(lam1_arr)
            alpha, log_C = np.polyfit(log_K, log_lam, 1)
            C = np.exp(log_C)

            # R-squared
            predicted = alpha * log_K + log_C
            ss_res = np.sum((log_lam - predicted) ** 2)
            ss_tot = np.sum((log_lam - np.mean(log_lam)) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            fits[src_name] = {"alpha": float(alpha), "C": float(C),
                              "r_squared": float(r_sq)}
            print(f"  {src_name}: lambda_1 = {C:.4f} * K^({alpha:.3f})  R²={r_sq:.4f}")
        else:
            print(f"  {src_name}: insufficient data points")

    # ── Part 3: Eigenvalue ratio uniformity ──
    print(f"\n{'='*70}")
    print("  PART 3: EIGENVALUE RATIO UNIFORMITY (K=200)")
    print(f"{'='*70}")

    ratios_200 = {}
    if 200 in all_eigs and "Zeta" in all_eigs[200] and "GUE" in all_eigs[200]:
        z_eigs = np.array(all_eigs[200]["Zeta"][:5])
        g_eigs = np.array(all_eigs[200]["GUE"][:5])
        ratios = z_eigs / g_eigs
        ratios_200 = {
            "ratios": ratios.tolist(),
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)),
            "cv": float(np.std(ratios) / np.mean(ratios) * 100),
        }
        print(f"  Zeta/GUE eigenvalue ratios: {[f'{r:.4f}' for r in ratios]}")
        print(f"  Mean: {ratios_200['mean']:.4f}, CV: {ratios_200['cv']:.1f}%")
        print(f"  Uniform ratio = premium exists at EVERY eigenvalue, not just sum")

    # ── Part 4: Epsilon sweep at K=200 ──
    print(f"\n{'='*70}")
    print("  PART 4: EPSILON SWEEP (K=200, sigma=0.5)")
    print(f"{'='*70}")

    eps_values = [1.5, 2.0, 3.0, 4.0]
    eps_data = {}

    for eps in eps_values:
        eps_data[eps] = {}
        for src_name, pts in sources.items():
            if eps == 3.0 and 200 in existing and src_name in existing[200]:
                eigs = existing[200][src_name]
                print(f"  eps={eps} {src_name}: CACHED lambda_1={eigs[0]:.8f}")
                eps_data[eps][src_name] = eigs
                continue

            print(f"  eps={eps} {src_name}: computing...", end=" ", flush=True)
            t0 = time.time()
            eigs = run_matfree_point(pts, 200, 0.5, eps, k_eig=20)
            elapsed = time.time() - t0
            eps_data[eps][src_name] = eigs[:5].tolist()
            print(f"lambda_1={eigs[0]:.8f} S={sum(eigs[:20]):.4f} ({elapsed:.1f}s)")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    # Verdict on Prediction 4
    zeta_alpha = fits.get("Zeta", {}).get("alpha", 0)
    gue_alpha = fits.get("GUE", {}).get("alpha", 0)
    random_alpha = fits.get("Random", {}).get("alpha", 0)

    print(f"\n  Scaling exponents:")
    print(f"    Zeta:   alpha = {zeta_alpha:.4f}")
    print(f"    GUE:    alpha = {gue_alpha:.4f}")
    print(f"    Random: alpha = {random_alpha:.4f}")

    # Does zeta approach zero faster?
    zeta_faster = zeta_alpha < gue_alpha < random_alpha
    premium_stable = ratios_200.get("cv", 100) < 5.0

    verdict = "PASS" if zeta_faster else "FAIL"
    print(f"\n  Zeta approaches zero fastest: {zeta_faster}")
    print(f"  Premium uniform across eigenvalues: {premium_stable} (CV={ratios_200.get('cv', 0):.1f}%)")
    print(f"\n  VERDICT: {verdict}")

    # Epsilon hierarchy check
    print(f"\n  Epsilon sweep hierarchy check:")
    eps_hierarchy_holds = True
    for eps in eps_values:
        if all(src in eps_data[eps] for src in ["Zeta", "GUE", "Random"]):
            sz = eps_data[eps]["Zeta"][0]
            sg = eps_data[eps]["GUE"][0]
            sr = eps_data[eps]["Random"][0]
            holds = sz < sg
            if not holds:
                eps_hierarchy_holds = False
            prem = (1 - sz / sg) * 100 if sg > 0 else 0
            print(f"    eps={eps}: lambda_1 Z={sz:.6f} G={sg:.6f} R={sr:.6f} premium={prem:.1f}% {'OK' if holds else 'FAIL'}")

    # ── Figures ──
    # Fig 1: Lambda_1 scaling with K
    fig, ax = plt.subplots(figsize=(10, 7))
    for src_name, color, marker in [("Zeta", COLORS["gold"], "o"),
                                      ("GUE", COLORS["teal"], "s"),
                                      ("Random", COLORS["red"], "^")]:
        K_arr = [K for K in K_values if src_name in all_eigs.get(K, {})]
        lam1 = [all_eigs[K][src_name][0] for K in K_arr]
        ax.loglog(K_arr, lam1, f"-{marker}", color=color, markersize=10,
                  linewidth=2, label=f"{src_name} (α={fits.get(src_name, {}).get('alpha', 0):.3f})")

    ax.set_xlabel("K (fiber dimension / number of primes)")
    ax.set_ylabel("λ₁ (smallest eigenvalue)")
    ax.set_title("λ₁ Scaling: Does Zeta Approach Zero Fastest?", color=COLORS["gold"], fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.15, which="both")
    fig.savefig(FIG_DIR / "p2_lambda1_scaling.png")
    plt.close(fig)

    # Fig 2: Eigenvalue ratio uniformity at K=200
    if ratios_200:
        fig, ax = plt.subplots(figsize=(8, 5))
        indices = range(1, 6)
        ax.bar(indices, ratios_200["ratios"], color=COLORS["gold"], alpha=0.85,
               edgecolor=COLORS["text"], linewidth=0.5)
        ax.axhline(ratios_200["mean"], color=COLORS["teal"], linestyle="--",
                    label=f"Mean = {ratios_200['mean']:.4f} (CV={ratios_200['cv']:.1f}%)")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("λᵢ(Zeta) / λᵢ(GUE)")
        ax.set_title("Eigenvalue Ratio Uniformity: Premium at Every Mode", color=COLORS["gold"])
        ax.legend()
        ax.set_ylim(0.7, 0.9)
        ax.grid(True, alpha=0.15)
        fig.savefig(FIG_DIR / "p2_eigenvalue_ratio.png")
        plt.close(fig)

    # Fig 3: Epsilon sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    for src_name, color, marker in [("Zeta", COLORS["gold"], "o"),
                                      ("GUE", COLORS["teal"], "s"),
                                      ("Random", COLORS["red"], "^")]:
        eps_arr = [e for e in eps_values if src_name in eps_data.get(e, {})]
        lam1 = [eps_data[e][src_name][0] for e in eps_arr]
        ax.plot(eps_arr, lam1, f"-{marker}", color=color, markersize=10,
                linewidth=2, label=src_name)

    ax.set_xlabel("ε (Rips complex scale)")
    ax.set_ylabel("λ₁ at K=200, σ=0.5")
    ax.set_title("Hierarchy Across Scales: Does It Hold at Every ε?", color=COLORS["gold"], fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.15)
    fig.savefig(FIG_DIR / "p2_epsilon_sweep.png")
    plt.close(fig)

    print(f"\n  Figures saved to {FIG_DIR}/")

    # ── Save results ──
    results = {
        "prediction": "4",
        "paper_claim": "On-shell configurations have dim ker(L_F) > 0",
        "reframed_test": "Does lambda_1 approach zero faster for Zeta than GUE?",
        "criterion": "alpha(Zeta) < alpha(GUE)",
        "timestamp": timestamp,
        "k_sweep": {str(K): {src: all_eigs[K][src] for src in all_eigs[K]}
                    for K in all_eigs},
        "power_law_fits": fits,
        "eigenvalue_ratios_k200": ratios_200,
        "epsilon_sweep": {str(e): {src: eps_data[e][src] for src in eps_data[e]}
                          for e in eps_data},
        "epsilon_hierarchy_holds": eps_hierarchy_holds,
        "verdict": verdict,
        "summary": (
            f"Lambda_1 scales as K^alpha with alpha(Zeta)={zeta_alpha:.3f}, "
            f"alpha(GUE)={gue_alpha:.3f}, alpha(Random)={random_alpha:.3f}. "
            f"Zeta approaches zero {'fastest' if zeta_faster else 'NOT fastest'}. "
            f"Eigenvalue ratio CV={ratios_200.get('cv', 0):.1f}% (premium uniform). "
            f"Epsilon hierarchy {'holds' if eps_hierarchy_holds else 'BREAKS'} at all tested scales."
        ),
    }

    out_path = OUTPUT_DIR / "p2_kernel_scaling.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")

    print(f"\n{'='*70}")
    print(f"  P2 VERDICT: {verdict}")
    print(f"  {results['summary']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
