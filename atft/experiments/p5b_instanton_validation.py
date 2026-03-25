#!/home/wb1/Desktop/Dev/JTopo/.venv/bin/python
"""P5b: BPST Instanton Discrimination — ATFT Prediction 2 Validation.

Paper claim: Parity-complete feature map distinguishes Q=+1 from Q=-1
instantons via mirror-image persistence signatures.

Protocol:
1. Generate BPST instanton configs on 12⁴ lattice at Q = 0, +1, -1, +2
2. Apply parity-complete feature map φ(x) = (s_μν, q_μν) ∈ R¹²
3. Compute H₀ persistence barcode for each Q
4. Compare: do Q=+1 and Q=-1 produce different persistence signatures?
5. Does |Q|=1 differ from |Q|=2?

PASS: Q=+1 and Q=-1 produce distinguishable persistence profiles
FAIL: Indistinguishable signatures
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp

OUTPUT_DIR = Path("output/atft_validation")
FIG_DIR = Path("assets/validation")

COLORS = {"gold": "#c5a03f", "teal": "#45a8b0", "red": "#e94560",
          "purple": "#9b59b6", "bg": "#0f0d08", "text": "#d6d0be",
          "muted": "#817a66"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.facecolor": COLORS["bg"], "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.edgecolor": COLORS["muted"], "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

LATTICE = (8, 8, 8, 8)  # 8⁴ for tractability (paper uses 12⁴)
Q_VALUES = [0, 1, -1, 2]


def gini(values):
    n = len(values)
    if n <= 1 or np.sum(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1.0) / n)


def h0_persistence_subsample(points, n_sample=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(points)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        points = points[idx]
        n = n_sample

    dists = pdist(points)
    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dists[k], i, j))
            k += 1
    edges.sort()

    bars = []
    for dist, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank_uf[ri] < rank_uf[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank_uf[ri] == rank_uf[rj]:
                rank_uf[ri] += 1
            bars.append(float(dist))

    return np.array(bars)


def main():
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print("  P5b: BPST INSTANTON DISCRIMINATION — PREDICTION 2")
    print(f"  Lattice: {LATTICE}, Q values: {Q_VALUES}")
    print(f"  {timestamp}")
    print("=" * 70)

    from atft.lattice.instanton import generate_instanton_config
    from atft.lattice.su2 import parity_complete_feature_map

    results = {}

    for Q in Q_VALUES:
        print(f"\n  Q = {Q:+d}:")
        t0 = time.time()

        # Generate config
        config = generate_instanton_config(
            lattice_shape=LATTICE,
            Q=Q,
            rho=2.5,
        )
        gen_time = time.time() - t0
        print(f"    Config generated ({gen_time:.1f}s)")

        # Feature map
        t0 = time.time()
        features = parity_complete_feature_map(config, LATTICE)
        feat_time = time.time() - t0
        print(f"    Feature map: {features.shape} ({feat_time:.1f}s)")

        # Separate s_μν (action density) and q_μν (topological charge density)
        n_pairs = features.shape[1] // 2
        s_features = features[:, :n_pairs]     # parity-even
        q_features = features[:, n_pairs:]     # parity-odd

        # Statistics
        mean_s = np.mean(np.linalg.norm(s_features, axis=1))
        mean_q = np.mean(np.linalg.norm(q_features, axis=1))
        print(f"    |s| mean = {mean_s:.6f}, |q| mean = {mean_q:.6f}")

        # H₀ persistence on full features
        bars_full = h0_persistence_subsample(features)
        bars_s = h0_persistence_subsample(s_features)
        bars_q = h0_persistence_subsample(q_features)

        g_full = gini(bars_full)
        g_s = gini(bars_s)
        g_q = gini(bars_q)

        print(f"    H₀ bars: {len(bars_full)} (full), {len(bars_s)} (s), {len(bars_q)} (q)")
        print(f"    Gini: {g_full:.4f} (full), {g_s:.4f} (s), {g_q:.4f} (q)")

        results[Q] = {
            "Q": Q,
            "gen_time": gen_time,
            "feat_time": feat_time,
            "mean_s_norm": float(mean_s),
            "mean_q_norm": float(mean_q),
            "bars_full": bars_full.tolist(),
            "bars_q": bars_q.tolist(),
            "gini_full": g_full,
            "gini_s": g_s,
            "gini_q": g_q,
            "median_bar_full": float(np.median(bars_full)) if len(bars_full) > 0 else 0,
            "median_bar_q": float(np.median(bars_q)) if len(bars_q) > 0 else 0,
        }

    # ── Analysis: Q=+1 vs Q=-1 discrimination ──
    print(f"\n{'='*70}")
    print("  ANALYSIS: INSTANTON vs ANTI-INSTANTON")
    print(f"{'='*70}")

    if 1 in results and -1 in results:
        bars_p1 = np.array(results[1]["bars_q"])
        bars_m1 = np.array(results[-1]["bars_q"])

        # KS test on q-persistence distributions
        ks_stat, ks_p = ks_2samp(bars_p1, bars_m1)
        print(f"  Q=+1 vs Q=-1 (q-features only):")
        print(f"    KS statistic = {ks_stat:.4f}, p = {ks_p:.2e}")
        print(f"    Median bar Q=+1: {results[1]['median_bar_q']:.6f}")
        print(f"    Median bar Q=-1: {results[-1]['median_bar_q']:.6f}")

        # The parity-complete feature map should distinguish them via q_μν sign
        # q_μν flips sign under Q → -Q
        mean_q_p1 = results[1]["mean_q_norm"]
        mean_q_m1 = results[-1]["mean_q_norm"]
        print(f"    |q| norm Q=+1: {mean_q_p1:.6f}")
        print(f"    |q| norm Q=-1: {mean_q_m1:.6f}")

        discriminates_pm1 = ks_p < 0.05

    # Q=0 vs Q=+1
    if 0 in results and 1 in results:
        bars_0 = np.array(results[0]["bars_full"])
        bars_1 = np.array(results[1]["bars_full"])
        ks_01, p_01 = ks_2samp(bars_0, bars_1)
        print(f"\n  Q=0 (vacuum) vs Q=+1 (instanton):")
        print(f"    KS = {ks_01:.4f}, p = {p_01:.2e}")
        discriminates_01 = p_01 < 0.05

    # |Q|=1 vs |Q|=2
    if 1 in results and 2 in results:
        bars_q1 = np.array(results[1]["bars_full"])
        bars_q2 = np.array(results[2]["bars_full"])
        ks_12, p_12 = ks_2samp(bars_q1, bars_q2)
        print(f"\n  |Q|=1 vs |Q|=2:")
        print(f"    KS = {ks_12:.4f}, p = {p_12:.2e}")
        discriminates_12 = p_12 < 0.05

    # ── Figures ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, Q in enumerate(Q_VALUES):
        ax = axes[i]
        if Q in results:
            bars = np.sort(np.array(results[Q]["bars_full"]))[::-1][:50]
            colors_bar = [COLORS["gold"] if Q > 0 else COLORS["teal"] if Q < 0
                         else COLORS["muted"] if Q == 0 else COLORS["purple"]]
            ax.barh(range(len(bars)), bars, height=0.7,
                    color=colors_bar[0], alpha=0.8)
            ax.set_title(f"Q = {Q:+d} (G={results[Q]['gini_full']:.3f})",
                        color=colors_bar[0])
        ax.set_xlabel("Persistence")
        ax.set_ylabel("Bar index")
        ax.invert_yaxis()

    fig.suptitle("P5b: BPST Instanton Persistence Barcodes",
                 color=COLORS["gold"], fontsize=15)
    fig.savefig(FIG_DIR / "p5b_instanton_barcodes.png")
    plt.close(fig)

    # ── Verdict ──
    print(f"\n{'='*70}")
    verdict_parts = []
    if discriminates_pm1:
        verdict_parts.append("Q=+1 vs Q=-1: DISCRIMINATED")
    else:
        verdict_parts.append("Q=+1 vs Q=-1: NOT DISCRIMINATED")

    if discriminates_01:
        verdict_parts.append("vacuum vs instanton: DISCRIMINATED")
    if discriminates_12:
        verdict_parts.append("|Q|=1 vs |Q|=2: DISCRIMINATED")

    overall = "PASS" if discriminates_pm1 else "FAIL"
    print(f"  P5b VERDICT: {overall}")
    for v in verdict_parts:
        print(f"    {v}")
    print(f"{'='*70}")

    # ── Save ──
    save_data = {
        "prediction": "2",
        "timestamp": timestamp,
        "lattice": list(LATTICE),
        "results": {str(Q): {k: v for k, v in r.items() if k not in ("bars_full", "bars_q")}
                    for Q, r in results.items()},
        "ks_pm1": {"stat": float(ks_stat), "p": float(ks_p)} if 1 in results and -1 in results else None,
        "verdict": overall,
    }

    out_path = OUTPUT_DIR / "p5b_instanton.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
