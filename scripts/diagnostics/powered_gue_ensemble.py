#!/usr/bin/env python
"""Powered GUE ensemble — resumable, shardable, CPU-only.

Replaces the underpowered Phase-3e Test 2 (10 D-E GUE realizations -> p<1/11)
with an ensemble large enough to give a real nonparametric tail bound on
S(zeta) against the GUE null. Uses the CPU SparseSheafLaplacian (scipy eigsh
shift-invert) — no torch, no GPU.

Each realization is independent (own seed), so the run is embarrassingly
parallel. A worker handles the subset of indices with (i %% n_workers ==
worker_id) and appends one JSON line per realization to its OWN shard file,
so concurrent workers never contend and any worker can resume by skipping
seeds already present in its shard.

  S(zeta) is computed once (worker 0 only) and written to <out>.zeta.json.

Aggregate with analyze_powered_gue.py once shards are populated.

Usage (single worker):
  python powered_gue_ensemble.py --K 200 --n-total 200 --out output/powered_gue_K200

Usage (parallel, launched by run_powered_gue.sh):
  python powered_gue_ensemble.py --K 200 --n-total 200 --worker-id 3 --n-workers 8 \
      --out output/powered_gue_K200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.spatial.distance import pdist

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

N = 1000
START_SEED = 2000


def gue_points(n, z_min, z_max, seed):
    """Dumitriu-Edelman GUE eigenvalues, spacing-preserving unfold to [z_min, z_max].

    Identical construction to phase3e_test2_rerun.generate_proper_gue, so the
    powered ensemble is a strict extension of the original 10-draw test.
    """
    rng = np.random.default_rng(seed)
    diag = rng.standard_normal(n)
    dof = 2.0 * np.arange(n - 1, 0, -1, dtype=np.float64)
    sub = np.sqrt(rng.chisquare(dof)) / np.sqrt(2.0)
    eigs = np.sort(eigvalsh_tridiagonal(diag, sub) / np.sqrt(2.0 * n))
    spacings = np.diff(eigs)
    target_mean = (z_max - z_min) / (n - 1)
    scaled = spacings * (target_mean / spacings.mean())
    pts = np.zeros(n)
    pts[0] = z_min
    pts[1:] = z_min + np.cumsum(scaled)
    return pts


def spectral_sum(pts, K, sigma, eps, k_eig):
    builder = TransportMapBuilder(K=K, sigma=sigma)
    lap = SparseSheafLaplacian(builder, pts, transport_mode="superposition")
    eigs = lap.smallest_eigenvalues(eps, k=k_eig)
    return float(np.sum(eigs)), eigs


def load_zeta(K, sigma, eps, k_eig):
    source = ZetaZerosSource(str(_ROOT / "data" / "odlyzko_zeros.txt"))
    zeta = SpectralUnfolding(method="zeta").transform(source.generate(N)).points[:, 0]
    z_min, z_max = float(zeta.min()), float(zeta.max())
    return zeta, z_min, z_max


def done_seeds(shard_path: Path) -> set[int]:
    seen = set()
    if shard_path.exists():
        for line in shard_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(json.loads(line)["seed"]))
            except Exception:
                pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--n-total", type=int, default=200)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=3.0)
    ap.add_argument("--keig", type=int, default=20)
    ap.add_argument("--out", type=str, default="output/powered_gue_K200")
    args = ap.parse_args()

    out_base = _ROOT / args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)
    shard_path = out_base.with_name(out_base.name + f".w{args.worker_id}.jsonl")

    zeta, z_min, z_max = load_zeta(args.K, args.sigma, args.eps, args.keig)

    # Worker 0 computes and records S(zeta) once.
    if args.worker_id == 0:
        zeta_path = out_base.with_name(out_base.name + ".zeta.json")
        if not zeta_path.exists():
            t0 = time.time()
            sz, ez = spectral_sum(zeta, args.K, args.sigma, args.eps, args.keig)
            n_e = int(np.sum(pdist(zeta.reshape(-1, 1)) <= args.eps))
            zeta_path.write_text(json.dumps({
                "K": args.K, "sigma": args.sigma, "eps": args.eps, "keig": args.keig,
                "S_zeta": sz, "edges": n_e, "eigs_top5": ez[:5].tolist(),
                "time_s": time.time() - t0,
            }, indent=2))
            print(f"[w0] S_zeta(K={args.K}) = {sz:.6f}  edges={n_e}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    my_indices = [i for i in range(args.n_total) if i % args.n_workers == args.worker_id]
    already = done_seeds(shard_path)
    todo = [i for i in my_indices if (START_SEED + i) not in already]
    print(f"[w{args.worker_id}] K={args.K} assigned={len(my_indices)} "
          f"done={len(already)} todo={len(todo)} -> {shard_path.name}", flush=True)

    with open(shard_path, "a", buffering=1) as fh:
        for i in todo:
            seed = START_SEED + i
            pts = gue_points(N, z_min, z_max, seed)
            n_e = int(np.sum(pdist(pts.reshape(-1, 1)) <= args.eps))
            t0 = time.time()
            s, eigs = spectral_sum(pts, args.K, args.sigma, args.eps, args.keig)
            rec = {"seed": seed, "S": s, "edges": n_e,
                   "eigs_top5": eigs[:5].tolist(), "time_s": time.time() - t0}
            fh.write(json.dumps(rec) + "\n")
            print(f"[w{args.worker_id}] seed={seed} S={s:.6f} |E|={n_e} "
                  f"({rec['time_s']:.1f}s)", flush=True)

    print(f"[w{args.worker_id}] DONE", flush=True)


if __name__ == "__main__":
    main()
