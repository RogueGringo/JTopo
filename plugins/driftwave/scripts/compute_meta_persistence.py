#!/usr/bin/env python3
"""Compute meta-persistence across accumulated session artifacts.

The same mathematics applied at two levels:
  Level 1: compute_persistence.py runs H₀ on file artifacts within a session
  Level 2: THIS SCRIPT runs H₀ on session artifacts across time

The barcode of barcodes. Persistence of persistence.
What ideas keep coming back? What patterns are structural?
What was the industry throwing away when they discarded speculative rejects?

Usage:
    cat /tmp/dw-artifacts/meta.json | python3 compute_meta_persistence.py

Input: MetaPersistence JSON (accumulated session history)
Output: Updated MetaPersistence with computed meta_barcode and convergence_signature
"""
from __future__ import annotations

import json
import sys
import numpy as np
from collections import Counter


def session_feature_vector(session: dict) -> np.ndarray:
    """Extract a feature vector from a session's artifact trail.

    Features:
    - Number of clusters found (L1)
    - Mean bar length (L1 persistence)
    - Gini slope (L2 trajectory direction)
    - Kernel dimension (L3 consistency)
    - Number of reprobes (routing complexity)
    - Number of speculative rejects (inference friction)
    """
    artifacts = session.get("artifacts", {})
    routing = session.get("routing_trace", [])
    rejects = session.get("speculative_rejects", [])

    # L1 features
    filtered = artifacts.get("filtered_topology", {})
    clusters = filtered.get("clusters", [])
    barcode = filtered.get("barcode", [])
    n_clusters = len(clusters)
    mean_bar = np.mean([b["death"] - b["birth"]
                        for b in barcode
                        if b.get("death", float("inf")) != float("inf")] or [0])

    # L2 features
    synthesis = artifacts.get("synthesis_map", {})
    trajectory = synthesis.get("trajectory", [])
    gini_slope = (trajectory[-1] - trajectory[0]) / max(len(trajectory), 1) if trajectory else 0

    # L3 features
    verdict = artifacts.get("sheaved_verdict", {})
    kernel_dim = verdict.get("kernel_dim", 0)

    # Routing features
    reprobes = sum(1 for r in routing if r.get("routing") == "REPROBE")
    n_rejects = len(rejects)

    return np.array([
        n_clusters,
        mean_bar,
        gini_slope,
        kernel_dim,
        reprobes,
        n_rejects
    ], dtype=np.float64)


def compute_session_distances(sessions: list[dict]) -> np.ndarray:
    """Compute pairwise distances between sessions in feature space."""
    features = np.array([session_feature_vector(s) for s in sessions])

    # Normalize
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (features - mins) / ranges

    n = len(sessions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((normed[i] - normed[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


def compute_h0_persistence(D: np.ndarray) -> list[dict]:
    """Same Union-Find as compute_persistence.py — applied at the meta level."""
    n = D.shape[0]
    parent = list(range(n))
    rank = [0] * n
    births = {i: 0.0 for i in range(n)}
    barcodes = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, eps):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        barcodes.append({
            "birth_session": str(rb),
            "death_session": str(ra),
            "birth": float(births[rb]),
            "death": float(eps),
            "dimension": 0,
            "bar_length": float(eps - births[rb])
        })
        del births[rb]

    edges = sorted((D[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    for eps, i, j in edges:
        union(i, j, eps)

    for comp in births:
        barcodes.append({
            "birth_session": str(comp),
            "death_session": "∞",
            "birth": float(births[comp]),
            "death": float("inf"),
            "dimension": 0,
            "bar_length": float("inf")
        })

    return barcodes


def compute_convergence_signature(meta: dict) -> dict:
    """Compute the topological fingerprint of the project trajectory."""
    sessions = meta.get("sessions", [])
    if not sessions:
        return {}

    # Dominant clusters: appear in >50% of sessions
    cluster_counts = Counter()
    for s in sessions:
        filtered = s.get("artifacts", {}).get("filtered_topology", {})
        for c in filtered.get("clusters", []):
            cluster_counts[c.get("label", "unknown")] += 1

    n_sessions = len(sessions)
    dominant = [label for label, count in cluster_counts.items()
                if count > n_sessions * 0.5]

    # Gini meta-trajectory: one Gini value per session
    gini_values = []
    for s in sessions:
        synthesis = s.get("artifacts", {}).get("synthesis_map", {})
        traj = synthesis.get("trajectory", [])
        if traj:
            gini_values.append(traj[-1])
        else:
            gini_values.append(0)

    # Sheaf consistency rate
    verdicts = meta.get("accumulated_verdicts", [])
    on_shell = sum(1 for v in verdicts if v.get("verdict") == "ON_SHELL")
    consistency_rate = on_shell / max(len(verdicts), 1)

    return {
        "dominant_clusters": dominant,
        "gini_meta_trajectory": gini_values,
        "sheaf_consistency_rate": consistency_rate
    }


def main():
    meta = json.load(sys.stdin)
    sessions = meta.get("sessions", [])

    if len(sessions) < 2:
        meta["meta_barcode"] = []
        meta["convergence_signature"] = compute_convergence_signature(meta)
        json.dump(meta, sys.stdout, indent=2)
        return

    # Compute distances between sessions
    D = compute_session_distances(sessions)

    # Run H₀ persistence on sessions
    barcode = compute_h0_persistence(D)

    # Label bars with session features
    for bar in barcode:
        if bar["birth_session"] != "∞":
            idx = int(bar["birth_session"])
            if idx < len(sessions):
                bar["feature"] = sessions[idx].get("prompt_summary", f"session_{idx}")

    meta["meta_barcode"] = barcode
    meta["convergence_signature"] = compute_convergence_signature(meta)

    json.dump(meta, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
