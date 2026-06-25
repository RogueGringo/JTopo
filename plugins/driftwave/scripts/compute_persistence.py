#!/usr/bin/env python3
"""Compute persistent homology on artifact distance matrices.

Same mathematics as atft/topology/ but applied to code/idea spaces
instead of zeta zero point clouds. Reads a RawCloud JSON artifact
from stdin, outputs barcode + distance matrix as JSON to stdout.

Usage:
    cat /tmp/dw-artifacts/raw.json | python3 compute_persistence.py
"""
from __future__ import annotations

import json
import sys
import hashlib
import numpy as np


def file_feature_vector(f: dict) -> np.ndarray:
    """Extract a numeric feature vector from a file entry."""
    size = f.get("size_bytes", 0)
    staleness = f.get("staleness_days", 0)

    # Language as a hash-derived float (deterministic)
    lang = f.get("language", "unknown")
    lang_hash = int(hashlib.md5(lang.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    return np.array([size, staleness, lang_hash], dtype=np.float64)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Min-max normalize each column to [0, 1]."""
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero
    return (features - mins) / ranges


def compute_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix."""
    n = features.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((features[i] - features[j]) ** 2))
            D[i, j] = d
            D[j, i] = d
    return D


def compute_h0_persistence(D: np.ndarray) -> list[dict]:
    """Union-Find H₀ persistence on a distance matrix.

    Returns a list of barcode entries {birth, death, dimension}.
    Same algorithm as Ripser for dimension 0.
    """
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
        # The younger component dies
        barcodes.append({
            "birth": float(births[rb]),
            "death": float(eps),
            "dimension": 0
        })
        del births[rb]

    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i, j], i, j))
    edges.sort()

    for eps, i, j in edges:
        union(i, j, eps)

    # Remaining components live forever
    for comp in births:
        barcodes.append({
            "birth": float(births[comp]),
            "death": float("inf"),
            "dimension": 0
        })

    return barcodes


def identify_clusters(D: np.ndarray, barcodes: list[dict],
                      files: list[dict]) -> tuple[list[dict], list[str]]:
    """Identify clusters from long-lived H₀ bars.

    A 'long bar' is one whose lifetime (death - birth) exceeds the
    median lifetime. These correspond to real structure; short bars
    are noise.
    """
    finite_bars = [b for b in barcodes if b["death"] != float("inf")]
    if not finite_bars:
        # Everything in one cluster
        return [{
            "id": 0,
            "label": "all",
            "members": [f["path"] for f in files],
            "bar_length": float("inf"),
            "centroid_description": ""
        }], []

    lifetimes = [b["death"] - b["birth"] for b in finite_bars]
    median_life = float(np.median(lifetimes))

    # Use the persistence threshold: cut at median lifetime
    # Components that survive past this threshold are real clusters
    eps_cut = median_life

    # Run Union-Find up to eps_cut to get cluster assignments
    n = len(files)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps_cut:
                union(i, j)

    # Group files by component
    clusters_map = {}
    noise = []
    for i in range(n):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(i)

    # Clusters with 1 member are noise
    clusters = []
    cid = 0
    for root, members in sorted(clusters_map.items(), key=lambda x: -len(x[1])):
        member_paths = [files[m]["path"] for m in members]
        if len(members) == 1:
            noise.append(member_paths[0])
        else:
            # Find the bar_length for this cluster (longest bar involving its members)
            max_bar = max(lifetimes) if lifetimes else 0
            clusters.append({
                "id": cid,
                "label": f"cluster_{cid}",
                "members": member_paths,
                "bar_length": float(max_bar),
                "centroid_description": ""
            })
            cid += 1

    return clusters, noise


def main():
    raw = json.load(sys.stdin)
    files = raw.get("files", [])
    n = len(files)

    if n < 2:
        result = {
            "barcode": [],
            "distances": [],
            "clusters": [],
            "noise": [f["path"] for f in files],
            "routing": "REPROBE",
            "routing_reason": "Fewer than 2 artifacts — need more input"
        }
        json.dump(result, sys.stdout, indent=2)
        return

    # Build feature matrix
    features = np.array([file_feature_vector(f) for f in files])
    features = normalize_features(features)

    # Compute distance matrix
    D = compute_distance_matrix(features)

    # Compute H₀ persistence
    barcodes = compute_h0_persistence(D)

    # Identify clusters
    clusters, noise = identify_clusters(D, barcodes, files)

    # Routing decision
    n_clusters = len(clusters)
    if n_clusters == 0:
        routing = "REPROBE"
        routing_reason = "No persistent clusters found — all noise"
    elif n_clusters > 3:
        routing = "SPLIT"
        routing_reason = f"{n_clusters} clusters detected — decompose into sub-pipelines"
    else:
        routing = "ASCEND"
        routing_reason = f"{n_clusters} stable cluster(s) identified"

    result = {
        "barcode": barcodes,
        "distances": D.tolist(),
        "clusters": clusters,
        "noise": noise,
        "routing": routing,
        "routing_reason": routing_reason
    }

    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
