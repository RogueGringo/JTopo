"""Benchmark Dumitriu-Edelman tridiagonal GUE generation."""
import time
import numpy as np
from atft.sources.gue import GUESource

src = GUESource(seed=42)

for n in [500, 1000, 2000, 5000, 10000, 50000]:
    t0 = time.time()
    cloud = src.generate(n)
    dt = time.time() - t0
    eigs = cloud.points[:, 0]
    print(f"N={n:>6d}:  {dt:7.3f}s  |  range=[{eigs.min():.4f}, {eigs.max():.4f}]  |  mem ~{3*n*8/1024:.0f} KB")
