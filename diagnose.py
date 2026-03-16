"""Diagnose the scale mismatch between sources."""
from pathlib import Path
import numpy as np

from atft.sources.zeta_zeros import ZetaZerosSource
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.feature_maps.identity import IdentityMap
from atft.topology.analytical_h0 import AnalyticalH0
from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.waypoint_extractor import WaypointExtractor

N = 500

# Load and unfold each source
zeta_cloud = ZetaZerosSource(Path("data/odlyzko_zeros.txt")).generate(N)
gue_cloud = GUESource(seed=42).generate(N)
poisson_cloud = PoissonSource(seed=43).generate(N)

zeta_unf = SpectralUnfolding(method="zeta").transform(zeta_cloud)
gue_unf = SpectralUnfolding(method="rank").transform(gue_cloud)
poisson_unf = IdentityMap().transform(poisson_cloud)

print("=== RAW POINT RANGES ===")
print(f"Zeta raw:     [{zeta_cloud.points.min():.2f}, {zeta_cloud.points.max():.2f}]")
print(f"GUE raw:      [{gue_cloud.points.min():.4f}, {gue_cloud.points.max():.4f}]")
print(f"Poisson raw:  [{poisson_cloud.points.min():.2f}, {poisson_cloud.points.max():.2f}]")

print("\n=== UNFOLDED POINT RANGES ===")
print(f"Zeta unf:     [{zeta_unf.points.min():.2f}, {zeta_unf.points.max():.2f}]")
print(f"GUE unf:      [{gue_unf.points.min():.2f}, {gue_unf.points.max():.2f}]")
print(f"Poisson unf:  [{poisson_unf.points.min():.2f}, {poisson_unf.points.max():.2f}]")

# Compute gaps
ph = AnalyticalH0()

zeta_pd = ph.compute(zeta_unf)
gue_pd = ph.compute(gue_unf)
poisson_pd = ph.compute(poisson_unf)

zeta_gaps = np.sort(zeta_pd.degree(0)[:, 1][np.isfinite(zeta_pd.degree(0)[:, 1])])
gue_gaps = np.sort(gue_pd.degree(0)[:, 1][np.isfinite(gue_pd.degree(0)[:, 1])])
poisson_gaps = np.sort(poisson_pd.degree(0)[:, 1][np.isfinite(poisson_pd.degree(0)[:, 1])])

print("\n=== GAP STATISTICS (death values = gap magnitudes) ===")
print(f"Zeta gaps:    mean={zeta_gaps.mean():.4f}, std={zeta_gaps.std():.4f}, max={zeta_gaps.max():.4f}")
print(f"GUE gaps:     mean={gue_gaps.mean():.4f}, std={gue_gaps.std():.4f}, max={gue_gaps.max():.4f}")
print(f"Poisson gaps: mean={poisson_gaps.mean():.4f}, std={poisson_gaps.std():.4f}, max={poisson_gaps.max():.4f}")

# Compute evolution curves
cc = EvolutionCurveComputer(n_steps=100)
zeta_curves = cc.compute(zeta_pd)
gue_curves = cc.compute(gue_pd)
poisson_curves = cc.compute(poisson_pd)

print("\n=== EPSILON GRID RANGES ===")
print(f"Zeta eps:     [{zeta_curves.betti[0].epsilon_grid[0]:.4f}, {zeta_curves.betti[0].epsilon_grid[-1]:.4f}]")
print(f"GUE eps:      [{gue_curves.betti[0].epsilon_grid[0]:.4f}, {gue_curves.betti[0].epsilon_grid[-1]:.4f}]")
print(f"Poisson eps:  [{poisson_curves.betti[0].epsilon_grid[0]:.4f}, {poisson_curves.betti[0].epsilon_grid[-1]:.4f}]")

# Waypoint signatures
we = WaypointExtractor(k_waypoints=2)
zeta_sig = we.extract(zeta_pd, zeta_curves)
gue_sig = we.extract(gue_pd, gue_curves)
poisson_sig = we.extract(poisson_pd, poisson_curves)

print("\n=== WAYPOINT SIGNATURES ===")
print(f"Zeta:    {zeta_sig.as_vector()}")
print(f"GUE:     {gue_sig.as_vector()}")
print(f"Poisson: {poisson_sig.as_vector()}")
