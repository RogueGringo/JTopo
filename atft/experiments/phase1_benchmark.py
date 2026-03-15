"""Phase 1 Experiment: Zeta vs GUE vs Poisson topological benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.statistical_tests import StatisticalValidator
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import (
    EvolutionCurveSet,
    ValidationResult,
    WaypointSignature,
)
from atft.feature_maps.identity import IdentityMap
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.analytical_h0 import AnalyticalH0


@dataclass
class Phase1Config:
    n_points: int = 10_000
    ensemble_size: int = 1_000
    k_waypoints: int = 2
    n_epsilon_steps: int = 1_000
    confidence_level: float = 0.99
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    cache_dir: Path = Path("cache/")
    device: str = "cuda"
    seed: int = 42


@dataclass
class Phase1Results:
    zeta_validation: ValidationResult
    poisson_validation: ValidationResult
    zeta_curves: EvolutionCurveSet
    gue_curves: list[EvolutionCurveSet]
    poisson_curves: list[EvolutionCurveSet]
    zeta_signature: WaypointSignature
    gue_signatures: list[WaypointSignature]
    poisson_signatures: list[WaypointSignature]


class Phase1Experiment:
    """Orchestrates the Zeta vs GUE vs Poisson benchmark."""

    def __init__(self, config: Phase1Config):
        self.config = config
        self.ph = AnalyticalH0()
        self.curve_computer = EvolutionCurveComputer(n_steps=config.n_epsilon_steps)
        self.waypoint_extractor = WaypointExtractor(k_waypoints=config.k_waypoints)
        self.validator = StatisticalValidator(confidence_level=config.confidence_level)

    def run(self) -> Phase1Results:
        print("Step 1/10: Loading zeta zeros...")
        zeta_src = ZetaZerosSource(self.config.zeta_data_path)
        zeta_cloud = zeta_src.generate(self.config.n_points)

        print("Step 2/10: Generating GUE ensemble...")
        gue_src = GUESource(seed=self.config.seed)
        gue_batch = gue_src.generate_batch(self.config.n_points, self.config.ensemble_size)

        print("Step 3/10: Generating Poisson baseline...")
        poisson_src = PoissonSource(seed=self.config.seed + 1)
        poisson_batch = poisson_src.generate_batch(self.config.n_points, self.config.ensemble_size)

        print("Step 4/10: Unfolding spectra...")
        zeta_unfolded = SpectralUnfolding(method="zeta").transform(zeta_cloud)
        gue_unfolded = SpectralUnfolding(method="rank").transform_batch(gue_batch)
        poisson_unfolded = IdentityMap().transform_batch(poisson_batch)

        print("Step 5/10: Computing persistence diagrams...")
        zeta_pd = self.ph.compute(zeta_unfolded)
        gue_pds = self.ph.compute_batch(gue_unfolded)
        poisson_pds = self.ph.compute_batch(poisson_unfolded)

        print("Step 6/10: Computing evolution curves...")
        zeta_curves = self.curve_computer.compute(zeta_pd)
        gue_curves = [self.curve_computer.compute(pd) for pd in gue_pds]
        poisson_curves = [self.curve_computer.compute(pd) for pd in poisson_pds]

        print("Step 7/10: Extracting waypoint signatures...")
        zeta_sig = self.waypoint_extractor.extract(zeta_pd, zeta_curves)
        gue_sigs = [self.waypoint_extractor.extract(pd, c) for pd, c in zip(gue_pds, gue_curves)]
        poisson_sigs = [self.waypoint_extractor.extract(pd, c) for pd, c in zip(poisson_pds, poisson_curves)]

        print("Step 8/10: Fitting statistical validator on GUE ensemble...")
        self.validator.fit_ensemble(gue_sigs, gue_curves)

        print("Step 9/10: Validating zeta zeros against GUE...")
        zeta_result = self.validator.validate(zeta_sig, zeta_curves)

        print("Step 10/10: Validating Poisson against GUE (negative control)...")
        poisson_result = self.validator.validate(poisson_sigs[0], poisson_curves[0])

        self._print_results(zeta_result, poisson_result)

        return Phase1Results(
            zeta_validation=zeta_result,
            poisson_validation=poisson_result,
            zeta_curves=zeta_curves,
            gue_curves=gue_curves,
            poisson_curves=poisson_curves,
            zeta_signature=zeta_sig,
            gue_signatures=gue_sigs,
            poisson_signatures=poisson_sigs,
        )

    @staticmethod
    def _print_results(zeta: ValidationResult, poisson: ValidationResult):
        print("\n" + "=" * 60)
        print("PHASE 1 RESULTS: Zeta vs GUE Topological Benchmark")
        print("=" * 60)
        print(f"\nZeta zeros:")
        print(f"  Mahalanobis distance: {zeta.mahalanobis_distance:.4f}")
        print(f"  p-value:              {zeta.p_value:.6f}")
        print(f"  Within 99% band:      {zeta.within_confidence_band}")
        print(f"  L2 (Betti):           {zeta.l2_distance_betti:.4f}")
        print(f"  L2 (Gini):            {zeta.l2_distance_gini:.4f}")
        print(f"\nPoisson (negative control):")
        print(f"  Mahalanobis distance: {poisson.mahalanobis_distance:.4f}")
        print(f"  p-value:              {poisson.p_value:.6f}")
        print(f"  Within 99% band:      {poisson.within_confidence_band}")
        print("=" * 60)
