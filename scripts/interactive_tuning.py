#!/usr/bin/env python
"""ATFT Interactive Tuner -- Textual TUI for parameter exploration.

Adjust K, sigma, epsilon, k_eig, and point cloud source on the fly
with granular pipeline caching so only necessary stages recompute.

Usage:
    python scripts/interactive_tuning.py

Keybindings:
    R / Enter  Run compute
    S          Save current parameter state to JSON
    Q          Quit

Caching rules (zero compute waste):
    Change sigma or K   -> only rebuild TransportMapBuilder
    Change epsilon/k_eig -> reuse cached builder AND zeros
    Change source or N   -> reload zeros (builder still cached)
"""
from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
)
from textual import work

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder

try:
    import torch
    from atft.topology.torch_sheaf_laplacian import TorchSheafLaplacian

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "odlyzko_zeros.txt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "tuner_states"


# ---------------------------------------------------------------------------
# Caching layer
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _get_builder(K: int, sigma: float) -> TransportMapBuilder:
    """Cache TransportMapBuilder by (K, sigma). Both are hashable."""
    return TransportMapBuilder(K=K, sigma=sigma)


_zeros_cache: dict[tuple[str, int], np.ndarray] = {}


def _load_zeros(source: str, n_points: int) -> np.ndarray:
    """Load and unfold zeros, cached by (source, n_points)."""
    key = (source, n_points)
    if key in _zeros_cache:
        return _zeros_cache[key]

    if source == "zeta":
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Odlyzko data not found at {DATA_PATH}. "
                "Clone the repo with data/ directory."
            )
        src = ZetaZerosSource(data_path=DATA_PATH)
        pc = src.generate(n_points=n_points)
        unfold = SpectralUnfolding(method="zeta")
        zeros = unfold.transform(pc).points.ravel()
    elif source == "poisson":
        src = PoissonSource(seed=42)
        pc = src.generate(n_points=n_points)
        zeros = pc.points.ravel()
    elif source == "gue":
        src = GUESource(seed=42)
        pc = src.generate(n_points=n_points)
        unfold = SpectralUnfolding(method="semicircle")
        zeros = unfold.transform(pc).points.ravel()
    else:
        raise ValueError(f"Unknown source: {source!r}")

    _zeros_cache[key] = zeros
    return zeros


# ---------------------------------------------------------------------------
# Textual App
# ---------------------------------------------------------------------------

SIDEBAR_CSS = """\
#sidebar {
    width: 40;
    padding: 1 2;
    background: $surface;
    border-right: thick $primary;
}

#results-panel {
    width: 1fr;
    padding: 1 2;
}

.section-title {
    text-style: bold;
    color: $text;
    margin-top: 1;
    margin-bottom: 0;
}

.param-label {
    margin-top: 1;
    margin-bottom: 0;
}

Input {
    margin-bottom: 0;
}

Button {
    margin-top: 1;
    width: 100%;
}

#btn-compute {
    margin-top: 2;
}

RadioSet {
    height: auto;
    margin-bottom: 0;
}
"""


class InteractiveTuner(App):
    """ATFT Interactive Parameter Tuner."""

    TITLE = "ATFT Interactive Tuner"
    CSS = SIDEBAR_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_compute", "Run"),
        ("s", "save_state", "Save"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Point Cloud Source", classes="section-title")
                yield RadioSet(
                    RadioButton("Zeta Zeros", value=True, id="src-zeta"),
                    RadioButton("Poisson Random", id="src-poisson"),
                    RadioButton("GUE Control", id="src-gue"),
                    id="source-select",
                )
                yield Label("N (points)", classes="param-label")
                yield Input(value="200", id="input-n")
                yield Label("K (fiber dim)", classes="param-label")
                yield Input(value="20", id="input-k")
                yield Label("sigma (0.0 - 1.0)", classes="param-label")
                yield Input(value="0.50", id="input-sigma")
                yield Label("epsilon (scale)", classes="param-label")
                yield Input(value="3.0", id="input-eps")
                yield Label("k_eig (eigenvalues)", classes="param-label")
                yield Input(value="20", id="input-keig")
                yield Button("Run Compute", id="btn-compute", variant="success")
                yield Button("Save State", id="btn-save", variant="primary")
            with Vertical(id="results-panel"):
                yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write("[bold]ATFT Interactive Tuner[/bold]")
        log.write("[dim]" + "─" * 50 + "[/dim]")

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                log.write(f"[green]GPU detected:[/green] {name}")
            else:
                log.write("[yellow]CPU mode[/yellow] (no CUDA available)")
        else:
            log.write("[yellow]CPU mode[/yellow] (PyTorch not installed)")

        log.write("")
        log.write("Adjust parameters in the sidebar, then press "
                   "[bold]Run Compute[/bold] or [bold]R[/bold].")
        log.write("Press [bold]S[/bold] to save parameter state, "
                   "[bold]Q[/bold] to quit.")
        log.write("")

    # -- Parameter reading --------------------------------------------------

    def _get_source(self) -> str:
        radio = self.query_one("#source-select", RadioSet)
        btn = radio.pressed_button
        if btn is None:
            return "zeta"
        return btn.id.replace("src-", "")

    def _get_params(self) -> dict:
        """Read current parameter values from the UI widgets."""
        try:
            N = int(self.query_one("#input-n", Input).value)
        except (ValueError, TypeError):
            N = 200
        try:
            K = int(self.query_one("#input-k", Input).value)
        except (ValueError, TypeError):
            K = 20
        try:
            sigma = float(self.query_one("#input-sigma", Input).value)
        except (ValueError, TypeError):
            sigma = 0.5
        try:
            epsilon = float(self.query_one("#input-eps", Input).value)
        except (ValueError, TypeError):
            epsilon = 3.0
        try:
            k_eig = int(self.query_one("#input-keig", Input).value)
        except (ValueError, TypeError):
            k_eig = 20

        return {
            "source": self._get_source(),
            "N": max(N, 2),
            "K": max(K, 1),
            "sigma": max(0.0, min(sigma, 1.0)),
            "epsilon": max(0.01, epsilon),
            "k_eig": max(1, k_eig),
        }

    # -- Actions ------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-compute":
            self.action_run_compute()
        elif event.button.id == "btn-save":
            self.action_save_state()

    def action_run_compute(self) -> None:
        params = self._get_params()
        self._run_compute(params)

    def action_save_state(self) -> None:
        params = self._get_params()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUT_DIR / f"state_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        log = self.query_one("#log", RichLog)
        log.write(f"[bold blue]Saved:[/bold blue] {path}")

    # -- Compute (background thread) ---------------------------------------

    @work(thread=True)
    def _run_compute(self, params: dict) -> None:
        """Run the ATFT pipeline in a background thread."""
        log = self.query_one("#log", RichLog)

        self.call_from_thread(log.write, "")
        self.call_from_thread(
            log.write,
            "[dim]" + "━" * 50 + "[/dim]",
        )
        self.call_from_thread(
            log.write,
            f"[bold cyan]Computing[/bold cyan]  "
            f"source={params['source']}  N={params['N']}  "
            f"K={params['K']}  σ={params['sigma']:.3f}  "
            f"ε={params['epsilon']:.2f}  k={params['k_eig']}",
        )

        t_start = time.perf_counter()

        try:
            # Step 1: Load zeros (cached by source + N)
            t1 = time.perf_counter()
            was_cached = (params["source"], params["N"]) in _zeros_cache
            zeros = _load_zeros(params["source"], params["N"])
            dt1 = time.perf_counter() - t1
            tag = "[dim](cached)[/dim]" if was_cached else ""
            self.call_from_thread(
                log.write,
                f"  Zeros: {len(zeros)} pts  {dt1:.3f}s {tag}",
            )

            # Step 2: Get builder (cached by K + sigma)
            t2 = time.perf_counter()
            builder = _get_builder(params["K"], params["sigma"])
            dt2 = time.perf_counter() - t2
            n_primes = len(builder.primes)
            self.call_from_thread(
                log.write,
                f"  Builder: K={params['K']} ({n_primes} primes)  "
                f"σ={params['sigma']:.3f}  {dt2:.3f}s",
            )

            # Step 3: Assemble Laplacian + eigensolver
            t3 = time.perf_counter()
            dim = params["N"] * params["K"]

            # Use SparseSheafLaplacian (reliable shift-invert eigsh).
            # TorchSheafLaplacian's Lanczos is unreliable for dim > 500.
            lap = SparseSheafLaplacian(
                builder, zeros, normalize=True,
            )
            backend_label = "SciPy eigsh (CPU)"

            spectral_sum = lap.spectral_sum(
                params["epsilon"], k=params["k_eig"],
            )
            dt3 = time.perf_counter() - t3

            # GPU cleanup if torch is loaded
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            dt_total = time.perf_counter() - t_start

            self.call_from_thread(
                log.write,
                f"  Backend: {backend_label}  dim={dim}",
            )
            self.call_from_thread(
                log.write,
                f"  Eigensolver: {dt3:.3f}s",
            )
            self.call_from_thread(log.write, "")
            self.call_from_thread(
                log.write,
                f"  [bold green]S(σ={params['sigma']:.3f}, "
                f"ε={params['epsilon']:.2f}) = "
                f"{spectral_sum:.8f}[/bold green]",
            )
            self.call_from_thread(
                log.write,
                f"  [dim]Total: {dt_total:.3f}s[/dim]",
            )

        except Exception as e:
            self.call_from_thread(
                log.write,
                f"  [bold red]Error:[/bold red] {e}",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = InteractiveTuner()
    app.run()
