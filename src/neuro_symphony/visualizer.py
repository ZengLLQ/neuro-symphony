"""Pure 7-bit ASCII Terminal Visualizer for NeuroSymphony.

Strictly 100% 7-bit standard ASCII characters (ord(c) <= 127) to ensure
absolute compatibility across Windows cp1252, POSIX C/UTF-8, macOS, and CI terminals.
Strictly NO Unicode box-drawing or special glyphs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from neuro_symphony.circuits.populations import PopulationType


class TerminalVisualizer:
    """ASCII Terminal Visualizer for Spiking Microcircuits and Closed-Loop DBS."""

    def __init__(self, default_width: int = 70) -> None:
        self.width = default_width

    @staticmethod
    def is_pure_ascii(text: str) -> bool:
        """Verify that string contains solely 7-bit ASCII characters (ord <= 127)."""
        return all(ord(c) < 128 for c in text)

    def render_header(self, title: str) -> str:
        """Render a section header with standard ASCII borders."""
        w = self.width
        border = "+" + "-" * (w - 2) + "+"
        padded_title = f"|  {title}".ljust(w - 1) + "|"
        return f"{border}\n{padded_title}\n{border}"

    def render_kuramoto_gauge(self, r_val: float, bar_width: int = 24) -> str:
        """Render horizontal Kuramoto order parameter synchronization meter.

        R in [0.0, 1.0]:
        0.0 - 0.35: Desynchronized (Healthy)
        0.35 - 0.60: Moderate coherence
        0.60 - 1.00: Pathological Hyper-Synchrony (Parkinsonian beta state)
        """
        r_clamped = max(0.0, min(1.0, r_val))
        filled_len = int(round(r_clamped * bar_width))
        empty_len = bar_width - filled_len

        bar = "#" * filled_len + "." * empty_len

        if r_clamped < 0.35:
            state_label = "DESYNCHRONIZED (Healthy)"
        elif r_clamped < 0.60:
            state_label = "MODERATE COHERENCE"
        else:
            state_label = "PATHOLOGICAL HYPER-SYNCHRONY (Beta)"

        return f"Kuramoto R: [{bar}] {r_clamped:.3f} | {state_label}"

    def render_dbs_status(
        self,
        active: bool,
        mode: str = "Adaptive Closed-Loop (aDBS)",
        energy_pct_saved: float = 0.0,
        pulses_delivered: int = 0
    ) -> str:
        """Render status indicator for DBS pulse delivery."""
        pulse_icon = "[* STIMULATING *]" if active else "[  IDLE / SENSING  ]"
        status_line = f"DBS Mode: {mode} | Status: {pulse_icon}"
        metrics_line = f"Pulses: {pulses_delivered} | Energy Savings: {energy_pct_saved:.1f}% vs cDBS"
        return f"{status_line}\n{metrics_line}"

    def render_raster_plot(
        self,
        spike_raster: Dict[PopulationType, List[Tuple[float, int]]],
        time_min: float,
        time_max: float,
        width: int = 60,
        height_per_pop: int = 3,
        populations: Optional[List[PopulationType]] = None
    ) -> str:
        """Render ASCII multi-unit spike raster plot across selected populations."""
        if populations is None:
            populations = [
                PopulationType.M1_PYRAMIDAL,
                PopulationType.STN,
                PopulationType.GPE
            ]

        dt_window = max(1e-3, time_max - time_min)
        lines: List[str] = []
        lines.append(f"+-- SPIKE RASTER [{time_min:.1f} ms to {time_max:.1f} ms] " + "-" * (width - 32) + "+")

        for pop in populations:
            pop_label = pop.value.upper()[:10].ljust(10)
            events = [
                (t, idx) for (t, idx) in spike_raster.get(pop, [])
                if time_min <= t <= time_max
            ]

            # Create 2D grid for this population
            grid = [["." for _ in range(width)] for _ in range(height_per_pop)]

            if events:
                max_neuron = max(idx for _, idx in events) + 1
                for t, idx in events:
                    col = int((t - time_min) / dt_window * (width - 1))
                    col = max(0, min(width - 1, col))
                    row = int((idx / max(1, max_neuron)) * (height_per_pop - 1))
                    row = max(0, min(height_per_pop - 1, row))
                    grid[row][col] = "|"

            for r_i, row in enumerate(grid):
                prefix = pop_label if r_i == height_per_pop // 2 else " " * 10
                lines.append(f"| {prefix} |" + "".join(row) + "|")

        lines.append("+" + "-" * (width + 13) + "+")
        return "\n".join(lines)

    def render_waveform_trace(
        self,
        time_vec: np.ndarray,
        lfp_vec: np.ndarray,
        envelope_vec: Optional[np.ndarray] = None,
        width: int = 60,
        height: int = 9,
        title: str = "LFP BETA OSCILLATION & ENVELOPE"
    ) -> str:
        """Render ASCII waveform trace with optional superimposed Hilbert envelope."""
        if len(lfp_vec) == 0:
            return "+-- NO SIGNAL DATA --+"

        # Resample signal to plot width
        indices = np.linspace(0, len(lfp_vec) - 1, width, dtype=int)
        sampled_lfp = lfp_vec[indices]
        sampled_env = envelope_vec[indices] if envelope_vec is not None else None

        v_min = float(np.min(sampled_lfp))
        v_max = float(np.max(sampled_lfp))
        v_span = max(1e-4, v_max - v_min)

        grid = [[" " for _ in range(width)] for _ in range(height)]
        zero_row = int(round((0.0 - v_min) / v_span * (height - 1)))
        zero_row = max(0, min(height - 1, zero_row))

        # Baseline horizontal reference axis
        for c in range(width):
            grid[zero_row][c] = "-"

        # Plot LFP trace
        for c in range(width):
            val = sampled_lfp[c]
            row = int(round((val - v_min) / v_span * (height - 1)))
            row = max(0, min(height - 1, row))
            grid[row][c] = "*"

        # Plot Envelope if provided
        if sampled_env is not None:
            for c in range(width):
                env_val = sampled_env[c]
                env_row = int(round((env_val - v_min) / v_span * (height - 1)))
                if 0 <= env_row < height:
                    if grid[env_row][c] == " ":
                        grid[env_row][c] = "^"

        # Construct frame
        lines: List[str] = []
        top_border = f"+-- {title} " + "-" * max(1, (width - len(title) - 6)) + "+"
        lines.append(top_border)

        for r in reversed(range(height)):
            axis_val = v_min + (r / (height - 1)) * v_span
            label = f"{axis_val:+5.1f}|"
            row_str = "".join(grid[r])
            lines.append(f"| {label}{row_str} |")

        bottom_border = "+" + "-" * (width + 8) + "+"
        legend = f"| Legend: * = LFP Trace, - = Baseline, ^ = Beta Envelope |"
        lines.append(bottom_border)
        lines.append(legend)
        return "\n".join(lines)

    def render_full_dashboard(
        self,
        time_ms: float,
        dopamine_level: float,
        kuramoto_r: float,
        stn_lfp: float,
        beta_power: float,
        dbs_active: bool,
        dbs_mode: str,
        pulses_count: int,
        energy_saved_pct: float
    ) -> str:
        """Render complete simulation cockpit dashboard in 100% 7-bit ASCII."""
        w = self.width
        h1 = self.render_header("NEUROSYMPHONY: COMPUTATIONAL NEUROPHYSIOLOGY & aDBS COCKPIT")

        da_state = "PARKINSONIAN DECAY" if dopamine_level < 0.4 else "HEALTHY BASAL STATE"
        info_lines = [
            f"| Simulation Time: {time_ms:7.1f} ms | Dopamine Tone: {dopamine_level:4.2f} [{da_state}]",
            f"| STN LFP Value:   {stn_lfp:+7.2f} mV | Beta Peak Power: {beta_power:8.2f} uV^2/Hz",
            "+" + "-" * (w - 2) + "+",
            "| " + self.render_kuramoto_gauge(kuramoto_r, bar_width=26),
            "+" + "-" * (w - 2) + "+",
            "| " + self.render_dbs_status(
                active=dbs_active,
                mode=dbs_mode,
                energy_pct_saved=energy_saved_pct,
                pulses_delivered=pulses_count
            ).replace("\n", "\n| ")
        ]

        full_dash = h1 + "\n" + "\n".join(info_lines) + "\n+" + "-" * (w - 2) + "+"
        assert self.is_pure_ascii(full_dash), "Visualizer must produce 100% 7-bit ASCII!"
        return full_dash
