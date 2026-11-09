"""Unit tests for pure 7-bit ASCII Terminal Visualizer."""

from __future__ import annotations

import numpy as np
import pytest

from neuro_symphony.circuits.populations import PopulationType
from neuro_symphony.visualizer import TerminalVisualizer


class TestVisualizer:
    """Test suite ensuring strictly 100% 7-bit ASCII characters (ord <= 127)."""

    def test_pure_ascii_validation_helper(self) -> None:
        """Helper must identify pure ASCII vs unicode characters."""
        assert TerminalVisualizer.is_pure_ascii("Standard ASCII text 12345!+-*") is True
        # Box drawing unicode character
        assert TerminalVisualizer.is_pure_ascii("Unicode box: \u2500") is False

    def test_header_pure_ascii(self) -> None:
        """Headers must be strictly 7-bit ASCII."""
        vis = TerminalVisualizer()
        hdr = vis.render_header("NEUROLOGICAL DASHBOARD")
        assert vis.is_pure_ascii(hdr) is True
        assert "+" in hdr
        assert "|" in hdr

    def test_kuramoto_gauge_levels(self) -> None:
        """Kuramoto gauge should format properly across all synchrony ranges."""
        vis = TerminalVisualizer()
        g_low = vis.render_kuramoto_gauge(0.15)
        g_med = vis.render_kuramoto_gauge(0.48)
        g_high = vis.render_kuramoto_gauge(0.85)

        for g in (g_low, g_med, g_high):
            assert vis.is_pure_ascii(g) is True
            assert "[" in g and "]" in g

        assert "DESYNCHRONIZED" in g_low
        assert "MODERATE" in g_med
        assert "PATHOLOGICAL" in g_high

    def test_dbs_status_pure_ascii(self) -> None:
        """DBS status text must be 7-bit ASCII."""
        vis = TerminalVisualizer()
        st_active = vis.render_dbs_status(active=True, energy_pct_saved=58.2, pulses_delivered=142)
        st_idle = vis.render_dbs_status(active=False, energy_pct_saved=0.0, pulses_delivered=0)

        assert vis.is_pure_ascii(st_active) is True
        assert vis.is_pure_ascii(st_idle) is True
        assert "STIMULATING" in st_active
        assert "IDLE" in st_idle

    def test_waveform_trace_pure_ascii(self) -> None:
        """Waveform plot must be 7-bit ASCII."""
        vis = TerminalVisualizer()
        t = np.linspace(0, 100, 50)
        sig = np.sin(t)
        env = np.ones_like(t)

        plot_str = vis.render_waveform_trace(
            time_vec=t,
            lfp_vec=sig,
            envelope_vec=env,
            width=50,
            height=7
        )

        assert vis.is_pure_ascii(plot_str) is True
        assert "*" in plot_str  # LFP trace glyph

    def test_raster_plot_pure_ascii(self) -> None:
        """Spike raster plot must be 7-bit ASCII."""
        vis = TerminalVisualizer()
        raster_data = {
            PopulationType.STN: [(10.0, 1), (15.0, 3), (25.0, 2)],
            PopulationType.GPE: [(12.0, 0), (22.0, 1)]
        }

        raster_str = vis.render_raster_plot(
            spike_raster=raster_data,
            time_min=0.0,
            time_max=30.0,
            width=40,
            populations=[PopulationType.STN, PopulationType.GPE]
        )

        assert vis.is_pure_ascii(raster_str) is True
        assert "STN" in raster_str
        assert "GPE" in raster_str

    def test_full_dashboard_pure_ascii(self) -> None:
        """Full cockpit dashboard must satisfy 100% 7-bit ASCII standard."""
        vis = TerminalVisualizer(default_width=70)
        dash = vis.render_full_dashboard(
            time_ms=250.0,
            dopamine_level=0.15,
            kuramoto_r=0.78,
            stn_lfp=-58.4,
            beta_power=14.2,
            dbs_active=True,
            dbs_mode="Adaptive Closed-Loop",
            pulses_count=180,
            energy_saved_pct=62.5
        )

        assert vis.is_pure_ascii(dash) is True
        for char in dash:
            assert ord(char) < 128
