"""Unit tests for signal processing, forward LFP, and spectral analysis."""

from __future__ import annotations

import math
import numpy as np
import pytest

from neuro_symphony.signal.hilbert_envelope import HilbertEnvelopeTracker
from neuro_symphony.signal.lfp import ForwardLFP, LFPConfig
from neuro_symphony.signal.spectral import SpectralAnalyzer, SpectralMetrics


class TestForwardLFP:
    """Test suite for forward biophysical LFP modeling."""

    def test_lfp_scaling_with_distance(self) -> None:
        """Electrode potential must drop proportionally with distance r."""
        lfp_model = ForwardLFP()
        currents = np.array([2.0, 3.0, 1.5])  # nA

        v_near = lfp_model.calculate_lfp(currents, distances_um=np.array([50.0, 50.0, 50.0]))
        v_far = lfp_model.calculate_lfp(currents, distances_um=np.array([200.0, 200.0, 200.0]))

        assert v_near > v_far
        # Inversely proportional: 200 um vs 50 um is 4x distance, potential should be ~4x lower
        assert math.isclose(v_near / v_far, 4.0, rel_tol=0.05)


class TestHilbertTracker:
    """Test suite for real-time and offline Hilbert envelope tracking."""

    def test_online_tracker_envelope(self) -> None:
        """Tracker should recover envelope of pure 20 Hz beta sinusoidal tone."""
        fs = 20000.0
        tracker = HilbertEnvelopeTracker(fs=fs, low_hz=13.0, high_hz=30.0)

        # 20 Hz tone with amplitude 10.0
        n_samples = 4000
        envelopes = []
        for i in range(n_samples):
            t = i / fs
            sample = 10.0 * math.sin(2 * math.pi * 20.0 * t)
            env, phase = tracker.update(sample)
            envelopes.append(env)
            assert -math.pi <= phase <= math.pi

        # After transient convergence, envelope should be non-zero and positive
        final_env = envelopes[-1]
        assert final_env > 1.0

    def test_offline_filtering_and_envelope(self) -> None:
        """Offline zero-phase Hilbert should extract envelope of modulated carrier."""
        fs = 2000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        # 20 Hz beta oscillation with slow 2 Hz amplitude modulation
        envelope_true = 5.0 + 3.0 * np.sin(2 * np.pi * 2.0 * t)
        signal_vec = envelope_true * np.sin(2 * np.pi * 20.0 * t)

        filt, env_est, phase = HilbertEnvelopeTracker.compute_offline_envelope_and_phase(
            signal_array=signal_vec,
            fs=fs,
            low_hz=13.0,
            high_hz=30.0
        )

        assert len(env_est) == len(t)
        assert len(phase) == len(t)
        # Mean envelope estimate should approximate mean true envelope (5.0)
        assert math.isclose(float(np.mean(env_est)), 5.0, rel_tol=0.25)


class TestSpectralAnalyzer:
    """Test suite for Welch PSD and beta power quantification."""

    def test_beta_peak_detection(self) -> None:
        """SpectralAnalyzer must detect peak frequency of 22 Hz synthetic beta oscillation."""
        fs = 1000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        beta_signal = 4.0 * np.sin(2 * np.pi * 22.0 * t) + np.random.normal(0, 0.2, len(t))

        analyzer = SpectralAnalyzer(fs=fs, nperseg=256)
        metrics = analyzer.analyze_beta_band(beta_signal)

        assert 20.0 <= metrics.peak_frequency_hz <= 24.0
        assert metrics.peak_power > 0.5
        assert metrics.beta_band_power > 0.0

    def test_beta_suppression_calculation(self) -> None:
        """Beta suppression percentage should reflect power decrease accurately."""
        analyzer = SpectralAnalyzer(fs=1000.0)
        t = np.arange(0, 2.0, 1.0 / 1000.0)
        signal_baseline = 10.0 * np.sin(2 * np.pi * 20.0 * t)
        signal_suppressed = 3.0 * np.sin(2 * np.pi * 20.0 * t)

        base_metrics = analyzer.analyze_beta_band(signal_baseline)
        supp_metrics = analyzer.analyze_beta_band(
            signal_suppressed,
            baseline_beta_power=base_metrics.beta_band_power
        )

        assert supp_metrics.beta_suppression_pct > 80.0
