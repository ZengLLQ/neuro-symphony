"""Spectral analysis and pathological beta-band power quantification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible across NumPy 1.x and 2.x."""
    if len(y) < 2 or len(x) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@dataclass
class SpectralMetrics:
    """Quantitative frequency-domain biomarker metrics."""
    peak_frequency_hz: float
    peak_power: float
    beta_band_power: float
    total_power: float
    beta_relative_ratio: float
    beta_suppression_pct: float = 0.0


class SpectralAnalyzer:
    """Computes Power Spectral Density (PSD) and quantifies Parkinsonian beta activity."""

    def __init__(
        self,
        fs: float = 20000.0,
        nperseg: int = 1024,
        noverlap: Optional[int] = None
    ) -> None:
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2

    def compute_psd(
        self,
        time_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density via Welch's averaged periodogram method.

        Returns:
            Tuple of (frequencies_hz, power_spectral_density)
        """
        if len(time_series) < 16:
            return np.array([0.0]), np.array([0.0])

        seg_len = min(len(time_series), self.nperseg)
        overlap = min(seg_len // 2, self.noverlap)

        freqs, psd = signal.welch(
            time_series,
            fs=self.fs,
            window="hann",
            nperseg=seg_len,
            noverlap=overlap,
            scaling="density"
        )
        return freqs, psd

    def analyze_beta_band(
        self,
        time_series: np.ndarray,
        baseline_beta_power: Optional[float] = None,
        beta_min_hz: float = 13.0,
        beta_max_hz: float = 30.0
    ) -> SpectralMetrics:
        """Quantify beta oscillation peak frequency, amplitude, and suppression."""
        freqs, psd = self.compute_psd(time_series)

        # Mask for beta band
        beta_mask = (freqs >= beta_min_hz) & (freqs <= beta_max_hz)
        broadband_mask = (freqs >= 4.0) & (freqs <= 100.0)

        if not np.any(beta_mask) or np.all(psd == 0):
            return SpectralMetrics(
                peak_frequency_hz=20.0,
                peak_power=0.0,
                beta_band_power=0.0,
                total_power=0.0,
                beta_relative_ratio=0.0,
                beta_suppression_pct=0.0
            )

        # Beta band power via numerical trapezoidal integration
        beta_freqs = freqs[beta_mask]
        beta_psd = psd[beta_mask]
        if len(beta_freqs) == 1:
            beta_power = float(beta_psd[0])
        else:
            beta_power = _integrate_trapezoid(beta_psd, beta_freqs)

        # Peak within beta band
        peak_idx = int(np.argmax(beta_psd))
        peak_freq = float(beta_freqs[peak_idx])
        peak_pow = float(beta_psd[peak_idx])

        # Broadband power (4-100 Hz)
        broad_freqs = freqs[broadband_mask]
        broad_psd = psd[broadband_mask]
        total_power = _integrate_trapezoid(broad_psd, broad_freqs) if len(broad_freqs) > 1 else 1e-6
        relative_ratio = beta_power / max(1e-9, total_power)

        # Beta suppression relative to un-stimulated baseline
        suppression = 0.0
        if baseline_beta_power is not None and baseline_beta_power > 0:
            suppression = max(0.0, (1.0 - beta_power / baseline_beta_power) * 100.0)

        return SpectralMetrics(
            peak_frequency_hz=peak_freq,
            peak_power=peak_pow,
            beta_band_power=beta_power,
            total_power=total_power,
            beta_relative_ratio=relative_ratio,
            beta_suppression_pct=suppression
        )
