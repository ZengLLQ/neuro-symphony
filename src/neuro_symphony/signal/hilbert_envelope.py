"""Hilbert transform and instantaneous beta envelope and phase tracking."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import signal


@dataclass
class EnvelopeResult:
    """Instantaneous signal parameters derived from analytic representation."""
    envelope: float
    phase_rad: float
    filtered_sample: float


class HilbertEnvelopeTracker:
    """Real-time causal tracker for instantaneous beta-band (13-30 Hz) envelope and phase.

    In closed-loop DBS devices (such as the Medtronic Percept PC / Activa PC+S),
    the LFP is filtered causally in real time without look-ahead, and the analytic signal
    z(t) = s(t) + i * H{s(t)} yields:
    - Envelope A(t) = |z(t)| (biomarker for pathological burst duration)
    - Phase phi(t) = arg(z(t)) (used for phase-locked desynchronization)
    """

    def __init__(
        self,
        fs: float = 20000.0,
        low_hz: float = 13.0,
        high_hz: float = 30.0,
        order: int = 2,
        buffer_len: int = 128
    ) -> None:
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self.buffer_len = buffer_len

        # Design digital IIR bandpass Butterworth filter
        nyquist = 0.5 * fs
        low = low_hz / nyquist
        high = high_hz / nyquist
        # Ensure within Nyquist interval
        low = max(0.001, min(0.99, low))
        high = max(low + 0.001, min(0.995, high))

        self.b, self.a = signal.butter(order, [low, high], btype="bandpass")
        self.zi = signal.lfilter_zi(self.b, self.a)

        # Ring buffer of filtered samples for Hilbert estimation
        self.filtered_buffer: deque = deque(maxlen=buffer_len)
        for _ in range(buffer_len):
            self.filtered_buffer.append(0.0)

        # Smoothed envelope accumulator
        self.current_envelope: float = 0.0
        self.current_phase: float = 0.0

    def update(self, raw_sample: float) -> Tuple[float, float]:
        """Update tracker with incoming LFP sample.

        Returns:
            Tuple of (instantaneous_envelope, instantaneous_phase_in_radians)
        """
        # 1. Causal IIR bandpass filtering step
        filtered_val, self.zi = signal.lfilter(self.b, self.a, [raw_sample], zi=self.zi)
        sample_filt = float(filtered_val[0])
        self.filtered_buffer.append(sample_filt)

        # 2. Approximate analytic signal via short window
        buf = np.array(self.filtered_buffer)
        n = len(buf)
        if n >= 16:
            # Short-time analytic signal via FFT/Hilbert
            analytic = signal.hilbert(buf)
            # Latest sample
            env = float(np.abs(analytic[-1]))
            phase = float(np.angle(analytic[-1]))
        else:
            env = abs(sample_filt)
            phase = 0.0

        # Exponential moving average smoothing on envelope to filter noise spikes
        alpha = 0.15
        self.current_envelope = alpha * env + (1.0 - alpha) * self.current_envelope
        self.current_phase = phase

        return self.current_envelope, self.current_phase

    @staticmethod
    def compute_offline_envelope_and_phase(
        signal_array: np.ndarray,
        fs: float,
        low_hz: float = 13.0,
        high_hz: float = 30.0,
        order: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Offline zero-phase forward-backward filtering and Hilbert transform.

        Returns:
            Tuple of (filtered_signal, instantaneous_envelope, instantaneous_phase)
        """
        nyquist = 0.5 * fs
        low = low_hz / nyquist
        high = high_hz / nyquist
        b, a = signal.butter(order, [low, high], btype="bandpass")

        # Zero-phase digital filtering
        filtered = signal.filtfilt(b, a, signal_array)

        # Hilbert transform
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)
        phase = np.angle(analytic)

        return filtered, envelope, phase
