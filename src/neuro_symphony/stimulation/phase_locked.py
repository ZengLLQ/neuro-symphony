"""Phase-locked deep brain stimulation (pDBS) for resonant desynchronization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from neuro_symphony.signal.hilbert_envelope import HilbertEnvelopeTracker
from neuro_symphony.stimulation.pulse_generator import BiphasicPulseConfig, PulseGenerator


@dataclass
class PhaseLockConfig:
    """Configuration for phase-locked desynchronization stimulation."""
    target_phase_rad: float = math.pi  # Optimal anti-phase (radians, in [-pi, pi])
    phase_tolerance_rad: float = 0.25   # Angle window for triggering
    min_refractory_ms: float = 35.0    # Refractory period preventing multiple pulses per cycle (~20-25 Hz)
    min_amplitude_threshold: float = 2.0  # Only fire when beta amplitude is significant


class PhaseLockedDBS:
    """Phase-Locked Deep Brain Stimulation (pDBS).

    Tracks the instantaneous phase phi(t) of the pathological STN beta oscillation
    and delivers timed micro-pulses at the vulnerable phase coordinate (phi* = pi).
    This phase reset causes destructive interference, dismantling the synchronized
    cluster with minimal energy consumption.
    """

    def __init__(
        self,
        config: Optional[PhaseLockConfig] = None,
        pulse_config: Optional[BiphasicPulseConfig] = None,
        fs: float = 20000.0
    ) -> None:
        self.config = config or PhaseLockConfig()
        self.pulse_config = pulse_config or BiphasicPulseConfig(amplitude=3.0, pulse_width_ms=0.09)
        self.pulse_gen = PulseGenerator(config=self.pulse_config)
        self.envelope_tracker = HilbertEnvelopeTracker(fs=fs, low_hz=13.0, high_hz=30.0)

        self.last_phase: float = 0.0
        self.time_since_trigger_ms: float = 9999.0
        self.total_time_ms: float = 0.0
        self.active_time_ms: float = 0.0

    @property
    def total_energy(self) -> float:
        return self.pulse_gen.cumulative_energy

    @property
    def total_pulses(self) -> int:
        return self.pulse_gen.total_pulses_delivered

    def compute_stimulus(
        self,
        lfp_sample: float,
        dt: float = 0.05,
        current_time: float = 0.0
    ) -> float:
        """Evaluate instantaneous phase and trigger pulse upon target phase crossing."""
        self.total_time_ms += dt
        self.time_since_trigger_ms += dt

        envelope, phase = self.envelope_tracker.update(lfp_sample)

        # Check phase crossing condition
        # Target phase is typically pi (or anti-phase)
        trigger = False
        target = self.config.target_phase_rad
        tol = self.config.phase_tolerance_rad

        phase_distance = abs(math.atan2(math.sin(phase - target), math.cos(phase - target)))
        refractory_ok = self.time_since_trigger_ms >= self.config.min_refractory_ms
        amplitude_ok = envelope >= self.config.min_amplitude_threshold

        if phase_distance <= tol and refractory_ok and amplitude_ok:
            trigger = True
            self.time_since_trigger_ms = 0.0

        self.last_phase = phase
        stim = self.pulse_gen.step(dt=dt, trigger=trigger)
        if abs(stim) > 1e-4:
            self.active_time_ms += dt

        return stim
