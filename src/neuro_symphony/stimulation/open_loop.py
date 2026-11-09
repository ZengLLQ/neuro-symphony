"""Classical open-loop constant deep brain stimulation (cDBS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from neuro_symphony.stimulation.pulse_generator import BiphasicPulseConfig, PulseGenerator


class OpenLoopDBS:
    """Continuous high-frequency open-loop stimulation (cDBS).

    Delivers continuous trains of biphasic charge-balanced pulses at a fixed
    frequency (standard clinical paradigm: 130 Hz), regardless of the ongoing
    pathological neural state.
    """

    def __init__(
        self,
        frequency_hz: float = 130.0,
        amplitude: float = 3.0,
        pulse_width_ms: float = 0.09,
        config: Optional[BiphasicPulseConfig] = None
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = BiphasicPulseConfig(
                frequency_hz=frequency_hz,
                amplitude=amplitude,
                pulse_width_ms=pulse_width_ms
            )

        self.pulse_gen = PulseGenerator(config=self.config)
        self.period_ms = 1000.0 / self.config.frequency_hz
        self.time_accumulator: float = 0.0
        self.enabled: bool = True

        # Diagnostic counters
        self.active_time_ms: float = 0.0
        self.total_time_ms: float = 0.0

    @property
    def total_energy(self) -> float:
        """Cumulative electrical energy proportional to integral(I^2 dt)."""
        return self.pulse_gen.cumulative_energy

    @property
    def total_pulses(self) -> int:
        return self.pulse_gen.total_pulses_delivered

    @property
    def duty_cycle(self) -> float:
        """Fraction of time stimulation is active."""
        if self.total_time_ms <= 0:
            return 0.0
        return self.active_time_ms / self.total_time_ms

    def compute_stimulus(
        self,
        lfp_sample: float = 0.0,
        dt: float = 0.05,
        current_time: float = 0.0
    ) -> float:
        """Calculate stimulation current for current time step.

        In open-loop DBS, lfp_sample is ignored.
        """
        self.total_time_ms += dt

        if not self.enabled:
            return 0.0

        self.time_accumulator += dt
        trigger = False

        if self.time_accumulator >= self.period_ms:
            trigger = True
            self.time_accumulator -= self.period_ms

        stim = self.pulse_gen.step(dt=dt, trigger=trigger)
        if abs(stim) > 1e-4:
            self.active_time_ms += dt

        return stim
