"""Adaptive closed-loop deep brain stimulation (aDBS) using biomarker envelope tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from neuro_symphony.signal.hilbert_envelope import HilbertEnvelopeTracker
from neuro_symphony.stimulation.pulse_generator import BiphasicPulseConfig, PulseGenerator


@dataclass
class AdaptiveThresholds:
    """Hysteresis thresholds for triggering adaptive stimulation bursts."""
    upper_threshold: float = 8.5   # Trigger DBS when beta amplitude exceeds this
    lower_threshold: float = 4.0   # Deactivate DBS when beta drops below this
    min_burst_duration_ms: float = 80.0  # Debounce / minimum duration per burst


class AdaptiveClosedLoopDBS:
    """Adaptive Closed-Loop Deep Brain Stimulation (aDBS).

    Continuously tracks local field potential (LFP) pathological beta (13-30 Hz)
    envelope and dynamically engages 130 Hz stimulation bursts only during
    pathological beta synchronization episodes.

    Reduces battery consumption by ~50-70% and minimizes stimulation-induced
    speech impairment (dysarthria) and motor side effects.
    """

    def __init__(
        self,
        thresholds: Optional[AdaptiveThresholds] = None,
        pulse_config: Optional[BiphasicPulseConfig] = None,
        fs: float = 20000.0  # Sampling rate in Hz (1.0 / 0.05 ms = 20 kHz)
    ) -> None:
        self.thresholds = thresholds or AdaptiveThresholds()
        self.pulse_config = pulse_config or BiphasicPulseConfig()
        self.pulse_gen = PulseGenerator(config=self.pulse_config)
        self.period_ms = 1000.0 / self.pulse_config.frequency_hz

        # Online envelope tracker
        self.envelope_tracker = HilbertEnvelopeTracker(fs=fs, low_hz=13.0, high_hz=30.0)

        # Controller state
        self.stimulating: bool = False
        self.current_burst_duration_ms: float = 0.0
        self.time_accumulator: float = 0.0

        # Diagnostics
        self.envelope_history: List[float] = []
        self.state_history: List[bool] = []
        self.total_time_ms: float = 0.0
        self.active_time_ms: float = 0.0
        self.burst_count: int = 0

    @property
    def total_energy(self) -> float:
        """Cumulative electrical energy proportional to integral(I^2 dt)."""
        return self.pulse_gen.cumulative_energy

    @property
    def total_pulses(self) -> int:
        return self.pulse_gen.total_pulses_delivered

    @property
    def duty_cycle(self) -> float:
        """Fraction of total runtime that stimulation bursts were active."""
        if self.total_time_ms <= 0:
            return 0.0
        return self.active_time_ms / self.total_time_ms

    def energy_savings_relative_to(self, c_dbs_energy: float) -> float:
        """Compute percentage energy saved compared to continuous open-loop DBS."""
        if c_dbs_energy <= 0:
            return 0.0
        saved = max(0.0, 1.0 - (self.total_energy / c_dbs_energy))
        return float(saved * 100.0)

    def compute_stimulus(
        self,
        lfp_sample: float,
        dt: float = 0.05,
        current_time: float = 0.0
    ) -> float:
        """Process incoming LFP sample and return instantaneous stimulation current."""
        self.total_time_ms += dt

        # Update envelope estimate
        envelope, phase = self.envelope_tracker.update(lfp_sample)
        self.envelope_history.append(envelope)

        # Dual-threshold hysteresis state machine
        if not self.stimulating:
            if envelope >= self.thresholds.upper_threshold:
                self.stimulating = True
                self.current_burst_duration_ms = 0.0
                self.burst_count += 1
        else:
            self.current_burst_duration_ms += dt
            if (
                envelope <= self.thresholds.lower_threshold
                and self.current_burst_duration_ms >= self.thresholds.min_burst_duration_ms
            ):
                self.stimulating = False

        self.state_history.append(self.stimulating)

        # Generate stimulation pulse if in active burst state
        stim_current = 0.0
        if self.stimulating:
            self.time_accumulator += dt
            trigger = False
            if self.time_accumulator >= self.period_ms:
                trigger = True
                self.time_accumulator -= self.period_ms

            stim_current = self.pulse_gen.step(dt=dt, trigger=trigger)
            if abs(stim_current) > 1e-4:
                self.active_time_ms += dt
        else:
            self.pulse_gen.step(dt=dt, trigger=False)

        return stim_current
