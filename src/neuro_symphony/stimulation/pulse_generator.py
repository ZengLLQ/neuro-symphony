"""Waveform generators for charge-balanced neurostimulation pulses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import numpy as np


class PulseShape(str, Enum):
    """Morphology of deep brain stimulation pulses."""
    BIPHASIC_RECTANGULAR = "biphasic_rectangular"
    BIPHASIC_GAUSSIAN = "biphasic_gaussian"
    ASYMMETRIC_CHARGE_BALANCED = "asymmetric_charge_balanced"


@dataclass
class BiphasicPulseConfig:
    """Parameters defining a charge-balanced biphasic stimulation pulse.

    Clinical DBS pulses require zero net direct current (DC) charge transfer
    to prevent electrochemical electrode corrosion and neurotoxic tissue damage:
    Q_net = integral(I(t) dt) = 0
    """
    amplitude: float = 3.0  # mA or normalized stimulus intensity
    pulse_width_ms: float = 0.09  # 90 microseconds (clinical standard: 60-120 us)
    interphase_gap_ms: float = 0.02  # 20 microseconds
    recovery_ratio: float = 1.0  # Anodal to cathodal amplitude ratio (1.0 = symmetric)
    frequency_hz: float = 130.0  # Clinical standard high-frequency DBS
    shape: PulseShape = PulseShape.BIPHASIC_RECTANGULAR


class PulseGenerator:
    """Generates continuous or triggered stimulation pulse waveforms."""

    def __init__(self, config: Optional[BiphasicPulseConfig] = None) -> None:
        self.config = config or BiphasicPulseConfig()
        self.period_ms = 1000.0 / self.config.frequency_hz
        self.time_since_last_pulse = 9999.0  # ready to fire
        self.cumulative_charge: float = 0.0
        self.cumulative_energy: float = 0.0
        self.total_pulses_delivered: int = 0

    def generate_single_pulse(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-resolution sample array of one biphasic pulse.

        Returns:
            Tuple of (time_array, current_array)
        """
        cfg = self.config
        t_cath = cfg.pulse_width_ms
        t_gap = cfg.interphase_gap_ms
        t_anod = cfg.pulse_width_ms / cfg.recovery_ratio

        total_duration = t_cath + t_gap + t_anod + 0.05
        time_vec = np.arange(0, total_duration, dt)
        current_vec = np.zeros_like(time_vec)

        for i, t in enumerate(time_vec):
            if t < t_cath:
                # Cathodal phase (negative / depolarizing)
                current_vec[i] = -cfg.amplitude
            elif t < t_cath + t_gap:
                # Interphase delay
                current_vec[i] = 0.0
            elif t < t_cath + t_gap + t_anod:
                # Anodal charge-recovery phase (positive)
                current_vec[i] = cfg.amplitude * cfg.recovery_ratio

        return time_vec, current_vec

    def evaluate_at_offset(self, time_in_pulse_ms: float) -> float:
        """Evaluate instantaneous pulse amplitude at elapsed offset since pulse trigger."""
        cfg = self.config
        t_cath = cfg.pulse_width_ms
        t_gap = cfg.interphase_gap_ms
        t_anod = cfg.pulse_width_ms / cfg.recovery_ratio

        if time_in_pulse_ms < 0:
            return 0.0
        elif time_in_pulse_ms < t_cath:
            if cfg.shape == PulseShape.BIPHASIC_GAUSSIAN:
                sigma = t_cath / 3.0
                center = t_cath / 2.0
                return -cfg.amplitude * math.exp(-0.5 * ((time_in_pulse_ms - center) / sigma) ** 2)
            return -cfg.amplitude
        elif time_in_pulse_ms < t_cath + t_gap:
            return 0.0
        elif time_in_pulse_ms < t_cath + t_gap + t_anod:
            anod_offset = time_in_pulse_ms - (t_cath + t_gap)
            if cfg.shape == PulseShape.BIPHASIC_GAUSSIAN:
                sigma = t_anod / 3.0
                center = t_anod / 2.0
                return cfg.amplitude * cfg.recovery_ratio * math.exp(-0.5 * ((anod_offset - center) / sigma) ** 2)
            return cfg.amplitude * cfg.recovery_ratio
        return 0.0

    def trigger_pulse(self) -> None:
        """Manually trigger a new pulse event."""
        self.time_since_last_pulse = 0.0
        self.total_pulses_delivered += 1

    def step(self, dt: float, trigger: bool = False) -> float:
        """Advance time by dt ms and compute current output."""
        if trigger:
            self.trigger_pulse()
        else:
            self.time_since_last_pulse += dt

        stim_current = self.evaluate_at_offset(self.time_since_last_pulse)

        # Update electrical metrics
        self.cumulative_charge += stim_current * dt
        self.cumulative_energy += (stim_current ** 2) * dt

        return stim_current
