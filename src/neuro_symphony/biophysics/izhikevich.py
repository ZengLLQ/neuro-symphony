"""Izhikevich 2D quadratic integrate-and-fire neuronal dynamics.

Supports diverse neuro-computational firing patterns:
- Regular Spiking (RS): typical cortical pyramidal excitatory neurons
- Fast Spiking (FS): GABAergic cortical/striatal interneurons
- Intrinsic Bursting (IB): layer V pyramidal neurons
- Chattering (CH): fast rhythmic bursting in visual cortex
- Low-Threshold Spiking (LTS): striatal interneurons
- Thalamocortical (TC): thalamic relay neurons with rebound bursting
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class IzhikevichFiringPattern(str, Enum):
    """Canonical electrophysiological firing patterns described by E.M. Izhikevich (2003)."""
    REGULAR_SPIKING = "regular_spiking"
    FAST_SPIKING = "fast_spiking"
    INTRINSIC_BURSTING = "intrinsic_bursting"
    CHATTERING = "chattering"
    LOW_THRESHOLD_SPIKING = "low_threshold_spiking"
    THALAMOCORTICAL = "thalamocortical"


@dataclass
class IzhikevichParameters:
    """Parameters governing Izhikevich model recovery and reset dynamics.

    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= v_peak:
        v = c
        u = u + d
    """
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v_peak: float = 30.0

    @classmethod
    def from_pattern(cls, pattern: Union[IzhikevichFiringPattern, str]) -> "IzhikevichParameters":
        """Factory method returning standard parameters for canonical firing types."""
        if isinstance(pattern, str):
            pattern = IzhikevichFiringPattern(pattern)

        presets = {
            IzhikevichFiringPattern.REGULAR_SPIKING: cls(a=0.02, b=0.2, c=-65.0, d=8.0),
            IzhikevichFiringPattern.FAST_SPIKING: cls(a=0.1, b=0.2, c=-65.0, d=2.0),
            IzhikevichFiringPattern.INTRINSIC_BURSTING: cls(a=0.02, b=0.2, c=-55.0, d=4.0),
            IzhikevichFiringPattern.CHATTERING: cls(a=0.02, b=0.2, c=-50.0, d=2.0),
            IzhikevichFiringPattern.LOW_THRESHOLD_SPIKING: cls(a=0.02, b=0.25, c=-65.0, d=2.0),
            IzhikevichFiringPattern.THALAMOCORTICAL: cls(a=0.02, b=0.25, c=-65.0, d=0.05),
        }
        return presets[pattern]


class IzhikevichNeuron:
    """Single point neuron model using Izhikevich 2D non-linear differential equations."""

    def __init__(
        self,
        params: Optional[IzhikevichParameters] = None,
        pattern: Optional[IzhikevichFiringPattern] = None,
        initial_v: float = -65.0
    ) -> None:
        if params is not None:
            self.params = params
        elif pattern is not None:
            self.params = IzhikevichParameters.from_pattern(pattern)
        else:
            self.params = IzhikevichParameters.from_pattern(IzhikevichFiringPattern.REGULAR_SPIKING)

        self.v: float = initial_v
        self.u: float = self.params.b * self.v
        self.spiked: bool = False
        self.total_spikes: int = 0
        self.last_spike_time: float = -1000.0

    def step(self, dt: float, current_input: float = 0.0, current_time: float = 0.0) -> bool:
        """Advance neuron state by dt ms using two-step sub-stepping for numerical stability."""
        p = self.params

        # If previous step ended in a spike, perform reset
        if self.v >= p.v_peak:
            self.v = p.c
            self.u += p.d

        # Bound membrane potential before nonlinear square term to prevent overflow
        v_bounded = max(-100.0, min(35.0, self.v))

        # 0.5 dt sub-steps for Euler integration (recommended by Izhikevich for stability)
        half_dt = dt * 0.5
        v_dot1 = 0.04 * (v_bounded ** 2) + 5.0 * v_bounded + 140.0 - self.u + current_input
        v_mid = max(-100.0, min(35.0, v_bounded + half_dt * v_dot1))

        v_dot2 = 0.04 * (v_mid ** 2) + 5.0 * v_mid + 140.0 - self.u + current_input
        self.v = max(-100.0, min(35.0, self.v + dt * v_dot2))

        u_dot = p.a * (p.b * self.v - self.u)
        self.u = self.u + dt * u_dot

        # Spike detection
        if self.v >= p.v_peak:
            self.spiked = True
            self.last_spike_time = current_time
            self.total_spikes += 1
        else:
            self.spiked = False

        return self.spiked


class IzhikevichPopulation:
    """Vectorized ensemble of N Izhikevich neurons for high-throughput network simulation."""

    def __init__(
        self,
        size: int,
        params: Optional[IzhikevichParameters] = None,
        pattern: Optional[IzhikevichFiringPattern] = None,
        heterogeneity: float = 0.05
    ) -> None:
        self.size = size
        base_params = params or (
            IzhikevichParameters.from_pattern(pattern)
            if pattern is not None
            else IzhikevichParameters.from_pattern(IzhikevichFiringPattern.REGULAR_SPIKING)
        )

        # Heterogeneous parameter distribution across ensemble
        rng = np.random.default_rng(seed=42)
        noise = 1.0 + heterogeneity * (rng.standard_normal(size))

        self.a = np.full(size, base_params.a) * noise
        self.b = np.full(size, base_params.b) * noise
        self.c = np.full(size, base_params.c)
        self.d = np.full(size, base_params.d) * noise
        self.v_peak = base_params.v_peak

        self.v = np.full(size, base_params.c)
        self.u = self.b * self.v
        self.spikes = np.zeros(size, dtype=bool)
        self.total_spikes = np.zeros(size, dtype=int)
        self.last_spike_times = np.full(size, -1000.0)

    def step(
        self,
        dt: float,
        current_input: Union[float, np.ndarray] = 0.0,
        current_time: float = 0.0
    ) -> np.ndarray:
        """Step all N neurons forward in parallel."""
        # Reset neurons that spiked in previous time step
        reset_mask = self.v >= self.v_peak
        self.v[reset_mask] = self.c[reset_mask]
        self.u[reset_mask] += self.d[reset_mask]

        # Bound voltage numerically
        v_bounded = np.clip(self.v, -100.0, 35.0)

        half_dt = dt * 0.5
        v_dot1 = 0.04 * (v_bounded ** 2) + 5.0 * v_bounded + 140.0 - self.u + current_input
        v_mid = np.clip(v_bounded + half_dt * v_dot1, -100.0, 35.0)

        v_dot2 = 0.04 * (v_mid ** 2) + 5.0 * v_mid + 140.0 - self.u + current_input
        self.v = np.clip(self.v + dt * v_dot2, -100.0, 35.0)

        u_dot = self.a * (self.b * self.v - self.u)
        self.u += dt * u_dot

        # Detect new spikes
        self.spikes = self.v >= self.v_peak
        self.total_spikes[self.spikes] += 1
        self.last_spike_times[self.spikes] = current_time

        return self.spikes
