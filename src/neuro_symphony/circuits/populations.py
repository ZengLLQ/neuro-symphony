"""Neuronal population representations across the basal ganglia-thalamocortical circuit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from neuro_symphony.biophysics.izhikevich import (
    IzhikevichFiringPattern,
    IzhikevichParameters,
    IzhikevichPopulation,
)


class PopulationType(str, Enum):
    """Anatomical brain regions and cell populations within the BGTC loop."""
    M1_PYRAMIDAL = "m1_pyramidal"
    M1_INTERNEURON = "m1_interneuron"
    STRIATUM_DMSN = "striatum_dmsn"
    STRIATUM_IMSN = "striatum_imsn"
    STN = "stn"  # Subthalamic Nucleus
    GPE = "gpe"  # Globus Pallidus externa
    GPI = "gpi"  # Globus Pallidus interna / SNr
    THALAMUS = "thalamus"  # Thalamocortical relay nucleus


@dataclass
class PopulationConfig:
    """Configuration for a single neuronal population ensemble."""
    pop_type: PopulationType
    size: int
    pattern: IzhikevichFiringPattern
    baseline_drive: float
    noise_amplitude: float = 1.5
    dopamine_sensitivity: float = 0.0  # + for D1 (dMSN), - for D2 (iMSN)


class NeuronalPopulation:
    """Spiking neuronal ensemble representing an anatomical nucleus."""

    def __init__(
        self,
        config: PopulationConfig,
        heterogeneity: float = 0.06,
        seed: Optional[int] = None
    ) -> None:
        self.config = config
        self.size = config.size
        self.pop_type = config.pop_type
        self.rng = np.random.default_rng(seed=seed if seed is not None else 100 + hash(config.pop_type) % 1000)

        # Initialize underlying vectorized Izhikevich population
        self.neurons = IzhikevichPopulation(
            size=self.size,
            pattern=config.pattern,
            heterogeneity=heterogeneity
        )

        # Buffers for currents and spikes
        self.synaptic_input = np.zeros(self.size, dtype=float)
        self.external_drive = np.zeros(self.size, dtype=float)
        self.dbs_current = np.zeros(self.size, dtype=float)
        self.spikes = np.zeros(self.size, dtype=bool)

        # Membrane potential history for monitoring
        self.mean_voltage: float = -65.0
        self.spike_count_recent: int = 0

    def set_dbs_current(self, current_val: Union[float, np.ndarray]) -> None:
        """Inject electrical stimulation current directly into this population."""
        if isinstance(current_val, (int, float)):
            self.dbs_current.fill(float(current_val))
        else:
            self.dbs_current = np.array(current_val, dtype=float)

    def clear_inputs(self) -> None:
        """Reset accumulated synaptic and stimulation inputs for next time step."""
        self.synaptic_input.fill(0.0)
        self.dbs_current.fill(0.0)

    def step(self, dt: float, dopamine_level: float, current_time: float) -> np.ndarray:
        """Advance population dynamics by dt ms.

        Args:
            dt: Time step in milliseconds.
            dopamine_level: Dopamine tone [0.0 (severely depleted) to 1.0 (healthy)].
            current_time: Current simulation time in milliseconds.

        Returns:
            Boolean array of length `size` indicating which neurons spiked.
        """
        # Baseline tonic drive modulated by dopamine receptor pharmacology
        # D1: excitation enhanced by DA (dMSN)
        # D2: inhibition suppressed by DA -> dopamine depletion excites iMSN
        da_modulation = self.config.dopamine_sensitivity * (dopamine_level - 1.0)
        effective_baseline = self.config.baseline_drive + da_modulation

        # Add Gaussian channel noise
        noise = self.rng.normal(0.0, self.config.noise_amplitude, size=self.size)

        total_current = (
            effective_baseline
            + noise
            + self.synaptic_input
            + self.dbs_current
        )

        self.spikes = self.neurons.step(
            dt=dt,
            current_input=total_current,
            current_time=current_time
        )

        self.mean_voltage = float(np.mean(self.neurons.v))
        self.spike_count_recent = int(np.sum(self.spikes))
        return self.spikes
