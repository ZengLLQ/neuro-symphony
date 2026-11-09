"""Synaptic connectivity matrices and axonal transmission projections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from neuro_symphony.biophysics.synapse import SynapseType
from neuro_symphony.circuits.populations import PopulationType


@dataclass
class ProjectionConfig:
    """Configuration for an axonal pathway linking two populations."""
    source: PopulationType
    target: PopulationType
    receptor: SynapseType
    weight: float
    probability: float = 0.3
    delay_ms: float = 2.0


class SynapticProjection:
    """Axonal projection connecting a source population to a target population."""

    def __init__(
        self,
        config: ProjectionConfig,
        source_size: int,
        target_size: int,
        dt: float = 0.05,
        seed: Optional[int] = None
    ) -> None:
        self.config = config
        self.source_size = source_size
        self.target_size = target_size
        self.dt = dt

        rng = np.random.default_rng(seed=seed)
        # Random sparse connection weight matrix
        mask = rng.random((target_size, source_size)) < config.probability
        self.weights = np.zeros((target_size, source_size), dtype=float)
        self.weights[mask] = config.weight

        # Synaptic delay ring buffer
        delay_steps = max(1, int(round(config.delay_ms / dt)))
        self.delay_buffer: deque = deque([np.zeros(source_size, dtype=bool) for _ in range(delay_steps)], maxlen=delay_steps)

        # Postsynaptic conductance state variables (dual exponential)
        self.tau_rise = 0.5 if config.receptor in (SynapseType.AMPA, SynapseType.GABA_A) else 4.0
        self.tau_decay = 2.5 if config.receptor == SynapseType.AMPA else (6.0 if config.receptor == SynapseType.GABA_A else 35.0)
        self.e_rev = 0.0 if config.receptor in (SynapseType.AMPA, SynapseType.NMDA) else -70.0

        self.r = np.zeros(target_size, dtype=float)
        self.d = np.zeros(target_size, dtype=float)

    def step(self, presynaptic_spikes: np.ndarray) -> np.ndarray:
        """Process incoming presynaptic spikes and compute synaptic current to targets."""
        # Push current spikes to delay buffer and retrieve delayed spikes
        self.delay_buffer.append(presynaptic_spikes.copy())
        delayed_spikes = self.delay_buffer.popleft()

        # Compute presynaptic input arrival at targets
        if np.any(delayed_spikes):
            spike_vec = delayed_spikes.astype(float)
            net_spike_input = self.weights @ spike_vec
            self.r += net_spike_input
            self.d += net_spike_input

        # Update continuous conductances
        self.r -= (self.dt / self.tau_rise) * self.r
        self.d -= (self.dt / self.tau_decay) * self.d

        conductance = np.maximum(0.0, self.d - self.r)
        return conductance


class ConnectivityMatrix:
    """Manages all anatomical projections across the BGTC circuit."""

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = dt
        self.projections: List[SynapticProjection] = []

    def add_projection(
        self,
        config: ProjectionConfig,
        source_size: int,
        target_size: int,
        seed: Optional[int] = None
    ) -> SynapticProjection:
        """Register a new axonal tract."""
        proj = SynapticProjection(
            config=config,
            source_size=source_size,
            target_size=target_size,
            dt=self.dt,
            seed=seed
        )
        self.projections.append(proj)
        return proj
