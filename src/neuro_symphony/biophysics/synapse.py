"""Conductance-based synaptic dynamics and transmission.

Models kinetic conductance profiles for:
- AMPA: Fast glutamatergic excitation
- NMDA: Voltage-dependent magnesium-blocked glutamatergic excitation
- GABA_A: Fast ionotropic chloride-mediated inhibition
- GABA_B: Slow metabotropic G-protein potassium-mediated inhibition
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class SynapseType(str, Enum):
    """Receptor classification for synaptic transmission."""
    AMPA = "ampa"
    NMDA = "nmda"
    GABA_A = "gaba_a"
    GABA_B = "gaba_b"


@dataclass
class SynapseParameters:
    """Biophysical kinetic parameters for synaptic conductances."""
    g_max: float = 1.0  # Maximal conductance (nS or mS/cm^2)
    e_rev: float = 0.0  # Reversal potential (mV)
    tau_rise: float = 0.5  # Rise time constant (ms)
    tau_decay: float = 2.0  # Decay time constant (ms)
    mg_conc: float = 1.0  # Extracellular Mg2+ concentration (mM, for NMDA)


class Synapse:
    """Base conductance-based synapse using dual-exponential kinetics.

    The normalized conductance waveform is given by:
    g(t) = g_max * factor * (exp(-t / tau_decay) - exp(-t / tau_rise))
    Implemented as a 2-variable linear differential system for continuous stepping:
    dr/dt = -r / tau_rise
    dd/dt = -d / tau_decay
    g = g_max * factor * (d - r)
    """

    def __init__(
        self,
        synapse_type: SynapseType,
        params: Optional[SynapseParameters] = None
    ) -> None:
        self.synapse_type = synapse_type
        self.params = params or self._default_params(synapse_type)

        # Calculate normalization factor so peak conductance equals g_max
        tau_r = self.params.tau_rise
        tau_d = self.params.tau_decay
        if abs(tau_d - tau_r) > 1e-6:
            t_peak = (tau_r * tau_d) / (tau_d - tau_r) * math.log(tau_d / tau_r)
            peak_val = math.exp(-t_peak / tau_d) - math.exp(-t_peak / tau_r)
            self.factor = 1.0 / peak_val if peak_val > 1e-9 else 1.0
        else:
            self.factor = math.e

        # Internal state variables for rise and decay
        self.r_state: float = 0.0
        self.d_state: float = 0.0
        self.conductance: float = 0.0

    @staticmethod
    def _default_params(syn_type: SynapseType) -> SynapseParameters:
        if syn_type == SynapseType.AMPA:
            return SynapseParameters(g_max=1.0, e_rev=0.0, tau_rise=0.5, tau_decay=2.0)
        elif syn_type == SynapseType.NMDA:
            return SynapseParameters(g_max=0.6, e_rev=0.0, tau_rise=4.0, tau_decay=40.0, mg_conc=1.0)
        elif syn_type == SynapseType.GABA_A:
            return SynapseParameters(g_max=1.2, e_rev=-70.0, tau_rise=0.5, tau_decay=6.0)
        elif syn_type == SynapseType.GABA_B:
            return SynapseParameters(g_max=0.4, e_rev=-95.0, tau_rise=20.0, tau_decay=100.0)
        raise ValueError(f"Unknown synapse type: {syn_type}")

    def on_presynaptic_spike(self, weight: float = 1.0) -> None:
        """Trigger conductance influx upon presynaptic action potential."""
        self.r_state += weight
        self.d_state += weight

    def step(self, dt: float) -> float:
        """Advance internal gating variables by dt ms and update conductance."""
        p = self.params
        self.r_state += - (dt / p.tau_rise) * self.r_state
        self.d_state += - (dt / p.tau_decay) * self.d_state

        raw_diff = max(0.0, self.d_state - self.r_state)
        self.conductance = p.g_max * self.factor * raw_diff
        return self.conductance

    def current(self, v_post: float) -> float:
        """Compute instantaneous postsynaptic current I_syn = g * (V - E_rev)."""
        return self.conductance * (v_post - self.params.e_rev)


class AMPASynapse(Synapse):
    """Fast glutamatergic ionotropic excitatory synapse."""

    def __init__(self, params: Optional[SynapseParameters] = None) -> None:
        super().__init__(synapse_type=SynapseType.AMPA, params=params)


class NMDASynapse(Synapse):
    """Slow glutamatergic excitatory synapse with voltage-dependent magnesium block.

    Block factor B(V) = 1 / (1 + [Mg2+] / 3.57 * exp(-0.062 * V))
    Current I_NMDA = g_NMDA * B(V) * (V - E_rev)
    """

    def __init__(self, params: Optional[SynapseParameters] = None) -> None:
        super().__init__(synapse_type=SynapseType.NMDA, params=params)

    def magnesium_block(self, v_post: float) -> float:
        """Voltage-dependent unblocking fraction B(V) in [0, 1]."""
        p = self.params
        return 1.0 / (1.0 + (p.mg_conc / 3.57) * math.exp(-0.062 * v_post))

    def current(self, v_post: float) -> float:
        """Calculate NMDA current modulated by magnesium block."""
        b_v = self.magnesium_block(v_post)
        return self.conductance * b_v * (v_post - self.params.e_rev)


class GABAASynapse(Synapse):
    """Fast ionotropic GABA_A inhibitory synapse (Cl- influx, E_rev ~ -70 mV)."""

    def __init__(self, params: Optional[SynapseParameters] = None) -> None:
        super().__init__(synapse_type=SynapseType.GABA_A, params=params)


class GABABSynapse(Synapse):
    """Slow metabotropic GABA_B inhibitory synapse (K+ efflux, E_rev ~ -95 mV)."""

    def __init__(self, params: Optional[SynapseParameters] = None) -> None:
        super().__init__(synapse_type=SynapseType.GABA_B, params=params)


class SynapticCurrentPool:
    """Aggregates multi-synaptic input currents (AMPA, NMDA, GABA_A, GABA_B) for a neuron."""

    def __init__(self) -> None:
        self.ampa = AMPASynapse()
        self.nmda = NMDASynapse()
        self.gaba_a = GABAASynapse()
        self.gaba_b = GABABSynapse()

    def receive_spike(self, syn_type: SynapseType, weight: float = 1.0) -> None:
        """Deliver presynaptic spike to corresponding receptor pool."""
        if syn_type == SynapseType.AMPA:
            self.ampa.on_presynaptic_spike(weight)
        elif syn_type == SynapseType.NMDA:
            self.nmda.on_presynaptic_spike(weight)
        elif syn_type == SynapseType.GABA_A:
            self.gaba_a.on_presynaptic_spike(weight)
        elif syn_type == SynapseType.GABABSynapse or syn_type == SynapseType.GABA_B:
            self.gaba_b.on_presynaptic_spike(weight)

    def step(self, dt: float) -> None:
        """Step all synaptic receptor pools forward."""
        self.ampa.step(dt)
        self.nmda.step(dt)
        self.gaba_a.step(dt)
        self.gaba_b.step(dt)

    def total_current(self, v_post: float) -> float:
        """Sum total synaptic current across all receptors."""
        return (
            self.ampa.current(v_post)
            + self.nmda.current(v_post)
            + self.gaba_a.current(v_post)
            + self.gaba_b.current(v_post)
        )
