"""Multi-compartmental biophysical Hodgkin-Huxley neuron dynamics.

Includes active ionic currents:
- Fast sodium current (I_Na) with activation (m) and inactivation (h)
- Delayed rectifier potassium current (I_K) with activation (n)
- Leak current (I_L)
- High-threshold L-type calcium current (I_Ca) with activation (s)
- Low-threshold T-type calcium current (I_T) with activation (a) and inactivation (b)
- Calcium-activated potassium current (I_AHP) for afterhyperpolarization
- Intracellular calcium dynamics d[Ca]/dt
- Multi-compartment soma-dendrite coupling via axial conductance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HodgkinHuxleyParameters:
    """Biophysical parameters for Hodgkin-Huxley ionic currents and kinetics."""
    # Capacitance (uF/cm^2)
    c_m: float = 1.0

    # Maximal conductances (mS/cm^2)
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    g_ca: float = 2.0
    g_t: float = 1.5
    g_ahp: float = 1.0

    # Reversal potentials (mV)
    e_na: float = 55.0
    e_k: float = -77.0
    e_l: float = -54.4
    e_ca: float = 120.0

    # Calcium dynamics
    ca_decay: float = 0.05  # ms^-1 (tau_ca = 20 ms)
    ca_influx_factor: float = 0.002  # uM / (nA * ms)
    ca_rest: float = 0.05  # uM
    k_ahp: float = 1.0  # uM (half-activation for I_AHP)

    # Spike detection threshold (mV)
    spike_threshold: float = 0.0


class HodgkinHuxleyNeuron:
    """Single-compartment biophysical neuron with multiple ionic conductances.

    Implements:
    C_m * dV/dt = - (I_Na + I_K + I_L + I_Ca + I_T + I_AHP + I_syn) + I_ext
    """

    def __init__(
        self,
        params: Optional[HodgkinHuxleyParameters] = None,
        initial_v: float = -65.0,
        initial_ca: float = 0.05
    ) -> None:
        self.params = params or HodgkinHuxleyParameters()
        self.v: float = initial_v
        self.ca: float = initial_ca

        # Steady-state gating variables at initial V
        self.m: float = self._alpha_m(self.v) / (self._alpha_m(self.v) + self._beta_m(self.v))
        self.h: float = self._alpha_h(self.v) / (self._alpha_h(self.v) + self._beta_h(self.v))
        self.n: float = self._alpha_n(self.v) / (self._alpha_n(self.v) + self._beta_n(self.v))
        self.s: float = self._s_inf(self.v)
        self.a: float = self._a_inf(self.v)
        self.b: float = self._b_inf(self.v)

        # State tracking
        self.spiked: bool = False
        self.last_spike_time: float = -1000.0
        self.total_spikes: int = 0
        self.current_i_na: float = 0.0
        self.current_i_k: float = 0.0
        self.current_i_ca: float = 0.0
        self.current_i_t: float = 0.0
        self.current_i_ahp: float = 0.0
        self.current_i_l: float = 0.0

    @staticmethod
    def _alpha_m(v: float) -> float:
        num = 0.1 * (v + 40.0)
        denom = 1.0 - math.exp(-0.1 * (v + 40.0)) if abs(v + 40.0) > 1e-6 else 1e-6
        return num / denom

    @staticmethod
    def _beta_m(v: float) -> float:
        return 4.0 * math.exp(-0.0556 * (v + 65.0))

    @staticmethod
    def _alpha_h(v: float) -> float:
        return 0.07 * math.exp(-0.05 * (v + 65.0))

    @staticmethod
    def _beta_h(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-0.1 * (v + 35.0)))

    @staticmethod
    def _alpha_n(v: float) -> float:
        num = 0.01 * (v + 55.0)
        denom = 1.0 - math.exp(-0.1 * (v + 55.0)) if abs(v + 55.0) > 1e-6 else 1e-6
        return num / denom

    @staticmethod
    def _beta_n(v: float) -> float:
        return 0.125 * math.exp(-0.0125 * (v + 65.0))

    @staticmethod
    def _s_inf(v: float) -> float:
        """High-threshold Ca current steady-state activation."""
        return 1.0 / (1.0 + math.exp(-(v + 20.0) / 9.0))

    @staticmethod
    def _a_inf(v: float) -> float:
        """Low-threshold T-type Ca steady-state activation."""
        return 1.0 / (1.0 + math.exp(-(v + 63.0) / 7.8))

    @staticmethod
    def _b_inf(v: float) -> float:
        """Low-threshold T-type Ca steady-state inactivation."""
        return 1.0 / (1.0 + math.exp((v + 84.0) / 4.0))

    def step(self, dt: float, i_ext: float = 0.0, i_syn: float = 0.0, current_time: float = 0.0) -> bool:
        """Advance the neuron by dt (ms).

        Returns:
            bool: True if an action potential spike occurred on this step.
        """
        p = self.params
        v = self.v

        # Ionic currents (uA/cm^2)
        self.current_i_na = p.g_na * (self.m ** 3) * self.h * (v - p.e_na)
        self.current_i_k = p.g_k * (self.n ** 4) * (v - p.e_k)
        self.current_i_l = p.g_l * (v - p.e_l)
        self.current_i_ca = p.g_ca * (self.s ** 2) * (v - p.e_ca)
        self.current_i_t = p.g_t * (self.a ** 3) * self.b * (v - p.e_ca)

        # Calcium-activated potassium current
        ahp_factor = self.ca / (self.ca + p.k_ahp) if (self.ca + p.k_ahp) > 0 else 0.0
        self.current_i_ahp = p.g_ahp * ahp_factor * (v - p.e_k)

        # Membrane potential derivative
        i_ionic = (
            self.current_i_na
            + self.current_i_k
            + self.current_i_l
            + self.current_i_ca
            + self.current_i_t
            + self.current_i_ahp
        )
        dv_dt = (i_ext - i_syn - i_ionic) / p.c_m

        # Gating variable kinetics
        dm_dt = self._alpha_m(v) * (1.0 - self.m) - self._beta_m(v) * self.m
        dh_dt = self._alpha_h(v) * (1.0 - self.h) - self._beta_h(v) * self.h
        dn_dt = self._alpha_n(v) * (1.0 - self.n) - self._beta_n(v) * self.n

        # Calcium channels kinetics (time constants approx 5-20 ms)
        tau_s = 8.0
        tau_a = 4.0
        tau_b = 30.0
        ds_dt = (self._s_inf(v) - self.s) / tau_s
        da_dt = (self._a_inf(v) - self.a) / tau_a
        db_dt = (self._b_inf(v) - self.b) / tau_b

        # Intracellular calcium dynamics
        total_ca_influx = max(0.0, -(self.current_i_ca + self.current_i_t))
        dca_dt = (
            p.ca_influx_factor * total_ca_influx
            - p.ca_decay * (self.ca - p.ca_rest)
        )

        # Update states (forward Euler with bounds)
        v_next = v + dt * dv_dt
        self.m = float(np.clip(self.m + dt * dm_dt, 0.0, 1.0))
        self.h = float(np.clip(self.h + dt * dh_dt, 0.0, 1.0))
        self.n = float(np.clip(self.n + dt * dn_dt, 0.0, 1.0))
        self.s = float(np.clip(self.s + dt * ds_dt, 0.0, 1.0))
        self.a = float(np.clip(self.a + dt * da_dt, 0.0, 1.0))
        self.b = float(np.clip(self.b + dt * db_dt, 0.0, 1.0))
        self.ca = float(max(0.001, self.ca + dt * dca_dt))

        # Spike detection: upward crossing of threshold
        was_below = self.v < p.spike_threshold
        now_above = v_next >= p.spike_threshold
        refractory_passed = (current_time - self.last_spike_time) > 1.5

        if was_below and now_above and refractory_passed:
            self.spiked = True
            self.last_spike_time = current_time
            self.total_spikes += 1
        else:
            self.spiked = False

        self.v = v_next
        return self.spiked


class CompartmentalNeuron:
    """Two-compartment (soma + dendritic arbor) biophysical neuron model.

    Features active spiking ionic channels in the soma and synaptic receptor/
    calcium conductances in the dendrite, coupled via axial conductance g_axial:
    I_axial = g_axial * (V_soma - V_dendrite)
    """

    def __init__(
        self,
        g_axial: float = 1.5,
        soma_params: Optional[HodgkinHuxleyParameters] = None,
        dend_params: Optional[HodgkinHuxleyParameters] = None,
        initial_v: float = -65.0
    ) -> None:
        self.g_axial = g_axial
        self.soma = HodgkinHuxleyNeuron(params=soma_params, initial_v=initial_v)
        self.dendrite = HodgkinHuxleyNeuron(params=dend_params, initial_v=initial_v)
        # In dendrites, fast sodium is typically lower and calcium higher
        self.dendrite.params.g_na = 20.0
        self.dendrite.params.g_ca = 5.0
        self.dendrite.params.g_t = 3.0

        self.spiked: bool = False

    @property
    def v_soma(self) -> float:
        return self.soma.v

    @property
    def v_dend(self) -> float:
        return self.dendrite.v

    def step(
        self,
        dt: float,
        i_soma_ext: float = 0.0,
        i_dend_syn: float = 0.0,
        current_time: float = 0.0
    ) -> bool:
        """Step both compartments forward with axial current exchange."""
        # Axial current flows between soma and dendrite
        i_axial_soma = self.g_axial * (self.soma.v - self.dendrite.v)
        i_axial_dend = -i_axial_soma

        # Soma receives external drive and axial current from dendrite
        soma_spike = self.soma.step(
            dt=dt,
            i_ext=i_soma_ext - i_axial_soma,
            i_syn=0.0,
            current_time=current_time
        )

        # Dendrite receives synaptic input and axial current from soma
        self.dendrite.step(
            dt=dt,
            i_ext=-i_axial_dend,
            i_syn=i_dend_syn,
            current_time=current_time
        )

        self.spiked = soma_spike
        return self.spiked
