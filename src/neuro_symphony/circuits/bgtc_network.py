"""Basal Ganglia-Thalamocortical (BGTC) looping network and Parkinsonian dynamics."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from neuro_symphony.biophysics.izhikevich import IzhikevichFiringPattern
from neuro_symphony.biophysics.synapse import SynapseType
from neuro_symphony.circuits.connectivity import ConnectivityMatrix, ProjectionConfig, SynapticProjection
from neuro_symphony.circuits.populations import (
    NeuronalPopulation,
    PopulationConfig,
    PopulationType,
)


@dataclass
class BGTCParameters:
    """Network-level parameters for the basal ganglia-thalamocortical circuit."""
    # Simulation timing
    dt: float = 0.05  # ms
    total_time: float = 1000.0  # ms

    # Dopamine tone [0.0 = severe Parkinson's, 1.0 = healthy basal state]
    dopamine_level: float = 1.0

    # Population ensemble sizes
    n_m1_pyr: int = 40
    n_m1_int: int = 10
    n_dmsn: int = 50
    n_imsn: int = 50
    n_stn: int = 40
    n_gpe: int = 40
    n_gpi: int = 30
    n_thalamus: int = 30

    # Synaptic pathway strengths (calibrated for beta emergence under dopamine depletion)
    w_m1_stn_hyperdirect: float = 1.8
    w_m1_str: float = 1.6
    w_dmsn_gpi: float = 2.4
    w_imsn_gpe: float = 2.8
    w_gpe_stn: float = 3.2
    w_stn_gpe: float = 3.0
    w_stn_gpi: float = 2.0
    w_gpi_thal: float = 3.5
    w_thal_m1: float = 1.9

    # Recurrent STN synchronizing collateral gain unmasked by dopamine depletion
    stn_collateral_sync_gain: float = 0.6


class KuramotoOrder:
    """Computes Kuramoto synchronization order parameter and Phase Locking Value (PLV).

    Order parameter:
    R(t) = 1/N * | sum_{j=1}^N exp(i * theta_j(t)) |
    where R in [0, 1] measures global phase coherence.
    """

    @staticmethod
    def instantaneous_order_parameter(phases: np.ndarray) -> float:
        """Compute Kuramoto R(t) from array of neuronal phase angles in radians."""
        if len(phases) == 0:
            return 0.0
        z = np.mean(np.exp(1j * phases))
        return float(np.abs(z))

    @staticmethod
    def phase_locking_value(phases1: np.ndarray, phases2: np.ndarray) -> float:
        """Compute Phase-Locking Value (PLV) between two phase time series:

        PLV = 1/T * | sum_{t=1}^T exp(i * (theta1(t) - theta2(t))) |
        """
        if len(phases1) == 0 or len(phases2) == 0:
            return 0.0
        diff = phases1 - phases2
        plv = np.abs(np.mean(np.exp(1j * diff)))
        return float(plv)


class BGTCNetwork:
    """Basal Ganglia-Thalamocortical Spiking Neural Network."""

    def __init__(
        self,
        params: Optional[BGTCParameters] = None,
        seed: Optional[int] = 42
    ) -> None:
        self.params = params or BGTCParameters()
        self.dt = self.params.dt
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.seed = seed

        # Build anatomical populations
        self.populations: Dict[PopulationType, NeuronalPopulation] = {}
        self._initialize_populations()

        # Build anatomical connections
        self.connectivity = ConnectivityMatrix(dt=self.dt)
        self.projections: Dict[str, SynapticProjection] = {}
        self._initialize_connectivity()

        # Buffers for history and diagnostics
        self.stn_lfp_history: List[float] = []
        self.m1_lfp_history: List[float] = []
        self.time_history: List[float] = []
        self.kuramoto_history: List[float] = []
        self.dbs_active_history: List[bool] = []
        self.spike_raster: Dict[PopulationType, List[Tuple[float, int]]] = {
            pop_type: [] for pop_type in PopulationType
        }

    def _initialize_populations(self) -> None:
        p = self.params

        # M1 Cortex: Regular Spiking Pyramidal & Fast Spiking Interneurons
        self.populations[PopulationType.M1_PYRAMIDAL] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.M1_PYRAMIDAL,
                size=p.n_m1_pyr,
                pattern=IzhikevichFiringPattern.REGULAR_SPIKING,
                baseline_drive=4.5,
                noise_amplitude=2.0
            ),
            seed=self.seed + 1 if self.seed else None
        )
        self.populations[PopulationType.M1_INTERNEURON] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.M1_INTERNEURON,
                size=p.n_m1_int,
                pattern=IzhikevichFiringPattern.FAST_SPIKING,
                baseline_drive=2.0,
                noise_amplitude=1.5
            ),
            seed=self.seed + 2 if self.seed else None
        )

        # Striatum: dMSN (D1: positive DA sensitivity) & iMSN (D2: negative DA sensitivity)
        self.populations[PopulationType.STRIATUM_DMSN] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.STRIATUM_DMSN,
                size=p.n_dmsn,
                pattern=IzhikevichFiringPattern.REGULAR_SPIKING,
                baseline_drive=3.0,
                noise_amplitude=1.5,
                dopamine_sensitivity=4.0  # DA increases dMSN activity
            ),
            seed=self.seed + 3 if self.seed else None
        )
        self.populations[PopulationType.STRIATUM_IMSN] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.STRIATUM_IMSN,
                size=p.n_imsn,
                pattern=IzhikevichFiringPattern.REGULAR_SPIKING,
                baseline_drive=4.5,
                noise_amplitude=1.5,
                dopamine_sensitivity=-5.0  # DA depletion strongly excites iMSN
            ),
            seed=self.seed + 4 if self.seed else None
        )

        # STN: Subthalamic Nucleus (excitatory, bursty, thalamocortical-like rebound)
        self.populations[PopulationType.STN] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.STN,
                size=p.n_stn,
                pattern=IzhikevichFiringPattern.THALAMOCORTICAL,
                baseline_drive=5.0,
                noise_amplitude=2.0
            ),
            seed=self.seed + 5 if self.seed else None
        )

        # GPe: Globus Pallidus externa (autonomous fast pacemakers, inhibited by iMSN)
        self.populations[PopulationType.GPE] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.GPE,
                size=p.n_gpe,
                pattern=IzhikevichFiringPattern.FAST_SPIKING,
                baseline_drive=8.0,
                noise_amplitude=2.0
            ),
            seed=self.seed + 6 if self.seed else None
        )

        # GPi: Globus Pallidus interna / SNr (output nucleus, high tonic inhibition to thalamus)
        self.populations[PopulationType.GPI] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.GPI,
                size=p.n_gpi,
                pattern=IzhikevichFiringPattern.FAST_SPIKING,
                baseline_drive=7.0,
                noise_amplitude=2.0
            ),
            seed=self.seed + 7 if self.seed else None
        )

        # Thalamus: Thalamocortical relay
        self.populations[PopulationType.THALAMUS] = NeuronalPopulation(
            PopulationConfig(
                pop_type=PopulationType.THALAMUS,
                size=p.n_thalamus,
                pattern=IzhikevichFiringPattern.THALAMOCORTICAL,
                baseline_drive=3.5,
                noise_amplitude=1.5
            ),
            seed=self.seed + 8 if self.seed else None
        )

    def _initialize_connectivity(self) -> None:
        p = self.params

        def add_proj(name: str, src: PopulationType, tgt: PopulationType, r: SynapseType, w: float, delay: float) -> None:
            config = ProjectionConfig(
                source=src,
                target=tgt,
                receptor=r,
                weight=w,
                probability=0.35,
                delay_ms=delay
            )
            proj = self.connectivity.add_projection(
                config=config,
                source_size=self.populations[src].size,
                target_size=self.populations[tgt].size,
                seed=self.seed + len(self.projections) if self.seed else None
            )
            self.projections[name] = proj

        # 1. Hyperdirect pathway (M1 Pyramidal -> STN)
        add_proj("m1_stn", PopulationType.M1_PYRAMIDAL, PopulationType.STN, SynapseType.AMPA, p.w_m1_stn_hyperdirect, 2.5)

        # 2. Corticostriatal pathway (M1 Pyramidal -> dMSN and iMSN)
        add_proj("m1_dmsn", PopulationType.M1_PYRAMIDAL, PopulationType.STRIATUM_DMSN, SynapseType.AMPA, p.w_m1_str, 3.0)
        add_proj("m1_imsn", PopulationType.M1_PYRAMIDAL, PopulationType.STRIATUM_IMSN, SynapseType.AMPA, p.w_m1_str, 3.0)

        # 3. Direct pathway (dMSN -> GPi)
        add_proj("dmsn_gpi", PopulationType.STRIATUM_DMSN, PopulationType.GPI, SynapseType.GABA_A, p.w_dmsn_gpi, 4.0)

        # 4. Indirect pathway (iMSN -> GPe -> STN -> GPi)
        add_proj("imsn_gpe", PopulationType.STRIATUM_IMSN, PopulationType.GPE, SynapseType.GABA_A, p.w_imsn_gpe, 4.0)
        add_proj("gpe_stn", PopulationType.GPE, PopulationType.STN, SynapseType.GABA_A, p.w_gpe_stn, 3.5)
        add_proj("stn_gpe", PopulationType.STN, PopulationType.GPE, SynapseType.AMPA, p.w_stn_gpe, 3.0)
        add_proj("stn_gpi", PopulationType.STN, PopulationType.GPI, SynapseType.AMPA, p.w_stn_gpi, 2.5)

        # 5. Output and feedback (GPi -> Thalamus -> M1)
        add_proj("gpi_thal", PopulationType.GPI, PopulationType.THALAMUS, SynapseType.GABA_A, p.w_gpi_thal, 3.0)
        add_proj("thal_m1", PopulationType.THALAMUS, PopulationType.M1_PYRAMIDAL, SynapseType.AMPA, p.w_thal_m1, 2.5)

    def set_dopamine_level(self, da_level: float) -> None:
        """Set dopamine tone [0.0 = Parkinsonian, 1.0 = healthy]."""
        self.params.dopamine_level = float(np.clip(da_level, 0.0, 1.0))

    def apply_dbs(self, target_population: PopulationType, current_amplitude: float) -> None:
        """Deliver DBS current pulse to target population."""
        self.populations[target_population].set_dbs_current(current_amplitude)

    def step(self, dbs_current: float = 0.0, dbs_target: PopulationType = PopulationType.STN) -> Dict[str, float]:
        """Perform one time step (dt ms) across the entire BGTC loop.

        Returns:
            Dictionary with current metrics (STN LFP, Kuramoto R, etc.)
        """
        # Inject DBS if active
        if dbs_current != 0.0:
            self.apply_dbs(dbs_target, dbs_current)
            self.dbs_active_history.append(True)
        else:
            self.dbs_active_history.append(False)

        # 1. Clear previous inputs
        for pop in self.populations.values():
            pop.synaptic_input.fill(0.0)

        # 2. Compute projections transmission
        for name, proj in self.projections.items():
            src_spikes = self.populations[proj.config.source].spikes
            g_syn = proj.step(src_spikes)
            tgt_pop = self.populations[proj.config.target]

            # Invert sign for inhibitory synapses (GABA_A reduces potential)
            if proj.config.receptor == SynapseType.GABA_A:
                current_contrib = -g_syn * (tgt_pop.neurons.v - (-70.0)) * 0.15
            else:
                current_contrib = -g_syn * (tgt_pop.neurons.v - 0.0) * 0.15

            tgt_pop.synaptic_input += current_contrib

        # 3. Recurrent STN collateral synchronizing drive unmasked under Parkinsonian dopamine loss
        da_depletion = max(0.0, 1.0 - self.params.dopamine_level)
        stn = self.populations[PopulationType.STN]
        if da_depletion > 0:
            v_stn_mean = float(np.mean(stn.neurons.v))
            # Collateral coupling driving phase alignment across STN ensemble
            stn.synaptic_input += self.params.stn_collateral_sync_gain * da_depletion * (v_stn_mean - stn.neurons.v)

        # 4. Advance all populations
        da = self.params.dopamine_level
        for pop_type, pop in self.populations.items():
            spikes = pop.step(self.dt, dopamine_level=da, current_time=self.current_time)

            # Record spike raster
            spike_indices = np.where(spikes)[0]
            for idx in spike_indices:
                self.spike_raster[pop_type].append((self.current_time, int(idx)))

        # 5. Synthesize STN Local Field Potential (LFP)
        stn_lfp = float(np.mean(stn.neurons.v) + 0.4 * np.mean(stn.synaptic_input))
        self.stn_lfp_history.append(stn_lfp)

        # M1 LFP
        m1 = self.populations[PopulationType.M1_PYRAMIDAL]
        m1_lfp = float(np.mean(m1.neurons.v))
        self.m1_lfp_history.append(m1_lfp)

        self.time_history.append(self.current_time)

        # 6. Compute STN population Kuramoto order parameter
        # Instantaneous phase on limit-cycle relative to baseline center (-55 mV, -11 uA)
        stn_phases = np.arctan2(stn.neurons.u - (-11.0), stn.neurons.v - (-55.0))
        r_kuramoto = KuramotoOrder.instantaneous_order_parameter(stn_phases)
        self.kuramoto_history.append(r_kuramoto)

        # Clear DBS input after step so next step requires explicit injection
        self.populations[dbs_target].dbs_current.fill(0.0)

        self.current_time += self.dt
        self.step_count += 1

        return {
            "time": self.current_time,
            "stn_lfp": stn_lfp,
            "kuramoto_r": r_kuramoto,
            "dbs_active": dbs_current != 0.0
        }

    def run_simulation(
        self,
        duration_ms: float,
        dbs_controller=None,
        dbs_target: PopulationType = PopulationType.STN
    ) -> Dict[str, np.ndarray]:
        """Run network for specified duration, optionally applying DBS controller."""
        n_steps = int(round(duration_ms / self.dt))

        for step_i in range(n_steps):
            dbs_val = 0.0
            if dbs_controller is not None:
                curr_lfp = self.stn_lfp_history[-1] if self.stn_lfp_history else -60.0
                dbs_val = dbs_controller.compute_stimulus(
                    lfp_sample=curr_lfp,
                    dt=self.dt,
                    current_time=self.current_time
                )

            self.step(dbs_current=dbs_val, dbs_target=dbs_target)

        return {
            "time": np.array(self.time_history),
            "stn_lfp": np.array(self.stn_lfp_history),
            "m1_lfp": np.array(self.m1_lfp_history),
            "kuramoto_r": np.array(self.kuramoto_history),
            "dbs_active": np.array(self.dbs_active_history, dtype=bool)
        }
