"""Unit tests for BGTC looping network, populations, and connectivity."""

from __future__ import annotations

import math
import numpy as np
import pytest

from neuro_symphony.biophysics.izhikevich import IzhikevichFiringPattern
from neuro_symphony.biophysics.synapse import SynapseType
from neuro_symphony.circuits.bgtc_network import (
    BGTCNetwork,
    BGTCParameters,
    KuramotoOrder,
)
from neuro_symphony.circuits.connectivity import (
    ConnectivityMatrix,
    ProjectionConfig,
    SynapticProjection,
)
from neuro_symphony.circuits.populations import (
    NeuronalPopulation,
    PopulationConfig,
    PopulationType,
)


class TestPopulations:
    """Test suite for neuronal populations."""

    def test_population_initialization(self) -> None:
        """Populations should initialize with correct dimensions and properties."""
        cfg = PopulationConfig(
            pop_type=PopulationType.STN,
            size=25,
            pattern=IzhikevichFiringPattern.THALAMOCORTICAL,
            baseline_drive=4.0
        )
        pop = NeuronalPopulation(config=cfg, seed=123)
        assert pop.size == 25
        assert pop.pop_type == PopulationType.STN
        assert len(pop.neurons.v) == 25

    def test_dopamine_modulation_striatum(self) -> None:
        """Dopamine depletion should differentially modulate dMSN and iMSN."""
        cfg_dmsn = PopulationConfig(
            pop_type=PopulationType.STRIATUM_DMSN,
            size=20,
            pattern=IzhikevichFiringPattern.REGULAR_SPIKING,
            baseline_drive=4.0,
            dopamine_sensitivity=5.0  # facilitated by DA
        )
        cfg_imsn = PopulationConfig(
            pop_type=PopulationType.STRIATUM_IMSN,
            size=20,
            pattern=IzhikevichFiringPattern.REGULAR_SPIKING,
            baseline_drive=4.0,
            dopamine_sensitivity=-5.0  # inhibited by DA
        )
        dmsn = NeuronalPopulation(cfg_dmsn, seed=10)
        imsn = NeuronalPopulation(cfg_imsn, seed=20)

        # Under low dopamine (DA = 0.1):
        dt = 0.05
        dmsn_spikes = 0
        imsn_spikes = 0
        for step_i in range(500):
            spk_d = dmsn.step(dt=dt, dopamine_level=0.1, current_time=step_i * dt)
            spk_i = imsn.step(dt=dt, dopamine_level=0.1, current_time=step_i * dt)
            dmsn_spikes += np.sum(spk_d)
            imsn_spikes += np.sum(spk_i)

        # Under Parkinsonian dopamine depletion, iMSN should be hyperactive relative to dMSN
        assert imsn_spikes > dmsn_spikes


class TestConnectivity:
    """Test suite for synaptic projections and axonal delay buffers."""

    def test_delay_buffer_mechanism(self) -> None:
        """Synaptic transmission should be delayed by specified axonal latency."""
        config = ProjectionConfig(
            source=PopulationType.M1_PYRAMIDAL,
            target=PopulationType.STN,
            receptor=SynapseType.AMPA,
            weight=2.0,
            probability=1.0,  # all-to-all
            delay_ms=2.0
        )
        dt = 0.05
        proj = SynapticProjection(config=config, source_size=10, target_size=10, dt=dt)

        spikes = np.ones(10, dtype=bool)  # synchronous pulse at t=0
        # Immediately after t=0, output conductance should still be zero because of 2.0 ms delay
        g_0 = proj.step(spikes)
        assert np.all(g_0 == 0.0)

        # After delay_steps (2.0 ms / 0.05 ms = 40 steps), conductance should rise
        g_peak = 0.0
        for _ in range(50):
            g = proj.step(np.zeros(10, dtype=bool))
            g_peak = max(g_peak, float(np.max(g)))

        assert g_peak > 0.1


class TestKuramotoOrder:
    """Test suite for synchronization metrics."""

    def test_perfect_in_phase_synchrony(self) -> None:
        """Identical phases must yield Kuramoto order parameter R = 1.0."""
        phases = np.full(50, 0.75)
        r = KuramotoOrder.instantaneous_order_parameter(phases)
        assert math.isclose(r, 1.0, abs_tol=1e-5)

    def test_uniform_random_phases_desynchrony(self) -> None:
        """Uniformly distributed phases should yield R near 0."""
        phases = np.linspace(-math.pi, math.pi, 1000, endpoint=False)
        r = KuramotoOrder.instantaneous_order_parameter(phases)
        assert r < 0.05

    def test_phase_locking_value(self) -> None:
        """Constant phase difference yields PLV = 1.0."""
        p1 = np.linspace(0, 10 * math.pi, 200)
        p2 = p1 + 0.5  # fixed phase shift
        plv = KuramotoOrder.phase_locking_value(p1, p2)
        assert math.isclose(plv, 1.0, abs_tol=1e-5)


class TestBGTCNetwork:
    """Integration test suite for the complete BGTC circuit."""

    def test_network_stepping(self) -> None:
        """BGTC network should step forward and record LFP and metrics."""
        params = BGTCParameters(
            dt=0.05,
            n_m1_pyr=15,
            n_m1_int=5,
            n_dmsn=15,
            n_imsn=15,
            n_stn=15,
            n_gpe=15,
            n_gpi=10,
            n_thalamus=10
        )
        net = BGTCNetwork(params=params, seed=42)
        res = net.step()

        assert "time" in res
        assert "stn_lfp" in res
        assert "kuramoto_r" in res
        assert 0.0 <= res["kuramoto_r"] <= 1.0

    def test_parkinsonian_beta_emergence(self) -> None:
        """Depleted dopamine should increase STN synchronization relative to healthy tone."""
        p_healthy = BGTCParameters(
            dt=0.05,
            dopamine_level=1.0,
            n_m1_pyr=20,
            n_stn=20,
            n_gpe=20,
            n_imsn=25,
            n_dmsn=25
        )
        p_pd = BGTCParameters(
            dt=0.05,
            dopamine_level=0.1,
            n_m1_pyr=20,
            n_stn=20,
            n_gpe=20,
            n_imsn=25,
            n_dmsn=25
        )

        net_h = BGTCNetwork(params=p_healthy, seed=42)
        net_pd = BGTCNetwork(params=p_pd, seed=42)

        res_h = net_h.run_simulation(duration_ms=150.0)
        res_pd = net_pd.run_simulation(duration_ms=150.0)

        # STN Kuramoto synchrony should be elevated in PD state
        mean_r_h = float(np.mean(res_h["kuramoto_r"]))
        mean_r_pd = float(np.mean(res_pd["kuramoto_r"]))
        assert mean_r_pd >= mean_r_h
