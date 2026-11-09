"""Unit tests for biophysical neuron and synaptic models."""

from __future__ import annotations

import math
import numpy as np
import pytest

from neuro_symphony.biophysics.hodgkin_huxley import (
    CompartmentalNeuron,
    HodgkinHuxleyNeuron,
    HodgkinHuxleyParameters,
)
from neuro_symphony.biophysics.izhikevich import (
    IzhikevichFiringPattern,
    IzhikevichNeuron,
    IzhikevichParameters,
    IzhikevichPopulation,
)
from neuro_symphony.biophysics.synapse import (
    AMPASynapse,
    GABAASynapse,
    GABABSynapse,
    NMDASynapse,
    SynapseParameters,
    SynapseType,
    SynapticCurrentPool,
)


class TestHodgkinHuxley:
    """Test suite for biophysical Hodgkin-Huxley neuron dynamics."""

    def test_resting_membrane_potential(self) -> None:
        """Neuron without current input should remain near resting potential."""
        neuron = HodgkinHuxleyNeuron()
        dt = 0.02
        for t in range(500):
            neuron.step(dt=dt, i_ext=0.0, current_time=t * dt)
        assert -75.0 < neuron.v < -55.0
        assert neuron.total_spikes == 0

    def test_action_potential_generation(self) -> None:
        """Sufficient depolarizing current step must trigger action potentials."""
        neuron = HodgkinHuxleyNeuron()
        dt = 0.02
        spike_detected = False
        v_max = -65.0

        for step_i in range(1500):
            t = step_i * dt
            i_ext = 12.0 if (20.0 <= t <= 45.0) else 0.0
            spiked = neuron.step(dt=dt, i_ext=i_ext, current_time=t)
            if spiked:
                spike_detected = True
            v_max = max(v_max, neuron.v)

        assert spike_detected is True
        assert neuron.total_spikes >= 1
        assert v_max > 10.0  # Action potential overshoot

    def test_gating_variable_bounds(self) -> None:
        """Gating variables (m, h, n, s, a, b) must strictly remain in [0, 1]."""
        neuron = HodgkinHuxleyNeuron()
        dt = 0.02
        for step_i in range(500):
            t = step_i * dt
            neuron.step(dt=dt, i_ext=15.0, current_time=t)
            assert 0.0 <= neuron.m <= 1.0
            assert 0.0 <= neuron.h <= 1.0
            assert 0.0 <= neuron.n <= 1.0
            assert 0.0 <= neuron.s <= 1.0
            assert 0.0 <= neuron.a <= 1.0
            assert 0.0 <= neuron.b <= 1.0

    def test_calcium_accumulation(self) -> None:
        """Calcium should accumulate during repetitive action potential firing."""
        neuron = HodgkinHuxleyNeuron()
        dt = 0.02
        initial_ca = neuron.ca
        for step_i in range(1000):
            neuron.step(dt=dt, i_ext=15.0, current_time=step_i * dt)
        assert neuron.ca > initial_ca

    def test_compartmental_neuron_axial_coupling(self) -> None:
        """Dendritic depolarization should propagate axially to soma."""
        comp = CompartmentalNeuron(g_axial=2.0)
        dt = 0.02
        initial_soma_v = comp.v_soma

        # Inject strong current into dendrite
        for step_i in range(500):
            comp.step(dt=dt, i_dend_syn=-20.0, current_time=step_i * dt)

        # Dendrite depolarization raises soma potential via axial current
        assert comp.v_soma != initial_soma_v


class TestIzhikevich:
    """Test suite for Izhikevich 2D non-linear integrate-and-fire model."""

    @pytest.mark.parametrize("pattern", [
        IzhikevichFiringPattern.REGULAR_SPIKING,
        IzhikevichFiringPattern.FAST_SPIKING,
        IzhikevichFiringPattern.INTRINSIC_BURSTING,
        IzhikevichFiringPattern.CHATTERING,
        IzhikevichFiringPattern.LOW_THRESHOLD_SPIKING,
        IzhikevichFiringPattern.THALAMOCORTICAL,
    ])
    def test_canonical_firing_patterns(self, pattern: IzhikevichFiringPattern) -> None:
        """All canonical firing patterns should step stably and fire under drive."""
        neuron = IzhikevichNeuron(pattern=pattern)
        dt = 0.05
        spiked_any = False
        for step_i in range(2000):
            spiked = neuron.step(dt=dt, current_input=10.0, current_time=step_i * dt)
            if spiked:
                spiked_any = True
        assert spiked_any is True
        assert neuron.total_spikes > 0

    def test_vectorized_population_stepping(self) -> None:
        """Vectorized IzhikevichPopulation should execute in parallel across N units."""
        pop = IzhikevichPopulation(size=50, pattern=IzhikevichFiringPattern.REGULAR_SPIKING)
        dt = 0.05
        assert pop.v.shape == (50,)
        assert pop.u.shape == (50,)

        # Run 200 ms with drive
        for step_i in range(4000):
            spikes = pop.step(dt=dt, current_input=10.0, current_time=step_i * dt)
            assert len(spikes) == 50

        assert np.sum(pop.total_spikes) > 0


class TestSynapses:
    """Test suite for conductance-based synaptic dynamics."""

    def test_ampa_dual_exponential_waveform(self) -> None:
        """AMPA conductance should rise and decay exponentially after presynaptic spike."""
        ampa = AMPASynapse()
        dt = 0.05
        ampa.on_presynaptic_spike(weight=1.0)

        conductances = []
        for _ in range(200):
            g = ampa.step(dt)
            conductances.append(g)

        peak_g = max(conductances)
        # Peak conductance should approximate g_max
        assert 0.8 < peak_g < 1.2
        # Conductance should decay toward zero
        assert conductances[-1] < 0.05

    def test_nmda_magnesium_block(self) -> None:
        """NMDA magnesium block should be strong at negative V and weak at depolarized V."""
        nmda = NMDASynapse()
        b_hyperpolarized = nmda.magnesium_block(-70.0)
        b_depolarized = nmda.magnesium_block(0.0)

        assert b_hyperpolarized < 0.25
        assert b_depolarized > 0.70
        assert b_hyperpolarized < b_depolarized

    def test_gaba_inhibitory_current(self) -> None:
        """GABA_A current should be hyperpolarizing (positive sign for V > E_rev)."""
        gaba = GABAASynapse()
        dt = 0.05
        gaba.on_presynaptic_spike(weight=1.0)
        for _ in range(20):
            gaba.step(dt)

        # At resting V = -65 mV, above E_rev = -70 mV, current I = g*(V - E_rev) > 0 (outward/hyperpolarizing)
        i_syn = gaba.current(v_post=-65.0)
        assert i_syn > 0.0

    def test_synaptic_pool_integration(self) -> None:
        """SynapticCurrentPool aggregates currents across multiple receptor types."""
        pool = SynapticCurrentPool()
        dt = 0.05

        pool.receive_spike(SynapseType.AMPA, weight=1.0)
        pool.receive_spike(SynapseType.GABA_A, weight=1.0)

        for _ in range(50):
            pool.step(dt)

        tot_curr = pool.total_current(v_post=-60.0)
        assert isinstance(tot_curr, float)
