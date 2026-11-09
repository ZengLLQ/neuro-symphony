"""Unit tests for neurostimulation waveforms and DBS control architectures."""

from __future__ import annotations

import math
import numpy as np
import pytest

from neuro_symphony.stimulation.adaptive_closed_loop import (
    AdaptiveClosedLoopDBS,
    AdaptiveThresholds,
)
from neuro_symphony.stimulation.open_loop import OpenLoopDBS
from neuro_symphony.stimulation.phase_locked import PhaseLockConfig, PhaseLockedDBS
from neuro_symphony.stimulation.pulse_generator import (
    BiphasicPulseConfig,
    PulseGenerator,
    PulseShape,
)


class TestPulseGenerator:
    """Test suite for electrical pulse waveforms and charge balance."""

    def test_charge_balance_rectangular(self) -> None:
        """Net electrical charge Q_net = integral(I dt) must be zero for biphasic pulse."""
        cfg = BiphasicPulseConfig(
            amplitude=3.0,
            pulse_width_ms=0.09,
            interphase_gap_ms=0.02,
            recovery_ratio=1.0,
            shape=PulseShape.BIPHASIC_RECTANGULAR
        )
        gen = PulseGenerator(cfg)
        dt = 0.001  # high resolution 1 us
        t_arr, i_arr = gen.generate_single_pulse(dt=dt)

        # Numerical integration of charge
        q_net = np.sum(i_arr) * dt
        assert math.isclose(q_net, 0.0, abs_tol=1e-3)

    def test_pulse_triggering(self) -> None:
        """Step function should output current following a trigger event."""
        gen = PulseGenerator(BiphasicPulseConfig(amplitude=2.5, pulse_width_ms=0.08))
        dt = 0.02
        # Before trigger
        assert gen.step(dt=dt, trigger=False) == 0.0

        # Trigger event
        i_0 = gen.step(dt=dt, trigger=True)
        assert i_0 < 0  # Cathodal phase is negative


class TestOpenLoopDBS:
    """Test suite for classical continuous 130 Hz cDBS."""

    def test_continuous_firing_frequency(self) -> None:
        """cDBS at 130 Hz should fire approximately 13 pulses per 100 ms."""
        cdbs = OpenLoopDBS(frequency_hz=130.0, amplitude=3.0)
        dt = 0.05
        total_time = 100.0  # ms
        steps = int(total_time / dt)

        for s in range(steps):
            cdbs.compute_stimulus(lfp_sample=0.0, dt=dt, current_time=s * dt)

        # 130 Hz * 0.1 s = 13 pulses
        assert 12 <= cdbs.total_pulses <= 14
        assert cdbs.total_energy > 0.0
        assert cdbs.duty_cycle > 0.0


class TestAdaptiveClosedLoopDBS:
    """Test suite for adaptive closed-loop aDBS."""

    def test_hysteresis_state_transitions(self) -> None:
        """aDBS should turn ON when envelope > upper_thresh and turn OFF when < lower_thresh."""
        thresholds = AdaptiveThresholds(
            upper_threshold=5.0,
            lower_threshold=2.0,
            min_burst_duration_ms=20.0
        )
        adbs = AdaptiveClosedLoopDBS(thresholds=thresholds, fs=20000.0)
        dt = 0.05

        # 1. Low input: should remain OFF
        for s in range(100):
            adbs.compute_stimulus(lfp_sample=0.5, dt=dt, current_time=s * dt)
        assert adbs.stimulating is False

        # 2. Surge in beta oscillation: should trigger ON
        for s in range(100, 300):
            # 20 Hz oscillation with large amplitude
            t = s * dt * 1e-3
            sample = 25.0 * math.sin(2 * math.pi * 20.0 * t)
            adbs.compute_stimulus(lfp_sample=sample, dt=dt, current_time=s * dt)

        assert adbs.stimulating is True
        assert adbs.burst_count >= 1

    def test_energy_savings_quantification(self) -> None:
        """Energy savings calculation should compute correct reduction percentage."""
        adbs = AdaptiveClosedLoopDBS()
        adbs.pulse_gen.cumulative_energy = 40.0
        c_dbs_energy = 100.0

        savings = adbs.energy_savings_relative_to(c_dbs_energy)
        assert math.isclose(savings, 60.0, abs_tol=1e-3)


class TestPhaseLockedDBS:
    """Test suite for phase-locked desynchronization pDBS."""

    def test_phase_locked_triggering(self) -> None:
        """pDBS should deliver pulses when tracking reaches target phase."""
        config = PhaseLockConfig(
            target_phase_rad=0.0,
            phase_tolerance_rad=0.5,
            min_refractory_ms=30.0,
            min_amplitude_threshold=1.0
        )
        pdbs = PhaseLockedDBS(config=config, fs=20000.0)
        dt = 0.05

        # Simulate 20 Hz sinusoidal LFP
        for s in range(500):
            t = s * dt * 1e-3
            sample = 15.0 * math.sin(2 * math.pi * 20.0 * t)
            pdbs.compute_stimulus(lfp_sample=sample, dt=dt, current_time=s * dt)

        assert pdbs.total_pulses > 0
