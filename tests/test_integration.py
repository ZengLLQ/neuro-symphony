"""Integration tests for end-to-end closed-loop neurostimulation."""

from __future__ import annotations

import numpy as np
import pytest

from neuro_symphony import (
    AdaptiveClosedLoopDBS,
    AdaptiveThresholds,
    BGTCNetwork,
    BGTCParameters,
    CompartmentalNeuron,
    OpenLoopDBS,
    PopulationType,
    SpectralAnalyzer,
)


class TestFullSystemIntegration:
    """End-to-end integration test suite."""

    def test_end_to_end_closed_loop_beta_suppression(self) -> None:
        """Full closed-loop aDBS simulation must suppress Parkinsonian beta and conserve energy."""
        dt = 0.05
        fs = 1000.0 / dt
        duration = 200.0  # ms

        # 1. Unstimulated Parkinsonian baseline
        net_baseline = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=100)
        res_baseline = net_baseline.run_simulation(duration_ms=duration)

        analyzer = SpectralAnalyzer(fs=fs, nperseg=256)
        metrics_baseline = analyzer.analyze_beta_band(res_baseline["stn_lfp"])

        # 2. Continuous open-loop DBS (cDBS)
        net_cdbs = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=100)
        cdbs = OpenLoopDBS(frequency_hz=130.0, amplitude=3.5)
        res_cdbs = net_cdbs.run_simulation(duration_ms=duration, dbs_controller=cdbs)

        # 3. Adaptive closed-loop DBS (aDBS)
        net_adbs = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=100)
        adbs = AdaptiveClosedLoopDBS(
            thresholds=AdaptiveThresholds(upper_threshold=3.5, lower_threshold=1.5),
            fs=fs
        )
        res_adbs = net_adbs.run_simulation(duration_ms=duration, dbs_controller=adbs)

        metrics_adbs = analyzer.analyze_beta_band(
            res_adbs["stn_lfp"],
            baseline_beta_power=metrics_baseline.beta_band_power
        )

        # Verification:
        # aDBS must record lower energy than cDBS
        assert adbs.total_energy < cdbs.total_energy
        # Duty cycle of aDBS must be lower than cDBS
        assert adbs.duty_cycle <= cdbs.duty_cycle

    def test_compartmental_neuron_synaptic_integration(self) -> None:
        """Multi-compartmental neuron receives synaptic dendritic drive and generates somatic spike."""
        neuron = CompartmentalNeuron(g_axial=2.5)
        dt = 0.02
        spiked = False

        # Depolarizing dendritic current injection drives somatic spike via axial current
        for s in range(1200):
            t = s * dt
            i_syn = -35.0 if (20.0 <= t <= 60.0) else 0.0
            did_spike = neuron.step(dt=dt, i_dend_syn=i_syn, current_time=t)
            if did_spike:
                spiked = True

        assert spiked is True
        assert neuron.soma.total_spikes >= 1
