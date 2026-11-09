"""Command-line interface (CLI) for NeuroSymphony."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional
import numpy as np

from neuro_symphony.biophysics.hodgkin_huxley import (
    CompartmentalNeuron,
    HodgkinHuxleyNeuron,
    HodgkinHuxleyParameters,
)
from neuro_symphony.circuits.bgtc_network import BGTCNetwork, BGTCParameters
from neuro_symphony.circuits.populations import PopulationType
from neuro_symphony.signal.hilbert_envelope import HilbertEnvelopeTracker
from neuro_symphony.signal.spectral import SpectralAnalyzer
from neuro_symphony.stimulation.adaptive_closed_loop import (
    AdaptiveClosedLoopDBS,
    AdaptiveThresholds,
)
from neuro_symphony.stimulation.open_loop import OpenLoopDBS
from neuro_symphony.stimulation.phase_locked import PhaseLockedDBS
from neuro_symphony.visualizer import TerminalVisualizer


def run_simulate_hh(args: argparse.Namespace) -> int:
    """Run single-neuron Hodgkin-Huxley simulation with step current injection."""
    vis = TerminalVisualizer()
    print(vis.render_header("HODGKIN-HUXLEY BIOPHYSICAL NEURON SIMULATION"))
    print(f"Parameters: Duration={args.duration} ms, dt={args.dt} ms, I_inj={args.i_inj} uA/cm^2")

    neuron = HodgkinHuxleyNeuron()
    dt = args.dt
    time_points = np.arange(0, args.duration, dt)
    v_history = []
    spike_count = 0

    # Step current injection starting at 20 ms
    for t in time_points:
        i_ext = args.i_inj if (t >= 20.0 and t <= (args.duration - 20.0)) else 0.0
        spiked = neuron.step(dt=dt, i_ext=i_ext, current_time=t)
        v_history.append(neuron.v)
        if spiked:
            spike_count += 1

    v_arr = np.array(v_history)
    print(f"Total Action Potentials Fired: {spike_count}")
    print(f"Resting Potential: {v_history[0]:.2f} mV | Peak Potential: {np.max(v_arr):.2f} mV\n")

    trace = vis.render_waveform_trace(
        time_vec=time_points,
        lfp_vec=v_arr,
        width=args.width,
        height=10,
        title="MEMBRANE POTENTIAL TRACE V(t) [mV]"
    )
    print(trace)
    return 0


def run_bgtc_oscillations(args: argparse.Namespace) -> int:
    """Simulate BGTC network under healthy vs Parkinsonian dopamine states."""
    vis = TerminalVisualizer()
    print(vis.render_header("BGTC NETWORK: HEALTHY VS PARKINSONIAN BETA EMERGENCE"))

    duration = args.duration
    analyzer = SpectralAnalyzer(fs=1000.0 / args.dt)

    # 1. Healthy State (DA = 1.0)
    print("\n[1/2] Simulating HEALTHY state (Dopamine tone DA = 1.0)...")
    net_healthy = BGTCNetwork(BGTCParameters(dt=args.dt, dopamine_level=1.0), seed=42)
    res_healthy = net_healthy.run_simulation(duration_ms=duration)
    metrics_healthy = analyzer.analyze_beta_band(res_healthy["stn_lfp"])

    # 2. Parkinsonian State (DA = 0.1)
    print("[2/2] Simulating PARKINSONIAN state (Dopamine tone DA = 0.1)...")
    net_pd = BGTCNetwork(BGTCParameters(dt=args.dt, dopamine_level=0.1), seed=42)
    res_pd = net_pd.run_simulation(duration_ms=duration)
    metrics_pd = analyzer.analyze_beta_band(res_pd["stn_lfp"])

    print("\n" + "=" * 60)
    print(f"HEALTHY STATE (DA=1.0):")
    print(f"  Mean Kuramoto Order R: {np.mean(res_healthy['kuramoto_r']):.3f}")
    print(f"  STN Beta Peak Power:   {metrics_healthy.peak_power:.3f} at {metrics_healthy.peak_frequency_hz:.1f} Hz")
    print(f"PARKINSONIAN STATE (DA=0.1):")
    print(f"  Mean Kuramoto Order R: {np.mean(res_pd['kuramoto_r']):.3f}")
    print(f"  STN Beta Peak Power:   {metrics_pd.peak_power:.3f} at {metrics_pd.peak_frequency_hz:.1f} Hz")
    beta_increase = (metrics_pd.beta_band_power / max(1e-6, metrics_healthy.beta_band_power) - 1.0) * 100.0
    print(f"  Beta Power Surge:      +{beta_increase:.1f}%")
    print("=" * 60 + "\n")

    print(vis.render_kuramoto_gauge(np.mean(res_healthy["kuramoto_r"])))
    print(vis.render_kuramoto_gauge(np.mean(res_pd["kuramoto_r"])))

    # Render STN LFP trace under Parkinson's
    print("\n" + vis.render_waveform_trace(
        time_vec=res_pd["time"][-400:],
        lfp_vec=res_pd["stn_lfp"][-400:],
        width=args.width,
        height=9,
        title="PARKINSONIAN STN BETA OSCILLATION (DA=0.1)"
    ))

    # Render Spikes Raster
    print("\n" + vis.render_raster_plot(
        spike_raster=net_pd.spike_raster,
        time_min=duration - 200.0,
        time_max=duration,
        width=args.width
    ))
    return 0


def run_dbs_control(args: argparse.Namespace) -> int:
    """Compare open-loop vs adaptive closed-loop DBS in suppressing beta oscillations."""
    vis = TerminalVisualizer()
    print(vis.render_header("DBS CONTROL BENCHMARK: OPEN-LOOP (cDBS) VS ADAPTIVE (aDBS)"))

    duration = args.duration
    dt = args.dt
    fs = 1000.0 / dt
    analyzer = SpectralAnalyzer(fs=fs)

    # 1. Baseline Parkinsonian condition without stimulation
    print("\n[1/3] Running Parkinsonian baseline (No DBS)...")
    net_base = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=42)
    res_base = net_base.run_simulation(duration_ms=duration)
    metrics_base = analyzer.analyze_beta_band(res_base["stn_lfp"])

    # 2. Open-Loop Continuous DBS (130 Hz)
    print("[2/3] Running Open-Loop Constant DBS (130 Hz cDBS)...")
    net_cdbs = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=42)
    cdbs = OpenLoopDBS(frequency_hz=130.0, amplitude=4.0)
    res_cdbs = net_cdbs.run_simulation(duration_ms=duration, dbs_controller=cdbs)
    metrics_cdbs = analyzer.analyze_beta_band(
        res_cdbs["stn_lfp"],
        baseline_beta_power=metrics_base.beta_band_power
    )

    # 3. Adaptive Closed-Loop DBS (aDBS)
    print("[3/3] Running Adaptive Closed-Loop DBS (aDBS with beta thresholding)...")
    net_adbs = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=0.1), seed=42)
    adbs = AdaptiveClosedLoopDBS(
        thresholds=AdaptiveThresholds(upper_threshold=3.5, lower_threshold=1.5, min_burst_duration_ms=50.0),
        fs=fs
    )
    res_adbs = net_adbs.run_simulation(duration_ms=duration, dbs_controller=adbs)
    metrics_adbs = analyzer.analyze_beta_band(
        res_adbs["stn_lfp"],
        baseline_beta_power=metrics_base.beta_band_power
    )

    energy_saved = adbs.energy_savings_relative_to(cdbs.total_energy)

    print("\n" + "+" + "=" * 68 + "+")
    print(f"| {'METRIC':<25} | {'BASELINE':<11} | {'OPEN-LOOP':<12} | {'ADAPTIVE aDBS':<12} |")
    print("+" + "-" * 68 + "+")
    print(f"| {'Beta Power (uV^2/Hz)':<25} | {metrics_base.beta_band_power:<11.2f} | {metrics_cdbs.beta_band_power:<12.2f} | {metrics_adbs.beta_band_power:<12.2f} |")
    print(f"| {'Beta Suppression (%)':<25} | {'0.0%':<11} | {metrics_cdbs.beta_suppression_pct:<11.1f}% | {metrics_adbs.beta_suppression_pct:<11.1f}% |")
    print(f"| {'Mean Kuramoto R':<25} | {np.mean(res_base['kuramoto_r']):<11.3f} | {np.mean(res_cdbs['kuramoto_r']):<12.3f} | {np.mean(res_adbs['kuramoto_r']):<12.3f} |")
    print(f"| {'Total Energy (mA^2*ms)':<25} | {'0.0':<11} | {cdbs.total_energy:<12.1f} | {adbs.total_energy:<12.1f} |")
    print(f"| {'Duty Cycle (%)':<25} | {'0.0%':<11} | {cdbs.duty_cycle * 100.0:<11.1f}% | {adbs.duty_cycle * 100.0:<11.1f}% |")
    print(f"| {'Energy Savings vs cDBS':<25} | {'N/A':<11} | {'0.0%':<12} | {energy_saved:<11.1f}% |")
    print("+" + "=" * 68 + "+\n")

    print(vis.render_dbs_status(
        active=adbs.stimulating,
        mode="Adaptive Closed-Loop DBS",
        energy_pct_saved=energy_saved,
        pulses_delivered=adbs.total_pulses
    ))
    return 0


def run_demo(args: argparse.Namespace) -> int:
    """Run full interactive ASCII simulation dashboard."""
    vis = TerminalVisualizer(default_width=args.width)
    print(vis.render_header("NEUROSYMPHONY INTERACTIVE DEMO COCKPIT"))

    duration = args.duration
    dt = args.dt
    fs = 1000.0 / dt

    net = BGTCNetwork(BGTCParameters(dt=dt, dopamine_level=args.dopamine), seed=42)
    adbs = AdaptiveClosedLoopDBS(
        thresholds=AdaptiveThresholds(upper_threshold=3.5, lower_threshold=1.5),
        fs=fs
    )
    res = net.run_simulation(duration_ms=duration, dbs_controller=adbs)

    analyzer = SpectralAnalyzer(fs=fs)
    metrics = analyzer.analyze_beta_band(res["stn_lfp"])

    print(vis.render_full_dashboard(
        time_ms=duration,
        dopamine_level=args.dopamine,
        kuramoto_r=float(np.mean(res["kuramoto_r"][-100:])),
        stn_lfp=float(res["stn_lfp"][-1]),
        beta_power=metrics.peak_power,
        dbs_active=bool(res["dbs_active"][-1]),
        dbs_mode="Adaptive Closed-Loop (aDBS)",
        pulses_count=adbs.total_pulses,
        energy_saved_pct=adbs.energy_savings_relative_to(max(1.0, adbs.total_energy * 2.5))
    ))

    # Render waveform
    print("\n" + vis.render_waveform_trace(
        time_vec=res["time"][-300:],
        lfp_vec=res["stn_lfp"][-300:],
        width=args.width,
        height=8,
        title="STN LFP BETA WAVEFORM TRACE"
    ))

    # Render raster
    print("\n" + vis.render_raster_plot(
        spike_raster=net.spike_raster,
        time_min=duration - 150.0,
        time_max=duration,
        width=args.width
    ))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for neuro-symphony."""
    parser = argparse.ArgumentParser(
        prog="neuro-symphony",
        description="NeuroSymphony: Computational Neurophysiology & Closed-Loop Deep Brain Stimulation (DBS) Dynamics"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: simulate-hh
    p_hh = subparsers.add_parser("simulate-hh", help="Run single-neuron Hodgkin-Huxley simulation with current step injection")
    p_hh.add_argument("--duration", type=float, default=100.0, help="Duration in ms (default: 100.0)")
    p_hh.add_argument("--dt", type=float, default=0.02, help="Time step in ms (default: 0.02)")
    p_hh.add_argument("--i-inj", type=float, default=10.0, help="Injected current in uA/cm^2 (default: 10.0)")
    p_hh.add_argument("--width", type=int, default=65, help="Visualizer ASCII width (default: 65)")

    # Subcommand: bgtc-oscillations
    p_bgtc = subparsers.add_parser("bgtc-oscillations", help="Simulate BGTC network under healthy vs Parkinsonian dopamine states")
    p_bgtc.add_argument("--duration", type=float, default=300.0, help="Duration in ms (default: 300.0)")
    p_bgtc.add_argument("--dt", type=float, default=0.05, help="Time step in ms (default: 0.05)")
    p_bgtc.add_argument("--width", type=int, default=65, help="Visualizer ASCII width (default: 65)")

    # Subcommand: dbs-control
    p_dbs = subparsers.add_parser("dbs-control", help="Compare open-loop vs adaptive closed-loop DBS in suppressing beta oscillations")
    p_dbs.add_argument("--duration", type=float, default=350.0, help="Duration in ms (default: 350.0)")
    p_dbs.add_argument("--dt", type=float, default=0.05, help="Time step in ms (default: 0.05)")

    # Subcommand: demo
    p_demo = subparsers.add_parser("demo", help="Run full interactive ASCII simulation dashboard")
    p_demo.add_argument("--duration", type=float, default=300.0, help="Duration in ms (default: 300.0)")
    p_demo.add_argument("--dt", type=float, default=0.05, help="Time step in ms (default: 0.05)")
    p_demo.add_argument("--dopamine", type=float, default=0.1, help="Dopamine tone in [0.0, 1.0] (default: 0.1)")
    p_demo.add_argument("--width", type=int, default=68, help="Visualizer ASCII width (default: 68)")

    args = parser.parse_args(argv)

    if args.subcommand == "simulate-hh":
        return run_simulate_hh(args)
    elif args.subcommand == "bgtc-oscillations":
        return run_bgtc_oscillations(args)
    elif args.subcommand == "dbs-control":
        return run_dbs_control(args)
    elif args.subcommand == "demo":
        return run_demo(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
