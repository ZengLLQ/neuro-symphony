"""Example: Closed-Loop Adaptive DBS (aDBS) vs Continuous Open-Loop DBS (cDBS).

Demonstrates real-time closed-loop sensing of pathological beta envelope,
dual-threshold burst stimulation, and quantitative comparison of:
- Pathological beta suppression percentage
- Delivered electrical energy
- Battery power savings
- Modulation duty cycle
"""

from __future__ import annotations

import sys
import numpy as np

from neuro_symphony import (
    AdaptiveClosedLoopDBS,
    AdaptiveThresholds,
    BGTCNetwork,
    BGTCParameters,
    OpenLoopDBS,
    SpectralAnalyzer,
    TerminalVisualizer,
)


def main() -> None:
    vis = TerminalVisualizer(default_width=72)
    print(vis.render_header("ADAPTIVE CLOSED-LOOP (aDBS) VS OPEN-LOOP (cDBS) BENCHMARK"))

    duration_ms = 400.0
    dt_ms = 0.05
    fs = 1000.0 / dt_ms
    analyzer = SpectralAnalyzer(fs=fs)

    print("\n[Stage 1/3] Simulating unmitigated Parkinsonian baseline (DA = 0.1)...")
    net_base = BGTCNetwork(BGTCParameters(dt=dt_ms, dopamine_level=0.1), seed=42)
    res_base = net_base.run_simulation(duration_ms=duration_ms)
    metrics_base = analyzer.analyze_beta_band(res_base["stn_lfp"])

    print("[Stage 2/3] Simulating Open-Loop Continuous DBS (130 Hz cDBS)...")
    net_cdbs = BGTCNetwork(BGTCParameters(dt=dt_ms, dopamine_level=0.1), seed=42)
    cdbs = OpenLoopDBS(frequency_hz=130.0, amplitude=4.0)
    res_cdbs = net_cdbs.run_simulation(duration_ms=duration_ms, dbs_controller=cdbs)
    metrics_cdbs = analyzer.analyze_beta_band(
        res_cdbs["stn_lfp"],
        baseline_beta_power=metrics_base.beta_band_power
    )

    print("[Stage 3/3] Simulating Adaptive Closed-Loop DBS (aDBS dual-threshold)...")
    net_adbs = BGTCNetwork(BGTCParameters(dt=dt_ms, dopamine_level=0.1), seed=42)
    adbs = AdaptiveClosedLoopDBS(
        thresholds=AdaptiveThresholds(upper_threshold=3.5, lower_threshold=1.5, min_burst_duration_ms=50.0),
        fs=fs
    )
    res_adbs = net_adbs.run_simulation(duration_ms=duration_ms, dbs_controller=adbs)
    metrics_adbs = analyzer.analyze_beta_band(
        res_adbs["stn_lfp"],
        baseline_beta_power=metrics_base.beta_band_power
    )

    energy_savings = adbs.energy_savings_relative_to(cdbs.total_energy)

    print("\n" + "+" + "=" * 70 + "+")
    print(f"| {'BENCHMARK PARAMETER':<26} | {'UNSTIMULATED':<12} | {'cDBS (130 Hz)':<12} | {'aDBS (Closed)':<12} |")
    print("+" + "-" * 70 + "+")
    print(f"| {'Beta Power (uV^2/Hz)':<26} | {metrics_base.beta_band_power:<12.2f} | {metrics_cdbs.beta_band_power:<12.2f} | {metrics_adbs.beta_band_power:<12.2f} |")
    print(f"| {'Beta Suppression (%)':<26} | {'0.0%':<12} | {metrics_cdbs.beta_suppression_pct:<11.1f}% | {metrics_adbs.beta_suppression_pct:<11.1f}% |")
    print(f"| {'Mean Kuramoto Order R':<26} | {np.mean(res_base['kuramoto_r']):<12.3f} | {np.mean(res_cdbs['kuramoto_r']):<12.3f} | {np.mean(res_adbs['kuramoto_r']):<12.3f} |")
    print(f"| {'Delivered Energy (mA^2*ms)':<26} | {'0.0':<12} | {cdbs.total_energy:<12.1f} | {adbs.total_energy:<12.1f} |")
    print(f"| {'Duty Cycle (%)':<26} | {'0.0%':<12} | {cdbs.duty_cycle * 100.0:<11.1f}% | {adbs.duty_cycle * 100.0:<11.1f}% |")
    print(f"| {'Energy Conservation vs cDBS':<26} | {'N/A':<12} | {'0.0%':<12} | {energy_savings:<11.1f}% |")
    print("+" + "=" * 70 + "+\n")

    print(vis.render_dbs_status(
        active=adbs.stimulating,
        mode="Adaptive Closed-Loop DBS (aDBS)",
        energy_pct_saved=energy_savings,
        pulses_delivered=adbs.total_pulses
    ))

    print("\n" + vis.render_waveform_trace(
        time_vec=res_adbs["time"][-350:],
        lfp_vec=res_adbs["stn_lfp"][-350:],
        width=70,
        height=8,
        title="aDBS STN LFP SUPPRESSED DYNAMICS"
    ))


if __name__ == "__main__":
    main()
