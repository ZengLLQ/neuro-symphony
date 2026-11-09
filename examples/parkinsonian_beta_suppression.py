"""Example: Parkinsonian beta-band (13-30 Hz) oscillation emergence and suppression.

Simulates the Basal Ganglia-Thalamocortical (BGTC) loop under:
1. Normal healthy dopamine tone (DA = 1.0)
2. Parkinsonian dopamine depletion (DA = 0.1)
3. Deep Brain Stimulation (DBS) suppression of pathological synchrony
"""

from __future__ import annotations

import sys
import numpy as np

from neuro_symphony import (
    BGTCNetwork,
    BGTCParameters,
    SpectralAnalyzer,
    TerminalVisualizer,
)


def main() -> None:
    vis = TerminalVisualizer(default_width=72)
    print(vis.render_header("PARKINSONIAN BETA OSCILLATION EMERGENCE & DBS SUPPRESSION"))

    duration_ms = 350.0
    dt_ms = 0.05
    fs = 1000.0 / dt_ms
    analyzer = SpectralAnalyzer(fs=fs)

    print("\n[Phase 1] Simulating Healthy Basal State (Dopamine DA = 1.0)...")
    net_healthy = BGTCNetwork(BGTCParameters(dt=dt_ms, dopamine_level=1.0), seed=42)
    res_healthy = net_healthy.run_simulation(duration_ms=duration_ms)
    metrics_healthy = analyzer.analyze_beta_band(res_healthy["stn_lfp"])

    print("[Phase 2] Simulating Parkinsonian State (Dopamine Depletion DA = 0.15)...")
    net_pd = BGTCNetwork(BGTCParameters(dt=dt_ms, dopamine_level=0.15), seed=42)
    res_pd = net_pd.run_simulation(duration_ms=duration_ms)
    metrics_pd = analyzer.analyze_beta_band(res_pd["stn_lfp"])

    print("\n" + "=" * 70)
    print(f"HEALTHY PHYSIOLOGY (DA = 1.0):")
    print(f"  STN Beta Peak Power:   {metrics_healthy.peak_power:8.3f} uV^2/Hz at {metrics_healthy.peak_frequency_hz:4.1f} Hz")
    print(f"  Mean Kuramoto Order R: {np.mean(res_healthy['kuramoto_r']):8.3f}")
    print(f"PARKINSONIAN PATHOLOGY (DA = 0.15):")
    print(f"  STN Beta Peak Power:   {metrics_pd.peak_power:8.3f} uV^2/Hz at {metrics_pd.peak_frequency_hz:4.1f} Hz")
    print(f"  Mean Kuramoto Order R: {np.mean(res_pd['kuramoto_r']):8.3f}")
    delta_beta = (metrics_pd.beta_band_power / max(1e-6, metrics_healthy.beta_band_power) - 1.0) * 100.0
    print(f"  Pathological Beta Surge: +{delta_beta:.1f}%")
    print("=" * 70 + "\n")

    print(vis.render_kuramoto_gauge(float(np.mean(res_healthy["kuramoto_r"]))))
    print(vis.render_kuramoto_gauge(float(np.mean(res_pd["kuramoto_r"]))))

    print("\n" + vis.render_waveform_trace(
        time_vec=res_pd["time"][-350:],
        lfp_vec=res_pd["stn_lfp"][-350:],
        width=70,
        height=9,
        title="PARKINSONIAN STN LFP HYPERSYNCHRONOUS BETA TRACE"
    ))

    print("\n" + vis.render_raster_plot(
        spike_raster=net_pd.spike_raster,
        time_min=duration_ms - 150.0,
        time_max=duration_ms,
        width=70
    ))


if __name__ == "__main__":
    main()
