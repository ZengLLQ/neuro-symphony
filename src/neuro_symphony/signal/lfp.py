"""Forward biophysical modeling of Local Field Potentials (LFP)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class LFPConfig:
    """Parameters for extracellular volume conduction and electrode placement."""
    extracellular_conductivity: float = 0.3  # S/m (Siemens per meter in brain tissue)
    electrode_distance_um: float = 150.0    # Distance from neuronal ensemble centroid (microns)
    lowpass_cutoff_hz: float = 300.0        # Upper cutoff for local field potentials


class ForwardLFP:
    """Forward model generating extracellular Local Field Potentials (LFP).

    Computes extracellular potential V_lfp from transmembrane current sources
    using the point-source volume conduction equation:
    V_lfp(t) = 1 / (4 * pi * sigma) * sum_k (I_k(t) / r_k)
    where sigma is extracellular medium conductivity (~0.3 S/m) and r_k is
    the distance from current source k to the microelectrode contact.
    """

    def __init__(self, config: Optional[LFPConfig] = None) -> None:
        self.config = config or LFPConfig()
        # Conversion constant: 1 / (4 * pi * sigma)
        # With r in micrometers (1e-6 m) and I in nanoamperes (1e-9 A):
        # V = (1e-9 / (4 * pi * 0.3 * 1e-6)) V = (1 / (1.2 * pi)) mV ~ 0.265 mV per (nA/um)
        self.scaling_factor = 1.0 / (4.0 * math.pi * self.config.extracellular_conductivity)

    def calculate_lfp(
        self,
        synaptic_currents: np.ndarray,
        distances_um: Optional[np.ndarray] = None
    ) -> float:
        """Compute instantaneous scalar LFP from ensemble synaptic currents.

        Args:
            synaptic_currents: Array of transmembrane currents from individual synapses/neurons.
            distances_um: Optional array of individual distances to electrode.
        """
        if len(synaptic_currents) == 0:
            return 0.0

        if distances_um is None:
            r = self.config.electrode_distance_um
            inv_r = 1.0 / max(1.0, r)
            lfp = self.scaling_factor * np.sum(synaptic_currents) * inv_r
        else:
            safe_r = np.maximum(1.0, distances_um)
            lfp = self.scaling_factor * np.sum(synaptic_currents / safe_r)

        return float(lfp)

    def synthesize_lfp_trace(
        self,
        membrane_voltages: np.ndarray,
        synaptic_inputs: np.ndarray
    ) -> np.ndarray:
        """Synthesize continuous LFP trace from membrane potentials and synaptic drive.

        In the neuropil, the low-frequency LFP predominantly reflects synchronized
        postsynaptic potentials (PSPs) and transmembrane currents rather than brief action potentials.
        """
        # Mean somatic voltage dipoles plus synaptic input currents
        raw_lfp = np.mean(membrane_voltages, axis=0) + 0.4 * np.mean(synaptic_inputs, axis=0)
        # Center signal around baseline zero
        centered = raw_lfp - np.mean(raw_lfp)
        return centered
