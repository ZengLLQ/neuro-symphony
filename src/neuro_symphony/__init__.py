"""NeuroSymphony: Computational Neurophysiology & Closed-Loop Deep Brain Stimulation Dynamics.

Multi-compartmental biophysical neuron modeling (Hodgkin-Huxley, Izhikevich),
basal ganglia-thalamocortical (BGTC) looping circuitry, parkinsonian beta-band
(13-30 Hz) pathogenetic synchronization, and adaptive closed-loop deep brain
stimulation (DBS) with phase-resetting control in Python.
"""

from __future__ import annotations

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
    Synapse,
    SynapseParameters,
    SynapseType,
    SynapticCurrentPool,
)
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
from neuro_symphony.signal.hilbert_envelope import (
    EnvelopeResult,
    HilbertEnvelopeTracker,
)
from neuro_symphony.signal.lfp import ForwardLFP, LFPConfig
from neuro_symphony.signal.spectral import SpectralAnalyzer, SpectralMetrics
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
from neuro_symphony.visualizer import TerminalVisualizer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HodgkinHuxleyNeuron",
    "CompartmentalNeuron",
    "HodgkinHuxleyParameters",
    "IzhikevichNeuron",
    "IzhikevichParameters",
    "IzhikevichPopulation",
    "IzhikevichFiringPattern",
    "Synapse",
    "AMPASynapse",
    "NMDASynapse",
    "GABAASynapse",
    "GABABSynapse",
    "SynapseParameters",
    "SynapseType",
    "SynapticCurrentPool",
    "BGTCNetwork",
    "BGTCParameters",
    "KuramotoOrder",
    "NeuronalPopulation",
    "PopulationConfig",
    "PopulationType",
    "ConnectivityMatrix",
    "ProjectionConfig",
    "SynapticProjection",
    "PulseGenerator",
    "BiphasicPulseConfig",
    "PulseShape",
    "OpenLoopDBS",
    "AdaptiveClosedLoopDBS",
    "AdaptiveThresholds",
    "PhaseLockedDBS",
    "PhaseLockConfig",
    "ForwardLFP",
    "LFPConfig",
    "HilbertEnvelopeTracker",
    "EnvelopeResult",
    "SpectralAnalyzer",
    "SpectralMetrics",
    "TerminalVisualizer",
]
