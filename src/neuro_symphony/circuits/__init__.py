"""Basal Ganglia-Thalamocortical (BGTC) loop circuits and network models."""

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

__all__ = [
    "BGTCNetwork",
    "BGTCParameters",
    "ConnectivityMatrix",
    "KuramotoOrder",
    "NeuronalPopulation",
    "PopulationConfig",
    "PopulationType",
    "ProjectionConfig",
    "SynapticProjection",
]
