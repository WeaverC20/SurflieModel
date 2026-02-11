"""
Surfzone Wave Propagation Runner

Forward ray tracing module for propagating waves from SWAN output
through the surfzone with energy deposition.

Components:
- swan_input_provider: Extract wave partitions from SWAN output
- wave_physics: Numba-accelerated wave physics calculations
- forward_ray_tracer: Forward ray tracing with energy deposition
- surfzone_runner: Main simulation orchestrator
- output_writer: Results storage
"""

from .swan_input_provider import SwanInputProvider, WavePartition, BoundaryConditions
from .wave_physics import (
    deep_water_properties,
    local_wave_properties,
    shoaling_coefficient,
    refraction_snell,
    wind_modification,
    BREAKER_TYPE_LABELS,
)
from .forward_ray_tracer import ForwardRayTracer, ForwardTracerConfig
from .surfzone_runner import ForwardSurfzoneRunner
from .surfzone_result import ForwardTracingResult
from .output_writer import save_forward_result, load_forward_result, BreakingField, load_breaking_field

__all__ = [
    # Swan input
    "SwanInputProvider",
    "WavePartition",
    "BoundaryConditions",
    # Physics
    "deep_water_properties",
    "local_wave_properties",
    "shoaling_coefficient",
    "refraction_snell",
    "wind_modification",
    "BREAKER_TYPE_LABELS",
    # Forward ray tracing
    "ForwardRayTracer",
    "ForwardTracerConfig",
    "ForwardSurfzoneRunner",
    "ForwardTracingResult",
    # Output
    "save_forward_result",
    "load_forward_result",
    "BreakingField",
    "load_breaking_field",
]
