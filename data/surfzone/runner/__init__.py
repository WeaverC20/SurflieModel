"""
Surfzone Wave Propagation Runner

Ray tracing module for propagating waves from SWAN output (2.5km offshore)
through the surfzone to the shore.

Components:
- swan_input_provider: Extract wave partitions from SWAN output
- wave_physics: Numba-accelerated wave physics calculations
- ray_tracer: Ray tracing engine
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
from .ray_tracer import RayTracer, RayResult
from .output_writer import OutputWriter, BreakingField, load_breaking_field

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
    # Ray tracing
    "RayTracer",
    "RayResult",
    # Output
    "OutputWriter",
    "BreakingField",
    "load_breaking_field",
]