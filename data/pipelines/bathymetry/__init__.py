"""
Bathymetry data pipeline for SWAN model integration.

This module provides tools for:
- Fetching GEBCO 2024 data (coarse bathymetry ~450m resolution)
- Fetching NCEI Coastal Relief Model data (fine bathymetry ~90m resolution)
- Generating outer SWAN grids from GEBCO
- Generating SWAN domains with WW3 boundary conditions
- Visualizing bathymetry and grid meshes
"""

from .config import REGIONS, SWAN_CONFIG
from .gebco_fetcher import GEBCOFetcher
from .ncei_fetcher import NCEIFetcher
from .outer_swan import OuterSwanGenerator
from .swan_domain import SwanDomainGenerator
from .visualizer import BathymetryVisualizer

__all__ = [
    "REGIONS",
    "SWAN_CONFIG",
    "GEBCOFetcher",
    "NCEIFetcher",
    "OuterSwanGenerator",
    "SwanDomainGenerator",
    "BathymetryVisualizer",
]
