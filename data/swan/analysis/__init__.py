"""
SWAN Analysis Module

Post-processing and visualization of SWAN model outputs.

Two plotting styles available:
- PAR (Parametric): Basic integrated plots (Hsig, Tps, Dir)
- Spectral: Period-colored arrows showing swell components
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.analysis.output_reader import SwanOutputReader, SwanOutput

# PAR (Parametric) plotter - basic integrated outputs
from data.swan.analysis.plot_results_par import SwanPlotter, plot_swan_results

# Spectral plotter - period-colored swell component arrows
from data.swan.analysis.plot_results_spectral import (
    SpectralPlotter,
    plot_spectral_results,
    MAX_SWELLS_TO_DISPLAY,
)

__all__ = [
    # Output reader
    "SwanOutputReader",
    "SwanOutput",
    # PAR plotter (basic)
    "SwanPlotter",
    "plot_swan_results",
    # Spectral plotter (swell components)
    "SpectralPlotter",
    "plot_spectral_results",
    "MAX_SWELLS_TO_DISPLAY",
]