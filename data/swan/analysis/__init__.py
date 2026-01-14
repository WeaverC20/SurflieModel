"""
SWAN Analysis Module

Post-processing and visualization of SWAN model outputs.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.analysis.output_reader import SwanOutputReader, SwanOutput
from data.swan.analysis.plot_results import SwanPlotter, plot_swan_results

__all__ = [
    "SwanOutputReader",
    "SwanOutput",
    "SwanPlotter",
    "plot_swan_results",
]