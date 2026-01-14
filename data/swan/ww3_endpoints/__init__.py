"""
WW3 Boundary Endpoint Extraction Module

Provides tools for finding WaveWatch III grid points along SWAN domain
boundaries for use as spectral boundary conditions.

Example:
    from data.swan.ww3_endpoints import (
        BoundaryLine,
        find_boundary_points,
        extract_region_boundaries,
    )

    # Extract west boundary points for a region
    boundaries = extract_region_boundaries(
        lon_min=-121.0,
        lon_max=-117.0,
        lat_min=32.0,
        lat_max=34.5,
        sides=['west'],
        region_name='socal',
    )
"""

from data.swan.ww3_endpoints.extract_ww3_endpoints import (
    WW3Grid,
    WW3_GRID,
    BoundaryLine,
    BoundaryType,
    BoundaryPointSet,
    find_boundary_points,
    extract_region_boundaries,
)

__all__ = [
    "WW3Grid",
    "WW3_GRID",
    "BoundaryLine",
    "BoundaryType",
    "BoundaryPointSet",
    "find_boundary_points",
    "extract_region_boundaries",
]