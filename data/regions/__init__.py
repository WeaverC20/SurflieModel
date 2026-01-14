"""
Regions module for SWAN wave modeling.

Provides Region and Mesh classes for defining geographic subregions
and computational grids for SWAN simulations.

Example usage:
    from data.regions import Region, Mesh, NORCAL, CA_SUBREGIONS

    # Get a specific region
    from data.regions import get_region
    norcal = get_region("norcal")

    # Create a mesh for a region
    mesh = Mesh(name="norcal_coarse", region=norcal, resolution_km=5.0)
"""

from data.regions.region import (
    Region,
    CALIFORNIA,
    NORCAL,
    CENTRAL_CAL,
    SOCAL,
    CA_SUBREGIONS,
    REGIONS,
    get_region,
)

from data.regions.mesh import Mesh

__all__ = [
    # Classes
    "Region",
    "Mesh",
    # California regions
    "CALIFORNIA",
    "NORCAL",
    "CENTRAL_CAL",
    "SOCAL",
    "CA_SUBREGIONS",
    # Utilities
    "REGIONS",
    "get_region",
]
