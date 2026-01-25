"""Bathymetry data processing module."""

# Lazy imports to avoid cartopy dependency for modules that don't need visualization
__all__ = ["GEBCOBathymetry", "USACELidar", "NCECRM"]

def __getattr__(name):
    if name == "GEBCOBathymetry":
        from .gebco import GEBCOBathymetry
        return GEBCOBathymetry
    elif name == "USACELidar":
        from .usace_lidar import USACELidar
        return USACELidar
    elif name == "NCECRM":
        from .ncei_crm import NCECRM
        return NCECRM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
