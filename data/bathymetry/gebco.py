#!/usr/bin/env python3
"""
GEBCO Bathymetry Handler

Object-oriented interface for loading, viewing, and manipulating GEBCO bathymetry data.
This will be extended to support SWAN domain creation.

GEBCO 2024 file location: data/raw/bathymetry/gebco_2024/gebco_2024_california.nc
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class GEBCOBathymetry:
    """
    Handler for GEBCO bathymetry data.

    GEBCO uses elevation convention:
        - Negative values = below sea level (ocean depth)
        - Positive values = above sea level (land)

    This class converts to depth convention for oceanographic use:
        - Positive values = water depth
        - Negative/zero values = land

    Attributes:
        filepath: Path to the GEBCO NetCDF file
        elevation: Raw elevation data (negative = ocean)
        depth: Converted depth data (positive = ocean depth)
        lats: Latitude coordinates
        lons: Longitude coordinates
    """

    # Default GEBCO file location relative to project root
    DEFAULT_GEBCO_PATH = Path("data/raw/bathymetry/gebco_2024/gebco_2024_california.nc")

    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize GEBCO bathymetry handler.

        Args:
            filepath: Path to GEBCO NetCDF file. If None, uses default California file.
        """
        if filepath is None:
            # Find project root (go up from this file's location)
            project_root = Path(__file__).parent.parent.parent
            filepath = project_root / self.DEFAULT_GEBCO_PATH

        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"GEBCO file not found: {self.filepath}")

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load GEBCO data from NetCDF file."""
        print(f"Loading GEBCO data from: {self.filepath}")

        ds = xr.open_dataset(self.filepath)

        self.elevation = ds['elevation'].values  # Raw GEBCO data
        self.lats = ds['lat'].values
        self.lons = ds['lon'].values

        ds.close()

        # Convert to depth (positive = ocean depth)
        self.depth = -self.elevation.astype(np.float64)
        self.depth[self.depth <= 0] = np.nan  # Mask land

        print(f"  Shape: {self.elevation.shape}")
        print(f"  Lat range: {self.lats.min():.2f} to {self.lats.max():.2f}")
        print(f"  Lon range: {self.lons.min():.2f} to {self.lons.max():.2f}")
        print(f"  Depth range: {np.nanmin(self.depth):.1f}m to {np.nanmax(self.depth):.1f}m")

    @property
    def bounds(self) -> dict:
        """Get geographic bounds of the data."""
        return {
            'lat_min': float(self.lats.min()),
            'lat_max': float(self.lats.max()),
            'lon_min': float(self.lons.min()),
            'lon_max': float(self.lons.max()),
        }

    @property
    def resolution_deg(self) -> float:
        """Get approximate resolution in degrees."""
        return float(np.abs(self.lats[1] - self.lats[0]))

    @property
    def resolution_m(self) -> float:
        """Get approximate resolution in meters (at center latitude)."""
        center_lat = (self.lats.min() + self.lats.max()) / 2
        return self.resolution_deg * 111000 * np.cos(np.radians(center_lat))

    def view(
        self,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        depth_max: float = 4000,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        View the bathymetry as a depth profile with coastline.

        Args:
            lat_range: (lat_min, lat_max) to display. None = full extent.
            lon_range: (lon_min, lon_max) to display. None = full extent.
            depth_max: Maximum depth for colorbar (meters).
            figsize: Figure size (width, height) in inches.
            title: Plot title. None = auto-generate.
            save_path: Path to save figure. None = don't save.
            show: Whether to display the plot.
        """
        # Default to full extent
        if lat_range is None:
            lat_range = (self.lats.min(), self.lats.max())
        if lon_range is None:
            lon_range = (self.lons.min(), self.lons.max())

        # Subset data
        lat_mask = (self.lats >= lat_range[0]) & (self.lats <= lat_range[1])
        lon_mask = (self.lons >= lon_range[0]) & (self.lons <= lon_range[1])

        depth_subset = self.depth[np.ix_(lat_mask, lon_mask)]
        lats_subset = self.lats[lat_mask]
        lons_subset = self.lons[lon_mask]

        LON, LAT = np.meshgrid(lons_subset, lats_subset)

        # Create figure
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Plot depth
        im = ax.pcolormesh(
            LON, LAT, depth_subset,
            cmap='Blues',
            vmin=0, vmax=depth_max,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                      crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Depth (m)')

        # Title
        if title is None:
            title = f'GEBCO 2024 Bathymetry\n{lat_range[0]:.1f}°N to {lat_range[1]:.1f}°N, {lon_range[0]:.1f}°W to {lon_range[1]:.1f}°W'
        ax.set_title(title, fontsize=12)

        plt.tight_layout()

        # Save
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        # Show
        if show:
            plt.show()
        else:
            plt.close()

    def __repr__(self) -> str:
        return (
            f"GEBCOBathymetry(\n"
            f"  file='{self.filepath.name}',\n"
            f"  shape={self.elevation.shape},\n"
            f"  bounds={self.bounds},\n"
            f"  resolution={self.resolution_deg:.4f}° (~{self.resolution_m:.0f}m)\n"
            f")"
        )


# Convenience function for quick viewing
def view_gebco(
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    **kwargs
) -> GEBCOBathymetry:
    """
    Quick function to load and view GEBCO bathymetry.

    Args:
        lat_range: (lat_min, lat_max) to display
        lon_range: (lon_min, lon_max) to display
        **kwargs: Additional arguments passed to GEBCOBathymetry.view()

    Returns:
        GEBCOBathymetry instance for further use

    Example:
        >>> from data.bathymetry import view_gebco
        >>> gebco = view_gebco(lat_range=(32, 42), lon_range=(-126, -117))
    """
    gebco = GEBCOBathymetry()
    gebco.view(lat_range=lat_range, lon_range=lon_range, **kwargs)
    return gebco


if __name__ == "__main__":
    # Example usage: view full California coast
    gebco = GEBCOBathymetry()
    print(gebco)
    gebco.view(
        lat_range=(32, 42),
        lon_range=(-126, -117),
        title="California Coast - GEBCO 2024 Bathymetry"
    )
