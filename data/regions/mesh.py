#!/usr/bin/env python3
"""
Mesh Class for SWAN Modeling

Defines computational meshes that are sampled from GEBCO bathymetry data.
Each mesh belongs to a Region and contains the bathymetry data needed for SWAN.

SWAN Grid Types:
    - Regular (REG): Rectangular grid - supports spectral wave input
    - Curvilinear: For complex coastlines (does NOT support spectral input)
    - Unstructured: Triangular meshes (does NOT support spectral input)

We use REGULAR grids exclusively to support spectral wave boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING, Union
from pathlib import Path
import json
import numpy as np

if TYPE_CHECKING:
    from data.bathymetry.gebco import GEBCOBathymetry
    import matplotlib.pyplot as plt

# Import Region for type hints and loading
from data.regions.region import Region, get_region, REGIONS


@dataclass
class Mesh:
    """
    A computational mesh for SWAN wave modeling.

    This mesh uses a REGULAR (rectangular) grid to support spectral wave
    boundary conditions from WW3.

    Attributes:
        name: Identifier for this mesh (e.g., "socal_coarse")
        region: Parent Region this mesh belongs to
        resolution_km: Grid resolution in kilometers (default 5km)

        # Grid definition (populated by from_gebco or set manually)
        origin: (lon, lat) of lower-left corner
        nx: Number of grid cells in x (longitude) direction
        ny: Number of grid cells in y (latitude) direction
        dx: Grid spacing in x direction (degrees)
        dy: Grid spacing in y direction (degrees)

        # Spectral discretization (for CGRID command)
        n_dir: Number of directional bins (default 36 = 10° resolution)
        freq_min: Lowest frequency in Hz (default 0.04 = 25s period)
        freq_max: Highest frequency in Hz (default 1.0 = 1s period)
        n_freq: Number of frequency bins (default 31)
        dir_type: "CIRCLE" for full circle or (dir1, dir2) tuple for SECTOR

        # Data
        depth_data: 2D array of depths (positive = ocean depth in meters)
        exception_value: Value marking land/invalid points (default -99)

    SWAN Convention:
        - Depths are positive downward
        - Land points use exception_value
        - Grid is specified from lower-left corner
    """

    name: str
    region: Optional[Region] = None
    resolution_km: float = 5.0  # Default 5km resolution

    # Grid definition
    origin: Optional[Tuple[float, float]] = None  # (lon, lat)
    nx: Optional[int] = None
    ny: Optional[int] = None
    dx: Optional[float] = None  # degrees
    dy: Optional[float] = None  # degrees

    # Spectral discretization (for CGRID)
    n_dir: int = 36  # 36 bins = 10° directional resolution
    freq_min: float = 0.04  # 0.04 Hz = 25s period (long swells)
    freq_max: float = 1.0  # 1.0 Hz = 1s period (short wind waves)
    n_freq: int = 31  # Number of frequency bins
    dir_type: str = "CIRCLE"  # Full directional circle

    # Data
    depth_data: Optional[np.ndarray] = None
    exception_value: float = -99.0

    # Constants
    GRID_TYPE: str = "regular"  # Only regular grids for spectral input

    def __post_init__(self):
        """Calculate dx/dy from resolution if region is set."""
        if self.region is not None and self.dx is None:
            self._calculate_grid_spacing()

    def _calculate_grid_spacing(self) -> None:
        """
        Calculate grid spacing in degrees from resolution_km.

        Uses center latitude of region for conversion.
        1 degree latitude ≈ 111 km
        1 degree longitude ≈ 111 km * cos(latitude)
        """
        if self.region is None:
            return

        center_lat = (self.region.lat_range[0] + self.region.lat_range[1]) / 2

        # Convert km to degrees
        self.dy = self.resolution_km / 111.0
        self.dx = self.resolution_km / (111.0 * np.cos(np.radians(center_lat)))

    # =========================================================================
    # Sampling from GEBCO
    # =========================================================================

    def from_gebco(self, gebco: GEBCOBathymetry) -> Mesh:
        """
        Sample depths from GEBCO bathymetry at mesh grid points.

        Args:
            gebco: GEBCOBathymetry instance to sample from

        Returns:
            self (for chaining)
        """
        if self.region is None:
            raise ValueError("Mesh must have a region assigned before sampling from GEBCO")

        if self.dx is None or self.dy is None:
            self._calculate_grid_spacing()

        # Get region bounds
        lon_min, lon_max = self.region.lon_range
        lat_min, lat_max = self.region.lat_range

        # Set origin (lower-left corner)
        self.origin = (lon_min, lat_min)

        # Calculate grid dimensions
        self.nx = int(np.ceil((lon_max - lon_min) / self.dx))
        self.ny = int(np.ceil((lat_max - lat_min) / self.dy))

        # Create mesh grid points
        mesh_lons = np.linspace(lon_min, lon_min + self.nx * self.dx, self.nx + 1)
        mesh_lats = np.linspace(lat_min, lat_min + self.ny * self.dy, self.ny + 1)

        # Interpolate GEBCO depth to mesh points
        # For now, use nearest-neighbor (can upgrade to bilinear later)
        self.depth_data = np.zeros((self.ny + 1, self.nx + 1))

        for i, lat in enumerate(mesh_lats):
            for j, lon in enumerate(mesh_lons):
                # Find nearest GEBCO point
                lat_idx = np.argmin(np.abs(gebco.lats - lat))
                lon_idx = np.argmin(np.abs(gebco.lons - lon))

                depth = gebco.depth[lat_idx, lon_idx]

                # Use exception value for land (NaN in GEBCO depth)
                if np.isnan(depth):
                    self.depth_data[i, j] = self.exception_value
                else:
                    self.depth_data[i, j] = depth

        print(f"Created mesh '{self.name}':")
        print(f"  Grid: {self.nx} x {self.ny} cells")
        print(f"  Resolution: {self.resolution_km} km ({self.dx:.4f}° x {self.dy:.4f}°)")
        print(f"  Origin: ({self.origin[0]:.2f}°, {self.origin[1]:.2f}°)")
        print(f"  Spectral: {self.n_dir} dirs, {self.n_freq} freqs ({self.freq_min}-{self.freq_max} Hz)")

        return self

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, directory: Union[str, Path], idla: int = 4) -> Path:
        """
        Save mesh with all metadata to a directory.

        Creates two files:
            - {name}.bot: SWAN bathymetry file (depth values)
            - {name}.json: Metadata (grid parameters, region, etc.)

        Args:
            directory: Directory to save files to
            idla: SWAN layout parameter for .bot file (default 4)

        Returns:
            Path to the directory containing saved files
        """
        if self.depth_data is None:
            raise ValueError("No depth data to save. Call from_gebco() first.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save .bot file (SWAN format)
        bot_path = directory / f"{self.name}.bot"
        self._write_bot_file(bot_path, idla)

        # Save .json metadata
        json_path = directory / f"{self.name}.json"
        self._write_json_metadata(json_path, idla)

        print(f"Saved mesh to: {directory}")
        return directory

    def _write_bot_file(self, filepath: Path, idla: int = 4) -> None:
        """Write SWAN bathymetry .bot file."""
        with open(filepath, 'w') as f:
            for i in range(self.depth_data.shape[0]):
                row = self.depth_data[i, :]
                line = ' '.join(f'{val:.2f}' for val in row)
                f.write(line + '\n')
        print(f"  Wrote: {filepath.name}")

    def _write_json_metadata(self, filepath: Path, idla: int = 4) -> None:
        """Write mesh metadata to JSON file."""
        metadata = {
            "name": self.name,
            "region_name": self.region.name if self.region else None,
            "resolution_km": self.resolution_km,
            "grid_type": self.GRID_TYPE,
            "origin": list(self.origin) if self.origin else None,
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "spectral": {
                "n_dir": self.n_dir,
                "freq_min": self.freq_min,
                "freq_max": self.freq_max,
                "n_freq": self.n_freq,
                "dir_type": self.dir_type,
            },
            "exception_value": self.exception_value,
            "idla": idla,
            "depth_shape": list(self.depth_data.shape) if self.depth_data is not None else None,
            "depth_stats": {
                "min": float(np.nanmin(self.depth_data[self.depth_data != self.exception_value])),
                "max": float(np.nanmax(self.depth_data[self.depth_data != self.exception_value])),
                "n_ocean_cells": int(np.sum(self.depth_data != self.exception_value)),
                "n_land_cells": int(np.sum(self.depth_data == self.exception_value)),
            } if self.depth_data is not None else None,
            "swan_commands": {
                "cgrid": self.generate_cgrid_command(),
                "inpgrid": self.generate_inpgrid_command(),
                "readinp": self.generate_readinp_command(f"{self.name}.bot", idla),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Wrote: {filepath.name}")

    @classmethod
    def load(cls, directory: Union[str, Path], name: Optional[str] = None) -> Mesh:
        """
        Load a mesh from saved files.

        Args:
            directory: Directory containing .bot and .json files
            name: Mesh name (optional if only one mesh in directory)

        Returns:
            Loaded Mesh object
        """
        directory = Path(directory)

        # Find the mesh name if not provided
        if name is None:
            json_files = list(directory.glob("*.json"))
            if len(json_files) == 0:
                raise FileNotFoundError(f"No mesh metadata found in {directory}")
            elif len(json_files) > 1:
                names = [f.stem for f in json_files]
                raise ValueError(f"Multiple meshes found: {names}. Specify name parameter.")
            name = json_files[0].stem

        json_path = directory / f"{name}.json"
        bot_path = directory / f"{name}.bot"

        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        if not bot_path.exists():
            raise FileNotFoundError(f"Bathymetry file not found: {bot_path}")

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Load depth data
        depth_data = np.loadtxt(bot_path)

        # Get region if it exists
        region = None
        region_name = metadata.get("region_name")
        if region_name and region_name in REGIONS:
            region = get_region(region_name)

        # Get spectral parameters (with defaults for older meshes)
        spectral = metadata.get("spectral", {})

        # Create mesh
        mesh = cls(
            name=metadata["name"],
            region=region,
            resolution_km=metadata["resolution_km"],
            origin=tuple(metadata["origin"]) if metadata["origin"] else None,
            nx=metadata["nx"],
            ny=metadata["ny"],
            dx=metadata["dx"],
            dy=metadata["dy"],
            n_dir=spectral.get("n_dir", 36),
            freq_min=spectral.get("freq_min", 0.04),
            freq_max=spectral.get("freq_max", 1.0),
            n_freq=spectral.get("n_freq", 31),
            dir_type=spectral.get("dir_type", "CIRCLE"),
            depth_data=depth_data,
            exception_value=metadata["exception_value"],
        )

        print(f"Loaded mesh '{mesh.name}' from {directory}")
        return mesh

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 10),
        depth_max: float = 4000,
        cmap: str = 'Blues',
        show_colorbar: bool = True,
        show_grid: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Axes:
        """
        Plot the mesh bathymetry.

        Args:
            ax: Matplotlib axes to plot on (creates new figure if None)
            figsize: Figure size if creating new figure
            depth_max: Maximum depth for colorbar
            cmap: Colormap name
            show_colorbar: Whether to show colorbar
            show_grid: Whether to show lat/lon gridlines
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            show: Whether to display the plot

        Returns:
            The matplotlib axes
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if self.depth_data is None:
            raise ValueError("No depth data to plot. Call from_gebco() first or load().")

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
        else:
            fig = ax.get_figure()

        # Get bounds
        lon_min = self.origin[0]
        lat_min = self.origin[1]
        lon_max = lon_min + self.nx * self.dx
        lat_max = lat_min + self.ny * self.dy

        # Create mesh coordinates
        mesh_lons = np.linspace(lon_min, lon_max, self.nx + 1)
        mesh_lats = np.linspace(lat_min, lat_max, self.ny + 1)
        LON, LAT = np.meshgrid(mesh_lons, mesh_lats)

        # Mask exception values for plotting
        depth_plot = self.depth_data.copy()
        depth_plot[depth_plot == self.exception_value] = np.nan

        # Plot bathymetry
        im = ax.pcolormesh(
            LON, LAT, depth_plot,
            cmap=cmap,
            vmin=0, vmax=depth_max,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Gridlines
        if show_grid:
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
            cbar.set_label('Depth (m)')

        # Title
        if title is None:
            region_str = self.region.display_name if self.region else "Unknown Region"
            title = f"{region_str} - {self.name}\n({self.resolution_km}km, {self.nx}x{self.ny} cells)"
        ax.set_title(title, fontsize=12)

        plt.tight_layout()

        # Save
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        # Show
        if show:
            plt.show()
        elif ax is None:
            plt.close()

        return ax

    def summary(self) -> str:
        """Return a summary string of mesh properties."""
        lines = [
            f"Mesh: {self.name}",
            f"  Region: {self.region.display_name if self.region else 'None'}",
            f"  Resolution: {self.resolution_km} km",
            f"  Grid: {self.nx} x {self.ny} cells",
            f"  Origin: ({self.origin[0]:.4f}°, {self.origin[1]:.4f}°)" if self.origin else "  Origin: None",
            f"  Spacing: dx={self.dx:.6f}°, dy={self.dy:.6f}°" if self.dx else "  Spacing: None",
            f"  Spectral: {self.n_dir} dirs, {self.n_freq} freqs ({self.freq_min}-{self.freq_max} Hz)",
        ]

        if self.depth_data is not None:
            ocean_mask = self.depth_data != self.exception_value
            n_ocean = np.sum(ocean_mask)
            n_land = self.depth_data.size - n_ocean
            depth_min = np.nanmin(self.depth_data[ocean_mask])
            depth_max = np.nanmax(self.depth_data[ocean_mask])

            lines.extend([
                f"  Depth range: {depth_min:.1f}m to {depth_max:.1f}m",
                f"  Ocean cells: {n_ocean} ({100*n_ocean/self.depth_data.size:.1f}%)",
                f"  Land cells: {n_land} ({100*n_land/self.depth_data.size:.1f}%)",
            ])

        return '\n'.join(lines)

    # =========================================================================
    # SWAN Commands
    # =========================================================================

    def to_swan_file(self, filepath: Path, idla: int = 4) -> Path:
        """
        Export bathymetry in SWAN format (legacy method, use save() instead).

        Args:
            filepath: Path to write the .bot file
            idla: SWAN layout parameter (default 4 = left-to-right from lower-left)

        Returns:
            Path to the written file
        """
        if self.depth_data is None:
            raise ValueError("No depth data. Call from_gebco() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._write_bot_file(filepath, idla)
        return filepath

    def generate_cgrid_command(self) -> str:
        """
        Generate the SWAN CGRID command string.

        The CGRID defines the computational grid where SWAN calculates wave spectra.

        Returns:
            SWAN command string for computational grid
        """
        if self.origin is None or self.nx is None:
            raise ValueError("Mesh not initialized. Call from_gebco() first.")

        lon_origin, lat_origin = self.origin

        # Calculate domain extents
        xlenc = self.nx * self.dx  # Domain length in x (degrees)
        ylenc = self.ny * self.dy  # Domain length in y (degrees)

        # mxc, myc = number of meshes (cells), not grid points
        mxc = self.nx
        myc = self.ny

        # CGRID REGular xpc ypc alpc xlenc ylenc mxc myc &
        #       CIRCLE mdc flow fhigh [msc]
        cmd = (
            f"CGRID REG {lon_origin:.4f} {lat_origin:.4f} 0 "
            f"{xlenc:.4f} {ylenc:.4f} {mxc} {myc} "
            f"{self.dir_type} {self.n_dir} {self.freq_min} {self.freq_max} {self.n_freq}"
        )
        return cmd

    def generate_inpgrid_command(self) -> str:
        """
        Generate the SWAN INPGRID BOTTOM command string.

        Returns:
            SWAN command string for this mesh's bathymetry
        """
        if self.origin is None or self.nx is None:
            raise ValueError("Mesh not initialized. Call from_gebco() first.")

        lon_origin, lat_origin = self.origin

        # INPGRID BOTTOM REG xp yp alp mx my dx dy EXCEPTION exc
        cmd = (
            f"INPGRID BOTTOM REG {lon_origin:.4f} {lat_origin:.4f} 0 "
            f"{self.nx} {self.ny} {self.dx:.6f} {self.dy:.6f} "
            f"EXCEPTION {self.exception_value:.1f}"
        )
        return cmd

    def generate_readinp_command(self, filename: str, idla: int = 4) -> str:
        """
        Generate the SWAN READINP BOTTOM command string.

        Args:
            filename: Name of the bathymetry file
            idla: Layout parameter (default 4)

        Returns:
            SWAN command string
        """
        # READINP BOTTOM fac 'filename' idla nhedf form
        return f"READINP BOTTOM 1 '{filename}' {idla} 0 FREE"

    def __repr__(self) -> str:
        region_str = f"region='{self.region.name}'" if self.region else "region=None"
        grid_str = f"{self.nx}x{self.ny}" if self.nx else "uninitialized"
        return (
            f"Mesh(name='{self.name}', {region_str}, "
            f"resolution={self.resolution_km}km, grid={grid_str})"
        )
