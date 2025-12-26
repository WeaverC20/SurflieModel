"""
Bathymetry visualization tools.

Visualizes stitched bathymetry grids with:
- Bathymetry depth coloring
- WW3 valid data boundary overlay
- Resolution zone boundaries (GEBCO vs NCEI)
- Distance to coast contours
- Surf spot locations

Usage:
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --show-ww3
    python -m data.pipelines.bathymetry.visualizer --region socal --zoom
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    import matplotlib.patheffects as pe
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.pipelines.bathymetry.config import (
    STITCHED_DIR,
    VIZ_DIR,
    REGIONS,
    SURF_SPOTS,
    SWAN_CONFIG,
)


class BathymetryVisualizer:
    """
    Visualize bathymetry grids and SWAN domain coverage.

    Features:
    - Bathymetry depth visualization with proper colormap
    - WW3 boundary overlay (from WaveWatchFetcher)
    - Resolution zone boundaries
    - Surf spot markers
    - Distance to coast contours
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving figures. Defaults to VIZ_DIR.
        """
        self.output_dir = output_dir or VIZ_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_bathymetry_colormap(self):
        """Create a colormap suitable for bathymetry (blue for deep, green/tan for shallow)."""
        colors = [
            (0.0, "#08306b"),   # Very deep (dark blue)
            (0.2, "#2171b5"),   # Deep
            (0.4, "#6baed6"),   # Medium depth
            (0.6, "#c6dbef"),   # Shallow
            (0.8, "#deebf7"),   # Very shallow
            (0.9, "#f7fbff"),   # Near surface
            (1.0, "#ffffcc"),   # Beach/land (tan)
        ]
        return LinearSegmentedColormap.from_list("bathymetry", colors)

    def plot_bathymetry(
        self,
        data_path: Path,
        region_bounds: Optional[Tuple[float, float, float, float]] = None,
        show_ww3_boundary: bool = True,
        show_resolution_zones: bool = True,
        show_surf_spots: bool = True,
        show_distance_contours: bool = True,
        output_name: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150,
    ) -> plt.Figure:
        """
        Create comprehensive bathymetry visualization.

        Args:
            data_path: Path to stitched bathymetry NetCDF
            region_bounds: (lat_min, lat_max, lon_min, lon_max) to zoom to
            show_ww3_boundary: Overlay WW3 valid data boundary
            show_resolution_zones: Show GEBCO/NCEI transition zone
            show_surf_spots: Mark surf spot locations
            show_distance_contours: Show distance to coast contours
            output_name: Filename for saved figure
            figsize: Figure size
            dpi: Resolution for saved figure

        Returns:
            matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required. Install with: pip install xarray")

        # Load data
        print(f"Loading: {data_path}")
        ds = xr.open_dataset(data_path)

        # Get elevation (try different variable names)
        elev_var = None
        for var in ["elevation", "z", "topo", "Band1"]:
            if var in ds.data_vars:
                elev_var = var
                break
        if elev_var is None:
            raise ValueError(f"No elevation variable found. Available: {list(ds.data_vars)}")
        elev = ds[elev_var].values
        print(f"  Using elevation variable: {elev_var}")

        # Get coordinates (try different names)
        lat_coord = "lat" if "lat" in ds.coords else "latitude" if "latitude" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "longitude" if "longitude" in ds.coords else "x"
        lats = ds[lat_coord].values
        lons = ds[lon_coord].values
        print(f"  Using coordinates: {lat_coord}, {lon_coord}")

        # Get optional layers
        ncei_weight = ds["ncei_weight"].values if "ncei_weight" in ds else None
        coast_distance = ds["coast_distance_km"].values if "coast_distance_km" in ds else None

        # Create figure
        if CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Plot bathymetry
        cmap = self.create_bathymetry_colormap()
        vmin = np.nanmin(elev)
        vmax = min(50, np.nanmax(elev))  # Cap at 50m for land visibility

        if CARTOPY_AVAILABLE:
            im = ax.pcolormesh(
                lon_grid, lat_grid, elev,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
        else:
            im = ax.pcolormesh(
                lon_grid, lat_grid, elev,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                shading="auto",
            )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Elevation (m)", fontsize=11)

        # Add coastlines
        if CARTOPY_AVAILABLE:
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="black")
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)
            ax.add_feature(cfeature.LAND, facecolor="#e0e0e0", alpha=0.3)

        # Show distance contours
        if show_distance_contours and coast_distance is not None:
            contour_levels = [0.5, 3, 10, 25]
            if CARTOPY_AVAILABLE:
                cs = ax.contour(
                    lon_grid, lat_grid, coast_distance,
                    levels=contour_levels,
                    colors=["green", "orange", "red", "purple"],
                    linewidths=1.5,
                    linestyles="--",
                    transform=ccrs.PlateCarree(),
                )
            else:
                cs = ax.contour(
                    lon_grid, lat_grid, coast_distance,
                    levels=contour_levels,
                    colors=["green", "orange", "red", "purple"],
                    linewidths=1.5,
                    linestyles="--",
                )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%d km")

        # Show resolution zones (NCEI weight boundary)
        if show_resolution_zones and ncei_weight is not None:
            # Contour at weight = 0.5 (transition midpoint)
            if CARTOPY_AVAILABLE:
                ax.contour(
                    lon_grid, lat_grid, ncei_weight,
                    levels=[0.5],
                    colors=["magenta"],
                    linewidths=2,
                    linestyles="-",
                    transform=ccrs.PlateCarree(),
                )
            else:
                ax.contour(
                    lon_grid, lat_grid, ncei_weight,
                    levels=[0.5],
                    colors=["magenta"],
                    linewidths=2,
                    linestyles="-",
                )

        # Show WW3 boundary
        if show_ww3_boundary:
            self._add_ww3_boundary(ax, lats, lons)

        # Show surf spots
        if show_surf_spots:
            self._add_surf_spots(ax, lats, lons)

        # Set extent
        if region_bounds:
            lat_min, lat_max, lon_min, lon_max = region_bounds
        else:
            lat_min, lat_max = lats.min(), lats.max()
            lon_min, lon_max = lons.min(), lons.max()

        if CARTOPY_AVAILABLE:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.grid(True, alpha=0.3, linestyle="--")

        # Title and labels
        ax.set_title(
            "SWAN Bathymetry Grid\n(GEBCO + NCEI CRM stitched)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)

        # Legend
        legend_elements = []
        if show_distance_contours:
            legend_elements.extend([
                mpatches.Patch(color="green", label="0.5 km (SWAN output)"),
                mpatches.Patch(color="orange", label="3 km (GEBCO/NCEI boundary)"),
                mpatches.Patch(color="red", label="10 km offshore"),
                mpatches.Patch(color="purple", label="25 km (outer SWAN)"),
            ])
        if show_resolution_zones:
            legend_elements.append(
                mpatches.Patch(color="magenta", label="Resolution transition")
            )
        if show_ww3_boundary:
            legend_elements.append(
                mpatches.Patch(color="red", label="WW3 valid data boundary")
            )

        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()

        # Save
        if output_name:
            output_path = self.output_dir / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {output_path}")

        ds.close()
        return fig

    def _add_ww3_boundary(self, ax, lats: np.ndarray, lons: np.ndarray):
        """Add WW3 valid data boundary to plot."""
        try:
            from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher

            fetcher = WaveWatchFetcher()

            # Get sample WW3 data to find boundary
            result = fetcher.fetch_grib(
                lat_min=float(lats.min()),
                lat_max=float(lats.max()),
                lon_min=float(lons.min()),
                lon_max=float(lons.max()),
            )

            if result is None:
                print("Could not fetch WW3 data for boundary")
                return

            data, ww3_lats, ww3_lons = result

            # Find easternmost valid point at each latitude
            boundary_lons = []
            boundary_lats = []

            for i, lat in enumerate(ww3_lats):
                row = data[i, :]
                valid_indices = np.where(~np.isnan(row))[0]
                if len(valid_indices) > 0:
                    easternmost_idx = valid_indices[-1]
                    boundary_lons.append(ww3_lons[easternmost_idx])
                    boundary_lats.append(lat)

            if boundary_lons:
                if CARTOPY_AVAILABLE:
                    ax.plot(
                        boundary_lons, boundary_lats,
                        color="red", linewidth=2.5,
                        transform=ccrs.PlateCarree(),
                        path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
                        label="WW3 boundary",
                    )
                else:
                    ax.plot(
                        boundary_lons, boundary_lats,
                        color="red", linewidth=2.5,
                        path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
                        label="WW3 boundary",
                    )

        except Exception as e:
            print(f"Could not add WW3 boundary: {e}")

    def _add_surf_spots(self, ax, lats: np.ndarray, lons: np.ndarray):
        """Add surf spot markers to plot."""
        for spot_id, spot_info in SURF_SPOTS.items():
            lat, lon = spot_info["lat"], spot_info["lon"]

            # Check if in bounds
            if lat < lats.min() or lat > lats.max():
                continue
            if lon < lons.min() or lon > lons.max():
                continue

            if CARTOPY_AVAILABLE:
                ax.plot(
                    lon, lat, "o",
                    color="yellow", markersize=8,
                    markeredgecolor="black", markeredgewidth=1,
                    transform=ccrs.PlateCarree(),
                )
                ax.annotate(
                    spot_info["name"],
                    (lon, lat),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                )
            else:
                ax.plot(
                    lon, lat, "o",
                    color="yellow", markersize=8,
                    markeredgecolor="black", markeredgewidth=1,
                )
                ax.annotate(
                    spot_info["name"],
                    (lon, lat),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    def plot_resolution_comparison(
        self,
        gebco_path: Path,
        ncei_path: Path,
        stitched_path: Path,
        zoom_bounds: Tuple[float, float, float, float],
        output_name: str = "resolution_comparison.png",
        figsize: Tuple[int, int] = (18, 6),
    ) -> plt.Figure:
        """
        Create side-by-side comparison of resolution sources.

        Shows GEBCO, NCEI, and stitched grids for the same area.

        Args:
            gebco_path: Path to GEBCO data
            ncei_path: Path to NCEI data
            stitched_path: Path to stitched data
            zoom_bounds: (lat_min, lat_max, lon_min, lon_max) area to compare
            output_name: Output filename
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required")

        lat_min, lat_max, lon_min, lon_max = zoom_bounds

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Load and plot each source
        sources = [
            ("GEBCO (~450m)", gebco_path),
            ("NCEI CRM (~90m)", ncei_path),
            ("Stitched (SWAN)", stitched_path),
        ]

        cmap = self.create_bathymetry_colormap()

        for ax, (title, path) in zip(axes, sources):
            if not path.exists():
                ax.set_title(f"{title}\n(file not found)")
                ax.set_visible(False)
                continue

            ds = xr.open_dataset(path)

            # Find elevation variable
            elev_var = None
            for var in ["elevation", "z", "topo", "Band1"]:
                if var in ds.data_vars:
                    elev_var = var
                    break

            if elev_var is None:
                ax.set_title(f"{title}\n(no elevation data)")
                continue

            # Find coordinate names
            lat_coord = "lat" if "lat" in ds.coords else "latitude" if "latitude" in ds.coords else "y"
            lon_coord = "lon" if "lon" in ds.coords else "longitude" if "longitude" in ds.coords else "x"

            # Subset to zoom bounds
            subset = ds.sel(
                **{
                    lat_coord: slice(lat_min, lat_max),
                    lon_coord: slice(lon_min, lon_max),
                }
            )

            elev = subset[elev_var].values
            lats = subset[lat_coord].values
            lons = subset[lon_coord].values

            lon_grid, lat_grid = np.meshgrid(lons, lats)

            im = ax.pcolormesh(
                lon_grid, lat_grid, elev,
                cmap=cmap,
                vmin=-500, vmax=50,
                shading="auto",
            )

            ax.set_title(f"{title}\n{elev.shape[0]}x{elev.shape[1]} grid cells")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")

            ds.close()

        # Common colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label("Elevation (m)")

        plt.suptitle(
            f"Resolution Comparison: {lat_min:.2f}°N to {lat_max:.2f}°N",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

        return fig

    def plot_mesh(
        self,
        data_path: Path,
        region_bounds: Optional[Tuple[float, float, float, float]] = None,
        show_every_n: int = 1,
        show_bathymetry: bool = True,
        mesh_color: str = "black",
        mesh_alpha: float = 0.5,
        mesh_linewidth: float = 0.3,
        output_name: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150,
        interactive: bool = False,
    ) -> plt.Figure:
        """
        Visualize the bathymetry grid as a mesh to inspect resolution.

        Args:
            data_path: Path to bathymetry NetCDF file
            region_bounds: (lat_min, lat_max, lon_min, lon_max) to zoom to
            show_every_n: Show every Nth grid line (1=all, 10=every 10th, etc.)
            show_bathymetry: Show bathymetry colors underneath mesh
            mesh_color: Color of mesh lines
            mesh_alpha: Transparency of mesh lines
            mesh_linewidth: Width of mesh lines
            output_name: Filename for saved figure (None = don't save)
            figsize: Figure size
            dpi: Resolution for saved figure
            interactive: If True, enable interactive zooming (don't close dataset)

        Returns:
            matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required. Install with: pip install xarray")

        # Load data
        print(f"Loading: {data_path}")
        ds = xr.open_dataset(data_path)

        # Get elevation variable
        elev_var = None
        for var in ["elevation", "z", "topo", "Band1"]:
            if var in ds.data_vars:
                elev_var = var
                break
        if elev_var is None:
            raise ValueError(f"No elevation variable found. Available: {list(ds.data_vars)}")

        # Get coordinates
        lat_coord = "lat" if "lat" in ds.coords else "latitude" if "latitude" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "longitude" if "longitude" in ds.coords else "x"

        # Subset if region specified
        if region_bounds:
            lat_min, lat_max, lon_min, lon_max = region_bounds
            ds = ds.sel(**{
                lat_coord: slice(lat_min, lat_max),
                lon_coord: slice(lon_min, lon_max),
            })
            print(f"  Subset to region: {lat_min:.2f}-{lat_max:.2f}°N, {lon_min:.2f}-{lon_max:.2f}°W")

        elev = ds[elev_var].values
        lats = ds[lat_coord].values
        lons = ds[lon_coord].values

        print(f"  Grid size: {len(lats)} × {len(lons)} cells")
        print(f"  Lat range: {lats.min():.4f} to {lats.max():.4f}")
        print(f"  Lon range: {lons.min():.4f} to {lons.max():.4f}")

        # Calculate cell sizes
        if len(lats) > 1:
            lat_spacing_deg = np.abs(lats[1] - lats[0])
            lat_spacing_m = lat_spacing_deg * 111000  # ~111km per degree
            print(f"  Lat spacing: {lat_spacing_deg:.6f}° (~{lat_spacing_m:.1f}m)")
        if len(lons) > 1:
            lon_spacing_deg = np.abs(lons[1] - lons[0])
            avg_lat = np.mean(lats)
            lon_spacing_m = lon_spacing_deg * 111000 * np.cos(np.radians(avg_lat))
            print(f"  Lon spacing: {lon_spacing_deg:.6f}° (~{lon_spacing_m:.1f}m)")

        # Downsample for visualization if needed
        if show_every_n > 1:
            lats_mesh = lats[::show_every_n]
            lons_mesh = lons[::show_every_n]
            elev_mesh = elev[::show_every_n, ::show_every_n]
            print(f"  Showing every {show_every_n}th grid line ({len(lats_mesh)} × {len(lons_mesh)} lines)")
        else:
            lats_mesh = lats
            lons_mesh = lons
            elev_mesh = elev

        # Create figure
        if CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons_mesh, lats_mesh)

        # Plot bathymetry underneath if requested
        if show_bathymetry:
            cmap = self.create_bathymetry_colormap()
            vmin = np.nanmin(elev_mesh)
            vmax = min(50, np.nanmax(elev_mesh))

            if CARTOPY_AVAILABLE:
                im = ax.pcolormesh(
                    lon_grid, lat_grid, elev_mesh,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                    alpha=0.7,
                )
            else:
                im = ax.pcolormesh(
                    lon_grid, lat_grid, elev_mesh,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    shading="auto",
                    alpha=0.7,
                )
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label("Elevation (m)", fontsize=11)

        # Draw mesh lines
        # Horizontal lines (constant latitude)
        for i, lat in enumerate(lats_mesh):
            if CARTOPY_AVAILABLE:
                ax.plot(
                    lons_mesh, [lat] * len(lons_mesh),
                    color=mesh_color, alpha=mesh_alpha, linewidth=mesh_linewidth,
                    transform=ccrs.PlateCarree(),
                )
            else:
                ax.plot(
                    lons_mesh, [lat] * len(lons_mesh),
                    color=mesh_color, alpha=mesh_alpha, linewidth=mesh_linewidth,
                )

        # Vertical lines (constant longitude)
        for j, lon in enumerate(lons_mesh):
            if CARTOPY_AVAILABLE:
                ax.plot(
                    [lon] * len(lats_mesh), lats_mesh,
                    color=mesh_color, alpha=mesh_alpha, linewidth=mesh_linewidth,
                    transform=ccrs.PlateCarree(),
                )
            else:
                ax.plot(
                    [lon] * len(lats_mesh), lats_mesh,
                    color=mesh_color, alpha=mesh_alpha, linewidth=mesh_linewidth,
                )

        # Add coastlines if available
        if CARTOPY_AVAILABLE:
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor="red")
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)

        # Set extent
        if region_bounds:
            lat_min, lat_max, lon_min, lon_max = region_bounds
        else:
            lat_min, lat_max = lats.min(), lats.max()
            lon_min, lon_max = lons.min(), lons.max()

        if CARTOPY_AVAILABLE:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--", color="blue")
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.grid(True, alpha=0.3, linestyle="--", color="blue")
            ax.set_aspect("equal")

        # Title with grid info
        title = f"Bathymetry Grid Mesh\n{len(lats)} × {len(lons)} cells"
        if show_every_n > 1:
            title += f" (showing every {show_every_n}th line)"
        if len(lats) > 1 and len(lons) > 1:
            title += f"\nResolution: ~{lat_spacing_m:.0f}m × ~{lon_spacing_m:.0f}m per cell"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)

        plt.tight_layout()

        # Save if requested
        if output_name:
            output_path = self.output_dir / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if not interactive:
            ds.close()

        return fig

    def plot_mesh_interactive(
        self,
        data_path: Path,
        initial_region: Optional[str] = None,
    ):
        """
        Launch interactive mesh viewer with zoom capabilities.

        Use matplotlib's interactive mode to zoom and pan around the grid.
        Press 'h' for help, scroll to zoom, drag to pan.

        Args:
            data_path: Path to bathymetry NetCDF
            initial_region: Start zoomed to a predefined region
        """
        print("=" * 60)
        print("INTERACTIVE MESH VIEWER")
        print("=" * 60)
        print()
        print("Controls:")
        print("  - Scroll wheel: Zoom in/out")
        print("  - Click and drag: Pan")
        print("  - 'h' key: Reset to full view")
        print("  - 'q' key: Quit")
        print()
        print("The mesh lines show individual grid cells.")
        print("Zoom in to see resolution at specific locations.")
        print()

        # Get initial bounds
        region_bounds = REGIONS.get(initial_region) if initial_region else None

        # Create the figure
        fig = self.plot_mesh(
            data_path=data_path,
            region_bounds=region_bounds,
            show_every_n=50,  # Start with sparse view
            show_bathymetry=True,
            interactive=True,
        )

        # Add instructions text
        ax = fig.axes[0]
        ax.text(
            0.02, 0.02,
            "Scroll to zoom | Drag to pan | 'h' to reset | 'q' to quit",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.ion()  # Interactive mode
        plt.show(block=True)

    def plot_coverage_overview(
        self,
        output_name: str = "coverage_overview.png",
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """
        Create schematic overview of bathymetry coverage zones.

        Shows conceptual diagram of:
        - WW3 domain boundary
        - GEBCO domain (25km → 3km)
        - NCEI domain (3km → 500m)
        - USACE LiDAR domain (500m → beach)
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Use California as example
        bounds = REGIONS["california"]
        lat_min, lat_max, lon_min, lon_max = bounds

        # Draw conceptual zones as bands
        # This is a simplified visualization showing the offshore distances

        ax.set_xlim(0, 30)  # km from coast
        ax.set_ylim(0, 4)

        # Zone colors
        zones = [
            (0, 0.5, "#2ecc71", "USACE LiDAR\n(1m res)"),
            (0.5, 3, "#3498db", "NCEI CRM\n(90m res)"),
            (3, 25, "#9b59b6", "GEBCO\n(450m res)"),
            (25, 30, "#e74c3c", "WW3\n(25km res)"),
        ]

        for x1, x2, color, label in zones:
            ax.axvspan(x1, x2, alpha=0.6, color=color, label=label)
            ax.axvline(x1, color="black", linewidth=0.5)

        ax.axvline(25, color="black", linewidth=2, linestyle="--", label="WW3 valid data boundary")

        ax.set_xlabel("Distance from Coast (km)", fontsize=12)
        ax.set_title(
            "SWAN Bathymetry Coverage Zones\n(Conceptual)",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="upper right")

        # Add annotations
        ax.annotate(
            "SWAN output\nterminates here",
            xy=(0.5, 3.5), fontsize=10,
            ha="center",
        )
        ax.annotate(
            "Transition\nzone",
            xy=(3, 3.5), fontsize=10,
            ha="center",
        )
        ax.annotate(
            "WW3 provides\nboundary conditions",
            xy=(27.5, 3.5), fontsize=10,
            ha="center",
        )

        ax.set_yticks([])

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bathymetry grids for SWAN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot stitched bathymetry
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc

    # Show with WW3 boundary overlay
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --show-ww3

    # Zoom to region
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --region socal

    # View grid as mesh to inspect resolution
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --mesh --show

    # Mesh view of specific region (e.g., around Huntington Beach)
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --mesh --region la_area --show

    # Sparse mesh view (every 10th line for large grids)
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --mesh --every-n 10 --show

    # Interactive mesh viewer with zoom/pan
    python -m data.pipelines.bathymetry.visualizer --file california_swan.nc --mesh-interactive

    # Create coverage overview schematic
    python -m data.pipelines.bathymetry.visualizer --overview

    # Compare resolutions
    python -m data.pipelines.bathymetry.visualizer --compare \\
        --gebco gebco.nc --ncei ncei.nc --stitched stitched.nc \\
        --zoom 33.5 34.2 -118.5 -117.5
""",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path to stitched bathymetry NetCDF",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=list(REGIONS.keys()),
        help="Zoom to predefined region",
    )
    parser.add_argument(
        "--show-ww3",
        action="store_true",
        help="Show WW3 valid data boundary",
    )
    parser.add_argument(
        "--no-spots",
        action="store_true",
        help="Hide surf spot markers",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Create coverage overview schematic",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare resolution sources",
    )
    parser.add_argument(
        "--gebco",
        type=str,
        help="GEBCO file for comparison",
    )
    parser.add_argument(
        "--ncei",
        type=str,
        help="NCEI file for comparison",
    )
    parser.add_argument(
        "--stitched",
        type=str,
        help="Stitched file for comparison",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Zoom bounds for comparison",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Show grid as mesh to inspect resolution",
    )
    parser.add_argument(
        "--mesh-interactive",
        action="store_true",
        help="Launch interactive mesh viewer with zoom/pan",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Show every Nth grid line (default: 1 = all lines)",
    )

    args = parser.parse_args()

    visualizer = BathymetryVisualizer()

    if args.overview:
        visualizer.plot_coverage_overview()
        if args.show:
            plt.show()
        return

    if args.compare:
        if not all([args.gebco, args.ncei, args.stitched, args.zoom]):
            print("--compare requires --gebco, --ncei, --stitched, and --zoom")
            return
        visualizer.plot_resolution_comparison(
            gebco_path=Path(args.gebco),
            ncei_path=Path(args.ncei),
            stitched_path=Path(args.stitched),
            zoom_bounds=tuple(args.zoom),
        )
        if args.show:
            plt.show()
        return

    if args.mesh_interactive and args.file:
        visualizer.plot_mesh_interactive(
            data_path=Path(args.file),
            initial_region=args.region,
        )
        return

    if args.mesh and args.file:
        region_bounds = REGIONS.get(args.region) if args.region else None
        visualizer.plot_mesh(
            data_path=Path(args.file),
            region_bounds=region_bounds,
            show_every_n=args.every_n,
            output_name=args.output or "mesh.png",
        )
        if args.show:
            plt.show()
        return

    if args.file:
        region_bounds = REGIONS.get(args.region) if args.region else None
        visualizer.plot_bathymetry(
            data_path=Path(args.file),
            region_bounds=region_bounds,
            show_ww3_boundary=args.show_ww3,
            show_surf_spots=not args.no_spots,
            output_name=args.output or "bathymetry.png",
        )
        if args.show:
            plt.show()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
