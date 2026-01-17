#!/usr/bin/env python3
"""
SWAN Results Plotter - Spectral Boundary Conditions

Visualizations of SWAN outputs with spectral/swell component breakdown.
Shows period-colored arrows indicating wave height and direction for multiple
swell components.

For basic integrated plots (Hsig, Tps, Dir), use plot_results_par.py.

Saves plots to data/swan/analysis/plots/{region}/{mesh}/
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.analysis.output_reader import SwanOutput, SwanOutputReader, WavePartitionGrid

logger = logging.getLogger(__name__)

# Base directory for saving plots
ANALYSIS_DIR = Path(__file__).parent
PLOTS_DIR = ANALYSIS_DIR / "plots"


# =============================================================================
# Configuration
# =============================================================================

# Number of top swells to display (configurable)
MAX_SWELLS_TO_DISPLAY = 3

# Period bands for coloring (seconds)
PERIOD_BANDS = {
    "wind_chop": (0, 8),      # Blue - short period wind chop
    "short_swell": (8, 12),    # Green/Yellow - medium period
    "long_swell": (12, 25),    # Orange/Red - long period groundswell
}

# Color mapping: period (seconds) -> RGB color
# Using a continuous colormap from blue (short period) to red (long period)
def period_to_color(period: float) -> str:
    """
    Map wave period to color.

    Short period (wind chop, <8s) -> Blue
    Medium period (8-12s) -> Yellow/Green
    Long period (>12s) -> Orange/Red

    Args:
        period: Wave period in seconds

    Returns:
        Matplotlib color string
    """
    if period < 8:
        # Blue range for wind chop
        return '#3498db'  # Bright blue
    elif period < 12:
        # Yellow/green for medium swells
        t = (period - 8) / 4  # 0 to 1
        return '#f39c12' if t > 0.5 else '#27ae60'  # Orange or green
    else:
        # Red/orange for long period groundswell
        return '#e74c3c'  # Red


@dataclass
class SwellComponent:
    """A single swell component with wave parameters."""
    hs: float           # Significant wave height (m)
    tp: float           # Peak period (s)
    direction: float    # Wave direction (degrees, nautical - coming FROM)
    spread: float = 25.0  # Directional spread (degrees)
    label: str = ""     # Component label (e.g., "wind_sea", "swell_1")

    @property
    def color(self) -> str:
        """Get color based on period."""
        return period_to_color(self.tp)

    @property
    def energy(self) -> float:
        """Approximate energy (proportional to Hs^2)."""
        return self.hs ** 2


@dataclass
class GridPointSpectrum:
    """Spectral data at a single grid point."""
    lon: float
    lat: float
    components: List[SwellComponent] = field(default_factory=list)

    def top_components(self, n: int = MAX_SWELLS_TO_DISPLAY) -> List[SwellComponent]:
        """Return top N components sorted by energy (Hs^2)."""
        sorted_components = sorted(self.components, key=lambda c: c.energy, reverse=True)
        return sorted_components[:n]


class SpectralBoundaryReader:
    """
    Reads spectral boundary data from SWAN run.

    Parses the boundary.sp2 file or reconstructs from WW3 partition data.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.boundary_file = self.run_dir / "boundary.sp2"

    def read_boundary_points(self) -> List[GridPointSpectrum]:
        """
        Read boundary spectral data.

        First tries to load WW3 partition data from cached JSON,
        then falls back to parsing boundary.sp2 file.

        Returns:
            List of GridPointSpectrum objects for boundary points
        """
        # Try to load cached partition data
        partition_cache = self.run_dir / "ww3_partitions.json"
        if partition_cache.exists():
            return self._read_from_partition_cache(partition_cache)

        # Fall back to parsing boundary.sp2
        if self.boundary_file.exists():
            return self._read_from_spec2d()

        logger.warning("No spectral boundary data found")
        return []

    def _read_from_partition_cache(self, cache_path: Path) -> List[GridPointSpectrum]:
        """Read from cached WW3 partition JSON."""
        with open(cache_path) as f:
            data = json.load(f)

        points = []
        for pt in data["points"]:
            spectrum = GridPointSpectrum(lon=pt["lon"], lat=pt["lat"])

            # Add wind sea component
            if pt.get("wind_sea"):
                ws = pt["wind_sea"]
                spectrum.components.append(SwellComponent(
                    hs=ws["hs"], tp=ws["tp"], direction=ws["dir"],
                    spread=ws.get("spread", 30.0), label="Wind Sea"
                ))

            # Add primary swell
            if pt.get("swell_1"):
                s1 = pt["swell_1"]
                spectrum.components.append(SwellComponent(
                    hs=s1["hs"], tp=s1["tp"], direction=s1["dir"],
                    spread=s1.get("spread", 20.0), label="Primary Swell"
                ))

            # Add secondary swell
            if pt.get("swell_2"):
                s2 = pt["swell_2"]
                spectrum.components.append(SwellComponent(
                    hs=s2["hs"], tp=s2["tp"], direction=s2["dir"],
                    spread=s2.get("spread", 20.0), label="Secondary Swell"
                ))

            points.append(spectrum)

        return points

    def _read_from_spec2d(self) -> List[GridPointSpectrum]:
        """
        Parse boundary.sp2 file and extract dominant components.

        This re-partitions the 2D spectrum into period bands.
        """
        if not self.boundary_file.exists():
            return []

        with open(self.boundary_file) as f:
            lines = f.readlines()

        points = []
        i = 0
        n_points = 0
        n_freq = 0
        n_dir = 0
        frequencies = []
        directions = []

        while i < len(lines):
            line = lines[i].strip()

            # Skip comments
            if line.startswith('$') or line.startswith('SWAN'):
                i += 1
                continue

            # Parse header sections
            if line == 'LONLAT':
                i += 1
                n_points = int(lines[i].strip())
                i += 1
                lons = []
                lats = []
                for j in range(n_points):
                    parts = lines[i].strip().split()
                    lons.append(float(parts[0]))
                    lats.append(float(parts[1]))
                    i += 1
                continue

            if line == 'AFREQ':
                i += 1
                n_freq = int(lines[i].strip())
                i += 1
                frequencies = []
                for j in range(n_freq):
                    frequencies.append(float(lines[i].strip()))
                    i += 1
                continue

            if line == 'NDIR':
                i += 1
                n_dir = int(lines[i].strip())
                i += 1
                directions = []
                for j in range(n_dir):
                    directions.append(float(lines[i].strip()))
                    i += 1
                continue

            if line == 'QUANT':
                # Skip QUANT section
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('FACTOR'):
                    i += 1
                continue

            if line == 'FACTOR':
                # Read spectrum for this point
                i += 1
                factor = float(lines[i].strip())
                i += 1

                # Read spectral values
                spectrum = np.zeros((n_freq, n_dir))
                for f_idx in range(n_freq):
                    values = lines[i].strip().split()
                    for d_idx, val in enumerate(values):
                        spectrum[f_idx, d_idx] = float(val) * factor
                    i += 1

                # Partition spectrum into components
                point_idx = len(points)
                if point_idx < n_points:
                    spectrum_obj = GridPointSpectrum(
                        lon=lons[point_idx],
                        lat=lats[point_idx]
                    )

                    # Extract components by period band
                    components = self._partition_spectrum(
                        spectrum, frequencies, directions
                    )
                    spectrum_obj.components = components
                    points.append(spectrum_obj)

                continue

            i += 1

        return points

    def _partition_spectrum(
        self,
        E: np.ndarray,
        frequencies: List[float],
        directions: List[float]
    ) -> List[SwellComponent]:
        """
        Partition 2D spectrum into swell components by period band.

        Args:
            E: 2D spectral density array (n_freq x n_dir)
            frequencies: Frequency values (Hz)
            directions: Direction values (degrees)

        Returns:
            List of SwellComponent objects
        """
        components = []
        periods = 1.0 / np.array(frequencies)
        directions = np.array(directions)

        # Define period bands
        bands = [
            ("Wind Chop", 0, 8),
            ("Short Swell", 8, 12),
            ("Groundswell", 12, 30),
        ]

        for label, t_min, t_max in bands:
            # Find frequency indices for this period band
            # Note: higher period = lower frequency
            f_max = 1.0 / t_min if t_min > 0 else np.inf
            f_min = 1.0 / t_max

            freq_mask = (np.array(frequencies) >= f_min) & (np.array(frequencies) < f_max)

            if not np.any(freq_mask):
                continue

            # Extract energy in this band
            E_band = E[freq_mask, :]

            # Calculate Hs for this band
            # Hs = 4 * sqrt(m0), m0 = integral of E
            df = np.diff(frequencies)
            df = np.append(df, df[-1])  # Extend to match length
            d_theta = 360.0 / len(directions)

            m0 = np.sum(E_band * df[freq_mask, np.newaxis] * d_theta)
            hs = 4.0 * np.sqrt(max(m0, 0))

            if hs < 0.1:  # Skip negligible components
                continue

            # Find peak direction in this band
            dir_energy = np.sum(E_band, axis=0)
            peak_dir_idx = np.argmax(dir_energy)
            peak_dir = directions[peak_dir_idx]

            # Find peak period in this band
            freq_energy = np.sum(E_band, axis=1)
            peak_freq_idx = np.argmax(freq_energy)
            peak_freq = frequencies[np.where(freq_mask)[0][peak_freq_idx]]
            peak_period = 1.0 / peak_freq

            components.append(SwellComponent(
                hs=hs,
                tp=peak_period,
                direction=peak_dir,
                label=label
            ))

        return components


@dataclass
class SpectralPlotter:
    """
    Creates spectral visualization plots from SWAN model output.

    Shows period-colored arrows representing different swell components.

    Plots are saved to: data/swan/analysis/plots/{region}/{mesh}/{timestamp}/

    Example usage:
        plotter = SpectralPlotter("data/swan/runs/socal/coarse/latest")
        plotter.plot_swell_components()
    """

    run_dir: str | Path
    output: Optional[SwanOutput] = None
    plots_dir: Optional[Path] = None
    max_swells: int = MAX_SWELLS_TO_DISPLAY
    extent_padding: float = 0.15  # Degrees to pad extent for arrow visibility

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)

        # Load SWAN output if not provided
        if self.output is None:
            reader = SwanOutputReader(self.run_dir)
            self.output = reader.read()

        # Setup plots directory
        if self.plots_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            self.plots_dir = (
                PLOTS_DIR
                / self.output.region_name
                / self.output.mesh_name.replace(f"{self.output.region_name}_", "")
                / timestamp
            )

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots will be saved to: {self.plots_dir}")

        # Load spectral boundary data
        self.boundary_reader = SpectralBoundaryReader(self.run_dir)

    def _get_padded_extent(self) -> Tuple[float, float, float, float]:
        """
        Get map extent with padding for arrow visibility at boundaries.

        Returns:
            (lon_min, lon_max, lat_min, lat_max) with padding applied
        """
        lon_min, lon_max, lat_min, lat_max = self.output.extent
        pad = self.extent_padding
        return (lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad)

    def _get_hsig_vmax(self, data: Optional[np.ndarray] = None) -> float:
        """
        Get a rounded-up max value for Hsig colorbar.

        Uses the actual maximum in the data, rounded up to the nearest integer.

        Args:
            data: Optional array to compute max from. If None, uses self.output.hsig.

        Returns:
            Rounded up max Hsig value (minimum 1.0)
        """
        if data is None:
            hsig_masked, _, _ = self.output.mask_land()
            max_val = np.nanmax(hsig_masked)
        else:
            max_val = np.nanmax(data[~np.isnan(data)]) if np.any(~np.isnan(data)) else 1.0

        # Round up to nearest integer, minimum of 1
        return max(1.0, float(np.ceil(max_val)))

    def plot_swell_components(
        self,
        arrow_scale: float = 0.15,
        arrow_width: float = 0.003,
        skip_grid: int = 4,
        hsig_vmax: Optional[float] = None,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Plot Hsig heatmap with period-colored swell arrows.

        At each grid point, shows up to max_swells arrows representing
        different swell components. Arrow color indicates period:
        - Blue: Wind chop (<8s)
        - Yellow/Green: Short-period swell (8-12s)
        - Red/Orange: Long-period groundswell (>12s)

        Args:
            arrow_scale: Scale factor for arrow length
            arrow_width: Arrow width
            skip_grid: Skip every N grid points for clarity
            hsig_vmax: Max value for Hsig colorbar (None = auto-scale to data)
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.patches import FancyArrow
        from matplotlib.lines import Line2D

        hsig_masked, tps_masked, dir_masked = self.output.mask_land()

        # Auto-scale colorbar if not specified
        if hsig_vmax is None:
            hsig_vmax = self._get_hsig_vmax(hsig_masked)

        fig, ax = plt.subplots(
            figsize=(14, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        # Plot Hsig as background
        im = ax.pcolormesh(
            LON, LAT, hsig_masked,
            cmap='YlOrRd',
            vmin=0, vmax=hsig_vmax,
            shading='auto',
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_extent(self._get_padded_extent(), crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Collect arrow data for quiver plot, grouped by period band
        arrows_by_period = {
            'wind_chop': {'x': [], 'y': [], 'u': [], 'v': []},
            'short_swell': {'x': [], 'y': [], 'u': [], 'v': []},
            'long_swell': {'x': [], 'y': [], 'u': [], 'v': []},
        }

        for i in range(0, len(self.output.lats), skip_grid):
            for j in range(0, len(self.output.lons), skip_grid):
                hs = self.output.hsig[i, j]
                tp = self.output.tps[i, j]
                wave_dir = self.output.dir[i, j]

                # Skip land/invalid points (check for exception value AND NaN)
                if hs == self.output.exception_value or np.isnan(hs):
                    continue
                if np.isnan(tp) or np.isnan(wave_dir):
                    continue

                # Convert nautical direction (FROM) to arrow direction (TO)
                # Nautical: 0=N, 90=E, direction waves come FROM
                # For arrow: we want to show where waves are going
                arrow_dir = (wave_dir + 180) % 360

                # Convert to radians (math convention: 0=E, CCW positive)
                theta = np.radians(90 - arrow_dir)

                # Arrow components proportional to Hs
                u = hs * arrow_scale * np.cos(theta)
                v = hs * arrow_scale * np.sin(theta)

                # Categorize by period band
                if tp < 8:
                    band = 'wind_chop'
                elif tp < 12:
                    band = 'short_swell'
                else:
                    band = 'long_swell'

                arrows_by_period[band]['x'].append(self.output.lons[j])
                arrows_by_period[band]['y'].append(self.output.lats[i])
                arrows_by_period[band]['u'].append(u)
                arrows_by_period[band]['v'].append(v)

        # Plot quiver for each period band
        period_colors = {
            'wind_chop': '#3498db',
            'short_swell': '#27ae60',
            'long_swell': '#e74c3c',
        }

        for band, data in arrows_by_period.items():
            if data['x']:
                ax.quiver(
                    data['x'], data['y'], data['u'], data['v'],
                    color=period_colors[band],
                    scale=1, scale_units='xy',
                    width=0.004,
                    headwidth=4, headlength=5,
                    alpha=0.85,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )

        # Colorbar for Hsig
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Significant Wave Height (m)')

        # Legend for period colors
        legend_elements = [
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#3498db',
                   markersize=12, label='Wind Chop (<8s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#27ae60',
                   markersize=12, label='Short Swell (8-12s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#e74c3c',
                   markersize=12, label='Groundswell (>12s)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title
        if title is None:
            title = f"Wave Field with Period-Colored Arrows - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / "spectral_arrows.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_partition_arrows(
        self,
        arrow_scale: float = 0.20,
        skip_grid: int = 3,
        hsig_vmax: Optional[float] = None,
        min_hs_threshold: float = 0.05,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Optional[Path]:
        """
        Plot multiple swell arrows per grid point from partition outputs.

        Uses SWAN partition outputs (wind sea + swells) to draw multiple
        period-colored arrows at each grid point, showing the individual
        wave components propagating through the domain.

        Arrow properties:
        - Color: Based on period (blue=short, green=medium, red=long)
        - Length: Proportional to wave height (Hs)
        - Direction: Wave propagation direction (TO, not FROM)

        Args:
            arrow_scale: Scale factor for arrow length (default 0.20)
            skip_grid: Skip every N grid points for clarity (default 3)
            hsig_vmax: Max value for Hsig colorbar (None = auto-scale to data)
            min_hs_threshold: Minimum Hs (m) to draw an arrow (default 0.05)
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot, or None if no partition data
        """
        if not self.output.has_partitions:
            logger.info("No partition data available - skipping partition arrows plot")
            logger.info("(Re-run SWAN with partition outputs enabled)")
            return None

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.lines import Line2D

        hsig_masked, _, _ = self.output.mask_land()

        # Auto-scale colorbar if not specified
        if hsig_vmax is None:
            hsig_vmax = self._get_hsig_vmax(hsig_masked)

        fig, ax = plt.subplots(
            figsize=(14, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        # Plot total Hsig as background
        im = ax.pcolormesh(
            LON, LAT, hsig_masked,
            cmap='YlOrRd',
            vmin=0, vmax=hsig_vmax,
            shading='auto',
            alpha=0.6,
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_extent(self._get_padded_extent(), crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Collect arrow data grouped by period band for each partition
        # Colors chosen to contrast with YlOrRd heatmap (avoiding red/orange/yellow)
        period_colors = {
            'wind_chop': '#17a2b8',    # Cyan - short period wind waves
            'short_swell': '#8e44ad',  # Purple - medium period swell
            'long_swell': '#1a252f',   # Dark navy - long period groundswell
        }

        arrows_by_period = {band: {'x': [], 'y': [], 'u': [], 'v': [], 'labels': []}
                           for band in period_colors}

        # Offset per partition to visually separate arrows at same grid point
        # Offsets are applied perpendicular to create a fan-like arrangement
        offset_scale = 0.035  # ~3.5km offset between partition arrows
        partition_offsets = {
            0: (0, 0),           # Primary partition - center
            1: (-0.8, -0.6),     # Secondary - SW offset
            2: (0.8, -0.6),      # Tertiary - SE offset
            3: (0, 0.8),         # Quaternary - N offset
        }

        for partition in self.output.partitions:
            hs_masked, tp_masked, dir_masked = partition.mask_invalid(self.output.exception_value)

            # Get offset for this partition
            off_x, off_y = partition_offsets.get(partition.partition_id, (0, 0))
            off_x *= offset_scale
            off_y *= offset_scale

            for i in range(0, len(self.output.lats), skip_grid):
                for j in range(0, len(self.output.lons), skip_grid):
                    hs = hs_masked[i, j]
                    tp = tp_masked[i, j]
                    wave_dir = dir_masked[i, j]

                    # Skip invalid/small points
                    if np.isnan(hs) or hs < min_hs_threshold:
                        continue
                    if np.isnan(tp) or np.isnan(wave_dir):
                        continue

                    # Convert nautical direction (FROM) to arrow direction (TO)
                    arrow_dir = (wave_dir + 180) % 360

                    # Convert to radians (math convention: 0=E, CCW positive)
                    theta = np.radians(90 - arrow_dir)

                    # Arrow components proportional to Hs
                    u = hs * arrow_scale * np.cos(theta)
                    v = hs * arrow_scale * np.sin(theta)

                    # Categorize by period band
                    if tp < 8:
                        band = 'wind_chop'
                    elif tp < 12:
                        band = 'short_swell'
                    else:
                        band = 'long_swell'

                    arrows_by_period[band]['x'].append(self.output.lons[j] + off_x)
                    arrows_by_period[band]['y'].append(self.output.lats[i] + off_y)
                    arrows_by_period[band]['u'].append(u)
                    arrows_by_period[band]['v'].append(v)
                    arrows_by_period[band]['labels'].append(partition.label)

        # Plot quiver for each period band
        for band, data in arrows_by_period.items():
            if data['x']:
                ax.quiver(
                    data['x'], data['y'], data['u'], data['v'],
                    color=period_colors[band],
                    scale=1, scale_units='xy',
                    width=0.003,
                    headwidth=2.5, headlength=3,
                    alpha=0.85,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )

        # Colorbar for total Hsig
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Total Significant Wave Height (m)')

        # Legend for period colors (matching arrow colors that contrast with heatmap)
        legend_elements = [
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#17a2b8',
                   markersize=12, label='Wind Chop (<8s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#8e44ad',
                   markersize=12, label='Short Swell (8-12s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#1a252f',
                   markersize=12, label='Groundswell (>12s)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title
        if title is None:
            n_partitions = len(self.output.partitions)
            partition_names = [p.label for p in self.output.partitions]
            title = f"Wave Partitions ({n_partitions}): {', '.join(partition_names)}\n{self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / "partition_arrows.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_wind_sea(
        self,
        arrow_scale: float = 0.20,
        skip_grid: int = 3,
        hsig_vmax: Optional[float] = None,
        min_hs_threshold: float = 0.05,
        period_threshold: float = 8.0,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Optional[Path]:
        """
        Plot short-period waves (wind sea) from ALL partitions.

        Aggregates ALL waves with Tp < period_threshold from any partition,
        not just partition 0. At each grid point, shows the dominant short-period
        wave (highest Hs among all partitions with Tp < threshold).

        Arrow properties:
        - Color: Blue (wind sea color)
        - Length: Proportional to wave height (Hs)
        - Direction: Wave propagation direction (TO, not FROM)

        Args:
            arrow_scale: Scale factor for arrow length
            skip_grid: Skip every N grid points for clarity
            hsig_vmax: Max value for Hsig colorbar (None = auto-scale to data)
            min_hs_threshold: Minimum Hs (m) to draw an arrow
            period_threshold: Maximum period (s) to consider as wind sea (default 8.0)
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot, or None if no short-period wave data
        """
        if not self.output.has_partitions:
            logger.info("No partition data available - skipping wind sea plot")
            return None

        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # Build aggregated short-period wave data from ALL partitions
        # For each grid cell, find the max Hs among all partitions with Tp < threshold
        grid_shape = self.output.hsig.shape
        agg_hs = np.full(grid_shape, np.nan)
        agg_tp = np.full(grid_shape, np.nan)
        agg_dir = np.full(grid_shape, np.nan)

        for partition in self.output.partitions:
            hs_masked, tp_masked, dir_masked = partition.mask_invalid(self.output.exception_value)

            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    hs = hs_masked[i, j]
                    tp = tp_masked[i, j]
                    wave_dir = dir_masked[i, j]

                    # Skip invalid points or periods >= threshold
                    if np.isnan(hs) or np.isnan(tp) or tp >= period_threshold:
                        continue

                    # Keep the highest Hs short-period wave at this point
                    if np.isnan(agg_hs[i, j]) or hs > agg_hs[i, j]:
                        agg_hs[i, j] = hs
                        agg_tp[i, j] = tp
                        agg_dir[i, j] = wave_dir

        # Check if there's any valid short-period data
        valid_count = np.sum(~np.isnan(agg_hs) & (agg_hs > min_hs_threshold))
        if valid_count == 0:
            logger.info("No significant short-period wave data - skipping wind sea plot")
            return None

        logger.info(f"Found {valid_count} grid points with Tp < {period_threshold}s waves")

        # Auto-scale colorbar if not specified
        if hsig_vmax is None:
            hsig_vmax = self._get_hsig_vmax(agg_hs)

        fig, ax = plt.subplots(
            figsize=(14, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        # Plot aggregated short-period Hs as background heatmap
        im = ax.pcolormesh(
            LON, LAT, agg_hs,
            cmap='Blues',
            vmin=0, vmax=hsig_vmax,
            shading='auto',
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_extent(self._get_padded_extent(), crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Collect arrow data
        arrow_x, arrow_y, arrow_u, arrow_v = [], [], [], []

        for i in range(0, len(self.output.lats), skip_grid):
            for j in range(0, len(self.output.lons), skip_grid):
                hs = agg_hs[i, j]
                tp = agg_tp[i, j]
                wave_dir = agg_dir[i, j]

                # Skip invalid/small points
                if np.isnan(hs) or hs < min_hs_threshold:
                    continue
                if np.isnan(tp) or np.isnan(wave_dir):
                    continue

                # Convert nautical direction (FROM) to arrow direction (TO)
                arrow_dir = (wave_dir + 180) % 360

                # Convert to radians (math convention: 0=E, CCW positive)
                theta = np.radians(90 - arrow_dir)

                # Arrow components proportional to Hs
                u = hs * arrow_scale * np.cos(theta)
                v = hs * arrow_scale * np.sin(theta)

                arrow_x.append(self.output.lons[j])
                arrow_y.append(self.output.lats[i])
                arrow_u.append(u)
                arrow_v.append(v)

        # Plot arrows
        if arrow_x:
            ax.quiver(
                arrow_x, arrow_y, arrow_u, arrow_v,
                color='#2980b9',  # Darker blue for arrows
                scale=1, scale_units='xy',
                width=0.004,
                headwidth=4, headlength=5,
                alpha=0.85,
                transform=ccrs.PlateCarree(),
                zorder=5
            )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Short-Period Wave Height (m)')

        # Title
        if title is None:
            valid_tp = agg_tp[~np.isnan(agg_tp)]
            avg_tp = np.mean(valid_tp) if len(valid_tp) > 0 else 0
            title = f"Short-Period Waves (Tp < {period_threshold:.0f}s, avg {avg_tp:.1f}s) - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / "wind_sea.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_boundary_components(
        self,
        arrow_scale: float = 0.2,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Optional[Path]:
        """
        Plot swell components at boundary points.

        Shows decomposed swell arrows (wind sea, swell 1, swell 2) at each
        boundary point, colored by period.

        Args:
            arrow_scale: Scale factor for arrow length
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.lines import Line2D

        # Load boundary spectral data
        boundary_points = self.boundary_reader.read_boundary_points()

        if not boundary_points:
            logger.info("No boundary spectral data available - skipping boundary components plot")
            logger.info("(To enable: cache WW3 partitions during SWAN run)")
            # Return None - will be filtered in plot_all
            return None

        hsig_masked, _, _ = self.output.mask_land()

        fig, ax = plt.subplots(
            figsize=(14, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        # Plot Hsig as background
        im = ax.pcolormesh(
            LON, LAT, hsig_masked,
            cmap='YlOrRd',
            vmin=0, vmax=5,
            shading='auto',
            alpha=0.6,
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_extent(self._get_padded_extent(), crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Collect arrow data grouped by period band
        arrows_by_period = {
            'wind_chop': {'x': [], 'y': [], 'u': [], 'v': []},
            'short_swell': {'x': [], 'y': [], 'u': [], 'v': []},
            'long_swell': {'x': [], 'y': [], 'u': [], 'v': []},
        }

        # Offset each arrow slightly to prevent overlap
        offset_scale = 0.02

        for point in boundary_points:
            top_components = point.top_components(self.max_swells)

            for idx, comp in enumerate(top_components):
                if comp.hs < 0.1:
                    continue

                # Convert nautical direction (FROM) to arrow direction (TO)
                arrow_dir = (comp.direction + 180) % 360
                theta = np.radians(90 - arrow_dir)

                # Arrow components proportional to Hs
                u = comp.hs * arrow_scale * np.cos(theta)
                v = comp.hs * arrow_scale * np.sin(theta)

                # Small perpendicular offset for multiple arrows
                perp_theta = theta + np.pi/2
                offset_x = idx * offset_scale * np.cos(perp_theta)
                offset_y = idx * offset_scale * np.sin(perp_theta)

                # Categorize by period band
                if comp.tp < 8:
                    band = 'wind_chop'
                elif comp.tp < 12:
                    band = 'short_swell'
                else:
                    band = 'long_swell'

                arrows_by_period[band]['x'].append(point.lon + offset_x)
                arrows_by_period[band]['y'].append(point.lat + offset_y)
                arrows_by_period[band]['u'].append(u)
                arrows_by_period[band]['v'].append(v)

        # Plot quiver for each period band
        period_colors = {
            'wind_chop': '#3498db',
            'short_swell': '#27ae60',
            'long_swell': '#e74c3c',
        }

        for band, data in arrows_by_period.items():
            if data['x']:
                ax.quiver(
                    data['x'], data['y'], data['u'], data['v'],
                    color=period_colors[band],
                    scale=1, scale_units='xy',
                    width=0.005,
                    headwidth=4, headlength=5,
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )

        # Mark boundary points
        for point in boundary_points:
            ax.plot(point.lon, point.lat, 'ko', markersize=3,
                   transform=ccrs.PlateCarree())

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Significant Wave Height (m)')

        # Legend for period colors
        legend_elements = [
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#3498db',
                   markersize=12, label='Wind Chop (<8s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#27ae60',
                   markersize=12, label='Short Swell (8-12s)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='#e74c3c',
                   markersize=12, label='Groundswell (>12s)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title
        if title is None:
            title = f"Boundary Swell Components - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / "boundary_components.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_all(self, show: bool = False) -> List[Path]:
        """
        Generate all spectral plots.

        Args:
            show: Display plots

        Returns:
            List of paths to saved plots
        """
        logger.info(f"Generating spectral plots for {self.output.region_name}/{self.output.mesh_name}")

        paths = [
            self.plot_swell_components(show=show),
            self.plot_partition_arrows(show=show),  # Multiple arrows per point from partitions
            self.plot_wind_sea(show=show),          # Wind sea only (Tp < 8s)
            self.plot_boundary_components(show=show),
        ]

        # Filter out empty paths (from skipped plots)
        paths = [p for p in paths if p and p.exists()]

        logger.info(f"Saved {len(paths)} plots to {self.plots_dir}")
        return paths


def plot_spectral_results(
    run_dir: str | Path,
    max_swells: int = MAX_SWELLS_TO_DISPLAY,
    show: bool = False,
) -> List[Path]:
    """
    Convenience function to plot spectral SWAN results.

    Args:
        run_dir: Path to run directory
        max_swells: Maximum number of swell components to display
        show: Display plots

    Returns:
        List of paths to saved plots
    """
    plotter = SpectralPlotter(run_dir, max_swells=max_swells)
    return plotter.plot_all(show=show)


# CLI
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Default run directory
    DEFAULT_REGION = "socal"
    DEFAULT_MESH = "coarse"
    RUNS_DIR = PROJECT_ROOT / "data" / "swan" / "runs"

    parser = argparse.ArgumentParser(
        description="Plot SWAN spectral results with period-colored swell arrows"
    )
    parser.add_argument("--region", "-r", default=DEFAULT_REGION,
                       help=f"Region name (default: {DEFAULT_REGION})")
    parser.add_argument("--mesh", "-m", default=DEFAULT_MESH,
                       help=f"Mesh name (default: {DEFAULT_MESH})")
    parser.add_argument("--max-swells", "-n", type=int, default=MAX_SWELLS_TO_DISPLAY,
                       help=f"Max swell components to display (default: {MAX_SWELLS_TO_DISPLAY})")
    parser.add_argument("--show", action="store_true", help="Display plots")

    args = parser.parse_args()

    # Build run directory path
    run_dir = RUNS_DIR / args.region / args.mesh / "latest"

    try:
        paths = plot_spectral_results(run_dir, max_swells=args.max_swells, show=args.show)
        print(f"\nGenerated {len(paths)} plots")
        for p in paths:
            print(f"  {p}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
