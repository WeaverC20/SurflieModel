#!/usr/bin/env python3
"""
SWAN Results Plotter

Creates visualizations of SWAN model outputs.
Saves plots to data/swan/analysis/plots/{region}/{mesh}/
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.analysis.output_reader import SwanOutput, SwanOutputReader

logger = logging.getLogger(__name__)

# Base directory for saving plots
ANALYSIS_DIR = Path(__file__).parent
PLOTS_DIR = ANALYSIS_DIR / "plots"


@dataclass
class SwanPlotter:
    """
    Creates plots from SWAN model output.

    Plots are saved to: data/swan/analysis/plots/{region}/{mesh}/{timestamp}/

    Example usage:
        plotter = SwanPlotter("data/swan/runs/socal/coarse/latest")
        plotter.plot_all()
    """

    run_dir: str | Path
    output: Optional[SwanOutput] = None
    plots_dir: Optional[Path] = None

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)

        # Load output if not provided
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

    def plot_hsig(
        self,
        vmin: float = 0,
        vmax: float = 5,
        cmap: str = "YlOrRd",
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Plot significant wave height.

        Args:
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            cmap: Colormap name
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        hsig_masked, _, _ = self.output.mask_land()

        fig, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        # Plot data
        im = ax.pcolormesh(
            LON, LAT, hsig_masked,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        ax.set_extent(self.output.extent, crs=ccrs.PlateCarree())

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Significant Wave Height (m)')

        # Title
        if title is None:
            title = f"Significant Wave Height - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        # Save
        filepath = self.plots_dir / "hsig.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_tps(
        self,
        vmin: float = 0,
        vmax: float = 20,
        cmap: str = "viridis",
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Plot peak wave period.

        Args:
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            cmap: Colormap name
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        _, tps_masked, _ = self.output.mask_land()

        fig, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        im = ax.pcolormesh(
            LON, LAT, tps_masked,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        ax.set_extent(self.output.extent, crs=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Peak Wave Period (s)')

        if title is None:
            title = f"Peak Wave Period - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        filepath = self.plots_dir / "tps.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath

    def plot_dir(
        self,
        cmap: str = "twilight",
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Plot mean wave direction.

        Args:
            cmap: Colormap name (circular colormap recommended)
            title: Plot title (auto-generated if None)
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        _, _, dir_masked = self.output.mask_land()

        fig, ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        im = ax.pcolormesh(
            LON, LAT, dir_masked,
            cmap=cmap,
            vmin=0, vmax=360,
            shading='auto',
            transform=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        ax.set_extent(self.output.extent, crs=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
        cbar.set_label('Mean Wave Direction (degrees)')
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])

        if title is None:
            title = f"Mean Wave Direction - {self.output.region_name}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        filepath = self.plots_dir / "dir.png"
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
        Generate all standard plots.

        Args:
            show: Display plots

        Returns:
            List of paths to saved plots
        """
        logger.info(f"Generating plots for {self.output.region_name}/{self.output.mesh_name}")

        paths = [
            self.plot_hsig(show=show),
            self.plot_tps(show=show),
            self.plot_dir(show=show),
        ]

        logger.info(f"Saved {len(paths)} plots to {self.plots_dir}")
        return paths

    def plot_combined(
        self,
        save: bool = True,
        show: bool = False,
    ) -> Path:
        """
        Create a combined 3-panel plot with Hsig, Tps, and Dir.

        Args:
            save: Save plot to file
            show: Display plot

        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        hsig_masked, tps_masked, dir_masked = self.output.mask_land()
        LON, LAT = np.meshgrid(self.output.lons, self.output.lats)

        fig, axes = plt.subplots(
            1, 3,
            figsize=(18, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Common settings
        extent = self.output.extent

        # Hsig
        ax = axes[0]
        im = ax.pcolormesh(LON, LAT, hsig_masked, cmap='YlOrRd', vmin=0, vmax=5, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.set_extent(extent)
        ax.set_title('Significant Wave Height (m)')
        plt.colorbar(im, ax=ax, shrink=0.6)

        # Tps
        ax = axes[1]
        im = ax.pcolormesh(LON, LAT, tps_masked, cmap='viridis', vmin=0, vmax=20, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.set_extent(extent)
        ax.set_title('Peak Wave Period (s)')
        plt.colorbar(im, ax=ax, shrink=0.6)

        # Dir
        ax = axes[2]
        im = ax.pcolormesh(LON, LAT, dir_masked, cmap='twilight', vmin=0, vmax=360, shading='auto')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.set_extent(extent)
        ax.set_title('Mean Wave Direction (deg)')
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_ticks([0, 90, 180, 270, 360])
        cbar.set_ticklabels(['N', 'E', 'S', 'W', 'N'])

        plt.suptitle(f"SWAN Results - {self.output.region_name}", fontsize=14)
        plt.tight_layout()

        filepath = self.plots_dir / "combined.png"
        if save:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")

        if show:
            plt.show()
        else:
            plt.close()

        return filepath


def plot_swan_results(
    run_dir: str | Path,
    show: bool = False,
    combined: bool = True,
) -> List[Path]:
    """
    Convenience function to plot SWAN results.

    Args:
        run_dir: Path to run directory
        show: Display plots
        combined: Also create combined plot

    Returns:
        List of paths to saved plots
    """
    plotter = SwanPlotter(run_dir)
    paths = plotter.plot_all(show=show)

    if combined:
        paths.append(plotter.plot_combined(show=show))

    return paths


# CLI
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Default run directory
    DEFAULT_REGION = "socal"
    DEFAULT_MESH = "coarse"
    RUNS_DIR = PROJECT_ROOT / "data" / "swan" / "runs"

    parser = argparse.ArgumentParser(description="Plot SWAN results")
    parser.add_argument("--region", "-r", default=DEFAULT_REGION,
                       help=f"Region name (default: {DEFAULT_REGION})")
    parser.add_argument("--mesh", "-m", default=DEFAULT_MESH,
                       help=f"Mesh name (default: {DEFAULT_MESH})")
    parser.add_argument("--show", action="store_true", help="Display plots")

    args = parser.parse_args()

    # Build run directory path
    run_dir = RUNS_DIR / args.region / args.mesh / "latest"

    try:
        paths = plot_swan_results(run_dir, show=args.show)
        print(f"\nGenerated {len(paths)} plots")
        for p in paths:
            print(f"  {p}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)