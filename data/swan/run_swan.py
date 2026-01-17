#!/usr/bin/env python3
"""
SWAN Run Orchestrator

Executes SWAN wave model simulations for specified regions and meshes.
Handles the complete workflow:
1. Fetch WW3 boundary conditions (from unified boundary config)
2. Generate SWAN INPUT file with spectral boundaries
3. Copy required files to run directory
4. Execute SWAN
5. Report results

Usage:
    python data/swan/run_swan.py --region socal --mesh coarse
    python data/swan/run_swan.py --region socal --mesh coarse --forecast-hour 24
    python data/swan/run_swan.py --region socal --mesh coarse --dry-run
"""

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.runner import (
    WW3BoundaryFetcher,
    BoundaryPoint,
    SwanInputGenerator,
    PhysicsSettings,
    SpectrumReconstructor,
    SpectralBoundaryWriter,
    WindProvider,
    WindData,
)

logger = logging.getLogger(__name__)

# Directory structure
DATA_DIR = PROJECT_ROOT / "data"
MESHES_DIR = DATA_DIR / "meshes"
WW3_ENDPOINTS_DIR = DATA_DIR / "swan" / "ww3_endpoints"
RUNS_DIR = DATA_DIR / "swan" / "runs"


def get_mesh_dir(region: str, mesh: str) -> Path:
    """Get path to mesh directory."""
    mesh_dir = MESHES_DIR / region / mesh
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    return mesh_dir


def get_boundary_file(region: str) -> Path:
    """Get path to unified WW3 boundary file."""
    boundary_file = WW3_ENDPOINTS_DIR / region / "ww3_boundaries.json"
    if not boundary_file.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {boundary_file}\n"
            f"Run: python data/swan/ww3_endpoints/{region}/extract.py --unified"
        )
    return boundary_file


def get_run_dir(region: str, mesh: str) -> Path:
    """Get path to run directory (latest)."""
    return RUNS_DIR / region / mesh / "latest"


class SwanRunner:
    """
    Orchestrates SWAN simulations.

    Handles the complete workflow from fetching boundary conditions
    to executing SWAN and collecting results.

    Uses unified boundary config with spectral (SPEC2D) boundary conditions.
    """

    def __init__(
        self,
        region: str,
        mesh: str,
        forecast_hours: Optional[List[int]] = None,
        physics: Optional[PhysicsSettings] = None,
    ):
        """
        Initialize SWAN runner.

        Args:
            region: Region name (e.g., "socal")
            mesh: Mesh name (e.g., "coarse")
            forecast_hours: Forecast hours to fetch (default: [0] for stationary)
            physics: Physics settings (uses defaults if None)
        """
        self.region = region
        self.mesh = mesh
        self.forecast_hours = forecast_hours or [0]  # Single hour for stationary
        self.physics = physics or PhysicsSettings()

        # Resolve paths
        self.mesh_dir = get_mesh_dir(region, mesh)
        self.run_dir = get_run_dir(region, mesh)
        self.boundary_file = get_boundary_file(region)

        # Find mesh files
        self.mesh_name = self._find_mesh_name()
        self.bot_file = self.mesh_dir / f"{self.mesh_name}.bot"

        if not self.bot_file.exists():
            raise FileNotFoundError(f"Bathymetry file not found: {self.bot_file}")

        logger.info(f"SwanRunner initialized for {region}/{mesh}")
        logger.info(f"Using boundary config: {self.boundary_file.name}")

    def _find_mesh_name(self) -> str:
        """Find the mesh name from directory contents."""
        json_files = [f for f in self.mesh_dir.glob("*.json") if not f.name.startswith("ww3_")]
        if not json_files:
            raise FileNotFoundError(f"No mesh metadata found in {self.mesh_dir}")
        return json_files[0].stem

    def _load_mesh_metadata(self) -> dict:
        """Load mesh metadata JSON."""
        mesh_json = self.mesh_dir / f"{self.mesh_name}.json"
        with open(mesh_json) as f:
            return json.load(f)

    async def fetch_boundary_conditions(self) -> tuple:
        """
        Fetch WW3 boundary conditions for all active boundaries.

        Returns:
            Tuple of (boundary_config dict, dict mapping side to List[BoundaryPoint])
        """
        logger.info(f"Fetching WW3 boundary conditions for hours: {self.forecast_hours}")

        fetcher = WW3BoundaryFetcher()
        config, boundary_points = await fetcher.fetch_boundary(
            self.boundary_file,
            forecast_hours=self.forecast_hours
        )

        # Log summary for each boundary
        for side, points in boundary_points.items():
            if points and points[0].hs:
                avg_hs = sum(p.hs[0] for p in points) / len(points)
                logger.info(f"  {side} boundary: avg Hs={avg_hs:.2f}m ({len(points)} points)")

        return config, boundary_points

    def prepare_wind_data(self) -> WindData:
        """
        Prepare wind data from downloaded GFS files.

        Extracts wind for the mesh region and interpolates to mesh resolution.

        Returns:
            WindData object with interpolated wind
        """
        logger.info("Preparing wind data from GFS")

        mesh_metadata = self._load_mesh_metadata()

        # Extract and interpolate wind
        wind_provider = WindProvider()
        wind_data = wind_provider.extract_for_mesh(
            mesh_metadata,
            forecast_hour=self.forecast_hours[0]
        )

        # Write wind files to run directory
        wind_provider.write_swan_files(wind_data, self.run_dir)

        # Log summary
        logger.info(wind_provider.summary(wind_data))

        return wind_data

    def generate_spectral_boundaries(
        self,
        boundary_points: Dict[str, List[BoundaryPoint]],
        mesh_metadata: dict
    ) -> Dict[str, str]:
        """
        Generate SPEC2D spectral boundary files for all boundaries.

        Args:
            boundary_points: Dict mapping side name to List[BoundaryPoint]
            mesh_metadata: Mesh metadata containing spectral configuration

        Returns:
            Dict mapping side name to spectral filename (just filename, not path)
        """
        logger.info(f"Generating spectral boundary files for {len(boundary_points)} boundaries")

        # Create spectrum reconstructor with mesh spectral config
        reconstructor = SpectrumReconstructor.from_mesh_metadata(mesh_metadata)

        # Create spectral boundary writer
        writer = SpectralBoundaryWriter(reconstructor)

        # Get timestamp from first boundary's first point
        first_side = list(boundary_points.keys())[0]
        first_points = boundary_points[first_side]
        timestamp = first_points[0].times[0] if first_points[0].times else None

        # Write SPEC2D files for each boundary
        spectral_files = writer.write_multi_boundary_spec2d(
            output_dir=self.run_dir,
            boundary_points=boundary_points,
            time_index=0,
            timestamp=timestamp,
        )

        # Return just filenames (not full paths) for SWAN INPUT
        return {side: path.name for side, path in spectral_files.items()}

    def generate_input_file(
        self,
        boundary_config: dict,
        boundary_points: Dict[str, List[BoundaryPoint]],
        wind_data: Optional[WindData] = None,
    ) -> Path:
        """
        Generate SWAN INPUT file with spectral boundary conditions.

        Args:
            boundary_config: Unified boundary config dict
            boundary_points: Dict mapping side name to List[BoundaryPoint]
            wind_data: Optional WindData object for wind forcing

        Returns:
            Path to generated INPUT file
        """
        active_boundaries = boundary_config["active_boundaries"]
        logger.info(f"Generating SWAN INPUT file (boundaries: {active_boundaries})")

        mesh_metadata = self._load_mesh_metadata()

        # Generate spectral boundary files for all boundaries
        spectral_files = self.generate_spectral_boundaries(
            boundary_points, mesh_metadata
        )

        # Create generator
        generator = SwanInputGenerator(
            mesh_dir=str(self.mesh_dir),
            boundary_file=str(self.boundary_file),
            physics=self.physics
        )

        # Generate wind commands if wind data provided
        wind_commands = None
        if wind_data is not None:
            wind_provider = WindProvider()
            wind_commands = [
                wind_provider.generate_inpgrid_command(wind_data),
                wind_provider.generate_readinp_command()
            ]

        # Generate INPUT file
        input_path = generator.generate(
            output_dir=self.run_dir,
            boundary_config=boundary_config,
            spectral_files=spectral_files,
            wind_commands=wind_commands,
            project_name=f"{self.region}_{self.mesh}",
            run_id="001"
        )

        return input_path

    def copy_bathymetry(self) -> Path:
        """
        Copy bathymetry file to run directory.

        Returns:
            Path to copied file
        """
        dest_path = self.run_dir / self.bot_file.name
        shutil.copy2(self.bot_file, dest_path)
        logger.info(f"Copied bathymetry: {dest_path.name}")
        return dest_path

    def execute_swan(self) -> subprocess.CompletedProcess:
        """
        Execute SWAN in the run directory.

        Returns:
            CompletedProcess result
        """
        logger.info(f"Executing SWAN in {self.run_dir}")

        # Run SWAN (assumes 'swan' or 'swanrun' is in PATH)
        result = subprocess.run(
            ["swan"],
            cwd=self.run_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("SWAN completed successfully")
        else:
            logger.error(f"SWAN failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")

        return result

    async def run(self, dry_run: bool = False) -> bool:
        """
        Execute complete SWAN workflow.

        Args:
            dry_run: If True, prepare files but don't execute SWAN

        Returns:
            True if successful, False otherwise
        """
        start_time = datetime.now()
        logger.info(f"Starting SWAN run for {self.region}/{self.mesh}")
        logger.info(f"Run directory: {self.run_dir}")

        # Clean and create run directory
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Fetch boundary conditions
            boundary_config, boundary_points = await self.fetch_boundary_conditions()

            # Step 2: Prepare wind data
            wind_data = self.prepare_wind_data()

            # Step 3: Generate INPUT file (with wind and spectral boundaries)
            input_path = self.generate_input_file(
                boundary_config, boundary_points, wind_data
            )

            # Step 4: Copy bathymetry
            self.copy_bathymetry()

            # Step 5: Execute SWAN (unless dry run)
            if dry_run:
                logger.info("Dry run - skipping SWAN execution")
                logger.info(f"Run directory ready: {self.run_dir}")
                self._print_run_contents()
                return True

            result = self.execute_swan()

            # Step 6: Check results
            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(f"SWAN run completed in {elapsed:.1f}s")
                self._print_output_files()
                return True
            else:
                logger.error(f"SWAN run failed after {elapsed:.1f}s")
                return False

        except Exception as e:
            logger.error(f"Run failed: {e}")
            raise

    def _print_run_contents(self) -> None:
        """Print contents of run directory."""
        print("\nRun directory contents:")
        for f in sorted(self.run_dir.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")

    def _print_output_files(self) -> None:
        """Print output files from SWAN run."""
        output_files = ["hsig.mat", "tps.mat", "dir.mat", "PRINT", "Errfile"]
        # Also check for boundary spectral files
        for f in self.run_dir.glob("boundary_*.sp2"):
            output_files.append(f.name)
        print("\nOutput files:")
        for name in output_files:
            path = self.run_dir / name
            if path.exists():
                size = path.stat().st_size
                print(f"  {name} ({size:,} bytes)")


async def run_swan(
    region: str,
    mesh: str,
    forecast_hours: Optional[List[int]] = None,
    dry_run: bool = False,
) -> bool:
    """
    Convenience function to run SWAN.

    Args:
        region: Region name
        mesh: Mesh name
        forecast_hours: Forecast hours to fetch
        dry_run: If True, prepare but don't execute

    Returns:
        True if successful
    """
    runner = SwanRunner(
        region=region,
        mesh=mesh,
        forecast_hours=forecast_hours,
    )
    return await runner.run(dry_run=dry_run)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SWAN wave model simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/swan/run_swan.py --region socal --mesh coarse
  python data/swan/run_swan.py --region socal --mesh coarse --forecast-hour 24
  python data/swan/run_swan.py --region socal --mesh coarse --dry-run
        """
    )

    parser.add_argument(
        "--region", "-r",
        required=True,
        help="Region name (e.g., socal, norcal, central)"
    )
    parser.add_argument(
        "--mesh", "-m",
        required=True,
        help="Mesh name (e.g., coarse, fine)"
    )
    parser.add_argument(
        "--forecast-hour", "-f",
        type=int,
        default=0,
        help="Forecast hour to run (default: 0 = current)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Prepare files but don't execute SWAN"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Run
    try:
        success = asyncio.run(run_swan(
            region=args.region,
            mesh=args.mesh,
            forecast_hours=[args.forecast_hour],
            dry_run=args.dry_run,
        ))

        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
