"""
SWAN domain configuration with WW3 boundary conditions.

Sets up the SWAN computational domain so that:
- Offshore boundary aligns with WW3 valid data extent
- WW3 wave spectra provide boundary conditions at this edge
- SWAN propagates waves from WW3 boundary toward coast

WW3 Coverage:
- WW3 global model has ~0.5° resolution (~50km)
- WW3 regional models may be finer
- Valid data typically extends to ~20-30km offshore (varies by location)
- Nearshore is masked out in WW3 (no valid data close to coast)

SWAN Boundary Condition Options:
1. Parametric: Hs, Tp, Dir, spreading -> simpler, less accurate
2. Spectral: Full 2D spectra from WW3 -> more accurate for swell

Usage:
    python -m data.pipelines.bathymetry.swan_domain --analyze california
    python -m data.pipelines.bathymetry.swan_domain --generate california --offshore-km 25
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.pipelines.bathymetry.config import (
    GEBCO_DIR,
    REGIONS,
)

GRIDS_DIR = Path(__file__).parent.parent.parent / "grids" / "swan_domains"


@dataclass
class SwanDomainConfig:
    """Configuration for a SWAN computational domain."""
    name: str
    region: str

    # Domain bounds
    lat_min: float
    lat_max: float
    lon_min: float  # Western boundary (offshore)
    lon_max: float  # Eastern boundary (coast)

    # Grid parameters
    resolution_m: float
    n_lat: int
    n_lon: int
    n_cells: int
    n_wet_cells: int

    # Boundary condition info
    offshore_boundary_km: float  # Distance from coast to offshore boundary
    ww3_boundary_description: str

    # File paths
    file_nc: str
    file_grd: str
    file_swan_input: str  # SWAN input file template

    created_at: str


class SwanDomainGenerator:
    """
    Generate SWAN domains with WW3 boundary conditions.

    The domain is set up so that:
    - Western (offshore) edge aligns with WW3 valid data extent
    - SWAN receives boundary conditions from WW3 at this edge
    - Domain extends from offshore boundary to near the coast
    """

    # Approximate WW3 offshore extent (km from coast where WW3 has valid data)
    # This varies by location - WW3 masks out shallow water
    DEFAULT_WW3_OFFSHORE_KM = 25.0

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or GRIDS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_ww3_coverage(self, region: str) -> dict:
        """
        Analyze WW3 data coverage for a region.

        Determines where WW3 has valid data (offshore extent).
        """
        print(f"\nAnalyzing WW3 coverage for: {region}")
        print("-" * 50)

        # Try to load WW3 data to find actual boundary
        try:
            from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher

            bounds = REGIONS[region]
            lat_min, lat_max, lon_min, lon_max = bounds

            fetcher = WaveWatchFetcher()
            # This would fetch WW3 data and analyze coverage
            # For now, return estimated values

            print("  WW3 fetcher available - would analyze actual coverage")
            print(f"  Estimated offshore extent: ~{self.DEFAULT_WW3_OFFSHORE_KM}km")

        except ImportError:
            print("  WW3 fetcher not available - using default estimates")

        # Return estimated coverage info
        return {
            "region": region,
            "estimated_offshore_km": self.DEFAULT_WW3_OFFSHORE_KM,
            "notes": "WW3 typically has valid data starting ~20-30km offshore",
            "boundary_type": "spectral",  # Recommended for accuracy
        }

    def generate(
        self,
        region: str,
        offshore_boundary_km: float = 25.0,
        nearshore_boundary_km: float = 0.5,
        resolution_m: float = 500.0,
        name: Optional[str] = None,
    ) -> SwanDomainConfig:
        """
        Generate SWAN domain with WW3-aligned boundaries.

        Args:
            region: Region name
            offshore_boundary_km: Distance from coast to offshore boundary (where WW3 provides BC)
            nearshore_boundary_km: Distance from coast to nearshore boundary (SWAN output limit)
            resolution_m: Grid resolution in meters
            name: Custom name for domain

        Returns:
            SwanDomainConfig with file paths and metadata
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required")

        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region}")

        bounds = REGIONS[region]
        lat_min, lat_max, lon_min, lon_max = bounds

        if name is None:
            name = f"{region}_swan_{int(resolution_m)}m"

        print("=" * 60)
        print(f"Generating SWAN Domain: {name}")
        print("=" * 60)
        print(f"  Region: {region}")
        print(f"  Offshore boundary: {offshore_boundary_km}km from coast (WW3 input)")
        print(f"  Nearshore boundary: {nearshore_boundary_km}km from coast (SWAN output)")
        print(f"  Resolution: {resolution_m}m")

        # Load GEBCO data
        gebco_path = GEBCO_DIR / f"gebco_2024_{region}.nc"
        if not gebco_path.exists():
            gebco_path = GEBCO_DIR / "gebco_2024_california.nc"

        print(f"\nLoading bathymetry: {gebco_path}")
        ds = xr.open_dataset(gebco_path)

        # Extract data
        elev_var = self._find_var(ds, ["elevation", "z", "topo"])
        lat_coord = self._find_coord(ds, ["lat", "latitude", "y"])
        lon_coord = self._find_coord(ds, ["lon", "longitude", "x"])

        elevation = ds[elev_var].values
        lats = ds[lat_coord].values
        lons = ds[lon_coord].values

        # Create grid at target resolution
        res_deg = resolution_m / 111000  # Approximate degrees

        out_lats = np.arange(lat_min, lat_max, res_deg)
        out_lons = np.arange(lon_min, lon_max, res_deg)

        print(f"  Grid dimensions: {len(out_lats)} × {len(out_lons)}")

        # Interpolate bathymetry to output grid
        from scipy.interpolate import RegularGridInterpolator

        # Ensure ascending order
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            elevation = elevation[::-1, :]
        if lons[0] > lons[-1]:
            lons = lons[::-1]
            elevation = elevation[:, ::-1]

        interp = RegularGridInterpolator(
            (lats, lons), elevation,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )

        lon_grid, lat_grid = np.meshgrid(out_lons, out_lats)
        points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
        out_elevation = interp(points).reshape(lat_grid.shape)

        # Calculate distance to coast for each point
        from scipy import ndimage
        land_mask = out_elevation >= 0
        distance_cells = ndimage.distance_transform_edt(~land_mask)
        distance_km = distance_cells * resolution_m / 1000

        # Create domain mask: only include cells within offshore_boundary_km of coast
        # and seaward of nearshore_boundary_km
        domain_mask = (distance_km <= offshore_boundary_km) & (distance_km >= nearshore_boundary_km) & (out_elevation < 0)

        # Apply mask - set out-of-domain cells to NaN
        masked_elevation = np.where(domain_mask, out_elevation, np.nan)

        n_wet = int(np.sum(domain_mask))
        n_total = len(out_lats) * len(out_lons)

        print(f"\n  Domain cells (in computational area): {n_wet:,}")
        print(f"  Total grid cells: {n_total:,}")
        print(f"  Domain efficiency: {100*n_wet/n_total:.1f}%")

        ds.close()

        # Create output directory
        domain_dir = self.output_dir / name
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Save NetCDF
        nc_path = domain_dir / f"{name}.nc"
        self._save_netcdf(
            out_elevation, masked_elevation, distance_km,
            out_lats, out_lons, nc_path, name,
            offshore_boundary_km, nearshore_boundary_km,
        )

        # Save SWAN grid file
        grd_path = domain_dir / f"{name}.grd"
        self._save_swan_grd(masked_elevation, out_lats, out_lons, grd_path)

        # Generate SWAN input file template
        swan_input_path = domain_dir / f"{name}.swn"
        self._generate_swan_input(
            name, out_lats, out_lons, resolution_m,
            offshore_boundary_km, grd_path, swan_input_path,
        )

        # Create config
        config = SwanDomainConfig(
            name=name,
            region=region,
            lat_min=float(out_lats.min()),
            lat_max=float(out_lats.max()),
            lon_min=float(out_lons.min()),
            lon_max=float(out_lons.max()),
            resolution_m=resolution_m,
            n_lat=len(out_lats),
            n_lon=len(out_lons),
            n_cells=n_total,
            n_wet_cells=n_wet,
            offshore_boundary_km=offshore_boundary_km,
            ww3_boundary_description=f"WW3 spectral boundary at {offshore_boundary_km}km offshore",
            file_nc=str(nc_path),
            file_grd=str(grd_path),
            file_swan_input=str(swan_input_path),
            created_at=datetime.now().isoformat(),
        )

        # Save config
        config_path = domain_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

        print(f"\nOutput files:")
        print(f"  NetCDF:     {nc_path}")
        print(f"  SWAN grid:  {grd_path}")
        print(f"  SWAN input: {swan_input_path}")

        return config

    def _find_var(self, ds, candidates):
        for name in candidates:
            if name in ds.data_vars:
                return name
        raise ValueError(f"Variable not found: {candidates}")

    def _find_coord(self, ds, candidates):
        for name in candidates:
            if name in ds.coords:
                return name
        raise ValueError(f"Coordinate not found: {candidates}")

    def _save_netcdf(
        self,
        full_elevation: np.ndarray,
        masked_elevation: np.ndarray,
        distance_km: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        path: Path,
        name: str,
        offshore_km: float,
        nearshore_km: float,
    ):
        """Save domain as NetCDF with full metadata."""
        ds = xr.Dataset(
            {
                "elevation": (["lat", "lon"], full_elevation, {
                    "units": "m",
                    "long_name": "Full bathymetry elevation",
                }),
                "elevation_masked": (["lat", "lon"], masked_elevation, {
                    "units": "m",
                    "long_name": "Bathymetry within SWAN domain",
                }),
                "distance_to_coast_km": (["lat", "lon"], distance_km, {
                    "units": "km",
                    "long_name": "Distance to nearest coastline",
                }),
                "domain_mask": (["lat", "lon"], ~np.isnan(masked_elevation), {
                    "long_name": "SWAN computational domain mask",
                }),
            },
            coords={
                "lat": ("lat", lats, {"units": "degrees_north"}),
                "lon": ("lon", lons, {"units": "degrees_east"}),
            },
            attrs={
                "title": f"SWAN domain: {name}",
                "offshore_boundary_km": offshore_km,
                "nearshore_boundary_km": nearshore_km,
                "boundary_condition_source": "WW3",
                "created": datetime.now().isoformat(),
            },
        )
        ds.to_netcdf(path)
        print(f"  Saved NetCDF: {path}")

    def _save_swan_grd(
        self,
        elevation: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        path: Path,
    ):
        """
        Save SWAN bathymetry grid in FREE format (no headers).

        SWAN FREE format expects only numbers.
        Depth values: positive = water depth, -999 = land/outside domain.
        """
        # Convert to SWAN depth (positive down, exception for land/out-of-domain)
        depth = np.where(
            np.isnan(elevation) | (elevation >= 0),
            -999.0,  # Exception value for land/outside domain
            -elevation,  # Positive depth (negate negative elevation)
        )

        with open(path, "w") as f:
            # No header - SWAN FREE format expects only numbers
            for row in depth:
                f.write(" ".join(f"{v:.2f}" for v in row) + "\n")

        print(f"  Saved SWAN grid: {path}")

    def _generate_swan_input(
        self,
        name: str,
        lats: np.ndarray,
        lons: np.ndarray,
        resolution_m: float,
        offshore_km: float,
        grd_path: Path,
        output_path: Path,
    ):
        """
        Generate SWAN input file template with WW3 boundary conditions.

        This creates a template that shows how to:
        1. Define the computational grid
        2. Read bathymetry
        3. Apply WW3 boundary conditions at offshore edge
        """

        # Calculate grid parameters
        n_lat = len(lats)
        n_lon = len(lons)
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()

        # For SPHERICAL coordinates, xlenc/ylenc are in DEGREES
        xlenc = lon_max - lon_min  # Domain width in degrees
        ylenc = lat_max - lat_min  # Domain height in degrees

        # Number of cells (not points)
        mxc = n_lon - 1
        myc = n_lat - 1

        # Grid spacing in degrees
        dx_deg = xlenc / mxc
        dy_deg = ylenc / myc

        # Shorten project name for SWAN (max ~16 chars)
        short_name = name[:16] if len(name) > 16 else name

        swan_input = f"""$ SWAN input file for {name}
$ Generated: {datetime.now().isoformat()}
$
$ This template shows how to set up SWAN with WW3 boundary conditions
$
$----------------------------------------------------------
$ PROJECT INFO
$----------------------------------------------------------
PROJECT '{short_name}' 'run01'

$----------------------------------------------------------
$ COMPUTATIONAL GRID
$----------------------------------------------------------
$ Regular grid in spherical coordinates
$ Origin at SW corner, grid aligned with lat/lon

SET LEVEL 0.0
SET NAUTICAL
SET DEPMIN 0.05
SET MAXERR 3

MODE NONSTATIONARY TWODIMENSIONAL

COORDINATES SPHERICAL

$ Computational grid definition
$ CGRID REGULAR: xpc ypc alpc xlenc ylenc mxc myc
$   xpc, ypc = origin (lon, lat)
$   alpc = rotation angle (0 = aligned with lat/lon)
$   xlenc, ylenc = domain size in degrees
$   mxc, myc = number of cells in x, y direction
CGRID REGULAR {lon_min:.4f} {lat_min:.4f} 0.0 {xlenc:.6f} {ylenc:.6f} {mxc} {myc} &
      CIRCLE 36 0.04 1.0

$----------------------------------------------------------
$ BATHYMETRY
$----------------------------------------------------------
$ Read bathymetry from SWAN grid file
$ Exception value -999 marks land/outside domain

INPGRID BOTTOM REGULAR {lon_min:.4f} {lat_min:.4f} 0.0 {mxc} {myc} {dx_deg:.6f} {dy_deg:.6f} &
        EXCEPTION -999.0

READINP BOTTOM 1.0 '{grd_path.name}' 1 0 FREE

$----------------------------------------------------------
$ BOUNDARY CONDITIONS FROM WW3 (SPECTRAL FORMAT)
$----------------------------------------------------------
$ The offshore (western) boundary receives 2D wave spectra from WW3
$ This preserves the full spectral shape including multiple swell systems
$
$ SPEC format advantages over TPAR:
$ - Preserves full 2D energy density spectrum E(f,theta)
$ - Multiple swell partitions maintained through propagation
$ - More accurate nearshore wave transformation
$
$ Boundary specification is generated at runtime by run_swan.py
$ using spectral files (.sp2) at each boundary point
$
$ Placeholder - replaced at runtime:
{{{{BOUNDSPEC_COMMANDS}}}}

$----------------------------------------------------------
$ PHYSICS
$----------------------------------------------------------
$ Wave breaking
BREAKING CONSTANT 1.0 0.73

$ Bottom friction (JONSWAP coefficient)
FRICTION JONSWAP 0.067

$ Triad wave-wave interactions (shallow water)
TRIAD

$ Whitecapping (Komen formulation)
WCAPPING KOMEN

$----------------------------------------------------------
$ NUMERICS
$----------------------------------------------------------
NUMERIC ACCUR 0.02 0.02 0.02 95 NONSTAT 10

$----------------------------------------------------------
$ OUTPUT
$----------------------------------------------------------
$
$ Combined wave parameters (full grid)
$
BLOCK 'COMPGRID' NOHEAD 'hsig.mat' LAY 3 HSIG 1.
BLOCK 'COMPGRID' NOHEAD 'tm01.mat' LAY 3 TM01 1.
BLOCK 'COMPGRID' NOHEAD 'tpeak.mat' LAY 3 RTP 1.
BLOCK 'COMPGRID' NOHEAD 'dir.mat' LAY 3 DIR 1.
BLOCK 'COMPGRID' NOHEAD 'depth.mat' LAY 3 DEPTH 1.

$----------------------------------------------------------
$ COMPUTATION
$----------------------------------------------------------
$ Nonstationary run through all forecast hours
$ Time stepping is set at runtime by run_swan.py

TEST 1 0
COMPUTE NONSTAT {{{{START_TIME}}}} 3.0 HR {{{{END_TIME}}}}

STOP
"""

        with open(output_path, "w") as f:
            f.write(swan_input)

        print(f"  Saved SWAN input template: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SWAN domains with WW3 boundary conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze WW3 coverage for region
    python -m data.pipelines.bathymetry.swan_domain --analyze california

    # Generate SWAN domain (default 25km offshore boundary)
    python -m data.pipelines.bathymetry.swan_domain --generate california

    # Custom offshore boundary (where WW3 data is valid)
    python -m data.pipelines.bathymetry.swan_domain --generate california --offshore-km 20

    # Higher resolution
    python -m data.pipelines.bathymetry.swan_domain --generate la_area --resolution 100

WW3 -> SWAN Workflow:
    1. WW3 provides wave spectra at ~20-30km offshore
    2. SWAN domain starts at this WW3 boundary
    3. WW3 spectra are applied as SWAN boundary conditions
    4. SWAN propagates waves from boundary toward coast
    5. SWAN outputs nearshore wave parameters for surf forecasting
""",
    )

    parser.add_argument("--analyze", type=str, metavar="REGION",
                       help="Analyze WW3 coverage for region")
    parser.add_argument("--generate", "-g", type=str, metavar="REGION",
                       help="Generate SWAN domain")
    parser.add_argument("--offshore-km", type=float, default=25.0,
                       help="Offshore boundary distance in km (default: 25)")
    parser.add_argument("--nearshore-km", type=float, default=0.5,
                       help="Nearshore boundary distance in km (default: 0.5)")
    parser.add_argument("--resolution", "-r", type=float, default=500.0,
                       help="Grid resolution in meters (default: 500)")
    parser.add_argument("--name", type=str, help="Custom domain name")

    args = parser.parse_args()

    generator = SwanDomainGenerator()

    if args.analyze:
        generator.analyze_ww3_coverage(args.analyze)
        return

    if args.generate:
        generator.generate(
            region=args.generate,
            offshore_boundary_km=args.offshore_km,
            nearshore_boundary_km=args.nearshore_km,
            resolution_m=args.resolution,
            name=args.name,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
