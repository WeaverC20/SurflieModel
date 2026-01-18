"""
SWAN Input File Generator

Generates SWAN INPUT files for stationary wave model runs.
Combines mesh bathymetry, WW3 boundary conditions, and physics settings.
Uses spectral (SPEC2D) boundary conditions with unified multi-boundary config.
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ww3_boundary_fetcher import WavePartition, BoundaryPoint

logger = logging.getLogger(__name__)


@dataclass
class PhysicsSettings:
    """
    Physics settings for SWAN simulation.

    Defaults are for wind-driven nearshore wave modeling:
    - GEN3 WESTH for wind-wave generation (van der Westhuysen formulation)
    - Bottom friction enabled (JONSWAP)
    - Depth-limited breaking enabled
    - Triads disabled (for swell propagation)

    The WESTH formulation is preferred over KOMEN for nearshore applications
    as it better handles mixed sea states (swell + wind sea).
    """
    # Bottom friction
    friction: str = "JONSWAP"
    friction_coefficient: float = 0.067

    # Wave breaking
    breaking: bool = True
    breaking_alpha: float = 1.0
    breaking_gamma: float = 0.73

    # GEN3 formulation: "WESTH" (recommended) or "KOMEN" (classic)
    gen3_formulation: str = "WESTH"

    # Triads (shallow water nonlinear interactions)
    triads: bool = False

    def to_swan_commands(self) -> List[str]:
        """Generate SWAN physics command strings."""
        commands = []

        # GEN3 enables wind-wave generation, whitecapping, and quadruplets
        # WESTH = van der Westhuysen formulation (better for mixed sea states)
        # KOMEN = classic Komen formulation
        commands.append(f"GEN3 {self.gen3_formulation.upper()}")

        # Friction
        if self.friction.upper() == "JONSWAP":
            commands.append(f"FRICTION JONSWAP {self.friction_coefficient}")
        elif self.friction.upper() == "COLLINS":
            commands.append(f"FRICTION COLLINS {self.friction_coefficient}")
        elif self.friction.upper() == "MADSEN":
            commands.append(f"FRICTION MADSEN {self.friction_coefficient}")

        # Breaking
        if self.breaking:
            commands.append(f"BREAKING CONSTANT {self.breaking_alpha} {self.breaking_gamma}")

        # Triads
        if self.triads:
            commands.append("TRIAD")

        return commands


@dataclass
class SpectralConfig:
    """Configuration for spectral discretization matching SWAN CGRID settings."""
    n_freq: int = 31          # Number of frequency bins
    freq_min: float = 0.04    # Minimum frequency (Hz)
    freq_max: float = 1.0     # Maximum frequency (Hz)
    n_dir: int = 36           # Number of direction bins

    def __post_init__(self):
        # Generate frequency and direction arrays
        # Frequencies are logarithmically spaced (SWAN default)
        self.frequencies = np.geomspace(self.freq_min, self.freq_max, self.n_freq)
        # Directions evenly spaced 0-360 (nautical convention)
        self.directions = np.linspace(0, 360 - 360/self.n_dir, self.n_dir)
        # Direction bin width
        self.d_theta = 360.0 / self.n_dir  # degrees


class SpectrumReconstructor:
    """
    Reconstructs 2D wave spectra from WW3 partition parameters.

    Uses JONSWAP frequency spectrum with cos^2s directional spreading.
    Combines multiple partitions (wind sea + swell) by superposition.

    The spectrum is E(f, θ) in units of m²/Hz/deg for SWAN input.
    """

    # JONSWAP default parameters
    GAMMA_WIND_SEA = 3.3    # Peak enhancement for wind sea
    GAMMA_SWELL = 1.5       # Lower enhancement for swell (narrower peak)

    def __init__(self, config: Optional[SpectralConfig] = None):
        """
        Initialize spectrum reconstructor.

        Args:
            config: Spectral discretization config (uses defaults if None)
        """
        self.config = config or SpectralConfig()

    @classmethod
    def from_mesh_metadata(cls, mesh_metadata: Dict) -> "SpectrumReconstructor":
        """Create reconstructor with spectral config from mesh metadata."""
        spectral = mesh_metadata.get("spectral", {})
        config = SpectralConfig(
            n_freq=spectral.get("n_freq", 31),
            freq_min=spectral.get("freq_min", 0.04),
            freq_max=spectral.get("freq_max", 1.0),
            n_dir=spectral.get("n_dir", 36)
        )
        return cls(config)

    def jonswap_spectrum(
        self,
        frequencies: np.ndarray,
        hs: float,
        tp: float,
        gamma: float = 3.3
    ) -> np.ndarray:
        """
        Calculate JONSWAP frequency spectrum.

        Args:
            frequencies: Array of frequencies (Hz)
            hs: Significant wave height (m)
            tp: Peak period (s)
            gamma: Peak enhancement factor

        Returns:
            1D array of spectral density S(f) in m²/Hz
        """
        if hs <= 0 or tp <= 0 or np.isnan(hs) or np.isnan(tp):
            return np.zeros_like(frequencies)

        fp = 1.0 / tp  # Peak frequency

        # JONSWAP parameters
        alpha = 0.0624 / (0.230 + 0.0336 * gamma - 0.185 / (1.9 + gamma))
        alpha = alpha * (hs ** 2) * (fp ** 4)

        # Spectral width parameter
        sigma = np.where(frequencies <= fp, 0.07, 0.09)

        # JONSWAP spectrum
        # S(f) = alpha * g^2 * (2*pi)^-4 * f^-5 * exp(-5/4 * (fp/f)^4) * gamma^r
        # where r = exp(-(f - fp)^2 / (2 * sigma^2 * fp^2))

        g = 9.81
        with np.errstate(divide='ignore', invalid='ignore'):
            pm_spectrum = (alpha * g**2 / (2 * np.pi)**4 *
                          frequencies**(-5) *
                          np.exp(-1.25 * (fp / frequencies)**4))

            # Peak enhancement
            r = np.exp(-((frequencies - fp)**2) / (2 * sigma**2 * fp**2))
            spectrum = pm_spectrum * gamma**r

            # Handle edge cases
            spectrum = np.where(np.isfinite(spectrum), spectrum, 0)
            spectrum = np.maximum(spectrum, 0)

        return spectrum

    def cos2s_spreading(
        self,
        directions: np.ndarray,
        mean_dir: float,
        spread: float
    ) -> np.ndarray:
        """
        Calculate cos^2s directional spreading function.

        Args:
            directions: Array of directions (degrees, nautical)
            mean_dir: Mean wave direction (degrees, direction FROM)
            spread: Directional spread (degrees, ~1 std dev)

        Returns:
            1D array of directional distribution D(θ), normalized to sum to 1
        """
        if np.isnan(mean_dir) or np.isnan(spread):
            # Return uniform distribution
            return np.ones_like(directions) / len(directions)

        # Convert spread to 's' parameter for cos^2s distribution
        # Approximate relationship: spread ≈ sqrt(2/s) * 180/π
        # So s ≈ 2 / (spread * π/180)^2
        spread_rad = np.radians(max(spread, 5.0))  # Minimum 5 degrees
        s = 2.0 / (spread_rad ** 2)
        s = min(max(s, 1), 100)  # Clamp to reasonable range

        # Calculate cos^2s spreading
        # D(θ) ∝ cos^2s((θ - θ_mean) / 2)
        theta_diff = np.radians(directions - mean_dir)

        # Use half-angle for cos^2s
        cos_half = np.cos(theta_diff / 2)
        D = np.abs(cos_half) ** (2 * s)

        # Normalize so integral over directions = 1
        d_theta = self.config.d_theta
        D = D / (np.sum(D) * d_theta) * d_theta  # Normalize

        return D

    def reconstruct_partition(
        self,
        partition: "WavePartition",
        is_wind_sea: bool = False
    ) -> np.ndarray:
        """
        Reconstruct 2D spectrum from a single partition.

        Args:
            partition: WavePartition with Hs, Tp, Dir, Spread
            is_wind_sea: If True, use wind sea gamma value

        Returns:
            2D array E(f, θ) in m²/Hz/deg, shape (n_freq, n_dir)
        """
        if partition is None:
            return np.zeros((self.config.n_freq, self.config.n_dir))

        gamma = self.GAMMA_WIND_SEA if is_wind_sea else self.GAMMA_SWELL

        # 1D frequency spectrum
        S_f = self.jonswap_spectrum(
            self.config.frequencies,
            partition.hs,
            partition.tp,
            gamma
        )

        # Directional spreading
        D_theta = self.cos2s_spreading(
            self.config.directions,
            partition.dir,
            partition.spread
        )

        # 2D spectrum: E(f, θ) = S(f) * D(θ)
        # Shape: (n_freq, n_dir)
        E = np.outer(S_f, D_theta)

        return E

    def reconstruct_from_partitions(
        self,
        wind_waves: Optional["WavePartition"],
        primary_swell: Optional["WavePartition"],
        secondary_swell: Optional["WavePartition"] = None
    ) -> np.ndarray:
        """
        Reconstruct full 2D spectrum by superimposing partitions.

        Energy is additive: E_total = E_wind + E_swell1 + E_swell2

        Args:
            wind_waves: Wind sea partition
            primary_swell: Primary swell partition
            secondary_swell: Optional secondary swell partition

        Returns:
            2D array E(f, θ) in m²/Hz/deg, shape (n_freq, n_dir)
        """
        E_total = np.zeros((self.config.n_freq, self.config.n_dir))

        if wind_waves is not None:
            E_total += self.reconstruct_partition(wind_waves, is_wind_sea=True)

        if primary_swell is not None:
            E_total += self.reconstruct_partition(primary_swell, is_wind_sea=False)

        if secondary_swell is not None:
            E_total += self.reconstruct_partition(secondary_swell, is_wind_sea=False)

        return E_total


class SpectralBoundaryWriter:
    """
    Writes SWAN SPEC2D format spectral boundary files.

    Creates a single file containing 2D spectra for all boundary points,
    suitable for use with BOUNDSPEC SEGMENT ... SPEC2D command.

    SWAN SPEC2D format (simplified):
        SWAN 1                           # Version identifier
        $ Comments
        TIME time_string                 # Optional timestamp
        LONLAT npoints                   # Number of locations
        lon1 lat1                        # Location 1
        lon2 lat2                        # ...
        APTS nfreq ndir                  # Absolute frequencies, nautical directions
        freq1 freq2 ... freqN            # Frequencies (Hz)
        CDIR                             # Cartesian/nautical direction convention
        dir1 dir2 ... dirM               # Directions (degrees)
        QUANT 1                          # One quantity (VaDens = variance density)
        VaDens                           # Quantity name
        m2/Hz/degr                       # Units
        -99.0                            # Exception value
        LOCATION 1                       # First location
        E(f1,d1) E(f1,d2) ... E(f1,dM)  # Spectrum row by row
        E(f2,d1) ...
        ...
        LOCATION 2                       # Second location
        ...
    """

    EXCEPTION_VALUE = -99.0

    def __init__(self, reconstructor: SpectrumReconstructor):
        """
        Initialize spectral boundary writer.

        Args:
            reconstructor: SpectrumReconstructor for building spectra
        """
        self.reconstructor = reconstructor
        self.config = reconstructor.config

    def write_spec2d(
        self,
        output_path: Path,
        boundary_points: List["BoundaryPoint"],
        time_index: int = 0,
        timestamp: Optional[datetime] = None,
        stationary: bool = True
    ) -> Path:
        """
        Write SPEC2D file for all boundary points.

        Args:
            output_path: Path to output file
            boundary_points: List of BoundaryPoint objects with partition data
            time_index: Index into time series (for stationary, use 0)
            timestamp: Optional timestamp for the spectra (only used as comment for stationary)
            stationary: If True, omit TIME keyword (required for SWAN stationary mode)

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header - following pyswan format exactly
        lines.append("SWAN 1")
        lines.append("$   Spectral boundary file generated by SurflieModel")
        if timestamp:
            lines.append(f"$   Timestamp: {timestamp.isoformat()}")
        lines.append("$")

        # Locations - one coordinate pair per line
        n_points = len(boundary_points)
        lines.append("LONLAT")
        lines.append(str(n_points))
        for point in boundary_points:
            lines.append(f"{point.lon:f} {point.lat:f}")

        # Frequencies - one per line (pyswan format)
        lines.append("AFREQ")
        lines.append(str(self.config.n_freq))
        for freq in self.config.frequencies:
            lines.append(f"{freq:g}")

        # Directions - one per line, NDIR for nautical convention
        lines.append("NDIR")
        lines.append(str(self.config.n_dir))
        for d in self.config.directions:
            lines.append(f"{d:g}")

        # Quantity specification - with descriptive text
        lines.append("QUANT")
        lines.append("1 number of quantities in table")
        lines.append("VaDens variance densities in m2/Hz/degr")
        lines.append("m2/Hz/degr unit")
        lines.append("-99 exception value")

        # Spectra for each location
        # Following pyswan format: FACTOR, then one line per frequency with all directions
        for i, point in enumerate(boundary_points):
            # Get partition data for this time index
            wind = point.wind_waves[time_index] if point.wind_waves else None
            swell1 = point.primary_swell[time_index] if point.primary_swell else None
            swell2 = point.secondary_swell[time_index] if point.secondary_swell else None

            # Reconstruct 2D spectrum
            E = self.reconstructor.reconstruct_from_partitions(wind, swell1, swell2)

            # Replace very small values with zero (avoid extreme exponents)
            E = np.where(E < 1e-20, 0.0, E)

            # Write FACTOR and spectral values (pyswan format)
            lines.append("FACTOR")
            lines.append("1")

            # One line per frequency, all directions on that line (space-separated)
            for f_idx in range(self.config.n_freq):
                row_strs = [f"{E[f_idx, d_idx]:g}" for d_idx in range(self.config.n_dir)]
                lines.append(" ".join(row_strs))

        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')

        logger.info(f"Wrote SPEC2D file: {output_path} ({n_points} points, "
                   f"{self.config.n_freq} freqs, {self.config.n_dir} dirs)")

        return output_path

    def write_multi_boundary_spec2d(
        self,
        output_dir: Path,
        boundary_points: Dict[str, List["BoundaryPoint"]],
        time_index: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Path]:
        """
        Write SPEC2D files for multiple boundaries.

        Creates one spectral file per boundary side, named boundary_{side}.sp2

        Args:
            output_dir: Directory to write files to
            boundary_points: Dict mapping side name to list of BoundaryPoint objects
            time_index: Index into time series (for stationary, use 0)
            timestamp: Optional timestamp for the spectra

        Returns:
            Dict mapping side name to Path of written file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        spectral_files = {}

        for side, points in boundary_points.items():
            spec_path = output_dir / f"boundary_{side}.sp2"
            self.write_spec2d(
                output_path=spec_path,
                boundary_points=points,
                time_index=time_index,
                timestamp=timestamp,
            )
            spectral_files[side] = spec_path

        logger.info(f"Wrote {len(spectral_files)} spectral boundary files")
        return spectral_files


@dataclass
class SwanInputGenerator:
    """
    Generates SWAN INPUT files for stationary runs with spectral boundaries.

    Combines:
    - Mesh (CGRID, INPGRID, READINP for bathymetry)
    - Boundary conditions (BOUNDSPEC SEGMENT with SPEC2D files)
    - Physics settings

    Example usage:
        generator = SwanInputGenerator(
            mesh_dir="data/meshes/socal/coarse",
            boundary_file="data/swan/ww3_endpoints/socal/ww3_boundaries.json"
        )
        input_path = generator.generate(
            output_dir="data/swan/runs/socal/latest",
            boundary_config=config,
            spectral_files={"west": "boundary_west.sp2", "south": "boundary_south.sp2"}
        )
    """

    mesh_dir: str
    boundary_file: str
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)

    # Loaded data (populated by load())
    mesh_metadata: Optional[Dict] = None
    boundary_config: Optional[Dict] = None

    def __post_init__(self):
        self.mesh_dir = Path(self.mesh_dir)
        self.boundary_file = Path(self.boundary_file)
        self._load()

    def _load(self) -> None:
        """Load mesh and boundary metadata."""
        # Find and load mesh JSON
        json_files = list(self.mesh_dir.glob("*.json"))
        # Filter out ww3 boundary files
        mesh_jsons = [f for f in json_files if not f.name.startswith("ww3_")]
        if not mesh_jsons:
            raise FileNotFoundError(f"No mesh metadata found in {self.mesh_dir}")

        with open(mesh_jsons[0]) as f:
            self.mesh_metadata = json.load(f)

        logger.info(f"Loaded mesh: {self.mesh_metadata['name']}")

        # Load boundary config
        with open(self.boundary_file) as f:
            self.boundary_config = json.load(f)

        active = self.boundary_config.get("active_boundaries", [])
        logger.info(f"Loaded boundary config: {len(active)} active boundaries ({', '.join(active)})")

    def _generate_boundspec_for_boundary(
        self,
        boundary_data: Dict,
        spec_file: str
    ) -> List[str]:
        """
        Generate BOUNDSPEC SEGMENT command for a single boundary.

        Args:
            boundary_data: Dict with boundary definition (from unified config)
            spec_file: Filename of the SPEC2D file for this boundary

        Returns:
            List of SWAN command lines for this boundary's spectral specification
        """
        points = boundary_data["points"]
        n_points = boundary_data["n_points"]

        # Get first and last points of boundary segment
        start_lon, start_lat = points[0]
        end_lon, end_lat = points[-1]

        lines = [
            f"BOUNDSPEC SEGMENT XY {start_lon:.6f} {start_lat:.6f} {end_lon:.6f} {end_lat:.6f} &",
            f"    VARIABLE FILE LEN {n_points} '{spec_file}' 1"
        ]

        return lines

    def _generate_boundspec_commands(
        self,
        boundary_config: Dict,
        spectral_files: Dict[str, str]
    ) -> List[str]:
        """
        Generate BOUNDSPEC SEGMENT commands for all active boundaries.

        Args:
            boundary_config: Unified boundary config dict with 'boundaries' and 'active_boundaries'
            spectral_files: Dict mapping boundary side to spectral filename
                           e.g., {"west": "boundary_west.sp2", "south": "boundary_south.sp2"}

        Returns:
            List of SWAN command lines for all boundary specifications
        """
        lines = []

        for side in boundary_config["active_boundaries"]:
            if side not in spectral_files:
                raise ValueError(f"No spectral file provided for boundary '{side}'")

            boundary_data = boundary_config["boundaries"][side]
            spec_file = spectral_files[side]

            # Add comment for this boundary
            lines.append(f"$ {side.capitalize()} boundary ({boundary_data['n_points']} points)")
            lines.extend(self._generate_boundspec_for_boundary(boundary_data, spec_file))
            lines.append("")

        return lines

    def generate(
        self,
        output_dir: str | Path,
        boundary_config: Dict,
        spectral_files: Dict[str, str],
        wind_commands: Optional[List[str]] = None,
        current_commands: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        run_id: str = "001"
    ) -> Path:
        """
        Generate SWAN INPUT file for stationary mode with spectral boundaries.

        Args:
            output_dir: Directory to write INPUT file
            boundary_config: Unified boundary config dict with 'boundaries' and 'active_boundaries'
            spectral_files: Dict mapping boundary side to spectral filename
                           e.g., {"west": "boundary_west.sp2", "south": "boundary_south.sp2"}
            wind_commands: List of SWAN wind commands (INPGRID WIND, READINP WIND)
            current_commands: List of SWAN current commands (INPGRID CURRENT, READINP CURRENT)
            project_name: Project name (default: mesh name)
            run_id: Run identifier (default: "001")

        Returns:
            Path to generated INPUT file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if project_name is None:
            project_name = self.mesh_metadata["name"]

        input_path = output_dir / "INPUT"

        # Get boundary info for header
        active_boundaries = boundary_config["active_boundaries"]
        n_boundaries = len(active_boundaries)

        lines = []

        # Header
        lines.append(f"$ SWAN INPUT file generated {datetime.now().isoformat()}")
        lines.append(f"$ Mesh: {self.mesh_metadata['name']}")
        lines.append(f"$ Region: {self.mesh_metadata['region_name']}")
        lines.append(f"$ Boundary conditions: spectral (SPEC2D) - {n_boundaries} boundaries")
        lines.append(f"$ Active boundaries: {', '.join(active_boundaries)}")
        lines.append("$")
        lines.append("")

        # Project
        lines.append(f"PROJECT '{project_name}' '{run_id}'")
        lines.append("")

        # Mode - stationary
        lines.append("$ Stationary mode")
        lines.append("MODE STAT")
        lines.append("")

        # Coordinates - spherical
        lines.append("$ Spherical coordinates (lat/lon)")
        lines.append("COORD SPHE")
        lines.append("")

        # Computational grid
        lines.append("$ Computational grid")
        lines.append(self.mesh_metadata["swan_commands"]["cgrid"])
        lines.append("")

        # Bathymetry input
        lines.append("$ Bathymetry")
        lines.append(self.mesh_metadata["swan_commands"]["inpgrid"])
        lines.append(self.mesh_metadata["swan_commands"]["readinp"])
        lines.append("")

        # Wind input (if provided)
        if wind_commands:
            lines.append("$ Wind forcing (GFS)")
            lines.extend(wind_commands)
            lines.append("")

        # Current input (if provided)
        if current_commands:
            lines.append("$ Ocean currents (RTOFS)")
            lines.extend(current_commands)
            lines.append("")

        # Boundary conditions
        lines.append("$ Boundary conditions (WW3 spectral)")
        boundspec_lines = self._generate_boundspec_commands(boundary_config, spectral_files)
        lines.extend(boundspec_lines)

        # Physics
        lines.append("$ Physics")
        physics_commands = self.physics.to_swan_commands()
        lines.extend(physics_commands)
        lines.append("")

        # Numerical settings
        lines.append("$ Numerical settings")
        lines.append("NUM ACCUR 0.02 0.02 0.02 95 STAT 50")
        lines.append("")

        # Output - ASCII format (LAY 3) for easy reading with numpy
        lines.append("$ Output - Integrated parameters")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'hsig.mat' LAY 3 HSIG 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'tps.mat' LAY 3 TPS 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'dir.mat' LAY 3 DIR 1.")
        lines.append("")

        # Partition outputs
        lines.append("$ Output - Wave partitions (up to 6 spectral peaks)")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'phs0.mat' LAY 3 PTHSIG 0")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'ptp0.mat' LAY 3 PTRTP 0")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'pdir0.mat' LAY 3 PTDIR 0")
        lines.append("")

        # Compute - just COMPUTE for stationary mode
        lines.append("$ Compute")
        lines.append("COMPUTE")
        lines.append("")

        # Stop
        lines.append("STOP")
        lines.append("")

        # Write file
        with open(input_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Generated SWAN INPUT file: {input_path} "
                   f"(boundaries: {', '.join(active_boundaries)})")
        return input_path

    def summary(self) -> str:
        """Return summary of input configuration."""
        active = self.boundary_config.get("active_boundaries", [])
        lines = [
            "SWAN Input Generator",
            f"  Mesh: {self.mesh_metadata['name']}",
            f"  Region: {self.mesh_metadata['region_name']}",
            f"  Grid: {self.mesh_metadata['nx']} x {self.mesh_metadata['ny']}",
            f"  Resolution: {self.mesh_metadata['resolution_km']} km",
            f"  Active boundaries: {', '.join(active)}",
            "",
            "Physics:",
            f"  GEN3: {self.physics.gen3_formulation}",
            f"  Friction: {self.physics.friction} ({self.physics.friction_coefficient})",
            f"  Breaking: {self.physics.breaking}",
            f"  Triads: {self.physics.triads}",
        ]
        return '\n'.join(lines)
