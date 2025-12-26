#!/usr/bin/env python3
"""
WW3 to SWAN Spectral Boundary Condition Converter

Converts WW3 parametric wave data (Hs, Tp, Dir) into SWAN 2D spectral format.
This preserves swell partitions better than simple TPAR format.

The converter:
1. Reads WW3 partitioned data (wind sea + primary swell)
2. Synthesizes 2D spectra using JONSWAP/PM spectral shapes
3. Writes SWAN .sp2 spectral files for each boundary point

Usage:
    from data.pipelines.swan.spectral_boundary import SpectralBoundaryGenerator

    generator = SpectralBoundaryGenerator(domain_config)
    generator.generate_boundary_spectra(ww3_ds, boundary_points, output_dir)
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralBoundaryGenerator:
    """
    Generates SWAN spectral boundary conditions from WW3 parametric data.

    Synthesizes 2D wave spectra E(f, θ) from bulk wave parameters using:
    - JONSWAP spectrum for wind sea (gamma=3.3)
    - Pierson-Moskowitz spectrum for swell (gamma=1.0)
    - cos^n directional spreading
    """

    # Spectral discretization (matching SWAN CGRID settings)
    N_FREQ = 36          # Number of frequency bins
    FREQ_MIN = 0.04      # Minimum frequency (Hz) - matches CGRID
    FREQ_MAX = 1.0       # Maximum frequency (Hz)
    N_DIR = 36           # Number of direction bins

    # Physical constants
    G = 9.81             # Gravity (m/s^2)

    def __init__(self, domain_config: Optional[Dict] = None):
        """
        Initialize spectral generator.

        Args:
            domain_config: Optional domain configuration dict
        """
        self.domain_config = domain_config or {}

        # Generate frequency and direction arrays
        self.frequencies = np.geomspace(self.FREQ_MIN, self.FREQ_MAX, self.N_FREQ)
        self.directions = np.linspace(0, 360 - 360/self.N_DIR, self.N_DIR)

        # Direction in radians for calculations
        self.dir_rad = np.radians(self.directions)

    def jonswap_spectrum(
        self,
        freq: np.ndarray,
        hs: float,
        tp: float,
        gamma: float = 3.3
    ) -> np.ndarray:
        """
        Generate JONSWAP frequency spectrum.

        Args:
            freq: Frequency array (Hz)
            hs: Significant wave height (m)
            tp: Peak period (s)
            gamma: Peak enhancement factor (3.3 for wind sea, 1.0 for swell)

        Returns:
            1D array of spectral density S(f) in m^2/Hz
        """
        if hs <= 0 or tp <= 0:
            return np.zeros_like(freq)

        fp = 1.0 / tp  # Peak frequency

        # JONSWAP parameters
        sigma = np.where(freq <= fp, 0.07, 0.09)

        # Phillips constant (alpha) from Hs
        alpha = 5.061 * (hs**2) * (fp**4) / self.G**2 * (1 - 0.287 * np.log(gamma))

        # Pierson-Moskowitz base spectrum
        pm = alpha * self.G**2 / ((2 * np.pi)**4 * freq**5) * \
             np.exp(-1.25 * (fp / freq)**4)

        # Peak enhancement factor
        r = np.exp(-0.5 * ((freq - fp) / (sigma * fp))**2)
        enhancement = gamma ** r

        spectrum = pm * enhancement

        # Handle numerical issues
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        return spectrum

    def directional_spreading(
        self,
        directions: np.ndarray,
        mean_dir: float,
        spreading: float = 25.0
    ) -> np.ndarray:
        """
        Generate directional spreading function using cos^n distribution.

        Args:
            directions: Direction array (degrees, nautical convention)
            mean_dir: Mean wave direction (degrees)
            spreading: Directional spreading parameter (degrees)

        Returns:
            1D array of directional distribution (integrates to 1)
        """
        # Convert to radians
        dir_rad = np.radians(directions)
        mean_rad = np.radians(mean_dir)

        # Spreading exponent (larger spreading param = narrower distribution)
        # s = 2 gives cos^2, s = 20 gives very narrow
        s = max(2, (90 / spreading) ** 2)

        # Cos^n distribution
        cos_diff = np.cos(dir_rad - mean_rad)
        D = np.maximum(cos_diff, 0) ** s

        # Normalize to integrate to 1
        D = D / (np.sum(D) * np.radians(360 / len(directions)))

        return D

    def synthesize_2d_spectrum(
        self,
        hs: float,
        tp: float,
        direction: float,
        spreading: float = 25.0,
        gamma: float = 3.3
    ) -> np.ndarray:
        """
        Synthesize a 2D wave spectrum E(f, θ) from bulk parameters.

        Args:
            hs: Significant wave height (m)
            tp: Peak period (s)
            direction: Mean wave direction (degrees, nautical)
            spreading: Directional spreading (degrees)
            gamma: JONSWAP peak enhancement (3.3=wind sea, 1.0=swell)

        Returns:
            2D array of shape (n_freq, n_dir) in m^2/Hz/deg
        """
        if hs <= 0 or tp <= 0:
            return np.zeros((self.N_FREQ, self.N_DIR))

        # 1D frequency spectrum
        S_f = self.jonswap_spectrum(self.frequencies, hs, tp, gamma)

        # Directional distribution
        D = self.directional_spreading(self.directions, direction, spreading)

        # 2D spectrum: E(f,θ) = S(f) * D(θ)
        spectrum_2d = np.outer(S_f, D)

        return spectrum_2d

    def combine_spectra(
        self,
        partitions: List[Dict]
    ) -> np.ndarray:
        """
        Combine multiple wave partitions into a single 2D spectrum.

        Args:
            partitions: List of dicts with keys: hs, tp, dir, spreading, gamma

        Returns:
            Combined 2D spectrum
        """
        combined = np.zeros((self.N_FREQ, self.N_DIR))

        for part in partitions:
            hs = part.get('hs', 0)
            tp = part.get('tp', 0)
            direction = part.get('dir', 270)
            spreading = part.get('spreading', 25)
            gamma = part.get('gamma', 3.3)

            if hs > 0 and tp > 0:
                spec = self.synthesize_2d_spectrum(hs, tp, direction, spreading, gamma)
                combined += spec

        return combined

    def write_swan_spectrum(
        self,
        output_path: Path,
        spectra: Dict[str, np.ndarray],
        location: Dict
    ):
        """
        Write SWAN .sp2 spectral file.

        SWAN spectral format:
        - Header with SWAN version
        - Frequency and direction definitions
        - Time-varying spectra

        Args:
            output_path: Path for output .sp2 file
            spectra: Dict mapping time_str -> 2D spectrum array
            location: Dict with lat, lon of boundary point
        """
        with open(output_path, 'w') as f:
            # Header
            f.write("SWAN   1                                Swan standard spectral file, version\n")
            f.write(f"$   Data produced by SurflieModel spectral_boundary.py\n")
            f.write(f"$   Location: lat={location['lat']:.4f} lon={location['lon']:.4f}\n")
            f.write("TIME                                    time-dependent data\n")
            f.write(f"     1                                  time coding option\n")
            f.write("LONLAT                                  locations in spherical coordinates\n")
            f.write(f"     1                                  number of locations\n")
            f.write(f"  {location['lon']:.6f}  {location['lat']:.6f}\n")
            f.write("APTS                                    absolute frequencies in Hz\n")
            f.write(f"    {self.N_FREQ}                                  number of frequencies\n")

            # Write frequencies
            for freq in self.frequencies:
                f.write(f"  {freq:.6f}\n")

            f.write("NDIR                                    spectral nautical directions in degr\n")
            f.write(f"    {self.N_DIR}                                  number of directions\n")

            # Write directions
            for dir_val in self.directions:
                f.write(f"  {dir_val:.1f}\n")

            f.write("QUANT\n")
            f.write("     1                                  number of quantities in table\n")
            f.write("VaDens                                  variance densities in m2/Hz/degr\n")
            f.write("m2/Hz/degr                              unit\n")
            f.write("   -0.9900E+02                          exception value\n")

            # Write time-varying spectra
            for time_str, spectrum in sorted(spectra.items()):
                # Parse time
                dt = datetime.fromisoformat(time_str.replace("T", " ").split(".")[0])
                swan_time = dt.strftime("%Y%m%d.%H%M%S")

                f.write(f"{swan_time}                         date and time\n")
                f.write("FACTOR\n")

                # Scale factor for numerical precision
                max_val = np.max(spectrum)
                if max_val > 0:
                    factor = max_val / 999.0  # Keep values < 1000
                else:
                    factor = 1.0
                f.write(f"  {factor:.6E}\n")

                # Write scaled spectrum (freq x dir)
                scaled = spectrum / factor if factor > 0 else spectrum
                for i_freq in range(self.N_FREQ):
                    row = scaled[i_freq, :]
                    # Write in groups of 7 values per line
                    for j in range(0, self.N_DIR, 7):
                        vals = row[j:min(j+7, self.N_DIR)]
                        f.write("  " + "  ".join(f"{v:.4f}" for v in vals) + "\n")

        logger.info(f"Written spectral file: {output_path}")

    def extract_ww3_partitions(
        self,
        ww3_ds,
        lat: float,
        lon: float,
        time_idx: int
    ) -> List[Dict]:
        """
        Extract wave partitions from WW3 data at a specific point.

        Args:
            ww3_ds: xarray Dataset with WW3 data
            lat: Latitude of point
            lon: Longitude of point
            time_idx: Time index

        Returns:
            List of partition dicts with hs, tp, dir, spreading, gamma
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy import ndimage

        ww3_lats = ww3_ds["lat"].values
        ww3_lons = ww3_ds["lon"].values

        # Ensure ascending order for interpolation
        lat_ascending = ww3_lats[0] < ww3_lats[-1]
        lon_ascending = ww3_lons[0] < ww3_lons[-1]

        if not lat_ascending:
            ww3_lats = ww3_lats[::-1]
        if not lon_ascending:
            ww3_lons = ww3_lons[::-1]

        def get_interpolated(var_name: str) -> float:
            """Interpolate a variable at the point."""
            if var_name not in ww3_ds:
                return np.nan

            data = ww3_ds[var_name].isel(time=time_idx).values

            if not lat_ascending:
                data = data[::-1, :]
            if not lon_ascending:
                data = data[:, ::-1]

            # Fill NaN
            mask = np.isnan(data)
            if mask.all():
                return np.nan
            if mask.any():
                indices = ndimage.distance_transform_edt(
                    mask, return_distances=False, return_indices=True
                )
                data = data[tuple(indices)]

            interp = RegularGridInterpolator(
                (ww3_lats, ww3_lons), data,
                method='linear', bounds_error=False, fill_value=np.nan
            )
            return float(interp(np.array([[lat, lon]]))[0])

        partitions = []

        # Wind sea partition
        hs_wind = get_interpolated('hs_wind')
        if not np.isnan(hs_wind) and hs_wind > 0.05:
            # Estimate wind sea period from combined if not available
            tp_combined = get_interpolated('tp')
            partitions.append({
                'hs': hs_wind,
                'tp': min(7.0, tp_combined * 0.6) if not np.isnan(tp_combined) else 5.0,
                'dir': get_interpolated('dp'),
                'spreading': 35.0,  # Wind sea has wider spreading
                'gamma': 3.3,       # JONSWAP for wind sea
                'type': 'windsea'
            })

        # Primary swell partition
        hs_swell = get_interpolated('hs_swell')
        tp_swell = get_interpolated('tp_swell')
        dp_swell = get_interpolated('dp_swell')

        if not np.isnan(hs_swell) and hs_swell > 0.05:
            partitions.append({
                'hs': hs_swell,
                'tp': tp_swell if not np.isnan(tp_swell) else 12.0,
                'dir': dp_swell if not np.isnan(dp_swell) else 270.0,
                'spreading': 20.0,  # Swell has narrower spreading
                'gamma': 1.0,       # PM spectrum for swell
                'type': 'swell'
            })

        # If no partitions found, use combined parameters
        if not partitions:
            hs = get_interpolated('hs')
            tp = get_interpolated('tp')
            dp = get_interpolated('dp')

            if not np.isnan(hs) and hs > 0.05:
                partitions.append({
                    'hs': hs,
                    'tp': tp if not np.isnan(tp) else 10.0,
                    'dir': dp if not np.isnan(dp) else 270.0,
                    'spreading': 25.0,
                    'gamma': 2.0,  # Mixed sea state
                    'type': 'combined'
                })

        return partitions

    def generate_boundary_spectra(
        self,
        ww3_ds,
        boundary_points: List[Dict],
        output_dir: Path
    ) -> List[Path]:
        """
        Generate spectral boundary condition files for all points.

        Args:
            ww3_ds: xarray Dataset with WW3 data
            boundary_points: List of boundary point dicts with lat, lon
            output_dir: Directory for output files

        Returns:
            List of paths to generated .sp2 files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        times = ww3_ds["time"].values
        n_times = len(times)
        n_points = len(boundary_points)

        logger.info(f"Generating spectral boundaries: {n_points} points x {n_times} times")

        sp2_files = []

        for pt_idx, point in enumerate(boundary_points):
            lat = point['lat']
            lon = point['lon']

            # Generate spectra for all times
            spectra = {}
            for t_idx, time_val in enumerate(times):
                time_str = str(time_val)[:19]

                # Extract partitions at this point and time
                partitions = self.extract_ww3_partitions(ww3_ds, lat, lon, t_idx)

                # Combine into 2D spectrum
                spectrum = self.combine_spectra(partitions)
                spectra[time_str] = spectrum

            # Write spectral file
            sp2_path = output_dir / f"boundary_pt{pt_idx:03d}.sp2"
            self.write_swan_spectrum(sp2_path, spectra, point)
            sp2_files.append(sp2_path)

            if (pt_idx + 1) % 10 == 0:
                logger.info(f"  Generated {pt_idx + 1}/{n_points} spectral files")

        logger.info(f"Generated {len(sp2_files)} spectral boundary files")
        return sp2_files

    def generate_boundspec_commands(
        self,
        boundary_points: List[Dict],
        sp2_files: List[Path]
    ) -> str:
        """
        Generate SWAN BOUNDSPEC commands for spectral boundary conditions.

        Args:
            boundary_points: List of boundary point dicts
            sp2_files: List of spectral file paths

        Returns:
            String with BOUNDSPEC commands for SWAN input file
        """
        lines = []
        lines.append("$ Spectral boundary conditions from WW3")
        lines.append(f"$ {len(boundary_points)} boundary points with 2D spectra")
        lines.append("$")

        for pt_idx, (point, sp2_path) in enumerate(zip(boundary_points, sp2_files)):
            lat = point['lat']
            lon = point['lon']
            filename = sp2_path.name

            # BOUNDSPEC command for this point
            # BOUNDSPEC SEGMENT XY lon1 lat1 lon2 lat2 VARIABLE FILE 'filename.sp2'
            # For single points, use the same coordinates
            lines.append(f"BOUNDSPEC SEGMENT XY {lon:.4f} {lat:.4f} {lon:.4f} {lat:.4f} &")
            lines.append(f"          VARIABLE FILE '{filename}'")

        return "\n".join(lines)


def main():
    """Test spectral generator."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    generator = SpectralBoundaryGenerator()

    # Test spectrum synthesis
    print("Testing spectrum synthesis...")

    # Wind sea: Hs=1m, Tp=6s, Dir=270, narrow
    wind_spec = generator.synthesize_2d_spectrum(1.0, 6.0, 270, 35, 3.3)
    print(f"Wind sea spectrum shape: {wind_spec.shape}, max: {wind_spec.max():.4f}")

    # Swell: Hs=2m, Tp=14s, Dir=285
    swell_spec = generator.synthesize_2d_spectrum(2.0, 14.0, 285, 20, 1.0)
    print(f"Swell spectrum shape: {swell_spec.shape}, max: {swell_spec.max():.4f}")

    # Combined
    combined = wind_spec + swell_spec
    print(f"Combined spectrum max: {combined.max():.4f}")

    # Verify energy conservation (approximately)
    # Hs = 4 * sqrt(m0) where m0 = integral of spectrum
    df = generator.frequencies[1] / generator.frequencies[0]  # Log spacing ratio
    d_theta = np.radians(360 / generator.N_DIR)

    m0_wind = np.sum(wind_spec) * df * d_theta * generator.frequencies[0]
    m0_swell = np.sum(swell_spec) * df * d_theta * generator.frequencies[0]

    print(f"Wind sea Hs from spectrum: {4 * np.sqrt(m0_wind):.2f}m (target: 1.0m)")
    print(f"Swell Hs from spectrum: {4 * np.sqrt(m0_swell):.2f}m (target: 2.0m)")

    print("\nSpectral generator test complete!")


if __name__ == "__main__":
    main()
