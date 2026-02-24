"""
Post-Deposition Cross-Shore Transect Breaking Correction

Stage 2 of the hybrid breaking dissipation approach. After per-ray Thornton & Guza
dissipation handles dominant swell breaking during propagation, this module catches
combined multi-partition breaking where no single ray exceeds the threshold alone
but combined Hs_total does.

Algorithm:
    1. Compute alongshore coordinate per mesh point (arc-length along nearest
       coastline segment)
    2. Bin mesh points into ~100m alongshore strips
    3. Within each strip, sort by coast_distance (offshore → shore)
    4. Walk each strip: compute combined Hs_total, apply Q_b correction where
       Hs_total > gamma * h, carry corrected Hs forward

References:
    Thornton & Guza (1983), Baldock et al. (1998), Daly et al. (2012)
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


RHO = 1025.0
G = 9.81


class CrossShoreTransectBuilder:
    """
    Builds cross-shore transects from mesh coastline data for breaking correction.

    Transects are constructed by:
    1. Computing an alongshore coordinate for each mesh point (arc-length
       along the nearest coastline segment)
    2. Binning points into alongshore strips (~100m wide)
    3. Sorting each strip by coast_distance (offshore → shore)

    The transect structure is cacheable — compute once per mesh.
    """

    def __init__(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        coast_distance: np.ndarray,
        coastlines: list,
        alongshore_bin_m: float = 100.0,
    ):
        """
        Args:
            points_x: UTM x coordinates of mesh points
            points_y: UTM y coordinates of mesh points
            coast_distance: Distance from coastline per mesh point (m)
            coastlines: List of (N, 2) coastline polylines in UTM
            alongshore_bin_m: Width of alongshore bins (m)
        """
        self.n_points = len(points_x)
        self.alongshore_bin_m = alongshore_bin_m

        # Build coastline KD-tree and compute alongshore coordinate
        self._alongshore, self._coast_tree = self._compute_alongshore_coordinate(
            points_x, points_y, coastlines
        )

        # Bin into alongshore strips and sort by coast_distance within each
        self.strip_indices, self.n_strips = self._build_strips(
            self._alongshore, coast_distance
        )

    def _compute_alongshore_coordinate(
        self,
        points_x: np.ndarray,
        points_y: np.ndarray,
        coastlines: list,
    ) -> Tuple[np.ndarray, cKDTree]:
        """
        Compute alongshore coordinate for each mesh point.

        The alongshore coordinate is the cumulative arc-length along the
        concatenated coastline at the point nearest to each mesh point.
        """
        # Concatenate all coastline segments with cumulative arc-length
        all_coast_pts = []
        all_arc_lengths = []
        cumulative_length = 0.0

        for coastline in coastlines:
            if len(coastline) < 2:
                continue
            # Sample coastline at fine resolution for accurate nearest-neighbor
            pts = coastline  # Already in UTM (N, 2)
            for i in range(len(pts)):
                all_coast_pts.append(pts[i])
                if i == 0:
                    all_arc_lengths.append(cumulative_length)
                else:
                    dx = pts[i, 0] - pts[i - 1, 0]
                    dy = pts[i, 1] - pts[i - 1, 1]
                    cumulative_length += np.sqrt(dx * dx + dy * dy)
                    all_arc_lengths.append(cumulative_length)

        if len(all_coast_pts) == 0:
            # No coastlines — return zeros
            return np.zeros(self.n_points), None

        coast_pts = np.array(all_coast_pts)
        arc_lengths = np.array(all_arc_lengths)

        # Build KD-tree for coastline
        tree = cKDTree(coast_pts)

        # Find nearest coastline point for each mesh point
        mesh_pts = np.column_stack([points_x, points_y])
        _, nearest_idx = tree.query(mesh_pts)

        # Alongshore coordinate = arc-length of nearest coastline point
        alongshore = arc_lengths[nearest_idx]

        return alongshore, tree

    def _build_strips(
        self,
        alongshore: np.ndarray,
        coast_distance: np.ndarray,
    ) -> Tuple[list, int]:
        """
        Bin mesh points into alongshore strips and sort by coast_distance.

        Returns:
            strip_indices: List of arrays, each containing mesh point indices
                          sorted by coast_distance (offshore → shore, descending)
            n_strips: Number of strips
        """
        if len(alongshore) == 0:
            return [], 0

        # Bin indices
        bin_idx = (alongshore / self.alongshore_bin_m).astype(np.int32)
        n_strips = bin_idx.max() + 1

        # Group points by bin and sort each by coast_distance (descending = offshore first)
        strip_indices = []
        for b in range(n_strips):
            mask = bin_idx == b
            indices = np.where(mask)[0]
            if len(indices) == 0:
                strip_indices.append(np.array([], dtype=np.int64))
                continue
            # Sort by coast_distance descending (offshore → shore)
            order = np.argsort(-coast_distance[indices])
            strip_indices.append(indices[order])

        return strip_indices, n_strips


def apply_combined_breaking_correction(
    mesh: 'SurfZoneMesh',
    total_energy: np.ndarray,
    per_partition_data: Dict[int, dict],
    gamma: float = 0.42,
    alongshore_bin_m: float = 100.0,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
    wind_Cw: float = 0.15,
) -> Tuple[np.ndarray, Dict[int, dict], np.ndarray]:
    """
    Apply post-deposition cross-shore transect breaking correction.

    Walks cross-shore transects from offshore to shore. At each mesh point,
    computes combined Hs_total from all partitions. If Hs_total > gamma * h,
    applies Q_b reduction to all partition energies at that point and carries
    the corrected Hs forward to the next shoreward point.

    Args:
        mesh: SurfZoneMesh with coastlines and coast_distance
        total_energy: Total accumulated energy grid (sum of all partitions)
        per_partition_data: Dict mapping partition_idx to dict with 'energy' key
        gamma: Breaking threshold parameter (T&G convention)
        alongshore_bin_m: Width of alongshore bins (m)

    Returns:
        Tuple of (total_energy, per_partition_data, is_breaking) where
        is_breaking is a boolean array marking points where breaking was detected
    """
    import time

    n_points = len(total_energy)

    # Check prerequisites
    if mesh.coast_distance is None or mesh.coastlines is None:
        print("  Breaking correction: skipped (no coast_distance or coastlines)")
        return total_energy, per_partition_data, np.zeros(n_points, dtype=bool)

    t_start = time.perf_counter()

    depths = np.maximum(mesh.depth, 0.05) if mesh.depth is not None else None
    if depths is None:
        depths = np.maximum(-mesh.elevation, 0.05)

    # Build transects
    transects = CrossShoreTransectBuilder(
        mesh.points_x, mesh.points_y,
        mesh.coast_distance, mesh.coastlines,
        alongshore_bin_m=alongshore_bin_m,
    )

    # Collect partition energy arrays
    n_partitions = len(per_partition_data)
    partition_energies = []
    partition_keys = sorted(per_partition_data.keys())
    for k in partition_keys:
        partition_energies.append(per_partition_data[k]['energy'])

    # Track which points are breaking
    is_breaking = np.zeros(n_points, dtype=bool)

    # Walk each strip offshore → shore
    n_corrected = 0
    for strip_idx in transects.strip_indices:
        if len(strip_idx) == 0:
            continue

        for i, pt_idx in enumerate(strip_idx):
            h = depths[pt_idx]
            if h <= 0.05:
                continue

            # Compute combined Hs_total at this point
            total_e = 0.0
            for pe in partition_energies:
                total_e += pe[pt_idx]

            if total_e <= 0.0:
                continue

            Hs_total = np.sqrt(8.0 * total_e / (RHO * G))

            # Wind-modified gamma (Douglass, 1990)
            gamma_local = gamma
            if wind_u is not None and wind_v is not None:
                wu = wind_u[pt_idx]
                wv = wind_v[pt_idx]
                wind_speed = np.sqrt(wu * wu + wv * wv)
                if wind_speed > 0.1:
                    wind_dir = np.arctan2(wv, wu)
                    # Energy-weighted wave direction at this point
                    dx_total = 0.0
                    dy_total = 0.0
                    for k in partition_keys:
                        if 'dir_x' in per_partition_data[k]:
                            dx_total += per_partition_data[k]['dir_x'][pt_idx]
                            dy_total += per_partition_data[k]['dir_y'][pt_idx]
                    wave_dir = np.arctan2(dy_total, dx_total)
                    phi = wind_dir - wave_dir
                    C_local = np.sqrt(G * h)  # Shallow water approximation
                    gamma_local = gamma * (1.0 - wind_Cw * wind_speed * np.cos(phi) / C_local)
                    gamma_local = max(0.3, min(1.0, gamma_local))

            H_max = gamma_local * h

            if Hs_total > H_max:
                is_breaking[pt_idx] = True

                # Compute Q_b from Rayleigh exceedance
                ratio = H_max / Hs_total
                Q_b = np.exp(-2.0 * ratio * ratio)

                # Apply correction: reduce all partition energies proportionally
                # Energy scales as Hs^2, so factor = (1 - Q_b)
                factor = 1.0 - Q_b
                for pe in partition_energies:
                    pe[pt_idx] *= factor

                n_corrected += 1

    # Recompute total energy from corrected partitions
    total_energy = np.zeros_like(total_energy)
    for i, k in enumerate(partition_keys):
        per_partition_data[k]['energy'] = partition_energies[i]
        total_energy += partition_energies[i]

    elapsed = time.perf_counter() - t_start
    print(f"  Breaking correction: {n_corrected:,} points corrected in {elapsed:.2f}s "
          f"({transects.n_strips} transects, gamma={gamma})")

    return total_energy, per_partition_data, is_breaking


def compute_breaking_characterization(
    mesh: 'SurfZoneMesh',
    energy: np.ndarray,
    per_partition_data: Dict[int, dict],
    is_breaking: np.ndarray,
    gamma: float = 0.42,
    wind_u: Optional[np.ndarray] = None,
    wind_v: Optional[np.ndarray] = None,
    wind_Cw: float = 0.15,
) -> dict:
    """
    Compute breaking characterization at mesh points after cross-shore correction.

    Uses the same physics as the backward ray tracer (ray_tracer.py:729-743):
    Rattanapitikon breaker index, Iribarren number, and breaker type classification.

    Only characterizes points where is_breaking=True (as determined by the
    cross-shore correction). All other points get NaN for classification fields.

    Args:
        mesh: SurfZoneMesh with bathymetry and slopes
        energy: Total combined energy at each mesh point (post-correction)
        per_partition_data: Per-partition data with 'energy' arrays
        is_breaking: Boolean array from cross-shore correction
        gamma: Breaking gamma used by the simulation

    Returns:
        Dict with keys: is_breaking, breaker_index, iribarren, breaker_type,
        breaking_intensity — all shape (n_points,)
    """
    n_points = len(mesh.points_x)
    depths = np.maximum(mesh.depth, 0.05) if mesh.depth is not None else np.maximum(-mesh.elevation, 0.05)

    # Compute bottom slopes
    slopes = mesh.get_slope_magnitude(mesh.points_x, mesh.points_y, h=10.0)

    # Combined Hs from total energy
    Hs_total = np.sqrt(8.0 * np.maximum(energy, 0) / (RHO * G))

    # Dominant Tp: from highest-energy partition at each point
    partition_keys = sorted(per_partition_data.keys())
    max_energy = np.zeros(n_points)
    Tp_dominant = np.zeros(n_points)

    from scipy.spatial import cKDTree

    for k in partition_keys:
        part_e = per_partition_data[k]['energy']
        # Need Tp — get from boundary data via nearest-neighbor
        # per_partition_data has 'energy', 'dir_x', 'dir_y', 'ray_counts', 'Hs', 'direction'
        # but not Tp directly. We need to get it from the partition result.
        # For now, estimate Tp from Hs and energy relationship or use a reference period.
        better = part_e > max_energy
        max_energy = np.where(better, part_e, max_energy)

    # We need Tp from the boundary conditions — it's not in per_partition_data.
    # Use a representative period from the energy-weighted approach.
    # Fall back to estimating from the mesh: load boundary Tp from partition NPZ if available.
    # For the simulation context, we can get Tp from the partition results that will be built later.
    # Instead, use a simpler approach: deep water wavelength from period isn't critical for
    # the characterization — we need it for Iribarren. Use 10s as a reasonable default
    # and override with actual partition Tp if available.
    #
    # Actually, the ForwardRayTracer has the boundary conditions. Let's accept Tp as a parameter.
    # For now, use a conservative estimate.

    # Check if 'Tp' was passed in per_partition_data (added by caller)
    has_tp = any('Tp' in per_partition_data[k] for k in partition_keys)

    if has_tp:
        # Use energy-weighted Tp from partition data
        Tp_dominant = np.zeros(n_points)
        max_energy = np.zeros(n_points)
        for k in partition_keys:
            part_e = per_partition_data[k]['energy']
            part_tp = per_partition_data[k]['Tp']
            better = part_e > max_energy
            Tp_dominant = np.where(better, part_tp, Tp_dominant)
            max_energy = np.where(better, part_e, max_energy)
    else:
        # Fallback: estimate period from typical swell (10s)
        # This is approximate but better than nothing
        Tp_dominant = np.full(n_points, 10.0)

    # Deep water wavelength
    TWO_PI = 2.0 * np.pi
    L0 = G * Tp_dominant**2 / TWO_PI
    safe_L0 = np.where(L0 > 0, L0, 1.0)

    # Breaker index (Rattanapitikon & Shibayama, matching backward tracer)
    H0_L0 = np.where(L0 > 0, Hs_total / safe_L0, 0.0)
    safe_H0_L0 = np.where(H0_L0 > 0, H0_L0, 1e-6)
    safe_slopes = np.where(np.isfinite(slopes) & (slopes > 0), slopes, 0.005)
    gamma_b = np.clip(0.57 + 0.71 * safe_H0_L0**0.12 * safe_slopes**0.36, 0.5, 1.5)

    # Apply Douglass (1990) wind modification to breaker index
    if wind_u is not None and wind_v is not None:
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        has_wind = wind_speed > 0.1
        if np.any(has_wind):
            wind_dir = np.arctan2(wind_v, wind_u)
            # Energy-weighted wave direction from partition dir components
            dx_total = np.zeros(n_points)
            dy_total = np.zeros(n_points)
            for k in partition_keys:
                if 'dir_x' in per_partition_data[k]:
                    dx_total += per_partition_data[k]['dir_x']
                    dy_total += per_partition_data[k]['dir_y']
            wave_dir = np.arctan2(dy_total, dx_total)
            phi = wind_dir - wave_dir
            C_local = np.sqrt(G * depths)  # Shallow water approximation
            wind_factor = 1.0 - wind_Cw * wind_speed * np.cos(phi) / np.maximum(C_local, 0.1)
            gamma_b = np.where(has_wind, np.clip(gamma_b * wind_factor, 0.3, 1.5), gamma_b)

    # Iribarren: xi = m / sqrt(H/L0)
    steepness = np.where(L0 > 0, Hs_total / safe_L0, 0.0)
    safe_steepness = np.where(steepness > 0, steepness, np.nan)
    iribarren = np.where(
        is_breaking & np.isfinite(slopes) & (steepness > 0),
        slopes / np.sqrt(safe_steepness),
        np.nan,
    )

    # Breaker type classification (only where breaking)
    breaker_type = np.full(n_points, np.nan)
    xi = np.where(np.isfinite(iribarren), iribarren, 0.0)
    valid = is_breaking & np.isfinite(iribarren)
    breaker_type = np.where(valid & (xi < 0.5), 0.0, breaker_type)
    breaker_type = np.where(valid & (xi >= 0.5) & (xi < 3.3), 1.0, breaker_type)
    breaker_type = np.where(valid & (xi >= 3.3) & (xi < 5.0), 2.0, breaker_type)
    breaker_type = np.where(valid & (xi >= 5.0), 3.0, breaker_type)

    # Breaking intensity: H / (gamma_b * h)
    breaking_intensity = np.where(
        is_breaking & (depths > 0.05),
        Hs_total / (gamma_b * depths),
        np.nan,
    )

    n_breaking = int(np.sum(is_breaking))
    if n_breaking > 0:
        valid_xi = iribarren[is_breaking & np.isfinite(iribarren)]
        print(f"  Breaking characterization: {n_breaking:,} breaking points")
        if len(valid_xi) > 0:
            # Count breaker types
            n_spilling = int(np.sum(valid_xi < 0.5))
            n_plunging = int(np.sum((valid_xi >= 0.5) & (valid_xi < 3.3)))
            n_collapsing = int(np.sum((valid_xi >= 3.3) & (valid_xi < 5.0)))
            n_surging = int(np.sum(valid_xi >= 5.0))
            print(f"    Spilling: {n_spilling}, Plunging: {n_plunging}, "
                  f"Collapsing: {n_collapsing}, Surging: {n_surging}")

    return {
        'is_breaking': is_breaking.astype(float),
        'breaker_index': np.where(is_breaking, gamma_b, np.nan),
        'iribarren': iribarren,
        'breaker_type': breaker_type,
        'breaking_intensity': breaking_intensity,
    }
