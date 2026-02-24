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
) -> Tuple[np.ndarray, Dict[int, dict]]:
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
        Updated (total_energy, per_partition_data) with breaking correction applied
    """
    import time

    # Check prerequisites
    if mesh.coast_distance is None or mesh.coastlines is None:
        print("  Breaking correction: skipped (no coast_distance or coastlines)")
        return total_energy, per_partition_data

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
            H_max = gamma * h

            if Hs_total > H_max:
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

    return total_energy, per_partition_data
