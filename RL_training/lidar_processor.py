"""
lidar_processor.py

Compact 3D LiDAR processing for the UAV RL environment.

The RL agent does NOT receive the full point cloud.  Instead, this module
converts the point cloud into the same 9 obstacle features that the old
DistanceFront/DistanceLeft/... sensors produced:

    front, front_left, front_right, left, right, back, down,
    horizontal_min, collision risk is computed later by ObservationBuilder.

Coordinate convention expected from AirSim/Cosys-AirSim when the LiDAR uses
DataFrame="SensorLocalFrame":
    x > 0 : forward
    y > 0 : right
    z > 0 : down

If your settings.json uses a different DataFrame, switch the LiDAR to
SensorLocalFrame. That keeps the sector logic independent of global heading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import math
import numpy as np


@dataclass
class LidarProcessorConfig:
    max_range_m: float = 20.0
    min_valid_range_m: float = 0.05

    # Horizontal obstacle sectors should not be polluted by the road/ground.
    # In SensorLocalFrame, ground below the drone usually has positive z.
    # Points with |z| above this value are ignored for horizontal sectors.
    horizontal_abs_z_max_m: float = 1.75

    # Down/ground estimate near the drone footprint.
    down_xy_radius_m: float = 2.0
    down_min_z_m: float = 0.05

    # Minimum radius around the LiDAR origin treated as drone/self returns.
    # Raw AirSim/Cosys point clouds can include points at ~0.10-0.15m even
    # when the vehicle is meters above the ground. Those points are not
    # external obstacles and must not drive horizontal safety.
    self_ignore_radius_m: float = 0.75

    # Sector boundaries in degrees around the horizontal plane.
    # angle = atan2(y_right, x_forward).
    front_half_angle_deg: float = 30.0
    diagonal_inner_deg: float = 30.0
    diagonal_outer_deg: float = 75.0
    side_inner_deg: float = 60.0
    side_outer_deg: float = 135.0


def point_cloud_to_array(point_cloud: Iterable[float]) -> np.ndarray:
    """Convert AirSim flat xyz list into an Nx3 float32 array."""
    arr = np.asarray(list(point_cloud), dtype=np.float32)
    if arr.size < 3:
        return np.empty((0, 3), dtype=np.float32)
    usable = (arr.size // 3) * 3
    if usable <= 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr[:usable].reshape((-1, 3))


class LidarProcessor:
    def __init__(self, config: LidarProcessorConfig | None = None):
        self.config = config or LidarProcessorConfig()

    def compute_sector_distances(
        self,
        points_xyz: np.ndarray,
        altitude_fallback_m: float | None = None,
    ) -> Dict[str, float | int | bool | str]:
        """Return old-distance-sensor-compatible sector distances in meters."""
        cfg = self.config
        max_d = float(cfg.max_range_m)

        result = {
            "front_dist_m": max_d,
            "front_left_dist_m": max_d,
            "front_right_dist_m": max_d,
            "left_dist_m": max_d,
            "right_dist_m": max_d,
            "back_dist_m": max_d,
            "down_dist_m": max_d if altitude_fallback_m is None else float(np.clip(altitude_fallback_m, 0.0, max_d)),
            "min_obstacle_dist_m": max_d,
            "lidar_valid": False,
            "lidar_point_count": 0,
            "obstacle_source": "lidar_empty",
        }

        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            return result

        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            return result

        x = pts[:, 0].astype(np.float32)
        y = pts[:, 1].astype(np.float32)
        z = pts[:, 2].astype(np.float32)

        ranges = np.sqrt(x * x + y * y + z * z)
        valid_range = (ranges >= float(cfg.min_valid_range_m)) & (ranges <= max_d)
        if not np.any(valid_range):
            return result

        x = x[valid_range]
        y = y[valid_range]
        z = z[valid_range]
        ranges = ranges[valid_range]

        point_count = int(ranges.size)
        result["lidar_valid"] = point_count > 0
        result["lidar_point_count"] = point_count
        result["obstacle_source"] = "lidar"

        # -----------------------------
        # Horizontal obstacle sectors
        # -----------------------------
        self_clear_mask = ranges >= float(cfg.self_ignore_radius_m)
        horizontal_mask = (np.abs(z) <= float(cfg.horizontal_abs_z_max_m)) & self_clear_mask
        if np.any(horizontal_mask):
            hx = x[horizontal_mask]
            hy = y[horizontal_mask]
            hr = ranges[horizontal_mask]
            angles = np.degrees(np.arctan2(hy, hx))

            front = np.abs(angles) <= float(cfg.front_half_angle_deg)
            front_right = (angles >= float(cfg.diagonal_inner_deg)) & (angles < float(cfg.diagonal_outer_deg))
            front_left = (angles <= -float(cfg.diagonal_inner_deg)) & (angles > -float(cfg.diagonal_outer_deg))
            right = (angles >= float(cfg.side_inner_deg)) & (angles < float(cfg.side_outer_deg))
            left = (angles <= -float(cfg.side_inner_deg)) & (angles > -float(cfg.side_outer_deg))
            back = np.abs(angles) >= float(cfg.side_outer_deg)

            result["front_dist_m"] = self._sector_min(hr, front, max_d)
            result["front_left_dist_m"] = self._sector_min(hr, front_left, max_d)
            result["front_right_dist_m"] = self._sector_min(hr, front_right, max_d)
            result["left_dist_m"] = self._sector_min(hr, left, max_d)
            result["right_dist_m"] = self._sector_min(hr, right, max_d)
            result["back_dist_m"] = self._sector_min(hr, back, max_d)

        horizontal_min = min(
            float(result["front_dist_m"]),
            float(result["front_left_dist_m"]),
            float(result["front_right_dist_m"]),
            float(result["left_dist_m"]),
            float(result["right_dist_m"]),
            float(result["back_dist_m"]),
        )
        result["min_obstacle_dist_m"] = horizontal_min

        # -----------------------------
        # Down / ground clearance
        # -----------------------------
        xy_radius = np.sqrt(x * x + y * y)
        down_mask = (
            (z > float(cfg.down_min_z_m))
            & (xy_radius <= float(cfg.down_xy_radius_m))
            & (ranges >= float(cfg.self_ignore_radius_m))
        )
        if np.any(down_mask):
            # z is the vertical-down component in SensorLocalFrame.  It is a more
            # stable AGL estimate than full Euclidean range for downward points.
            result["down_dist_m"] = float(np.clip(np.min(z[down_mask]), 0.0, max_d))
        elif altitude_fallback_m is not None and np.isfinite(float(altitude_fallback_m)):
            result["down_dist_m"] = float(np.clip(float(altitude_fallback_m), 0.0, max_d))

        return result

    @staticmethod
    def _sector_min(ranges: np.ndarray, mask: np.ndarray, fallback: float) -> float:
        if not np.any(mask):
            return float(fallback)
        return float(np.clip(np.min(ranges[mask]), 0.0, fallback))
