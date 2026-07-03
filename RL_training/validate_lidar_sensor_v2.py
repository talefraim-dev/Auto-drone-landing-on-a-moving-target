"""
validate_lidar_sensor_v2.py

Checks raw LiDAR and processed sector distances using the same calibrated
LidarProcessor settings as DroneEnv.
"""

from __future__ import annotations

import numpy as np
import time

try:
    import cosysairsim as airsim
except Exception:
    import airsim

from weights_config import EnvConfig
from lidar_processor import LidarProcessor, LidarProcessorConfig, point_cloud_to_array


def main():
    cfg = EnvConfig()
    client = airsim.MultirotorClient()
    client.confirmConnection()

    processor = LidarProcessor(
        LidarProcessorConfig(
            max_range_m=float(cfg.lidar_max_dist_m),
            min_valid_range_m=float(cfg.lidar_min_valid_range_m),
            horizontal_abs_z_max_m=float(cfg.lidar_horizontal_abs_z_max_m),
            down_xy_radius_m=float(cfg.lidar_down_xy_radius_m),
            down_min_z_m=float(cfg.lidar_down_min_z_m),
            self_ignore_radius_m=float(getattr(cfg, "lidar_self_ignore_radius_m", 0.75)),
        )
    )

    print("[LIDAR V2] calibrated processor check")
    print(f"[LIDAR V2] self_ignore_radius_m={getattr(cfg, 'lidar_self_ignore_radius_m', 0.75)}")
    print(f"[LIDAR V2] use_lidar_down_as_altitude={getattr(cfg, 'use_lidar_down_as_altitude', False)}")
    print()

    for i in range(60):
        state = client.getMultirotorState(vehicle_name=cfg.vehicle_name)
        alt = max(0.0, -float(state.kinematics_estimated.position.z_val))

        data = client.getLidarData(lidar_name=cfg.lidar_sensor_name, vehicle_name=cfg.vehicle_name)
        pts = point_cloud_to_array(data.point_cloud)

        if pts.size:
            ranges = np.linalg.norm(pts, axis=1)
            raw_min = float(np.nanmin(ranges))
            raw_p05 = float(np.nanpercentile(ranges, 5))
        else:
            raw_min = float("nan")
            raw_p05 = float("nan")

        sectors = processor.compute_sector_distances(pts, altitude_fallback_m=alt)

        print(
            f"[{i:03d}] alt_pose={alt:.2f} "
            f"raw_points={len(pts)} raw_min={raw_min:.3f} raw_p05={raw_p05:.3f} | "
            f"valid={sectors.get('lidar_valid')} pts={sectors.get('lidar_point_count')} "
            f"F={float(sectors['front_dist_m']):.2f} "
            f"FL={float(sectors['front_left_dist_m']):.2f} "
            f"FR={float(sectors['front_right_dist_m']):.2f} "
            f"L={float(sectors['left_dist_m']):.2f} "
            f"R={float(sectors['right_dist_m']):.2f} "
            f"B={float(sectors['back_dist_m']):.2f} "
            f"D={float(sectors['down_dist_m']):.2f} "
            f"HMIN={float(sectors['min_obstacle_dist_m']):.2f}"
        )
        time.sleep(0.2)


if __name__ == "__main__":
    main()
