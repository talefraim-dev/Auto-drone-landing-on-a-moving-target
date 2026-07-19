"""
validate_lidar_sensor.py

Run this before training after adding the LiDAR to Cosys-AirSim settings.json.
It prints the same sector distances that DroneEnv will feed to the agent.

Run from inside RL_training/:
    python validate_lidar_sensor.py
"""

from __future__ import annotations

import time
import numpy as np
import cosysairsim as airsim

from weights_config import EnvConfig
from lidar_processor import LidarProcessor, LidarProcessorConfig, point_cloud_to_array


def main() -> None:
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
        )
    )

    print("=" * 100)
    print("[LIDAR VALIDATION]")
    print(f"vehicle_name      : {cfg.vehicle_name}")
    print(f"lidar_sensor_name : {cfg.lidar_sensor_name}")
    print("Expected DataFrame: SensorLocalFrame")
    print("Press Ctrl+C to stop.")
    print("=" * 100)

    while True:
        try:
            data = client.getLidarData(
                lidar_name=str(cfg.lidar_sensor_name),
                vehicle_name=str(cfg.vehicle_name),
            )
            pts = point_cloud_to_array(getattr(data, "point_cloud", []))

            try:
                ms = client.getMultirotorState(vehicle_name=str(cfg.vehicle_name))
                alt = max(0.0, float(-ms.kinematics_estimated.position.z_val))
            except Exception:
                alt = None

            sectors = processor.compute_sector_distances(pts, altitude_fallback_m=alt)
            print(
                "[LIDAR] "
                f"valid={sectors['lidar_valid']} "
                f"points={sectors['lidar_point_count']} "
                f"F={sectors['front_dist_m']:.2f} "
                f"FL={sectors['front_left_dist_m']:.2f} "
                f"FR={sectors['front_right_dist_m']:.2f} "
                f"L={sectors['left_dist_m']:.2f} "
                f"R={sectors['right_dist_m']:.2f} "
                f"B={sectors['back_dist_m']:.2f} "
                f"D={sectors['down_dist_m']:.2f} "
                f"HMIN={sectors['min_obstacle_dist_m']:.2f}"
            )
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[LIDAR VALIDATION] stopped")
            break
        except Exception as e:
            print(f"[LIDAR VALIDATION] error: {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
