"""Calibrated five-sensor downward range-finder array for final UAV landing.

The module keeps the original nine observation features while adding:
- persistent per-sensor bias calibration;
- roll/pitch compensation to obtain vertical surface distance;
- estimated world-Z of the hit surface for diagnostics;
- median-based aggregation for robustness against one bad corner ray.

Sensor names are fixed by the AirSim settings file:
TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, CENTER.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import time

import numpy as np

SENSOR_NAMES = ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT", "CENTER")
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "config" / "range_finder_calibration.json"


@dataclass
class RangeFinderArrayConfig:
    min_distance_m: float = 0.05
    max_distance_m: float = 20.0
    min_valid_count: int = 4
    safe_spread_m: float = 0.25
    sensor_final_entry_m: float = 1.50
    sensor_final_stop_m: float = 0.35
    closing_rate_clip_mps: float = 3.0
    calibration_path: str | None = str(DEFAULT_CALIBRATION_PATH)
    enable_tilt_compensation: bool = True
    min_vertical_cosine: float = 0.50


class RangeFinderArray:
    FEATURE_NAMES = [
        "range_top_left_norm", "range_top_right_norm",
        "range_bottom_left_norm", "range_bottom_right_norm",
        "range_center_norm", "range_mean_norm", "range_spread_norm",
        "range_valid_ratio", "range_closing_rate_norm",
    ]

    def __init__(self, config: RangeFinderArrayConfig | None = None):
        self.config = config or RangeFinderArrayConfig()
        self.sensor_bias_m = {name: 0.0 for name in SENSOR_NAMES}
        self.mount_positions_m = {name: (0.0, 0.0, 0.60) for name in SENSOR_NAMES}
        self.calibration_loaded = False
        self.calibration_metadata: dict[str, Any] = {}
        self._load_calibration()
        self.reset()

    def _load_calibration(self) -> None:
        path_value = self.config.calibration_path
        if not path_value:
            return
        path = Path(path_value)
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validation_passed = bool(
                payload.get("calibration_status") == "PASS"
                or payload.get("validation", {}).get("overall_pass", False)
                or payload.get("calibration_test", {}).get("passed", False)
            )
            if not validation_passed:
                raise ValueError("Range-finder calibration file is not marked PASS")
            bias = payload.get("sensor_bias_m", {})
            mounts = payload.get("sensor_mount_positions_m", {})
            for name in SENSOR_NAMES:
                if name in bias and math.isfinite(float(bias[name])):
                    self.sensor_bias_m[name] = float(bias[name])
                if name in mounts and len(mounts[name]) == 3:
                    self.mount_positions_m[name] = tuple(float(v) for v in mounts[name])
            self.calibration_metadata = payload
            self.calibration_loaded = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.calibration_loaded = False

    def reset(self) -> None:
        self._previous_median_m: float | None = None
        self._previous_time: float | None = None

    @staticmethod
    def _quaternion_to_rotation_matrix(q: Any) -> np.ndarray:
        w = float(getattr(q, "w_val", 1.0))
        x = float(getattr(q, "x_val", 0.0))
        y = float(getattr(q, "y_val", 0.0))
        z = float(getattr(q, "z_val", 0.0))
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm <= 1e-9:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        return np.asarray([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _vehicle_pose(self, client: Any, vehicle_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            state = client.getMultirotorState(vehicle_name=vehicle_name)
            kin = state.kinematics_estimated
            p = kin.position
            position = np.asarray([float(p.x_val), float(p.y_val), float(p.z_val)], dtype=np.float64)
            rotation = self._quaternion_to_rotation_matrix(kin.orientation)
            return position, rotation
        except Exception:
            return None

    def read(self, client: Any, vehicle_name: str) -> dict[str, Any]:
        now = time.monotonic()
        pose = self._vehicle_pose(client, vehicle_name)
        vertical_cosine = 1.0
        if pose is not None:
            _, rotation = pose
            body_down_world = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            vertical_cosine = abs(float(body_down_world[2]))
        vertical_cosine = float(np.clip(vertical_cosine, self.config.min_vertical_cosine, 1.0))

        raw_distances: dict[str, float] = {}
        distances: dict[str, float] = {}
        vertical_distances: dict[str, float] = {}
        surface_world_z: dict[str, float] = {}
        valid: dict[str, bool] = {}

        for name in SENSOR_NAMES:
            raw = float("inf")
            try:
                data = client.getDistanceSensorData(distance_sensor_name=name, vehicle_name=vehicle_name)
                raw = float(getattr(data, "distance", float("inf")))
            except Exception:
                raw = float("inf")

            corrected = raw - self.sensor_bias_m.get(name, 0.0) if math.isfinite(raw) else float("inf")
            ok = bool(math.isfinite(corrected) and self.config.min_distance_m <= corrected <= self.config.max_distance_m)
            raw_distances[name] = raw
            distances[name] = corrected if ok else float("inf")
            valid[name] = ok

            vertical = corrected * vertical_cosine if ok and self.config.enable_tilt_compensation else corrected
            vertical_distances[name] = vertical if ok else float("inf")

            world_z = float("inf")
            if ok and pose is not None:
                vehicle_position, rotation = pose
                mount_body = np.asarray(self.mount_positions_m.get(name, (0.0, 0.0, 0.60)), dtype=np.float64)
                sensor_world = vehicle_position + rotation @ mount_body
                world_z = float(sensor_world[2] + vertical)
            surface_world_z[name] = world_z

        values = [vertical_distances[n] for n in SENSOR_NAMES if valid[n]]
        valid_count = len(values)
        median_m = float(np.median(values)) if values else float("inf")
        mean_m = float(np.mean(values)) if values else float("inf")
        spread_m = float(max(values) - min(values)) if len(values) >= 2 else float("inf")

        world_z_values = [surface_world_z[n] for n in SENSOR_NAMES if math.isfinite(surface_world_z[n])]
        surface_world_z_median = float(np.median(world_z_values)) if world_z_values else float("inf")
        surface_world_z_spread = float(max(world_z_values) - min(world_z_values)) if len(world_z_values) >= 2 else float("inf")

        closing_rate = 0.0
        if self._previous_median_m is not None and self._previous_time is not None and math.isfinite(median_m):
            dt = max(1e-3, now - self._previous_time)
            closing_rate = float((self._previous_median_m - median_m) / dt)
        if math.isfinite(median_m):
            self._previous_median_m = median_m
            self._previous_time = now

        height_reliable = bool(
            self.calibration_loaded
            and valid_count >= self.config.min_valid_count
            and math.isfinite(median_m)
            and math.isfinite(spread_m)
            and spread_m <= self.config.safe_spread_m
            and vertical_cosine >= self.config.min_vertical_cosine
        )
        ready = bool(
            height_reliable
            and self.config.sensor_final_stop_m < median_m <= self.config.sensor_final_entry_m
        )

        result: dict[str, Any] = {
            "range_raw_distances_m": raw_distances,
            "range_distances_m": distances,
            "range_vertical_distances_m": vertical_distances,
            "range_surface_world_z_m": surface_world_z,
            "range_surface_world_z_median": surface_world_z_median,
            "range_surface_world_z_spread": surface_world_z_spread,
            "range_valid": valid,
            "range_valid_count": valid_count,
            "range_valid_ratio": valid_count / len(SENSOR_NAMES),
            "range_mean_m": median_m,
            "range_arithmetic_mean_m": mean_m,
            "range_spread_m": spread_m,
            "range_closing_rate_mps": closing_rate,
            "range_vertical_cosine": vertical_cosine,
            "range_calibration_loaded": self.calibration_loaded,
            "range_height_reliable": height_reliable,
            "range_sensor_final_ready": ready,
        }
        result["features"] = self.features(result)
        return result

    def features(self, result: dict[str, Any]) -> np.ndarray:
        cfg = self.config
        distances = result["range_vertical_distances_m"]

        def norm_distance(value: float) -> float:
            if not math.isfinite(value):
                return 1.0
            return float(np.clip(value / cfg.max_distance_m, 0.0, 1.0))

        median_m = float(result["range_mean_m"])
        spread_m = float(result["range_spread_m"])
        closing = float(result["range_closing_rate_mps"])
        values = [norm_distance(distances[n]) for n in SENSOR_NAMES]
        values += [
            norm_distance(median_m),
            1.0 if not math.isfinite(spread_m) else float(np.clip(spread_m / cfg.safe_spread_m, 0, 1)),
            float(result["range_valid_ratio"]),
            float(np.clip(closing / cfg.closing_rate_clip_mps, -1, 1)),
        ]
        return np.asarray(values, dtype=np.float32)

    def sensor_final_descent_speed(self, mean_m: float) -> float:
        # Measurements below 0.35 m are outside the experimentally verified
        # reliable envelope. The environment handles that region with a short,
        # bounded terminal-contact descent rather than interpreting the new
        # sensor value as height.
        if not math.isfinite(mean_m) or mean_m <= self.config.sensor_final_stop_m:
            return 0.0
        if mean_m <= 0.45:
            return 0.14
        if mean_m <= 0.80:
            return 0.24
        return 0.35
