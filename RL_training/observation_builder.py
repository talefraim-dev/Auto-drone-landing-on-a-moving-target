"""
Observation Builder for UAV RL agents — v37.

Observation size:
- Base OBS: 28 features
- Obstacle OBS: 9 features
- Total: 37 features

Important design decision:
- min_obstacle_dist_m / collision_risk_score are based on horizontal sensors only.
- down_dist_m is kept separately for vertical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class ObservationBuilderConfig:
    image_width: int = 960
    image_height: int = 540

    max_img_velocity: float = 2.0
    max_img_acceleration: float = 5.0

    max_altitude_m: float = 30.0
    max_drone_speed_mps: float = 8.0
    max_vertical_speed_mps: float = 5.0
    max_roll_rad: float = math.radians(35.0)
    max_pitch_rad: float = math.radians(35.0)
    max_yaw_rate_radps: float = math.radians(180.0)

    max_obstacle_range_m: float = 20.0
    safe_obstacle_distance_m: float = 5.0

    max_lost_target_time_s: float = 2.0

    eps: float = 1e-6


@dataclass
class BBox:
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0


@dataclass
class DroneState:
    altitude_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    roll_rad: float
    pitch_rad: float
    yaw_rate_radps: float


@dataclass
class ObstacleState:
    front_dist_m: float
    front_left_dist_m: float
    front_right_dist_m: float
    left_dist_m: float
    right_dist_m: float
    back_dist_m: float
    down_dist_m: float
    min_obstacle_dist_m: Optional[float] = None

    @property
    def horizontal_min_obstacle_dist_m(self) -> float:
        if self.min_obstacle_dist_m is not None:
            return float(self.min_obstacle_dist_m)

        return min(
            self.front_dist_m,
            self.front_left_dist_m,
            self.front_right_dist_m,
            self.left_dist_m,
            self.right_dist_m,
            self.back_dist_m,
        )


class ObservationBuilder:
    FEATURE_NAMES: List[str] = [
        "has_target",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "bbox_conf",
        "err_x",
        "err_y",
        "img_vx",
        "img_vy",
        "img_ax",
        "img_ay",
        "area_delta_norm",
        "distance_proxy_norm",
        "distance_proxy_delta",
        "altitude_norm",
        "drone_vx_norm",
        "drone_vy_norm",
        "drone_vz_norm",
        "drone_roll_norm",
        "drone_pitch_norm",
        "drone_yaw_rate_norm",
        "drone_speed_xy_norm",
        "centered_score",
        "target_stability_score",
        "landing_allowed",
        "lost_target_time_norm",
        "front_dist_norm",
        "front_left_dist_norm",
        "front_right_dist_norm",
        "left_dist_norm",
        "right_dist_norm",
        "back_dist_norm",
        "down_dist_norm",
        "min_obstacle_dist_norm",
        "collision_risk_score",
    ]

    def __init__(self, config: Optional[ObservationBuilderConfig] = None):
        self.config = config or ObservationBuilderConfig()
        self.reset()

    def reset(self) -> None:
        self.prev_bbox_cx: Optional[float] = None
        self.prev_bbox_cy: Optional[float] = None
        self.prev_bbox_area: Optional[float] = None

        self.prev_img_vx: float = 0.0
        self.prev_img_vy: float = 0.0
        self.prev_distance_proxy: Optional[float] = None

        self.prev_err_x: float = 0.0
        self.prev_err_y: float = 0.0
        self.lost_target_time_s: float = 0.0

    def build(
        self,
        bbox: Optional[BBox],
        drone_state: DroneState,
        obstacle_state: ObstacleState,
        dt: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        cfg = self.config
        dt = max(float(dt), cfg.eps)

        if bbox is None or bbox.conf <= 0.0 or bbox.w <= 0.0 or bbox.h <= 0.0:
            vision = self._build_lost_target_vision(dt)
        else:
            vision = self._build_detected_target_vision(bbox, dt)

        drone = self._build_drone_state(drone_state)
        obstacle = self._build_obstacle_state(obstacle_state)

        values = [
            vision["has_target"],
            vision["bbox_cx"],
            vision["bbox_cy"],
            vision["bbox_w"],
            vision["bbox_h"],
            vision["bbox_area"],
            vision["bbox_conf"],
            vision["err_x"],
            vision["err_y"],
            vision["img_vx"],
            vision["img_vy"],
            vision["img_ax"],
            vision["img_ay"],
            vision["area_delta_norm"],
            vision["distance_proxy_norm"],
            vision["distance_proxy_delta"],
            drone["altitude_norm"],
            drone["drone_vx_norm"],
            drone["drone_vy_norm"],
            drone["drone_vz_norm"],
            drone["drone_roll_norm"],
            drone["drone_pitch_norm"],
            drone["drone_yaw_rate_norm"],
            drone["drone_speed_xy_norm"],
            vision["centered_score"],
            vision["target_stability_score"],
            vision["landing_allowed"],
            vision["lost_target_time_norm"],
            obstacle["front_dist_norm"],
            obstacle["front_left_dist_norm"],
            obstacle["front_right_dist_norm"],
            obstacle["left_dist_norm"],
            obstacle["right_dist_norm"],
            obstacle["back_dist_norm"],
            obstacle["down_dist_norm"],
            obstacle["min_obstacle_dist_norm"],
            obstacle["collision_risk_score"],
        ]

        obs_vector = np.asarray(values, dtype=np.float32)
        obs_dict = dict(zip(self.FEATURE_NAMES, map(float, obs_vector)))

        return obs_vector, obs_dict

    def _build_detected_target_vision(self, bbox: BBox, dt: float) -> Dict[str, float]:
        cfg = self.config

        bbox_cx = self._clip01(bbox.cx / cfg.image_width)
        bbox_cy = self._clip01(bbox.cy / cfg.image_height)
        bbox_w = self._clip01(bbox.w / cfg.image_width)
        bbox_h = self._clip01(bbox.h / cfg.image_height)
        bbox_area = self._clip01(bbox_w * bbox_h)
        bbox_conf = self._clip01(bbox.conf)

        err_x = self._clip(2.0 * (bbox_cx - 0.5), -1.0, 1.0)
        err_y = self._clip(2.0 * (bbox_cy - 0.5), -1.0, 1.0)

        if self.prev_bbox_cx is None:
            img_vx = img_vy = img_ax = img_ay = 0.0
            area_delta_norm = 0.0
            distance_proxy_delta = 0.0
        else:
            img_vx = self._clip(
                (bbox_cx - self.prev_bbox_cx) / dt,
                -cfg.max_img_velocity,
                cfg.max_img_velocity,
            )
            img_vy = self._clip(
                (bbox_cy - self.prev_bbox_cy) / dt,
                -cfg.max_img_velocity,
                cfg.max_img_velocity,
            )

            img_ax = self._clip(
                (img_vx - self.prev_img_vx) / dt,
                -cfg.max_img_acceleration,
                cfg.max_img_acceleration,
            )
            img_ay = self._clip(
                (img_vy - self.prev_img_vy) / dt,
                -cfg.max_img_acceleration,
                cfg.max_img_acceleration,
            )

            area_delta_norm = (bbox_area - self.prev_bbox_area) / (self.prev_bbox_area + cfg.eps)
            area_delta_norm = self._clip(area_delta_norm, -1.0, 1.0)

        distance_proxy_norm = self._clip01(1.0 - math.sqrt(max(bbox_area, 0.0)))

        if self.prev_distance_proxy is None:
            distance_proxy_delta = 0.0
        else:
            distance_proxy_delta = self._clip(
                distance_proxy_norm - self.prev_distance_proxy,
                -1.0,
                1.0,
            )

        center_error = math.sqrt(err_x * err_x + err_y * err_y)
        centered_score = 1.0 - self._clip(center_error / math.sqrt(2.0), 0.0, 1.0)

        target_stability_score = self._compute_target_stability(
            img_vx=img_vx,
            img_vy=img_vy,
            img_ax=img_ax,
            img_ay=img_ay,
            area_delta_norm=area_delta_norm,
        )

        landing_allowed = 1.0 if (
            centered_score > 0.85 and target_stability_score > 0.75
        ) else 0.0

        self.prev_bbox_cx = bbox_cx
        self.prev_bbox_cy = bbox_cy
        self.prev_bbox_area = bbox_area
        self.prev_img_vx = img_vx
        self.prev_img_vy = img_vy
        self.prev_distance_proxy = distance_proxy_norm
        self.prev_err_x = err_x
        self.prev_err_y = err_y
        self.lost_target_time_s = 0.0

        return {
            "has_target": 1.0,
            "bbox_cx": bbox_cx,
            "bbox_cy": bbox_cy,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_area": bbox_area,
            "bbox_conf": bbox_conf,
            "err_x": err_x,
            "err_y": err_y,
            "img_vx": img_vx,
            "img_vy": img_vy,
            "img_ax": img_ax,
            "img_ay": img_ay,
            "area_delta_norm": area_delta_norm,
            "distance_proxy_norm": distance_proxy_norm,
            "distance_proxy_delta": distance_proxy_delta,
            "centered_score": centered_score,
            "target_stability_score": target_stability_score,
            "landing_allowed": landing_allowed,
            "lost_target_time_norm": 0.0,
        }

    def _build_lost_target_vision(self, dt: float) -> Dict[str, float]:
        cfg = self.config
        self.lost_target_time_s += dt
        lost_target_time_norm = self._clip01(
            self.lost_target_time_s / cfg.max_lost_target_time_s
        )

        return {
            "has_target": 0.0,
            "bbox_cx": 0.5,
            "bbox_cy": 0.5,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
            "bbox_area": 0.0,
            "bbox_conf": 0.0,
            "err_x": self.prev_err_x,
            "err_y": self.prev_err_y,
            "img_vx": 0.0,
            "img_vy": 0.0,
            "img_ax": 0.0,
            "img_ay": 0.0,
            "area_delta_norm": 0.0,
            "distance_proxy_norm": 1.0,
            "distance_proxy_delta": 0.0,
            "centered_score": 0.0,
            "target_stability_score": 0.0,
            "landing_allowed": 0.0,
            "lost_target_time_norm": lost_target_time_norm,
        }

    def _build_drone_state(self, state: DroneState) -> Dict[str, float]:
        cfg = self.config
        speed_xy = math.sqrt(state.vx_mps ** 2 + state.vy_mps ** 2)

        return {
            "altitude_norm": self._clip01(state.altitude_m / cfg.max_altitude_m),
            "drone_vx_norm": self._clip(state.vx_mps / cfg.max_drone_speed_mps, -1.0, 1.0),
            "drone_vy_norm": self._clip(state.vy_mps / cfg.max_drone_speed_mps, -1.0, 1.0),
            "drone_vz_norm": self._clip(state.vz_mps / cfg.max_vertical_speed_mps, -1.0, 1.0),
            "drone_roll_norm": self._clip(state.roll_rad / cfg.max_roll_rad, -1.0, 1.0),
            "drone_pitch_norm": self._clip(state.pitch_rad / cfg.max_pitch_rad, -1.0, 1.0),
            "drone_yaw_rate_norm": self._clip(state.yaw_rate_radps / cfg.max_yaw_rate_radps, -1.0, 1.0),
            "drone_speed_xy_norm": self._clip01(speed_xy / cfg.max_drone_speed_mps),
        }

    def _build_obstacle_state(self, state: ObstacleState) -> Dict[str, float]:
        cfg = self.config

        horizontal_min = state.horizontal_min_obstacle_dist_m
        collision_risk = compute_collision_risk(
            min_obstacle_dist_m=horizontal_min,
            safe_distance_m=cfg.safe_obstacle_distance_m,
        )

        return {
            "front_dist_norm": self._normalize_dist(state.front_dist_m),
            "front_left_dist_norm": self._normalize_dist(state.front_left_dist_m),
            "front_right_dist_norm": self._normalize_dist(state.front_right_dist_m),
            "left_dist_norm": self._normalize_dist(state.left_dist_m),
            "right_dist_norm": self._normalize_dist(state.right_dist_m),
            "back_dist_norm": self._normalize_dist(state.back_dist_m),
            "down_dist_norm": self._normalize_dist(state.down_dist_m),
            "min_obstacle_dist_norm": self._normalize_dist(horizontal_min),
            "collision_risk_score": collision_risk,
        }

    def _compute_target_stability(
        self,
        img_vx: float,
        img_vy: float,
        img_ax: float,
        img_ay: float,
        area_delta_norm: float,
    ) -> float:
        motion_mag = math.sqrt(img_vx * img_vx + img_vy * img_vy)
        acc_mag = math.sqrt(img_ax * img_ax + img_ay * img_ay)
        area_motion = abs(area_delta_norm)

        raw_instability = (
            0.5 * min(motion_mag / self.config.max_img_velocity, 1.0)
            + 0.3 * min(acc_mag / self.config.max_img_acceleration, 1.0)
            + 0.2 * min(area_motion, 1.0)
        )

        return 1.0 - self._clip01(raw_instability)

    def _normalize_dist(self, dist_m: float) -> float:
        return self._clip01(dist_m / self.config.max_obstacle_range_m)

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, float(value)))

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


def compute_collision_risk(min_obstacle_dist_m: float, safe_distance_m: float) -> float:
    if safe_distance_m <= 0:
        raise ValueError("safe_distance_m must be positive.")

    risk = 1.0 - min(max(min_obstacle_dist_m, 0.0) / safe_distance_m, 1.0)
    return max(0.0, min(1.0, risk))
