from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RewardStats:
    total: float = 0.0
    center: float = 0.0
    area: float = 0.0
    focus: float = 0.0
    pred: float = 0.0
    no_bbox: float = 0.0
    calm_search: float = 0.0
    obstacle: float = 0.0
    energy: float = 0.0
    smooth: float = 0.0
    vertical: float = 0.0
    range_term: float = 0.0
    progress: float = 0.0
    altitude: float = 0.0
    bearing: float = 0.0
    time: float = 0.0
    timeout: float = 0.0
    collision: float = 0.0
    low_alt: float = 0.0
    too_close: float = 0.0
    stuck: float = 0.0
    ground_contact: float = 0.0


class RewardManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats = RewardStats()
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_center_error: Optional[float] = None
        self.prev_range_error: Optional[float] = None

    def reset_episode(self):
        self.stats = RewardStats()
        self.prev_action[:] = 0.0
        self.prev_center_error = None
        self.prev_range_error = None

    def _reward_energy(self, action: np.ndarray) -> float:
        return -float(self.cfg.energy_penalty_k) * float(np.abs(action).sum())

    def _reward_smooth(self, action: np.ndarray) -> float:
        da = np.abs(action - self.prev_action)
        return -float(self.cfg.w_smooth_delta) * float(da.sum())

    def _reward_time(self, dt: float) -> float:
        return -float(self.cfg.w_time_penalty) * float(dt)

    def _reward_center(self, center_error: float) -> float:
        return float(self.cfg.w_center) * float(np.exp(-center_error * float(self.cfg.center_decay)))

    def _reward_area(self, area01: float) -> float:
        return float(self.cfg.w_area) * float(np.sqrt(np.clip(area01, 0.0, 1.0)))

    def _reward_focus(self, focus_streak: float, is_match: bool, is_pred: bool, dt: float) -> Dict[str, float]:
        out = {"focus": 0.0, "pred": 0.0}
        if is_match:
            out["focus"] += float(self.cfg.match_warmup_reward)
        elif is_pred:
            out["focus"] += float(self.cfg.pred_warmup_reward)

        out["focus"] += float(self.cfg.w_focus) * float(min(2.0, focus_streak / 4.0))
        if is_pred:
            out["pred"] -= float(self.cfg.pred_penalty_per_sec) * float(dt)
        return out

    def _reward_no_bbox(self, action: np.ndarray, dt: float) -> Dict[str, float]:
        vx_n = float(np.clip(abs(action[0]), 0.0, 1.0))
        vy_n = float(np.clip(abs(action[1]), 0.0, 1.0))
        yaw_n = float(np.clip(abs(action[3]), 0.0, 1.0))
        calm = 1.0 - (0.45 * vx_n + 0.45 * vy_n + 0.10 * yaw_n)
        calm = float(np.clip(calm, 0.0, 1.0))
        return {
            "no_bbox": -float(self.cfg.penalty_no_bbox),
            "calm_search": float(self.cfg.w_calm_no_bbox) * calm * float(dt),
        }

    def _reward_obstacle(self, min_obst_m: Optional[float], dt: float) -> float:
        if (not self.cfg.use_obstacle_penalty) or min_obst_m is None:
            return 0.0
        d = float(min_obst_m)
        if d <= 0.0:
            return 0.0
        r = 0.0
        if d < float(self.cfg.obstacle_safe_dist_m):
            r -= float(self.cfg.obstacle_penalty_k) * (float(self.cfg.obstacle_safe_dist_m) - d) * float(dt)
        if d < float(self.cfg.obstacle_danger_dist_m):
            r -= float(self.cfg.obstacle_danger_penalty_k) * float(dt)
        return r

    def _reward_vertical(self, vz_cmd: float, dt: float) -> float:
        return -float(self.cfg.w_vz) * abs(float(vz_cmd)) * float(dt)

    def _reward_range(self, range_error: float) -> float:
        return -float(self.cfg.w_range_error) * abs(float(range_error))

    def _reward_progress(self, center_error: float, range_error: float) -> float:
        reward = 0.0
        if self.prev_center_error is not None:
            reward += float(self.cfg.w_progress) * (float(self.prev_center_error) - float(center_error))
        if self.prev_range_error is not None:
            reward += 0.5 * float(self.cfg.w_progress) * (
                abs(float(self.prev_range_error)) - abs(float(range_error))
            )
        return reward

    def _reward_altitude(self, alt_error: float) -> float:
        return -float(self.cfg.w_altitude_error) * abs(float(alt_error))

    def _reward_bearing(self, bearing_error: float) -> float:
        return -float(self.cfg.w_bearing_error) * abs(float(bearing_error))

    def _reward_low_alt(self, alt_agl_m: Optional[float]) -> float:
        if (not getattr(self.cfg, "use_low_alt_penalty", False)) or alt_agl_m is None:
            return 0.0
        deficit = float(self.cfg.min_follow_alt_m) - float(alt_agl_m)
        if deficit <= 0.0:
            return 0.0
        return -float(self.cfg.w_low_alt) * deficit

    def _reward_too_close(self, range_to_target: float, desired_range: float) -> float:
        if not getattr(self.cfg, "use_too_close_penalty", False):
            return 0.0
        threshold = float(desired_range) - float(self.cfg.close_margin_norm)
        if float(range_to_target) >= threshold:
            return 0.0
        return -float(self.cfg.w_too_close) * (threshold - float(range_to_target))

    def _reward_stuck_follow(self, stuck_triggered: bool) -> float:
        if (not getattr(self.cfg, "use_stuck_penalty", False)) or (not stuck_triggered):
            return 0.0
        return -float(self.cfg.penalty_stuck_follow)

    def _accumulate(self, terms: Dict[str, float]) -> float:
        total = 0.0
        for k, v in terms.items():
            fv = float(v)
            total += fv
            if hasattr(self.stats, k):
                setattr(self.stats, k, getattr(self.stats, k) + fv)
        self.stats.total += total
        return total

    def compute_no_bbox_reward(
        self,
        *,
        action: np.ndarray,
        dt: float,
        vz_cmd: float,
        min_obst_m: Optional[float],
        focus_timeout: bool,
        pred_timeout: bool,
        collision: bool,
        alt_agl_m: Optional[float] = None,
        ground_contact: bool = False,
    ) -> float:
        terms: Dict[str, float] = {}
        terms.update(self._reward_no_bbox(action, dt))
        terms["energy"] = self._reward_energy(action)
        terms["smooth"] = self._reward_smooth(action)
        terms["vertical"] = self._reward_vertical(vz_cmd, dt)
        terms["obstacle"] = self._reward_obstacle(min_obst_m, dt)
        terms["time"] = self._reward_time(dt)
        terms["low_alt"] = self._reward_low_alt(alt_agl_m)
        if ground_contact:
            terms["ground_contact"] = -float(self.cfg.penalty_ground_contact)
        if focus_timeout:
            terms["timeout"] = -float(self.cfg.penalty_focus_timeout)
        if pred_timeout:
            terms["timeout"] = terms.get("timeout", 0.0) - float(self.cfg.penalty_pred_focus_timeout)
        if collision:
            terms["collision"] = terms.get("collision", 0.0) - float(self.cfg.penalty_collision)
        total = self._accumulate(terms)
        self.prev_action = action.copy()
        self.prev_center_error = None
        self.prev_range_error = None
        return total

    def compute_tracking_reward(
        self,
        *,
        action: np.ndarray,
        dt: float,
        vz_cmd: float,
        area01: float,
        center_error: float,
        focus_streak: float,
        is_match: bool,
        is_pred: bool,
        range_error: float,
        range_to_target: float,
        desired_range: float,
        alt_error: float,
        bearing_error: float,
        min_obst_m: Optional[float],
        focus_timeout: bool,
        pred_timeout: bool,
        collision: bool,
        stuck_triggered: bool = False,
        alt_agl_m: Optional[float] = None,
        ground_contact: bool = False,
    ) -> float:
        terms: Dict[str, float] = {}
        terms["center"] = self._reward_center(center_error)
        terms["area"] = self._reward_area(area01)
        terms.update(self._reward_focus(focus_streak, is_match, is_pred, dt))
        terms["range_term"] = self._reward_range(range_error)
        terms["too_close"] = self._reward_too_close(range_to_target, desired_range)
        terms["progress"] = self._reward_progress(center_error, range_error)
        terms["altitude"] = self._reward_altitude(alt_error)
        terms["low_alt"] = self._reward_low_alt(alt_agl_m)
        terms["bearing"] = self._reward_bearing(bearing_error)
        terms["obstacle"] = self._reward_obstacle(min_obst_m, dt)
        terms["energy"] = self._reward_energy(action)
        terms["smooth"] = self._reward_smooth(action)
        terms["vertical"] = self._reward_vertical(vz_cmd, dt)
        terms["time"] = self._reward_time(dt)
        terms["stuck"] = self._reward_stuck_follow(stuck_triggered)

        if ground_contact:
            terms["ground_contact"] = -float(self.cfg.penalty_ground_contact)
        if focus_timeout:
            terms["timeout"] = -float(self.cfg.penalty_focus_timeout)
        if pred_timeout:
            terms["timeout"] = terms.get("timeout", 0.0) - float(self.cfg.penalty_pred_focus_timeout)
        if collision:
            terms["collision"] = terms.get("collision", 0.0) - float(self.cfg.penalty_collision)

        total = self._accumulate(terms)
        self.prev_action = action.copy()
        self.prev_center_error = float(center_error)
        self.prev_range_error = float(range_error)
        return total
