"""Scale-independent identity-stability reward for alternating co-training."""

from __future__ import annotations

from typing import Any

import numpy as np

# Identity-stability shaping. RPC remains overwhelmingly dominant. The shaping
# is deliberately scale-independent: it rewards accepted LIVE identity and
# temporal continuity, while penalizing prediction/stale/lost states and only
# non-physical *changes* in bbox scale.
IDENTITY_LIVE_BASE_REWARD = 0.25
IDENTITY_STREAK_MAX_BONUS = 0.50
IDENTITY_STREAK_FULL_STEPS = 20
IDENTITY_PREDICTION_PENALTY = 0.10
IDENTITY_STALE_PENALTY = 0.25
IDENTITY_LOST_PENALTY = 0.75
IDENTITY_SCALE_JUMP_PENALTY = 0.50
IDENTITY_EPISODE_CAP = 400.0
AGENT2_IDENTITY_REWARD_SCALE = 0.15
AGENT2_DESCENDING_SCALE_JUMP_PENALTY = 0.50
BBOX_SCALE_RATIO_MAX = 2.50
BBOX_SCALE_RATIO_MIN = 0.40


def _bbox_area_from_info(info: dict[str, Any]) -> float | None:
    """Return a finite positive bbox area from the most reliable exposed bbox."""
    for key in ("bottom_bbox_control_xyxy", "bottom_bbox_raw_xyxy", "bbox_xyxy"):
        raw = info.get(key)
        if raw is None:
            continue
        try:
            box = np.asarray(raw, dtype=np.float64).reshape(-1)
            if box.size < 4 or not np.all(np.isfinite(box[:4])):
                continue
            width = max(0.0, float(box[2] - box[0]))
            height = max(0.0, float(box[3] - box[1]))
            area = width * height
            if area > 1.0:
                return area
        except (TypeError, ValueError):
            continue
    return None


class IdentityStabilityReward:
    """Stateful, bounded identity reward shared by both co-training phases.

    The reward never depends on absolute bbox size. A close target may correctly
    fill the frame. Only a sudden per-step area ratio outside the physical guard
    is penalized. The episode accumulator is clipped so hovering cannot compete
    with the terminal RPC signal.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.live_streak = 0
        self.episode_total = 0.0
        self.previous_bbox_area: float | None = None

    @staticmethod
    def _classify(info: dict[str, Any]) -> tuple[str, bool]:
        bottom_live = bool(
            info.get("bottom_match_live", info.get("bottom_guidance_live", False))
        )
        bottom_confirmed = bool(
            info.get("bottom_match_confirmed", bottom_live)
        )
        prediction_only = bool(
            info.get("bottom_guidance_prediction_only", False)
        )
        tracker_mode = str(
            info.get("tracker_mode", info.get("agent1_tracking_mode", ""))
        ).upper()

        if bottom_live and bottom_confirmed:
            return "LIVE", True
        if prediction_only or tracker_mode.startswith("PRED"):
            return "PRED", False
        if bool(info.get("bottom_match_recent", False)) or tracker_mode == "STALE":
            return "STALE", False
        # During the front-camera chase phase, a genuine MATCH is also a valid
        # identity observation even before bottom handoff.
        if tracker_mode == "MATCH" and bool(info.get("fusion_has_target", True)):
            return "LIVE", True
        return "LOST", False

    def step(self, info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        state, live = self._classify(info)
        if live:
            self.live_streak += 1
            streak_fraction = min(
                float(self.live_streak) / float(IDENTITY_STREAK_FULL_STEPS), 1.0
            )
            raw_reward = (
                IDENTITY_LIVE_BASE_REWARD
                + IDENTITY_STREAK_MAX_BONUS * streak_fraction
            )
        else:
            self.live_streak = 0
            if state == "PRED":
                raw_reward = -IDENTITY_PREDICTION_PENALTY
            elif state == "STALE":
                raw_reward = -IDENTITY_STALE_PENALTY
            else:
                raw_reward = -IDENTITY_LOST_PENALTY

        area = _bbox_area_from_info(info)
        scale_ratio = 1.0
        scale_jump = False
        # Compare only consecutive LIVE accepted boxes. A reacquisition after
        # PRED/STALE/LOST may legitimately have a very different scale.
        if live and area is not None and self.previous_bbox_area is not None:
            scale_ratio = area / max(self.previous_bbox_area, 1.0)
            scale_jump = bool(
                scale_ratio > BBOX_SCALE_RATIO_MAX
                or scale_ratio < BBOX_SCALE_RATIO_MIN
            )
            if scale_jump:
                raw_reward -= IDENTITY_SCALE_JUMP_PENALTY
        self.previous_bbox_area = area if (live and area is not None) else None

        before = self.episode_total
        after = float(np.clip(
            before + raw_reward, -IDENTITY_EPISODE_CAP, IDENTITY_EPISODE_CAP
        ))
        applied = after - before
        self.episode_total = after

        diagnostics = {
            "identity_state": state,
            "identity_live_streak": int(self.live_streak),
            "identity_bbox_scale_ratio": float(scale_ratio),
            "identity_bbox_scale_jump": bool(scale_jump),
            "identity_reward_raw": float(raw_reward),
            "identity_reward_applied": float(applied),
            "identity_episode_total": float(self.episode_total),
            "identity_episode_cap": float(IDENTITY_EPISODE_CAP),
        }
        return float(applied), diagnostics

