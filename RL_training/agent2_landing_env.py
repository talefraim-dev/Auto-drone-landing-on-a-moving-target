"""Independent Agent-2 landing environment.

This module intentionally does not subclass or mutate ``DroneEnv``. Agent 1
keeps its original configuration, initialized reward objects, safety objects,
tracker memory and control pipeline. Agent 2 owns a separate control loop that
uses:

* bottom camera only;
* immutable user-selected visual identity;
* horizontal LiDAR sectors only;
* AirSim API Z for all vertical state;
* AirSim collision API as the touchdown signal;
* a collision-gated landing reward bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
import cosysairsim as airsim

from lidar_processor import LidarProcessor, LidarProcessorConfig, point_cloud_to_array
from observation_builder import (
    BBox,
    DroneState,
    ObstacleState,
    ObservationBuilder,
    ObservationBuilderConfig,
)
from resnet_yolo_tracker import YoloResNetTracker


@dataclass
class Agent2Config:
    vehicle_name: str = "Drone1"
    bottom_camera_name: str = "bottom_center"
    lidar_sensor_name: str = "LidarSensor1"
    image_width: int = 960
    image_height: int = 720

    cmd_duration_s: float = 0.10
    vx_scale_mps: float = 1.20
    vy_scale_mps: float = 1.20
    vz_scale_mps: float = 0.65
    yaw_scale_dps: float = 25.0
    max_episode_steps: int = 900

    # YOLO class is used only during the first click so a bbox can be obtained.
    # After that, the selected instance is internally named ``user_target`` and
    # every YOLO candidate is judged only by ResNet visual similarity.
    strict_class_gate: bool = False
    yolo_proposal_conf: float = 0.05
    # ResNet remains the identity judge. These gates only decide whether a
    # visual match is reliable enough to be treated as a LIVE control input.
    min_match_similarity: float = 0.60
    min_match_margin: float = 0.03
    high_conf_reacquire_similarity: float = 0.78
    max_reacquire_center_jump_norm: float = 0.35
    match_confirmation_steps: int = 3
    max_prediction_steps: int = 20
    show_camera: bool = True

    # Vertical state machine. AirSim NED uses positive vz for DOWN.
    # Agent 2 is a landing-only controller: policy commands may request hover
    # or descent, but never climb. Descent is additionally blocked unless the
    # target is detected in the current frame, confirmed over consecutive
    # frames and sufficiently aligned.
    descent_min_live_match_streak: int = 3
    descent_min_similarity: float = 0.65

    # Horizontal visual-servo controller. PPO does not directly command the
    # full XY velocity anymore; it learns only a small residual around a
    # deterministic bottom-camera PD controller.
    horizontal_pd_kp_y_to_vx: float = 1.10
    horizontal_pd_kd_y_to_vx: float = 0.16
    horizontal_pd_kp_x_to_vy: float = 0.82
    horizontal_pd_kd_x_to_vy: float = 0.12
    horizontal_pd_max_action: float = 0.90
    horizontal_ppo_residual_max_action: float = 0.12
    horizontal_deadband_error: float = 0.025
    horizontal_deadband_velocity: float = 0.08
    horizontal_velocity_ema_alpha: float = 0.35
    horizontal_velocity_clip_per_s: float = 4.0

    # Moving-target velocity feed-forward. The actor pose is sampled read-only
    # from AirSim, converted from world NED XY into the drone body frame, and
    # added to the visual centering correction. This does not change the PPO
    # observation or action spaces.
    target_velocity_feedforward_enabled: bool = True
    target_velocity_feedforward_gain: float = 1.0
    target_velocity_ema_alpha: float = 0.35
    target_velocity_max_valid_mps: float = 8.0
    target_velocity_stale_after_s: float = 1.0
    horizontal_total_speed_max_mps: float = 6.0
    horizontal_velocity_hold_max_s: float = 1.0

    # Speed shrinks near touchdown. The final limit also considers bbox area,
    # so a wrong actor-Z estimate cannot make close-range commands aggressive.
    horizontal_speed_far_mps: float = 0.90
    horizontal_speed_mid_mps: float = 0.60
    horizontal_speed_near_mps: float = 0.35
    horizontal_speed_touchdown_mps: float = 0.20

    # Alignment hysteresis: enter DESCEND only after several strongly aligned
    # frames, and immediately return to RECENTER when the looser exit limits
    # are exceeded.
    alignment_enter_center_error: float = 0.20
    alignment_enter_bbox_rel_error: float = 0.35
    alignment_exit_center_error: float = 0.32
    alignment_exit_bbox_rel_error: float = 0.58
    alignment_streak_required: int = 5

    lidar_max_range_m: float = 20.0
    obstacle_emergency_m: float = 0.55
    obstacle_warning_m: float = 1.20

    # Touchdown reward classification uses only evidence from the collision
    # frame. A target collision is successful when either a centered LIVE
    # bottom-camera match exists, or the current bottom image itself looks like
    # the selected target directly under the camera. Historical latches remain
    # diagnostic only and cannot grant terminal reward.
    good_collision_center_error: float = 0.45
    good_collision_bbox_rel_error: float = 0.75
    good_collision_recent_match_steps: int = 4
    collision_latch_max_age_steps: int = 6
    collision_latch_center_error: float = 0.10
    collision_latch_bbox_rel_error: float = 0.15
    collision_latch_min_similarity: float = 0.65

    # Direct contact-frame appearance test. Several centered crops are compared
    # with the immutable bottom-view target anchor. Corner crops act as a
    # current-frame floor/background control. A high absolute similarity or a
    # clear center-over-corners margin is required.
    contact_appearance_center_crop_scales: tuple[float, ...] = (0.35, 0.50, 0.70, 0.90, 1.00)
    contact_appearance_corner_crop_scale: float = 0.35
    contact_appearance_min_similarity: float = 0.60
    contact_appearance_strong_similarity: float = 0.72
    contact_appearance_min_center_margin: float = 0.03

    # Authorized-descent state is retained only for diagnostics. It no longer
    # participates in touchdown success classification.
    authorized_descent_min_vz_mps: float = 1.0e-4
    authorized_descent_extreme_live_center_error: float = 0.55

    # The first verified, aligned, non-ground touchdown auto-locks the AirSim
    # collision object name (for example Porsche_BP_C_1). Later rewards require
    # the same object. This prevents Floor_0/terrain contacts from becoming
    # successes while avoiding a hard-coded Unreal actor name.
    collision_object_auto_lock: bool = True
    collision_ground_tokens: tuple[str, ...] = ("floor", "ground", "landscape", "terrain")

    success_base_reward: float = 2500.0
    success_min_reward: float = 800.0
    success_max_reward: float = 6000.0
    wrong_collision_penalty: float = 1500.0

    target_lost_limit_steps: int = 120

    # In AGENT_1P2, a sustained absence of a LIVE bottom-camera match requests
    # a same-episode return to Agent 1. A short grace window handles detector
    # gaps; near contact, direct bottom-image appearance can suppress a false
    # recovery request when the target fills the frame.
    recovery_no_live_match_timeout_s: float = 1.0
    recovery_no_live_match_min_steps: int = 2
    recovery_contact_guard_max_height_m: float = 0.75
    # Contact appearance may postpone recovery only briefly. It must never
    # create a low-altitude hover deadlock after the target has really escaped.
    recovery_contact_guard_max_no_live_s: float = 1.50
    # In AGENT_1P2 v12, Agent 2 never owns XY/Yaw and never requests a
    # recovery handoff. It continuously owns only Z while Agent 1 remains
    # active on both cameras.
    parallel_dual_agent_mode: bool = False

    static_target_actor_name: str = ""
    static_target_surface_altitude_m: float = 0.0


class Agent2LandingEnv(gym.Env):
    """Bottom-camera landing controller used by AGENT_2 and AGENT_1P2."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Agent2Config | None = None):
        super().__init__()
        self.cfg = cfg or Agent2Config()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(37,), dtype=np.float32)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.observation_builder = ObservationBuilder(
            ObservationBuilderConfig(
                image_width=self.cfg.image_width,
                image_height=self.cfg.image_height,
                max_altitude_m=30.0,
                max_drone_speed_mps=8.0,
                max_vertical_speed_mps=5.0,
                max_obstacle_range_m=self.cfg.lidar_max_range_m,
                safe_obstacle_distance_m=5.0,
                max_lost_target_time_s=3.0,
            )
        )
        self.lidar_processor = LidarProcessor(
            LidarProcessorConfig(max_range_m=self.cfg.lidar_max_range_m)
        )
        self.tracker = YoloResNetTracker(
            yolo_model_path="yolo11s.pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            target_classes=None,
            yolo_conf=float(self.cfg.yolo_proposal_conf),
            click_pad=20,
            min_match_score=0.45,
            appearance_weight=1.0,
            motion_weight=0.0,
            search_window_scale=4.0,
            use_search_window=False,
            ema_alpha=1.0,
            verbose=False,
        )

        self._original_embedding: Optional[torch.Tensor] = None
        self._bottom_anchor_embedding: Optional[torch.Tensor] = None
        self._reference_embeddings: list[torch.Tensor] = []
        self._target_id = "user_target"
        self._target_class_id: Optional[int] = None
        self._last_bbox_xyxy: Optional[np.ndarray] = None
        self._last_similarity = 0.0
        self._last_candidate_class_id: Optional[int] = None
        self._last_candidate_confidence = 0.0
        self._last_candidate_count = 0
        self._last_candidate_scores: list[dict[str, float | int]] = []
        self._prediction_steps = 0
        self._last_match_step = -999999
        self._live_match_streak = 0
        self._last_match_margin = 0.0
        self._last_spatial_jump_norm = 0.0
        self._last_match_reject_reason = ""

        self._standalone_initial_pose = None
        self._attached_from_agent1 = False
        self._target_actor_name = str(self.cfg.static_target_actor_name or "")
        # Keep target and drone in the same raw AirSim NED-Z coordinate
        # system. Height above target is target_z_ned - drone_z_ned.
        self._target_surface_z_ned = -float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_altitude_m = float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_source = "configured_static"

        self._step = 0
        self._reward_bank = 0.0
        self._episode_return = 0.0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_relative_height_m: Optional[float] = None
        self._lost_steps = 0
        self._collision_timestamp_at_reset = 0

        # Strict visual-alignment latch used only for touchdown classification.
        # It is reset every episode; the expected collision object may persist
        # after being safely auto-locked from a verified target touchdown.
        self._last_verified_alignment_step = -999999
        self._last_verified_alignment_center_error = 999.0
        self._last_verified_alignment_bbox_rel_error = 999.0
        self._last_verified_alignment_similarity = 0.0

        # Physical descent authorization evidence. Unlike the visual latch,
        # this is refreshed only when a non-zero descent command is actually
        # sent after passing the full vertical safety gate.
        self._authorized_descent_latched = False
        self._last_authorized_descent_step = -999999
        self._last_authorized_descent_center_error = 999.0
        self._last_authorized_descent_bbox_rel_error = 999.0
        self._last_authorized_descent_similarity = 0.0
        self._last_authorized_descent_vz_mps = 0.0
        self._authorized_descent_invalidated_reason = "never_authorized"

        self._expected_collision_object_name = ""
        self._expected_collision_object_source = "unlocked"

        # Most recent raw bottom-camera frame. It is used only when AirSim
        # reports a new collision, so contact appearance adds no per-step
        # inference cost.
        self._last_bottom_frame: Optional[np.ndarray] = None

        self._last_info: dict[str, Any] = {}
        self._last_vertical_control_state = "HOLD_INIT"
        self._last_descent_block_reason = "waiting_for_first_control_step"
        self._last_raw_vz_action = 0.0
        self._last_requested_vz_mps = 0.0
        self._last_applied_vz_mps = 0.0
        self._last_climb_command_blocked = False

        self._last_observation_monotonic: Optional[float] = None
        self._last_control_had_live_match = False
        self._last_control_err_x = 0.0
        self._last_control_err_y = 0.0
        self._control_img_vel_x = 0.0
        self._control_img_vel_y = 0.0
        self._target_pose_xy: Optional[tuple[float, float]] = None
        self._target_pose_monotonic: Optional[float] = None
        self._target_velocity_world_x_mps = 0.0
        self._target_velocity_world_y_mps = 0.0
        self._target_velocity_body_vx_mps = 0.0
        self._target_velocity_body_vy_mps = 0.0
        self._target_velocity_speed_mps = 0.0
        self._target_velocity_valid = False
        self._target_velocity_age_s = float("inf")
        self._non_live_started_monotonic: Optional[float] = None
        self._non_live_duration_s = 0.0
        self._alignment_ready_streak = 0
        self._descent_alignment_latched = False
        self._last_horizontal_control_state = "HOLD_INIT"
        self._last_pd_action_vx = 0.0
        self._last_pd_action_vy = 0.0
        self._last_residual_action_vx = 0.0
        self._last_residual_action_vy = 0.0
        self._last_horizontal_action_vx = 0.0
        self._last_horizontal_action_vy = 0.0
        self._last_horizontal_speed_limit_mps = 0.0
        self._last_velocity_ff_vx_mps = 0.0
        self._last_velocity_ff_vy_mps = 0.0
        self._last_horizontal_correction_vx_mps = 0.0
        self._last_horizontal_correction_vy_mps = 0.0
        self._last_horizontal_command_vx_mps = 0.0
        self._last_horizontal_command_vy_mps = 0.0
        self._last_horizontal_total_speed_limit_mps = 0.0
        self._episode_recenter_steps = 0
        self._episode_xy_hold_steps = 0

        self._episode_live_match_steps = 0
        self._episode_predicted_steps = 0
        self._episode_no_target_steps = 0
        self._episode_descent_requested_steps = 0
        self._episode_descent_allowed_steps = 0
        self._episode_descent_blocked_steps = 0
        self._episode_climb_command_blocked_steps = 0
        self._episode_best_center_error = float("inf")
        self._episode_best_similarity = 0.0

        # Optional single-command executor supplied by Agent1P2Env. When set,
        # Agent 2 computes only the gated Z command; the executor runs frozen
        # Agent 1 and transmits the fused XY/Yaw/Z command exactly once.
        self._external_command_executor: Optional[
            Callable[[float], dict[str, Any]]
        ] = None
        self._last_external_command_info: dict[str, Any] = {}

    def set_external_command_executor(
        self,
        executor: Optional[Callable[[float], dict[str, Any]]],
    ) -> None:
        self._external_command_executor = executor

    def get_target_velocity_feedforward_body(self) -> tuple[float, float, bool]:
        """Return the current filtered target velocity in drone body axes."""
        valid = bool(
            self.cfg.target_velocity_feedforward_enabled
            and self._target_velocity_valid
            and self._target_velocity_age_s
            <= float(self.cfg.target_velocity_stale_after_s)
        )
        if not valid:
            return 0.0, 0.0, False
        gain = float(self.cfg.target_velocity_feedforward_gain)
        return (
            float(self._target_velocity_body_vx_mps) * gain,
            float(self._target_velocity_body_vy_mps) * gain,
            True,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def _set_identity(self, fingerprint: Any, class_id: Any = None) -> None:
        arr = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("Agent 2 received an empty target fingerprint.")

        emb = torch.from_numpy(arr).float().to(self.tracker.device)
        emb = F.normalize(emb, dim=0)
        self._original_embedding = emb.detach().clone()
        self._bottom_anchor_embedding = None
        self._reference_embeddings = [self._original_embedding.detach().clone()]
        self._target_class_id = None if class_id is None else int(class_id)

        # The internal tracker is used only as a detector/embedding backend.
        # target_class_id is kept only for diagnostics; it is never a gate.
        self.tracker.target_embedding = self._original_embedding.detach().clone()
        self.tracker.target_class_id = self._target_class_id
        self.tracker.last_bbox = None
        self.tracker.last_good_bbox = None
        self.tracker.last_score = 0.0
        self.tracker.last_mode = "IDLE"

    def _set_bottom_anchor_from_bbox(self, frame: np.ndarray, bbox_xyxy: Any) -> bool:
        """Create an immutable bottom-view reference for the same user target."""
        arr = self._validated_xyxy(bbox_xyxy, frame.shape)
        if arr is None:
            return False

        # ResNet crop coordinates must be integer pixel indices. Agent 1 stores
        # handoff boxes as float32, so normalize them at this ownership boundary.
        bbox_xywh = self._xyxy_to_xywh(arr)
        emb = self.tracker._embedding_from_bbox(frame, bbox_xywh)
        if emb is None:
            return False

        self._bottom_anchor_embedding = emb.detach().clone()
        self._reference_embeddings = [self._original_embedding.detach().clone()]
        self._reference_embeddings.append(self._bottom_anchor_embedding.detach().clone())
        return True

    @staticmethod
    def _validated_xyxy(bbox_xyxy: Any, frame_shape: tuple[int, ...]) -> Optional[np.ndarray]:
        if bbox_xyxy is None:
            return None
        try:
            arr = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            return None

        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in arr[:4]]
        x1 = float(np.clip(x1, 0.0, max(0.0, float(w - 1))))
        y1 = float(np.clip(y1, 0.0, max(0.0, float(h - 1))))
        x2 = float(np.clip(x2, x1 + 1.0, max(x1 + 1.0, float(w))))
        y2 = float(np.clip(y2, y1 + 1.0, max(y1 + 1.0, float(h))))
        if x2 <= x1 or y2 <= y1:
            return None
        return np.asarray([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _extract_agent1_handoff_bbox(agent1_env: Any) -> Optional[np.ndarray]:
        """Read the verified bottom bbox left by Agent 1 without mutating Agent 1."""
        for name in (
            "_bottom_stable_bbox_xyxy",
            "_bottom_bbox_xyxy",
            "_bottom_last_trusted_raw_bbox_xyxy",
        ):
            value = getattr(agent1_env, name, None)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
                return arr[:4].copy()
        return None

    def _sync_image_geometry(self, frame: np.ndarray) -> None:
        """Keep observation normalization aligned with the real AirSim frame."""
        h, w = frame.shape[:2]
        if int(self.cfg.image_width) == int(w) and int(self.cfg.image_height) == int(h):
            return
        self.cfg.image_width = int(w)
        self.cfg.image_height = int(h)
        self.observation_builder.config.image_width = int(w)
        self.observation_builder.config.image_height = int(h)

    def _select_target_from_bottom_click(self, frame: np.ndarray) -> None:
        window = "Agent 2 - click static target in BOTTOM camera"
        selected: dict[str, Any] = {"done": False, "x": 0, "y": 0}

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected.update(done=True, x=int(x), y=int(y))

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        shown = frame.copy()
        cv2.putText(
            shown,
            "Click the landing target (bottom camera)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        while not selected["done"]:
            cv2.imshow(window, shown)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(window)
                raise RuntimeError("Agent-2 target selection was cancelled.")

        bbox = self.tracker.select_target(frame, selected["x"], selected["y"])
        cv2.destroyWindow(window)
        if bbox is None or self.tracker.target_embedding is None:
            raise RuntimeError("Bottom click did not produce a target bbox and fingerprint.")

        fp = self.tracker.target_embedding.detach().cpu().numpy().copy()
        self._set_identity(fp, self.tracker.target_class_id)
        self._last_bbox_xyxy = self._xywh_to_xyxy(bbox)
        self._set_bottom_anchor_from_bbox(frame, self._last_bbox_xyxy)
        print(
            f"[AGENT_2] Static target selected: target_id={self._target_id} "
            f"initial_yolo_class={self._target_class_id} "
            f"bbox={self._last_bbox_xyxy.astype(int).tolist()}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def attach_from_agent1(self, agent1_env: Any, handoff_info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Attach Agent 2 to the exact physical state left by Agent 1."""
        self.client = agent1_env.client
        self.cfg.vehicle_name = str(agent1_env.cfg.vehicle_name)
        self.cfg.bottom_camera_name = str(getattr(agent1_env.cfg, "downward_camera_name", "bottom_center"))
        self.cfg.lidar_sensor_name = str(getattr(agent1_env.cfg, "lidar_sensor_name", "LidarSensor1"))

        fingerprint = getattr(agent1_env, "target_fingerprint", None)
        class_id = getattr(agent1_env, "target_class_id", None)
        self._set_identity(fingerprint, class_id)

        handoff_bbox = self._extract_agent1_handoff_bbox(agent1_env)

        self._target_actor_name = str(getattr(agent1_env, "train_target_car", "") or "")
        self._read_target_surface_altitude()
        self._attached_from_agent1 = True
        self._reset_runtime_state()

        try:
            self.client.moveByVelocityBodyFrameAsync(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                duration=0.10,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
        except Exception:
            pass

        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        validated_handoff_bbox = self._validated_xyxy(handoff_bbox, frame.shape)
        if validated_handoff_bbox is not None:
            self._last_bbox_xyxy = validated_handoff_bbox.copy()
            self._last_match_step = int(self._step)
            self._last_similarity = 1.0
            self.tracker.last_bbox = self._xyxy_to_xywh(validated_handoff_bbox)
            self.tracker.last_good_bbox = self.tracker.last_bbox.copy()
            self.tracker.last_mode = "HANDOFF_INIT"
            self._set_bottom_anchor_from_bbox(frame, validated_handoff_bbox)

        obs, info = self._observe(frame=frame)
        info.update(
            {
                "agent": "AGENT_2",
                "attached_from_agent1": True,
                "agent1_handoff_reason": str(handoff_info.get("termination_reason", "handoff_success")),
                "handoff_bbox_transferred": validated_handoff_bbox is not None,
                "bottom_anchor_created": self._bottom_anchor_embedding is not None,
            }
        )
        print(
            "[AGENT_1P2] Landing Z authority enabled; Agent 1 remains active | "
            f"target_id={self._target_id} initial_yolo_class={self._target_class_id} "
            f"bbox_transferred={int(validated_handoff_bbox is not None)} "
            f"bottom_anchor={int(self._bottom_anchor_embedding is not None)} "
            f"targetZ_NED={self._target_surface_z_ned:+.3f}m "
            f"source={self._target_surface_source}"
        )
        return obs, info

    def reattach_after_recovery(
        self,
        agent1_env: Any,
        handoff_info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return control to Agent 2 after Agent 1 reacquires the same target.

        The physical world, target identity, PPO episode counters, reward bank,
        and expected collision object are preserved. Only short-term landing
        perception/control state is cleared so stale PRED/Kalman data cannot
        leak across the recovery boundary.
        """
        saved_progress = {
            "step": int(self._step),
            "reward_bank": float(self._reward_bank),
            "episode_return": float(self._episode_return),
            "collision_timestamp": int(self._collision_timestamp_at_reset),
            "live_steps": int(self._episode_live_match_steps),
            "pred_steps": int(self._episode_predicted_steps),
            "none_steps": int(self._episode_no_target_steps),
            "descent_requested": int(self._episode_descent_requested_steps),
            "descent_allowed": int(self._episode_descent_allowed_steps),
            "descent_blocked": int(self._episode_descent_blocked_steps),
            "climb_blocked": int(self._episode_climb_command_blocked_steps),
            "best_center": float(self._episode_best_center_error),
            "best_similarity": float(self._episode_best_similarity),
            "recenter_steps": int(self._episode_recenter_steps),
            "xy_hold_steps": int(self._episode_xy_hold_steps),
        }

        self.client = agent1_env.client
        self.cfg.vehicle_name = str(agent1_env.cfg.vehicle_name)
        self.cfg.bottom_camera_name = str(
            getattr(agent1_env.cfg, "downward_camera_name", "bottom_center")
        )
        self.cfg.lidar_sensor_name = str(
            getattr(agent1_env.cfg, "lidar_sensor_name", "LidarSensor1")
        )

        if self._original_embedding is None:
            self._set_identity(
                getattr(agent1_env, "target_fingerprint", None),
                getattr(agent1_env, "target_class_id", None),
            )

        handoff_bbox = self._extract_agent1_handoff_bbox(agent1_env)
        self._target_actor_name = str(getattr(agent1_env, "train_target_car", "") or "")
        self._attached_from_agent1 = True
        self._reset_runtime_state()

        self._step = saved_progress["step"]
        self._reward_bank = saved_progress["reward_bank"]
        self._episode_return = saved_progress["episode_return"]
        self._collision_timestamp_at_reset = saved_progress["collision_timestamp"]
        self._episode_live_match_steps = saved_progress["live_steps"]
        self._episode_predicted_steps = saved_progress["pred_steps"]
        self._episode_no_target_steps = saved_progress["none_steps"]
        self._episode_descent_requested_steps = saved_progress["descent_requested"]
        self._episode_descent_allowed_steps = saved_progress["descent_allowed"]
        self._episode_descent_blocked_steps = saved_progress["descent_blocked"]
        self._episode_climb_command_blocked_steps = saved_progress["climb_blocked"]
        self._episode_best_center_error = saved_progress["best_center"]
        self._episode_best_similarity = saved_progress["best_similarity"]
        self._episode_recenter_steps = saved_progress["recenter_steps"]
        self._episode_xy_hold_steps = saved_progress["xy_hold_steps"]

        try:
            self.client.moveByVelocityBodyFrameAsync(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                duration=0.10,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
        except Exception:
            pass

        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        validated_handoff_bbox = self._validated_xyxy(handoff_bbox, frame.shape)
        if validated_handoff_bbox is not None:
            self._last_bbox_xyxy = validated_handoff_bbox.copy()
            self._last_match_step = int(self._step)
            self._last_similarity = 1.0
            self.tracker.last_bbox = self._xyxy_to_xywh(validated_handoff_bbox)
            self.tracker.last_good_bbox = self.tracker.last_bbox.copy()
            self.tracker.last_mode = "RECOVERY_HANDOFF_INIT"
            if self._bottom_anchor_embedding is None:
                self._set_bottom_anchor_from_bbox(frame, validated_handoff_bbox)

        self._read_target_surface_altitude()
        obs, info = self._observe(frame=frame)
        info.update(
            {
                "agent": "AGENT_2",
                "reattached_after_recovery": True,
                "agent1_recovery_handoff_reason": str(
                    handoff_info.get("termination_reason", "handoff_success")
                ),
                "handoff_bbox_transferred": validated_handoff_bbox is not None,
                "bottom_anchor_preserved": self._bottom_anchor_embedding is not None,
            }
        )
        print(
            "[AGENT_1P2] Agent 1 RECOVERY OUT -> Agent 2 BACK IN | "
            f"agent2_step={self._step} bbox_transferred="
            f"{int(validated_handoff_bbox is not None)} "
            f"targetSpeed={self._target_velocity_speed_mps:.2f}m/s"
        )
        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._attached_from_agent1 = False
        self.client.enableApiControl(True, vehicle_name=self.cfg.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.cfg.vehicle_name)

        if self._standalone_initial_pose is None:
            self._standalone_initial_pose = self.client.simGetVehiclePose(vehicle_name=self.cfg.vehicle_name)
        else:
            self.client.simSetVehiclePose(
                self._standalone_initial_pose,
                True,
                vehicle_name=self.cfg.vehicle_name,
            )
            self.client.moveByVelocityBodyFrameAsync(
                0.0,
                0.0,
                0.0,
                0.15,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0.0),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
            time.sleep(0.20)

        self._reset_runtime_state()
        self._read_target_surface_altitude()
        frame = self._get_bottom_frame()
        self._sync_image_geometry(frame)
        if self._original_embedding is None:
            self._select_target_from_bottom_click(frame)

        obs, info = self._observe(frame=frame)
        info.update({"agent": "AGENT_2", "standalone": True, "target_actor_moved": False})
        print(
            "[AGENT_2 RESET] target actor untouched | "
            f"targetZ_NED={self._target_surface_z_ned:+.3f}m source={self._target_surface_source}"
        )
        return obs, info

    def _reset_runtime_state(self) -> None:
        self.observation_builder.reset()
        self._step = 0
        self._reward_bank = 0.0
        self._episode_return = 0.0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_relative_height_m = None
        self._lost_steps = 0
        self._prediction_steps = 0
        self._last_match_step = -999999
        self._last_verified_alignment_step = -999999
        self._last_verified_alignment_center_error = 999.0
        self._last_verified_alignment_bbox_rel_error = 999.0
        self._last_verified_alignment_similarity = 0.0
        self._authorized_descent_latched = False
        self._last_authorized_descent_step = -999999
        self._last_authorized_descent_center_error = 999.0
        self._last_authorized_descent_bbox_rel_error = 999.0
        self._last_authorized_descent_similarity = 0.0
        self._last_authorized_descent_vz_mps = 0.0
        self._authorized_descent_invalidated_reason = "never_authorized"
        self._live_match_streak = 0
        self._last_match_margin = 0.0
        self._last_spatial_jump_norm = 0.0
        self._last_match_reject_reason = ""
        self._last_similarity = 0.0
        self._last_candidate_class_id = None
        self._last_candidate_confidence = 0.0
        self._last_candidate_count = 0
        self._last_candidate_scores = []
        self._last_bbox_xyxy = None
        self._last_bottom_frame = None
        self._last_info = {}
        self._last_vertical_control_state = "HOLD_INIT"
        self._last_descent_block_reason = "waiting_for_first_control_step"
        self._last_raw_vz_action = 0.0
        self._last_requested_vz_mps = 0.0
        self._last_applied_vz_mps = 0.0
        self._last_climb_command_blocked = False
        self._last_observation_monotonic = None
        self._last_control_had_live_match = False
        self._last_control_err_x = 0.0
        self._last_control_err_y = 0.0
        self._control_img_vel_x = 0.0
        self._control_img_vel_y = 0.0
        self._reset_target_motion_estimator()
        self._non_live_started_monotonic = None
        self._non_live_duration_s = 0.0
        self._alignment_ready_streak = 0
        self._descent_alignment_latched = False
        self._last_horizontal_control_state = "HOLD_INIT"
        self._last_pd_action_vx = 0.0
        self._last_pd_action_vy = 0.0
        self._last_residual_action_vx = 0.0
        self._last_residual_action_vy = 0.0
        self._last_horizontal_action_vx = 0.0
        self._last_horizontal_action_vy = 0.0
        self._last_horizontal_speed_limit_mps = 0.0
        self._last_velocity_ff_vx_mps = 0.0
        self._last_velocity_ff_vy_mps = 0.0
        self._last_horizontal_correction_vx_mps = 0.0
        self._last_horizontal_correction_vy_mps = 0.0
        self._last_horizontal_command_vx_mps = 0.0
        self._last_horizontal_command_vy_mps = 0.0
        self._last_horizontal_total_speed_limit_mps = 0.0
        self._episode_recenter_steps = 0
        self._episode_xy_hold_steps = 0
        self._episode_live_match_steps = 0
        self._episode_predicted_steps = 0
        self._episode_no_target_steps = 0
        self._episode_descent_requested_steps = 0
        self._episode_descent_allowed_steps = 0
        self._episode_descent_blocked_steps = 0
        self._episode_climb_command_blocked_steps = 0
        self._episode_best_center_error = float("inf")
        self._episode_best_similarity = 0.0
        try:
            collision = self.client.simGetCollisionInfo(vehicle_name=self.cfg.vehicle_name)
            self._collision_timestamp_at_reset = int(getattr(collision, "time_stamp", 0) or 0)
        except Exception:
            self._collision_timestamp_at_reset = 0

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy(bbox: Any) -> np.ndarray:
        x, y, w, h = [float(v) for v in bbox[:4]]
        return np.asarray([x, y, x + max(1.0, w), y + max(1.0, h)], dtype=np.float32)

    @staticmethod
    def _xyxy_to_xywh(bbox: Any) -> list[int]:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x = int(round(x1))
        y = int(round(y1))
        w = max(1, int(round(x2 - x1)))
        h = max(1, int(round(y2 - y1)))
        return [x, y, w, h]

    def _get_bottom_frame(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.cfg.bottom_camera_name, airsim.ImageType.Scene, False, False)],
            vehicle_name=self.cfg.vehicle_name,
        )
        if not responses or not responses[0].image_data_uint8:
            return np.zeros((self.cfg.image_height, self.cfg.image_width, 3), dtype=np.uint8)
        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        frame = img.reshape(response.height, response.width, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _strict_track(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], float, str]:
        """Track ``user_target`` using ResNet identity only.

        YOLO is a proposal generator. Its class ID and confidence are recorded
        for diagnostics, but neither participates in target acceptance after
        the initial user click/handoff.
        """
        references = list(self._reference_embeddings)
        if not references and self._original_embedding is not None:
            references = [self._original_embedding]
        if not references:
            return None, 0.0, "NO_IDENTITY"

        candidates = self.tracker._detect_candidates(frame)
        self._last_candidate_count = int(len(candidates))
        self._last_candidate_scores = []

        if not candidates:
            self._last_candidate_class_id = None
            self._last_candidate_confidence = 0.0
            self._prediction_steps += 1
            self._live_match_streak = 0
            self._last_match_margin = 0.0
            self._last_spatial_jump_norm = 0.0
            self._last_match_reject_reason = "no_detection"
            if self._last_bbox_xyxy is not None and self._prediction_steps <= self.cfg.max_prediction_steps:
                return self._last_bbox_xyxy.copy(), float(self._last_similarity), "PRED_NO_DETECTION"
            return None, 0.0, "NO_DETECTION"

        best = None
        best_similarity = -1.0
        best_anchor_index = -1
        scored_candidates: list[tuple[float, Any, int]] = []

        for candidate_index, candidate in enumerate(candidates):
            emb = self.tracker._embedding_from_bbox(frame, candidate.bbox)
            if emb is None:
                self._last_candidate_scores.append(
                    {
                        "candidate_index": int(candidate_index),
                        "yolo_class_id": int(candidate.cls_id),
                        "yolo_confidence": float(candidate.conf),
                        "resnet_similarity": -1.0,
                        "best_anchor_index": -1,
                    }
                )
                continue

            anchor_scores = [
                float(torch.dot(reference, emb).detach().cpu().item())
                for reference in references
            ]
            candidate_similarity = max(anchor_scores)
            candidate_anchor_index = int(np.argmax(anchor_scores))
            self._last_candidate_scores.append(
                {
                    "candidate_index": int(candidate_index),
                    "yolo_class_id": int(candidate.cls_id),
                    "yolo_confidence": float(candidate.conf),
                    "resnet_similarity": float(candidate_similarity),
                    "best_anchor_index": int(candidate_anchor_index),
                }
            )

            if candidate_similarity > best_similarity:
                best_similarity = float(candidate_similarity)
                best = candidate
                best_anchor_index = candidate_anchor_index

            scored_candidates.append(
                (float(candidate_similarity), candidate, int(candidate_anchor_index))
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        second_best_similarity = (
            float(scored_candidates[1][0]) if len(scored_candidates) > 1 else -1.0
        )
        match_margin = (
            float(best_similarity - second_best_similarity)
            if second_best_similarity >= -0.5
            else 1.0
        )

        spatial_jump_norm = 0.0
        spatial_ok = True
        if best is not None and self._last_bbox_xyxy is not None:
            current_xyxy = self._xywh_to_xyxy(best.bbox)
            h, w = frame.shape[:2]
            prev_cx = 0.5 * float(self._last_bbox_xyxy[0] + self._last_bbox_xyxy[2])
            prev_cy = 0.5 * float(self._last_bbox_xyxy[1] + self._last_bbox_xyxy[3])
            curr_cx = 0.5 * float(current_xyxy[0] + current_xyxy[2])
            curr_cy = 0.5 * float(current_xyxy[1] + current_xyxy[3])
            spatial_jump_norm = float(
                math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                / max(1.0, math.hypot(float(w), float(h)))
            )
            spatial_ok = bool(
                spatial_jump_norm <= float(self.cfg.max_reacquire_center_jump_norm)
                or best_similarity >= float(self.cfg.high_conf_reacquire_similarity)
            )

        similarity_ok = bool(best is not None and best_similarity >= self.cfg.min_match_similarity)
        margin_ok = bool(
            best is not None
            and (
                len(scored_candidates) <= 1
                or match_margin >= float(self.cfg.min_match_margin)
                or best_similarity >= float(self.cfg.high_conf_reacquire_similarity)
            )
        )

        self._last_match_margin = float(match_margin)
        self._last_spatial_jump_norm = float(spatial_jump_norm)

        if best is None or not similarity_ok or not margin_ok or not spatial_ok:
            self._last_candidate_class_id = None if best is None else int(best.cls_id)
            self._last_candidate_confidence = 0.0 if best is None else float(best.conf)
            self._last_similarity = max(0.0, float(best_similarity))
            self._prediction_steps += 1
            self._live_match_streak = 0
            if best is None:
                self._last_match_reject_reason = "no_embedded_candidate"
            elif not similarity_ok:
                self._last_match_reject_reason = "low_similarity"
            elif not margin_ok:
                self._last_match_reject_reason = "ambiguous_similarity_margin"
            else:
                self._last_match_reject_reason = "implausible_spatial_jump"
            if self._last_bbox_xyxy is not None and self._prediction_steps <= self.cfg.max_prediction_steps:
                return self._last_bbox_xyxy.copy(), max(0.0, best_similarity), "PRED_REJECTED_MATCH"
            return None, max(0.0, best_similarity), "REJECTED_MATCH"

        self._prediction_steps = 0
        self._live_match_streak += 1
        self._last_match_reject_reason = ""
        self._last_bbox_xyxy = self._xywh_to_xyxy(best.bbox)
        self._last_similarity = float(best_similarity)
        self._last_match_step = int(self._step)
        self._last_candidate_class_id = int(best.cls_id)
        self._last_candidate_confidence = float(best.conf)

        # Keep detector metadata for diagnostics only. The identity remains
        # ``user_target`` and the immutable reference embeddings never change.
        self.tracker.last_bbox = list(best.bbox)
        self.tracker.last_good_bbox = list(best.bbox)
        self.tracker.last_score = float(best_similarity)
        self.tracker.last_mode = f"MATCH_USER_TARGET_ANCHOR_{best_anchor_index}"
        return self._last_bbox_xyxy.copy(), float(best_similarity), "MATCH"

    def _get_api_state(self) -> tuple[DroneState, float, Any]:
        state = self.client.getMultirotorState(vehicle_name=self.cfg.vehicle_name)
        k = state.kinematics_estimated
        drone_z_ned = float(k.position.z_val)
        altitude = max(0.0, -drone_z_ned)
        velocity = k.linear_velocity
        angular = k.angular_velocity
        roll = 0.0
        pitch = 0.0
        try:
            pitch, roll, _yaw = airsim.to_eularian_angles(k.orientation)
        except Exception:
            pass
        drone_state = DroneState(
            altitude_m=altitude,
            vx_mps=float(velocity.x_val),
            vy_mps=float(velocity.y_val),
            vz_mps=float(velocity.z_val),
            roll_rad=float(roll),
            pitch_rad=float(pitch),
            yaw_rate_radps=float(angular.z_val),
        )
        # Both values are raw AirSim NED-Z coordinates. A drone above the
        # target has a smaller/more-negative Z, so this separation is positive.
        relative_height = float(self._target_surface_z_ned - drone_z_ned)
        return drone_state, relative_height, state

    def _read_target_surface_altitude(self) -> None:
        actor = str(self._target_actor_name or self.cfg.static_target_actor_name or "")
        previous_z = float(self._target_surface_z_ned)
        previous_source = str(self._target_surface_source)
        if actor:
            try:
                pose = self.client.simGetObjectPose(actor)
                self._update_target_velocity_from_pose(pose)
                z_ned = float(pose.position.z_val)
                if np.isfinite(z_ned):
                    self._target_surface_z_ned = z_ned
                    # Diagnostic conversion only. Relative height never uses it.
                    self._target_surface_altitude_m = max(0.0, -z_ned)
                    self._target_surface_source = "api_object_pose_z_ned"
                    return
            except Exception:
                self._target_velocity_valid = False

            # A single read failure must not replace a previously valid moving
            # platform Z with the configured ground fallback. Keep the last
            # valid API/collision value until the actor query succeeds again.
            if (
                np.isfinite(previous_z)
                and (
                    previous_source.startswith("api_object_pose_z_ned")
                    or previous_source.startswith("verified_collision_api_z_ned")
                )
            ):
                self._target_surface_z_ned = previous_z
                self._target_surface_altitude_m = max(0.0, -previous_z)
                self._target_surface_source = f"{previous_source.split('_stale')[0]}_stale"
                return

        self._target_surface_altitude_m = float(self.cfg.static_target_surface_altitude_m)
        self._target_surface_z_ned = -self._target_surface_altitude_m
        self._target_surface_source = "configured_static"

    def _reset_target_motion_estimator(self) -> None:
        self._target_pose_xy = None
        self._target_pose_monotonic = None
        self._target_velocity_world_x_mps = 0.0
        self._target_velocity_world_y_mps = 0.0
        self._target_velocity_body_vx_mps = 0.0
        self._target_velocity_body_vy_mps = 0.0
        self._target_velocity_speed_mps = 0.0
        self._target_velocity_valid = False
        self._target_velocity_age_s = float("inf")

    def _update_target_velocity_from_pose(
        self,
        pose: Any,
        now: Optional[float] = None,
    ) -> None:
        """Estimate target world-NED XY velocity from consecutive actor poses."""
        if not bool(self.cfg.target_velocity_feedforward_enabled):
            self._target_velocity_valid = False
            return

        try:
            x = float(pose.position.x_val)
            y = float(pose.position.y_val)
        except Exception:
            self._target_velocity_valid = False
            return
        if not np.isfinite(x) or not np.isfinite(y):
            self._target_velocity_valid = False
            return

        sample_time = float(time.monotonic() if now is None else now)
        previous_xy = self._target_pose_xy
        previous_time = self._target_pose_monotonic
        self._target_pose_xy = (x, y)
        self._target_pose_monotonic = sample_time

        if previous_xy is None or previous_time is None:
            self._target_velocity_valid = False
            self._target_velocity_age_s = 0.0
            return

        dt = float(sample_time - previous_time)
        if not np.isfinite(dt) or dt < 0.02 or dt > 3.0:
            self._target_velocity_valid = False
            self._target_velocity_age_s = 0.0
            return

        raw_vx = float((x - previous_xy[0]) / dt)
        raw_vy = float((y - previous_xy[1]) / dt)
        raw_speed = float(math.hypot(raw_vx, raw_vy))
        if (
            not np.isfinite(raw_speed)
            or raw_speed > float(self.cfg.target_velocity_max_valid_mps)
        ):
            # A reset/teleport must never become a feed-forward speed command.
            self._target_velocity_valid = False
            self._target_velocity_world_x_mps = 0.0
            self._target_velocity_world_y_mps = 0.0
            self._target_velocity_speed_mps = 0.0
            self._target_velocity_age_s = 0.0
            return

        alpha = float(np.clip(self.cfg.target_velocity_ema_alpha, 0.0, 1.0))
        if not self._target_velocity_valid:
            filtered_vx = raw_vx
            filtered_vy = raw_vy
        else:
            filtered_vx = (
                alpha * raw_vx
                + (1.0 - alpha) * float(self._target_velocity_world_x_mps)
            )
            filtered_vy = (
                alpha * raw_vy
                + (1.0 - alpha) * float(self._target_velocity_world_y_mps)
            )

        self._target_velocity_world_x_mps = float(filtered_vx)
        self._target_velocity_world_y_mps = float(filtered_vy)
        self._target_velocity_speed_mps = float(math.hypot(filtered_vx, filtered_vy))
        self._target_velocity_valid = True
        self._target_velocity_age_s = 0.0

    def _update_target_body_velocity(self, api_state: Any) -> None:
        """Convert target world velocity into the current drone body frame."""
        now = time.monotonic()
        if self._target_pose_monotonic is None:
            age = float("inf")
        else:
            age = max(0.0, float(now - self._target_pose_monotonic))
        self._target_velocity_age_s = age

        valid = bool(
            self.cfg.target_velocity_feedforward_enabled
            and self._target_velocity_valid
            and age <= float(self.cfg.target_velocity_stale_after_s)
        )
        if not valid:
            self._target_velocity_body_vx_mps = 0.0
            self._target_velocity_body_vy_mps = 0.0
            self._target_velocity_valid = False
            return

        yaw = 0.0
        try:
            _pitch, _roll, yaw = airsim.to_eularian_angles(
                api_state.kinematics_estimated.orientation
            )
        except Exception:
            yaw = 0.0

        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        world_vx = float(self._target_velocity_world_x_mps)
        world_vy = float(self._target_velocity_world_y_mps)
        self._target_velocity_body_vx_mps = float(
            cos_yaw * world_vx + sin_yaw * world_vy
        )
        self._target_velocity_body_vy_mps = float(
            -sin_yaw * world_vx + cos_yaw * world_vy
        )

    def _get_obstacles(self, api_altitude_m: float, relative_height_m: float) -> dict[str, Any]:
        max_d = float(self.cfg.lidar_max_range_m)
        try:
            data = self.client.getLidarData(
                lidar_name=self.cfg.lidar_sensor_name,
                vehicle_name=self.cfg.vehicle_name,
            )
            points = point_cloud_to_array(getattr(data, "point_cloud", []))
            result = self.lidar_processor.compute_sector_distances(points, altitude_fallback_m=api_altitude_m)
        except Exception:
            result = {
                "front_dist_m": max_d,
                "front_left_dist_m": max_d,
                "front_right_dist_m": max_d,
                "left_dist_m": max_d,
                "right_dist_m": max_d,
                "back_dist_m": max_d,
                "down_dist_m": max(0.0, float(relative_height_m)),
                "min_obstacle_dist_m": max_d,
                "lidar_valid": False,
                "lidar_point_count": 0,
                "obstacle_source": "api_z_only_fallback",
            }

        # Vertical observation uses only the API NED separation to the target.
        result["down_dist_m"] = max(0.0, float(relative_height_m))
        result["obstacle_source"] = "lidar_horizontal+api_z_vertical"
        return result

    @staticmethod
    def _bbox_metrics(bbox_xyxy: Optional[np.ndarray], frame_shape: tuple[int, ...]) -> dict[str, float]:
        if bbox_xyxy is None:
            return {
                "err_x": 0.0,
                "err_y": 0.0,
                "center_error": 999.0,
                "bbox_rel_error": 999.0,
                "area_norm": 0.0,
            }
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        err_x = (cx - 0.5 * w) / max(1.0, 0.5 * w)
        err_y = (cy - 0.5 * h) / max(1.0, 0.5 * h)
        rel_x = abs(err_x) / max(1e-6, bw / max(1.0, w))
        rel_y = abs(err_y) / max(1e-6, bh / max(1.0, h))
        return {
            "err_x": float(err_x),
            "err_y": float(err_y),
            "center_error": float(math.hypot(err_x, err_y)),
            "bbox_rel_error": float(max(rel_x, rel_y)),
            "area_norm": float((bw * bh) / max(1.0, w * h)),
        }

    def _observe(self, frame: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
        frame = self._get_bottom_frame() if frame is None else frame
        self._sync_image_geometry(frame)
        self._last_bottom_frame = frame.copy()
        bbox_xyxy, similarity, tracker_mode = self._strict_track(frame)
        metrics = self._bbox_metrics(bbox_xyxy, frame.shape)

        # The moving platform may change world Z on uneven roads. Refresh its
        # API pose before every vertical-state calculation. No actor movement is
        # performed here; this is a read-only query.
        if self._target_actor_name:
            self._read_target_surface_altitude()

        drone_state, relative_height, api_state = self._get_api_state()
        self._update_target_body_velocity(api_state)
        drone_z_ned = float(api_state.kinematics_estimated.position.z_val)
        obstacle = self._get_obstacles(drone_state.altitude_m, relative_height)
        obstacle_state = ObstacleState(
            front_dist_m=float(obstacle["front_dist_m"]),
            front_left_dist_m=float(obstacle["front_left_dist_m"]),
            front_right_dist_m=float(obstacle["front_right_dist_m"]),
            left_dist_m=float(obstacle["left_dist_m"]),
            right_dist_m=float(obstacle["right_dist_m"]),
            back_dist_m=float(obstacle["back_dist_m"]),
            down_dist_m=max(0.0, relative_height),
            min_obstacle_dist_m=float(obstacle["min_obstacle_dist_m"]),
        )

        bbox = None
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            conf = float(similarity if tracker_mode == "MATCH" else 0.25)
            bbox = BBox(
                cx=0.5 * (x1 + x2),
                cy=0.5 * (y1 + y2),
                w=max(1.0, x2 - x1),
                h=max(1.0, y2 - y1),
                conf=conf,
            )

        obs, obs_dict = self.observation_builder.build(
            bbox=bbox,
            drone_state=drone_state,
            obstacle_state=obstacle_state,
            dt=float(self.cfg.cmd_duration_s),
        )

        live_match = bool(tracker_mode == "MATCH")
        self._update_control_image_velocity(metrics, live_match)
        match_recent = bool(
            self._step - self._last_match_step <= self.cfg.good_collision_recent_match_steps
        )
        match_confirmed = bool(
            live_match
            and self._live_match_streak >= int(self.cfg.match_confirmation_steps)
        )
        perception_now = time.monotonic()
        if live_match:
            self._lost_steps = 0
            self._non_live_started_monotonic = None
            self._non_live_duration_s = 0.0
        else:
            self._lost_steps += 1
            if self._non_live_started_monotonic is None:
                self._non_live_started_monotonic = perception_now
            self._non_live_duration_s = max(
                0.0,
                float(perception_now - self._non_live_started_monotonic),
            )

        if live_match:
            self._episode_live_match_steps += 1
            self._episode_best_similarity = max(
                self._episode_best_similarity,
                float(similarity),
            )
            if np.isfinite(metrics["center_error"]):
                self._episode_best_center_error = min(
                    self._episode_best_center_error,
                    float(metrics["center_error"]),
                )
        elif tracker_mode.startswith("PRED"):
            self._episode_predicted_steps += 1
        else:
            self._episode_no_target_steps += 1

        info: dict[str, Any] = {
            "obs_dict": obs_dict,
            "tracker_mode": tracker_mode,
            "target_id": self._target_id,
            "target_class_id": int(self._target_class_id) if self._target_class_id is not None else -1,
            "initial_yolo_class_id": int(self._target_class_id) if self._target_class_id is not None else -1,
            "strict_class_gate": False,
            "identity_judge": "resnet_only",
            "yolo_proposal_conf_threshold": float(self.cfg.yolo_proposal_conf),
            "reference_anchor_count": int(len(self._reference_embeddings)),
            "has_front_anchor": self._original_embedding is not None,
            "has_bottom_anchor": self._bottom_anchor_embedding is not None,
            "yolo_candidate_count": int(self._last_candidate_count),
            "selected_candidate_yolo_class_id": (
                int(self._last_candidate_class_id)
                if self._last_candidate_class_id is not None
                else -1
            ),
            "selected_candidate_yolo_confidence": float(self._last_candidate_confidence),
            "candidate_resnet_scores": list(self._last_candidate_scores),
            "bottom_match": live_match,
            # Fresh now means a LIVE ResNet acceptance from the current frame.
            # Recent history is exposed separately for collision evaluation.
            "bottom_match_fresh": live_match,
            "bottom_match_live": live_match,
            "bottom_match_confirmed": match_confirmed,
            "bottom_match_recent": match_recent,
            "bottom_live_match_streak": int(self._live_match_streak),
            "bottom_match_margin": float(self._last_match_margin),
            "bottom_spatial_jump_norm": float(self._last_spatial_jump_norm),
            "bottom_match_reject_reason": self._last_match_reject_reason,
            "bottom_similarity": float(similarity),
            "bottom_err_x": metrics["err_x"],
            "bottom_err_y": metrics["err_y"],
            "bottom_center_error": metrics["center_error"],
            "bottom_bbox_rel_err": metrics["bbox_rel_error"],
            "bottom_bbox_area_norm": metrics["area_norm"],
            "bottom_img_vel_x_control": float(self._control_img_vel_x),
            "bottom_img_vel_y_control": float(self._control_img_vel_y),
            "alignment_ready_streak": int(self._alignment_ready_streak),
            "descent_alignment_latched": bool(self._descent_alignment_latched),
            "horizontal_control_state": self._last_horizontal_control_state,
            "horizontal_pd_action_vx": float(self._last_pd_action_vx),
            "horizontal_pd_action_vy": float(self._last_pd_action_vy),
            "horizontal_residual_action_vx": float(self._last_residual_action_vx),
            "horizontal_residual_action_vy": float(self._last_residual_action_vy),
            "horizontal_final_action_vx": float(self._last_horizontal_action_vx),
            "horizontal_final_action_vy": float(self._last_horizontal_action_vy),
            "horizontal_speed_limit_mps": float(self._last_horizontal_speed_limit_mps),
            "target_velocity_valid": bool(self._target_velocity_valid),
            "target_velocity_age_s": float(self._target_velocity_age_s),
            "target_velocity_world_x_mps": float(self._target_velocity_world_x_mps),
            "target_velocity_world_y_mps": float(self._target_velocity_world_y_mps),
            "target_velocity_body_vx_mps": float(self._target_velocity_body_vx_mps),
            "target_velocity_body_vy_mps": float(self._target_velocity_body_vy_mps),
            "target_velocity_speed_mps": float(self._target_velocity_speed_mps),
            "horizontal_velocity_ff_vx_mps": float(self._last_velocity_ff_vx_mps),
            "horizontal_velocity_ff_vy_mps": float(self._last_velocity_ff_vy_mps),
            "horizontal_correction_vx_mps": float(self._last_horizontal_correction_vx_mps),
            "horizontal_correction_vy_mps": float(self._last_horizontal_correction_vy_mps),
            "horizontal_command_vx_mps": float(self._last_horizontal_command_vx_mps),
            "horizontal_command_vy_mps": float(self._last_horizontal_command_vy_mps),
            "horizontal_total_speed_limit_mps": float(self._last_horizontal_total_speed_limit_mps),
            "alt_agl_m": float(drone_state.altitude_m),
            "drone_z_ned": float(drone_z_ned),
            "target_surface_z_ned": float(self._target_surface_z_ned),
            "target_surface_altitude_m": float(self._target_surface_altitude_m),
            "target_surface_source": self._target_surface_source,
            "relative_height_to_target_m": float(relative_height),
            "lidar_vertical_used": False,
            "obstacle_source": obstacle["obstacle_source"],
            "lidar_valid": bool(obstacle.get("lidar_valid", False)),
            "lidar_point_count": int(obstacle.get("lidar_point_count", 0)),
            "front_dist_m": float(obstacle["front_dist_m"]),
            "left_dist_m": float(obstacle["left_dist_m"]),
            "right_dist_m": float(obstacle["right_dist_m"]),
            "back_dist_m": float(obstacle["back_dist_m"]),
            "lost_steps": int(self._lost_steps),
            "bottom_no_live_duration_s": float(self._non_live_duration_s),
            "reward_bank": float(self._reward_bank),
            "vertical_control_state": self._last_vertical_control_state,
            "descent_block_reason": self._last_descent_block_reason,
            "raw_vz_action": float(self._last_raw_vz_action),
            "requested_vz_mps": float(self._last_requested_vz_mps),
            "applied_vz_mps": float(self._last_applied_vz_mps),
            "climb_command_blocked": bool(self._last_climb_command_blocked),
        }

        if self.cfg.show_camera:
            vis = frame.copy()
            h, w = vis.shape[:2]
            cv2.drawMarker(vis, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 28, 2)
            display_mode = tracker_mode
            if live_match and not match_confirmed:
                display_mode = (
                    f"MATCH_PENDING({self._live_match_streak}/"
                    f"{int(self.cfg.match_confirmation_steps)})"
                )
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
                color = (0, 255, 0) if match_confirmed else (0, 255, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"AGENT 2 | {self._target_id} | {display_mode} sim={similarity:.3f}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(
                vis,
                f"API height above target={relative_height:.2f}m YOLOcls(diag)={self._last_candidate_class_id}",
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Z_CTRL={self._last_vertical_control_state} "
                f"raw={self._last_raw_vz_action:+.2f} "
                f"down={self._last_requested_vz_mps:+.2f} applied={self._last_applied_vz_mps:+.2f}",
                (15, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"XY_CTRL={self._last_horizontal_control_state} "
                f"PD=({self._last_pd_action_vx:+.2f},{self._last_pd_action_vy:+.2f}) "
                f"RES=({self._last_residual_action_vx:+.2f},{self._last_residual_action_vy:+.2f}) "
                f"CMD=({self._last_horizontal_action_vx:+.2f},{self._last_horizontal_action_vy:+.2f})",
                (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"TARGET_VEL={'VALID' if self._target_velocity_valid else 'WAIT'} "
                f"body=({self._target_velocity_body_vx_mps:+.2f},"
                f"{self._target_velocity_body_vy_mps:+.2f})m/s "
                f"lost={self._lost_steps}/{self._non_live_duration_s:.1f}s",
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Agent 2 - Bottom Landing", vis)
            cv2.waitKey(1)

        self._last_info = info
        return np.asarray(obs, dtype=np.float32), info

    # ------------------------------------------------------------------
    # Control/reward
    # ------------------------------------------------------------------
    def _update_control_image_velocity(self, metrics: dict[str, float], live_match: bool) -> None:
        """Estimate live target motion using real wall-clock spacing between frames."""
        now = time.monotonic()
        if live_match:
            err_x = float(metrics.get("err_x", 0.0) or 0.0)
            err_y = float(metrics.get("err_y", 0.0) or 0.0)
            if self._last_control_had_live_match and self._last_observation_monotonic is not None:
                dt = float(np.clip(now - self._last_observation_monotonic, 0.03, 1.0))
                clip_v = float(self.cfg.horizontal_velocity_clip_per_s)
                raw_vx = float(np.clip((err_x - self._last_control_err_x) / dt, -clip_v, clip_v))
                raw_vy = float(np.clip((err_y - self._last_control_err_y) / dt, -clip_v, clip_v))
                alpha = float(np.clip(self.cfg.horizontal_velocity_ema_alpha, 0.0, 1.0))
                self._control_img_vel_x = float(alpha * raw_vx + (1.0 - alpha) * self._control_img_vel_x)
                self._control_img_vel_y = float(alpha * raw_vy + (1.0 - alpha) * self._control_img_vel_y)
            else:
                self._control_img_vel_x = 0.0
                self._control_img_vel_y = 0.0
            self._last_control_err_x = err_x
            self._last_control_err_y = err_y
            self._last_control_had_live_match = True
        else:
            self._control_img_vel_x = 0.0
            self._control_img_vel_y = 0.0
            self._last_control_had_live_match = False
        self._last_observation_monotonic = now

    def _horizontal_speed_limit(self, info: dict[str, Any]) -> float:
        """Return a conservative XY speed limit from API height and visual scale."""
        try:
            height = float(info.get("relative_height_to_target_m", float("inf")))
        except (TypeError, ValueError):
            height = float("inf")
        try:
            area = float(info.get("bottom_bbox_area_norm", 0.0))
        except (TypeError, ValueError):
            area = 0.0

        if not np.isfinite(height):
            height_limit = float(self.cfg.horizontal_speed_mid_mps)
        elif height <= 0.8:
            height_limit = float(self.cfg.horizontal_speed_touchdown_mps)
        elif height <= 1.6:
            height_limit = float(self.cfg.horizontal_speed_near_mps)
        elif height <= 3.5:
            height_limit = float(self.cfg.horizontal_speed_mid_mps)
        else:
            height_limit = float(self.cfg.horizontal_speed_far_mps)

        if area >= 0.20:
            area_limit = float(self.cfg.horizontal_speed_touchdown_mps)
        elif area >= 0.10:
            area_limit = float(self.cfg.horizontal_speed_near_mps)
        elif area >= 0.045:
            area_limit = float(self.cfg.horizontal_speed_mid_mps)
        else:
            area_limit = float(self.cfg.horizontal_speed_far_mps)

        return float(max(0.05, min(height_limit, area_limit)))

    def _horizontal_visual_servo(
        self,
        raw: np.ndarray,
        info: dict[str, Any],
    ) -> tuple[float, float, dict[str, Any]]:
        """Combine target-speed feed-forward, visual PD, and PPO residual."""
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        err_x = float(info.get("bottom_err_x", 0.0) or 0.0)
        err_y = float(info.get("bottom_err_y", 0.0) or 0.0)
        vel_x = float(
            info.get(
                "bottom_img_vel_x_control",
                getattr(self, "_control_img_vel_x", 0.0),
            )
            or 0.0
        )
        vel_y = float(
            info.get(
                "bottom_img_vel_y_control",
                getattr(self, "_control_img_vel_y", 0.0),
            )
            or 0.0
        )
        correction_speed_limit = self._horizontal_speed_limit(info)

        velocity_valid = bool(info.get("target_velocity_valid", False))
        ff_gain = float(self.cfg.target_velocity_feedforward_gain)
        ff_vx = (
            ff_gain * float(info.get("target_velocity_body_vx_mps", 0.0) or 0.0)
            if velocity_valid
            else 0.0
        )
        ff_vy = (
            ff_gain * float(info.get("target_velocity_body_vy_mps", 0.0) or 0.0)
            if velocity_valid
            else 0.0
        )
        ff_speed = float(math.hypot(ff_vx, ff_vy))
        max_total = max(0.10, float(self.cfg.horizontal_total_speed_max_mps))
        total_speed_limit = float(
            min(max_total, max(correction_speed_limit, ff_speed + correction_speed_limit))
        )

        def clip_vector(vx: float, vy: float, limit: float) -> tuple[float, float]:
            magnitude = float(math.hypot(vx, vy))
            if magnitude <= max(1.0e-9, limit):
                return float(vx), float(vy)
            scale = float(limit / magnitude)
            return float(vx * scale), float(vy * scale)

        if not live_match:
            self._episode_xy_hold_steps = int(
                getattr(self, "_episode_xy_hold_steps", 0)
            ) + 1
            no_live_duration_s = float(
                info.get("bottom_no_live_duration_s", 0.0) or 0.0
            )
            velocity_hold = bool(
                velocity_valid
                and no_live_duration_s <= float(
                    self.cfg.horizontal_velocity_hold_max_s
                )
            )
            if velocity_hold:
                vx, vy = clip_vector(ff_vx, ff_vy, total_speed_limit)
                state = "VELOCITY_HOLD_NO_LIVE_MATCH"
            else:
                vx, vy = 0.0, 0.0
                state = "HOLD_NO_LIVE_MATCH"
            details = {
                "state": state,
                "pd_ax": 0.0,
                "pd_ay": 0.0,
                "residual_ax": 0.0,
                "residual_ay": 0.0,
                "final_ax": 0.0,
                "final_ay": 0.0,
                "speed_limit_mps": correction_speed_limit,
                "ff_vx_mps": float(ff_vx if velocity_hold else 0.0),
                "ff_vy_mps": float(ff_vy if velocity_hold else 0.0),
                "correction_vx_mps": 0.0,
                "correction_vy_mps": 0.0,
                "command_vx_mps": float(vx),
                "command_vy_mps": float(vy),
                "total_speed_limit_mps": total_speed_limit,
            }
            return float(vx), float(vy), details

        pd_ax = (
            -float(self.cfg.horizontal_pd_kp_y_to_vx) * err_y
            - float(self.cfg.horizontal_pd_kd_y_to_vx) * vel_y
        )
        pd_ay = (
            float(self.cfg.horizontal_pd_kp_x_to_vy) * err_x
            + float(self.cfg.horizontal_pd_kd_x_to_vy) * vel_x
        )

        if (
            abs(err_y) <= float(self.cfg.horizontal_deadband_error)
            and abs(vel_y) <= float(self.cfg.horizontal_deadband_velocity)
        ):
            pd_ax = 0.0
        if (
            abs(err_x) <= float(self.cfg.horizontal_deadband_error)
            and abs(vel_x) <= float(self.cfg.horizontal_deadband_velocity)
        ):
            pd_ay = 0.0

        pd_max = float(np.clip(self.cfg.horizontal_pd_max_action, 0.05, 1.0))
        pd_ax = float(np.clip(pd_ax, -pd_max, pd_max))
        pd_ay = float(np.clip(pd_ay, -pd_max, pd_max))

        residual_max = float(
            np.clip(self.cfg.horizontal_ppo_residual_max_action, 0.0, 0.30)
        )
        if confirmed:
            residual_ax = float(
                np.clip(float(raw[0]) * residual_max, -residual_max, residual_max)
            )
            residual_ay = float(
                np.clip(float(raw[1]) * residual_max, -residual_max, residual_max)
            )
            state = "VELOCITY_FF_PD_PLUS_RESIDUAL"
        else:
            residual_ax = 0.0
            residual_ay = 0.0
            state = "VELOCITY_FF_PD_MATCH_PENDING"

        final_ax = float(np.clip(pd_ax + residual_ax, -1.0, 1.0))
        final_ay = float(np.clip(pd_ay + residual_ay, -1.0, 1.0))
        correction_vx = float(final_ax * correction_speed_limit)
        correction_vy = float(final_ay * correction_speed_limit)
        vx, vy = clip_vector(
            ff_vx + correction_vx,
            ff_vy + correction_vy,
            total_speed_limit,
        )

        details = {
            "state": state,
            "pd_ax": pd_ax,
            "pd_ay": pd_ay,
            "residual_ax": residual_ax,
            "residual_ay": residual_ay,
            "final_ax": final_ax,
            "final_ay": final_ay,
            "speed_limit_mps": correction_speed_limit,
            "ff_vx_mps": ff_vx,
            "ff_vy_mps": ff_vy,
            "correction_vx_mps": correction_vx,
            "correction_vy_mps": correction_vy,
            "command_vx_mps": vx,
            "command_vy_mps": vy,
            "total_speed_limit_mps": total_speed_limit,
        }
        return float(vx), float(vy), details

    def _horizontal_lidar_guard(self, vx: float, vy: float, info: dict[str, Any]) -> tuple[float, float, list[str]]:
        reasons: list[str] = []
        emergency = float(self.cfg.obstacle_emergency_m)
        warning = float(self.cfg.obstacle_warning_m)

        front = float(info["front_dist_m"])
        back = float(info["back_dist_m"])
        left = float(info["left_dist_m"])
        right = float(info["right_dist_m"])

        if vx > 0 and front < emergency:
            vx = 0.0
            reasons.append("front_emergency")
        elif vx > 0 and front < warning:
            vx *= max(0.0, (front - emergency) / max(1e-6, warning - emergency))
            reasons.append("front_slow")

        if vx < 0 and back < emergency:
            vx = 0.0
            reasons.append("back_emergency")
        if vy > 0 and right < emergency:
            vy = 0.0
            reasons.append("right_emergency")
        if vy < 0 and left < emergency:
            vy = 0.0
            reasons.append("left_emergency")
        return float(vx), float(vy), reasons

    @staticmethod
    def _scaled_frame_crop(
        frame: np.ndarray,
        scale: float,
        anchor_x: float,
        anchor_y: float,
    ) -> Optional[np.ndarray]:
        """Return an image crop centered at a normalized anchor position."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        scale = float(np.clip(scale, 0.05, 1.0))
        crop_w = max(3, int(round(w * scale)))
        crop_h = max(3, int(round(h * scale)))
        cx = float(np.clip(anchor_x, 0.0, 1.0)) * float(w)
        cy = float(np.clip(anchor_y, 0.0, 1.0)) * float(h)
        x1 = int(round(cx - 0.5 * crop_w))
        y1 = int(round(cy - 0.5 * crop_h))
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w]
        if crop.size == 0:
            return None
        return crop

    def _appearance_similarity_to_bottom_anchor(self, crop: np.ndarray) -> float:
        """Compare one current-frame crop with the immutable bottom anchor."""
        reference = self._bottom_anchor_embedding
        if reference is None:
            # Standalone Agent-2 clicks the target from the bottom camera, so
            # its original embedding is a valid fallback when no explicit
            # handoff bottom anchor exists.
            reference = self._original_embedding
        if reference is None:
            return 0.0
        embedding = self.tracker._embedding_from_crop(crop)
        if embedding is None:
            return 0.0
        ref = F.normalize(reference.reshape(-1), dim=0)
        emb = F.normalize(embedding.reshape(-1), dim=0)
        return float(torch.dot(ref, emb).detach().cpu().item())

    def _contact_appearance_evidence(self) -> dict[str, Any]:
        """Measure whether the selected target is under the bottom camera now.

        This uses only the collision-frame image. Center crops are target
        candidates; same-frame corner crops are background controls. This
        distinguishes a top touchdown from a side impact where the camera sees
        mostly floor beneath the drone.
        """
        frame = self._last_bottom_frame
        if frame is None or frame.size == 0:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "missing_collision_frame",
            }
        if self._bottom_anchor_embedding is None and self._original_embedding is None:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "missing_target_anchor",
            }

        center_scores: list[tuple[float, float]] = []
        for scale in self.cfg.contact_appearance_center_crop_scales:
            crop = self._scaled_frame_crop(frame, float(scale), 0.5, 0.5)
            if crop is None:
                continue
            center_scores.append((float(scale), self._appearance_similarity_to_bottom_anchor(crop)))

        corner_scale = float(self.cfg.contact_appearance_corner_crop_scale)
        corner_scores: list[float] = []
        # Anchors place each control crop inside a corner without going outside
        # the frame. They deliberately avoid the image center.
        offset = 0.5 * corner_scale
        for ax, ay in (
            (offset, offset),
            (1.0 - offset, offset),
            (offset, 1.0 - offset),
            (1.0 - offset, 1.0 - offset),
        ):
            crop = self._scaled_frame_crop(frame, corner_scale, ax, ay)
            if crop is None:
                continue
            corner_scores.append(self._appearance_similarity_to_bottom_anchor(crop))

        if not center_scores:
            return {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": max(corner_scores, default=0.0),
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "no_valid_center_crop",
            }

        best_scale, best_center = max(center_scores, key=lambda item: item[1])
        best_corner = max(corner_scores, default=0.0)
        margin = float(best_center - best_corner)
        absolute_ok = best_center >= float(self.cfg.contact_appearance_min_similarity)
        strong_ok = best_center >= float(self.cfg.contact_appearance_strong_similarity)
        contrast_ok = margin >= float(self.cfg.contact_appearance_min_center_margin)
        valid = bool(absolute_ok and (strong_ok or contrast_ok))

        if not absolute_ok:
            reason = "center_similarity_below_threshold"
        elif not strong_ok and not contrast_ok:
            reason = "center_not_stronger_than_background"
        else:
            reason = ""

        return {
            "valid": valid,
            "center_similarity": float(best_center),
            "corner_similarity": float(best_corner),
            "center_margin": margin,
            "best_center_scale": float(best_scale),
            "reason": reason,
        }

    @staticmethod
    def _normalized_collision_object_name(value: str) -> str:
        return str(value or "").strip().casefold()

    def _is_ground_collision_object(self, object_name: str) -> bool:
        normalized = self._normalized_collision_object_name(object_name)
        if not normalized:
            return True
        return any(
            str(token).strip().casefold() in normalized
            for token in self.cfg.collision_ground_tokens
            if str(token).strip()
        )

    def _update_verified_alignment_latch(self, info: dict[str, Any]) -> None:
        """Store only a strict, confirmed LIVE alignment from this frame."""
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))
        similarity = float(info.get("bottom_similarity", 0.0))
        strict_alignment = bool(
            live_match
            and confirmed
            and np.isfinite(center_error)
            and np.isfinite(bbox_rel)
            and np.isfinite(similarity)
            and center_error <= float(self.cfg.collision_latch_center_error)
            and bbox_rel <= float(self.cfg.collision_latch_bbox_rel_error)
            and similarity >= float(self.cfg.collision_latch_min_similarity)
        )
        if not strict_alignment:
            return
        self._last_verified_alignment_step = int(self._step)
        self._last_verified_alignment_center_error = float(center_error)
        self._last_verified_alignment_bbox_rel_error = float(bbox_rel)
        self._last_verified_alignment_similarity = float(similarity)

    def _record_authorized_descent(
        self,
        pre_info: dict[str, Any],
        descent_allowed: bool,
        applied_vz_mps: float,
    ) -> None:
        """Latch only an actual descent command that passed the safety gate."""
        vz = float(applied_vz_mps)
        if not bool(descent_allowed) or vz <= float(self.cfg.authorized_descent_min_vz_mps):
            return

        center_error = float(pre_info.get("bottom_center_error", 999.0))
        bbox_rel = float(pre_info.get("bottom_bbox_rel_err", 999.0))
        similarity = float(pre_info.get("bottom_similarity", 0.0))
        live_match = bool(pre_info.get("bottom_match_live", False))
        confirmed = bool(pre_info.get("bottom_match_confirmed", False))

        # This is intentionally redundant with _vertical_control_state(). It
        # prevents a future caller from creating physical authorization without
        # current, confirmed visual evidence.
        valid_evidence = bool(
            live_match
            and confirmed
            and np.isfinite(center_error)
            and np.isfinite(bbox_rel)
            and np.isfinite(similarity)
            and center_error <= float(self.cfg.alignment_exit_center_error)
            and bbox_rel <= float(self.cfg.alignment_exit_bbox_rel_error)
            and similarity >= float(self.cfg.descent_min_similarity)
        )
        if not valid_evidence:
            return

        self._authorized_descent_latched = True
        self._last_authorized_descent_step = int(self._step)
        self._last_authorized_descent_center_error = float(center_error)
        self._last_authorized_descent_bbox_rel_error = float(bbox_rel)
        self._last_authorized_descent_similarity = float(similarity)
        self._last_authorized_descent_vz_mps = float(vz)
        self._authorized_descent_invalidated_reason = ""

    def _update_authorized_descent_latch_after_observation(
        self,
        info: dict[str, Any],
    ) -> None:
        """Keep episode-level descent authorization until reset.

        Close to contact, visual detections can disappear or become geometrically
        unreliable. Therefore age and intermediate visual drift no longer erase
        evidence that a real, alignment-authorized descent command occurred.
        A clearly off-center LIVE frame is evaluated only at collision time.
        """
        _ = info
        return

    def _recovery_request_decision(
        self,
        info: dict[str, Any],
        collision_now: bool,
    ) -> dict[str, Any]:
        """Decide whether AGENT_1P2 must return control to Agent 1."""
        result = {
            "requested": False,
            "reason": "",
            "contact_guard_active": False,
            "contact_guard_similarity": 0.0,
            "contact_guard_margin": 0.0,
        }
        if collision_now or not self._attached_from_agent1:
            return result
        if bool(info.get("bottom_match_live", False)):
            return result

        lost_steps = int(info.get("lost_steps", 0) or 0)
        no_live_duration_s = float(
            info.get("bottom_no_live_duration_s", 0.0) or 0.0
        )
        if lost_steps < int(self.cfg.recovery_no_live_match_min_steps):
            return result
        if no_live_duration_s < float(self.cfg.recovery_no_live_match_timeout_s):
            return result

        relative_height = float(info.get("relative_height_to_target_m", float("inf")))
        if (
            relative_height <= float(self.cfg.recovery_contact_guard_max_height_m)
            and no_live_duration_s
            <= float(self.cfg.recovery_contact_guard_max_no_live_s)
        ):
            contact = self._contact_appearance_evidence()
            result["contact_guard_similarity"] = float(contact["center_similarity"])
            result["contact_guard_margin"] = float(contact["center_margin"])
            if bool(contact["valid"]):
                # This is only a short collision/contact grace period. Descent
                # remains blocked without LIVE MATCH, and after the grace limit
                # Agent 1 recovery is requested even if appearance stays high.
                result["contact_guard_active"] = True
                return result

        result["requested"] = True
        result["reason"] = (
            f"bottom_live_match_missing_{no_live_duration_s:.2f}s_"
            f"{lost_steps}_steps"
        )
        return result

    def _collision_reward_decision(
        self,
        info: dict[str, Any],
        collision_object: str,
        collision_now: bool = False,
    ) -> dict[str, Any]:
        """Classify touchdown from current collision-frame evidence only.

        Historical visual latches and prior descent authorization are logged for
        diagnosis, but they cannot grant terminal reward. Success requires a
        target-object collision plus either a centered LIVE bottom-camera match
        or direct target appearance under the camera in the current frame.
        """
        live_match = bool(info.get("bottom_match_live", False))
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))
        similarity = float(info.get("bottom_similarity", 0.0))

        live_alignment = bool(
            live_match
            and np.isfinite(center_error)
            and np.isfinite(bbox_rel)
            and center_error <= float(self.cfg.good_collision_center_error)
            and bbox_rel <= float(self.cfg.good_collision_bbox_rel_error)
        )

        # Direct image evidence is evaluated only on a real collision step.
        # This avoids extra ResNet inference during normal control.
        if collision_now:
            contact = self._contact_appearance_evidence()
        else:
            contact = {
                "valid": False,
                "center_similarity": 0.0,
                "corner_similarity": 0.0,
                "center_margin": 0.0,
                "best_center_scale": 0.0,
                "reason": "not_a_collision_step",
            }

        if live_alignment:
            success_path = "LIVE_MATCH"
        elif bool(contact["valid"]):
            success_path = "CONTACT_APPEARANCE"
        else:
            success_path = "NONE"

        # Historical values remain in the output for comparison with v6-v9.
        last_verified_step = int(getattr(self, "_last_verified_alignment_step", -999999))
        last_verified_center = float(
            getattr(self, "_last_verified_alignment_center_error", 999.0)
        )
        last_verified_bbox_rel = float(
            getattr(self, "_last_verified_alignment_bbox_rel_error", 999.0)
        )
        last_verified_similarity = float(
            getattr(self, "_last_verified_alignment_similarity", 0.0)
        )
        latch_age = int(self._step - last_verified_step)

        last_authorized_step = int(
            getattr(self, "_last_authorized_descent_step", -999999)
        )
        authorized_age = int(self._step - last_authorized_step)
        last_authorized_center = float(
            getattr(self, "_last_authorized_descent_center_error", 999.0)
        )
        last_authorized_bbox_rel = float(
            getattr(self, "_last_authorized_descent_bbox_rel_error", 999.0)
        )
        last_authorized_similarity = float(
            getattr(self, "_last_authorized_descent_similarity", 0.0)
        )
        last_authorized_vz = float(
            getattr(self, "_last_authorized_descent_vz_mps", 0.0)
        )

        normalized_object = self._normalized_collision_object_name(collision_object)
        ground_contact = self._is_ground_collision_object(collision_object)
        object_matches_target = False
        object_lock_created = False

        current_bottom_evidence = success_path != "NONE"
        expected_name = str(getattr(self, "_expected_collision_object_name", "") or "")
        expected_normalized = self._normalized_collision_object_name(expected_name)
        object_lock_eligible = bool(current_bottom_evidence)
        if current_bottom_evidence and normalized_object and not ground_contact:
            if expected_normalized:
                object_matches_target = normalized_object == expected_normalized
            elif bool(self.cfg.collision_object_auto_lock) and object_lock_eligible:
                self._expected_collision_object_name = str(collision_object)
                self._expected_collision_object_source = "current_bottom_evidence_touchdown"
                object_matches_target = True
                object_lock_created = True

        if not collision_now:
            reject_reason = ""
        elif ground_contact:
            reject_reason = "ground_or_terrain_collision"
        elif not normalized_object:
            reject_reason = "empty_collision_object"
        elif not current_bottom_evidence:
            reject_reason = str(contact.get("reason", "bottom_target_not_verified")) or "bottom_target_not_verified"
        elif not object_matches_target:
            reject_reason = "collision_object_mismatch"
        else:
            reject_reason = ""

        return {
            "success": bool(collision_now and current_bottom_evidence and object_matches_target),
            "success_path": success_path,
            "live_match_at_collision": live_match,
            "live_center_error": center_error,
            "live_bbox_rel_error": bbox_rel,
            "live_similarity": similarity,
            "contact_appearance_valid": bool(contact["valid"]),
            "contact_center_similarity": float(contact["center_similarity"]),
            "contact_corner_similarity": float(contact["corner_similarity"]),
            "contact_center_margin": float(contact["center_margin"]),
            "contact_best_center_scale": float(contact["best_center_scale"]),
            "contact_appearance_reason": str(contact["reason"]),
            "alignment_latch_age": latch_age,
            "last_verified_center_error": last_verified_center,
            "last_verified_bbox_rel_error": last_verified_bbox_rel,
            "last_verified_similarity": last_verified_similarity,
            "authorized_descent_latched": bool(
                getattr(self, "_authorized_descent_latched", False)
            ),
            "authorized_descent_latch_age": authorized_age,
            "last_authorized_descent_center_error": last_authorized_center,
            "last_authorized_descent_bbox_rel_error": last_authorized_bbox_rel,
            "last_authorized_descent_similarity": last_authorized_similarity,
            "last_authorized_descent_vz_mps": last_authorized_vz,
            "authorized_descent_invalidated_reason": "diagnostic_only",
            "collision_object_matches_target": bool(object_matches_target),
            "collision_object_lock_created": bool(object_lock_created),
            "collision_object_lock_eligible": bool(object_lock_eligible),
            "expected_collision_object_name": str(
                getattr(self, "_expected_collision_object_name", "") or ""
            ),
            "expected_collision_object_source": str(
                getattr(self, "_expected_collision_object_source", "unlocked") or "unlocked"
            ),
            "collision_ground_contact": bool(ground_contact),
            "reject_reason": reject_reason,
        }

    def _new_collision(self) -> tuple[bool, str, int]:
        """Return a collision only when it is new relative to the episode reset."""
        try:
            collision = self.client.simGetCollisionInfo(
                vehicle_name=self.cfg.vehicle_name
            )
            collided = bool(getattr(collision, "has_collided", False))
            timestamp = int(getattr(collision, "time_stamp", 0) or 0)
            object_name = str(getattr(collision, "object_name", "") or "")
            is_new = bool(
                collided
                and (
                    timestamp == 0
                    or timestamp != self._collision_timestamp_at_reset
                )
            )
            return is_new, object_name, timestamp
        except Exception:
            return False, "", 0

    def _vertical_control_state(self, info: dict[str, Any]) -> tuple[str, bool, str]:
        """Alignment-gated descent with hysteresis and recentering."""
        live_match = bool(info.get("bottom_match_live", False))
        confirmed = bool(info.get("bottom_match_confirmed", False))
        similarity = float(info.get("bottom_similarity", 0.0))
        center_error = float(info.get("bottom_center_error", 999.0))
        bbox_rel = float(info.get("bottom_bbox_rel_err", 999.0))
        streak = int(info.get("bottom_live_match_streak", 0))

        if not live_match:
            self._alignment_ready_streak = 0
            self._descent_alignment_latched = False
            return "HOLD_NO_LIVE_MATCH", False, "target_not_detected_current_frame"
        if not confirmed or streak < int(self.cfg.descent_min_live_match_streak):
            self._alignment_ready_streak = 0
            self._descent_alignment_latched = False
            return "HOLD_MATCH_CONFIRM", False, "live_match_streak_too_short"
        if similarity < float(self.cfg.descent_min_similarity):
            self._alignment_ready_streak = 0
            self._descent_alignment_latched = False
            return "HOLD_LOW_SIMILARITY", False, "resnet_similarity_below_descent_threshold"

        if bool(getattr(self, "_descent_alignment_latched", False)):
            if center_error > float(self.cfg.alignment_exit_center_error):
                self._alignment_ready_streak = 0
                self._descent_alignment_latched = False
                self._episode_recenter_steps = int(getattr(self, "_episode_recenter_steps", 0)) + 1
                return "RECENTER_XY", False, "center_error_exceeded_descent_exit_limit"
            if bbox_rel > float(self.cfg.alignment_exit_bbox_rel_error):
                self._alignment_ready_streak = 0
                self._descent_alignment_latched = False
                self._episode_recenter_steps = int(getattr(self, "_episode_recenter_steps", 0)) + 1
                return "RECENTER_BBOX", False, "bbox_error_exceeded_descent_exit_limit"
            return "DESCEND_TRACKING", True, ""

        enter_ok = bool(
            center_error <= float(self.cfg.alignment_enter_center_error)
            and bbox_rel <= float(self.cfg.alignment_enter_bbox_rel_error)
        )
        if enter_ok:
            self._alignment_ready_streak = int(getattr(self, "_alignment_ready_streak", 0)) + 1
        else:
            self._alignment_ready_streak = 0

        required = max(1, int(self.cfg.alignment_streak_required))
        if self._alignment_ready_streak >= required:
            self._descent_alignment_latched = True
            return "DESCEND_TRACKING", True, ""
        if center_error > float(self.cfg.alignment_enter_center_error):
            return "ALIGN_CENTER", False, "target_center_error_too_large"
        if bbox_rel > float(self.cfg.alignment_enter_bbox_rel_error):
            return "ALIGN_BBOX", False, "target_bbox_relative_error_too_large"
        return "ALIGN_STABLE_PENDING", False, "alignment_streak_too_short"

    def step(self, action):
        self._step += 1
        raw = np.asarray(action, dtype=np.float32).reshape(4)
        raw = np.clip(raw, -1.0, 1.0)

        # Use the observation already returned by reset()/the previous step for
        # pre-action safety. Perception is executed exactly once per physical
        # control step, after the command, so tracker/lost counters and visual
        # dynamics advance once rather than twice.
        if not self._last_info:
            raise RuntimeError("Agent2LandingEnv.step() called before reset()/handoff observation.")
        pre_info = dict(self._last_info)

        vx, vy, horizontal = self._horizontal_visual_servo(raw, pre_info)
        self._last_horizontal_control_state = str(horizontal["state"])
        self._last_pd_action_vx = float(horizontal["pd_ax"])
        self._last_pd_action_vy = float(horizontal["pd_ay"])
        self._last_residual_action_vx = float(horizontal["residual_ax"])
        self._last_residual_action_vy = float(horizontal["residual_ay"])
        self._last_horizontal_action_vx = float(horizontal["final_ax"])
        self._last_horizontal_action_vy = float(horizontal["final_ay"])
        self._last_horizontal_speed_limit_mps = float(horizontal["speed_limit_mps"])
        self._last_velocity_ff_vx_mps = float(horizontal["ff_vx_mps"])
        self._last_velocity_ff_vy_mps = float(horizontal["ff_vy_mps"])
        self._last_horizontal_correction_vx_mps = float(horizontal["correction_vx_mps"])
        self._last_horizontal_correction_vy_mps = float(horizontal["correction_vy_mps"])
        self._last_horizontal_command_vx_mps = float(horizontal["command_vx_mps"])
        self._last_horizontal_command_vy_mps = float(horizontal["command_vy_mps"])
        self._last_horizontal_total_speed_limit_mps = float(
            horizontal["total_speed_limit_mps"]
        )

        # Landing-only vertical contract:
        #   raw_z > 0  -> request descent (positive AirSim NED vz)
        #   raw_z <= 0 -> hover vertically
        # A fresh/untrained PPO policy can therefore never escape upward.
        raw_vz_action = float(raw[2])
        climb_command_blocked = bool(raw_vz_action < 0.0)
        requested_vz = max(0.0, raw_vz_action) * float(self.cfg.vz_scale_mps)
        vertical_state, descent_allowed, descent_block_reason = self._vertical_control_state(pre_info)
        descent_requested = bool(requested_vz > 0.0)
        descent_blocked = bool(descent_requested and not descent_allowed)
        if descent_allowed:
            vz = float(requested_vz)
        else:
            vz = 0.0

        self._last_vertical_control_state = vertical_state
        self._last_descent_block_reason = descent_block_reason
        self._last_raw_vz_action = float(raw_vz_action)
        self._last_requested_vz_mps = float(requested_vz)
        self._last_applied_vz_mps = float(vz)
        self._last_climb_command_blocked = bool(climb_command_blocked)

        if climb_command_blocked:
            self._episode_climb_command_blocked_steps += 1
        if descent_requested:
            self._episode_descent_requested_steps += 1
            if descent_blocked:
                self._episode_descent_blocked_steps += 1
            else:
                self._episode_descent_allowed_steps += 1
        yaw_rate = float(np.clip(raw[3], -0.20, 0.20)) * self.cfg.yaw_scale_dps
        vx, vy, guard_reasons = self._horizontal_lidar_guard(vx, vy, pre_info)

        # Final invariant at the AirSim boundary: Agent 2 can never transmit a
        # negative NED vz. Even a future upstream regression cannot command up.
        vz = max(0.0, float(vz))
        self._last_applied_vz_mps = float(vz)
        self._record_authorized_descent(pre_info, descent_allowed, vz)

        external_info: dict[str, Any] = {}
        if self._external_command_executor is not None:
            # Agent 1 owns XY/Yaw and sends the only physical command. Agent 2
            # contributes only its already-gated Z command.
            external_info = dict(self._external_command_executor(vz) or {})
            self._last_external_command_info = external_info
            vx = float(external_info.get("agent1_commanded_vx_mps", 0.0))
            vy = float(external_info.get("agent1_commanded_vy_mps", 0.0))
            vz = max(
                0.0,
                float(external_info.get("agent1_commanded_vz_mps", vz)),
            )
            yaw_rate = float(
                external_info.get("agent1_commanded_yaw_rate_dps", 0.0)
            )
        else:
            self.client.moveByVelocityBodyFrameAsync(
                vx=vx,
                vy=vy,
                vz=vz,
                duration=float(self.cfg.cmd_duration_s),
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                vehicle_name=self.cfg.vehicle_name,
            ).join()
            self._last_external_command_info = {}

        obs, info = self._observe()
        collision_now, collision_object, collision_timestamp = self._new_collision()

        center_error = float(info["bottom_center_error"])
        bbox_rel = float(info["bottom_bbox_rel_err"])
        relative_height = float(info["relative_height_to_target_m"])
        visible = bool(info["bottom_match_live"])

        center_score = 0.0 if not np.isfinite(center_error) or center_error > 2.0 else math.exp(-4.0 * center_error)
        center_bank = 8.0 * center_score if visible else 0.0
        progress_bank = 0.0
        if self._prev_relative_height_m is not None and visible:
            descent_progress = float(self._prev_relative_height_m - relative_height)
            if descent_progress > 0.0 and center_error <= 0.45:
                progress_bank = min(12.0, 80.0 * descent_progress)
        if bool(self.cfg.parallel_dual_agent_mode):
            # XY/Yaw actions remain in the 4-D space only for checkpoint
            # compatibility, but they are masked physically. Do not shape the
            # landing policy with action dimensions it does not own.
            smooth_bank = -0.15 * abs(float(raw[2] - self._prev_action[2]))
        else:
            smooth_bank = -0.15 * float(np.linalg.norm(raw - self._prev_action))
        shaping = float(center_bank + progress_bank + smooth_bank)
        self._reward_bank += max(0.0, shaping)

        # Update touchdown evidence before evaluating a collision from this
        # physical step. The visual latch covers a short detector gap; the
        # physical flag records that an ACTUAL, alignment-authorized descent
        # occurred at least once in the current episode.
        self._update_verified_alignment_latch(info)
        self._update_authorized_descent_latch_after_observation(info)
        collision_decision = self._collision_reward_decision(
            info, collision_object, collision_now=collision_now
        )
        if bool(self.cfg.parallel_dual_agent_mode):
            recovery_decision = {
                "requested": False,
                "reason": "parallel_agent1_continuous",
                "contact_guard_active": False,
                "contact_guard_similarity": 0.0,
                "contact_guard_margin": 0.0,
            }
        else:
            recovery_decision = self._recovery_request_decision(
                info, collision_now
            )
        if bool(recovery_decision["requested"]):
            print(
                "[A2 RECOVERY REQUEST] "
                f"reason={recovery_decision['reason']} "
                f"height={relative_height:.2f}m "
                f"noLive={self._non_live_duration_s:.2f}s "
                f"targetSpeed={self._target_velocity_speed_mps:.2f}m/s"
            )

        done = False
        reason = ""
        reward = 0.0
        good_xy = bool(collision_decision["success"])

        if collision_now:
            done = True
            if good_xy:
                reason = "landing_collision_success"
                reward = float(np.clip(
                    self.cfg.success_base_reward + self._reward_bank,
                    self.cfg.success_min_reward,
                    self.cfg.success_max_reward,
                ))
                # Calibrate in raw AirSim NED-Z, the same coordinate used by
                # the drone state. No abs()/sign conversion is involved.
                self._target_surface_z_ned = float(info["drone_z_ned"])
                self._target_surface_altitude_m = max(0.0, -self._target_surface_z_ned)
                self._target_surface_source = "verified_collision_api_z_ned"
            elif collision_decision["success_path"] != "NONE":
                reason = "landing_collision_wrong_object"
                reward = -float(self.cfg.wrong_collision_penalty)
            else:
                reason = "landing_collision_bad_xy"
                reward = -float(self.cfg.wrong_collision_penalty)
        elif self._step >= int(self.cfg.max_episode_steps):
            done = True
            reason = "landing_timeout_no_collision"
            reward = 0.0
        elif (
            not bool(self.cfg.parallel_dual_agent_mode)
            and self._lost_steps >= int(self.cfg.target_lost_limit_steps)
        ):
            done = True
            reason = "landing_target_lost"
            reward = 0.0

        self._episode_return += float(reward)
        self._prev_action = raw.copy()
        self._prev_relative_height_m = relative_height
        info.update(
            {
                "agent": "AGENT_2",
                "raw_action": raw.copy(),
                "parallel_dual_agent": bool(self.cfg.parallel_dual_agent_mode),
                "xy_yaw_owner": "AGENT_1" if bool(self.cfg.parallel_dual_agent_mode) else "AGENT_2",
                "z_owner": "AGENT_2",
                "agent2_xy_yaw_masked": bool(self.cfg.parallel_dual_agent_mode),
                "parallel_agent1_tracking_mode": str(external_info.get("agent1_tracking_mode", "")),
                "parallel_agent1_active_camera": str(external_info.get("agent1_active_camera", "")),
                "parallel_agent1_camera_authority": str(external_info.get("agent1_camera_authority", "")),
                "parallel_agent1_fusion_has_target": bool(external_info.get("agent1_fusion_has_target", False)),
                "parallel_agent1_bottom_match": bool(external_info.get("agent1_bottom_match", False)),
                "parallel_target_velocity_ff_valid": bool(external_info.get("target_velocity_ff_valid", False)),
                "parallel_target_velocity_ff_vx_mps": float(external_info.get("target_velocity_ff_vx_mps", 0.0)),
                "parallel_target_velocity_ff_vy_mps": float(external_info.get("target_velocity_ff_vy_mps", 0.0)),
                "parallel_target_velocity_ff_blocked_by_safety": bool(external_info.get("target_velocity_ff_blocked_by_safety", False)),
                "parallel_physical_steps": int(external_info.get("parallel_physical_steps", 0)),
                "commanded_vx_mps": vx,
                "commanded_vy_mps": vy,
                "commanded_vz_mps": vz,
                "raw_vz_action": raw_vz_action,
                "requested_vz_mps": requested_vz,
                "applied_vz_mps": vz,
                "climb_command_blocked": bool(climb_command_blocked),
                "vertical_control_state": vertical_state,
                "descent_allowed": bool(descent_allowed),
                "descent_requested": bool(descent_requested),
                "descent_blocked": bool(descent_blocked),
                "descent_block_reason": descent_block_reason,
                "commanded_yaw_rate_dps": yaw_rate,
                "horizontal_guard_reasons": guard_reasons,
                "horizontal_control_state": self._last_horizontal_control_state,
                "horizontal_pd_action_vx": float(self._last_pd_action_vx),
                "horizontal_pd_action_vy": float(self._last_pd_action_vy),
                "horizontal_residual_action_vx": float(self._last_residual_action_vx),
                "horizontal_residual_action_vy": float(self._last_residual_action_vy),
                "horizontal_final_action_vx": float(self._last_horizontal_action_vx),
                "horizontal_final_action_vy": float(self._last_horizontal_action_vy),
                "horizontal_speed_limit_mps": float(self._last_horizontal_speed_limit_mps),
                "target_velocity_valid": bool(self._target_velocity_valid),
                "target_velocity_age_s": float(self._target_velocity_age_s),
                "target_velocity_world_x_mps": float(self._target_velocity_world_x_mps),
                "target_velocity_world_y_mps": float(self._target_velocity_world_y_mps),
                "target_velocity_body_vx_mps": float(self._target_velocity_body_vx_mps),
                "target_velocity_body_vy_mps": float(self._target_velocity_body_vy_mps),
                "target_velocity_speed_mps": float(self._target_velocity_speed_mps),
                "horizontal_velocity_ff_vx_mps": float(self._last_velocity_ff_vx_mps),
                "horizontal_velocity_ff_vy_mps": float(self._last_velocity_ff_vy_mps),
                "horizontal_correction_vx_mps": float(self._last_horizontal_correction_vx_mps),
                "horizontal_correction_vy_mps": float(self._last_horizontal_correction_vy_mps),
                "horizontal_total_speed_limit_mps": float(self._last_horizontal_total_speed_limit_mps),
                "recovery_requested": bool(recovery_decision["requested"]),
                "recovery_request_reason": str(recovery_decision["reason"]),
                "recovery_contact_guard_active": bool(recovery_decision["contact_guard_active"]),
                "recovery_contact_guard_similarity": float(recovery_decision["contact_guard_similarity"]),
                "recovery_contact_guard_margin": float(recovery_decision["contact_guard_margin"]),
                "alignment_ready_streak": int(getattr(self, "_alignment_ready_streak", 0)),
                "descent_alignment_latched": bool(getattr(self, "_descent_alignment_latched", False)),
                "reward_bank": float(self._reward_bank),
                "landing_shaping_bank_delta": shaping,
                "collision_new": collision_now,
                "collision_object_name": collision_object,
                "collision_timestamp": collision_timestamp,
                "collision_good_xy": good_xy,
                "collision_alignment_success_path": collision_decision["success_path"],
                "collision_live_match_at_contact": collision_decision["live_match_at_collision"],
                "collision_contact_appearance_valid": collision_decision["contact_appearance_valid"],
                "collision_contact_center_similarity": collision_decision["contact_center_similarity"],
                "collision_contact_corner_similarity": collision_decision["contact_corner_similarity"],
                "collision_contact_center_margin": collision_decision["contact_center_margin"],
                "collision_contact_best_center_scale": collision_decision["contact_best_center_scale"],
                "collision_contact_appearance_reason": collision_decision["contact_appearance_reason"],
                "collision_alignment_latch_age": collision_decision["alignment_latch_age"],
                "collision_last_verified_center_error": collision_decision["last_verified_center_error"],
                "collision_last_verified_bbox_rel_error": collision_decision["last_verified_bbox_rel_error"],
                "collision_last_verified_similarity": collision_decision["last_verified_similarity"],
                "collision_authorized_descent_latched": collision_decision["authorized_descent_latched"],
                "collision_authorized_descent_latch_age": collision_decision["authorized_descent_latch_age"],
                "collision_last_authorized_descent_center_error": collision_decision["last_authorized_descent_center_error"],
                "collision_last_authorized_descent_bbox_rel_error": collision_decision["last_authorized_descent_bbox_rel_error"],
                "collision_last_authorized_descent_similarity": collision_decision["last_authorized_descent_similarity"],
                "collision_last_authorized_descent_vz_mps": collision_decision["last_authorized_descent_vz_mps"],
                "collision_authorized_descent_invalidated_reason": collision_decision["authorized_descent_invalidated_reason"],
                "collision_object_matches_target": collision_decision["collision_object_matches_target"],
                "collision_object_lock_created": collision_decision["collision_object_lock_created"],
                "expected_collision_object_name": collision_decision["expected_collision_object_name"],
                "expected_collision_object_source": collision_decision["expected_collision_object_source"],
                "collision_reject_reason": collision_decision["reject_reason"],
                "termination_reason": reason,
                "episode_return": float(self._episode_return),
            }
        )

        if done:
            total_perception = max(
                1,
                self._episode_live_match_steps
                + self._episode_predicted_steps
                + self._episode_no_target_steps,
            )
            live_pct = 100.0 * self._episode_live_match_steps / total_perception
            pred_pct = 100.0 * self._episode_predicted_steps / total_perception
            lost_pct = 100.0 * self._episode_no_target_steps / total_perception
            best_center = (
                self._episode_best_center_error
                if np.isfinite(self._episode_best_center_error)
                else 999.0
            )
            print(
                f"[A2 EP] steps={self._step} result={reason or 'running'} "
                f"reward={reward:+.1f} bank={self._reward_bank:+.1f} "
                f"live={live_pct:.1f}% pred={pred_pct:.1f}% lost={lost_pct:.1f}% "
                f"xy={self._last_horizontal_control_state} recenter={int(getattr(self, '_episode_recenter_steps', 0))} "
                f"xyHold={int(getattr(self, '_episode_xy_hold_steps', 0))} "
                f"descent={self._episode_descent_allowed_steps}/"
                f"{self._episode_descent_requested_steps} "
                f"blocked={self._episode_descent_blocked_steps} "
                f"climbBlocked={self._episode_climb_command_blocked_steps} "
                f"centerFinal={center_error:.3f} centerBest={best_center:.3f} "
                f"simBest={self._episode_best_similarity:.3f} "
                f"A1cam={external_info.get('agent1_active_camera', 'n/a')} "
                f"A1mode={external_info.get('agent1_tracking_mode', 'n/a')} "
                f"A1fusion={int(bool(external_info.get('agent1_fusion_has_target', False)))} "
                f"FF=({float(external_info.get('target_velocity_ff_vx_mps', 0.0)):+.2f},"
                f"{float(external_info.get('target_velocity_ff_vy_mps', 0.0)):+.2f}) "
                f"collision={collision_object or 'none'} "
                f"liveAtTouch={int(bool(collision_decision['live_match_at_collision']))} "
                f"contact={int(bool(collision_decision['contact_appearance_valid']))} "
                f"contactSim={float(collision_decision['contact_center_similarity']):.3f} "
                f"cornerSim={float(collision_decision['contact_corner_similarity']):.3f} "
                f"contactMargin={float(collision_decision['contact_center_margin']):+.3f} "
                f"contactScale={float(collision_decision['contact_best_center_scale']):.2f} "
                f"latchAge={int(collision_decision['alignment_latch_age'])} "
                f"lastCenter={float(collision_decision['last_verified_center_error']):.3f} "
                f"lastBBoxRel={float(collision_decision['last_verified_bbox_rel_error']):.3f} "
                f"lastSim={float(collision_decision['last_verified_similarity']):.3f} "
                f"authAge={int(collision_decision['authorized_descent_latch_age'])} "
                f"authCenter={float(collision_decision['last_authorized_descent_center_error']):.3f} "
                f"authBBoxRel={float(collision_decision['last_authorized_descent_bbox_rel_error']):.3f} "
                f"authSim={float(collision_decision['last_authorized_descent_similarity']):.3f} "
                f"authVz={float(collision_decision['last_authorized_descent_vz_mps']):.3f} "
                f"authActive={int(bool(collision_decision['authorized_descent_latched']))} "
                f"successPath={collision_decision['success_path']} "
                f"targetCollision={int(bool(collision_decision['collision_object_matches_target']))} "
                f"reject={collision_decision['reject_reason'] or 'none'} "
                f"authReject={collision_decision['authorized_descent_invalidated_reason'] or 'none'}"
            )

        return obs, float(reward), bool(done), False, info

    def register_recovery_penalty(self, penalty: float) -> None:
        """Keep Agent-2 episode diagnostics consistent with wrapper rewards."""
        value = max(0.0, float(penalty))
        self._episode_return -= value

    def close(self):
        try:
            cv2.destroyWindow("Agent 2 - Bottom Landing")
        except Exception:
            pass
