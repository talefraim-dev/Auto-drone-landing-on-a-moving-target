"""Run_diag v3: comprehensive, non-invasive diagnostics for the current system.

Run from RL_training:
    python Run_diag.py

The wrapper imports and executes the existing Run_train.main() unchanged. It
monkey-patches observation-only diagnostics around the live implementation and
always packages the result into RL_training/diag.zip.

This version is focused on the current parallel dual-agent landing problem. It
records, on the same control timeline:
- Bottom LIVE/PRED/LOST transitions and every landing-lock gate decision.
- Current and predicted image error, image velocity, horizon and catch-up state.
- Agent-1 XY request, target-velocity feed-forward, Bottom correction, fused XY,
  final AirSim command, safety scaling and LiDAR distances.
- Agent-2 raw/requested/applied Z, descent blockers, prohibited climb requests,
  NED-Z values and relative-height sign checks.
- Adaptive appearance-bank changes, candidates, ResNet evidence and frames at
  every important state transition.
- Active checkpoints, source/config snapshots, runtime-created TensorBoard/log
  files, code-audit findings and an automatically generated diagnosis report.

Important:
- No control, reward, tracker or training decision is reimplemented here.
- Model/checkpoint writes are suppressed during diagnostics.
- Ctrl+C, an exception, or a diagnostic limit still produces diag.zip.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
import weakref
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


# Keep the original verbose Agent-1 console stream during diagnostics. Normal
# Run_train.py remains concise; the diagnostic archive still captures every
# legacy line and every structured state snapshot.
os.environ["DRONE_DIAG_VERBOSE"] = "1"


# ======================================================================================
# Diagnostic limits and output policy
# ======================================================================================
DIAG_VERSION = "3.0-parallel-control-authority"
MAX_PHYSICAL_STEPS = 7_000
MAX_TRAINABLE_STEPS = 7_000
MAX_WALL_SECONDS = 600.0
# At the current simulator rate this normally gives a frame pair every few
# seconds. Important state changes are captured independently of this interval.
FRAME_SAVE_EVERY = 10
YOLO_ANNOTATED_SAVE_EVERY = 10
OBJECT_STATE_EVERY = 1
FULL_RESNET_EMBEDDINGS = True
METHOD_CALL_LOG_LIMIT = 2_000_000
OBJECT_CREATION_LOG_LIMIT = 500_000
RUNTIME_ARTIFACT_MAX_FILE_BYTES = 25 * 1024 * 1024
RUNTIME_ARTIFACT_TOTAL_BYTES = 150 * 1024 * 1024
EVENT_FRAME_MIN_STEP_GAP = 1

BASE_DIR = Path(__file__).resolve().parent
DIAG_ZIP = BASE_DIR / "diag.zip"
DIAG_PARENT = BASE_DIR / "diagnostics"
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = DIAG_PARENT / f"diag_{RUN_STAMP}"


class DiagnosticStop(RuntimeError):
    """Internal stop signal used only to end a diagnostic run cleanly."""


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fp = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.RLock()
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.fp.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.count += 1

    def close(self) -> None:
        try:
            self.fp.flush()
            self.fp.close()
        except Exception:
            pass


class DiagnosticSession:
    LOG_FILES = {
        "events": "01_events.jsonl",
        "objects": "02_object_lifecycle.jsonl",
        "object_state": "03_object_state_snapshots.jsonl",
        "flow": "04_run_train_flow.jsonl",
        "sb3": "05_sb3.jsonl",
        "agent1_steps": "06_agent1_env_steps.jsonl",
        "agent2_steps": "07_agent2_env_steps.jsonl",
        "control": "08_control_commands.jsonl",
        "observations": "09_observations.jsonl",
        "rewards": "10_rewards.jsonl",
        "handoff": "11_handoff.jsonl",
        "tracker": "12_tracker_state.jsonl",
        "yolo_raw": "13_yolo_raw.jsonl",
        "yolo_candidates": "14_yolo_candidates.jsonl",
        "resnet_embeddings": "15_resnet_embeddings.jsonl",
        "resnet_similarity": "16_resnet_similarity.jsonl",
        "lidar": "17_lidar.jsonl",
        "airsim": "18_airsim_calls.jsonl",
        "physics": "19_physics.jsonl",
        "frames": "20_frames.jsonl",
        "methods": "21_component_method_calls.jsonl",
        "exceptions": "22_exceptions.jsonl",
        "config": "23_config_runtime_changes.jsonl",
        "identity": "24_user_target_identity.jsonl",
        "horizontal": "25_agent2_horizontal_controller.jsonl",
        "parallel": "26_parallel_xy_arbitration.jsonl",
        "landing_gate": "27_landing_lock_and_z_gate.jsonl",
        "predictive": "28_bottom_predictive_controller.jsonl",
        "vertical": "29_vertical_authority_and_height.jsonl",
        "safety_detail": "30_safety_lidar_command_suppression.jsonl",
        "fusion": "31_camera_fusion_authority.jsonl",
        "kinematics": "32_target_drone_kinematics.jsonl",
        "bookmarks": "33_critical_event_bookmarks.jsonl",
        "checkpoints": "34_checkpoint_loading.jsonl",
        "audit": "35_static_code_audit.jsonl",
        "adaptive_bank": "36_adaptive_embedding_bank.jsonl",
    }

    def __init__(self) -> None:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "frames" / "front").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "frames" / "bottom").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "frames" / "yolo_raw").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "frames" / "events").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "source_snapshot").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "active_checkpoints").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "runtime_artifacts").mkdir(parents=True, exist_ok=True)

        self.start_monotonic = time.monotonic()
        self.start_wall = time.time()
        self.physical_steps = 0
        self.trainable_steps = 0
        self.agent1_steps = 0
        self.agent2_steps = 0
        self.reset_count = 0
        self.stop_reason = "running"
        self.stop_requested = False
        self.failure: BaseException | None = None
        self.lock = threading.RLock()
        self.writers = {
            name: JsonlWriter(RUN_DIR / filename)
            for name, filename in self.LOG_FILES.items()
        }
        self.event_counts: Counter[str] = Counter()
        self.method_counts: Counter[str] = Counter()
        self.method_total_ms: defaultdict[str, float] = defaultdict(float)
        self.object_count = 0
        self.method_log_count = 0
        self.registry: dict[int, dict[str, Any]] = {}
        self.last_modes: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        self.last_agent = "UNKNOWN"
        self.last_command: dict[str, Any] = {}
        self.last_multirotor_state: Any = None
        self.last_object_poses: dict[str, Any] = {}
        self.control_ticks = 0
        self.last_agent1_step_info: dict[str, Any] = {}
        self.last_agent2_step_info: dict[str, Any] = {}
        self.last_parallel_override: dict[str, Any] = {}
        self.last_parallel_fuse: dict[str, Any] = {}
        self.last_predictive_guidance: dict[str, Any] = {}
        self.last_vertical_gate: dict[str, Any] = {}
        self.last_horizontal_servo: dict[str, Any] = {}
        self.last_critical_state: dict[str, Any] = {}
        self.last_event_frame_step = -999999
        self.transition_counts: Counter[str] = Counter()
        self.lock_block_counts: Counter[str] = Counter()
        self.safety_reason_counts: Counter[str] = Counter()
        self.anomaly_counts: Counter[str] = Counter()
        self.loaded_checkpoints: list[dict[str, Any]] = []
        self.collision_events: list[dict[str, Any]] = []
        self.episode_results: list[dict[str, Any]] = []
        self.timeline_fp = (RUN_DIR / "00_master_timeline.csv").open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.timeline_fields = [
            "wall_s", "physical_step", "trainable_step", "agent", "episode_step",
            "reward", "done", "truncated", "termination_reason", "tracker_mode",
            "front_mode", "bottom_mode", "bottom_match", "bottom_similarity",
            "bottom_err_x", "bottom_err_y", "bottom_bbox_rel_err", "speed_stage",
            "horizontal_control_state", "horizontal_pd_ax", "horizontal_pd_ay",
            "horizontal_residual_ax", "horizontal_residual_ay",
            "horizontal_final_ax", "horizontal_final_ay", "horizontal_speed_limit_mps",
            "alignment_ready_streak", "descent_alignment_latched",
            "handoff_ready", "alt_agl_m", "drone_z_ned", "target_surface_z_ned",
            "relative_height_to_target_m", "vertical_control_state", "raw_vz_action",
            "requested_vz_mps", "applied_vz_mps", "climb_command_blocked",
            "descent_allowed", "descent_blocked", "cmd_vx", "cmd_vy", "cmd_vz",
            "cmd_yaw_rate", "actual_vx",
            "actual_vy", "actual_vz", "collision", "collision_object",
            "collision_success_path", "collision_live_match_at_contact",
            "collision_alignment_latch_age", "collision_last_verified_center_error",
            "collision_last_verified_bbox_rel_error", "collision_last_verified_similarity",
            "collision_object_matches_target", "expected_collision_object_name",
            "collision_reject_reason",
        ]
        self.timeline = csv.DictWriter(self.timeline_fp, fieldnames=self.timeline_fields)
        self.timeline.writeheader()

        self.critical_fp = (RUN_DIR / "00_control_authority_timeline.csv").open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.critical_fields = [
            "wall_s", "control_tick", "physical_step", "trainable_step", "episode_step",
            "tracker_mode", "bottom_live", "bottom_confirmed", "bottom_similarity",
            "bottom_reject_reason", "candidate_count", "adaptive_anchor_count",
            "adaptive_updates", "err_x", "err_y", "center_error", "bbox_rel_error",
            "bbox_area", "image_vel_x", "image_vel_y", "control_dt_s",
            "predicted_err_x", "predicted_err_y", "predicted_center_error",
            "prediction_horizon_s", "outward_speed_per_s", "catchup_active",
            "catchup_release_streak", "no_live_duration_s", "lost_steps",
            "landing_lock", "alignment_streak", "lock_visual_gap_steps",
            "lock_bad_live_steps", "lock_age_steps", "vertical_gate_state",
            "descent_allowed", "descent_block_reason", "raw_vz_action",
            "requested_vz_mps", "applied_vz_mps", "climb_command_blocked",
            "drone_z_ned", "target_surface_z_ned", "relative_height_m", "alt_agl_m",
            "target_velocity_valid", "target_velocity_age_s", "target_vel_body_vx",
            "actual_drone_vx", "actual_drone_vy", "actual_drone_vz",
            "drone_world_x", "drone_world_y", "drone_world_z",
            "target_world_x", "target_world_y", "target_world_z",
            "relative_world_x", "relative_world_y", "relative_world_z",
            "target_vel_body_vy", "target_speed_mps", "agent1_tracking_mode",
            "agent1_active_camera", "agent1_fusion_has_target", "agent1_bottom_match",
            "agent1_xy_weight", "agent1_pre_fuse_vx", "agent1_pre_fuse_vy",
            "ff_requested_vx", "ff_requested_vy", "ff_applied_vx", "ff_applied_vy",
            "bottom_requested_vx", "bottom_requested_vy", "bottom_applied_vx",
            "bottom_applied_vy", "fused_expected_vx", "fused_expected_vy",
            "physical_cmd_vx", "physical_cmd_vy", "physical_cmd_vz",
            "horizontal_speed_limit_mps", "safety_intervention", "safety_reasons",
            "ff_bottom_blocked_by_safety", "front_dist_m", "back_dist_m",
            "left_dist_m", "right_dist_m", "collision", "collision_object",
            "termination_reason", "reward", "anomalies",
        ]
        self.critical = csv.DictWriter(
            self.critical_fp, fieldnames=self.critical_fields, extrasaction="ignore"
        )
        self.critical.writeheader()

    @property
    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.start_monotonic)

    def base(self, event: str) -> dict[str, Any]:
        return {
            "ts_unix": time.time(),
            "elapsed_s": round(self.elapsed, 6),
            "event": event,
            "physical_step": self.physical_steps,
            "trainable_step": self.trainable_steps,
            "thread": threading.current_thread().name,
        }

    def write(self, log_name: str, event: str, **fields: Any) -> None:
        row = self.base(event)
        row.update({key: safe_value(value) for key, value in fields.items()})
        writer = self.writers.get(log_name)
        if writer is not None:
            writer.write(row)
        self.event_counts[f"{log_name}:{event}"] += 1

    def register_object(self, obj: Any, created_by: str = "", args: Any = None, kwargs: Any = None) -> None:
        if obj is None:
            return
        oid = id(obj)
        if oid in self.registry:
            return
        with self.lock:
            if oid in self.registry:
                return
            self.object_count += 1
            record = {
                "object_id": hex(oid),
                "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "created_by": created_by,
                "creation_index": self.object_count,
            }
            self.registry[oid] = record
            if self.object_count <= OBJECT_CREATION_LOG_LIMIT:
                self.write(
                    "objects",
                    "object_created",
                    **record,
                    args=args,
                    kwargs=kwargs,
                    initial_state=snapshot_object(obj),
                )

    def snapshot_registered(self, trigger: str) -> None:
        if OBJECT_STATE_EVERY <= 0 or self.physical_steps % OBJECT_STATE_EVERY != 0:
            return
        for oid, obj in list(_OBJECT_REFERENCES.items()):
            meta = self.registry.get(oid)
            if meta is None:
                continue
            self.write(
                "object_state",
                "object_state",
                trigger=trigger,
                object_id=meta["object_id"],
                class_name=meta["class"],
                state=snapshot_object(obj),
            )

    def check_limits(self) -> None:
        reason = None
        if self.physical_steps >= MAX_PHYSICAL_STEPS:
            reason = f"physical_step_limit_{MAX_PHYSICAL_STEPS}"
        elif self.trainable_steps >= MAX_TRAINABLE_STEPS:
            reason = f"trainable_step_limit_{MAX_TRAINABLE_STEPS}"
        elif self.elapsed >= MAX_WALL_SECONDS:
            reason = f"wall_time_limit_{int(MAX_WALL_SECONDS)}s"
        if reason:
            self.stop_requested = True
            self.stop_reason = reason
            raise DiagnosticStop(reason)

    def increment_physical(self, agent: str) -> None:
        self.physical_steps += 1
        self.last_agent = agent
        if agent == "AGENT_1":
            self.agent1_steps += 1
        elif agent == "AGENT_2":
            self.agent2_steps += 1

    def write_timeline(self, agent: str, reward: Any, done: Any, truncated: Any, info: dict[str, Any]) -> None:
        self.last_info = dict(info or {})
        actual = extract_actual_velocity_from_info(info)
        row = {
            "wall_s": round(self.elapsed, 6),
            "physical_step": self.physical_steps,
            "trainable_step": self.trainable_steps,
            "agent": agent,
            "episode_step": first_value(info, "episode_steps", "step", "agent_step", default=""),
            "reward": scalar_or_empty(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "termination_reason": first_value(info, "termination_reason", "reason", default=""),
            "tracker_mode": first_value(info, "tracker_mode", "tracking_mode", default=""),
            "front_mode": first_value(info, "front_tracking_mode", "front_mode", "FF", default=""),
            "bottom_mode": first_value(info, "bottom_tracking_mode", "bottom_mode", "BF", default=""),
            "bottom_match": first_value(info, "bottom_match", default=""),
            "bottom_similarity": first_value(info, "bottom_similarity", "bottom_match_score", default=""),
            "bottom_err_x": first_value(info, "bottom_err_x", "bottom_center_x", default=""),
            "bottom_err_y": first_value(info, "bottom_err_y", "bottom_center_y", default=""),
            "bottom_bbox_rel_err": first_value(info, "bottom_bbox_rel_err", default=""),
            "speed_stage": first_value(info, "speed_stage", default=""),
            "horizontal_control_state": first_value(info, "horizontal_control_state", default=""),
            "horizontal_pd_ax": first_value(info, "horizontal_pd_action_vx", default=""),
            "horizontal_pd_ay": first_value(info, "horizontal_pd_action_vy", default=""),
            "horizontal_residual_ax": first_value(info, "horizontal_residual_action_vx", default=""),
            "horizontal_residual_ay": first_value(info, "horizontal_residual_action_vy", default=""),
            "horizontal_final_ax": first_value(info, "horizontal_final_action_vx", default=""),
            "horizontal_final_ay": first_value(info, "horizontal_final_action_vy", default=""),
            "horizontal_speed_limit_mps": first_value(info, "horizontal_speed_limit_mps", default=""),
            "alignment_ready_streak": first_value(info, "alignment_ready_streak", default=""),
            "descent_alignment_latched": first_value(info, "descent_alignment_latched", default=""),
            "handoff_ready": first_value(info, "handoff_ready", default=""),
            "alt_agl_m": first_value(info, "alt_agl_m", "altitude_m", default=""),
            "drone_z_ned": first_value(info, "drone_z_ned", default=""),
            "target_surface_z_ned": first_value(info, "target_surface_z_ned", default=""),
            "relative_height_to_target_m": first_value(info, "relative_height_to_target_m", default=""),
            "vertical_control_state": first_value(info, "vertical_control_state", default=""),
            "raw_vz_action": first_value(info, "raw_vz_action", default=""),
            "requested_vz_mps": first_value(info, "requested_vz_mps", default=""),
            "applied_vz_mps": first_value(info, "applied_vz_mps", default=""),
            "climb_command_blocked": first_value(info, "climb_command_blocked", default=""),
            "descent_allowed": first_value(info, "descent_allowed", default=""),
            "descent_blocked": first_value(info, "descent_blocked", default=""),
            "cmd_vx": first_value(info, "commanded_vx_mps", "cmd_vx", "final_vx", default=self.last_command.get("vx", "")),
            "cmd_vy": first_value(info, "commanded_vy_mps", "cmd_vy", "final_vy", default=self.last_command.get("vy", "")),
            "cmd_vz": first_value(info, "commanded_vz_mps", "cmd_vz", "final_vz", default=self.last_command.get("vz", "")),
            "cmd_yaw_rate": first_value(info, "commanded_yaw_rate_dps", "yaw_rate_cmd_dps", default=self.last_command.get("yaw_rate", "")),
            "actual_vx": actual.get("vx", ""),
            "actual_vy": actual.get("vy", ""),
            "actual_vz": actual.get("vz", ""),
            "collision": first_value(info, "collision_new", "collision", "has_collided", default=""),
            "collision_object": first_value(info, "collision_object_name", "collision_object", default=""),
            "collision_success_path": first_value(info, "collision_alignment_success_path", default=""),
            "collision_live_match_at_contact": first_value(info, "collision_live_match_at_contact", default=""),
            "collision_alignment_latch_age": first_value(info, "collision_alignment_latch_age", default=""),
            "collision_last_verified_center_error": first_value(info, "collision_last_verified_center_error", default=""),
            "collision_last_verified_bbox_rel_error": first_value(info, "collision_last_verified_bbox_rel_error", default=""),
            "collision_last_verified_similarity": first_value(info, "collision_last_verified_similarity", default=""),
            "collision_object_matches_target": first_value(info, "collision_object_matches_target", default=""),
            "expected_collision_object_name": first_value(info, "expected_collision_object_name", default=""),
            "collision_reject_reason": first_value(info, "collision_reject_reason", default=""),
        }
        try:
            self.timeline.writerow(row)
        except Exception as exc:
            self.write("exceptions", "timeline_write_failed", error=repr(exc))

    def close(self) -> None:
        try:
            self.timeline_fp.flush()
            self.timeline_fp.close()
        except Exception:
            pass
        try:
            self.critical_fp.flush()
            self.critical_fp.close()
        except Exception:
            pass
        for writer in self.writers.values():
            writer.close()


SESSION: DiagnosticSession | None = None
_OBJECT_REFERENCES: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_PATCHES: list[tuple[Any, str, Any]] = []


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_or_array_summary(value: Any, full: bool = False) -> dict[str, Any]:
    try:
        import numpy as np
        if hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
            source = "torch"
        else:
            arr = np.asarray(value)
            source = "numpy"
        flat = arr.reshape(-1)
        finite = flat[np.isfinite(flat)] if flat.size else flat
        raw = arr.tobytes()
        result: dict[str, Any] = {
            "source": source,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(flat.size),
            "sha256": sha256_bytes(raw),
        }
        if finite.size:
            result.update(
                min=float(finite.min()),
                max=float(finite.max()),
                mean=float(finite.mean()),
                std=float(finite.std()),
                l2=float(np.linalg.norm(finite.astype(np.float64))),
            )
        result["head"] = [clean_float(x) for x in flat[:32].tolist()]
        if full or flat.size <= 64:
            result["values"] = [clean_float(x) for x in flat.tolist()]
        return result
    except Exception as exc:
        return {"type": type(value).__name__, "summary_error": repr(exc)}


def clean_float(value: Any) -> Any:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    return f


def safe_value(value: Any, depth: int = 0, *, full_tensor: bool = False) -> Any:
    if depth > 5:
        return f"<{type(value).__name__}:depth-limit>"
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return clean_float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": sha256_bytes(value)}

    try:
        import numpy as np
        if isinstance(value, np.generic):
            return safe_value(value.item(), depth + 1)
        if isinstance(value, np.ndarray):
            return tensor_or_array_summary(value, full=full_tensor)
    except Exception:
        pass

    try:
        import torch
        if isinstance(value, torch.Tensor):
            return tensor_or_array_summary(value, full=full_tensor)
    except Exception:
        pass

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return safe_value(dataclasses.asdict(value), depth + 1)
        except Exception:
            pass
    if isinstance(value, dict):
        out = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 500:
                out["<truncated>"] = len(value) - 500
                break
            out[str(key)] = safe_value(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        result = [safe_value(item, depth + 1) for item in seq[:500]]
        if len(seq) > 500:
            result.append({"truncated": len(seq) - 500})
        return result

    # AirSim and lightweight third-party structures often expose public fields.
    public = {}
    for attr in (
        "x_val", "y_val", "z_val", "w_val", "position", "orientation",
        "linear_velocity", "linear_acceleration", "angular_velocity",
        "angular_acceleration", "kinematics_estimated", "gps_location",
        "timestamp", "time_stamp", "has_collided", "object_name", "impact_point",
        "normal", "penetration_depth", "height", "width", "camera_position",
        "camera_orientation", "image_type", "pixels_as_float", "compress",
        "point_cloud", "pose", "boxes", "bbox", "cls_id", "conf",
        "appearance_score", "motion_score", "total_score",
    ):
        if hasattr(value, attr):
            try:
                item = getattr(value, attr)
                if attr == "point_cloud":
                    public[attr] = tensor_or_array_summary(item, full=False)
                elif attr == "boxes":
                    public[attr] = summarize_yolo_boxes(item)
                else:
                    public[attr] = safe_value(item, depth + 1)
            except Exception as exc:
                public[attr] = f"<error:{exc}>"
    if public:
        public["_type"] = f"{type(value).__module__}.{type(value).__qualname__}"
        return public

    try:
        return repr(value)[:2000]
    except Exception:
        return f"<{type(value).__name__}>"


def snapshot_object(obj: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    try:
        items = vars(obj).items()
    except Exception:
        return {"repr": safe_value(obj)}
    heavy_names = {
        "client", "model", "agent1_model", "yolo", "resnet", "preprocess",
        "policy", "rollout_buffer", "logger",
    }
    reference_names = {
        "tracker", "core", "observation_builder", "lidar_processor",
        "agent1_env", "agent2_env", "cfg",
    }
    for index, (name, value) in enumerate(items):
        if index >= 500:
            state["<truncated>"] = len(vars(obj)) - 500
            break
        if name in heavy_names:
            state[name] = {
                "object_id": hex(id(value)),
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        elif name in reference_names and value is not None:
            state[name] = {
                "object_id": hex(id(value)),
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": safe_value(value) if dataclasses.is_dataclass(value) else None,
            }
        else:
            state[name] = safe_value(value)
    return state


def first_value(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def scalar_or_empty(value: Any) -> Any:
    try:
        return clean_float(value)
    except Exception:
        return ""


def extract_actual_velocity_from_info(info: dict[str, Any]) -> dict[str, Any]:
    for key in ("drone_state", "api_state", "kinematics", "kinematics_estimated"):
        value = info.get(key)
        if value is None:
            continue
        obj = value
        if hasattr(obj, "kinematics_estimated"):
            obj = obj.kinematics_estimated
        vel = getattr(obj, "linear_velocity", None)
        if vel is not None:
            return {
                "vx": getattr(vel, "x_val", ""),
                "vy": getattr(vel, "y_val", ""),
                "vz": getattr(vel, "z_val", ""),
            }
    return {}


def summarize_yolo_boxes(boxes: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    try:
        if boxes is None:
            return output
        for index, box in enumerate(boxes):
            if index >= 500:
                break
            xyxy = getattr(box, "xyxy", None)
            cls = getattr(box, "cls", None)
            conf = getattr(box, "conf", None)
            output.append(
                {
                    "index": index,
                    "xyxy": tensor_or_array_summary(xyxy, full=True) if xyxy is not None else None,
                    "cls": tensor_or_array_summary(cls, full=True) if cls is not None else None,
                    "conf": tensor_or_array_summary(conf, full=True) if conf is not None else None,
                }
            )
    except Exception as exc:
        output.append({"error": repr(exc)})
    return output


def class_name_from_model(model: Any, cls_id: int) -> str:
    try:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
    except Exception:
        pass
    return str(cls_id)


def patch_attr(owner: Any, name: str, replacement: Any) -> None:
    original = getattr(owner, name)
    _PATCHES.append((owner, name, original))
    setattr(owner, name, replacement)


def restore_patches() -> None:
    for owner, name, original in reversed(_PATCHES):
        try:
            setattr(owner, name, original)
        except Exception:
            pass
    _PATCHES.clear()


def log_exception(where: str, exc: BaseException) -> None:
    if SESSION is None:
        return
    SESSION.write(
        "exceptions",
        "exception",
        where=where,
        exception_type=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def wrap_project_class_initializers(modules: Iterable[Any]) -> None:
    """Log construction of every class defined in project modules.

    This is intentionally limited to __init__ and does not add a line tracer,
    because line tracing would materially change control-loop timing.
    """
    seen: set[type] = set()
    for module in modules:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            if BASE_DIR not in Path(module_file).resolve().parents and Path(module_file).resolve() != BASE_DIR:
                continue
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen or cls.__module__ != module.__name__:
                continue
            seen.add(cls)
            original = cls.__dict__.get("__init__")
            if original is None or getattr(original, "_diag_wrapped", False):
                continue

            @functools.wraps(original)
            def init_wrapper(self, *args, __orig=original, __cls=cls, **kwargs):
                started = time.perf_counter()
                try:
                    __orig(self, *args, **kwargs)
                except BaseException as exc:
                    log_exception(f"{__cls.__module__}.{__cls.__qualname__}.__init__", exc)
                    raise
                finally:
                    if SESSION is not None:
                        try:
                            _OBJECT_REFERENCES[id(self)] = self
                        except TypeError:
                            pass
                        SESSION.register_object(
                            self,
                            created_by=f"{__cls.__module__}.{__cls.__qualname__}.__init__",
                            args=args,
                            kwargs=kwargs,
                        )
                        SESSION.write(
                            "methods",
                            "constructor",
                            class_name=f"{__cls.__module__}.{__cls.__qualname__}",
                            object_id=hex(id(self)),
                            duration_ms=(time.perf_counter() - started) * 1000.0,
                        )
            init_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
            patch_attr(cls, "__init__", init_wrapper)


def wrap_method(
    cls: type,
    name: str,
    *,
    log_name: str = "methods",
    event: str | None = None,
    post: Callable[[Any, tuple[Any, ...], dict[str, Any], Any], None] | None = None,
    pre: Callable[[Any, tuple[Any, ...], dict[str, Any]], None] | None = None,
) -> None:
    if not hasattr(cls, name):
        if SESSION is not None:
            SESSION.write("events", "instrumentation_method_missing", class_name=str(cls), method=name)
        return
    original = getattr(cls, name)
    if getattr(original, "_diag_wrapped", False):
        return
    event_name = event or f"{cls.__name__}.{name}"

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        started = time.perf_counter()
        if pre is not None:
            try:
                pre(self, args, kwargs)
            except Exception as exc:
                log_exception(f"diagnostic_pre:{event_name}", exc)
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            log_exception(event_name, exc)
            raise
        duration_ms = (time.perf_counter() - started) * 1000.0
        if SESSION is not None:
            key = f"{cls.__module__}.{cls.__qualname__}.{name}"
            SESSION.method_counts[key] += 1
            SESSION.method_total_ms[key] += duration_ms
            if SESSION.method_log_count < METHOD_CALL_LOG_LIMIT:
                SESSION.method_log_count += 1
                SESSION.write(
                    log_name,
                    event_name,
                    object_id=hex(id(self)),
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    duration_ms=duration_ms,
                    object_state=snapshot_object(self),
                )
        if post is not None:
            try:
                post(self, args, kwargs, result)
            except DiagnosticStop:
                raise
            except Exception as exc:
                log_exception(f"diagnostic_post:{event_name}", exc)
        return result

    wrapper._diag_wrapped = True  # type: ignore[attr-defined]
    patch_attr(cls, name, wrapper)


def frame_stats(frame: Any) -> dict[str, Any]:
    try:
        import numpy as np
        arr = np.asarray(frame)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "min": clean_float(arr.min()) if arr.size else None,
            "max": clean_float(arr.max()) if arr.size else None,
            "mean": clean_float(arr.mean()) if arr.size else None,
            "std": clean_float(arr.std()) if arr.size else None,
            "sha256": sha256_bytes(arr.tobytes()),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def save_frame(frame: Any, kind: str, label: str = "") -> str | None:
    if SESSION is None:
        return None
    step = SESSION.physical_steps
    should_save = step <= 5 or step % FRAME_SAVE_EVERY == 0
    if not should_save:
        SESSION.write("frames", "frame_metadata", kind=kind, label=label, stats=frame_stats(frame), saved=False)
        return None
    try:
        import cv2
        import numpy as np
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.size == 0:
            return None
        folder = RUN_DIR / "frames" / kind
        folder.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:60]
        path = folder / f"step_{step:07d}_{safe_label or kind}.jpg"
        cv2.imwrite(str(path), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        SESSION.write("frames", "frame_saved", kind=kind, label=label, stats=frame_stats(arr), path=str(path.relative_to(RUN_DIR)), saved=True)
        return str(path)
    except Exception as exc:
        log_exception("save_frame", exc)
        return None


def install_yolo_and_resnet_instrumentation(resnet_module: Any) -> None:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        SESSION.write("events", "ultralytics_patch_skipped", error=repr(exc))
    else:
        original_predict = YOLO.predict

        @functools.wraps(original_predict)
        def predict_wrapper(self, source=None, *args, **kwargs):
            started = time.perf_counter()
            try:
                result = original_predict(self, source, *args, **kwargs)
            except BaseException as exc:
                log_exception("ultralytics.YOLO.predict", exc)
                raise
            detections: list[dict[str, Any]] = []
            try:
                for result_index, item in enumerate(result or []):
                    boxes = getattr(item, "boxes", None)
                    if boxes is None:
                        continue
                    for box_index, box in enumerate(boxes):
                        xyxy = box.xyxy[0].detach().cpu().numpy().tolist() if getattr(box, "xyxy", None) is not None else None
                        cls_id = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
                        conf = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 0.0
                        detections.append(
                            {
                                "result_index": result_index,
                                "box_index": box_index,
                                "xyxy": xyxy,
                                "cls_id": cls_id,
                                "class_name": class_name_from_model(self, cls_id),
                                "confidence": conf,
                            }
                        )
            except Exception as exc:
                detections.append({"parse_error": repr(exc)})
            SESSION.write(
                "yolo_raw",
                "yolo_predict_return",
                model=repr(getattr(self, "model", self))[:1000],
                conf=kwargs.get("conf"),
                iou=kwargs.get("iou"),
                input_frame=frame_stats(source) if source is not None else None,
                detections=detections,
                detection_count=len([d for d in detections if "cls_id" in d]),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            # Save an annotated raw-detection frame periodically without changing the result.
            if source is not None and (SESSION.physical_steps <= 5 or SESSION.physical_steps % YOLO_ANNOTATED_SAVE_EVERY == 0):
                try:
                    import cv2
                    import numpy as np
                    vis = np.asarray(source).copy()
                    for det in detections:
                        if "xyxy" not in det or det["xyxy"] is None:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 1)
                        cv2.putText(vis, f"{det['cls_id']}:{det['class_name']} {det['confidence']:.3f}", (max(0, x1), max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                    save_frame(vis, "yolo_raw", "raw_detections")
                except Exception as exc:
                    log_exception("yolo_raw_annotated_frame", exc)
            return result

        predict_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
        patch_attr(YOLO, "predict", predict_wrapper)

    Tracker = resnet_module.YoloResNetTracker

    if hasattr(Tracker, "_detect_candidates"):
        original_detect = Tracker._detect_candidates

        @functools.wraps(original_detect)
        def detect_wrapper(self, frame, *args, **kwargs):
            result = original_detect(self, frame, *args, **kwargs)
            candidates = []
            for index, cand in enumerate(result or []):
                candidates.append(
                    {
                        "index": index,
                        "bbox_xywh": list(getattr(cand, "bbox", []) or []),
                        "cls_id": int(getattr(cand, "cls_id", -1)),
                        "confidence": float(getattr(cand, "conf", 0.0)),
                        "appearance_score": float(getattr(cand, "appearance_score", 0.0)),
                        "motion_score": float(getattr(cand, "motion_score", 0.0)),
                        "total_score": float(getattr(cand, "total_score", 0.0)),
                        "passes_target_class": (
                            getattr(self, "target_class_id", None) is None
                            or int(getattr(cand, "cls_id", -1)) == int(getattr(self, "target_class_id"))
                        ),
                    }
                )
            setattr(self, "_diag_last_candidates", result)
            SESSION.write(
                "yolo_candidates",
                "tracker_candidates_return",
                tracker_id=hex(id(self)),
                target_class_id=getattr(self, "target_class_id", None),
                yolo_conf=getattr(self, "yolo_conf", None),
                frame=frame_stats(frame),
                candidates=candidates,
            )
            return result

        detect_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
        patch_attr(Tracker, "_detect_candidates", detect_wrapper)

    if hasattr(Tracker, "_embedding_from_crop"):
        original_embedding = Tracker._embedding_from_crop

        @functools.wraps(original_embedding)
        def embedding_wrapper(self, crop, *args, **kwargs):
            started = time.perf_counter()
            result = original_embedding(self, crop, *args, **kwargs)
            SESSION.write(
                "resnet_embeddings",
                "resnet_embedding_return",
                tracker_id=hex(id(self)),
                target_class_id=getattr(self, "target_class_id", None),
                crop=frame_stats(crop),
                embedding=tensor_or_array_summary(result, full=FULL_RESNET_EMBEDDINGS) if result is not None else None,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            return result

        embedding_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
        patch_attr(Tracker, "_embedding_from_crop", embedding_wrapper)

    # torch.dot is the actual cosine-similarity operation used by this project.
    try:
        import torch
        original_dot = torch.dot

        @functools.wraps(original_dot)
        def dot_wrapper(input_tensor, other_tensor, *args, **kwargs):
            result = original_dot(input_tensor, other_tensor, *args, **kwargs)
            try:
                if int(input_tensor.numel()) >= 64 and int(other_tensor.numel()) >= 64:
                    caller = inspect.currentframe().f_back
                    SESSION.write(
                        "resnet_similarity",
                        "torch_dot_similarity",
                        caller_file=Path(caller.f_code.co_filename).name if caller else "",
                        caller_function=caller.f_code.co_name if caller else "",
                        lhs=tensor_or_array_summary(input_tensor, full=False),
                        rhs=tensor_or_array_summary(other_tensor, full=False),
                        similarity=clean_float(result.detach().cpu().item()),
                    )
            except Exception as exc:
                log_exception("torch_dot_diagnostics", exc)
            return result

        dot_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
        patch_attr(torch, "dot", dot_wrapper)
    except Exception as exc:
        SESSION.write("events", "torch_dot_patch_skipped", error=repr(exc))


def install_airsim_instrumentation(airsim_module: Any) -> None:
    Client = getattr(airsim_module, "MultirotorClient", None)
    if Client is None:
        SESSION.write("events", "airsim_client_missing")
        return
    method_names = [
        "confirmConnection", "enableApiControl", "armDisarm", "reset",
        "getMultirotorState", "simGetVehiclePose", "simSetVehiclePose",
        "simGetObjectPose", "simSetObjectPose", "simGetImages", "getLidarData",
        "simGetCollisionInfo", "moveByVelocityBodyFrameAsync", "moveByVelocityAsync",
        "moveToPositionAsync", "takeoffAsync", "hoverAsync", "simListSceneObjects",
        "getDistanceSensorData",
    ]
    for method_name in method_names:
        if not hasattr(Client, method_name):
            continue
        original = getattr(Client, method_name)
        if getattr(original, "_diag_wrapped", False):
            continue

        @functools.wraps(original)
        def method_wrapper(self, *args, __orig=original, __name=method_name, **kwargs):
            started = time.perf_counter()
            command = __name.startswith("move") or __name in {"takeoffAsync", "hoverAsync", "simSetVehiclePose", "simSetObjectPose"}
            if command:
                command_fields = command_from_call(__name, args, kwargs)
                SESSION.last_command = command_fields
                SESSION.write("control", "airsim_command_issued", method=__name, command=command_fields, args=args, kwargs=kwargs)
            try:
                result = __orig(self, *args, **kwargs)
            except BaseException as exc:
                log_exception(f"AirSim.{__name}", exc)
                raise
            duration = (time.perf_counter() - started) * 1000.0
            SESSION.write("airsim", "airsim_call", method=__name, args=args, kwargs=kwargs, result=result, duration_ms=duration)
            if __name == "getMultirotorState":
                SESSION.last_multirotor_state = result
                SESSION.write("physics", "multirotor_state", state=result)
            elif __name == "simGetObjectPose":
                object_name = str(kwargs.get("object_name", args[0] if args else "") or "")
                if object_name:
                    SESSION.last_object_poses[object_name] = result
                SESSION.write("physics", "object_pose", object_name=object_name, pose=result)
            elif __name == "simGetCollisionInfo":
                SESSION.write("physics", "collision_state", collision=result)
            elif __name == "getLidarData":
                SESSION.write("lidar", "raw_lidar_return", lidar=result)
            return result

        method_wrapper._diag_wrapped = True  # type: ignore[attr-defined]
        patch_attr(Client, method_name, method_wrapper)


def command_from_call(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    command: dict[str, Any] = {"method": name}
    if name in {"moveByVelocityBodyFrameAsync", "moveByVelocityAsync"}:
        keys = ["vx", "vy", "vz", "duration"]
        for index, key in enumerate(keys):
            command[key] = kwargs.get(key, args[index] if index < len(args) else None)
        yaw_mode = kwargs.get("yaw_mode", args[4] if len(args) > 4 else None)
        command["yaw_mode"] = safe_value(yaw_mode)
        try:
            command["yaw_rate"] = getattr(yaw_mode, "yaw_or_rate", None)
        except Exception:
            pass
        command["vehicle_name"] = kwargs.get("vehicle_name", "")
    else:
        command["args"] = safe_value(args)
        command["kwargs"] = safe_value(kwargs)
    return command


def env_step_post(agent: str) -> Callable[[Any, tuple[Any, ...], dict[str, Any], Any], None]:
    def post(self, args, kwargs, result):
        try:
            obs, reward, done, truncated, info = result
        except Exception:
            SESSION.write("exceptions", "unexpected_env_step_result", agent=agent, result=result)
            return
        info = dict(info or {})
        SESSION.increment_physical(agent)
        if agent == "AGENT_1":
            SESSION.last_agent1_step_info = dict(info)
        else:
            SESSION.last_agent2_step_info = dict(info)
        log_name = "agent1_steps" if agent == "AGENT_1" else "agent2_steps"
        action = args[0] if args else kwargs.get("action")
        SESSION.write(
            log_name,
            "environment_step",
            agent=agent,
            env_id=hex(id(self)),
            action=action,
            observation=obs,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
            env_state=snapshot_object(self),
        )
        SESSION.write("observations", "observation_return", agent=agent, observation=obs, info_obs_dict=info.get("obs_dict"))
        SESSION.write("rewards", "reward_return", agent=agent, reward=reward, reward_parts=extract_reward_parts(info), termination_reason=info.get("termination_reason"))
        SESSION.write("handoff", "handoff_state", agent=agent, state=extract_handoff(info))
        SESSION.write("tracker", "tracker_state_from_info", agent=agent, state=extract_tracker(info), env_tracker_state=tracker_state_from_env(self))
        if agent == "AGENT_2":
            SESSION.write("identity", "user_target_identity_step", agent=agent, state=extract_identity(info))
            SESSION.write(
                "horizontal",
                "agent2_horizontal_control_step",
                agent=agent,
                state={
                    "horizontal_control_state": info.get("horizontal_control_state"),
                    "pd_action_vx": info.get("horizontal_pd_action_vx"),
                    "pd_action_vy": info.get("horizontal_pd_action_vy"),
                    "residual_action_vx": info.get("horizontal_residual_action_vx"),
                    "residual_action_vy": info.get("horizontal_residual_action_vy"),
                    "final_action_vx": info.get("horizontal_final_action_vx"),
                    "final_action_vy": info.get("horizontal_final_action_vy"),
                    "speed_limit_mps": info.get("horizontal_speed_limit_mps"),
                    "commanded_vx_mps": info.get("commanded_vx_mps"),
                    "commanded_vy_mps": info.get("commanded_vy_mps"),
                    "bottom_err_x": info.get("bottom_err_x"),
                    "bottom_err_y": info.get("bottom_err_y"),
                    "bottom_img_vel_x_control": info.get("bottom_img_vel_x_control"),
                    "bottom_img_vel_y_control": info.get("bottom_img_vel_y_control"),
                    "bottom_center_error": info.get("bottom_center_error"),
                    "bottom_bbox_rel_err": info.get("bottom_bbox_rel_err"),
                    "alignment_ready_streak": info.get("alignment_ready_streak"),
                    "descent_alignment_latched": info.get("descent_alignment_latched"),
                    "vertical_control_state": info.get("vertical_control_state"),
                    "descent_allowed": info.get("descent_allowed"),
                },
            )
        SESSION.write("control", "step_control_state", agent=agent, action=action, command=extract_command(info), last_airsim_command=SESSION.last_command, guards=extract_guards(info))
        SESSION.write_timeline(agent, reward, done, truncated, info)
        if agent == "AGENT_2":
            record_current_problem_step(self, info, reward, done, truncated)
        SESSION.snapshot_registered(f"{agent}_step")
        if SESSION.physical_steps <= 5 or SESSION.physical_steps % 20 == 0 or done:
            print(
                f"[DIAG] physical={SESSION.physical_steps}/{MAX_PHYSICAL_STEPS} "
                f"trainable={SESSION.trainable_steps}/{MAX_TRAINABLE_STEPS} "
                f"elapsed={SESSION.elapsed:.1f}/{MAX_WALL_SECONDS:.0f}s agent={agent} "
                f"mode={first_value(info, 'tracker_mode', 'tracking_mode', default='')} "
                f"reason={first_value(info, 'termination_reason', default='running')}"
            )
        SESSION.check_limits()
    return post


def env_reset_post(agent: str) -> Callable[[Any, tuple[Any, ...], dict[str, Any], Any], None]:
    def post(self, args, kwargs, result):
        SESSION.reset_count += 1
        try:
            obs, info = result
        except Exception:
            obs, info = result, {}
        SESSION.write(
            "flow",
            "environment_reset_return",
            agent=agent,
            env_id=hex(id(self)),
            observation=obs,
            info=info,
            env_state=snapshot_object(self),
            reset_count=SESSION.reset_count,
        )
        SESSION.write("handoff", "reset_handoff_state", agent=agent, state=extract_handoff(info if isinstance(info, dict) else {}))
        if agent == "AGENT_2" and isinstance(info, dict):
            SESSION.write("identity", "user_target_identity_reset", agent=agent, state=extract_identity(info))
    return post


def extract_reward_parts(info: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in info.items():
        lowered = key.lower()
        if lowered.startswith("r_") or "reward" in lowered or "penalty" in lowered or "bonus" in lowered or "bank" in lowered:
            result[key] = value
    for nested_key in ("reward_parts", "reward_diag", "reward_diagnostics"):
        nested = info.get(nested_key)
        if isinstance(nested, dict):
            result[nested_key] = nested
    return result


def extract_handoff(info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "handoff_ready", "handoff_state", "handoff_reason", "termination_reason",
        "bottom_match", "bottom_match_fresh", "bottom_confirmed",
        "bottom_match_live", "bottom_match_confirmed", "bottom_match_recent",
        "bottom_live_match_streak", "bottom_match_margin",
        "bottom_spatial_jump_norm", "bottom_match_reject_reason",
        "bottom_match_streak", "bottom_velocity_ready", "bottom_velocity_streak",
        "bottom_similarity", "bottom_bbox_rel_err", "bottom_center_error",
        "camera_authority", "control_authority", "speed_stage", "safety_phase",
        "agent1_handoff_reason", "attached_from_agent1",
    ]
    return {key: info.get(key) for key in keys if key in info}


def extract_tracker(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if any(token in key.lower() for token in ("track", "bbox", "match", "similar", "candidate", "class", "fingerprint", "mode", "anchor", "identity", "target_id", "yolo"))
    }


def extract_identity(info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "target_id",
        "identity_judge",
        "yolo_proposal_conf_threshold",
        "initial_yolo_class_id",
        "strict_class_gate",
        "reference_anchor_count",
        "has_front_anchor",
        "has_bottom_anchor",
        "handoff_bbox_transferred",
        "bottom_anchor_created",
        "yolo_candidate_count",
        "selected_candidate_yolo_class_id",
        "selected_candidate_yolo_confidence",
        "candidate_resnet_scores",
        "tracker_mode",
        "bottom_similarity",
        "bottom_match",
        "bottom_match_fresh",
        "bottom_match_live",
        "bottom_match_confirmed",
        "bottom_match_recent",
        "bottom_live_match_streak",
        "bottom_match_margin",
        "bottom_spatial_jump_norm",
        "bottom_match_reject_reason",
        "lost_steps",
    ]
    return {key: info.get(key) for key in keys if key in info}


def extract_command(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if any(token in key.lower() for token in ("command", "cmd_", "safe_action", "raw_action", "final_v", "yaw_rate"))
    }


def extract_guards(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in info.items()
        if any(token in key.lower() for token in ("guard", "safety", "block", "obstacle", "emergency", "risk"))
    }



def _num(mapping: dict[str, Any], *names: str, default: float = 0.0) -> float:
    value = first_value(mapping, *names, default=default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _truth(mapping: dict[str, Any], *names: str, default: bool = False) -> bool:
    value = first_value(mapping, *names, default=default)
    return bool(value)


def _string(mapping: dict[str, Any], *names: str, default: str = "") -> str:
    value = first_value(mapping, *names, default=default)
    return str(value if value is not None else default)


def _flatten_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _frame_candidate(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        try:
            value = getattr(obj, name, None)
        except Exception:
            continue
        if value is not None:
            try:
                import numpy as np
                arr = np.asarray(value)
                if arr.ndim == 3 and arr.size > 0:
                    return arr
            except Exception:
                continue
    return None


def capture_critical_frames(agent2_env: Any, label: str) -> None:
    """Save synchronized front/bottom evidence for important control transitions."""
    if SESSION is None:
        return
    if SESSION.physical_steps - SESSION.last_event_frame_step < EVENT_FRAME_MIN_STEP_GAP:
        return
    SESSION.last_event_frame_step = SESSION.physical_steps
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))[:100]
    bottom = _frame_candidate(
        agent2_env,
        ("_last_bottom_frame", "last_bottom_frame", "_bottom_frame"),
    )
    wrapper = None
    executor = getattr(agent2_env, "_external_command_executor", None)
    if executor is not None:
        wrapper = getattr(executor, "__self__", None)
    agent1_env = getattr(wrapper, "agent1_env", None) if wrapper is not None else None
    front = _frame_candidate(
        agent1_env,
        ("_last_frame", "last_frame", "_front_frame", "_last_front_frame"),
    )
    agent1_bottom = _frame_candidate(
        agent1_env,
        ("_last_downward_frame", "_last_bottom_frame", "last_downward_frame"),
    )

    saved: dict[str, str] = {}
    for kind, frame in (("bottom", bottom), ("front", front), ("agent1_bottom", agent1_bottom)):
        if frame is None:
            continue
        try:
            import cv2
            folder = RUN_DIR / "frames" / "events"
            file_path = folder / f"tick_{SESSION.control_ticks:06d}_{safe_label}_{kind}.jpg"
            cv2.imwrite(str(file_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved[kind] = str(file_path.relative_to(RUN_DIR))
        except Exception as exc:
            log_exception("capture_critical_frames", exc)
    SESSION.write("bookmarks", "critical_frames", label=label, files=saved)


def _critical_anomalies(row: dict[str, Any]) -> list[str]:
    anomalies: list[str] = []
    bottom_live = bool(row.get("bottom_live"))
    a1_weight = float(row.get("agent1_xy_weight") or 0.0)
    if bottom_live and a1_weight > 0.05:
        anomalies.append("bottom_live_but_agent1_xy_not_muted")
    if not bottom_live and a1_weight < 0.95:
        anomalies.append("bottom_not_live_but_agent1_xy_still_muted")

    requested_x = (
        float(row.get("agent1_pre_fuse_vx") or 0.0) * a1_weight
        + float(row.get("ff_requested_vx") or 0.0)
        + float(row.get("bottom_requested_vx") or 0.0)
    )
    requested_y = (
        float(row.get("agent1_pre_fuse_vy") or 0.0) * a1_weight
        + float(row.get("ff_requested_vy") or 0.0)
        + float(row.get("bottom_requested_vy") or 0.0)
    )
    requested_mag = math.hypot(requested_x, requested_y)
    physical_mag = math.hypot(
        float(row.get("physical_cmd_vx") or 0.0),
        float(row.get("physical_cmd_vy") or 0.0),
    )
    if requested_mag >= 0.30 and physical_mag <= 0.08:
        anomalies.append("nonzero_xy_request_but_physical_xy_near_zero")
    if bool(row.get("ff_bottom_blocked_by_safety")):
        anomalies.append("target_ff_and_bottom_correction_zeroed_by_safety")
    if bool(row.get("climb_command_blocked")):
        anomalies.append("negative_z_action_climb_request_blocked")
    if float(row.get("relative_height_m") or 0.0) < -0.25:
        anomalies.append("negative_relative_height_sign_or_post_collision_anomaly")
    if bool(row.get("bottom_live")) and float(row.get("center_error") or 999.0) < 0.30 and not bool(row.get("landing_lock")):
        anomalies.append("bottom_centered_but_landing_lock_not_active")
    if bool(row.get("catchup_active")) and float(row.get("applied_vz_mps") or 0.0) > 1.0e-4:
        anomalies.append("descent_applied_during_predictive_catchup")
    if not bool(row.get("bottom_live")) and float(row.get("no_live_duration_s") or 0.0) > 1.0:
        anomalies.append("bottom_target_missing_over_one_second")
    return anomalies


def record_current_problem_step(
    agent2_env: Any,
    info: dict[str, Any],
    reward: Any,
    done: Any,
    truncated: Any,
) -> None:
    """Write one joined row covering perception, control authority and physics."""
    if SESSION is None:
        return
    SESSION.control_ticks += 1
    SESSION.last_agent2_step_info = dict(info)
    a1 = dict(SESSION.last_agent1_step_info)
    fuse = dict(SESSION.last_parallel_fuse)
    override = dict(SESSION.last_parallel_override)
    guidance = dict(SESSION.last_predictive_guidance)
    vertical_gate = dict(SESSION.last_vertical_gate)

    safety_reasons = _flatten_reasons(a1.get("safety_reasons"))
    for reason in safety_reasons:
        SESSION.safety_reason_counts[reason] += 1

    drone_pos = drone_vel = None
    multirotor_state = SESSION.last_multirotor_state
    if multirotor_state is not None:
        kin = getattr(multirotor_state, "kinematics_estimated", None)
        if kin is not None:
            drone_pos = getattr(kin, "position", None)
            drone_vel = getattr(kin, "linear_velocity", None)
    target_name = str(getattr(agent2_env, "_target_actor_name", "") or "")
    target_pose = SESSION.last_object_poses.get(target_name)
    target_pos = getattr(target_pose, "position", None) if target_pose is not None else None
    drone_x = float(getattr(drone_pos, "x_val", 0.0) or 0.0) if drone_pos is not None else 0.0
    drone_y = float(getattr(drone_pos, "y_val", 0.0) or 0.0) if drone_pos is not None else 0.0
    drone_z = float(getattr(drone_pos, "z_val", 0.0) or 0.0) if drone_pos is not None else 0.0
    target_x = float(getattr(target_pos, "x_val", 0.0) or 0.0) if target_pos is not None else 0.0
    target_y = float(getattr(target_pos, "y_val", 0.0) or 0.0) if target_pos is not None else 0.0
    target_z = float(getattr(target_pos, "z_val", 0.0) or 0.0) if target_pos is not None else 0.0

    row: dict[str, Any] = {
        "wall_s": round(SESSION.elapsed, 6),
        "control_tick": SESSION.control_ticks,
        "physical_step": SESSION.physical_steps,
        "trainable_step": SESSION.trainable_steps,
        "episode_step": int(getattr(agent2_env, "_step", 0)),
        "tracker_mode": _string(info, "tracker_mode"),
        "bottom_live": _truth(info, "bottom_match_live"),
        "bottom_confirmed": _truth(info, "bottom_match_confirmed"),
        "bottom_similarity": _num(info, "bottom_similarity"),
        "bottom_reject_reason": _string(info, "bottom_match_reject_reason"),
        "candidate_count": int(_num(info, "yolo_candidate_count")),
        "adaptive_anchor_count": int(_num(info, "adaptive_anchor_count")),
        "adaptive_updates": int(_num(info, "adaptive_embedding_updates")),
        "err_x": _num(info, "bottom_err_x", default=999.0),
        "err_y": _num(info, "bottom_err_y", default=999.0),
        "center_error": _num(info, "bottom_center_error", default=999.0),
        "bbox_rel_error": _num(info, "bottom_bbox_rel_err", default=999.0),
        "bbox_area": _num(info, "bottom_bbox_area_norm"),
        "image_vel_x": _num(info, "bottom_img_vel_x_control"),
        "image_vel_y": _num(info, "bottom_img_vel_y_control"),
        "control_dt_s": _num(info, "bottom_control_dt_s"),
        "predicted_err_x": _num(info, "predicted_bottom_err_x", "parallel_bottom_predicted_err_x", default=_num(guidance, "predicted_err_x")),
        "predicted_err_y": _num(info, "predicted_bottom_err_y", "parallel_bottom_predicted_err_y", default=_num(guidance, "predicted_err_y")),
        "predicted_center_error": _num(info, "predicted_bottom_center_error", "parallel_bottom_predicted_center_error", default=_num(guidance, "predicted_center_error", default=999.0)),
        "prediction_horizon_s": _num(info, "parallel_bottom_prediction_horizon_s", default=_num(guidance, "prediction_horizon_s")),
        "outward_speed_per_s": _num(info, "parallel_bottom_outward_speed_per_s", default=_num(guidance, "outward_speed_per_s")),
        "catchup_active": _truth(info, "predictive_bottom_catchup_active", "parallel_bottom_predictive_catchup", default=_truth(guidance, "catchup_active")),
        "catchup_release_streak": int(getattr(agent2_env, "_predictive_catchup_release_streak", 0)),
        "no_live_duration_s": _num(info, "bottom_no_live_duration_s"),
        "lost_steps": int(_num(info, "lost_steps")),
        "landing_lock": _truth(info, "landing_lock_active", "descent_alignment_latched"),
        "alignment_streak": int(_num(info, "alignment_ready_streak")),
        "lock_visual_gap_steps": int(_num(info, "landing_lock_visual_gap_steps")),
        "lock_bad_live_steps": int(_num(info, "landing_lock_bad_live_steps")),
        "lock_age_steps": int(_num(info, "landing_lock_age_steps", default=-1)),
        "vertical_gate_state": _string(info, "vertical_control_state", default=_string(vertical_gate, "state")),
        "descent_allowed": _truth(info, "descent_allowed"),
        "descent_block_reason": _string(info, "descent_block_reason", default=_string(vertical_gate, "reason")),
        "raw_vz_action": _num(info, "raw_vz_action"),
        "requested_vz_mps": _num(info, "requested_vz_mps"),
        "applied_vz_mps": _num(info, "applied_vz_mps", "commanded_vz_mps"),
        "climb_command_blocked": _truth(info, "climb_command_blocked"),
        "drone_z_ned": _num(info, "drone_z_ned"),
        "target_surface_z_ned": _num(info, "target_surface_z_ned"),
        "relative_height_m": _num(info, "relative_height_to_target_m"),
        "alt_agl_m": _num(info, "alt_agl_m"),
        "target_velocity_valid": _truth(info, "target_velocity_valid", "parallel_target_velocity_ff_valid"),
        "target_velocity_age_s": _num(info, "target_velocity_age_s"),
        "actual_drone_vx": float(getattr(drone_vel, "x_val", 0.0) or 0.0) if drone_vel is not None else 0.0,
        "actual_drone_vy": float(getattr(drone_vel, "y_val", 0.0) or 0.0) if drone_vel is not None else 0.0,
        "actual_drone_vz": float(getattr(drone_vel, "z_val", 0.0) or 0.0) if drone_vel is not None else 0.0,
        "drone_world_x": drone_x,
        "drone_world_y": drone_y,
        "drone_world_z": drone_z,
        "target_world_x": target_x,
        "target_world_y": target_y,
        "target_world_z": target_z,
        "relative_world_x": target_x - drone_x,
        "relative_world_y": target_y - drone_y,
        "relative_world_z": target_z - drone_z,
        "target_vel_body_vx": _num(info, "target_velocity_body_vx_mps"),
        "target_vel_body_vy": _num(info, "target_velocity_body_vy_mps"),
        "target_speed_mps": _num(info, "target_velocity_speed_mps"),
        "agent1_tracking_mode": _string(info, "parallel_agent1_tracking_mode", default=_string(a1, "tracking_mode")),
        "agent1_active_camera": _string(info, "parallel_agent1_active_camera", default=_string(a1, "active_camera")),
        "agent1_fusion_has_target": _truth(info, "parallel_agent1_fusion_has_target", default=_truth(a1, "fusion_has_target")),
        "agent1_bottom_match": _truth(info, "parallel_agent1_bottom_match", default=_truth(a1, "bottom_match")),
        "agent1_xy_weight": _num(info, "parallel_agent1_xy_weight", default=_num(override, "agent1_xy_weight", default=1.0)),
        "agent1_pre_fuse_vx": _num(fuse, "agent1_vx"),
        "agent1_pre_fuse_vy": _num(fuse, "agent1_vy"),
        "ff_requested_vx": _num(info, "parallel_target_velocity_ff_vx_mps", default=_num(override, "feedforward_vx_mps")),
        "ff_requested_vy": _num(info, "parallel_target_velocity_ff_vy_mps", default=_num(override, "feedforward_vy_mps")),
        "ff_applied_vx": _num(a1, "parallel_target_velocity_ff_vx_mps"),
        "ff_applied_vy": _num(a1, "parallel_target_velocity_ff_vy_mps"),
        "bottom_requested_vx": _num(info, "parallel_bottom_guidance_vx_mps", default=_num(override, "bottom_correction_vx_mps")),
        "bottom_requested_vy": _num(info, "parallel_bottom_guidance_vy_mps", default=_num(override, "bottom_correction_vy_mps")),
        "bottom_applied_vx": _num(a1, "parallel_bottom_correction_vx_mps"),
        "bottom_applied_vy": _num(a1, "parallel_bottom_correction_vy_mps"),
        "fused_expected_vx": _num(fuse, "result_vx"),
        "fused_expected_vy": _num(fuse, "result_vy"),
        "physical_cmd_vx": _num(info, "commanded_vx_mps", default=_num(a1, "commanded_vx_mps")),
        "physical_cmd_vy": _num(info, "commanded_vy_mps", default=_num(a1, "commanded_vy_mps")),
        "physical_cmd_vz": _num(info, "commanded_vz_mps", default=_num(a1, "commanded_vz_mps")),
        "horizontal_speed_limit_mps": _num(info, "horizontal_total_speed_limit_mps", "horizontal_speed_limit_mps", default=_num(fuse, "speed_limit_mps")),
        "safety_intervention": _truth(a1, "safety_intervention"),
        "safety_reasons": "|".join(safety_reasons),
        "ff_bottom_blocked_by_safety": _truth(info, "parallel_target_velocity_ff_blocked_by_safety", default=_truth(a1, "parallel_target_velocity_ff_blocked_by_safety")),
        "front_dist_m": _num(a1, "front_dist_m", default=_num(info, "front_dist_m", default=999.0)),
        "back_dist_m": _num(a1, "back_dist_m", default=_num(info, "back_dist_m", default=999.0)),
        "left_dist_m": _num(a1, "left_dist_m", default=_num(info, "left_dist_m", default=999.0)),
        "right_dist_m": _num(a1, "right_dist_m", default=_num(info, "right_dist_m", default=999.0)),
        "collision": _truth(info, "collision_new"),
        "collision_object": _string(info, "collision_object_name"),
        "termination_reason": _string(info, "termination_reason"),
        "reward": scalar_or_empty(reward),
    }
    anomalies = _critical_anomalies(row)
    row["anomalies"] = "|".join(anomalies)
    for anomaly in anomalies:
        SESSION.anomaly_counts[anomaly] += 1

    block_reason = str(row["descent_block_reason"] or "")
    if block_reason:
        SESSION.lock_block_counts[block_reason] += 1

    previous = SESSION.last_critical_state
    transition_keys = (
        "tracker_mode", "bottom_live", "landing_lock", "catchup_active",
        "vertical_gate_state", "agent1_active_camera", "agent1_xy_weight",
        "ff_bottom_blocked_by_safety", "collision", "termination_reason",
    )
    changed = [
        key for key in transition_keys
        if previous and previous.get(key) != row.get(key)
    ]
    for key in changed:
        SESSION.transition_counts[f"{key}:{previous.get(key)}->{row.get(key)}"] += 1

    try:
        SESSION.critical.writerow({key: scalar_or_empty(row.get(key, "")) for key in SESSION.critical_fields})
    except Exception as exc:
        SESSION.write("exceptions", "critical_timeline_write_failed", error=repr(exc), row=row)

    SESSION.write("parallel", "joined_parallel_control_tick", state=row, override=override, fuse=fuse)
    SESSION.write("landing_gate", "joined_landing_gate_tick", state={
        key: row.get(key) for key in (
            "bottom_live", "bottom_confirmed", "bottom_similarity", "center_error",
            "bbox_rel_error", "predicted_center_error", "catchup_active",
            "landing_lock", "alignment_streak", "lock_visual_gap_steps",
            "lock_bad_live_steps", "vertical_gate_state", "descent_allowed",
            "descent_block_reason", "raw_vz_action", "requested_vz_mps",
            "applied_vz_mps", "climb_command_blocked",
        )
    }, method_gate=vertical_gate)
    SESSION.write("predictive", "joined_predictive_tick", state={
        key: row.get(key) for key in (
            "err_x", "err_y", "image_vel_x", "image_vel_y", "control_dt_s",
            "predicted_err_x", "predicted_err_y", "predicted_center_error",
            "prediction_horizon_s", "outward_speed_per_s", "catchup_active",
            "bottom_requested_vx", "bottom_requested_vy", "physical_cmd_vx",
            "physical_cmd_vy",
        )
    }, guidance=guidance, horizontal_servo=SESSION.last_horizontal_servo)
    SESSION.write("vertical", "joined_vertical_tick", state={
        key: row.get(key) for key in (
            "raw_vz_action", "requested_vz_mps", "applied_vz_mps",
            "climb_command_blocked", "descent_allowed", "descent_block_reason",
            "drone_z_ned", "target_surface_z_ned", "relative_height_m", "alt_agl_m",
        )
    })
    SESSION.write("safety_detail", "joined_safety_tick", state={
        key: row.get(key) for key in (
            "safety_intervention", "safety_reasons", "ff_bottom_blocked_by_safety",
            "ff_requested_vx", "ff_requested_vy", "ff_applied_vx", "ff_applied_vy",
            "bottom_requested_vx", "bottom_requested_vy", "bottom_applied_vx",
            "bottom_applied_vy", "fused_expected_vx", "fused_expected_vy",
            "physical_cmd_vx", "physical_cmd_vy", "front_dist_m", "back_dist_m",
            "left_dist_m", "right_dist_m",
        )
    })
    SESSION.write("fusion", "joined_camera_authority_tick", state={
        key: row.get(key) for key in (
            "tracker_mode", "bottom_live", "agent1_tracking_mode",
            "agent1_active_camera", "agent1_fusion_has_target", "agent1_bottom_match",
            "agent1_xy_weight",
        )
    })
    SESSION.write("kinematics", "joined_kinematics_tick", state={
        key: row.get(key) for key in (
            "target_velocity_valid", "target_velocity_age_s", "target_vel_body_vx",
            "actual_drone_vx", "actual_drone_vy", "actual_drone_vz",
            "drone_world_x", "drone_world_y", "drone_world_z",
            "target_world_x", "target_world_y", "target_world_z",
            "relative_world_x", "relative_world_y", "relative_world_z",
            "target_vel_body_vy", "target_speed_mps", "physical_cmd_vx",
            "physical_cmd_vy", "physical_cmd_vz", "relative_height_m",
        )
    })

    previous_anomalies = {item for item in str(previous.get("anomalies", "")).split("|") if item}
    event_reasons = [item for item in anomalies if item not in previous_anomalies]
    event_reasons.extend(f"transition_{key}" for key in changed)
    if bool(done) or bool(truncated):
        event_reasons.append("episode_finished")
        episode = {
            "control_tick": SESSION.control_ticks,
            "episode_step": row["episode_step"],
            "termination_reason": row["termination_reason"],
            "reward": row["reward"],
            "collision_object": row["collision_object"],
            "bottom_live": row["bottom_live"],
            "center_error": row["center_error"],
            "landing_lock": row["landing_lock"],
        }
        SESSION.episode_results.append(episode)
    if bool(row["collision"]):
        event_reasons.append("collision")
        SESSION.collision_events.append(dict(row))

    if event_reasons:
        label = "__".join(event_reasons[:4])
        SESSION.write("bookmarks", "critical_transition", reasons=event_reasons, state=row)
        capture_critical_frames(agent2_env, label)

    SESSION.last_critical_state = dict(row)


def install_current_problem_instrumentation(modules: dict[str, Any]) -> None:
    """Install exact probes around the current v12.x control-authority path."""
    agent2_module = modules.get("agent2_landing_env")
    wrapper_module = modules.get("agent1p2_env")
    drone_module = modules.get("drone_env")

    if agent2_module is not None:
        Agent2 = agent2_module.Agent2LandingEnv

        if hasattr(Agent2, "_vertical_control_state"):
            original_vertical = Agent2._vertical_control_state
            @functools.wraps(original_vertical)
            def vertical_wrapper(self, info, *args, __orig=original_vertical, **kwargs):
                before = {
                    "alignment_streak": getattr(self, "_alignment_ready_streak", None),
                    "landing_lock": getattr(self, "_descent_alignment_latched", None),
                    "visual_gap_steps": getattr(self, "_landing_lock_visual_gap_steps", None),
                    "bad_live_steps": getattr(self, "_landing_lock_bad_live_steps", None),
                    "predictive_catchup": getattr(self, "_predictive_catchup_active", None),
                }
                result = __orig(self, info, *args, **kwargs)
                state, allowed, reason = result
                record = {
                    "state": state,
                    "allowed": bool(allowed),
                    "reason": reason,
                    "input": {key: info.get(key) for key in (
                        "tracker_mode", "bottom_match_live", "bottom_match_confirmed",
                        "bottom_similarity", "bottom_center_error", "bottom_bbox_rel_err",
                        "bottom_bbox_area_norm", "predicted_bottom_center_error",
                        "predictive_bottom_catchup_active", "bottom_no_live_duration_s",
                    )},
                    "before": before,
                    "after": {
                        "alignment_streak": getattr(self, "_alignment_ready_streak", None),
                        "landing_lock": getattr(self, "_descent_alignment_latched", None),
                        "visual_gap_steps": getattr(self, "_landing_lock_visual_gap_steps", None),
                        "bad_live_steps": getattr(self, "_landing_lock_bad_live_steps", None),
                        "lock_acquired_step": getattr(self, "_landing_lock_acquired_step", None),
                    },
                    "thresholds": {
                        "descent_min_similarity": getattr(self.cfg, "descent_min_similarity", None),
                        "alignment_enter_center_error": getattr(self.cfg, "alignment_enter_center_error", None),
                        "alignment_enter_bbox_rel_error": getattr(self.cfg, "alignment_enter_bbox_rel_error", None),
                        "alignment_exit_center_error": getattr(self.cfg, "alignment_exit_center_error", None),
                        "alignment_exit_bbox_rel_error": getattr(self.cfg, "alignment_exit_bbox_rel_error", None),
                        "alignment_streak_required": getattr(self.cfg, "alignment_streak_required", None),
                        "landing_lock_max_visual_gap_steps": getattr(self.cfg, "landing_lock_max_visual_gap_steps", None),
                        "landing_lock_bad_live_release_steps": getattr(self.cfg, "landing_lock_bad_live_release_steps", None),
                    },
                }
                SESSION.last_vertical_gate = record
                SESSION.write("landing_gate", "vertical_control_state_decision", **record)
                return result
            patch_attr(Agent2, "_vertical_control_state", vertical_wrapper)

        if hasattr(Agent2, "_compute_predictive_bottom_guidance"):
            original_guidance = Agent2._compute_predictive_bottom_guidance
            @functools.wraps(original_guidance)
            def guidance_wrapper(self, info, *args, __orig=original_guidance, **kwargs):
                before_catch = bool(getattr(self, "_predictive_catchup_active", False))
                result = __orig(self, info, *args, **kwargs)
                record = dict(result or {})
                record["input"] = {key: info.get(key) for key in (
                    "bottom_match_live", "bottom_similarity", "bottom_err_x",
                    "bottom_err_y", "bottom_center_error", "bottom_bbox_area_norm",
                    "bottom_img_vel_x_control", "bottom_img_vel_y_control",
                    "bottom_control_dt_s", "relative_height_to_target_m",
                )}
                record["catchup_before"] = before_catch
                record["catchup_after"] = bool(getattr(self, "_predictive_catchup_active", False))
                record["release_streak"] = int(getattr(self, "_predictive_catchup_release_streak", 0))
                record["thresholds"] = {
                    name: getattr(self.cfg, name, None) for name in (
                        "predictive_bottom_extra_latency_s",
                        "predictive_bottom_horizon_min_s",
                        "predictive_bottom_horizon_max_s",
                        "predictive_bottom_kp_y_to_vx_mps",
                        "predictive_bottom_kd_y_to_vx_mps",
                        "predictive_bottom_kp_x_to_vy_mps",
                        "predictive_bottom_kd_x_to_vy_mps",
                        "predictive_bottom_normal_correction_max_mps",
                        "predictive_bottom_catchup_correction_max_mps",
                        "predictive_bottom_touchdown_catchup_max_mps",
                        "predictive_bottom_catchup_enter_center_error",
                        "predictive_bottom_catchup_exit_center_error",
                        "predictive_bottom_catchup_enter_outward_speed_per_s",
                        "predictive_bottom_catchup_exit_outward_speed_per_s",
                        "predictive_bottom_catchup_exit_image_speed_per_s",
                    )
                }
                SESSION.last_predictive_guidance = record
                SESSION.write("predictive", "predictive_guidance_decision", state=record)
                return result
            patch_attr(Agent2, "_compute_predictive_bottom_guidance", guidance_wrapper)

        if hasattr(Agent2, "_horizontal_visual_servo"):
            original_servo = Agent2._horizontal_visual_servo
            @functools.wraps(original_servo)
            def servo_wrapper(self, raw, info, *args, __orig=original_servo, **kwargs):
                result = __orig(self, raw, info, *args, **kwargs)
                vx, vy, details = result
                record = {
                    "raw_action": raw,
                    "input": {key: info.get(key) for key in (
                        "bottom_match_live", "bottom_match_confirmed", "bottom_err_x",
                        "bottom_err_y", "bottom_center_error", "bottom_bbox_rel_err",
                        "relative_height_to_target_m", "target_velocity_valid",
                        "target_velocity_body_vx_mps", "target_velocity_body_vy_mps",
                    )},
                    "vx": vx,
                    "vy": vy,
                    "details": details,
                }
                SESSION.last_horizontal_servo = record
                SESSION.write("horizontal", "horizontal_visual_servo_decision", **record)
                return result
            patch_attr(Agent2, "_horizontal_visual_servo", servo_wrapper)

        if hasattr(Agent2, "_update_control_image_velocity"):
            original_img_vel = Agent2._update_control_image_velocity
            @functools.wraps(original_img_vel)
            def image_velocity_wrapper(self, metrics, live_match, *args, __orig=original_img_vel, **kwargs):
                before = {
                    "vel_x": getattr(self, "_control_img_vel_x", 0.0),
                    "vel_y": getattr(self, "_control_img_vel_y", 0.0),
                    "last_err_x": getattr(self, "_last_control_err_x", 0.0),
                    "last_err_y": getattr(self, "_last_control_err_y", 0.0),
                    "last_live": getattr(self, "_last_control_had_live_match", False),
                }
                result = __orig(self, metrics, live_match, *args, **kwargs)
                after = {
                    "vel_x": getattr(self, "_control_img_vel_x", 0.0),
                    "vel_y": getattr(self, "_control_img_vel_y", 0.0),
                    "dt_s": getattr(self, "_last_control_dt_s", 0.0),
                    "last_err_x": getattr(self, "_last_control_err_x", 0.0),
                    "last_err_y": getattr(self, "_last_control_err_y", 0.0),
                    "last_live": getattr(self, "_last_control_had_live_match", False),
                }
                SESSION.write("predictive", "image_velocity_update", metrics=metrics, live_match=live_match, before=before, after=after)
                return result
            patch_attr(Agent2, "_update_control_image_velocity", image_velocity_wrapper)

        if hasattr(Agent2, "_maybe_update_adaptive_embedding_bank"):
            original_bank = Agent2._maybe_update_adaptive_embedding_bank
            @functools.wraps(original_bank)
            def bank_wrapper(self, *args, __orig=original_bank, **kwargs):
                before = {
                    "count": len(getattr(self, "_adaptive_embeddings", [])),
                    "steps": list(getattr(self, "_adaptive_embedding_steps", [])),
                    "updates": getattr(self, "_adaptive_embedding_updates", 0),
                }
                result = __orig(self, *args, **kwargs)
                after = {
                    "count": len(getattr(self, "_adaptive_embeddings", [])),
                    "steps": list(getattr(self, "_adaptive_embedding_steps", [])),
                    "updates": getattr(self, "_adaptive_embedding_updates", 0),
                }
                SESSION.write("adaptive_bank", "adaptive_bank_update_attempt", args=args, kwargs=kwargs, result=result, before=before, after=after)
                return result
            patch_attr(Agent2, "_maybe_update_adaptive_embedding_bank", bank_wrapper)

    if wrapper_module is not None:
        Agent1P2 = wrapper_module.Agent1P2Env
        if hasattr(Agent1P2, "_execute_parallel_command"):
            original_execute = Agent1P2._execute_parallel_command
            @functools.wraps(original_execute)
            def execute_wrapper(self, agent2_vz_mps, *args, __orig=original_execute, **kwargs):
                result = __orig(self, agent2_vz_mps, *args, **kwargs)
                SESSION.write(
                    "parallel", "parallel_command_executor_return",
                    agent2_vz_mps=agent2_vz_mps,
                    result=result,
                    agent1_info=getattr(self, "_agent1_last_info", {}),
                    bottom_guidance=getattr(getattr(self, "agent2_env", None), "_last_predictive_guidance", {}),
                )
                return result
            patch_attr(Agent1P2, "_execute_parallel_command", execute_wrapper)

    if drone_module is not None:
        DroneEnv = drone_module.DroneEnv
        if hasattr(DroneEnv, "set_parallel_control_overrides"):
            original_overrides = DroneEnv.set_parallel_control_overrides
            @functools.wraps(original_overrides)
            def overrides_wrapper(self, *args, __orig=original_overrides, **kwargs):
                names = (
                    "vz_mps", "feedforward_vx_mps", "feedforward_vy_mps",
                    "bottom_correction_vx_mps", "bottom_correction_vy_mps",
                    "agent1_xy_weight", "horizontal_speed_limit_mps",
                )
                values = {name: kwargs.get(name, args[index] if index < len(args) else None) for index, name in enumerate(names)}
                SESSION.last_parallel_override = values
                SESSION.write("parallel", "parallel_override_requested", state=values)
                return __orig(self, *args, **kwargs)
            patch_attr(DroneEnv, "set_parallel_control_overrides", overrides_wrapper)

        if hasattr(DroneEnv, "_fuse_parallel_xy"):
            original_fuse = DroneEnv._fuse_parallel_xy
            @functools.wraps(original_fuse)
            def fuse_wrapper(*args, __orig=original_fuse, **kwargs):
                names = (
                    "agent1_vx", "agent1_vy", "agent1_weight", "feedforward_vx",
                    "feedforward_vy", "bottom_correction_vx", "bottom_correction_vy",
                    "speed_limit_mps",
                )
                values = {name: kwargs.get(name, args[index] if index < len(args) else None) for index, name in enumerate(names)}
                result = __orig(*args, **kwargs)
                record = dict(values)
                record["result_vx"] = result[0]
                record["result_vy"] = result[1]
                SESSION.last_parallel_fuse = record
                SESSION.write("parallel", "parallel_xy_fuse", state=record)
                return result
            patch_attr(DroneEnv, "_fuse_parallel_xy", staticmethod(fuse_wrapper))


def snapshot_model_inventory() -> None:
    entries: list[dict[str, Any]] = []
    model_root = BASE_DIR / "models"
    if model_root.exists():
        for path in sorted(model_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                record = {
                    "path": str(path.relative_to(BASE_DIR)),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "sha256": sha256_bytes(path.read_bytes()),
                }
                match = re.search(r"(\d+)_steps", path.name)
                record["numbered_steps"] = int(match.group(1)) if match else None
                entries.append(record)
            except Exception as exc:
                entries.append({"path": str(path), "error": repr(exc)})
    (RUN_DIR / "model_checkpoint_inventory.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    SESSION.write("checkpoints", "checkpoint_inventory", entries=entries)


def snapshot_code_audit() -> None:
    """Record exact source lines relevant to the current control deadlock."""
    patterns = [
        ("negative_vz_clipping", re.compile(r"max\s*\(\s*0\.0\s*,.*vz", re.IGNORECASE)),
        ("parallel_safety_zeroing", re.compile(r"ff_vx\s*=\s*0\.0|bottom_vx\s*=\s*0\.0")),
        ("safety_obstacle_token_gate", re.compile(r"obstacle.*horizontal.*lidar|for token in \(\"obstacle\"", re.IGNORECASE)),
        ("bottom_live_xy_authority", re.compile(r"bottom_live_agent1_xy_weight|if bottom_live else 1\.0")),
        ("landing_lock_gate", re.compile(r"_descent_alignment_latched|alignment_streak_required")),
        ("predictive_catchup_gate", re.compile(r"HOLD_PREDICTIVE_CATCHUP|predictive_bottom_catchup")),
        ("relative_height_equation", re.compile(r"target_surface_z_ned\s*-\s*drone_z_ned")),
    ]
    findings: list[dict[str, Any]] = []
    for file_name in ("drone_env.py", "agent1p2_env.py", "agent2_landing_env.py", "config/flow_config.py"):
        file_path = BASE_DIR / file_name
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, line in enumerate(lines, start=1):
            for finding_name, pattern in patterns:
                if pattern.search(line):
                    findings.append({
                        "finding": finding_name,
                        "file": file_name,
                        "line": line_no,
                        "text": line.strip(),
                    })
    (RUN_DIR / "static_code_audit.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    md = ["# Static control-path audit", "", "These are observations only; Run_diag does not alter the source.", ""]
    for item in findings:
        md.append(f"- **{item['finding']}** — `{item['file']}:{item['line']}` — `{item['text']}`")
    (RUN_DIR / "static_code_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    SESSION.write("audit", "static_code_audit", findings=findings)


def copy_runtime_artifacts() -> None:
    """Copy files created or modified by this diagnostic run from logs/results."""
    destination = RUN_DIR / "runtime_artifacts"
    total = 0
    copied: list[dict[str, Any]] = []
    cutoff = SESSION.start_wall - 3.0
    for root_name in ("logs", "results"):
        root = BASE_DIR / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                if stat.st_mtime < cutoff:
                    continue
                if stat.st_size > RUNTIME_ARTIFACT_MAX_FILE_BYTES:
                    copied.append({"path": str(path.relative_to(BASE_DIR)), "skipped": "file_too_large", "size": stat.st_size})
                    continue
                if total + stat.st_size > RUNTIME_ARTIFACT_TOTAL_BYTES:
                    copied.append({"path": str(path.relative_to(BASE_DIR)), "skipped": "total_limit", "size": stat.st_size})
                    continue
                target = destination / path.relative_to(BASE_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                total += stat.st_size
                copied.append({"path": str(path.relative_to(BASE_DIR)), "copied": True, "size": stat.st_size})
            except Exception as exc:
                copied.append({"path": str(path), "error": repr(exc)})
    (RUN_DIR / "runtime_artifacts_manifest.json").write_text(
        json.dumps(copied, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def write_automatic_findings() -> None:
    """Produce a human-readable first-pass diagnosis from the collected counters."""
    lines = [
        "# Automatic diagnostic findings",
        "",
        f"Diagnostic version: `{DIAG_VERSION}`",
        "",
        "This report is generated from runtime evidence. It does not replace inspection of",
        "`00_control_authority_timeline.csv` and the event-frame pairs.",
        "",
        "## Run totals",
        "",
        f"- Control ticks: **{SESSION.control_ticks}**",
        f"- Instrumented environment step returns: **{SESSION.physical_steps}**",
        f"- Trainable PPO steps: **{SESSION.trainable_steps}**",
        f"- Episodes completed: **{len(SESSION.episode_results)}**",
        f"- Collisions captured: **{len(SESSION.collision_events)}**",
        "",
        "## Most frequent descent/landing-lock blockers",
        "",
    ]
    if SESSION.lock_block_counts:
        for reason, count in SESSION.lock_block_counts.most_common(20):
            lines.append(f"- `{reason}`: **{count}**")
    else:
        lines.append("- No block reason was recorded.")

    lines.extend(["", "## Safety reasons", ""])
    if SESSION.safety_reason_counts:
        for reason, count in SESSION.safety_reason_counts.most_common(20):
            lines.append(f"- `{reason}`: **{count}**")
    else:
        lines.append("- No safety reason was recorded.")

    lines.extend(["", "## Automatically detected anomalies", ""])
    if SESSION.anomaly_counts:
        for reason, count in SESSION.anomaly_counts.most_common(30):
            lines.append(f"- `{reason}`: **{count}**")
    else:
        lines.append("- No predefined anomaly was detected.")

    lines.extend(["", "## Authority/state transitions", ""])
    if SESSION.transition_counts:
        for transition, count in SESSION.transition_counts.most_common(30):
            lines.append(f"- `{transition}`: **{count}**")
    else:
        lines.append("- No transition was recorded.")

    lines.extend(["", "## Episode results", ""])
    if SESSION.episode_results:
        for index, episode in enumerate(SESSION.episode_results, start=1):
            lines.append(
                f"- Episode {index}: `{episode.get('termination_reason')}` | "
                f"reward={episode.get('reward')} | collision=`{episode.get('collision_object')}` | "
                f"center={episode.get('center_error')} | lock={episode.get('landing_lock')}"
            )
    else:
        lines.append("- The diagnostic stopped before an episode completed.")

    lines.extend([
        "",
        "## Files to inspect first",
        "",
        "1. `00_control_authority_timeline.csv` — joined perception/control/safety/Z timeline.",
        "2. `27_landing_lock_and_z_gate.jsonl` — exact reason for every Z decision.",
        "3. `26_parallel_xy_arbitration.jsonl` — requested versus fused XY components.",
        "4. `30_safety_lidar_command_suppression.jsonl` — commands removed by safety.",
        "5. `28_bottom_predictive_controller.jsonl` — t+1 prediction and derivative state.",
        "6. `frames/events/` — synchronized visual evidence at each anomaly/transition.",
        "7. `static_code_audit.md` — exact source lines for clipping and authority gates.",
        "",
    ])
    (RUN_DIR / "DIAGNOSTIC_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def tracker_state_from_env(env: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in ("tracker", "bottom_tracker", "downward_tracker"):
        obj = getattr(env, name, None)
        if obj is not None:
            state[name] = snapshot_object(obj)
    return state


def frame_return_post(kind: str, label: str) -> Callable[[Any, tuple[Any, ...], dict[str, Any], Any], None]:
    def post(self, args, kwargs, result):
        save_frame(result, kind, label)
    return post


def install_component_instrumentation(modules: dict[str, Any]) -> None:
    # Log all project class construction first.
    wrap_project_class_initializers(modules.values())

    drone_env = modules.get("drone_env")
    agent2 = modules.get("agent2_landing_env")
    agent1p2 = modules.get("agent1p2_env")
    object_tracker = modules.get("object_tracker")
    resnet_tracker = modules.get("resnet_yolo_tracker")
    observation_builder = modules.get("observation_builder")
    lidar_processor = modules.get("lidar_processor")
    safety_filter_module = modules.get("safety_filter")
    reward_module = modules.get("follow_reward_v37")

    if resnet_tracker is not None:
        install_yolo_and_resnet_instrumentation(resnet_tracker)
    airsim_module = None
    for module in (drone_env, agent2):
        candidate = getattr(module, "airsim", None) if module is not None else None
        if candidate is not None:
            airsim_module = candidate
            break
    if airsim_module is not None:
        install_airsim_instrumentation(airsim_module)

    if drone_env is not None:
        DroneEnv = drone_env.DroneEnv
        wrap_method(DroneEnv, "reset", log_name="flow", post=env_reset_post("AGENT_1"))
        wrap_method(DroneEnv, "step", log_name="methods", post=env_step_post("AGENT_1"))
        for method in (
            "_get_drone_state", "_get_alt_agl_m", "_get_drone_to_target_distance_m",
            "_get_current_yaw_rad", "_compute_safety_phase", "_get_obstacle_state_m",
            "_read_lidar_obstacle_state_m", "_compute_visual_lidar_handoff_trigger",
            "_update_soft_handoff_state", "_detect_bottom_target_by_fingerprint",
            "_scan_bottom_target_candidate_by_fingerprint", "_update_bottom_full_tracker",
            "_update_stable_tracking", "_update_bottom_stable_tracking",
        ):
            wrap_method(DroneEnv, method, log_name="methods")
        wrap_method(DroneEnv, "_get_frame", log_name="frames", post=frame_return_post("front", "agent1_front"))
        wrap_method(DroneEnv, "_get_downward_frame", log_name="frames", post=frame_return_post("bottom", "agent1_bottom"))

    if agent2 is not None:
        Agent2 = agent2.Agent2LandingEnv
        wrap_method(Agent2, "reset", log_name="flow", post=env_reset_post("AGENT_2"))
        wrap_method(Agent2, "attach_from_agent1", log_name="handoff")
        wrap_method(Agent2, "step", log_name="methods", post=env_step_post("AGENT_2"))
        wrap_method(Agent2, "_get_bottom_frame", log_name="frames", post=frame_return_post("bottom", "agent2_bottom"))
        for method, log in (
            ("_set_identity", "tracker"),
            ("_select_target_from_bottom_click", "tracker"),
            ("_strict_track", "tracker"),
            ("_get_api_state", "physics"),
            ("_read_target_surface_altitude", "physics"),
            ("_get_obstacles", "lidar"),
            ("_observe", "observations"),
            ("_horizontal_lidar_guard", "control"),
            ("_new_collision", "physics"),
        ):
            wrap_method(Agent2, method, log_name=log)

    if agent1p2 is not None:
        Agent1P2 = agent1p2.Agent1P2Env
        wrap_method(Agent1P2, "reset", log_name="handoff")
        wrap_method(Agent1P2, "step", log_name="flow")
        for function_name in ("find_agent1_checkpoint", "exact_env_config_for_checkpoint"):
            if hasattr(agent1p2, function_name):
                original = getattr(agent1p2, function_name)
                @functools.wraps(original)
                def function_wrapper(*args, __orig=original, __name=function_name, **kwargs):
                    started = time.perf_counter()
                    result = __orig(*args, **kwargs)
                    SESSION.write("flow", __name, args=args, kwargs=kwargs, result=result, duration_ms=(time.perf_counter() - started) * 1000.0)
                    return result
                patch_attr(agent1p2, function_name, function_wrapper)

    if object_tracker is not None:
        TrackerAdapter = object_tracker.tracker
        for method in (
            "select_target_and_get_class", "auto_lock_on_fingerprint", "update",
            "get_target_fingerprint", "set_target_fingerprint", "set_target_class",
        ):
            wrap_method(TrackerAdapter, method, log_name="tracker")

    if resnet_tracker is not None:
        Core = resnet_tracker.YoloResNetTracker
        for method in (
            "select_target", "select_target_by_bbox", "update", "_embedding_from_bbox",
            "_candidate_in_search_window", "_motion_score", "_smooth_bbox",
        ):
            wrap_method(Core, method, log_name="tracker")

    if observation_builder is not None:
        Builder = observation_builder.ObservationBuilder
        wrap_method(Builder, "reset", log_name="observations")
        wrap_method(Builder, "build", log_name="observations")
        for method in ("_build_detected_target_vision", "_build_lost_target_vision", "_build_drone_state", "_build_obstacle_state", "_compute_target_stability"):
            wrap_method(Builder, method, log_name="observations")

    if lidar_processor is not None:
        Processor = lidar_processor.LidarProcessor
        wrap_method(Processor, "compute_sector_distances", log_name="lidar")
        if hasattr(lidar_processor, "point_cloud_to_array"):
            original = lidar_processor.point_cloud_to_array
            @functools.wraps(original)
            def pc_wrapper(*args, __orig=original, **kwargs):
                result = __orig(*args, **kwargs)
                SESSION.write("lidar", "point_cloud_to_array", args=args, result=result)
                return result
            patch_attr(lidar_processor, "point_cloud_to_array", pc_wrapper)
            # Modules imported the function directly, so update their references too.
            if drone_env is not None and getattr(drone_env, "point_cloud_to_array", None) is original:
                patch_attr(drone_env, "point_cloud_to_array", pc_wrapper)
            if agent2 is not None and getattr(agent2, "point_cloud_to_array", None) is original:
                patch_attr(agent2, "point_cloud_to_array", pc_wrapper)

    if safety_filter_module is not None and hasattr(safety_filter_module, "safety_filter"):
        original = safety_filter_module.safety_filter
        @functools.wraps(original)
        def safety_wrapper(*args, __orig=original, **kwargs):
            started = time.perf_counter()
            result = __orig(*args, **kwargs)
            SESSION.write("control", "safety_filter_return", args=args, kwargs=kwargs, result=result, duration_ms=(time.perf_counter() - started) * 1000.0)
            return result
        patch_attr(safety_filter_module, "safety_filter", safety_wrapper)
        if drone_env is not None and getattr(drone_env, "safety_filter", None) is original:
            patch_attr(drone_env, "safety_filter", safety_wrapper)

    if reward_module is not None and hasattr(reward_module, "compute_follow_reward"):
        original = reward_module.compute_follow_reward
        @functools.wraps(original)
        def reward_wrapper(*args, __orig=original, **kwargs):
            result = __orig(*args, **kwargs)
            SESSION.write("rewards", "compute_follow_reward_return", args=args, kwargs=kwargs, result=result)
            return result
        patch_attr(reward_module, "compute_follow_reward", reward_wrapper)
        if drone_env is not None and getattr(drone_env, "compute_follow_reward", None) is original:
            patch_attr(drone_env, "compute_follow_reward", reward_wrapper)

    install_current_problem_instrumentation(modules)


def install_sb3_instrumentation(run_train_module: Any) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    except Exception as exc:
        raise RuntimeError(f"Stable-Baselines3 diagnostics could not be installed: {exc}") from exc

    original_init = PPO.__init__
    @functools.wraps(original_init)
    def init_wrapper(self, *args, **kwargs):
        started = time.perf_counter()
        original_init(self, *args, **kwargs)
        try:
            _OBJECT_REFERENCES[id(self)] = self
        except TypeError:
            pass
        SESSION.register_object(self, created_by="stable_baselines3.PPO.__init__", args=args, kwargs=kwargs)
        SESSION.write("sb3", "ppo_created", model_id=hex(id(self)), args=args, kwargs=kwargs, duration_ms=(time.perf_counter() - started) * 1000.0, policy=repr(getattr(self, "policy", None)))
    patch_attr(PPO, "__init__", init_wrapper)

    original_load = PPO.load
    @functools.wraps(original_load)
    def load_wrapper(*args, **kwargs):
        started = time.perf_counter()
        model = original_load(*args, **kwargs)
        try:
            _OBJECT_REFERENCES[id(model)] = model
        except TypeError:
            pass
        SESSION.register_object(model, created_by="stable_baselines3.PPO.load", args=args, kwargs=kwargs)
        checkpoint_record: dict[str, Any] = {}
        try:
            requested = kwargs.get("path", args[0] if args else None)
            source_path = Path(str(requested))
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            if not source_path.exists() and source_path.suffix.lower() != ".zip":
                zipped = source_path.with_suffix(".zip")
                if zipped.exists():
                    source_path = zipped
            if source_path.exists() and source_path.is_file():
                target = RUN_DIR / "active_checkpoints" / source_path.name
                if target.exists():
                    target = target.with_name(f"{target.stem}_{len(SESSION.loaded_checkpoints)}{target.suffix}")
                shutil.copy2(source_path, target)
                checkpoint_record = {
                    "requested": str(requested),
                    "resolved": str(source_path),
                    "copied_to": str(target.relative_to(RUN_DIR)),
                    "size": source_path.stat().st_size,
                    "sha256": sha256_bytes(source_path.read_bytes()),
                }
                SESSION.loaded_checkpoints.append(checkpoint_record)
                SESSION.write("checkpoints", "active_checkpoint_copied", **checkpoint_record)
        except Exception as exc:
            SESSION.write("exceptions", "active_checkpoint_copy_failed", error=repr(exc), args=args, kwargs=kwargs)
        SESSION.write("sb3", "ppo_loaded", args=args, kwargs=kwargs, model_id=hex(id(model)), observation_space=getattr(model, "observation_space", None), action_space=getattr(model, "action_space", None), duration_ms=(time.perf_counter() - started) * 1000.0, checkpoint=checkpoint_record)
        return model
    patch_attr(PPO, "load", load_wrapper)

    original_predict = PPO.predict
    @functools.wraps(original_predict)
    def predict_wrapper(self, observation, *args, **kwargs):
        started = time.perf_counter()
        result = original_predict(self, observation, *args, **kwargs)
        SESSION.write("sb3", "policy_predict", model_id=hex(id(self)), observation=observation, args=args, kwargs=kwargs, action=result[0] if isinstance(result, tuple) else result, state=result[1] if isinstance(result, tuple) and len(result) > 1 else None, duration_ms=(time.perf_counter() - started) * 1000.0)
        return result
    patch_attr(PPO, "predict", predict_wrapper)

    class DiagnosticCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)

        def _on_step(self) -> bool:
            SESSION.trainable_steps = int(self.num_timesteps)
            if SESSION.trainable_steps <= 5 or SESSION.trainable_steps % 20 == 0:
                SESSION.write(
                    "sb3",
                    "trainable_step",
                    num_timesteps=self.num_timesteps,
                    n_calls=self.n_calls,
                    locals_keys=sorted(str(key) for key in self.locals.keys()),
                    rewards=self.locals.get("rewards"),
                    dones=self.locals.get("dones"),
                    actions=self.locals.get("actions"),
                    new_obs=self.locals.get("new_obs"),
                    infos=self.locals.get("infos"),
                )
            if SESSION.elapsed >= MAX_WALL_SECONDS or SESSION.trainable_steps >= MAX_TRAINABLE_STEPS or SESSION.physical_steps >= MAX_PHYSICAL_STEPS:
                SESSION.stop_reason = (
                    f"trainable_step_limit_{MAX_TRAINABLE_STEPS}" if SESSION.trainable_steps >= MAX_TRAINABLE_STEPS
                    else f"physical_step_limit_{MAX_PHYSICAL_STEPS}" if SESSION.physical_steps >= MAX_PHYSICAL_STEPS
                    else f"wall_time_limit_{int(MAX_WALL_SECONDS)}s"
                )
                SESSION.stop_requested = True
                return False
            return True

    original_learn = PPO.learn
    @functools.wraps(original_learn)
    def learn_wrapper(self, *args, **kwargs):
        callback = kwargs.get("callback")
        diag_cb = DiagnosticCallback()
        if callback is None:
            kwargs["callback"] = diag_cb
        elif isinstance(callback, (list, tuple)):
            kwargs["callback"] = CallbackList([*callback, diag_cb])
        else:
            kwargs["callback"] = CallbackList([callback, diag_cb])
        SESSION.write("sb3", "learn_begin", model_id=hex(id(self)), args=args, kwargs=kwargs)
        try:
            result = original_learn(self, *args, **kwargs)
            SESSION.write("sb3", "learn_end", model_id=hex(id(self)), num_timesteps=getattr(self, "num_timesteps", None), result=repr(result))
            return result
        except DiagnosticStop:
            raise
        except BaseException as exc:
            log_exception("PPO.learn", exc)
            raise
    patch_attr(PPO, "learn", learn_wrapper)

    original_save = PPO.save
    @functools.wraps(original_save)
    def save_wrapper(self, path, *args, **kwargs):
        SESSION.write("sb3", "model_save_suppressed", model_id=hex(id(self)), requested_path=path, args=args, kwargs=kwargs)
        print(f"[DIAG] Model save suppressed during diagnostics: {path}")
        return None
    patch_attr(PPO, "save", save_wrapper)

    if hasattr(CheckpointCallback, "_on_step"):
        original_checkpoint = CheckpointCallback._on_step
        @functools.wraps(original_checkpoint)
        def checkpoint_wrapper(self):
            SESSION.write("sb3", "checkpoint_write_suppressed", n_calls=getattr(self, "n_calls", None), num_timesteps=getattr(self, "num_timesteps", None), save_path=getattr(self, "save_path", None), name_prefix=getattr(self, "name_prefix", None))
            return True
        patch_attr(CheckpointCallback, "_on_step", checkpoint_wrapper)

    # Run_train imported these symbols directly. Keep its PPO reference aligned.
    run_train_module.PPO = PPO


def snapshot_configs(modules: dict[str, Any]) -> None:
    config_data: dict[str, Any] = {}
    for name, module in modules.items():
        if name.startswith("config.") or name in {"weights_config"}:
            values = {}
            for key in dir(module):
                if key.startswith("__"):
                    continue
                try:
                    value = getattr(module, key)
                except Exception:
                    continue
                if inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                    continue
                values[key] = safe_value(value)
            config_data[name] = values
    (RUN_DIR / "config_snapshot.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    SESSION.write("config", "config_snapshot", configs=config_data)


def snapshot_runtime_environment() -> None:
    import subprocess
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "base_dir": str(BASE_DIR),
        "argv": sys.argv,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {
                "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH", "PATH", "COMPUTERNAME", "USERNAME",
            }
        },
    }
    try:
        import torch
        runtime["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        runtime["torch_error"] = repr(exc)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30
        )
        (RUN_DIR / "pip_freeze.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    except Exception as exc:
        runtime["pip_freeze_error"] = repr(exc)
    (RUN_DIR / "runtime_environment.json").write_text(
        json.dumps(safe_value(runtime), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def snapshot_source() -> None:
    snapshot_root = RUN_DIR / "source_snapshot"
    manifest: list[dict[str, Any]] = []
    allowed = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".txt"}
    excluded_parts = {"diagnostics", "logs", "models", "__pycache__", ".git", ".idea"}
    for path in BASE_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        rel = path.relative_to(BASE_DIR)
        if any(part in excluded_parts for part in rel.parts):
            continue
        try:
            target = snapshot_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            data = path.read_bytes()
            manifest.append({"path": str(rel), "size": len(data), "sha256": sha256_bytes(data)})
        except Exception as exc:
            manifest.append({"path": str(rel), "error": repr(exc)})
    (RUN_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def import_project_modules() -> dict[str, Any]:
    import importlib
    module_names = [
        "Run_train", "Run_train_agent1_original", "drone_env", "agent1p2_env",
        "agent2_landing_env", "object_tracker", "resnet_yolo_tracker",
        "observation_builder", "lidar_processor", "safety_filter",
        "follow_reward_v37", "weights_config", "tracking.target_tracker_manager",
        "tracking.deep_tracker_memory", "tracking.target_memory", "tracking.kalman_bbox",
        "config.flow_config", "config.tracking_config", "config.dynamic_landing_config",
        "config.static_landing_config", "config.servo_config", "config.tracker_config",
        "config.deep_tracker_config", "config.training_config",
    ]
    modules: dict[str, Any] = {}
    for name in module_names:
        try:
            modules[name] = importlib.import_module(name)
            SESSION.write("events", "module_imported", module=name, file=getattr(modules[name], "__file__", None))
        except Exception as exc:
            SESSION.write("exceptions", "module_import_failed", module=name, error=repr(exc), traceback=traceback.format_exc())
    if "Run_train" not in modules:
        raise RuntimeError("Could not import the current Run_train.py")
    return modules


def write_method_statistics() -> None:
    rows = []
    for key, count in SESSION.method_counts.most_common():
        total = SESSION.method_total_ms[key]
        rows.append(
            {
                "method": key,
                "count": count,
                "total_ms": total,
                "mean_ms": total / max(1, count),
            }
        )
    with (RUN_DIR / "method_statistics.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["method", "count", "total_ms", "mean_ms"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary() -> None:
    summary = {
        "diagnostic_version": DIAG_VERSION,
        "started_unix": SESSION.start_wall,
        "finished_unix": time.time(),
        "elapsed_s": SESSION.elapsed,
        "stop_reason": SESSION.stop_reason,
        "physical_steps": SESSION.physical_steps,
        "control_ticks": SESSION.control_ticks,
        "trainable_steps": SESSION.trainable_steps,
        "agent1_physical_steps": SESSION.agent1_steps,
        "agent2_physical_steps": SESSION.agent2_steps,
        "reset_count": SESSION.reset_count,
        "object_count": SESSION.object_count,
        "last_agent": SESSION.last_agent,
        "last_info": safe_value(SESSION.last_info),
        "last_command": safe_value(SESSION.last_command),
        "failure": None if SESSION.failure is None else {
            "type": type(SESSION.failure).__name__,
            "message": str(SESSION.failure),
            "traceback": "".join(traceback.format_exception(type(SESSION.failure), SESSION.failure, SESSION.failure.__traceback__)),
        },
        "log_counts": {name: writer.count for name, writer in SESSION.writers.items()},
        "event_counts": dict(SESSION.event_counts),
        "transition_counts": dict(SESSION.transition_counts),
        "landing_lock_block_counts": dict(SESSION.lock_block_counts),
        "safety_reason_counts": dict(SESSION.safety_reason_counts),
        "anomaly_counts": dict(SESSION.anomaly_counts),
        "loaded_checkpoints": SESSION.loaded_checkpoints,
        "episode_results": SESSION.episode_results,
        "collision_events": safe_value(SESSION.collision_events),
        "limits": {
            "physical_steps": MAX_PHYSICAL_STEPS,
            "trainable_steps": MAX_TRAINABLE_STEPS,
            "wall_seconds": MAX_WALL_SECONDS,
            "frame_save_every": FRAME_SAVE_EVERY,
            "full_resnet_embeddings": FULL_RESNET_EMBEDDINGS,
        },
        "architecture_note": "Run_diag imported and executed the existing Run_train.main(); it did not reimplement the training flow.",
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def package_zip() -> None:
    if DIAG_ZIP.exists():
        DIAG_ZIP.unlink()
    with zipfile.ZipFile(DIAG_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in RUN_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(RUN_DIR)))


def main() -> None:
    global SESSION
    SESSION = DiagnosticSession()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_fp = (RUN_DIR / "00_console_full.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(original_stdout, console_fp)
    sys.stderr = Tee(original_stderr, console_fp)

    print("=" * 110)
    print(f"[DIAG] Run_diag {DIAG_VERSION} wrapping the CURRENT Run_train.py")
    print(f"[DIAG] Project root          : {BASE_DIR}")
    print(f"[DIAG] Physical step limit   : {MAX_PHYSICAL_STEPS}")
    print(f"[DIAG] Trainable step limit  : {MAX_TRAINABLE_STEPS}")
    print(f"[DIAG] Wall-time limit       : {MAX_WALL_SECONDS:.0f}s")
    print(f"[DIAG] Full ResNet vectors   : {FULL_RESNET_EMBEDDINGS}")
    print("[DIAG] Joined authority CSV  : 00_control_authority_timeline.csv")
    print("[DIAG] Critical event frames : frames/events/")
    print(f"[DIAG] Final archive         : {DIAG_ZIP}")
    print("[DIAG] Model/checkpoint writes are disabled for this diagnostic run")
    print("=" * 110)

    failure: BaseException | None = None
    try:
        snapshot_runtime_environment()
        snapshot_source()
        snapshot_model_inventory()
        snapshot_code_audit()
        modules = import_project_modules()
        snapshot_configs(modules)
        install_component_instrumentation(modules)
        install_sb3_instrumentation(modules["Run_train"])

        run_train_path = Path(modules["Run_train"].__file__).resolve()
        SESSION.write(
            "flow",
            "current_run_train_selected",
            path=str(run_train_path),
            sha256=sha256_bytes(run_train_path.read_bytes()),
            training_mode=getattr(modules.get("config.flow_config"), "TRAINING_MODE", None),
        )
        modules["Run_train"].main()
        if SESSION.stop_reason == "running":
            SESSION.stop_reason = "run_train_returned_normally"
    except DiagnosticStop as exc:
        SESSION.stop_reason = str(exc)
        print(f"[DIAG] Stop limit reached: {exc}")
    except KeyboardInterrupt as exc:
        SESSION.stop_reason = "keyboard_interrupt"
        failure = exc
        print("[DIAG] Ctrl+C received. Packaging all collected diagnostics.")
    except BaseException as exc:
        SESSION.stop_reason = f"exception_{type(exc).__name__}"
        SESSION.failure = exc
        failure = exc
        log_exception("Run_diag.main", exc)
        print(f"[DIAG] Run failed with {type(exc).__name__}: {exc}")
        print("[DIAG] All collected data will still be packaged.")
    finally:
        try:
            copy_runtime_artifacts()
            write_automatic_findings()
            write_method_statistics()
            write_summary()
        except Exception as exc:
            print(f"[DIAG] Summary generation failed: {exc}")
        try:
            SESSION.close()
        except Exception:
            pass
        try:
            restore_patches()
        except Exception:
            pass
        try:
            console_fp.flush()
            console_fp.close()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            package_zip()
        except Exception as exc:
            print(f"[DIAG] ZIP packaging failed: {exc}")
            raise

        print("=" * 110)
        print(f"[DIAG] Finished. Stop reason      : {SESSION.stop_reason}")
        print(f"[DIAG] Physical environment steps: {SESSION.physical_steps}")
        print(f"[DIAG] Trainable PPO steps       : {SESSION.trainable_steps}")
        print(f"[DIAG] Agent-1 physical steps    : {SESSION.agent1_steps}")
        print(f"[DIAG] Agent-2 physical steps    : {SESSION.agent2_steps}")
        print(f"[DIAG] Objects registered        : {SESSION.object_count}")
        print(f"[DIAG] Package ready             : {DIAG_ZIP}")
        print("=" * 110)

    # A normal diagnostic stop or Ctrl+C is not re-raised. A real failure is
    # re-raised after diag.zip exists so the console still communicates it.
    if failure is not None and not isinstance(failure, KeyboardInterrupt):
        raise failure


if __name__ == "__main__":
    main()
