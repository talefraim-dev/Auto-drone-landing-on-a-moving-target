"""Read-only 100-landing benchmark for the final cooperative UAV models.

The benchmark uses only:
    models/FINAL_MODELS/agent1_final.zip
    models/FINAL_MODELS/agent2_final.zip

It never trains, saves, overwrites, or creates model checkpoints. The only
artifact it creates is a CSV file containing one row per landing attempt.

The CSV is designed as raw input for later statistical aggregation and plots
across different target types and target sizes.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NoReturn
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from alternating_cotraining_env import RpcDominantAgent2RewardEnv
from Run_train_alternating_agents import build_agent2_config, build_parallel_env


AGENT1_MODEL = Path("models") / "FINAL_MODELS" / "agent1_final.zip"
AGENT2_MODEL = Path("models") / "FINAL_MODELS" / "agent2_final.zip"

LEGACY_AGENT2_OBS_DIM = 37
RUNTIME_AGENT2_OBS_DIM = 46
SUCCESS_REASONS = {"landing_collision_success"}


class LegacyAgent2ObservationAdapter(gym.ObservationWrapper):
    """Expose the legacy 37-value observation prefix to a frozen policy."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        runtime_shape = tuple(getattr(env.observation_space, "shape", ()) or ())
        if runtime_shape != (RUNTIME_AGENT2_OBS_DIM,):
            raise RuntimeError(
                f"Expected runtime observation shape {(RUNTIME_AGENT2_OBS_DIM,)}, "
                f"got {runtime_shape}."
            )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(LEGACY_AGENT2_OBS_DIM,),
            dtype=np.float32,
        )

    def observation(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.size != RUNTIME_AGENT2_OBS_DIM:
            raise RuntimeError(
                f"Expected {RUNTIME_AGENT2_OBS_DIM} runtime observations, "
                f"got {values.size}."
            )
        return values[:LEGACY_AGENT2_OBS_DIM].copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _space_dim(space: gym.Space) -> int:
    shape = tuple(getattr(space, "shape", ()) or ())
    if len(shape) != 1:
        raise RuntimeError(f"Expected a one-dimensional space, got {shape}.")
    return int(shape[0])


def _forbidden_operation(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise RuntimeError(
        "Training and checkpoint writing are disabled in this benchmark."
    )


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("nan")


def _median(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.median(finite_values)) if finite_values else float("nan")


def _maximum(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return max(finite_values) if finite_values else float("nan")


def _minimum(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return min(finite_values) if finite_values else float("nan")


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / max(1, int(denominator))


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if value is None:
        return ""
    return value


def _landing_xy_error(final_info: dict[str, Any]) -> tuple[float, str]:
    """Choose the strongest available Euclidean target-center error in metres."""

    candidates = (
        ("collision_xy_combined_error_m", "collision_xy_combined"),
        ("collision_xy_geometric_error_m", "collision_xy_geometric"),
        (
            "collision_terminal_snapshot_center_error_m",
            "terminal_snapshot_center",
        ),
        ("collision_xy_legacy_error_m", "collision_xy_legacy"),
        ("range_terminal_snapshot_center_error_m", "range_terminal_snapshot"),
    )
    for key, source in candidates:
        value = _finite(final_info.get(key))
        if math.isfinite(value):
            return value, source

    x_value = _finite(
        final_info.get(
            "bottom_relative_position_x_m",
            final_info.get("visual_relative_position_body_x_m"),
        )
    )
    y_value = _finite(
        final_info.get(
            "bottom_relative_position_y_m",
            final_info.get("visual_relative_position_body_y_m"),
        )
    )
    if math.isfinite(x_value) and math.isfinite(y_value):
        return math.hypot(x_value, y_value), "relative_xy_fallback"

    return float("nan"), "unavailable"


@dataclass
class AttemptTelemetry:
    """Accumulate full-flight and cooperative-phase metrics."""

    physical_agent1_steps: int = 0
    agent1_match_steps: int = 0
    agent1_pred_steps: int = 0
    agent1_lost_steps: int = 0
    bottom_live_steps_full: int = 0
    bottom_confirmed_steps_full: int = 0

    cooperative_steps: int = 0
    bottom_live_steps_coop: int = 0
    bottom_confirmed_steps_coop: int = 0
    predicted_steps_coop: int = 0
    lost_steps_coop: int = 0

    terminal_steps: int = 0
    range_valid_steps: int = 0
    range_reliable_steps: int = 0
    descent_steps: int = 0
    hold_steps: int = 0
    climb_steps: int = 0

    first_bbox_area_norm: float = float("nan")
    handoff_bbox_area_norm: float = float("nan")
    final_bbox_area_norm: float = float("nan")

    bbox_areas: list[float] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    center_errors_norm: list[float] = field(default_factory=list)
    range_medians_m: list[float] = field(default_factory=list)
    range_spreads_m: list[float] = field(default_factory=list)
    commanded_xy_speeds_mps: list[float] = field(default_factory=list)
    commanded_vz_mps: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    handoff_step: int | None = None
    handoff_elapsed_s: float | None = None
    terminal_start_step: int | None = None
    terminal_start_elapsed_s: float | None = None

    def record_agent1(self, info: dict[str, Any]) -> None:
        """Record one physical Agent-1 command, including preparation steps."""

        self.physical_agent1_steps += 1

        mode = str(
            info.get(
                "tracking_mode",
                info.get("front_full_tracker_mode", "LOST"),
            )
            or "LOST"
        ).upper()
        if mode == "MATCH" or mode.startswith("MATCH_"):
            self.agent1_match_steps += 1
        elif mode == "PRED" or mode.startswith("PRED"):
            self.agent1_pred_steps += 1
        else:
            self.agent1_lost_steps += 1

        bottom_live = bool(
            info.get("bottom_match_live", info.get("bottom_match", False))
        )
        bottom_confirmed = bool(
            info.get(
                "bottom_match_confirmed",
                info.get("bottom_confirmed", False),
            )
        )
        self.bottom_live_steps_full += int(bottom_live)
        self.bottom_confirmed_steps_full += int(bottom_live and bottom_confirmed)

        self._record_common_visual_metrics(info)

    def record_cooperative(
        self,
        info: dict[str, Any],
        reward: float,
        elapsed_s: float,
    ) -> None:
        """Record one Agent-2/cooperative environment step."""

        self.cooperative_steps += 1
        self.rewards.append(float(reward))

        bottom_live = bool(info.get("bottom_match_live", False))
        bottom_confirmed = bool(info.get("bottom_match_confirmed", False))
        self.bottom_live_steps_coop += int(bottom_live)
        self.bottom_confirmed_steps_coop += int(bottom_live and bottom_confirmed)

        tracker_mode = str(
            info.get(
                "bottom_full_tracker_mode",
                info.get("tracker_mode", "LOST"),
            )
            or "LOST"
        ).upper()
        if tracker_mode == "PRED" or tracker_mode.startswith("PRED"):
            self.predicted_steps_coop += 1
        if not bottom_live and not tracker_mode.startswith("PRED"):
            self.lost_steps_coop += 1

        handoff = bool(info.get("range_terminal_handoff_active", False))
        if handoff and self.handoff_step is None:
            self.handoff_step = self.cooperative_steps
            self.handoff_elapsed_s = elapsed_s
            self.handoff_bbox_area_norm = _finite(
                info.get("bottom_bbox_area_norm")
            )

        terminal = bool(
            handoff
            or info.get("terminal_range_governed_z", False)
            or str(info.get("vertical_control_state", "")).startswith(
                "SENSOR_FINAL_TERMINAL"
            )
        )
        if terminal:
            self.terminal_steps += 1
            if self.terminal_start_step is None:
                self.terminal_start_step = self.cooperative_steps
                self.terminal_start_elapsed_s = elapsed_s

        valid_count = int(info.get("range_valid_count", 0) or 0)
        self.range_valid_steps += int(valid_count >= 4)
        reliable = bool(info.get("range_height_reliable", False))
        self.range_reliable_steps += int(reliable)

        median_range = _finite(info.get("range_mean_m"))
        spread = _finite(info.get("range_spread_m"))
        if math.isfinite(median_range):
            self.range_medians_m.append(median_range)
        if reliable and math.isfinite(spread):
            self.range_spreads_m.append(spread)

        vx = _finite(
            info.get(
                "commanded_vx_mps",
                info.get("agent1_commanded_vx_mps", 0.0),
            ),
            0.0,
        )
        vy = _finite(
            info.get(
                "commanded_vy_mps",
                info.get("agent1_commanded_vy_mps", 0.0),
            ),
            0.0,
        )
        vz = _finite(
            info.get(
                "applied_vz_mps",
                info.get("agent1_commanded_vz_mps", 0.0),
            ),
            0.0,
        )
        self.commanded_xy_speeds_mps.append(math.hypot(vx, vy))
        self.commanded_vz_mps.append(vz)
        self.descent_steps += int(vz > 1.0e-4)
        self.climb_steps += int(vz < -1.0e-4)
        self.hold_steps += int(abs(vz) <= 1.0e-4)

        self._record_common_visual_metrics(info)
        self.final_bbox_area_norm = _finite(
            info.get("bottom_bbox_area_norm"),
            self.final_bbox_area_norm,
        )

    def _record_common_visual_metrics(self, info: dict[str, Any]) -> None:
        area = _finite(info.get("bottom_bbox_area_norm"))
        if math.isfinite(area) and area > 0.0:
            if not math.isfinite(self.first_bbox_area_norm):
                self.first_bbox_area_norm = area
            self.bbox_areas.append(area)

        similarity = _finite(info.get("bottom_similarity"))
        if math.isfinite(similarity):
            self.similarities.append(similarity)

        center_error = _finite(
            info.get(
                "bottom_touchdown_bbox_center_error",
                info.get("bottom_center_error"),
            )
        )
        if math.isfinite(center_error):
            self.center_errors_norm.append(center_error)


CSV_FIELDS = [
    "run_id",
    "attempt",
    "attempt_started_at",
    "target_label",
    "target_physical_area_m2",
    "target_width_m",
    "target_length_m",
    "initial_altitude_m",
    "success",
    "physical_contact",
    "landing_accepted",
    "rpc_latch_success",
    "termination_reason",
    "failure_reason",
    "error_type",
    "error_message",
    "total_flight_steps",
    "agent1_prepare_steps",
    "cooperative_landing_steps",
    "total_flight_duration_s",
    "handoff_step",
    "handoff_time_s",
    "terminal_steps",
    "terminal_duration_s",
    "total_reward",
    "agent1_match_percent_full_flight",
    "agent1_pred_percent_full_flight",
    "agent1_lost_percent_full_flight",
    "bottom_live_match_percent_full_flight",
    "bottom_confirmed_match_percent_full_flight",
    "bottom_live_match_percent_landing_phase",
    "bottom_confirmed_match_percent_landing_phase",
    "bottom_pred_percent_landing_phase",
    "bottom_lost_percent_landing_phase",
    "landing_xy_euclidean_error_m",
    "landing_xy_error_source",
    "collision_xy_threshold_m",
    "center_error_norm_first",
    "center_error_norm_mean",
    "center_error_norm_median",
    "center_error_norm_best",
    "center_error_norm_final",
    "bottom_similarity_mean",
    "bottom_similarity_best",
    "bottom_similarity_final",
    "target_bbox_area_norm_first",
    "target_bbox_area_norm_handoff",
    "target_bbox_area_norm_mean",
    "target_bbox_area_norm_max",
    "target_bbox_area_norm_final",
    "range_valid_percent",
    "range_reliable_percent",
    "minimum_range_m",
    "median_range_m",
    "maximum_reliable_range_spread_m",
    "descent_steps",
    "hold_steps",
    "climb_steps",
    "average_horizontal_speed_mps",
    "maximum_horizontal_speed_mps",
    "average_vertical_command_mps",
    "maximum_descent_command_mps",
    "maximum_climb_command_mps",
    "contact_similarity",
    "contact_corner_similarity",
    "contact_similarity_margin",
    "target_collision_object",
    "target_collision_matches_expected",
    "agent1_model_sha256",
    "agent2_model_sha256",
    "models_unchanged_after_attempt",
]


def _parse_attempt_exception(error_type: str, error_message: str) -> tuple[str, str]:
    """Convert benchmark exceptions into stable failure labels."""

    message = (error_message or "").strip()
    if not error_type:
        return "", ""

    if error_type == "RuntimeError":
        if "Frozen Agent 1 did not reach landing-ready state" in message:
            match = re.search(r"Last reason=([^\s]+)", message)
            last_reason = match.group(1) if match else "unknown"
            termination_reason = "agent1_landing_ready_timeout"
            failure_reason = (
                f"agent1_not_landing_ready:last_reason={last_reason}"
            )
            return termination_reason, failure_reason

    termination_reason = "tester_exception"
    failure_reason = message or error_type
    return termination_reason, failure_reason


def _resolve_failure_reason(
    *,
    success: bool,
    termination_reason: str,
    error_type: str,
    error_message: str,
) -> str:
    if success:
        return ""
    if error_type:
        parsed_termination_reason, parsed_failure_reason = _parse_attempt_exception(
            error_type, error_message
        )
        if parsed_failure_reason:
            return parsed_failure_reason
        if parsed_termination_reason:
            return parsed_termination_reason
    return termination_reason or error_message or error_type or "unknown_failure"




def _build_row(
    *,
    run_id: str,
    attempt: int,
    attempt_started_at: str,
    target_label: str,
    target_physical_area_m2: float,
    target_width_m: float,
    target_length_m: float,
    initial_altitude_m: float,
    telemetry: AttemptTelemetry,
    final_info: dict[str, Any],
    termination_reason: str,
    duration_s: float,
    agent1_prepare_steps: int,
    total_reward: float,
    error_type: str,
    error_message: str,
    models_unchanged: bool,
    agent1_hash: str,
    agent2_hash: str,
) -> dict[str, Any]:
    physical_contact = bool(
        final_info.get("target_collision", False)
        or final_info.get("collision_detected", False)
        or "collision" in termination_reason
    )
    landing_accepted = bool(
        termination_reason in SUCCESS_REASONS
        or final_info.get("success", False)
        or final_info.get("rpc_success", False)
    )
    latch_success = bool(final_info.get("latch_succeeded", False))
    success = bool(
        physical_contact
        and landing_accepted
        and latch_success
        and not error_type
        and models_unchanged
    )

    failure_reason = _resolve_failure_reason(
        success=success,
        termination_reason=termination_reason,
        error_type=error_type,
        error_message=error_message,
    )

    landing_xy_error, landing_xy_source = _landing_xy_error(final_info)

    terminal_duration_s = float("nan")
    if telemetry.terminal_start_elapsed_s is not None:
        terminal_duration_s = max(
            0.0,
            duration_s - telemetry.terminal_start_elapsed_s,
        )

    total_flight_steps = int(
        agent1_prepare_steps + telemetry.cooperative_steps
    )

    vertical_values = telemetry.commanded_vz_mps
    descent_values = [value for value in vertical_values if value > 0.0]
    climb_values = [abs(value) for value in vertical_values if value < 0.0]

    return {
        "run_id": run_id,
        "attempt": attempt,
        "attempt_started_at": attempt_started_at,
        "target_label": target_label,
        "target_physical_area_m2": target_physical_area_m2,
        "target_width_m": target_width_m,
        "target_length_m": target_length_m,
        "initial_altitude_m": initial_altitude_m,
        "success": success,
        "physical_contact": physical_contact,
        "landing_accepted": landing_accepted,
        "rpc_latch_success": latch_success,
        "termination_reason": termination_reason,
        "failure_reason": failure_reason,
        "error_type": error_type,
        "error_message": error_message,
        "total_flight_steps": total_flight_steps,
        "agent1_prepare_steps": agent1_prepare_steps,
        "cooperative_landing_steps": telemetry.cooperative_steps,
        "total_flight_duration_s": duration_s,
        "handoff_step": telemetry.handoff_step,
        "handoff_time_s": telemetry.handoff_elapsed_s,
        "terminal_steps": telemetry.terminal_steps,
        "terminal_duration_s": terminal_duration_s,
        "total_reward": total_reward,
        "agent1_match_percent_full_flight": _percent(
            telemetry.agent1_match_steps,
            telemetry.physical_agent1_steps,
        ),
        "agent1_pred_percent_full_flight": _percent(
            telemetry.agent1_pred_steps,
            telemetry.physical_agent1_steps,
        ),
        "agent1_lost_percent_full_flight": _percent(
            telemetry.agent1_lost_steps,
            telemetry.physical_agent1_steps,
        ),
        "bottom_live_match_percent_full_flight": _percent(
            telemetry.bottom_live_steps_full,
            telemetry.physical_agent1_steps,
        ),
        "bottom_confirmed_match_percent_full_flight": _percent(
            telemetry.bottom_confirmed_steps_full,
            telemetry.physical_agent1_steps,
        ),
        "bottom_live_match_percent_landing_phase": _percent(
            telemetry.bottom_live_steps_coop,
            telemetry.cooperative_steps,
        ),
        "bottom_confirmed_match_percent_landing_phase": _percent(
            telemetry.bottom_confirmed_steps_coop,
            telemetry.cooperative_steps,
        ),
        "bottom_pred_percent_landing_phase": _percent(
            telemetry.predicted_steps_coop,
            telemetry.cooperative_steps,
        ),
        "bottom_lost_percent_landing_phase": _percent(
            telemetry.lost_steps_coop,
            telemetry.cooperative_steps,
        ),
        "landing_xy_euclidean_error_m": landing_xy_error,
        "landing_xy_error_source": landing_xy_source,
        "collision_xy_threshold_m": _finite(
            final_info.get("collision_xy_threshold_m")
        ),
        "center_error_norm_first": (
            telemetry.center_errors_norm[0]
            if telemetry.center_errors_norm
            else float("nan")
        ),
        "center_error_norm_mean": _mean(telemetry.center_errors_norm),
        "center_error_norm_median": _median(telemetry.center_errors_norm),
        "center_error_norm_best": _minimum(telemetry.center_errors_norm),
        "center_error_norm_final": (
            telemetry.center_errors_norm[-1]
            if telemetry.center_errors_norm
            else float("nan")
        ),
        "bottom_similarity_mean": _mean(telemetry.similarities),
        "bottom_similarity_best": _maximum(telemetry.similarities),
        "bottom_similarity_final": _finite(
            final_info.get("bottom_similarity")
        ),
        "target_bbox_area_norm_first": telemetry.first_bbox_area_norm,
        "target_bbox_area_norm_handoff": telemetry.handoff_bbox_area_norm,
        "target_bbox_area_norm_mean": _mean(telemetry.bbox_areas),
        "target_bbox_area_norm_max": _maximum(telemetry.bbox_areas),
        "target_bbox_area_norm_final": telemetry.final_bbox_area_norm,
        "range_valid_percent": _percent(
            telemetry.range_valid_steps,
            telemetry.cooperative_steps,
        ),
        "range_reliable_percent": _percent(
            telemetry.range_reliable_steps,
            telemetry.cooperative_steps,
        ),
        "minimum_range_m": _minimum(telemetry.range_medians_m),
        "median_range_m": _median(telemetry.range_medians_m),
        "maximum_reliable_range_spread_m": _maximum(
            telemetry.range_spreads_m
        ),
        "descent_steps": telemetry.descent_steps,
        "hold_steps": telemetry.hold_steps,
        "climb_steps": telemetry.climb_steps,
        "average_horizontal_speed_mps": _mean(
            telemetry.commanded_xy_speeds_mps
        ),
        "maximum_horizontal_speed_mps": _maximum(
            telemetry.commanded_xy_speeds_mps
        ),
        "average_vertical_command_mps": _mean(vertical_values),
        "maximum_descent_command_mps": _maximum(descent_values),
        "maximum_climb_command_mps": _maximum(climb_values),
        "contact_similarity": _finite(
            final_info.get("collision_contact_center_similarity")
        ),
        "contact_corner_similarity": _finite(
            final_info.get("collision_contact_corner_similarity")
        ),
        "contact_similarity_margin": _finite(
            final_info.get("collision_contact_center_margin")
        ),
        "target_collision_object": str(
            final_info.get("collision_object_name", "") or ""
        ),
        "target_collision_matches_expected": bool(
            final_info.get("collision_object_matches_target", False)
        ),
        "agent1_model_sha256": agent1_hash,
        "agent2_model_sha256": agent2_hash,
        "models_unchanged_after_attempt": models_unchanged,
    }


def _append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                field: _csv_value(row.get(field))
                for field in CSV_FIELDS
            }
        )
        handle.flush()


def _install_agent1_telemetry_hook(
    parallel_env: Any,
    telemetry_supplier: Callable[[], AttemptTelemetry],
) -> None:
    """Observe Agent-1 physical steps without altering its actions."""

    original_step = parallel_env._agent1_step

    def wrapped_step(action: Any):
        result = original_step(action)
        try:
            info = dict(result[4] or {})
            telemetry_supplier().record_agent1(info)
        except Exception as exc:
            print(
                "[BENCHMARK] Warning: Agent-1 telemetry sample failed: "
                f"{type(exc).__name__}: {exc}"
            )
        return result

    parallel_env._agent1_step = wrapped_step


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only landing attempts and save one statistical CSV row "
            "per attempt."
        )
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=100,
        help="Number of landing attempts. Default: 100.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=700,
        help="Maximum cooperative Agent-2 steps per attempt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV output path. Default: "
            "statistics_data/landing_trials_<timestamp>.csv"
        ),
    )
    parser.add_argument(
        "--target-label",
        default="unspecified_target",
        help="Human-readable target name stored in every CSV row.",
    )
    parser.add_argument(
        "--target-area-m2",
        type=float,
        default=float("nan"),
        help="Optional physical landing-surface area in square metres.",
    )
    parser.add_argument(
        "--target-width-m",
        type=float,
        default=float("nan"),
        help="Optional physical landing-surface width in metres.",
    )
    parser.add_argument(
        "--target-length-m",
        type=float,
        default=float("nan"),
        help="Optional physical landing-surface length in metres.",
    )
    parser.add_argument(
        "--initial-altitude-m",
        type=float,
        default=None,
        help=(
            "Optional experiment override for both reset takeoff altitude "
            "and Agent-1 altitude-hold target. When omitted, the frozen "
            "Agent-1 training snapshot value is preserved."
        ),
    )
    parser.add_argument(
        "--continue-after-exception",
        action="store_true",
        help=(
            "Record an exception row and continue. By default, unexpected "
            "exceptions stop the benchmark after preserving completed rows."
        ),
    )
    args = parser.parse_args()

    if args.attempts <= 0:
        raise ValueError("--attempts must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.initial_altitude_m is not None and args.initial_altitude_m <= 0.0:
        raise ValueError("--initial-altitude-m must be greater than zero.")

    missing = [
        path for path in (AGENT1_MODEL, AGENT2_MODEL) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Final model file(s) missing:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = (
        args.output
        if args.output is not None
        else Path("statistics_data") / f"landing_trials_{run_id}.csv"
    )

    initial_hashes = {
        "agent1": _sha256(AGENT1_MODEL),
        "agent2": _sha256(AGENT2_MODEL),
    }

    print("=" * 110)
    print("[LANDING BENCHMARK] READ-ONLY FINAL-MODELS DATA COLLECTION")
    print(f"[LANDING BENCHMARK] Attempts : {args.attempts}")
    print(f"[LANDING BENCHMARK] Agent 1  : {AGENT1_MODEL}")
    print(f"[LANDING BENCHMARK] Agent 2  : {AGENT2_MODEL}")
    print(f"[LANDING BENCHMARK] Device   : {device}")
    print(f"[LANDING BENCHMARK] CSV      : {csv_path}")
    print("[LANDING BENCHMARK] learn/save/checkpoint creation: BLOCKED")
    print("=" * 110)

    agent2_model = PPO.load(str(AGENT2_MODEL), env=None, device=device)
    agent2_model.policy.set_training_mode(False)
    agent2_model.learn = _forbidden_operation  # type: ignore[method-assign]
    agent2_model.save = _forbidden_operation  # type: ignore[method-assign]

    checkpoint_obs_dim = _space_dim(agent2_model.observation_space)

    agent2_cfg = build_agent2_config()
    agent2_cfg.force_descent_while_bottom_match = False

    parallel_env = build_parallel_env(
        agent1_checkpoint=AGENT1_MODEL,
        device=device,
        target_identity=None,
        agent2_config=agent2_cfg,
    )

    # Experiment-only override applied after the checkpoint snapshot has
    # created Agent 1's environment. This keeps the frozen model and snapshot
    # untouched while allowing controlled altitude-vs-target-size tests.
    if args.initial_altitude_m is not None:
        altitude_m = float(args.initial_altitude_m)
        parallel_env.agent1_env.cfg.reset_takeoff_altitude_m = altitude_m
        parallel_env.agent1_env.cfg.altitude_hold_target_m = altitude_m

    effective_initial_altitude_m = float(
        parallel_env.agent1_env.cfg.reset_takeoff_altitude_m
    )
    effective_hold_altitude_m = float(
        parallel_env.agent1_env.cfg.altitude_hold_target_m
    )
    print(
        "[LANDING BENCHMARK] Initial altitude override: "
        f"reset={effective_initial_altitude_m:.2f}m "
        f"hold={effective_hold_altitude_m:.2f}m "
        f"source={'tester' if args.initial_altitude_m is not None else 'snapshot'}"
    )

    # Block training/checkpoint writes for the frozen Agent-1 policy too.
    parallel_env.agent1_model.policy.set_training_mode(False)
    parallel_env.agent1_model.learn = _forbidden_operation
    parallel_env.agent1_model.save = _forbidden_operation

    current_telemetry = AttemptTelemetry()

    def telemetry_supplier() -> AttemptTelemetry:
        return current_telemetry

    _install_agent1_telemetry_hook(parallel_env, telemetry_supplier)

    base_env: gym.Env = RpcDominantAgent2RewardEnv(
        parallel_env,
        collapse_terminal_rollout=False,
    )

    runtime_obs_dim = _space_dim(base_env.observation_space)
    if checkpoint_obs_dim == runtime_obs_dim:
        env: gym.Env = base_env
        compatibility = "NATIVE"
    elif (
        checkpoint_obs_dim == LEGACY_AGENT2_OBS_DIM
        and runtime_obs_dim == RUNTIME_AGENT2_OBS_DIM
    ):
        env = LegacyAgent2ObservationAdapter(base_env)
        compatibility = "LEGACY_37_TO_RUNTIME_46_READ_ONLY"
    else:
        base_env.close()
        raise RuntimeError(
            "Agent-2 observation mismatch: "
            f"checkpoint={checkpoint_obs_dim}, runtime={runtime_obs_dim}."
        )

    print(
        "[LANDING BENCHMARK] Observation compatibility: "
        f"{compatibility} ({checkpoint_obs_dim} -> {runtime_obs_dim})"
    )

    successful_attempts = 0
    failed_attempts = 0

    try:
        for attempt in range(1, args.attempts + 1):
            current_telemetry = AttemptTelemetry()
            attempt_started_at = datetime.now().isoformat(timespec="seconds")
            attempt_started = time.monotonic()
            final_info: dict[str, Any] = {}
            termination_reason = ""
            total_reward = 0.0
            error_type = ""
            error_message = ""
            terminated = False
            truncated = False

            print()
            print("-" * 110)
            print(
                f"[LANDING BENCHMARK] Attempt {attempt}/{args.attempts} "
                f"| target={args.target_label}"
            )
            print("-" * 110)

            try:
                observation, reset_info = env.reset()
                reset_info = dict(reset_info or {})
                final_info = reset_info

                for _step in range(1, args.max_steps + 1):
                    with torch.inference_mode():
                        action, _state = agent2_model.predict(
                            observation,
                            deterministic=True,
                        )

                    observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    final_info = dict(info or {})
                    total_reward += float(reward)
                    elapsed_s = time.monotonic() - attempt_started
                    current_telemetry.record_cooperative(
                        final_info,
                        float(reward),
                        elapsed_s,
                    )

                    if terminated or truncated:
                        termination_reason = str(
                            final_info.get("termination_reason", "")
                            or "UNKNOWN"
                        )
                        break
                else:
                    termination_reason = "tester_max_steps_reached"

            except KeyboardInterrupt:
                print(
                    "[LANDING BENCHMARK] Interrupted by user. "
                    "Completed CSV rows are already saved."
                )
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                termination_reason, parsed_failure_reason = _parse_attempt_exception(
                    error_type, error_message
                )
                print(
                    f"[LANDING BENCHMARK] Attempt exception: "
                    f"{error_type}: {error_message}"
                )
                if parsed_failure_reason:
                    print(
                        "[LANDING BENCHMARK] Exception converted to failure row: "
                        f"termination_reason={termination_reason} "
                        f"failure_reason={parsed_failure_reason}"
                    )
                traceback.print_exc()

            duration_s = time.monotonic() - attempt_started

            current_hashes = {
                "agent1": _sha256(AGENT1_MODEL),
                "agent2": _sha256(AGENT2_MODEL),
            }
            models_unchanged = current_hashes == initial_hashes
            if not models_unchanged:
                raise RuntimeError(
                    "A final model file changed during the benchmark. "
                    "Execution was stopped immediately."
                )

            # Agent1P2Env exposes the cumulative count after reset. The hook
            # gives the exact per-attempt physical count, so preparation steps
            # are the physical Agent-1 steps not paired with cooperative steps.
            agent1_prepare_steps = max(
                0,
                current_telemetry.physical_agent1_steps
                - current_telemetry.cooperative_steps,
            )

            row = _build_row(
                run_id=run_id,
                attempt=attempt,
                attempt_started_at=attempt_started_at,
                target_label=args.target_label,
                target_physical_area_m2=args.target_area_m2,
                target_width_m=args.target_width_m,
                target_length_m=args.target_length_m,
                initial_altitude_m=effective_initial_altitude_m,
                telemetry=current_telemetry,
                final_info=final_info,
                termination_reason=termination_reason,
                duration_s=duration_s,
                agent1_prepare_steps=agent1_prepare_steps,
                total_reward=total_reward,
                error_type=error_type,
                error_message=error_message,
                models_unchanged=models_unchanged,
                agent1_hash=initial_hashes["agent1"],
                agent2_hash=initial_hashes["agent2"],
            )
            _append_csv(csv_path, row)

            if bool(row["success"]):
                successful_attempts += 1
            else:
                failed_attempts += 1

            xy_text = (
                f"{row['landing_xy_euclidean_error_m']:.3f}m"
                if isinstance(row["landing_xy_euclidean_error_m"], float)
                and math.isfinite(row["landing_xy_euclidean_error_m"])
                else "n/a"
            )
            print(
                "[LANDING BENCHMARK] "
                f"Attempt={attempt} "
                f"success={int(bool(row['success']))} "
                f"steps={row['total_flight_steps']} "
                f"time={row['total_flight_duration_s']:.2f}s "
                f"confirmedMatch="
                f"{row['bottom_confirmed_match_percent_full_flight']:.1f}% "
                f"landingXY={xy_text} "
                f"reason={termination_reason}"
            )
            print(f"[LANDING BENCHMARK] CSV row saved: {csv_path}")

            if error_type and not args.continue_after_exception:
                print(
                    "[LANDING BENCHMARK] Exception recorded as failure. "
                    "Benchmark continues to the next attempt."
                )

    finally:
        env.close()

    final_hashes = {
        "agent1": _sha256(AGENT1_MODEL),
        "agent2": _sha256(AGENT2_MODEL),
    }
    models_unchanged = final_hashes == initial_hashes

    completed = successful_attempts + failed_attempts
    success_rate = _percent(successful_attempts, completed)

    print()
    print("=" * 110)
    print("[LANDING BENCHMARK] FINAL SUMMARY")
    print(f"Completed attempts : {completed}")
    print(f"Successful         : {successful_attempts}")
    print(f"Failed             : {failed_attempts}")
    print(f"Success rate       : {success_rate:.2f}%")
    print(f"CSV                : {csv_path}")
    print(
        f"Models unchanged   : {'PASS' if models_unchanged else 'FAIL'}"
    )
    print(
        f"LANDING_BENCHMARK_OK={1 if models_unchanged and completed > 0 else 0}"
    )
    print("=" * 110)

    return 0 if models_unchanged and completed > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
