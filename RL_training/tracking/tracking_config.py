"""
Task-specific training config.

Python-only config file.
No CLI.
No environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str

    total_timesteps: int
    checkpoint_freq: int
    eval_freq: int
    run_name_prefix: str

    freeze_vz: bool
    altitude_hold_enabled: bool
    enable_forward_motion: bool
    enable_yaw_control: bool
    enable_z_control: bool

    desired_distance_proxy: float
    distance_tolerance: float
    min_target_distance_proxy: float
    block_forward_when_too_close: bool

    reset_takeoff_altitude_m: float
    altitude_hold_target_m: float
    min_safe_altitude_m: float
    min_termination_altitude_m: float
    max_termination_altitude_m: float

    max_episode_steps: int
    focus_fail_sec: float

    non_match_timeout_sec: float
    approach_warmup_sec: float
    not_approaching_timeout_sec: float
    approach_target_distance_proxy: float
    approach_min_improvement: float
    approach_goal_distance_m: float
    approach_min_improvement_m: float
    not_approaching_penalty_growth_per_sec: float
    max_initial_yaw_delta_deg: float
    target_lost_hard_fail_penalty: float

    w_center: float
    w_distance: float
    w_visibility: float
    w_lost_target: float
    w_altitude_safe: float
    w_altitude_low_penalty: float
    w_altitude_high_penalty: float
    w_smooth_follow: float
    w_control: float
    w_action_delta: float
    w_obstacle: float
    w_safety_intervention: float
    w_slow_or_stuck: float

    penalty_collision: float
    penalty_timeout: float


TASK_CONFIG = TaskConfig(
    name="tracking",
    description=(
        "Chase/front-camera tracking fine-tuning agent. The policy learns to chase the target, "
        "stay strongly centered, and avoid unnecessary yaw while altitude is stabilized."
    ),

    total_timesteps=40_000,
    checkpoint_freq=10_000,
    eval_freq=10_000,
    run_name_prefix="tracking",

    # Tracking phase should not learn landing/descent yet.
    # Altitude hold keeps the drone alive and isolates yaw/forward behavior.
    freeze_vz=True,
    altitude_hold_enabled=True,
    enable_forward_motion=True,
    enable_yaw_control=True,
    enable_z_control=False,

    desired_distance_proxy=0.65,
    distance_tolerance=0.10,
    min_target_distance_proxy=0.45,
    block_forward_when_too_close=False,

    reset_takeoff_altitude_m=5.0,
    altitude_hold_target_m=5.0,
    min_safe_altitude_m=2.0,
    min_termination_altitude_m=0.75,
    max_termination_altitude_m=12.0,

    max_episode_steps=700,
    focus_fail_sec=3.0,

    # Strict chase/center fine-tuning termination rules.
    non_match_timeout_sec=3.0,
    approach_warmup_sec=6.0,
    not_approaching_timeout_sec=5.0,
    approach_target_distance_proxy=0.88,  # legacy bbox-proxy fallback, no longer primary
    approach_min_improvement=0.010,      # legacy bbox-proxy fallback, no longer primary
    approach_goal_distance_m=2.50,
    approach_min_improvement_m=0.10,
    not_approaching_penalty_growth_per_sec=1000.0,
    max_initial_yaw_delta_deg=25.0,
    target_lost_hard_fail_penalty=12000.0,

    # Tracking: prioritize stable centering and smooth behavior.
    w_center=45.0,
    w_distance=12.0,
    w_visibility=2.0,
    w_lost_target=22.0,
    w_altitude_safe=5.0,
    w_altitude_low_penalty=14.0,
    w_altitude_high_penalty=4.0,
    w_smooth_follow=1.5,
    w_control=0.06,
    w_action_delta=0.35,
    w_obstacle=6.0,
    w_safety_intervention=2.0,
    w_slow_or_stuck=0.8,

    penalty_collision=80.0,
    penalty_timeout=4.0,
)
