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
    name="static_landing",
    description=(
        "Static target landing agent. The target vehicle is not moving. "
        "The policy learns yaw/forward/Z control for safe descent onto a fixed visual target."
    ),

    total_timesteps=250_000,
    checkpoint_freq=10_000,
    eval_freq=10_000,
    run_name_prefix="static_landing",

    # Static landing needs Z control enabled.
    freeze_vz=False,
    altitude_hold_enabled=False,
    enable_forward_motion=True,
    enable_yaw_control=True,
    enable_z_control=True,

    # Distance proxy is bbox-size based. Higher means closer/larger target.
    desired_distance_proxy=0.96,
    distance_tolerance=0.08,
    min_target_distance_proxy=0.88,
    block_forward_when_too_close=True,

    reset_takeoff_altitude_m=5.0,
    altitude_hold_target_m=5.0,
    min_safe_altitude_m=1.4,
    min_termination_altitude_m=0.55,
    max_termination_altitude_m=12.0,

    max_episode_steps=450,
    focus_fail_sec=4.0,

    # Strong non-linear center reward is handled in follow_reward_v37.py.
    # These are task weights.
    w_center=18.0,
    w_distance=8.0,
    w_visibility=5.0,
    w_lost_target=10.0,
    w_altitude_safe=4.0,
    w_altitude_low_penalty=12.0,
    w_altitude_high_penalty=4.0,
    w_smooth_follow=2.0,
    w_control=0.05,
    w_action_delta=0.35,
    w_obstacle=6.0,
    w_safety_intervention=2.5,
    w_slow_or_stuck=0.4,

    penalty_collision=80.0,
    penalty_timeout=8.0,
)
