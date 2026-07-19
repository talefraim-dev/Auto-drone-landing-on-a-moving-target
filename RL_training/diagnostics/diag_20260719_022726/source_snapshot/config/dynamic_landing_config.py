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
    name="dynamic_landing",
    description=(
        "Dynamic target landing agent. The hardest task: the target may move, "
        "so the policy must track, approach, and descend safely."
    ),

    total_timesteps=500_000,
    checkpoint_freq=10_000,
    eval_freq=10_000,
    run_name_prefix="dynamic_landing",

    # Dynamic landing needs all controls.
    freeze_vz=False,
    altitude_hold_enabled=False,
    enable_forward_motion=True,
    enable_yaw_control=True,
    enable_z_control=True,

    desired_distance_proxy=0.96,
    distance_tolerance=0.08,
    min_target_distance_proxy=0.88,
    block_forward_when_too_close=True,

    reset_takeoff_altitude_m=5.0,
    altitude_hold_target_m=5.0,
    min_safe_altitude_m=1.3,
    min_termination_altitude_m=0.50,
    max_termination_altitude_m=12.0,

    max_episode_steps=850,
    focus_fail_sec=4.0,

    # Dynamic landing needs stronger center + visibility + smoothness constraints.
    w_center=24.0,
    w_distance=8.0,
    w_visibility=6.0,
    w_lost_target=14.0,
    w_altitude_safe=4.0,
    w_altitude_low_penalty=14.0,
    w_altitude_high_penalty=5.0,
    w_smooth_follow=3.0,
    w_control=0.07,
    w_action_delta=0.65,
    w_obstacle=7.0,
    w_safety_intervention=2.5,
    w_slow_or_stuck=0.7,

    penalty_collision=90.0,
    penalty_timeout=6.0,
)
