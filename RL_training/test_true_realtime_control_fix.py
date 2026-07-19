"""Isolated checks for the safe post-pulse bridge and shared bottom perception."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent


def _install_stubs() -> None:
    gym = types.ModuleType("gymnasium")
    gym.Env = object
    spaces = types.ModuleType("gymnasium.spaces")
    spaces.Box = object
    gym.spaces = spaces
    sys.modules["gymnasium"] = gym
    sys.modules["gymnasium.spaces"] = spaces

    airsim = types.ModuleType("cosysairsim")

    class YawMode:
        def __init__(self, is_rate=True, yaw_or_rate=0.0):
            self.is_rate = is_rate
            self.yaw_or_rate = yaw_or_rate

    airsim.YawMode = YawMode
    airsim.to_eularian_angles = lambda orientation: tuple(orientation)
    sys.modules["cosysairsim"] = airsim

    object_tracker = types.ModuleType("object_tracker")
    object_tracker.tracker = object
    sys.modules["object_tracker"] = object_tracker

    tracking = types.ModuleType("tracking")
    manager = types.ModuleType("tracking.target_tracker_manager")
    manager.TargetTrackerManager = object
    sys.modules["tracking"] = tracking
    sys.modules["tracking.target_tracker_manager"] = manager

    weights = types.ModuleType("weights_config")
    weights.EnvConfig = object
    sys.modules["weights_config"] = weights

    observation = types.ModuleType("observation_builder")
    for name in ("ObservationBuilder", "ObservationBuilderConfig", "BBox", "DroneState", "ObstacleState"):
        setattr(observation, name, object)
    sys.modules["observation_builder"] = observation

    safety = types.ModuleType("safety_filter")
    safety.SafetyConfig = object
    safety.safety_filter = lambda **kwargs: (kwargs.get("raw_action"), {})
    sys.modules["safety_filter"] = safety

    lidar = types.ModuleType("lidar_processor")
    lidar.LidarProcessor = object
    lidar.LidarProcessorConfig = object
    lidar.point_cloud_to_array = lambda value: value
    sys.modules["lidar_processor"] = lidar

    reward = types.ModuleType("follow_reward_v37")
    reward.FollowRewardConfig = object
    reward.compute_follow_reward = lambda **kwargs: (0.0, {})
    sys.modules["follow_reward_v37"] = reward


def _load_drone_env_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("drone_env_truefix_test", ROOT / "drone_env.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Future:
    def join(self):
        return self


class _Client:
    def __init__(self, body_vx=0.42, body_vy=0.08, pitch_deg=0.0, roll_deg=0.0):
        self.commands = []
        self.state = SimpleNamespace(
            kinematics_estimated=SimpleNamespace(
                orientation=(np.radians(pitch_deg), np.radians(roll_deg), 0.0),
                linear_velocity=SimpleNamespace(x_val=body_vx, y_val=body_vy),
            )
        )

    def getMultirotorState(self, vehicle_name=""):
        return self.state

    def moveByVelocityBodyFrameAsync(self, **kwargs):
        self.commands.append(dict(kwargs))
        return _Future()


def _cfg():
    return SimpleNamespace(
        vehicle_name="Drone1",
        cmd_duration_s=0.10,
        command_bridge_enabled=True,
        command_bridge_period_initial_s=0.55,
        command_bridge_period_ema_alpha=0.25,
        command_bridge_duration_scale=1.15,
        command_bridge_min_duration_s=0.25,
        command_bridge_max_duration_s=0.90,
        command_bridge_match_max_speed_mps=1.20,
        command_bridge_pred_max_speed_mps=0.85,
        command_bridge_lost_max_speed_mps=0.45,
        command_bridge_bottom_max_speed_mps=0.70,
        command_bridge_landing_max_speed_mps=0.38,
        command_bridge_landing_catchup_max_speed_mps=0.70,
        command_bridge_velocity_ema_alpha=0.65,
        command_bridge_min_speed_mps=0.03,
        command_bridge_attitude_soft_limit_deg=10.0,
        command_bridge_attitude_hard_limit_deg=16.0,
        command_bridge_zero_vertical_velocity=True,
        command_bridge_parallel_vertical_enabled=True,
        command_bridge_parallel_vertical_max_mps=0.40,
        command_bridge_parallel_vertical_min_mps=1.0e-4,
        parallel_vertical_near_ground_max_mps=0.18,
        command_bridge_zero_yaw_rate=True,
        parallel_share_agent2_bottom_perception=True,
        parallel_bottom_snapshot_stale_after_s=4.00,
        parallel_xy_slew_enabled=True,
        parallel_xy_max_delta_per_step_mps=0.60,
        parallel_bottom_live_speed_cap_mps=1.60,
        parallel_bottom_pred_speed_cap_mps=1.10,
    )


def main() -> None:
    module = _load_drone_env_module()
    env = module.DroneEnv.__new__(module.DroneEnv)
    env.cfg = _cfg()
    env.client = _Client()
    env._speed_stage = "CHASE_FAST"
    env._last_commanded_vz_mps = 0.6
    env._last_commanded_yaw_rate_dps = 90.0
    env._reset_command_bridge_state()

    safe_vz, blocked, limited = env._apply_parallel_vertical_safety(
        0.80, ["landing_down_proximity_limit_descent"], hard_safety=False
    )
    assert abs(safe_vz - 0.18) < 1.0e-9 and not blocked and limited
    safe_vz, blocked, limited = env._apply_parallel_vertical_safety(
        0.80, ["down_obstacle_warning_block_descent"], hard_safety=False
    )
    assert safe_vz == 0.0 and blocked and not limited
    safe_vz, blocked, limited = env._apply_parallel_vertical_safety(
        -0.45, ["down_obstacle_warning_block_descent"], hard_safety=False
    )
    assert safe_vz == -0.45 and not blocked and not limited
    safe_vz, blocked, limited = env._apply_parallel_vertical_safety(
        -0.45, ["collision_emergency"], hard_safety=True
    )
    assert safe_vz == 0.0 and blocked and not limited
    print("PASS parallel Z ownership cannot bypass down-LiDAR or hard safety")

    env._issue_command_bridge(
        requested_vx_mps=5.80,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=2.0,
    )
    command = env.client.commands[-1]
    assert command["vx"] < 1.20, command
    assert abs(command["vx"]) <= 0.42 + 1.0e-6, command
    assert abs(command["vy"]) <= 0.08 + 1.0e-6, command
    assert command["duration"] <= 0.90, command
    assert command["vz"] == 0.0 and command["yaw_mode"].yaw_or_rate == 0.0
    assert env._command_bridge_last_reason == "ACHIEVED_VELOCITY_HOLD"
    print("PASS bridge holds achieved velocity, never the raw 5.8 m/s pulse")

    env._speed_stage = "BOTTOM_LANDING_READY"
    env._parallel_bottom_measurement_live = True
    env._parallel_bottom_predictive_catchup = False
    env.client.state.kinematics_estimated.linear_velocity.x_val = 0.9
    env._issue_command_bridge(
        requested_vx_mps=5.80,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=1.0,
    )
    command = env.client.commands[-1]
    assert abs(command["vx"]) <= 0.38 + 1.0e-6, command
    print("PASS aligned landing-stage bridge keeps the strict 0.38 m/s XY cap")

    env._parallel_bottom_predictive_catchup = True
    env.client = _Client(body_vx=0.9, body_vy=0.0)
    env._issue_command_bridge(
        requested_vx_mps=1.20,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=1.0,
    )
    command = env.client.commands[-1]
    assert 0.38 < abs(command["vx"]) <= 0.70 + 1.0e-6, command
    print("PASS landing-ready LIVE catch-up temporarily restores the 0.70 m/s bridge cap")

    env.client = _Client(body_vx=0.5, pitch_deg=18.0)
    command_count_before = len(env.client.commands)
    env._issue_command_bridge(
        requested_vx_mps=5.80,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=1.0,
    )
    assert len(env.client.commands) == command_count_before
    assert not env._command_bridge_active
    assert env._command_bridge_last_reason == "ATTITUDE_GUARD"
    print("PASS hard pitch/roll guard suppresses the bridge without a braking pulse")


    env._parallel_dual_agent_mode = True
    env._parallel_external_vz_mps = 0.80
    env.client = _Client(body_vx=0.0, body_vy=0.0)
    env._issue_command_bridge(
        requested_vx_mps=0.0,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=2.0,
    )
    command = env.client.commands[-1]
    assert command["vx"] == 0.0 and command["vy"] == 0.0, command
    assert abs(command["vz"] - 0.40) < 1.0e-9, command
    assert env._command_bridge_last_reason == "LANDING_Z_HOLD"
    print("PASS parallel landing Z is refreshed independently and capped at 0.40 m/s")

    env._parallel_external_vz_mps = 0.80
    env._last_parallel_z_override_mps = 0.18
    env.client = _Client(body_vx=0.0, body_vy=0.0)
    env._issue_command_bridge(
        requested_vx_mps=0.0,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": ["landing_down_proximity_limit_descent"]},
        control_period_s=2.0,
    )
    command = env.client.commands[-1]
    assert abs(command["vz"] - 0.18) < 1.0e-9, command
    print("PASS bridge follows the safety-limited Z actually sent, not raw Agent-2 Z")

    env._parallel_external_vz_mps = -0.45
    env._last_parallel_z_override_mps = -0.45
    env.client = _Client(body_vx=0.0, body_vy=0.0)
    command_count_before = len(env.client.commands)
    env._issue_command_bridge(
        requested_vx_mps=0.0,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=2.0,
    )
    assert len(env.client.commands) == command_count_before
    assert env._command_bridge_last_vz_mps == 0.0
    assert env._command_bridge_last_reason == "NO_ACHIEVED_XY_OR_LANDING_Z"
    print("PASS negative NED-Z reacquire climb is never persisted by the bridge")

    env._parallel_external_vz_mps = 0.28
    env._last_parallel_z_override_mps = 0.28
    env.client = _Client(body_vx=0.0, body_vy=0.0)
    env._issue_command_bridge(
        requested_vx_mps=0.0,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": ["collision_detected"]},
        control_period_s=2.0,
    )
    command = env.client.commands[-1]
    assert command["vx"] == 0.0 and command["vy"] == 0.0 and command["vz"] == 0.0
    assert env._command_bridge_last_reason == "HARD_SAFETY"
    assert not env._command_bridge_active
    print("PASS hard safety cancels the vertical bridge with an explicit zero command")

    env.client = _Client(body_vx=0.4, pitch_deg=18.0)
    command_count_before = len(env.client.commands)
    env._issue_command_bridge(
        requested_vx_mps=1.0,
        requested_vy_mps=0.0,
        tracking_mode="MATCH",
        safety_info={"safety_reasons": []},
        control_period_s=1.0,
    )
    assert len(env.client.commands) == command_count_before
    assert env._command_bridge_last_vz_mps == 0.0
    assert env._command_bridge_last_reason == "ATTITUDE_GUARD"
    print("PASS hard attitude guard suppresses both XY and landing-Z bridge")

    env._parallel_dual_agent_mode = True
    env.step_in_episode = 10
    env._bottom_match_streak = 0
    shared_frame = np.full((32, 48, 3), 77, dtype=np.uint8)
    env.set_parallel_bottom_perception_snapshot(
        {
            "match": True,
            "mode": "MATCH",
            "bbox_xyxy": [100, 100, 200, 200],
            "frame_bgr": shared_frame,
            "similarity": 0.82,
            "err_x": 0.04,
            "err_y": -0.06,
            "bbox_area_norm": 0.08,
            "bbox_rel_err": 0.25,
            "live_match_streak": 3,
        }
    )
    assert env._apply_parallel_bottom_perception_snapshot()
    assert env._agent1_bottom_perception_shared
    assert env._bottom_match and env._bottom_match_streak >= 3
    assert np.allclose(env._bottom_bbox_xyxy, [100, 100, 200, 200])
    assert np.array_equal(env._cached_downward_frame, shared_frame)
    assert env._cached_downward_frame is not shared_frame
    print("PASS Agent 1 consumes Agent-2 bbox and its exact source frame")

    env._parallel_fused_prev_vx_mps = 0.0
    env._parallel_fused_prev_vy_mps = 0.0
    vx1, vy1 = env._stabilize_parallel_xy_command(
        3.0, 4.0, bottom_live=True, bottom_guidance_active=True
    )
    assert np.hypot(vx1, vy1) <= 0.60 + 1.0e-9
    vx2, vy2 = env._stabilize_parallel_xy_command(
        -3.0, -4.0, bottom_live=True, bottom_guidance_active=True
    )
    assert np.hypot(vx2 - vx1, vy2 - vy1) <= 0.60 + 1.0e-9
    assert np.hypot(vx2, vy2) <= 1.60 + 1.0e-9
    raw_vx, raw_vy = env._stabilize_parallel_xy_command(
        2.5, -1.5, bottom_live=False, bottom_guidance_active=False
    )
    assert raw_vx == 2.5 and raw_vy == -1.5
    print("PASS bottom-guided XY is speed-capped and slew-limited; search is unchanged")

    source = (ROOT / "drone_env.py").read_text(encoding="utf-8")
    pulse_index = source.index("self.client.moveByVelocityBodyFrameAsync(\n            vx=vx_cmd")
    bridge_index = source.index("self._issue_command_bridge(", pulse_index)
    assert pulse_index < bridge_index
    assert "duration=float(self.cfg.cmd_duration_s)" in source[pulse_index:bridge_index]
    assert ").join()" in source[pulse_index:bridge_index]
    print("PASS original 0.10-second PPO pulse executes before the bounded bridge")

    print("ALL TRUE REAL-TIME CONTROL FIX TESTS PASSED")


if __name__ == "__main__":
    main()
