"""Regression tests for Agent-2 vertical direction and NED-Z consistency.

No Unreal/AirSim connection is required. The tests prove that:
1. Agent 2 can never send a negative NED vz (climb).
2. Positive policy Z produces positive NED vz (descent) only when aligned.
3. Height above target is computed from raw API NED coordinates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load_module():
    gym = ModuleType("gymnasium")
    gym.Env = object
    spaces = ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    spaces.Box = Box
    gym.spaces = spaces
    sys.modules["gymnasium"] = gym
    sys.modules["gymnasium.spaces"] = spaces

    airsim = ModuleType("cosysairsim")
    airsim.MultirotorClient = object
    airsim.YawMode = lambda **kwargs: kwargs
    airsim.ImageRequest = lambda *args, **kwargs: (args, kwargs)
    airsim.ImageType = SimpleNamespace(Scene=0)
    airsim.to_eularian_angles = lambda _q: (0.0, 0.0, 0.0)
    sys.modules["cosysairsim"] = airsim

    lidar = ModuleType("lidar_processor")
    lidar.LidarProcessor = object
    lidar.LidarProcessorConfig = object
    lidar.point_cloud_to_array = lambda value: np.asarray(value)
    sys.modules["lidar_processor"] = lidar

    observation = ModuleType("observation_builder")
    observation.BBox = type("BBox", (), {})
    observation.DroneState = lambda **kwargs: SimpleNamespace(**kwargs)
    observation.ObstacleState = type("ObstacleState", (), {})
    observation.ObservationBuilder = object
    observation.ObservationBuilderConfig = object
    sys.modules["observation_builder"] = observation

    tracker_module = ModuleType("resnet_yolo_tracker")
    tracker_module.YoloResNetTracker = object
    sys.modules["resnet_yolo_tracker"] = tracker_module

    spec = importlib.util.spec_from_file_location(
        "agent2_landing_env_vertical_v4_test",
        ROOT / "agent2_landing_env.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
Agent2Config = MODULE.Agent2Config
Agent2LandingEnv = MODULE.Agent2LandingEnv


def _base_info(*, aligned: bool) -> dict:
    return {
        "bottom_match_live": aligned,
        "bottom_match_confirmed": aligned,
        "bottom_live_match_streak": 3 if aligned else 0,
        "bottom_similarity": 0.90 if aligned else 0.0,
        "bottom_center_error": 0.10 if aligned else 999.0,
        "bottom_bbox_rel_err": 0.20 if aligned else 999.0,
        "front_dist_m": 20.0,
        "back_dist_m": 20.0,
        "left_dist_m": 20.0,
        "right_dist_m": 20.0,
    }


def _step_env(*, aligned: bool):
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._step = 0
    env._last_info = _base_info(aligned=aligned)
    env._episode_descent_requested_steps = 0
    env._episode_descent_allowed_steps = 0
    env._episode_descent_blocked_steps = 0
    env._episode_climb_command_blocked_steps = 0
    env._episode_live_match_steps = 0
    env._episode_predicted_steps = 0
    env._episode_no_target_steps = 0
    env._episode_best_center_error = float("inf")
    env._episode_best_similarity = 0.0
    env._prev_relative_height_m = None
    env._prev_action = np.zeros(4, dtype=np.float32)
    env._reward_bank = 0.0
    env._episode_return = 0.0
    env._lost_steps = 0
    env._target_surface_z_ned = 0.0
    env._target_surface_altitude_m = 0.0
    env._target_surface_source = "test"
    env._collision_timestamp_at_reset = 0
    env._last_vertical_control_state = "HOLD_INIT"
    env._last_descent_block_reason = ""
    env._last_raw_vz_action = 0.0
    env._last_requested_vz_mps = 0.0
    env._last_applied_vz_mps = 0.0
    env._last_climb_command_blocked = False
    env._control_img_vel_x = 0.0
    env._control_img_vel_y = 0.0
    env._episode_xy_hold_steps = 0
    env._alignment_ready_streak = env.cfg.alignment_streak_required if aligned else 0
    env._descent_alignment_latched = bool(aligned)
    env._episode_recenter_steps = 0
    env._prev_center_error_for_reward = None
    env._episode_dense_reward = 0.0

    sent = {}

    class Future:
        @staticmethod
        def join():
            return None

    class Client:
        @staticmethod
        def moveByVelocityBodyFrameAsync(**kwargs):
            sent.update(kwargs)
            return Future()

    env.client = Client()
    env._observe = lambda: (
        np.zeros(37, dtype=np.float32),
        {
            "bottom_center_error": 0.10 if aligned else 999.0,
            "bottom_bbox_rel_err": 0.20 if aligned else 999.0,
            "relative_height_to_target_m": 5.0,
            "bottom_match_live": aligned,
            "bottom_match_recent": aligned,
            "alt_agl_m": 5.0,
            "drone_z_ned": -5.0,
        },
    )
    env._new_collision = lambda: (False, "", 0)
    return env, sent


def test_negative_policy_z_can_never_command_climb():
    env, sent = _step_env(aligned=True)
    _obs, _reward, _done, _truncated, info = env.step(
        np.asarray([0.0, 0.0, -1.0, 0.0], dtype=np.float32)
    )
    assert sent["vz"] == 0.0, sent
    assert info["climb_command_blocked"] is True
    assert info["requested_vz_mps"] == 0.0
    assert env._episode_climb_command_blocked_steps == 1


def test_positive_policy_z_descends_when_target_is_aligned():
    env, sent = _step_env(aligned=True)
    _obs, _reward, _done, _truncated, info = env.step(
        np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
    )
    expected = 0.5 * env.cfg.vz_scale_mps
    assert abs(sent["vz"] - expected) < 1e-9, sent
    assert sent["vz"] > 0.0
    assert info["descent_allowed"] is True


def test_positive_policy_z_hovers_when_target_is_not_aligned():
    env, sent = _step_env(aligned=False)
    _obs, _reward, _done, _truncated, info = env.step(
        np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    )
    assert sent["vz"] == 0.0, sent
    assert info["descent_blocked"] is True


def test_relative_height_uses_raw_ned_difference():
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config()
    env._target_surface_z_ned = 1.72

    kinematics = SimpleNamespace(
        position=SimpleNamespace(z_val=-4.88),
        linear_velocity=SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0),
        angular_velocity=SimpleNamespace(z_val=0.0),
        orientation=SimpleNamespace(),
    )
    env.client = SimpleNamespace(
        getMultirotorState=lambda **_kwargs: SimpleNamespace(
            kinematics_estimated=kinematics
        )
    )

    _drone_state, relative_height, _state = env._get_api_state()
    assert abs(relative_height - 6.60) < 1e-6, relative_height
    assert relative_height > 0.0


def test_actor_pose_z_is_not_converted_with_abs():
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config()
    env._target_actor_name = "TargetActor"
    env.client = SimpleNamespace(
        simGetObjectPose=lambda _actor: SimpleNamespace(
            position=SimpleNamespace(z_val=1.72)
        )
    )
    env._read_target_surface_altitude()
    assert env._target_surface_z_ned == 1.72
    assert env._target_surface_source == "api_object_pose_z_ned"


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} tests")
