"""Focused regression tests for final-metre controller and dense Z reward."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    for name in ("BBox", "DroneState", "ObstacleState", "ObservationBuilder", "ObservationBuilderConfig"):
        setattr(observation, name, object)
    sys.modules["observation_builder"] = observation

    tracker = ModuleType("resnet_yolo_tracker")
    tracker.YoloResNetTracker = object
    sys.modules["resnet_yolo_tracker"] = tracker

    spec = importlib.util.spec_from_file_location(
        "agent2_progress_reward_test", ROOT / "agent2_landing_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
Agent2Config = MODULE.Agent2Config
Agent2LandingEnv = MODULE.Agent2LandingEnv


def _env(prev_height=1.2):
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._descent_alignment_latched = True
    env._prev_relative_height_m = prev_height
    return env


def _info(height=0.8, center=0.08, live=True, confirmed=True):
    return {
        "bottom_match_live": live,
        "bottom_match_confirmed": confirmed,
        "bottom_center_error": center,
        "relative_height_to_target_m": height,
    }


def test_final_metre_descent_cap_is_decisive_but_bounded():
    env = _env()
    assert env._catchup_descent_speed_limit(_info(height=2.0)) == 0.40
    assert env._catchup_descent_speed_limit(_info(height=0.8)) == 0.32
    vz, limit, soft = env._bounded_descent_command(
        0.60, "DESCEND_SOFT_CATCHUP_LANDING_LOCK", True, _info(height=0.8)
    )
    assert soft is True
    assert abs(vz - 0.32) < 1.0e-9
    assert abs(limit - 0.32) < 1.0e-9


def test_real_aligned_height_progress_receives_immediate_positive_reward():
    env = _env(prev_height=1.2)
    reward, parts = env._dense_landing_reward(
        _info(height=0.8),
        raw_vz_action=0.8,
        descent_allowed=True,
        applied_vz_mps=0.32,
    )
    assert parts["aligned_descent_progress"] > 0.0
    assert parts["landing_lock_time"] < 0.0
    assert parts["hesitation"] == 0.0
    assert reward > 0.0


def test_aligned_hover_after_lock_is_negative():
    env = _env(prev_height=0.8)
    reward, parts = env._dense_landing_reward(
        _info(height=0.8),
        raw_vz_action=-0.8,
        descent_allowed=True,
        applied_vz_mps=0.0,
    )
    assert parts["aligned_descent_progress"] == 0.0
    assert parts["landing_lock_time"] < 0.0
    assert parts["hesitation"] < 0.0
    assert reward < 0.0


def test_safety_blocked_step_does_not_get_hesitation_penalty():
    env = _env(prev_height=0.8)
    reward, parts = env._dense_landing_reward(
        _info(height=0.8),
        raw_vz_action=-0.8,
        descent_allowed=False,
        applied_vz_mps=0.0,
    )
    assert parts["hesitation"] == 0.0
    assert reward == parts["landing_lock_time"]


def test_pose_drop_without_applied_descent_is_not_rewarded():
    env = _env(prev_height=1.2)
    reward, parts = env._dense_landing_reward(
        _info(height=0.6),
        raw_vz_action=0.8,
        descent_allowed=True,
        applied_vz_mps=0.0,
    )
    assert parts["aligned_descent_progress"] == 0.0
    assert reward < 0.0


def test_unsafe_descent_request_is_penalized_but_not_executed_by_reward():
    env = _env(prev_height=1.2)
    env._descent_alignment_latched = False
    reward, parts = env._dense_landing_reward(
        _info(height=1.2, center=0.70),
        raw_vz_action=0.9,
        descent_allowed=False,
        applied_vz_mps=0.0,
    )
    assert parts["unsafe_descent"] < 0.0
    assert reward < 0.0


def test_controller_config_keeps_more_xy_authority_near_touch():
    from config import flow_config as flow

    assert flow.PARALLEL_LANDING_BRIDGE_MAX_SPEED_MPS == 0.55
    assert flow.PARALLEL_LANDING_CATCHUP_BRIDGE_MAX_SPEED_MPS == 0.75
    assert flow.PARALLEL_NEAR_GROUND_DESCENT_MAX_MPS == 0.32
    source = (ROOT / "agent1p2_env.py").read_text(encoding="utf-8")
    assert "agent1_cfg.command_bridge_landing_max_speed_mps" in source
    assert "agent1_cfg.parallel_vertical_near_ground_max_mps" in source


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} reward/controller progress tests")
