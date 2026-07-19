"""Regression tests for Agent-2 PD + residual horizontal control.

These tests are runtime-light and do not connect to Unreal or AirSim.
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
        "agent2_landing_env_horizontal_v5_test",
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


def _bare_env():
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._control_img_vel_x = 0.0
    env._control_img_vel_y = 0.0
    env._episode_xy_hold_steps = 0
    env._alignment_ready_streak = 0
    env._descent_alignment_latched = False
    env._episode_recenter_steps = 0
    return env


def _info(**overrides):
    info = {
        "bottom_match_live": True,
        "bottom_match_confirmed": True,
        "bottom_live_match_streak": 10,
        "bottom_similarity": 0.90,
        "bottom_center_error": 0.10,
        "bottom_bbox_rel_err": 0.20,
        "bottom_bbox_area_norm": 0.03,
        "bottom_err_x": 0.0,
        "bottom_err_y": 0.0,
        "bottom_img_vel_x_control": 0.0,
        "bottom_img_vel_y_control": 0.0,
        "relative_height_to_target_m": 5.0,
    }
    info.update(overrides)
    return info


def test_no_live_match_holds_xy_even_with_full_policy_action():
    env = _bare_env()
    vx, vy, details = env._horizontal_visual_servo(
        np.asarray([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
        _info(bottom_match_live=False, bottom_match_confirmed=False),
    )
    assert vx == 0.0
    assert vy == 0.0
    assert details["state"] == "HOLD_NO_LIVE_MATCH"


def test_pd_axis_mapping_matches_agent1_controller():
    env = _bare_env()
    # Target below image center -> move body-X backward (negative vx).
    vx, _vy, details = env._horizontal_visual_servo(
        np.zeros(4, dtype=np.float32),
        _info(bottom_err_y=0.30, bottom_center_error=0.30),
    )
    assert details["pd_ax"] < 0.0
    assert vx < 0.0

    # Target right of image center -> move body-Y right (positive vy).
    _vx, vy, details = env._horizontal_visual_servo(
        np.zeros(4, dtype=np.float32),
        _info(bottom_err_x=0.30, bottom_center_error=0.30),
    )
    assert details["pd_ay"] > 0.0
    assert vy > 0.0


def test_policy_is_only_a_bounded_residual():
    env = _bare_env()
    _vx, _vy, details = env._horizontal_visual_servo(
        np.asarray([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
        _info(),
    )
    limit = env.cfg.horizontal_ppo_residual_max_action
    assert abs(details["residual_ax"] - limit) < 1e-9
    assert abs(details["residual_ay"] + limit) < 1e-9
    assert abs(details["final_ax"]) <= env.cfg.horizontal_pd_max_action + limit + 1e-9
    assert abs(details["final_ay"]) <= env.cfg.horizontal_pd_max_action + limit + 1e-9


def test_unconfirmed_match_uses_pd_without_policy_residual():
    env = _bare_env()
    _vx, _vy, details = env._horizontal_visual_servo(
        np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        _info(bottom_match_confirmed=False, bottom_err_x=0.2),
    )
    assert details["state"] == "PREDICTIVE_BOTTOM_MATCH_PENDING"
    assert details["residual_ax"] == 0.0
    assert details["residual_ay"] == 0.0
    assert details["pd_ay"] > 0.0


def test_speed_limit_shrinks_near_touchdown_and_large_bbox():
    env = _bare_env()
    far = env._horizontal_speed_limit(
        _info(relative_height_to_target_m=8.0, bottom_bbox_area_norm=0.01)
    )
    near = env._horizontal_speed_limit(
        _info(relative_height_to_target_m=0.5, bottom_bbox_area_norm=0.25)
    )
    assert far == env.cfg.horizontal_speed_far_mps
    assert near == env.cfg.horizontal_speed_touchdown_mps
    assert near < far


def test_zero_height_is_not_mistaken_for_infinity():
    env = _bare_env()
    limit = env._horizontal_speed_limit(
        _info(relative_height_to_target_m=0.0, bottom_bbox_area_norm=0.01)
    )
    assert limit == env.cfg.horizontal_speed_touchdown_mps


def test_descent_requires_stable_alignment_then_recenters_on_drift():
    env = _bare_env()
    aligned = _info(
        bottom_center_error=0.10,
        bottom_bbox_rel_err=0.20,
    )
    required = env.cfg.alignment_streak_required
    for _ in range(required - 1):
        state, allowed, _ = env._vertical_control_state(aligned)
        assert allowed is False
        assert state == "ALIGN_LOCK_PENDING"

    state, allowed, reason = env._vertical_control_state(aligned)
    assert state == "DESCEND_LANDING_LOCK_ACQUIRED"
    assert allowed is True
    assert reason == ""
    assert env._descent_alignment_latched is True

    drifted = _info(
        bottom_center_error=0.50,
        bottom_bbox_rel_err=0.20,
    )
    for attempt in range(env.cfg.landing_lock_bad_live_release_steps):
        state, allowed, reason = env._vertical_control_state(drifted)
        assert allowed is False
        if attempt < env.cfg.landing_lock_bad_live_release_steps - 1:
            assert state == "RECENTER_LANDING_LOCK_HELD"
            assert env._descent_alignment_latched is True
        else:
            assert state == "RECENTER_LANDING_LOCK_RELEASED"
            assert "sustained_live_misalignment" in reason
            assert env._descent_alignment_latched is False


def test_perfect_zero_alignment_is_valid_numeric_input():
    env = _bare_env()
    aligned = _info(bottom_center_error=0.0, bottom_bbox_rel_err=0.0)
    for _ in range(env.cfg.alignment_streak_required):
        state, allowed, _ = env._vertical_control_state(aligned)
    assert state == "DESCEND_LANDING_LOCK_ACQUIRED"
    assert allowed is True


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
