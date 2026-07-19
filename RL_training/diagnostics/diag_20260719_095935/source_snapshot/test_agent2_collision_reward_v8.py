"""Regression tests for Safe Training v8 authorized-descent reward evidence.

The physical controller is unchanged. These tests protect the v8 behavior:
- only an ACTUAL positive-NED descent command that passed the visual gate can
  create physical touchdown authorization;
- a short AirSim collision-report delay can use that authorization when the
  target is no longer visible;
- later LIVE drift, staleness, ground contact, or a wrong object rejects it.
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
        "agent2_landing_env_collision_v8_test",
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


def _bare_env(expected_object: str = "Porsche_BP_C_1"):
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._step = 100
    env._last_verified_alignment_step = -999999
    env._last_verified_alignment_center_error = 999.0
    env._last_verified_alignment_bbox_rel_error = 999.0
    env._last_verified_alignment_similarity = 0.0
    env._authorized_descent_latched = False
    env._last_authorized_descent_step = -999999
    env._last_authorized_descent_center_error = 999.0
    env._last_authorized_descent_bbox_rel_error = 999.0
    env._last_authorized_descent_similarity = 0.0
    env._last_authorized_descent_vz_mps = 0.0
    env._authorized_descent_invalidated_reason = "never_authorized"
    env._expected_collision_object_name = expected_object
    env._expected_collision_object_source = "test" if expected_object else "unlocked"
    return env


def _authorized_pre_info(**overrides):
    info = {
        "bottom_match_live": True,
        "bottom_match_confirmed": True,
        "bottom_center_error": 0.18,
        "bottom_bbox_rel_err": 0.30,
        "bottom_similarity": 0.80,
    }
    info.update(overrides)
    return info


def _no_live_info():
    return {
        "bottom_match_live": False,
        "bottom_match_confirmed": False,
        "bottom_center_error": 999.0,
        "bottom_bbox_rel_err": 999.0,
        "bottom_similarity": 0.0,
    }


def _record(env, vz: float = 0.25):
    env._record_authorized_descent(_authorized_pre_info(), True, vz)
    assert env._authorized_descent_latched is True
    assert env._last_authorized_descent_step == env._step


def test_v8_config_and_short_experiment_defaults():
    cfg = Agent2Config(show_camera=False)
    assert cfg.authorized_descent_max_age_steps == 20
    assert cfg.authorized_descent_min_vz_mps == 1.0e-4
    source = (ROOT / "config" / "flow_config.py").read_text(encoding="utf-8")
    assert "TOTAL_AGENT2_TIMESTEPS = 2_048" in source
    assert "CHECKPOINT_FREQUENCY = 2_048" in source
    assert "RESUME_AGENT_2 = False" in source


def test_actual_allowed_descent_creates_authorization():
    env = _bare_env()
    _record(env, 0.30)
    assert env._last_authorized_descent_vz_mps == 0.30
    assert env._authorized_descent_invalidated_reason == ""


def test_blocked_or_zero_descent_cannot_create_authorization():
    env = _bare_env()
    env._record_authorized_descent(_authorized_pre_info(), False, 0.30)
    assert env._authorized_descent_latched is False
    env._record_authorized_descent(_authorized_pre_info(), True, 0.0)
    assert env._authorized_descent_latched is False


def test_unconfirmed_or_bad_visual_evidence_cannot_create_authorization():
    env = _bare_env()
    env._record_authorized_descent(
        _authorized_pre_info(bottom_match_confirmed=False), True, 0.30
    )
    assert env._authorized_descent_latched is False
    env._record_authorized_descent(
        _authorized_pre_info(bottom_center_error=0.50), True, 0.30
    )
    assert env._authorized_descent_latched is False


def test_authorized_descent_accepts_delayed_target_contact_without_live_match():
    env = _bare_env()
    _record(env)
    env._step += 12
    env._update_authorized_descent_latch_after_observation(_no_live_info())
    decision = env._collision_reward_decision(_no_live_info(), "Porsche_BP_C_1")
    assert decision["success"] is True
    assert decision["success_path"] == "AUTHORIZED_DESCENT"
    assert decision["authorized_descent_latch_age"] == 12


def test_age_twenty_is_accepted_and_twenty_one_is_rejected():
    env = _bare_env()
    _record(env)
    env._step += 20
    env._update_authorized_descent_latch_after_observation(_no_live_info())
    decision = env._collision_reward_decision(_no_live_info(), "Porsche_BP_C_1")
    assert decision["success"] is True

    env._step += 1
    env._update_authorized_descent_latch_after_observation(_no_live_info())
    decision = env._collision_reward_decision(_no_live_info(), "Porsche_BP_C_1")
    assert decision["success"] is False
    assert decision["success_path"] == "NONE"
    assert decision["authorized_descent_invalidated_reason"] == "authorized_descent_stale"


def test_later_live_drift_invalidates_authorization():
    env = _bare_env()
    _record(env)
    env._step += 2
    env._update_authorized_descent_latch_after_observation(
        _authorized_pre_info(bottom_center_error=0.70)
    )
    assert env._authorized_descent_latched is False
    assert env._authorized_descent_invalidated_reason == "live_drift_after_authorized_descent"
    decision = env._collision_reward_decision(_no_live_info(), "Porsche_BP_C_1")
    assert decision["success"] is False


def test_visible_bad_contact_cannot_fall_back_to_authorized_descent():
    env = _bare_env()
    _record(env)
    env._step += 2
    bad_live = _authorized_pre_info(bottom_center_error=0.70)
    decision = env._collision_reward_decision(bad_live, "Porsche_BP_C_1")
    assert decision["success"] is False
    assert decision["success_path"] == "NONE"


def test_ground_and_wrong_object_are_rejected():
    env = _bare_env()
    _record(env)
    env._step += 8
    ground = env._collision_reward_decision(_no_live_info(), "Floor_0")
    assert ground["success"] is False
    assert ground["success_path"] == "AUTHORIZED_DESCENT"
    assert ground["reject_reason"] == "ground_or_terrain_collision"

    wrong = env._collision_reward_decision(_no_live_info(), "OtherCar_C_1")
    assert wrong["success"] is False
    assert wrong["success_path"] == "AUTHORIZED_DESCENT"
    assert wrong["reject_reason"] == "collision_object_mismatch"


def test_first_authorized_target_contact_can_lock_collision_object():
    env = _bare_env(expected_object="")
    _record(env)
    env._step += 8
    decision = env._collision_reward_decision(_no_live_info(), "Porsche_BP_C_1")
    assert decision["success"] is True
    assert decision["success_path"] == "AUTHORIZED_DESCENT"
    assert decision["collision_object_lock_created"] is True
    assert env._expected_collision_object_name == "Porsche_BP_C_1"


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
