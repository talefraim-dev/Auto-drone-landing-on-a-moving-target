"""Runtime-light regression tests for the v6 touchdown reward classifier.

No Unreal/AirSim connection is required. These tests protect both sides of the
reward decision:
- a brief contact-frame visual loss may use a strict recent alignment latch;
- stale alignment, visible drift, ground contact, or the wrong collision object
  can never receive the success reward.
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
        "agent2_landing_env_collision_v6_test",
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
    env._step = 100
    env._last_verified_alignment_step = -999999
    env._last_verified_alignment_center_error = 999.0
    env._last_verified_alignment_bbox_rel_error = 999.0
    env._last_verified_alignment_similarity = 0.0
    env._expected_collision_object_name = ""
    env._expected_collision_object_source = "unlocked"
    return env


def _info(**overrides):
    info = {
        "bottom_match_live": True,
        "bottom_match_confirmed": True,
        "bottom_center_error": 0.10,
        "bottom_bbox_rel_err": 0.20,
        "bottom_similarity": 0.90,
    }
    info.update(overrides)
    return info


def _store_strict_alignment(env):
    env._update_verified_alignment_latch(_info())
    assert env._last_verified_alignment_step == env._step


def test_strict_live_alignment_updates_latch():
    env = _bare_env()
    _store_strict_alignment(env)
    assert env._last_verified_alignment_center_error == 0.10
    assert env._last_verified_alignment_bbox_rel_error == 0.20
    assert env._last_verified_alignment_similarity == 0.90


def test_unconfirmed_or_loose_alignment_does_not_update_latch():
    env = _bare_env()
    env._update_verified_alignment_latch(_info(bottom_match_confirmed=False))
    assert env._last_verified_alignment_step < 0
    env._update_verified_alignment_latch(_info(bottom_center_error=0.21))
    assert env._last_verified_alignment_step < 0
    env._update_verified_alignment_latch(_info(bottom_bbox_rel_err=0.36))
    assert env._last_verified_alignment_step < 0


def test_live_aligned_touchdown_autolocks_target_collision_object():
    env = _bare_env()
    decision = env._collision_reward_decision(_info(), "Porsche_BP_C_1")
    assert decision["success"] is True
    assert decision["success_path"] == "LIVE_MATCH"
    assert decision["collision_object_matches_target"] is True
    assert decision["collision_object_lock_created"] is True
    assert env._expected_collision_object_name == "Porsche_BP_C_1"



def test_first_object_lock_requires_strict_alignment():
    env = _bare_env()
    decision = env._collision_reward_decision(
        _info(bottom_center_error=0.40, bottom_bbox_rel_err=0.60),
        "Porsche_BP_C_1",
    )
    assert decision["success_path"] == "LIVE_MATCH"
    assert decision["collision_object_lock_eligible"] is False
    assert decision["success"] is False
    assert env._expected_collision_object_name == ""

def test_recent_strict_alignment_handles_contact_frame_detection_loss():
    env = _bare_env()
    env._expected_collision_object_name = "Porsche_BP_C_1"
    env._expected_collision_object_source = "test"
    _store_strict_alignment(env)
    env._step += 2
    decision = env._collision_reward_decision(
        _info(
            bottom_match_live=False,
            bottom_match_confirmed=False,
            bottom_center_error=999.0,
            bottom_bbox_rel_err=999.0,
            bottom_similarity=0.0,
        ),
        "Porsche_BP_C_1",
    )
    assert decision["success"] is True
    assert decision["success_path"] == "RECENT_ALIGNMENT"
    assert decision["alignment_latch_age"] == 2


def test_stale_alignment_cannot_receive_success():
    env = _bare_env()
    env._expected_collision_object_name = "Porsche_BP_C_1"
    _store_strict_alignment(env)
    env._step += env.cfg.collision_latch_max_age_steps + 1
    decision = env._collision_reward_decision(
        _info(bottom_match_live=False, bottom_match_confirmed=False),
        "Porsche_BP_C_1",
    )
    assert decision["success"] is False
    assert decision["success_path"] == "NONE"
    assert decision["reject_reason"] == "alignment_not_verified"


def test_visible_current_drift_cannot_fall_back_to_old_latch():
    env = _bare_env()
    env._expected_collision_object_name = "Porsche_BP_C_1"
    _store_strict_alignment(env)
    env._step += 1
    decision = env._collision_reward_decision(
        _info(bottom_match_live=True, bottom_center_error=0.70),
        "Porsche_BP_C_1",
    )
    assert decision["success"] is False
    assert decision["success_path"] == "NONE"


def test_floor_collision_is_never_a_success_even_with_recent_alignment():
    env = _bare_env()
    _store_strict_alignment(env)
    env._step += 1
    decision = env._collision_reward_decision(
        _info(bottom_match_live=False, bottom_match_confirmed=False),
        "Floor_0",
    )
    assert decision["success"] is False
    assert decision["success_path"] == "RECENT_ALIGNMENT"
    assert decision["collision_ground_contact"] is True
    assert decision["reject_reason"] == "ground_or_terrain_collision"
    assert env._expected_collision_object_name == ""


def test_locked_target_rejects_different_collision_object():
    env = _bare_env()
    env._expected_collision_object_name = "Porsche_BP_C_1"
    decision = env._collision_reward_decision(_info(), "OtherCar_C_1")
    assert decision["success"] is False
    assert decision["success_path"] == "LIVE_MATCH"
    assert decision["collision_object_matches_target"] is False
    assert decision["reject_reason"] == "collision_object_mismatch"


def test_experiment_defaults_are_clean_and_short():
    source = (ROOT / "config" / "flow_config.py").read_text(encoding="utf-8")
    assert 'TOTAL_AGENT2_TIMESTEPS = 10_240' in source
    assert 'CHECKPOINT_FREQUENCY = 10_240' in source
    assert 'RESUME_AGENT_2 = False' in source


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
