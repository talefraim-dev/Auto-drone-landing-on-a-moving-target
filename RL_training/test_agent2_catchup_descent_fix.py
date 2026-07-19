"""Regression checks for stable landing-gate behavior during bottom catch-up.

No AirSim/Unreal connection is required. These tests reproduce the two concrete
failures seen in the diagnostic screenshots:
  * a visually centered target at ~6 m was rejected because bbox-relative error
    grows when the target bbox is narrow;
  * a raw detector bbox jump was differentiated directly into noisy XY control.
"""
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
    for name in (
        "BBox",
        "DroneState",
        "ObstacleState",
        "ObservationBuilder",
        "ObservationBuilderConfig",
    ):
        setattr(observation, name, object)
    sys.modules["observation_builder"] = observation

    tracker = ModuleType("resnet_yolo_tracker")
    tracker.YoloResNetTracker = object
    sys.modules["resnet_yolo_tracker"] = tracker

    spec = importlib.util.spec_from_file_location(
        "agent2_stable_landing_test", ROOT / "agent2_landing_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()
Agent2Config = MODULE.Agent2Config
Agent2LandingEnv = MODULE.Agent2LandingEnv


def _env(*, catchup=True):
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._step = 10
    env._alignment_ready_streak = 0
    env._descent_alignment_latched = False
    env._landing_lock_visual_gap_steps = 0
    env._landing_lock_bad_live_steps = 0
    env._landing_lock_acquired_step = -999999
    env._episode_recenter_steps = 0
    env._predictive_catchup_active = bool(catchup)
    env._last_predictive_guidance = {
        "guidance_active": True,
        "measurement_live": True,
        "prediction_only": False,
        "catchup_active": bool(catchup),
        "predicted_center_error": 0.11,
    }
    env._control_bbox_xyxy = None
    env._control_bbox_history = []
    env._control_bbox_outlier_suppressed = False
    return env


def _info(*, height=6.0, center=0.15, bbox_rel=1.30):
    return {
        "bottom_match_live": True,
        "bottom_match_confirmed": True,
        "bottom_live_match_streak": 20,
        "bottom_similarity": 0.98,
        "bottom_center_error": center,
        "bottom_bbox_rel_err": bbox_rel,
        "visual_motion_attitude_valid": True,
        "relative_height_to_target_m": height,
    }


def test_high_altitude_bbox_relative_error_no_longer_deadlocks_descent():
    env = _env(catchup=True)
    # Screenshot-like geometry: good image-center error but bbox-relative error
    # above 1.0 because the vehicle is still narrow at ~6 m altitude.
    first = env._vertical_control_state(_info())
    assert first[0] == "ALIGN_LOCK_PENDING_SOFT_CATCHUP", first
    assert first[1] is False
    assert env._alignment_ready_streak == 1

    second = env._vertical_control_state(_info())
    assert second[0] == "DESCEND_SOFT_CATCHUP_LOCK_ACQUIRED", second
    assert second[1] is True
    assert env._descent_alignment_latched is True

    vz, limit, soft = env._bounded_descent_command(
        0.80, second[0], second[1], _info()
    )
    assert soft is True
    assert abs(vz - 0.28) < 1.0e-9
    assert abs(limit - 0.28) < 1.0e-9


def test_near_contact_reenables_bbox_footprint_gate():
    env = _env(catchup=True)
    state, allowed, reason = env._vertical_control_state(
        _info(height=1.0, center=0.12, bbox_rel=1.30)
    )
    assert state == "HOLD_PREDICTIVE_CATCHUP", (state, allowed, reason)
    assert allowed is False
    assert "bbox_relative" in reason
    assert env._alignment_ready_streak == 0


def test_predicted_target_escaping_still_blocks_soft_descent():
    env = _env(catchup=True)
    env._last_predictive_guidance["predicted_center_error"] = 0.55
    state, allowed, reason = env._vertical_control_state(_info())
    assert state == "HOLD_PREDICTIVE_CATCHUP", (state, allowed, reason)
    assert allowed is False
    assert reason == "catchup_descent_predicted_center_escaping"


def test_unconfirmed_or_non_live_target_never_descends():
    env = _env(catchup=True)
    unconfirmed = _info()
    unconfirmed["bottom_match_confirmed"] = False
    state, allowed, reason = env._vertical_control_state(unconfirmed)
    assert state == "HOLD_PREDICTIVE_CATCHUP"
    assert allowed is False
    assert reason == "catchup_descent_requires_confirmed_match"

    env = _env(catchup=True)
    not_live = _info()
    not_live["bottom_match_live"] = False
    state, allowed, _ = env._vertical_control_state(not_live)
    assert state == "HOLD_NO_LIVE_MATCH"
    assert allowed is False


def test_noncatchup_high_altitude_lock_uses_image_center_then_near_contact_uses_bbox():
    env = _env(catchup=False)
    first = env._vertical_control_state(_info(height=6.0, center=0.20, bbox_rel=1.20))
    second = env._vertical_control_state(_info(height=6.0, center=0.20, bbox_rel=1.20))
    assert first[0] == "ALIGN_LOCK_PENDING", first
    assert second[0] == "DESCEND_LANDING_LOCK_ACQUIRED", second
    assert second[1] is True

    env = _env(catchup=False)
    state, allowed, reason = env._vertical_control_state(
        _info(height=1.0, center=0.20, bbox_rel=1.20)
    )
    assert state == "ALIGN_BBOX", (state, allowed, reason)
    assert allowed is False


def test_bbox_filter_suppresses_one_frame_shape_and_center_jump():
    env = _env(catchup=True)
    shape = (1000, 1400, 3)
    stable = np.asarray([510.0, 300.0, 670.0, 700.0], dtype=np.float32)
    filtered0 = env._stabilize_control_bbox(
        stable, live_match=True, frame_shape=shape
    )
    assert np.allclose(filtered0, stable, atol=1.0)

    # A raw box suddenly selects a broad/right-side vehicle region. The control
    # bbox must remain near the stable history rather than follow that jump.
    outlier = np.asarray([820.0, 180.0, 1160.0, 820.0], dtype=np.float32)
    filtered1 = env._stabilize_control_bbox(
        outlier, live_match=True, frame_shape=shape
    )
    raw_center_jump = np.linalg.norm(
        np.asarray([(outlier[0] + outlier[2]) / 2, (outlier[1] + outlier[3]) / 2])
        - np.asarray([(stable[0] + stable[2]) / 2, (stable[1] + stable[3]) / 2])
    )
    filtered_center_jump = np.linalg.norm(
        np.asarray([(filtered1[0] + filtered1[2]) / 2, (filtered1[1] + filtered1[3]) / 2])
        - np.asarray([(filtered0[0] + filtered0[2]) / 2, (filtered0[1] + filtered0[3]) / 2])
    )
    assert filtered_center_jump < 0.20 * raw_center_jump, (
        raw_center_jump,
        filtered_center_jump,
        filtered1,
    )

    held = env._stabilize_control_bbox(None, live_match=False, frame_shape=shape)
    assert np.allclose(held, filtered1)


def _bbox_center(box):
    box = np.asarray(box, dtype=np.float64)
    return np.asarray([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])


def test_control_bbox_uses_true_xyxy_center_not_top_left_corner():
    env = _env(catchup=True)
    shape = (1000, 1400, 3)
    raw = np.asarray([510.0, 300.0, 670.0, 700.0], dtype=np.float32)
    control = env._stabilize_control_bbox(raw, live_match=True, frame_shape=shape)
    assert np.allclose(control, raw, atol=1.0), (raw, control)
    assert np.allclose(_bbox_center(control), [590.0, 500.0], atol=1.0)


def test_top_edge_clipping_follows_vehicle_center_instead_of_staying_on_road():
    env = _env(catchup=True)
    shape = (1000, 1400, 3)
    full = np.asarray([520.0, 260.0, 700.0, 700.0], dtype=np.float32)
    env._stabilize_control_bbox(full, live_match=True, frame_shape=shape)

    # Vehicle reaches the top image edge and only its visible portion is boxed.
    edge_raw = np.asarray([520.0, 0.0, 700.0, 220.0], dtype=np.float32)
    control = env._stabilize_control_bbox(
        edge_raw, live_match=True, frame_shape=shape
    )
    raw_center = _bbox_center(edge_raw)
    control_center = _bbox_center(control)

    assert env._control_bbox_filter_mode == "EDGE_FOLLOW"
    assert abs(control_center[0] - raw_center[0]) <= 2.0
    assert abs(control_center[1] - raw_center[1]) <= 25.0, (
        raw_center,
        control_center,
        control,
    )
    assert control[1] <= 1.0
    assert control[3] < 300.0, control


def test_size_outlier_does_not_freeze_a_valid_center_update():
    env = _env(catchup=True)
    shape = (1000, 1400, 3)
    first = np.asarray([500.0, 300.0, 700.0, 700.0], dtype=np.float32)
    control0 = env._stabilize_control_bbox(
        first, live_match=True, frame_shape=shape
    )

    # Same verified target moves by 100 px while YOLO changes its footprint.
    changed = np.asarray([520.0, 240.0, 880.0, 960.0], dtype=np.float32)
    control1 = env._stabilize_control_bbox(
        changed, live_match=True, frame_shape=shape
    )
    requested = _bbox_center(changed) - _bbox_center(control0)
    applied = _bbox_center(control1) - _bbox_center(control0)
    assert np.linalg.norm(applied) >= 0.70 * np.linalg.norm(requested), (
        requested,
        applied,
        control1,
    )


def test_soft_descent_touchdown_cap_is_lower():
    env = _env(catchup=True)
    assert env._catchup_descent_speed_limit(_info(height=6.0)) == 0.28
    assert env._catchup_descent_speed_limit(_info(height=0.7)) == 0.12


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} stable bbox/descent tests")
