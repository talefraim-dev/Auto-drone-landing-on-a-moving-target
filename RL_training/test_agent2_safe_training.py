"""Static/runtime-light regression tests for Agent-2 safe training controls.

These tests do not connect to Unreal or AirSim. They validate the pure control
and identity decisions that previously allowed blind descent and false matches.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import torch


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
    sys.modules["cosysairsim"] = airsim

    lidar = ModuleType("lidar_processor")
    lidar.LidarProcessor = object
    lidar.LidarProcessorConfig = object
    lidar.point_cloud_to_array = lambda value: np.asarray(value)
    sys.modules["lidar_processor"] = lidar

    observation = ModuleType("observation_builder")
    for name in ("BBox", "DroneState", "ObstacleState", "ObservationBuilder", "ObservationBuilderConfig"):
        setattr(observation, name, type(name, (), {}))
    sys.modules["observation_builder"] = observation

    tracker_module = ModuleType("resnet_yolo_tracker")
    tracker_module.YoloResNetTracker = object
    sys.modules["resnet_yolo_tracker"] = tracker_module

    spec = importlib.util.spec_from_file_location(
        "agent2_landing_env_safe_test",
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


def _bare_env() -> object:
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config()
    env._alignment_ready_streak = 0
    env._descent_alignment_latched = False
    env._episode_recenter_steps = 0
    env._control_img_vel_x = 0.0
    env._control_img_vel_y = 0.0
    env._episode_xy_hold_steps = 0
    return env


def test_descent_is_blocked_without_current_live_match():
    env = _bare_env()
    state, allowed, reason = env._vertical_control_state(
        {
            "bottom_match_live": False,
            "bottom_match_confirmed": False,
            "bottom_live_match_streak": 0,
            "bottom_similarity": 0.95,
            "bottom_center_error": 0.01,
            "bottom_bbox_rel_err": 0.01,
        }
    )
    assert state == "HOLD_NO_LIVE_MATCH"
    assert allowed is False
    assert reason == "target_not_detected_current_frame"


def test_descent_waits_for_consecutive_live_matches():
    env = _bare_env()
    state, allowed, _ = env._vertical_control_state(
        {
            "bottom_match_live": True,
            "bottom_match_confirmed": True,
            "bottom_live_match_streak": 2,
            "bottom_similarity": 0.95,
            "bottom_center_error": 0.05,
            "bottom_bbox_rel_err": 0.10,
        }
    )
    assert state == "ALIGN_LOCK_PENDING"
    assert allowed is False


def test_descent_is_allowed_only_when_live_confirmed_and_aligned():
    env = _bare_env()
    info = {
        "bottom_match_live": True,
        "bottom_match_confirmed": True,
        "bottom_live_match_streak": 10,
        "bottom_similarity": 0.90,
        "bottom_center_error": 0.12,
        "bottom_bbox_rel_err": 0.20,
    }
    for _ in range(env.cfg.alignment_streak_required):
        state, allowed, reason = env._vertical_control_state(info)
    assert state == "DESCEND_LANDING_LOCK_ACQUIRED"
    assert allowed is True
    assert reason == ""


class _FakeTracker:
    def __init__(self, candidates):
        self.candidates = candidates
        self.device = torch.device("cpu")
        self.last_bbox = None
        self.last_good_bbox = None
        self.last_score = 0.0
        self.last_mode = "IDLE"

    def _detect_candidates(self, _frame):
        return self.candidates

    @staticmethod
    def _embedding_from_bbox(_frame, bbox):
        return bbox.embedding


def _candidate(cls_id: int, embedding, x=430, y=260, w=100, h=180, conf=0.9):
    bbox = SimpleNamespace(
        __getitem__=None,
        embedding=torch.tensor(embedding, dtype=torch.float32),
    )
    # The production code treats bbox as a sequence and also passes it back to
    # the fake tracker. A list subclass gives us both behavior and an embedding.
    class BBoxList(list):
        pass

    box = BBoxList([x, y, w, h])
    box.embedding = torch.nn.functional.normalize(
        torch.tensor(embedding, dtype=torch.float32), dim=0
    )
    return SimpleNamespace(cls_id=cls_id, conf=conf, bbox=box)


def _tracking_env(candidates):
    env = _bare_env()
    env.cfg.adaptive_embedding_enabled = False
    env.tracker = _FakeTracker(candidates)
    env._reference_embeddings = [torch.tensor([1.0, 0.0], dtype=torch.float32)]
    env._original_embedding = env._reference_embeddings[0]
    env._last_bbox_xyxy = np.asarray([420, 250, 530, 450], dtype=np.float32)
    env._last_similarity = 0.9
    env._prediction_steps = 0
    env._live_match_streak = 0
    env._last_match_step = -999
    env._step = 10
    env._last_candidate_count = 0
    env._last_candidate_scores = []
    env._last_candidate_class_id = None
    env._last_candidate_confidence = 0.0
    env._last_match_margin = 0.0
    env._last_spatial_jump_norm = 0.0
    env._last_match_reject_reason = ""
    return env


def test_yolo_class_is_diagnostic_only_after_selection():
    candidate = _candidate(67, [0.98, 0.05])
    env = _tracking_env([candidate])
    bbox, similarity, mode = env._strict_track(np.zeros((720, 960, 3), dtype=np.uint8))
    assert mode == "MATCH"
    assert similarity > 0.9
    assert bbox is not None
    assert env._last_candidate_class_id == 67


def test_low_similarity_candidate_cannot_become_live_match():
    candidate = _candidate(14, [0.52, 0.85])
    env = _tracking_env([candidate])
    _bbox, similarity, mode = env._strict_track(np.zeros((720, 960, 3), dtype=np.uint8))
    assert similarity < env.cfg.min_match_similarity
    assert mode == "PRED_REJECTED_MATCH"
    assert env._live_match_streak == 0
    assert env._last_match_reject_reason == "low_similarity"


def test_source_contains_explicit_positive_vz_block():
    source = (ROOT / "agent2_landing_env.py").read_text(encoding="utf-8")
    assert "descent_blocked = bool(descent_requested and not descent_allowed)" in source
    assert "if descent_blocked:" in source
    assert "return 0.0, 0.0, False" in source
    assert "bottom_match_live" in source


def test_step_sends_zero_down_velocity_when_target_is_not_live():
    env = _bare_env()
    env._step = 0
    env._last_info = {
        "bottom_match_live": False,
        "bottom_match_confirmed": False,
        "bottom_live_match_streak": 0,
        "bottom_similarity": 0.0,
        "bottom_center_error": 999.0,
        "bottom_bbox_rel_err": 999.0,
        "front_dist_m": 20.0,
        "back_dist_m": 20.0,
        "left_dist_m": 20.0,
        "right_dist_m": 20.0,
    }
    env._episode_descent_requested_steps = 0
    env._episode_descent_allowed_steps = 0
    env._episode_descent_blocked_steps = 0
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
    env._target_surface_altitude_m = 0.0
    env._target_surface_source = "test"
    env._control_img_vel_x = 0.0
    env._control_img_vel_y = 0.0
    env._episode_xy_hold_steps = 0
    env._alignment_ready_streak = 0
    env._descent_alignment_latched = False
    env._episode_recenter_steps = 0
    env._prev_center_error_for_reward = None
    env._episode_dense_reward = 0.0
    env._episode_climb_command_blocked_steps = 0

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
            "bottom_center_error": 999.0,
            "bottom_bbox_rel_err": 999.0,
            "relative_height_to_target_m": 3.0,
            "bottom_match_live": False,
            "bottom_match_recent": False,
            "alt_agl_m": 5.0,
        },
    )
    env._new_collision = lambda: (False, "", 0)

    env.step(np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert sent["vz"] == 0.0, sent
    assert env._episode_descent_requested_steps == 1
    assert env._episode_descent_blocked_steps == 1


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
