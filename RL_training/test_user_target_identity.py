"""Static/runtime-light tests for Agent-2 user_target identity architecture."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _install_stubs() -> None:
    gym = types.ModuleType("gymnasium")

    class Env:
        pass

    class Box:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    gym.Env = Env
    gym.spaces = SimpleNamespace(Box=Box)
    sys.modules["gymnasium"] = gym

    airsim = types.ModuleType("cosysairsim")
    airsim.MultirotorClient = object
    airsim.YawMode = lambda **kwargs: SimpleNamespace(**kwargs)
    airsim.ImageRequest = lambda *args, **kwargs: None
    airsim.ImageType = SimpleNamespace(Scene=0)
    airsim.to_eularian_angles = lambda _q: (0.0, 0.0, 0.0)
    sys.modules["cosysairsim"] = airsim

    tracker_module = types.ModuleType("resnet_yolo_tracker")
    tracker_module.YoloResNetTracker = object
    sys.modules["resnet_yolo_tracker"] = tracker_module


class DummyCandidate:
    def __init__(self, cls_id: int, conf: float, bbox: list[float], embedding: torch.Tensor):
        self.cls_id = cls_id
        self.conf = conf
        self.bbox = bbox
        self.embedding = embedding


class DummyTracker:
    def __init__(self, candidates):
        self.candidates = candidates
        self.last_bbox = None
        self.last_good_bbox = None
        self.last_score = 0.0
        self.last_mode = "IDLE"

    def _detect_candidates(self, _frame):
        return list(self.candidates)

    @staticmethod
    def _embedding_from_bbox(_frame, bbox):
        for candidate in DummyTracker._active_candidates:
            if candidate.bbox is bbox or candidate.bbox == bbox:
                return candidate.embedding
        return None


DummyTracker._active_candidates = []


def _make_env(module, candidates):
    env = module.Agent2LandingEnv.__new__(module.Agent2LandingEnv)
    env.cfg = SimpleNamespace(
        min_match_similarity=0.60,
        min_match_margin=0.03,
        high_conf_reacquire_similarity=0.78,
        max_reacquire_center_jump_norm=0.35,
        max_prediction_steps=20,
        adaptive_embedding_identity_floor=0.52,
        adaptive_embedding_enabled=False,
    )
    env._reference_embeddings = [torch.tensor([1.0, 0.0])]
    env._original_embedding = env._reference_embeddings[0]
    env._bottom_anchor_embedding = None
    env._adaptive_embeddings = []
    env._adaptive_embedding_steps = []
    env._last_adaptive_embedding_update_step = -999999
    env._adaptive_embedding_updates = 0
    env._target_id = "user_target"
    env._target_class_id = 2
    env._last_bbox_xyxy = np.asarray([1, 2, 11, 12], dtype=np.float32)
    env._last_similarity = 0.9
    env._last_candidate_class_id = None
    env._last_candidate_confidence = 0.0
    env._last_candidate_count = 0
    env._last_candidate_scores = []
    env._prediction_steps = 0
    env._live_match_streak = 0
    env._last_match_margin = 0.0
    env._last_spatial_jump_norm = 0.0
    env._last_match_reject_reason = ""
    env._last_match_step = -999
    env._step = 10
    DummyTracker._active_candidates = list(candidates)
    env.tracker = DummyTracker(candidates)
    return env


def main() -> None:
    _install_stubs()
    module = importlib.import_module("agent2_landing_env")
    frame = np.zeros((720, 960, 3), dtype=np.uint8)

    # A candidate classified by YOLO as cell phone must still be accepted when
    # ResNet says it is the selected visual instance.
    cell_phone = DummyCandidate(
        cls_id=67,
        conf=0.97,
        bbox=[100.0, 120.0, 80.0, 60.0],
        embedding=torch.tensor([1.0, 0.0]),
    )
    env = _make_env(module, [cell_phone])
    bbox, score, mode = env._strict_track(frame)
    assert mode == "MATCH", (mode, score)
    assert score > 0.99
    assert env._last_candidate_class_id == 67
    assert env._target_class_id == 2, "Initial YOLO class must remain diagnostic-only."
    assert bbox.tolist() == [100.0, 120.0, 180.0, 180.0]
    print("PASS: YOLO cls=67 accepted as user_target by ResNet.")

    # YOLO confidence must not participate in identity judgement.
    low_conf_match = DummyCandidate(
        cls_id=23,
        conf=0.01,
        bbox=[20.0, 30.0, 40.0, 50.0],
        embedding=torch.tensor([1.0, 0.0]),
    )
    env = _make_env(module, [low_conf_match])
    _bbox, score, mode = env._strict_track(frame)
    assert mode == "MATCH" and score > 0.99
    assert env._last_candidate_class_id == 23
    print("PASS: YOLO confidence/class are diagnostic-only after selection.")

    # The highest ResNet similarity wins across different YOLO classes.
    candidates = [
        DummyCandidate(2, 0.99, [0.0, 0.0, 20.0, 20.0], torch.tensor([0.60, 0.80])),
        DummyCandidate(67, 0.20, [50.0, 60.0, 30.0, 40.0], torch.tensor([0.99, 0.01])),
    ]
    env = _make_env(module, candidates)
    bbox, score, mode = env._strict_track(frame)
    assert mode == "MATCH"
    assert env._last_candidate_class_id == 67
    assert bbox.tolist() == [50.0, 60.0, 80.0, 100.0]
    assert score > 0.98
    print("PASS: Best ResNet candidate wins regardless of YOLO class/confidence.")

    # No YOLO proposal keeps the last bbox temporarily instead of deleting it.
    env = _make_env(module, [])
    bbox, score, mode = env._strict_track(frame)
    assert mode == "PRED_NO_DETECTION"
    assert bbox is not None and bbox.tolist() == [1.0, 2.0, 11.0, 12.0]
    assert env._prediction_steps == 1
    print("PASS: No-detection continuity keeps the previous bbox.")

    # Real frame dimensions must update observation normalization.
    env = module.Agent2LandingEnv.__new__(module.Agent2LandingEnv)
    env.cfg = SimpleNamespace(image_width=960, image_height=540)
    env.observation_builder = SimpleNamespace(config=SimpleNamespace(image_width=960, image_height=540))
    env._sync_image_geometry(frame)
    assert env.cfg.image_height == 720
    assert env.observation_builder.config.image_height == 720
    print("PASS: Observation geometry follows the actual 960x720 frame.")

    # Agent 1 handoff boxes are float32 XYXY values. Agent 2 must convert
    # them to integer XYWH pixel coordinates before the ResNet crop call.
    captured = {}

    class AnchorTracker:
        device = "cpu"

        @staticmethod
        def _embedding_from_bbox(_frame, bbox):
            captured["bbox"] = list(bbox)
            assert all(isinstance(v, int) for v in bbox), bbox
            return torch.tensor([1.0, 0.0])

    env = module.Agent2LandingEnv.__new__(module.Agent2LandingEnv)
    env.tracker = AnchorTracker()
    env._original_embedding = torch.tensor([1.0, 0.0])
    env._bottom_anchor_embedding = None
    env._reference_embeddings = [env._original_embedding.clone()]
    handoff_xyxy = np.asarray([487.0, 38.0, 593.0, 242.0], dtype=np.float32)
    assert env._set_bottom_anchor_from_bbox(frame, handoff_xyxy)
    assert captured["bbox"] == [487, 38, 106, 204], captured
    assert len(env._reference_embeddings) == 2
    print("PASS: float32 handoff bbox becomes integer ResNet crop coordinates.")

    # Verify protected trackers/config remain unchanged and the authorized DroneEnv fix is present.
    expected = {
        "drone_env.py": "82f4bc21fadc96de1ad3e2a4cb75ff752bc9f7269b14c2fd776082941236115d",
        "object_tracker.py": "ceca68653b9b8a304a23184d33d81d5b1a0b9529b053c70d5770a5aaad725156",
        "resnet_yolo_tracker.py": "e7b5fb098738d27df4fb82b438fd865a477d791c95a7d288a546457053f3309f",
        "config/tracking_config.py": "a3282379e9b83f4aa4b3f265a4a85b5f8a266b6c34b1c4730a4d07cec3c26b5a",
        "Run_train.py": "a766c20e091bfc573c64f865f9ab8cf55cfa055cfcc6ad28fbd0560f5c96b823",
    }
    import hashlib

    for rel, wanted in expected.items():
        got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
        assert got == wanted, f"Agent-1 baseline changed: {rel}: {got} != {wanted}"
    print("PASS: protected runtime hashes and authorized DroneEnv hash match this package.")


if __name__ == "__main__":
    main()
