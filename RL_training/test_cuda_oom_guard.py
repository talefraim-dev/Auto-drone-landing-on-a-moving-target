from __future__ import annotations

import types

import numpy as np

import resnet_yolo_tracker as tracker_module


class FakeModel:
    def __init__(self, fail_cuda: bool = False):
        self.fail_cuda = fail_cuda
        self.moves = []

    def to(self, device):
        device = str(device)
        self.moves.append(device)
        if device.startswith("cuda") and self.fail_cuda:
            raise RuntimeError("CUDA error: out of memory")
        return self


class FakeYolo:
    def __init__(self):
        self.calls = []
        self.fail_cuda_once = True

    def predict(self, frame, conf, verbose, device):
        self.calls.append(str(device))
        if str(device).startswith("cuda") and self.fail_cuda_once:
            self.fail_cuda_once = False
            raise RuntimeError("cudaErrorMemoryAllocation: out of memory")
        return []


def test_resnet_oom_falls_back_once_and_is_shared():
    tracker_module._SHARED_RESNET_MODELS.clear()
    tracker_module._CUDA_REID_DISABLED_AFTER_OOM = False

    created = []

    def make_model():
        model = FakeModel(fail_cuda=(len(created) == 0))
        created.append(model)
        return model

    original = tracker_module.YoloResNetTracker._new_resnet18
    tracker_module.YoloResNetTracker._new_resnet18 = staticmethod(make_model)
    try:
        first = object.__new__(tracker_module.YoloResNetTracker)
        first.device = "cuda"
        model1 = first._build_resnet_feature_extractor()
        assert first.device == "cpu"
        assert model1 is tracker_module._SHARED_RESNET_MODELS["cpu"]
        assert tracker_module._CUDA_REID_DISABLED_AFTER_OOM is True

        second = object.__new__(tracker_module.YoloResNetTracker)
        second.device = "cuda"
        model2 = second._build_resnet_feature_extractor()
        assert second.device == "cpu"
        assert model2 is model1
        assert len(created) == 2, created
    finally:
        tracker_module.YoloResNetTracker._new_resnet18 = original
        tracker_module._SHARED_RESNET_MODELS.clear()
        tracker_module._CUDA_REID_DISABLED_AFTER_OOM = False


def test_yolo_oom_retries_cpu_and_disables_repeated_cuda_attempts():
    tracker_module._CUDA_YOLO_DISABLED_AFTER_OOM = False

    first = object.__new__(tracker_module.YoloResNetTracker)
    first.yolo = FakeYolo()
    first.yolo_conf = 0.25
    first.detector_device = "cuda"
    first._detector_cpu_fallback_used = False

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert first._detect_candidates(frame) == []
    assert first.yolo.calls == ["cuda", "cpu"]
    assert first.detector_device == "cpu"
    assert tracker_module._CUDA_YOLO_DISABLED_AFTER_OOM is True

    second = object.__new__(tracker_module.YoloResNetTracker)
    second.yolo = FakeYolo()
    second.yolo_conf = 0.25
    second.detector_device = "cuda"
    second._detector_cpu_fallback_used = False
    assert second._detect_candidates(frame) == []
    assert second.yolo.calls == ["cpu"]

    tracker_module._CUDA_YOLO_DISABLED_AFTER_OOM = False


def test_non_oom_errors_are_not_hidden():
    tracker_module._CUDA_YOLO_DISABLED_AFTER_OOM = False

    class BrokenYolo:
        def predict(self, *args, **kwargs):
            raise RuntimeError("invalid tensor shape")

    obj = object.__new__(tracker_module.YoloResNetTracker)
    obj.yolo = BrokenYolo()
    obj.yolo_conf = 0.25
    obj.detector_device = "cuda"
    obj._detector_cpu_fallback_used = False

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        obj._detect_candidates(frame)
    except RuntimeError as exc:
        assert "invalid tensor shape" in str(exc)
    else:
        raise AssertionError("non-OOM error was swallowed")


if __name__ == "__main__":
    test_resnet_oom_falls_back_once_and_is_shared()
    test_yolo_oom_retries_cpu_and_disables_repeated_cuda_attempts()
    test_non_oom_errors_are_not_hidden()
    print("PASS CUDA OOM guard")
