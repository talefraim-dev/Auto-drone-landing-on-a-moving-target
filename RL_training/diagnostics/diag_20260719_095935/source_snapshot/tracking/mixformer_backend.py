from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config.deep_tracker_config import DeepTrackerConfig
from tracking.deep_tracker_memory import xyxy_to_xywh, xywh_to_xyxy


class MixFormerBackend:
    """
    Runtime wrapper around the official MixFormer tracker.

    Expected official API:
        from lib.test.evaluation import Tracker
        wrapper = Tracker(name, param, "video", tracker_params={...})
        tracker = wrapper.create_tracker(wrapper.params)
        tracker.initialize(frame, {"init_bbox": [x, y, w, h]})
        out = tracker.track(frame)
        out["target_bbox"] -> [x, y, w, h]
    """

    def __init__(self, cfg: DeepTrackerConfig | None = None):
        self.cfg = cfg if cfg is not None else DeepTrackerConfig()
        self.wrapper = None
        self.tracker = None
        self.initialized = False
        self._load_backend()

    def _load_backend(self) -> None:
        root = Path(self.cfg.MIXFORMER_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"MixFormer root not found: {root}\n"
                "Set DeepTrackerConfig.MIXFORMER_ROOT in config/deep_tracker_config.py"
            )

        model_path = Path(self.cfg.MIXFORMER_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MixFormer model not found: {model_path}\n"
                "Set DeepTrackerConfig.MIXFORMER_MODEL_PATH in config/deep_tracker_config.py"
            )

        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            from lib.test.evaluation import Tracker
        except Exception as exc:
            raise ImportError(
                "Failed importing official MixFormer Tracker. "
                "Make sure MIXFORMER_ROOT points to the official MixFormer repo."
            ) from exc

        tracker_params = {
            "model": model_path.name,
            "search_area_scale": float(self.cfg.MIXFORMER_SEARCH_AREA_SCALE),
            "update_interval": int(self.cfg.MIXFORMER_UPDATE_INTERVAL),
            "online_sizes": int(self.cfg.MIXFORMER_ONLINE_SIZES),
            "max_score_decay": float(self.cfg.MIXFORMER_MAX_SCORE_DECAY),
        }

        self.wrapper = Tracker(
            self.cfg.MIXFORMER_TRACKER_NAME,
            self.cfg.MIXFORMER_PARAM_NAME,
            self.cfg.MIXFORMER_DATASET_NAME,
            tracker_params=tracker_params,
        )
        self.tracker = self.wrapper.create_tracker(self.wrapper.params)

    def initialize(self, frame_bgr: np.ndarray, bbox_xyxy) -> None:
        init_bbox_xywh = xyxy_to_xywh(bbox_xyxy)
        out = self.tracker.initialize(frame_bgr, {"init_bbox": init_bbox_xywh})
        self.initialized = True
        return out

    def track(self, frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float, dict]:
        if not self.initialized:
            return None, 0.0, {"reason": "not_initialized"}

        out = self.tracker.track(frame_bgr)
        if out is None:
            return None, 0.0, {"reason": "backend_returned_none"}

        bbox_xywh = out.get("target_bbox", None)
        if bbox_xywh is None:
            return None, 0.0, {"reason": "missing_target_bbox", "raw": out}

        bbox_xyxy = xywh_to_xyxy(bbox_xywh)

        score = None
        for key in ("best_score", "target_score", "score", "max_score", "conf"):
            if key in out:
                try:
                    score = float(out[key])
                    break
                except Exception:
                    pass

        if score is None or not np.isfinite(score):
            score = float(self.cfg.default_score_when_missing)

        return bbox_xyxy, float(score), dict(out)
