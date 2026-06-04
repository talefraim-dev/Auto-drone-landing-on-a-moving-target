from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


def clip_xyxy(bbox, frame_width: int, frame_height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, frame_width - 1.0))
    y1 = max(0.0, min(y1, frame_height - 1.0))
    x2 = max(0.0, min(x2, frame_width - 1.0))
    y2 = max(0.0, min(y2, frame_height - 1.0))
    if x2 <= x1:
        x2 = min(frame_width - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(frame_height - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_xywh(bbox) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def xywh_to_xyxy(bbox) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + max(1.0, w), y + max(1.0, h)], dtype=np.float32)


def bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return float(max(1.0, x2 - x1) * max(1.0, y2 - y1))


def bbox_aspect(bbox) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return float(max(1.0, x2 - x1) / max(1.0, y2 - y1))


def center_distance(a, b) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def touches_frame_edge(bbox, frame_width: int, frame_height: int, margin_px: int = 3) -> bool:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (
        x1 <= margin_px or y1 <= margin_px
        or x2 >= frame_width - 1 - margin_px
        or y2 >= frame_height - 1 - margin_px
    )


def crop_safe(frame, bbox):
    h, w = frame.shape[:2]
    b = clip_xyxy(bbox, w, h)
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop


class SimpleCenterKalman:
    def __init__(self):
        self.initialized = False
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 50.0
        self.Q = np.eye(8, dtype=np.float32) * 1.5
        self.R = np.eye(4, dtype=np.float32) * 20.0
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

    @staticmethod
    def _measure(bbox):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float32)

    @staticmethod
    def _state_to_xyxy(x):
        cx, cy, w, h = [float(v) for v in x[:4]]
        return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)

    def initialize(self, bbox):
        z = self._measure(bbox)
        self.x[:] = 0.0
        self.x[0:4, 0] = z
        self.P = np.eye(8, dtype=np.float32) * 30.0
        self.P[4:, 4:] = np.eye(4, dtype=np.float32) * 200.0
        self.initialized = True
        return self._state_to_xyxy(self.x[:, 0])

    def predict(self, dt: float = 1.0):
        if not self.initialized:
            raise RuntimeError("Kalman not initialized")
        F = np.eye(8, dtype=np.float32)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = float(max(1e-3, dt))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self._state_to_xyxy(self.x[:, 0])

    def update(self, bbox):
        if not self.initialized:
            return self.initialize(bbox)
        z = self._measure(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        return self._state_to_xyxy(self.x[:, 0])


@dataclass
class MemoryEntry:
    bbox_xyxy: np.ndarray
    score: float
    frame_index: int
    crop: Optional[np.ndarray]


class TargetMemoryBank:
    def __init__(self, max_templates: int = 12):
        self.max_templates = int(max_templates)
        self.original: Optional[MemoryEntry] = None
        self.recent: list[MemoryEntry] = []
        self.match_count = 0
        self.pred_count = 0
        self.lost_count = 0

    def initialize(self, frame, bbox_xyxy, score: float = 1.0, frame_index: int = 0):
        crop = crop_safe(frame, bbox_xyxy)
        self.original = MemoryEntry(np.asarray(bbox_xyxy, dtype=np.float32).copy(), float(score), int(frame_index), None if crop is None else crop.copy())
        self.recent = []
        self.match_count = self.pred_count = self.lost_count = 0

    def update_good(self, frame, bbox_xyxy, score: float, frame_index: int):
        self.match_count += 1
        crop = crop_safe(frame, bbox_xyxy)
        self.recent.append(MemoryEntry(np.asarray(bbox_xyxy, dtype=np.float32).copy(), float(score), int(frame_index), None if crop is None else crop.copy()))
        if len(self.recent) > self.max_templates:
            self.recent = self.recent[-self.max_templates:]

    def update_pred(self):
        self.pred_count += 1

    def update_lost(self):
        self.lost_count += 1
