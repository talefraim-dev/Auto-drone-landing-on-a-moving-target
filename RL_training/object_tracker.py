"""
object_tracker.py
YOLO + Legacy RGB Histogram + Kalman Velocity EMA + Hard Search Window + Yaw Shift.
NO ByteTrack. NO SAM2.

Drop-in replacement for RL_training/object_tracker.py.
Based on the older simple tracker that worked in Blocks, with minimal additions:
- hard search window
- corrected yaw shift
- freeze instead of switching target
- size/aspect guards
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.tracker_config import TrackerConfig


@dataclass
class Candidate:
    bbox: List[int]
    cls_id: Optional[int]
    conf: float
    feat_sim: float = 0.0
    spatial_sim: float = 0.0
    size_sim: float = 0.0
    aspect_sim: float = 0.0
    score: float = 0.0


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.16
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def reset(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        return self.kf.predict()

    def update(self, x: float, y: float):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))


class tracker:
    def __init__(self):
        self.cfg = TrackerConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weights = self.cfg.yolo_weights
        self.conf_match = float(self.cfg.yolo_conf)
        self.conf_click = float(self.cfg.click_conf)
        self.iou = float(self.cfg.yolo_iou)
        self.imgsz = int(self.cfg.imgsz)

        self.model = YOLO(self.weights).to(self.device)
        print(f"[TRACKER] YOLO loaded: {self.weights}")

        self.kf = KalmanFilter()
        self.STATE = "SEARCH"
        self.target_class_id = None
        self.target_features = None
        self.target_feature_bank = []
        self.last_bbox = None
        self.last_good_bbox = None
        self.last_mask = None
        self.target_area = None
        self.target_aspect = None

        self.MATCH_TH = float(self.cfg.match_threshold)
        self.AUTOLOCK_TH = float(self.cfg.autolock_threshold)
        self.FEAT_MIN_TH = float(self.cfg.feature_min_threshold)

        self.last_mode = "NONE"
        self.last_raw_mode = "NONE"
        self._prev_center = None
        self._vel_ema = np.zeros(2, dtype=np.float32)
        self.pred_frames = 0

        self.search_scale = float(self.cfg.search_window_scale)
        self.search_min_pad_px = float(self.cfg.search_min_pad_px)
        self.search_extra_yaw_pad_px = float(self.cfg.search_extra_yaw_pad_px)

        # Active reacquisition:
        # Local hard search prevents target switching, but if the bbox drifts away
        # while the real target is still visible, we need a controlled full-frame
        # identity search after several PRED frames.
        self.active_reacquire_enabled = bool(getattr(self.cfg, "active_reacquire_enabled", True))
        self.reacquire_after_pred_frames = int(getattr(self.cfg, "reacquire_after_pred_frames", 8))
        self.reacquire_every_n_frames = int(getattr(self.cfg, "reacquire_every_n_frames", 3))
        self.reacquire_min_score = float(getattr(self.cfg, "reacquire_min_score", 0.52))
        self.reacquire_feat_min = float(getattr(self.cfg, "reacquire_feat_min", 0.38))
        self.reacquire_max_candidates = int(getattr(self.cfg, "reacquire_max_candidates", 8))

        # Appearance bank + robust mid-pass similarity.
        self.appearance_bank_enabled = bool(getattr(self.cfg, "appearance_bank_enabled", True))
        self.appearance_bank_max_size = int(getattr(self.cfg, "appearance_bank_max_size", 24))
        self.appearance_bank_jitter_px = int(getattr(self.cfg, "appearance_bank_jitter_px", 8))
        self.appearance_bank_scale_jitter = float(getattr(self.cfg, "appearance_bank_scale_jitter", 0.10))
        self.appearance_bank_update_min_score = float(getattr(self.cfg, "appearance_bank_update_min_score", 0.74))
        self.appearance_bank_update_every_n_matches = int(getattr(self.cfg, "appearance_bank_update_every_n_matches", 3))
        self.midpass_low_quantile = float(getattr(self.cfg, "midpass_low_quantile", 0.25))
        self.midpass_high_quantile = float(getattr(self.cfg, "midpass_high_quantile", 0.85))
        self.candidate_midpass_variants = int(getattr(self.cfg, "candidate_midpass_variants", 7))
        self._match_update_counter = 0

        self.class_gate = bool(self.cfg.class_gate)
        self.size_min_score = float(self.cfg.size_min_score)
        self.aspect_min_score = float(self.cfg.aspect_min_score)
        self.max_pred_frames = int(self.cfg.max_pred_frames)
        self.freeze_without_yaw = bool(self.cfg.freeze_without_yaw)

        self.last_yaw_rate_cmd_dps = 0.0
        self.yaw_enabled = bool(self.cfg.yaw_shift_enabled)
        self.yaw_deadband = float(self.cfg.yaw_cmd_deadband)
        self.yaw_pix_per_cmd = float(self.cfg.yaw_pix_per_cmd)
        self.yaw_shift_sign = float(self.cfg.yaw_shift_sign)
        self.yaw_max_shift_px = float(self.cfg.yaw_max_shift_px)

        self.draw_search_window = bool(self.cfg.draw_search_window)
        self.draw_expected = bool(self.cfg.draw_expected)

        print(
            "[TRACKER] LEGACY-HIST+KALMAN+SEARCH-WINDOW initialized. "
            f"device={self.device} conf={self.conf_match} match_th={self.MATCH_TH} "
            f"feat_min={self.FEAT_MIN_TH} search_scale={self.search_scale} "
            f"yaw_pix_per_cmd={self.yaw_pix_per_cmd} "
            f"active_reacquire={self.active_reacquire_enabled} "
            f"appearance_bank={self.appearance_bank_enabled}"
        )

    @staticmethod
    def _clip_box_to_frame(b: List[int], frame) -> List[int]:
        H, W = frame.shape[:2]
        x1 = int(np.clip(b[0], 0, W - 1))
        y1 = int(np.clip(b[1], 0, H - 1))
        x2 = int(np.clip(b[2], 0, W - 1))
        y2 = int(np.clip(b[3], 0, H - 1))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def _bbox_center(b: List[int]) -> Tuple[float, float]:
        return (float(b[0] + b[2]) / 2.0, float(b[1] + b[3]) / 2.0)

    @staticmethod
    def _bbox_area(b: List[int]) -> float:
        return float(max(1, b[2] - b[0]) * max(1, b[3] - b[1]))

    @staticmethod
    def _bbox_aspect(b: List[int]) -> float:
        return float(max(1, b[2] - b[0])) / float(max(1, b[3] - b[1]))

    @staticmethod
    def _center_inside(cx: float, cy: float, b: List[int]) -> bool:
        return float(b[0]) <= cx <= float(b[2]) and float(b[1]) <= cy <= float(b[3])

    def _jitter_xyxy_boxes(self, frame, bbox, jitter_px: int, scale_jitter: float):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        offsets = [
            (0, 0, 1.00),
            (-jitter_px, 0, 1.00),
            (jitter_px, 0, 1.00),
            (0, -jitter_px, 1.00),
            (0, jitter_px, 1.00),
            (-jitter_px, -jitter_px, 1.0 + scale_jitter),
            (jitter_px, jitter_px, 1.0 + scale_jitter),
            (-jitter_px, jitter_px, 1.0 - 0.5 * scale_jitter),
            (jitter_px, -jitter_px, 1.0 - 0.5 * scale_jitter),
        ]

        boxes = []
        for dx, dy, sc in offsets:
            nbw = max(2.0, bw * sc)
            nbh = max(2.0, bh * sc)
            ncx = cx + dx
            ncy = cy + dy
            b = [
                int(round(ncx - nbw * 0.5)),
                int(round(ncy - nbh * 0.5)),
                int(round(ncx + nbw * 0.5)),
                int(round(ncy + nbh * 0.5)),
            ]
            boxes.append(self._clip_box_to_frame(b, frame))
        return boxes

    def _features_for_xyxy_robust(self, frame, bbox):
        """
        Mid-pass candidate fingerprint.

        Uses several jittered crops and keeps the median histogram. This reduces
        noise caused by drone shake, motion blur and small bbox shifts.
        """
        if not self.appearance_bank_enabled:
            return self._features_for_xyxy(frame, bbox)

        boxes = self._jitter_xyxy_boxes(
            frame,
            bbox,
            jitter_px=max(2, int(self.appearance_bank_jitter_px * 0.5)),
            scale_jitter=max(0.01, self.appearance_bank_scale_jitter * 0.5),
        )[: max(1, self.candidate_midpass_variants)]

        feats = []
        for b in boxes:
            f = self._features_for_xyxy(frame, b)
            if f is not None:
                feats.append(f.astype(np.float32))

        if not feats:
            return self._features_for_xyxy(frame, bbox)

        arr = np.stack(feats, axis=0)
        med = np.median(arr, axis=0).astype(np.float32)
        return cv2.normalize(med, med).flatten()

    def _build_target_feature_bank(self, frame, bbox):
        """
        Build a bank of possible target fingerprints from jittered versions of
        the initial target crop.
        """
        bank = []
        base = self._features_for_xyxy_robust(frame, bbox)
        if base is not None:
            bank.append(base.astype(np.float32))

        if self.appearance_bank_enabled:
            boxes = self._jitter_xyxy_boxes(
                frame,
                bbox,
                jitter_px=self.appearance_bank_jitter_px,
                scale_jitter=self.appearance_bank_scale_jitter,
            )
            for b in boxes:
                f = self._features_for_xyxy(frame, b)
                if f is not None:
                    bank.append(f.astype(np.float32))

        clean = []
        for f in bank:
            f = cv2.normalize(f, f).flatten().astype(np.float32)
            clean.append(f)

        return clean[: self.appearance_bank_max_size]

    def _midpass_similarity(self, similarities):
        if not similarities:
            return 0.0

        vals = np.asarray(similarities, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0

        vals.sort()
        n = vals.size
        lo = int(np.floor(self.midpass_low_quantile * (n - 1)))
        hi = int(np.ceil(self.midpass_high_quantile * (n - 1))) + 1
        lo = max(0, min(lo, n - 1))
        hi = max(lo + 1, min(hi, n))
        return float(np.mean(vals[lo:hi]))

    def _compare_to_feature_bank(self, candidate_feature):
        if candidate_feature is None:
            return 0.0

        if not self.target_feature_bank:
            return self.compare_features(self.target_features, candidate_feature)

        sims = [self.compare_features(f, candidate_feature) for f in self.target_feature_bank]
        return self._midpass_similarity(sims)

    def _maybe_update_feature_bank(self, frame, bbox, score: float):
        """
        Conservative online memory update. Only strong matches are allowed to
        enter the bank, so the target identity does not drift easily.
        """
        if not self.appearance_bank_enabled:
            return
        if score < self.appearance_bank_update_min_score:
            return

        self._match_update_counter += 1
        if self.appearance_bank_update_every_n_matches > 1:
            if self._match_update_counter % self.appearance_bank_update_every_n_matches != 0:
                return

        f = self._features_for_xyxy_robust(frame, bbox)
        if f is None:
            return

        self.target_feature_bank.append(f.astype(np.float32))

        if len(self.target_feature_bank) > self.appearance_bank_max_size:
            original = self.target_feature_bank[0]
            recent = self.target_feature_bank[-(self.appearance_bank_max_size - 1):]
            self.target_feature_bank = [original] + recent

    def get_features(self, frame, bbox_xywh):
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return None
        H, W = frame.shape[:2]
        x0 = int(np.clip(x, 0, W - 1))
        y0 = int(np.clip(y, 0, H - 1))
        x1 = int(np.clip(x + w, 0, W))
        y1 = int(np.clip(y + h, 0, H))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return cv2.normalize(hist, hist).flatten()

    def _features_for_xyxy(self, frame, b):
        return self.get_features(frame, [b[0], b[1], b[2] - b[0], b[3] - b[1]])

    def compare_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        val = float(cv2.compareHist(f1.astype(np.float32), f2.astype(np.float32), cv2.HISTCMP_CORREL))
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, -1.0, 1.0))

    def get_target_fingerprint(self):
        return None if self.target_features is None else self.target_features.copy()

    def set_target_fingerprint(self, fp):
        self.target_features = None if fp is None else fp.copy()
        self.target_feature_bank = [] if fp is None else [fp.copy()]

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)

    def _yaw_shift_px(self) -> float:
        if not self.yaw_enabled:
            return 0.0
        yaw_cmd = float(getattr(self, "last_yaw_rate_cmd_dps", 0.0) or 0.0)
        if abs(yaw_cmd) < self.yaw_deadband:
            return 0.0
        dx = self.yaw_shift_sign * yaw_cmd * self.yaw_pix_per_cmd
        return float(np.clip(dx, -self.yaw_max_shift_px, self.yaw_max_shift_px))

    def _shift_box_x(self, box: List[int], frame, dx: float) -> List[int]:
        if abs(dx) < 1e-6:
            return box.copy()
        return self._clip_box_to_frame([box[0] + dx, box[1], box[2] + dx, box[3]], frame)

    def _expected_bbox(self, frame) -> Optional[List[int]]:
        if self.last_good_bbox is None and self.last_bbox is None:
            return None
        base = self.last_good_bbox if self.last_good_bbox is not None else self.last_bbox
        dx = self._yaw_shift_px()
        if self.freeze_without_yaw and abs(dx) < 1e-6:
            return base.copy()
        return self._shift_box_x(base, frame, dx)

    def _make_search_window(self, frame, expected_bbox: List[int]) -> List[int]:
        cx, cy = self._bbox_center(expected_bbox)
        bw = max(1.0, float(expected_bbox[2] - expected_bbox[0]))
        bh = max(1.0, float(expected_bbox[3] - expected_bbox[1]))
        pad_x = max(self.search_min_pad_px, (self.search_scale - 1.0) * bw / 2.0)
        pad_y = max(self.search_min_pad_px, (self.search_scale - 1.0) * bh / 2.0)
        if abs(self._yaw_shift_px()) > 1e-6:
            pad_x += self.search_extra_yaw_pad_px
        return self._clip_box_to_frame(
            [int(round(cx - bw / 2.0 - pad_x)), int(round(cy - bh / 2.0 - pad_y)),
             int(round(cx + bw / 2.0 + pad_x)), int(round(cy + bh / 2.0 + pad_y))],
            frame,
        )

    def _run_yolo(self, frame, conf=None):
        conf = self.conf_match if conf is None else conf
        results = self.model.predict(frame, conf=conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        candidates = []
        if results.boxes is None or len(results.boxes) == 0:
            return candidates
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int).tolist()
            b = self._clip_box_to_frame(b, frame)
            cls_id = int(box.cls[0])
            det_conf = float(box.conf[0]) if box.conf is not None else 0.0
            candidates.append(Candidate(bbox=b, cls_id=cls_id, conf=det_conf))
        return candidates

    def _score_candidate(self, frame, cand: Candidate, expected_bbox: List[int], search_window: List[int]) -> Optional[Candidate]:
        if self.class_gate and self.target_class_id is not None and cand.cls_id != self.target_class_id:
            return None
        cx, cy = self._bbox_center(cand.bbox)
        if not self._center_inside(cx, cy, search_window):
            return None
        feat = self._features_for_xyxy_robust(frame, cand.bbox)
        feat_sim = self._compare_to_feature_bank(feat)
        if feat_sim < self.FEAT_MIN_TH:
            return None
        exp_cx, exp_cy = self._bbox_center(expected_bbox)
        sw = max(1.0, float(search_window[2] - search_window[0]))
        sh = max(1.0, float(search_window[3] - search_window[1]))
        dist_norm = math.hypot((cx - exp_cx) / sw, (cy - exp_cy) / sh)
        spatial_sim = float(np.exp(-dist_norm * 3.0))
        size_sim = 0.5
        if self.target_area is not None and self.target_area > 1.0:
            ratio = self._bbox_area(cand.bbox) / self.target_area
            size_sim = float(np.exp(-abs(math.log(max(ratio, 1e-6)))))
        if size_sim < self.size_min_score:
            return None
        aspect_sim = 0.5
        if self.target_aspect is not None and self.target_aspect > 1e-6:
            ratio = self._bbox_aspect(cand.bbox) / self.target_aspect
            aspect_sim = float(np.exp(-abs(math.log(max(ratio, 1e-6)))))
        if aspect_sim < self.aspect_min_score:
            return None
        score = 0.62 * feat_sim + 0.24 * spatial_sim + 0.06 * size_sim + 0.04 * aspect_sim + 0.04 * cand.conf
        cand.feat_sim = feat_sim
        cand.spatial_sim = spatial_sim
        cand.size_sim = size_sim
        cand.aspect_sim = aspect_sim
        cand.score = float(score)
        return cand

    def _choose_best(self, frame, candidates, expected_bbox, search_window) -> Optional[Candidate]:
        best = None
        for c in candidates:
            scored = self._score_candidate(frame, c, expected_bbox, search_window)
            if scored is None:
                continue
            if best is None or scored.score > best.score:
                best = scored
        return best

    def _score_reacquire_candidate(self, frame, cand: Candidate) -> Optional[Candidate]:
        """
        Full-frame identity-gated candidate scoring for active reacquisition.

        This is called only after several PRED frames. It does not replace the
        local search window during normal tracking.
        """
        if self.class_gate and self.target_class_id is not None and cand.cls_id != self.target_class_id:
            return None

        feat = self._features_for_xyxy_robust(frame, cand.bbox)
        feat_sim = self._compare_to_feature_bank(feat)
        if feat_sim < self.reacquire_feat_min:
            return None

        size_sim = 0.5
        if self.target_area is not None and self.target_area > 1.0:
            ratio = self._bbox_area(cand.bbox) / self.target_area
            size_sim = float(np.exp(-abs(math.log(max(ratio, 1e-6)))))
        if size_sim < self.size_min_score:
            return None

        aspect_sim = 0.5
        if self.target_aspect is not None and self.target_aspect > 1e-6:
            ratio = self._bbox_aspect(cand.bbox) / self.target_aspect
            aspect_sim = float(np.exp(-abs(math.log(max(ratio, 1e-6)))))
        if aspect_sim < self.aspect_min_score:
            return None

        # Mild image-center prior only. Identity remains dominant.
        h, w = frame.shape[:2]
        cx, cy = self._bbox_center(cand.bbox)
        dx = (cx - 0.5 * w) / max(1.0, 0.5 * w)
        dy = (cy - 0.5 * h) / max(1.0, 0.5 * h)
        center_prior = float(np.exp(-0.7 * (dx * dx + dy * dy)))

        cand.feat_sim = feat_sim
        cand.spatial_sim = center_prior
        cand.size_sim = size_sim
        cand.aspect_sim = aspect_sim
        cand.score = float(
            0.62 * feat_sim
            + 0.14 * size_sim
            + 0.08 * aspect_sim
            + 0.08 * cand.conf
            + 0.08 * center_prior
        )
        return cand

    def _active_reacquire(self, frame) -> Optional[Candidate]:
        """
        Controlled full-frame search.

        Use case:
            The car is visible in the frame, but the local bbox/search window is
            no longer near it due to a sharp drone movement.

        Safety:
            Reacquire is delayed and identity-gated, so it should not reintroduce
            constant target switching.
        """
        if not self.active_reacquire_enabled:
            return None

        if self.pred_frames < self.reacquire_after_pred_frames:
            return None

        if self.reacquire_every_n_frames > 1 and (self.pred_frames % self.reacquire_every_n_frames) != 0:
            return None

        candidates = self._run_yolo(frame, conf=self.conf_match)
        if not candidates:
            self.last_raw_mode = f"REACQUIRE_NO_DETECTIONS pred={self.pred_frames}"
            return None

        candidates = sorted(candidates, key=lambda c: c.conf, reverse=True)[: self.reacquire_max_candidates]

        best = None
        for cand in candidates:
            scored = self._score_reacquire_candidate(frame, cand)
            if scored is None:
                continue
            if best is None or scored.score > best.score:
                best = scored

        if best is not None and best.score >= self.reacquire_min_score:
            print(
                "[TRACKER] ACTIVE REACQUIRE "
                f"score={best.score:.3f} feat={best.feat_sim:.3f} "
                f"size={best.size_sim:.3f} aspect={best.aspect_sim:.3f} "
                f"conf={best.conf:.3f} bbox={best.bbox}"
            )
            return best

        if best is not None:
            self.last_raw_mode = f"REACQUIRE_REJECT score={best.score:.2f} feat={best.feat_sim:.2f}"
        else:
            self.last_raw_mode = "REACQUIRE_NO_IDENTITY_MATCH"

        return None

    def select_target_and_get_class(self, frame, x, y):
        """
        Click target selection.

        Important fix:
        In the clean one-car level, YOLO may sometimes miss the car exactly on the
        click frame or the user may click on an un-detected mesh part. The previous
        cleaned version returned None in that case, so the OpenCV click appeared to
        "do nothing".

        This version:
        1. Tries YOLO detection under the click.
        2. If none is under the click, chooses nearest YOLO detection.
        3. If YOLO returns no candidates at all, creates a small fallback bbox
           around the click and still locks a visual fingerprint.
        """
        candidates = self._run_yolo(frame, conf=self.conf_click)
        print(f"[TRACKER CLICK] x={x} y={y} candidates={len(candidates)}")

        selected = None
        for cand in candidates:
            b = cand.bbox
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                selected = cand
                break

        if selected is None and candidates:
            selected = min(
                candidates,
                key=lambda c: math.hypot(self._bbox_center(c.bbox)[0] - x, self._bbox_center(c.bbox)[1] - y),
            )

        if selected is None:
            # Last-resort manual crop lock. This prevents the UI from getting stuck
            # when YOLO misses on the exact click frame.
            radius = int(getattr(self.cfg, "click_fallback_radius_px", 45))
            b = self._clip_box_to_frame([x - radius, y - radius, x + radius, y + radius], frame)
            selected = Candidate(bbox=b, cls_id=None, conf=0.20)
            print(f"[TRACKER CLICK] YOLO missed. Using fallback click bbox={b}")

        b = selected.bbox
        self.target_class_id = selected.cls_id
        self.target_features = self._features_for_xyxy_robust(frame, b)
        self.target_feature_bank = self._build_target_feature_bank(frame, b)
        if self.target_features is None and self.target_feature_bank:
            self.target_features = self.target_feature_bank[0]
        print(f"[TRACKER] appearance bank size={len(self.target_feature_bank)}")
        self.target_area = self._bbox_area(b)
        self.target_aspect = self._bbox_aspect(b)

        cx, cy = self._bbox_center(b)
        self.kf.reset(cx, cy)

        self.last_bbox = b.copy()
        self.last_good_bbox = b.copy()
        self.STATE = "TRACK"
        self.last_mode = "MATCH"
        self.last_raw_mode = "CLICK_SELECT_LEGACY_SEARCH"
        self._prev_center = (cx, cy)
        self._vel_ema[:] = 0.0
        self.pred_frames = 0

        print(f"[TRACKER] LEGACY SEARCH LOCK(click) cid={self.target_class_id} bbox={self.last_bbox}")
        return self.target_class_id

    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        if self.target_features is None:
            return False
        expected = self._expected_bbox(frame)
        if expected is None:
            h, w = frame.shape[:2]
            expected = [w // 2 - 32, h // 2 - 32, w // 2 + 32, h // 2 + 32]
        search_window = self._make_search_window(frame, expected)
        candidates = self._run_yolo(frame, conf=self.conf_match)
        best = self._choose_best(frame, candidates, expected, search_window)
        if best is None or best.score < self.AUTOLOCK_TH:
            return False
        self._accept(frame, best, "AUTOLOCK_LEGACY_SEARCH")
        return True

    def update(self, frame):
        if self.STATE == "SEARCH":
            if self.auto_lock_on_fingerprint(frame, use_class_gate=True):
                return self.last_bbox
            self.last_mode = "NONE"
            self.last_raw_mode = "SEARCH"
            return None
        expected = self._expected_bbox(frame)
        if expected is None:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            self.last_raw_mode = "NO_EXPECTED_BBOX"
            return None
        search_window = self._make_search_window(frame, expected)
        candidates = self._run_yolo(frame, conf=self.conf_match)
        best = self._choose_best(frame, candidates, expected, search_window)
        if best is not None and best.score >= self.MATCH_TH:
            self._accept(frame, best, "MATCH_LEGACY_SEARCH")
            return self.last_bbox
        if self.last_bbox is None:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            self.last_raw_mode = "NO_LAST_BBOX"
            return None
        self.pred_frames += 1

        # After the local tracker has been in PRED for several frames, try a
        # controlled full-frame reacquisition. This lets the tracker actively find
        # the target again when the car is visible but no longer near the bbox.
        reacquired = self._active_reacquire(frame)
        if reacquired is not None:
            self._accept(frame, reacquired, "ACTIVE_REACQUIRE")
            return self.last_bbox

        if self.pred_frames > self.max_pred_frames:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            self.last_raw_mode = "PRED_TIMEOUT"
            print("[TRACKER] LOST: prediction timeout")
            return None

        self.last_bbox = expected.copy()
        self.last_mode = "PRED"
        dx = self._yaw_shift_px()
        if not str(self.last_raw_mode).startswith("REACQUIRE_"):
            self.last_raw_mode = f"PRED_LEGACY_YAW dx={dx:+.1f}" if abs(dx) > 1e-6 else "PRED_LEGACY_FREEZE"
        return self.last_bbox

    def _accept(self, frame, cand: Candidate, raw_mode: str):
        b = cand.bbox.copy()
        cx, cy = self._bbox_center(b)
        self.kf.update(cx, cy)
        if self._prev_center is not None:
            dv = np.array([cx - self._prev_center[0], cy - self._prev_center[1]], dtype=np.float32)
            self._vel_ema = 0.8 * self._vel_ema + 0.2 * dv
            self.kf.kf.statePost[2, 0] = self._vel_ema[0]
            self.kf.kf.statePost[3, 0] = self._vel_ema[1]
        self._prev_center = (cx, cy)
        self.last_bbox = b
        self.last_good_bbox = b.copy()
        self.last_mode = "MATCH"
        self.last_raw_mode = raw_mode
        self.STATE = "TRACK"
        self.pred_frames = 0
        area = self._bbox_area(b)
        aspect = self._bbox_aspect(b)
        self.target_area = area if self.target_area is None else 0.97 * self.target_area + 0.03 * area
        self.target_aspect = aspect if self.target_aspect is None else 0.97 * self.target_aspect + 0.03 * aspect
        feat = self._features_for_xyxy(frame, b)
        if feat is not None and self.target_features is not None:
            sim = self.compare_features(self.target_features, feat)
            if sim > float(self.cfg.feature_update_min):
                alpha = float(self.cfg.feature_update_alpha)
                mixed = (1.0 - alpha) * self.target_features + alpha * feat
                self.target_features = cv2.normalize(mixed, mixed).flatten()
        elif feat is not None:
            self.target_features = feat

    def draw(self, frame, fps=None):
        if self.last_good_bbox is not None:
            expected = self._expected_bbox(frame)
            if expected is not None:
                if self.draw_expected:
                    ex1, ey1, ex2, ey2 = map(int, expected)
                    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (180, 180, 180), 1)
                if self.draw_search_window:
                    sw = self._make_search_window(frame, expected)
                    sx1, sy1, sx2, sy2 = map(int, sw)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 180, 0), 1)
        if self.last_bbox is None:
            return frame
        x1, y1, x2, y2 = map(int, self.last_bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = (0, 255, 0) if self.last_mode == "MATCH" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"{self.last_mode}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"RAW: {self.last_raw_mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
        cv2.putText(frame, f"yaw={float(self.last_yaw_rate_cmd_dps):+.4f} pred={self.pred_frames}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        if fps is not None:
            cv2.putText(frame, f"FPS: {float(fps):.1f}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        return frame

    def close(self):
        pass
