"""
Author:   Tal Efraim
Version:  POC
Abstract: YOLO v11n, Kalman filter and deep embedding
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        return self.kf.predict()

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))


class tracker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11n.pt").to(self.device)

        self.kf = KalmanFilter()

        self.STATE = "SEARCH"
        self.target_class_id = None      # optional gate
        self.target_features = None      # embedding fingerprint
        self.last_bbox = None            # [x1,y1,x2,y2] ints

        # Thresholds (keep your logic; only tune if needed)
        self.MATCH_TH = 0.5
        self.AUTOLOCK_TH = 0.35

        # Debug telemetry
        self.last_mode = "NONE"          # MATCH / PRED / NONE
        self.last_score = 0.0
        self.last_feat_sim = 0.0
        self.last_spatial_sim = 0.0
        self.last_dist = 0.0

        # For making PRED "alive" (does NOT change matching logic)
        self._prev_center = None         # (cx, cy)
        self._vel_ema = np.array([0.0, 0.0], dtype=np.float32)  # vx, vy

    # -----------------------------
    # Embedding
    # -----------------------------
    def get_features(self, frame, bbox_xywh):
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return None
        roi = frame[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            return None
        hist = cv2.calcHist([roi], [0, 1, 2], None,
                            [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def compare_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL))

    # -----------------------------
    # Persistence API (reset by embedding)
    # -----------------------------
    def get_target_fingerprint(self):
        return None if self.target_features is None else self.target_features.copy()

    def set_target_fingerprint(self, fp):
        self.target_features = None if fp is None else fp.copy()

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)

    # -----------------------------
    # Manual click lock
    # -----------------------------
    def select_target_and_get_class(self, frame, x, y):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return None

        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                self.target_class_id = int(box.cls[0])

                roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                self.target_features = self.get_features(frame, roi)

                cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
                self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

                self.last_bbox = b.tolist()
                self.STATE = "TRACK"

                self.last_mode = "MATCH"
                self.last_score = 1.0
                self.last_feat_sim = 1.0
                self.last_spatial_sim = 1.0
                self.last_dist = 0.0

                self._prev_center = (float(cx), float(cy))
                self._vel_ema[:] = 0.0

                print(f"[TRACKER] LOCKED(click): {self.model.names[self.target_class_id].upper()}  cid={self.target_class_id}")
                return self.target_class_id

        return None

    # -----------------------------
    # Auto-lock by embedding (reset relock)
    # -----------------------------
    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        if self.target_features is None:
            return False

        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return False

        pred_cx, pred_cy = frame.shape[1] / 2, frame.shape[0] / 2
        best_box, best_score = None, -1.0
        best_feat, best_spatial, best_dist = 0.0, 0.0, 0.0

        for box in results.boxes:
            if use_class_gate and self.target_class_id is not None:
                if int(box.cls[0]) != self.target_class_id:
                    continue

            b = box.xyxy[0].cpu().numpy().astype(int)
            roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]

            feat = self.get_features(frame, roi)
            feat_sim = self.compare_features(self.target_features, feat)

            cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
            dist = float(np.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2))
            spatial_sim = float(np.exp(-dist / 100.0))

            score = feat_sim * 0.7 + spatial_sim * 0.3

            if score > best_score:
                best_score = score
                best_box = b
                best_feat, best_spatial, best_dist = feat_sim, spatial_sim, dist

        if best_box is None or best_score < self.AUTOLOCK_TH:
            self.last_mode = "NONE"
            self.last_score = 0.0
            return False

        roi = [best_box[0], best_box[1],
               best_box[2] - best_box[0],
               best_box[3] - best_box[1]]

        cx, cy = best_box[0] + roi[2] / 2, best_box[1] + roi[3] / 2
        self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        self.last_bbox = best_box.tolist()
        self.STATE = "TRACK"

        self.last_mode = "MATCH"
        self.last_score = float(best_score)
        self.last_feat_sim = float(best_feat)
        self.last_spatial_sim = float(best_spatial)
        self.last_dist = float(best_dist)

        self._prev_center = (float(cx), float(cy))
        self._vel_ema[:] = 0.0

        print(f"[TRACKER] RELOCK(auto): OK  gate={int(use_class_gate)}  score={self.last_score:.2f}  cid={self.target_class_id}")
        return True

    # -----------------------------
    # Update loop (YOUR logic)
    # -----------------------------
    def update(self, frame):
        if self.STATE == "SEARCH":
            self.last_mode = "NONE"
            self.last_score = 0.0
            return None

        predicted = self.kf.predict()
        pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])

        results = self.model.predict(frame, conf=0.25, verbose=False)[0]

        best_box, best_score = None, -1.0
        best_feat, best_spatial, best_dist = 0.0, 0.0, 0.0

        if results.boxes:
            for box in results.boxes:
                if self.target_class_id is None:
                    continue
                if int(box.cls[0]) != self.target_class_id:
                    continue

                b = box.xyxy[0].cpu().numpy().astype(int)
                roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]

                current_feat = self.get_features(frame, roi)
                feat_sim = self.compare_features(self.target_features, current_feat)

                cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
                dist = float(np.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2))
                spatial_sim = float(np.exp(-dist / 100.0))

                score = feat_sim * 0.7 + spatial_sim * 0.3

                if score > best_score and score > self.MATCH_TH:
                    best_score = score
                    best_box = b
                    best_feat, best_spatial, best_dist = feat_sim, spatial_sim, dist

        if best_box is not None:
            # Correct KF
            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2
            self.kf.update(cx, cy)

            # Update vx/vy estimate (makes PRED move naturally; does NOT change matching)
            if self._prev_center is not None:
                dvx = float(cx - self._prev_center[0])
                dvy = float(cy - self._prev_center[1])
                self._vel_ema = 0.8 * self._vel_ema + 0.2 * np.array([dvx, dvy], dtype=np.float32)
                # Inject into KF statePost velocities
                self.kf.kf.statePost[2, 0] = np.float32(self._vel_ema[0])
                self.kf.kf.statePost[3, 0] = np.float32(self._vel_ema[1])

            self._prev_center = (float(cx), float(cy))

            self.last_bbox = best_box.tolist()

            # Smoothly update visual fingerprint (your EMA)
            new_feat = self.get_features(
                frame,
                [best_box[0], best_box[1],
                 best_box[2] - best_box[0],
                 best_box[3] - best_box[1]]
            )
            if self.target_features is not None and new_feat is not None:
                self.target_features = self.target_features * 0.9 + new_feat * 0.1

            self.last_mode = "MATCH"
            self.last_score = float(best_score)
            self.last_feat_sim = float(best_feat)
            self.last_spatial_sim = float(best_spatial)
            self.last_dist = float(best_dist)
            return self.last_bbox

        # No match -> prediction bbox (PRED mode)
        if self.last_bbox is None:
            self.last_mode = "NONE"
            self.last_score = 0.0
            return None

        # If KF has some velocity, pred will move -> bbox won't be static
        x1, y1, x2, y2 = self.last_bbox
        w, h = (x2 - x1), (y2 - y1)
        self.last_bbox = [
            int(pred_cx - w / 2), int(pred_cy - h / 2),
            int(pred_cx + w / 2), int(pred_cy + h / 2)
        ]

        self.last_mode = "PRED"
        self.last_score = 0.0
        self.last_feat_sim = 0.0
        self.last_spatial_sim = 0.0
        self.last_dist = 0.0
        return self.last_bbox

    # -----------------------------
    # HUD
    # -----------------------------
    def draw(self, frame, fps):
        target_set = int(self.target_features is not None)

        if self.last_mode == "MATCH":
            mode_txt = "MATCH"
            color = (0, 255, 0)
        elif self.last_mode == "PRED":
            mode_txt = "PRED"
            color = (0, 255, 255)  # yellow
        else:
            mode_txt = "SEARCH"
            color = (0, 0, 255)

        cid_txt = "None" if self.target_class_id is None else str(self.target_class_id)

        cv2.putText(frame,
                    f"TARGET_SET={target_set} | STATE={self.STATE} | MODE={mode_txt} | CID={cid_txt} | FPS={fps}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame,
                    f"SCORE: {self.last_score:.2f}  FEAT: {self.last_feat_sim:.2f}  SPAT: {self.last_spatial_sim:.2f}  D: {self.last_dist:.1f}",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.STATE == "TRACK" and self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Tactical Corner Brackets
            s, t = 20, 3
            cv2.line(frame, (x1, y1), (x1 + s, y1), color, t)
            cv2.line(frame, (x1, y1), (x1, y1 + s), color, t)
            cv2.line(frame, (x2, y1), (x2 - s, y1), color, t)
            cv2.line(frame, (x2, y1), (x2, y1 + s), color, t)
            cv2.line(frame, (x1, y2), (x1 + s, y2), color, t)
            cv2.line(frame, (x1, y2), (x1, y2 - s), color, t)
            cv2.line(frame, (x2, y2), (x2 - s, y2), color, t)
            cv2.line(frame, (x2, y2), (x2, y2 - s), color, t)

            # Crosshair
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), color, 2)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), color, 2)

            # Offset readout
            dx, dy = cx - (frame.shape[1] // 2), cy - (frame.shape[0] // 2)
            cv2.putText(frame, f"REL X: {dx} Y: {dy}",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
