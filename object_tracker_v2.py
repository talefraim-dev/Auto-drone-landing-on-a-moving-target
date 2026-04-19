# Baseline V2: Stable tracking + CNN/Histogram fusion + illumination robustness (low FPS)
# changes in algo core.

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from collections import deque


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
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.15

    def predict(self):
        return self.kf.predict()

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class tracker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11n.pt").to(self.device)

        self.kf = KalmanFilter()

        self.STATE = "SEARCH"
        self.target_class_id = None

        # feature memories
        self.target_hist_features = None
        self.target_cnn_features = None

        self.last_bbox = None

        # thresholds
        self.MATCH_TH = 0.50
        self.AUTOLOCK_TH = 0.38
        self.UPDATE_TH = 0.72

        self.last_mode = "NONE"

        self._prev_center = None
        self._vel_ema = np.zeros(2, dtype=np.float32)

        # Ego-motion compensation
        self.last_yaw_rate_cmd_dps = 0.0
        self.YAW_PIX_PER_DPS = 0.12

        # CNN extractor
        self.cnn = CNNFeatureExtractor().to(self.device).eval()
        self.cnn_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        # signal + debug
        self.last_hist_sim = 0.0
        self.last_cnn_sim = 0.0
        self.last_spatial_sim = 0.0
        self.last_score = 0.0
        self.signal_history = deque(maxlen=300)

        # weighted fusion
        self.W_HIST = 0.25
        self.W_CNN = 0.55
        self.W_SPATIAL = 0.20

    # -------------------------------------------------
    # Preprocess for CNN robustness
    # -------------------------------------------------
    def _preprocess_for_cnn(self, roi):
        if roi is None or roi.size == 0:
            return None

        # LAB normalize illumination
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        roi = cv2.merge([y, cr, cb])

        return roi

    # -------------------------------------------------
    # Histogram features
    # -------------------------------------------------
    def get_hist_features(self, frame, bbox_xywh):
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

    def compare_hist_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        v = float(cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL))
        return float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))

    # -------------------------------------------------
    # CNN features
    # -------------------------------------------------
    def get_cnn_features(self, frame, bbox_xywh):
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

        roi = self._preprocess_for_cnn(roi)
        if roi is None:
            return None

        t = self.cnn_transform(roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.cnn(t)
            feat = torch.nn.functional.normalize(feat, dim=1)

        return feat.squeeze(0).cpu().numpy()

    def compare_cnn_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        sim = float(np.dot(f1, f2))
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    # -------------------------------------------------
    # Fusion
    # -------------------------------------------------
    def _compute_fused_score(self, hist_sim, cnn_sim, spatial_sim):
        score = (
            self.W_HIST * hist_sim +
            self.W_CNN * cnn_sim +
            self.W_SPATIAL * spatial_sim
        )
        return float(np.clip(score, 0.0, 1.0))

    def _update_signal(self, score, hist_sim, cnn_sim, spatial_sim):
        self.last_score = score
        self.last_hist_sim = hist_sim
        self.last_cnn_sim = cnn_sim
        self.last_spatial_sim = spatial_sim
        self.signal_history.append(score)

    def get_signal_value(self):
        return self.last_score

    def get_signal_history(self):
        return list(self.signal_history)

    # -------------------------------------------------
    # Memory API
    # -------------------------------------------------
    def get_target_fingerprint(self):
        return {
            "hist": None if self.target_hist_features is None else self.target_hist_features.copy(),
            "cnn": None if self.target_cnn_features is None else self.target_cnn_features.copy(),
            "class_id": self.target_class_id
        }

    def set_target_fingerprint(self, fp):
        if fp is None:
            self.target_hist_features = None
            self.target_cnn_features = None
            self.target_class_id = None
            return

        self.target_hist_features = None if fp.get("hist") is None else fp["hist"].copy()
        self.target_cnn_features = None if fp.get("cnn") is None else fp["cnn"].copy()
        self.target_class_id = fp.get("class_id", None)

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)

    # -------------------------------------------------
    # IMPORTANT: env depends on it
    # -------------------------------------------------
    def select_target_and_get_class(self, frame, x, y):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return None

        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                self.target_class_id = int(box.cls[0])
                roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]

                self.target_hist_features = self.get_hist_features(frame, roi)
                self.target_cnn_features = self.get_cnn_features(frame, roi)

                cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
                self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

                self.last_bbox = b.tolist()
                self.STATE = "TRACK"
                self.last_mode = "MATCH"
                self._prev_center = (cx, cy)
                self._vel_ema[:] = 0.0

                self._update_signal(1.0, 1.0, 1.0, 1.0)

                print(f"[TRACKER_V2] LOCKED(click) cid={self.target_class_id}")
                return self.target_class_id

        return None

    # -------------------------------------------------
    # Re-lock using detector + fused features
    # -------------------------------------------------
    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        if self.target_hist_features is None and self.target_cnn_features is None:
            return False

        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return False

        h, w = frame.shape[:2]
        pred_cx, pred_cy = w / 2, h / 2

        best_box, best_score = None, -1.0
        best_hist_sim, best_cnn_sim, best_spatial_sim = 0.0, 0.0, 0.0

        for box in results.boxes:
            if use_class_gate and self.target_class_id is not None:
                if int(box.cls[0]) != self.target_class_id:
                    continue

            b = box.xyxy[0].cpu().numpy().astype(int)
            roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]

            hist_feat = self.get_hist_features(frame, roi)
            cnn_feat = self.get_cnn_features(frame, roi)

            hist_sim = self.compare_hist_features(self.target_hist_features, hist_feat)
            cnn_sim = self.compare_cnn_features(self.target_cnn_features, cnn_feat)

            cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
            dist = np.hypot(cx - pred_cx, cy - pred_cy)
            spatial_sim = float(np.exp(-dist / 100.0))

            score = self._compute_fused_score(hist_sim, cnn_sim, spatial_sim)

            if score > best_score:
                best_score = score
                best_box = b
                best_hist_sim = hist_sim
                best_cnn_sim = cnn_sim
                best_spatial_sim = spatial_sim

        self._update_signal(best_score if best_score >= 0 else 0.0,
                            best_hist_sim, best_cnn_sim, best_spatial_sim)

        if best_box is None or best_score < self.AUTOLOCK_TH:
            return False

        cx = (best_box[0] + best_box[2]) / 2
        cy = (best_box[1] + best_box[3]) / 2
        self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        self.last_bbox = best_box.tolist()
        self.STATE = "TRACK"
        self.last_mode = "MATCH"
        self._prev_center = (cx, cy)
        self._vel_ema[:] = 0.0

        print("[TRACKER_V2] RELOCK(auto)")
        return True

    # -------------------------------------------------
    # EMA update for target identity (strong matches only)
    # -------------------------------------------------
    def _update_target_memory(self, hist_feat, cnn_feat, alpha=0.08):
        if hist_feat is not None and self.target_hist_features is not None:
            self.target_hist_features = (
                (1.0 - alpha) * self.target_hist_features + alpha * hist_feat
            )

        if cnn_feat is not None and self.target_cnn_features is not None:
            mixed = (1.0 - alpha) * self.target_cnn_features + alpha * cnn_feat
            norm = np.linalg.norm(mixed) + 1e-6
            self.target_cnn_features = mixed / norm

    # -------------------------------------------------
    def update(self, frame):
        if self.STATE == "SEARCH":
            if self.auto_lock_on_fingerprint(frame, use_class_gate=True):
                return self.last_bbox
            self.last_mode = "NONE"
            self._update_signal(0.0, 0.0, 0.0, 0.0)
            return None

        predicted = self.kf.predict()
        pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])

        results = self.model.predict(frame, conf=0.25, verbose=False)[0]

        best_box, best_score = None, -1.0
        best_hist_feat, best_cnn_feat = None, None
        best_hist_sim, best_cnn_sim, best_spatial_sim = 0.0, 0.0, 0.0

        if results.boxes:
            for box in results.boxes:
                if self.target_class_id is not None and int(box.cls[0]) != self.target_class_id:
                    continue

                b = box.xyxy[0].cpu().numpy().astype(int)
                roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]

                hist_feat = self.get_hist_features(frame, roi)
                cnn_feat = self.get_cnn_features(frame, roi)

                hist_sim = self.compare_hist_features(
                    self.target_hist_features, hist_feat
                )
                cnn_sim = self.compare_cnn_features(
                    self.target_cnn_features, cnn_feat
                )

                cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
                dist = np.hypot(cx - pred_cx, cy - pred_cy)
                spatial_sim = float(np.exp(-dist / 100.0))

                score = self._compute_fused_score(hist_sim, cnn_sim, spatial_sim)

                if score > best_score and score > self.MATCH_TH:
                    best_score = score
                    best_box = b
                    best_hist_feat = hist_feat
                    best_cnn_feat = cnn_feat
                    best_hist_sim = hist_sim
                    best_cnn_sim = cnn_sim
                    best_spatial_sim = spatial_sim

        if best_box is not None:
            cx = (best_box[0] + best_box[2]) / 2
            cy = (best_box[1] + best_box[3]) / 2
            self.kf.update(cx, cy)

            if self._prev_center is not None:
                dv = np.array([cx - self._prev_center[0], cy - self._prev_center[1]])
                self._vel_ema = 0.8 * self._vel_ema + 0.2 * dv
                self.kf.kf.statePost[2, 0] = self._vel_ema[0]
                self.kf.kf.statePost[3, 0] = self._vel_ema[1]

            self._prev_center = (cx, cy)
            self.last_bbox = best_box.tolist()
            self.last_mode = "MATCH"

            if best_score >= self.UPDATE_TH:
                self._update_target_memory(best_hist_feat, best_cnn_feat, alpha=0.08)

            self._update_signal(best_score, best_hist_sim, best_cnn_sim, best_spatial_sim)
            return self.last_bbox

        # No match => PRED
        if self.last_bbox is None:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            self._update_signal(0.0, 0.0, 0.0, 0.0)
            return None

        pred_cx += -float(self.last_yaw_rate_cmd_dps) * float(self.YAW_PIX_PER_DPS)

        x1, y1, x2, y2 = self.last_bbox
        w, h = x2 - x1, y2 - y1

        self.last_bbox = [
            int(pred_cx - w / 2), int(pred_cy - h / 2),
            int(pred_cx + w / 2), int(pred_cy + h / 2)
        ]

        self.last_mode = "PRED"
        self._update_signal(max(self.last_score * 0.85, 0.0), 0.0, 0.0, 0.0)
        return self.last_bbox

    # -------------------------------------------------
    def draw(self, frame, fps):
        if self.last_bbox is None:
            return

        x1, y1, x2, y2 = self.last_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        color = (0, 255, 0) if self.last_mode == "MATCH" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        cv2.putText(frame, f"STATE: {self.STATE}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MODE: {self.last_mode}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"SCORE: {self.last_score:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"H:{self.last_hist_sim:.2f} C:{self.last_cnn_sim:.2f} S:{self.last_spatial_sim:.2f}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)