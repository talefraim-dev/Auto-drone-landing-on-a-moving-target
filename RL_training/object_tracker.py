import cv2
import numpy as np
import torch
from ultralytics import YOLO


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
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
        self.target_class_id = None
        self.target_features = None
        self.last_bbox = None

        self.MATCH_TH = 0.40 # TODO: raise the TH along the training
        self.AUTOLOCK_TH = 0.35

        self.last_mode = "NONE"

        self._prev_center = None
        self._vel_ema = np.zeros(2, dtype=np.float32)

    # -----------------------------
    def get_features(self, frame, bbox_xywh):
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return None
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return cv2.normalize(hist, hist).flatten()

    def compare_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL))

    # -----------------------------
    def get_target_fingerprint(self):
        return None if self.target_features is None else self.target_features.copy()

    def set_target_fingerprint(self, fp):
        self.target_features = None if fp is None else fp.copy()

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)

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
                self._prev_center = (cx, cy)
                self._vel_ema[:] = 0.0

                print(f"[TRACKER] LOCKED(click) cid={self.target_class_id}")
                return self.target_class_id
        return None

    # -----------------------------
    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        if self.target_features is None:
            return False

        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return False

        h, w = frame.shape[:2]
        pred_cx, pred_cy = w / 2, h / 2

        best_box, best_score = None, -1.0

        for box in results.boxes:
            if use_class_gate and self.target_class_id is not None:
                if int(box.cls[0]) != self.target_class_id:
                    continue

            b = box.xyxy[0].cpu().numpy().astype(int)
            roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            feat = self.get_features(frame, roi)
            feat_sim = self.compare_features(self.target_features, feat)

            cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
            dist = np.hypot(cx - pred_cx, cy - pred_cy)
            spatial_sim = np.exp(-dist / 100.0)

            score = feat_sim * 0.7 + spatial_sim * 0.3
            if score > best_score:
                best_score, best_box = score, b

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

        print("[TRACKER] RELOCK(auto)")
        return True

    # -----------------------------
    def update(self, frame):
        if self.STATE == "SEARCH":
            if self.auto_lock_on_fingerprint(frame, use_class_gate=True):
                return self.last_bbox
            self.last_mode = "NONE"
            return None

        predicted = self.kf.predict()
        pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])

        results = self.model.predict(frame, conf=0.25, verbose=False)[0]

        best_box, best_score = None, -1.0

        if results.boxes:
            for box in results.boxes:
                if int(box.cls[0]) != self.target_class_id:
                    continue

                b = box.xyxy[0].cpu().numpy().astype(int)
                roi = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                feat_sim = self.compare_features(
                    self.target_features,
                    self.get_features(frame, roi)
                )

                cx, cy = b[0] + roi[2] / 2, b[1] + roi[3] / 2
                dist = np.hypot(cx - pred_cx, cy - pred_cy)
                spatial_sim = np.exp(-dist / 100.0)

                score = feat_sim * 0.7 + spatial_sim * 0.3
                if score > best_score and score > self.MATCH_TH:
                    best_score, best_box = score, b

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
            return self.last_bbox

        if self.last_bbox is None:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            return None

        x1, y1, x2, y2 = self.last_bbox
        w, h = x2 - x1, y2 - y1
        self.last_bbox = [
            int(pred_cx - w / 2), int(pred_cy - h / 2),
            int(pred_cx + w / 2), int(pred_cy + h / 2)
        ]

        self.last_mode = "PRED"
        return self.last_bbox

    # -----------------------------
    def draw(self, frame, fps):
        if self.last_bbox is None:
            return
        x1, y1, x2, y2 = self.last_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        color = (0, 255, 0) if self.last_mode == "MATCH" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
