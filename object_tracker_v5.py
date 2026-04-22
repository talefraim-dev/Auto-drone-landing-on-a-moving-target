import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import json
import time

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    models = None
    transforms = None


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


if TORCH_AVAILABLE:
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
else:
    class CNNFeatureExtractor:
        def __init__(self):
            pass


class tracker:
    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model = YOLO("yolo11n.pt")
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        self.kf = KalmanFilter()

        self.STATE = "SEARCH"
        self.target_class_id = None
        self.last_bbox = None
        self._last_match_bbox = None  # last confirmed detection (prevents drift/spiral)
        self._last_size_wh = None  # (w, h) for prediction bbox
        self.last_mode = "NONE"
        self._prev_state = self.STATE

        self._prev_center = None
        self._vel_ema = np.zeros(2, dtype=np.float32)

        self.last_yaw_rate_cmd_dps = 0.0
        self.YAW_PIX_PER_DPS = 0.12

        self.cnn = None
        self.cnn_transform = None
        if TORCH_AVAILABLE:
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

        # identity memory
        self.target_hist_features = None
        self.target_cnn_features = None

        self.pos_hist_bank = deque(maxlen=10)
        self.pos_cnn_bank = deque(maxlen=10)

        self.neg_hist_bank = deque(maxlen=14)
        self.neg_cnn_bank = deque(maxlen=14)

        # identical-object stability helpers
        self.CROP_PAD_RATIO = 0.18
        self.SPATIAL_DIST_SCALE_PX = 110.0
        self.SPATIAL_IOU_WEIGHT = 0.75  # more "stickiness" on same instance
        self.MIN_MATCH_IOU = 0.12       # hard gate (when last_bbox exists)
        self.MIN_MATCH_SPATIAL = 0.22   # hard gate (prevents jumps)
        self.MAX_CENTER_JUMP_FACTOR = 1.2  # max center jump in units of bbox diag (prevents close-car swaps)
        # Recovery must be stricter to avoid hijacking a neighboring identical object
        self.RECOVER_MAX_CENTER_JUMP_FACTOR = 0.65
        self.RECOVER_MIN_IOU = 0.05

        # thresholds (TRACK)
        self.MATCH_TH = 0.44
        self.UPDATE_TH = 0.74
        self.IDENTITY_MARGIN_TH = 0.10

        # thresholds (RECOVERY)
        self.RECOVER_TH = 0.42   # יותר סלחני
        self.RECOVER_STRICT_TH = 0.50

        # timing / recovery
        self.lost_frames = 0
        self.max_pred_frames = 18
        self.frame_count = 0
        self.recovery_interval = 4
        self._reacquire_streak = 0
        self.REACQUIRE_MIN_STREAK = 3
        self.REACQUIRE_LOCAL_ONLY_FRAMES = 60  # while within this horizon, never "hop lanes" via global search

        # performance knobs
        self.DETECT_EVERY_N = 1           # run YOLO once every N frames (stability first)
        self.FEATURE_EVERY_N = 2          # compute hist/cnn once every N detect-frames per candidate
        self.MAX_CANDIDATES = 3           # compute features only for top-K spatial candidates
        self.SEARCH_REGION_SCALE = 3.0    # crop around predicted bbox (bigger = more robust)
        self.YOLO_IMGSZ = 416
        self.YOLO_CONF_TRACK = 0.25
        self.YOLO_CONF_SELECT = 0.30
        self._use_half = (TORCH_AVAILABLE and self.device == "cuda")

        # signal + debug
        self.last_hist_sim = 0.0
        self.last_cnn_sim = 0.0
        self.last_spatial_sim = 0.0
        self.last_neg_penalty = 0.0
        self.last_score = 0.0
        self.signal_history = deque(maxlen=500)

        # weights (fallback if torch is not usable)
        self.W_HIST = 0.28 if TORCH_AVAILABLE else 0.55
        self.W_CNN = 0.60 if TORCH_AVAILABLE else 0.0
        self.W_SPATIAL = 0.32 if TORCH_AVAILABLE else 0.45

        # --- debug logging (NDJSON) ---
        self._dbg_enabled = True
        self._dbg_log_path = "debug-1e6537.log"
        self._dbg_session_id = "1e6537"
        self._dbg_run_id = f"run_{int(time.time())}"
        self._dbg_throttle_n = 15

    # ---------------------------
    def _dbg(self, hypothesis_id: str, location: str, message: str, data: dict | None = None):
        if not self._dbg_enabled:
            return
        payload = {
            "sessionId": self._dbg_session_id,
            "runId": self._dbg_run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        try:
            # #region agent log
            with open(self._dbg_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            # #endregion agent log
        except Exception:
            # never crash tracking due to logging
            pass

    @staticmethod
    def _clip_bbox_xyxy(b, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        # Ensure non-degenerate box even at borders
        if x2 <= x1:
            x1 = max(0, min(x1, w - 2))
            x2 = x1 + 1
        if y2 <= y1:
            y1 = max(0, min(y1, h - 2))
            y2 = y1 + 1
        return [x1, y1, x2, y2]

    def _bbox_from_center(self, cx, cy):
        if self._last_size_wh is None:
            return None
        w, h = self._last_size_wh
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))
        return [x1, y1, x2, y2]

    @staticmethod
    def _xywh_to_xyxy(b):
        x, y, w, h = b
        return [x, y, x + w, y + h]

    @staticmethod
    def _iou_xyxy(a, b):
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0

    def _padded_crop(self, frame, bbox_xywh):
        x, y, w, h = bbox_xywh
        if w <= 1 or h <= 1:
            return None
        pad_x = int(round(w * self.CROP_PAD_RATIO))
        pad_y = int(round(h * self.CROP_PAD_RATIO))

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size != 0 else None

    def _crop_search_region(self, frame, pred_bbox_xyxy, lost_frames: int = 0):
        if pred_bbox_xyxy is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = pred_bbox_xyxy
        w = max(2, x2 - x1)
        h = max(2, y2 - y1)
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))

        # Expand search region as we stay lost (Kalman drift / occlusion robustness)
        scale = float(min(self.SEARCH_REGION_SCALE * (1.0 + 0.10 * max(0, lost_frames)), 6.0))
        sw = int(round(w * scale))
        sh = int(round(h * scale))
        rx1 = max(0, cx - sw // 2)
        ry1 = max(0, cy - sh // 2)
        rx2 = min(frame.shape[1], cx + sw // 2)
        ry2 = min(frame.shape[0], cy + sh // 2)
        if rx2 <= rx1 or ry2 <= ry1:
            return frame, (0, 0)
        return frame[ry1:ry2, rx1:rx2], (rx1, ry1)

    @staticmethod
    def _center_xyxy(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _within_center_jump(self, candidate_xyxy, ref_xyxy):
        if candidate_xyxy is None or ref_xyxy is None:
            return True
        cx, cy = self._center_xyxy(candidate_xyxy)
        rx, ry = self._center_xyxy(ref_xyxy)
        w = max(2.0, ref_xyxy[2] - ref_xyxy[0])
        h = max(2.0, ref_xyxy[3] - ref_xyxy[1])
        diag = float(np.hypot(w, h))
        return float(np.hypot(cx - rx, cy - ry)) <= (self.MAX_CENTER_JUMP_FACTOR * diag)

    def _within_center_jump_recover(self, candidate_xyxy, ref_xyxy):
        if candidate_xyxy is None or ref_xyxy is None:
            return True
        cx, cy = self._center_xyxy(candidate_xyxy)
        rx, ry = self._center_xyxy(ref_xyxy)
        w = max(2.0, ref_xyxy[2] - ref_xyxy[0])
        h = max(2.0, ref_xyxy[3] - ref_xyxy[1])
        diag = float(np.hypot(w, h))
        return float(np.hypot(cx - rx, cy - ry)) <= (self.RECOVER_MAX_CENTER_JUMP_FACTOR * diag)

    def _yolo_predict(self, img, conf, classes=None):
        # ultralytics predict accepts BGR numpy images directly
        return self.model.predict(
            img,
            conf=conf,
            imgsz=self.YOLO_IMGSZ,
            classes=classes,
            half=self._use_half,
            verbose=False
        )[0]

    def _preprocess_for_cnn(self, roi):
        if roi is None or roi.size == 0:
            return None
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        roi = cv2.merge([y, cr, cb])
        return roi

    # ---------------------------
    def get_hist_features(self, frame, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        roi = self._padded_crop(frame, bbox)
        if roi is None:
            return None
        hist = cv2.calcHist([roi], [0,1,2], None, [8,8,8], [0,256]*3)
        return cv2.normalize(hist, hist).flatten()

    def compare_hist(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        v = cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
        return float(np.clip((v+1)/2, 0, 1))

    def hist_bank(self, f):
        if f is None or not self.pos_hist_bank:
            return 0.0
        return max(self.compare_hist(f, b) for b in self.pos_hist_bank)

    # ---------------------------
    def get_cnn_features(self, frame, bbox):
        if not TORCH_AVAILABLE or self.cnn is None or self.cnn_transform is None:
            return None
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        roi = self._padded_crop(frame, bbox)
        if roi is None:
            return None
        roi = self._preprocess_for_cnn(roi)
        t = self.cnn_transform(roi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.cnn(t)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy()

    def cnn_bank(self, f):
        if f is None or not self.pos_cnn_bank:
            return 0.0
        return max(np.dot(f, b) for b in self.pos_cnn_bank)

    def _neg_penalty(self, hist_f, cnn_f):
        neg_hist = 0.0
        neg_cnn = 0.0
        if hist_f is not None and self.neg_hist_bank:
            neg_hist = max(self.compare_hist(hist_f, b) for b in self.neg_hist_bank)
        if cnn_f is not None and self.neg_cnn_bank:
            neg_cnn = max(float(np.dot(cnn_f, b)) for b in self.neg_cnn_bank)
        return float(max(neg_hist, neg_cnn))

    # ---------------------------
    def _score(self, hist_pos, cnn_pos, spatial, neg_penalty):
        return np.clip(
            self.W_HIST*hist_pos +
            self.W_CNN*cnn_pos +
            self.W_SPATIAL*spatial -
            0.28*neg_penalty,
            0, 1
        )

    # ---------------------------
    def _recovery_score(self, hist_pos, cnn_pos):
        # 🔥 בלי spatial ובלי negative כמעט
        return np.clip(0.4*hist_pos + 0.6*cnn_pos, 0, 1)

    # ---------------------------
    def select_target_and_get_class(self, frame, x, y):
        res = self._yolo_predict(frame, conf=self.YOLO_CONF_SELECT, classes=None)
        if not res.boxes:
            return None

        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                self.target_class_id = int(box.cls[0])

                roi = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                h = self.get_hist_features(frame, roi)
                c = self.get_cnn_features(frame, roi)

                self.pos_hist_bank.clear()
                self.pos_cnn_bank.clear()
                self.neg_hist_bank.clear()
                self.neg_cnn_bank.clear()

                self.pos_hist_bank.append(h)
                self.pos_cnn_bank.append(c)

                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                self.kf.kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)

                self.last_bbox = b.tolist()
                self._last_match_bbox = self.last_bbox.copy()
                self._last_size_wh = (int(b[2] - b[0]), int(b[3] - b[1]))
                self.STATE = "TRACK"
                self.last_mode = "MATCH"
                self._dbg(
                    "H1",
                    "object_tracker_v5.py:select_target_and_get_class",
                    "target_selected",
                    {
                        "class_id": self.target_class_id,
                        "bbox": self.last_bbox,
                        "torch": TORCH_AVAILABLE,
                        "device": self.device,
                    },
                )

                return self.target_class_id
        return None

    # ---------------------------
    def _try_recovery(self, frame, pred_bbox=None, strict=False):
        # Recovery should not "jump" across identical objects:
        # - restrict search to a region near prediction (when available)
        # - require spatial agreement and multi-frame confirmation
        # IMPORTANT: do NOT call Kalman predict() here; update() already did it.
        # IMPORTANT: during recovery we must follow the *predicted motion*.
        # Anchoring to a stale last-match location causes "lane hopping" to any similar object
        # that stays near the old bbox while the real target has moved.
        if pred_bbox is not None:
            pred_bbox = self._clip_bbox_xyxy(pred_bbox, frame.shape)

        img, (ox, oy) = self._crop_search_region(frame, pred_bbox, lost_frames=self.lost_frames)
        res = self._yolo_predict(img, conf=self.YOLO_CONF_SELECT, classes=[self.target_class_id])
        # IMPORTANT: no full-frame fallback while we still expect the target near prediction.
        # This prevents "lane hopping" to any other same-class object.
        if (not res.boxes) and (self.lost_frames >= self.REACQUIRE_LOCAL_ONLY_FRAMES):
            ox, oy = 0, 0
            img = frame
            res = self._yolo_predict(img, conf=self.YOLO_CONF_SELECT, classes=[self.target_class_id])
        if not res.boxes:
            self._reacquire_streak = 0
            self._dbg(
                "H3",
                "object_tracker_v5.py:_try_recovery",
                "recovery_no_dets",
                {"strict": strict, "lost_frames": self.lost_frames, "used_global": self.lost_frames >= self.REACQUIRE_LOCAL_ONLY_FRAMES},
            )
            return False

        best_score = 0
        best_box = None
        second_best = 0.0

        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            # shift back to full-frame coordinates
            b[0] += ox; b[2] += ox
            b[1] += oy; b[3] += oy
            if pred_bbox is not None:
                iou_sim = self._iou_xyxy(pred_bbox, b.tolist())
                if iou_sim < self.RECOVER_MIN_IOU and (not self._within_center_jump_recover(b.tolist(), pred_bbox)):
                    continue
            roi = [b[0], b[1], b[2]-b[0], b[3]-b[1]]

            h = self.get_hist_features(frame, roi)
            c = self.get_cnn_features(frame, roi)

            hist_pos = self.hist_bank(h)
            cnn_pos = self.cnn_bank(c)
            neg_penalty = self._neg_penalty(h, c)

            # spatial agreement is critical in recovery
            if pred_bbox is not None:
                pcx = (pred_bbox[0] + pred_bbox[2]) / 2
                pcy = (pred_bbox[1] + pred_bbox[3]) / 2
            else:
                pcx, pcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            dist_sim = float(np.exp(-np.hypot((b[0] + b[2]) / 2 - pcx, (b[1] + b[3]) / 2 - pcy) / self.SPATIAL_DIST_SCALE_PX))
            iou_sim = self._iou_xyxy(pred_bbox, b.tolist()) if pred_bbox is not None else 0.0
            spatial = float(self.SPATIAL_IOU_WEIGHT * iou_sim + (1.0 - self.SPATIAL_IOU_WEIGHT) * dist_sim)

            score = float(np.clip(self._recovery_score(hist_pos, cnn_pos) + 0.25 * spatial - 0.35 * neg_penalty, 0, 1))

            if score > best_score:
                second_best = best_score
                best_score = score
                best_box = b
            elif score > second_best:
                second_best = score

        th = self.RECOVER_STRICT_TH if strict else self.RECOVER_TH
        margin = float(best_score - second_best)

        # require both score and separation to avoid hijacking to a twin object
        if best_box is None or best_score < th or margin < max(0.08, self.IDENTITY_MARGIN_TH * 0.8):
            self._reacquire_streak = 0
            self._dbg(
                "H3",
                "object_tracker_v5.py:_try_recovery",
                "recovery_reject",
                {"best_score": best_score, "second_best": second_best, "margin": margin, "th": th, "strict": strict},
            )
            return False

        cx = (best_box[0]+best_box[2])/2
        cy = (best_box[1]+best_box[3])/2

        self.kf.kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)

        self.last_bbox = best_box.tolist()
        self.last_mode = "RECOVER"

        self._reacquire_streak += 1
        if self._reacquire_streak >= self.REACQUIRE_MIN_STREAK:
            self.STATE = "TRACK"
            self.last_mode = "MATCH"
            self._reacquire_streak = 0
            self.lost_frames = 0
            print("[RECOVERY LOCKED]")
            self._dbg(
                "H3",
                "object_tracker_v5.py:_try_recovery",
                "recovery_locked",
                {"best_score": best_score, "margin": margin, "bbox": self.last_bbox},
            )
            return True
        return False

    # ---------------------------
    def update(self, frame):
        self.frame_count += 1

        # 🔥 חשוב: Kalman ממשיך לזוז גם ב SEARCH
        pred = self.kf.predict()
        pred_cx, pred_cy = float(pred[0][0]), float(pred[1][0])
        # Clamp runaway predictions back into frame to avoid degenerate crops
        fh, fw = frame.shape[:2]
        if not (0.0 <= pred_cx <= (fw - 1)) or not (0.0 <= pred_cy <= (fh - 1)):
            pred_cx = float(np.clip(pred_cx, 0.0, float(fw - 1)))
            pred_cy = float(np.clip(pred_cy, 0.0, float(fh - 1)))
            try:
                self.kf.kf.statePost[0][0] = np.float32(pred_cx)
                self.kf.kf.statePost[1][0] = np.float32(pred_cy)
                # damp velocity when we've hit image bounds
                self.kf.kf.statePost[2][0] = np.float32(0.0)
                self.kf.kf.statePost[3][0] = np.float32(0.0)
            except Exception:
                pass

        pred_bbox = None
        if self._last_size_wh is not None:
            pb = self._bbox_from_center(pred_cx, pred_cy)
            if pb is not None:
                pred_bbox = self._clip_bbox_xyxy(pb, frame.shape)

        if (self.frame_count % self._dbg_throttle_n) == 0 or self.STATE != self._prev_state:
            self._dbg(
                "H1",
                "object_tracker_v5.py:update",
                "frame_state",
                {
                    "frame": self.frame_count,
                    "state": self.STATE,
                    "lost": self.lost_frames,
                    "last_mode": self.last_mode,
                    "last_bbox": self.last_bbox,
                    "last_match_bbox": self._last_match_bbox,
                    "pred_bbox": pred_bbox,
                },
            )
            self._prev_state = self.STATE

        # recovery only when actually needed (prevents identity jumps)
        if (self.STATE == "SEARCH" or self.lost_frames > 0) and (self.frame_count % self.recovery_interval == 0):
            self._try_recovery(frame, pred_bbox=pred_bbox, strict=(self.STATE == "SEARCH"))

        if self.STATE == "SEARCH":
            # keep exposing a reasonable confidence signal
            self.last_hist_sim = 0.0
            self.last_cnn_sim = 0.0
            self.last_spatial_sim = 0.0
            self.last_neg_penalty = 0.0
            self.last_score = 0.0
            # If recovery has a tentative bbox, keep showing it (and keep Kalman running)
            if self.last_mode == "RECOVER" and self.last_bbox is not None:
                return self.last_bbox
            return None

        # performance: skip detection on most frames
        if (self.frame_count % self.DETECT_EVERY_N) != 0:
            if self._last_size_wh is not None:
                pred_bbox = self._bbox_from_center(pred_cx, pred_cy)
                if pred_bbox is not None:
                    self.last_bbox = self._clip_bbox_xyxy(pred_bbox, frame.shape)
                    self.last_mode = "PRED"
                    # keep a decaying confidence signal while we coast
                    self.last_score = float(max(self.last_score * 0.98, 0.0))
                    self.signal_history.append(self.last_score)
                    return self.last_bbox
            return None

        # Anchor tracking crop:
        # - if we're lost, rely on prediction to keep following motion
        # - otherwise use last confirmed match (more stable than instantaneous prediction)
        if self.lost_frames > 0:
            anchor_bbox = pred_bbox
        else:
            anchor_bbox = self._last_match_bbox if self._last_match_bbox is not None else pred_bbox

        img, (ox, oy) = self._crop_search_region(frame, anchor_bbox, lost_frames=self.lost_frames)
        res = self._yolo_predict(img, conf=self.YOLO_CONF_TRACK, classes=[self.target_class_id])

        best_box = None
        best_score = 0
        second_best_score = 0.0
        second_box = None
        best_hist_pos = 0.0
        best_cnn_pos = 0.0
        best_spatial = 0.0
        best_h = None
        best_c = None
        best_neg_penalty = 0.0
        second_h = None
        second_c = None

        # stage 1: collect candidates and rank by spatial first (cheap)
        candidates = []
        pre_gate = 0
        if res.boxes:
            for box in res.boxes:
                pre_gate += 1
                b = box.xyxy[0].cpu().numpy().astype(int)
                b[0] += ox; b[2] += ox
                b[1] += oy; b[3] += oy

                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                dist_sim = float(np.exp(-np.hypot(cx - pred_cx, cy - pred_cy) / self.SPATIAL_DIST_SCALE_PX))
                ref_bbox = self._last_match_bbox if self._last_match_bbox is not None else self.last_bbox
                iou_sim = self._iou_xyxy(ref_bbox, b.tolist()) if ref_bbox is not None else 0.0
                spatial = float(self.SPATIAL_IOU_WEIGHT * iou_sim + (1.0 - self.SPATIAL_IOU_WEIGHT) * dist_sim)
                # hard spatial gate to prevent identity switching between identical neighbors
                if ref_bbox is not None:
                    if iou_sim < self.MIN_MATCH_IOU and spatial < self.MIN_MATCH_SPATIAL:
                        continue
                    if not self._within_center_jump(b.tolist(), ref_bbox):
                        continue
                candidates.append((spatial, b, iou_sim))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:max(1, self.MAX_CANDIDATES)]

        if (self.frame_count % self._dbg_throttle_n) == 0:
            self._dbg(
                "H2",
                "object_tracker_v5.py:update",
                "candidates_after_gate",
                {
                    "pre_gate": pre_gate,
                    "post_gate": len(candidates),
                    "anchor_bbox": anchor_bbox,
                    "ox_oy": [ox, oy],
                    "top": [
                        {"spatial": float(s), "iou": float(i), "bbox": bb.tolist()}
                        for (s, bb, i) in candidates[:3]
                    ],
                },
            )

        compute_features_all = ((self.frame_count // self.DETECT_EVERY_N) % self.FEATURE_EVERY_N) == 0

        for idx, (spatial, b, iou_sim) in enumerate(candidates):

            roi = [b[0], b[1], b[2]-b[0], b[3]-b[1]]

            # stage 2: compute expensive appearance only for top-K and not every frame
            # IMPORTANT: we must compute appearance for at least the best spatial candidate,
            # otherwise the score is capped by W_SPATIAL and will never pass MATCH_TH.
            compute_features = compute_features_all or (idx == 0)
            if compute_features:
                h = self.get_hist_features(frame, roi)
                c = self.get_cnn_features(frame, roi)
            else:
                h = None
                c = None

            hist_pos = self.hist_bank(h) if compute_features else 0.0
            cnn_pos = self.cnn_bank(c) if compute_features else 0.0

            neg_penalty = self._neg_penalty(h, c) if compute_features else 0.0

            score = self._score(hist_pos, cnn_pos, spatial, neg_penalty)

            if score > best_score:
                second_best_score = best_score
                second_box = best_box
                second_h = best_h
                second_c = best_c
                best_score = score
                best_box = b
                best_hist_pos = hist_pos
                best_cnn_pos = cnn_pos
                best_spatial = float(spatial)
                best_h = h
                best_c = c
                best_neg_penalty = float(neg_penalty)
            elif score > second_best_score:
                second_best_score = score
                second_box = b
                second_h = h
                second_c = c

        # update debug signal even if we fail thresholds
        self.last_hist_sim = float(best_hist_pos)
        self.last_cnn_sim = float(best_cnn_pos)
        self.last_spatial_sim = float(best_spatial)
        self.last_neg_penalty = float(best_neg_penalty)
        self.last_score = float(np.clip(best_score, 0.0, 1.0))
        self.signal_history.append(self.last_score)

        if best_box is not None and best_score > self.MATCH_TH:
            cx = (best_box[0]+best_box[2])/2
            cy = (best_box[1]+best_box[3])/2
            self.kf.update(cx, cy)

            bb = best_box.tolist()
            self.last_bbox = self._clip_bbox_xyxy(bb, frame.shape)
            self._last_match_bbox = self.last_bbox.copy()
            self._last_size_wh = (int(best_box[2] - best_box[0]), int(best_box[3] - best_box[1]))
            self.last_mode = "MATCH"
            self.lost_frames = 0
            self._dbg(
                "H2",
                "object_tracker_v5.py:update",
                "match_accept",
                {
                    "score": float(best_score),
                    "second": float(second_best_score),
                    "margin": float(best_score - second_best_score),
                    "hist": float(best_hist_pos),
                    "cnn": float(best_cnn_pos),
                    "spatial": float(best_spatial),
                    "neg": float(best_neg_penalty),
                    "bbox": self.last_bbox,
                },
            )

            # update identity memory only when we're confident and not ambiguous
            margin = float(best_score - second_best_score)
            if best_score >= self.UPDATE_TH and margin >= self.IDENTITY_MARGIN_TH and best_h is not None and best_c is not None:
                if best_h is not None:
                    self.pos_hist_bank.append(best_h)
                if best_c is not None:
                    self.pos_cnn_bank.append(best_c)
            else:
                # NOTE: we intentionally avoid auto-populating negatives from close confusers,
                # because when two cars overlap/drive close, it can poison the true identity.
                return self.last_bbox

        # lost
        self.lost_frames += 1
        if (self.lost_frames == 1) or ((self.frame_count % self._dbg_throttle_n) == 0):
            self._dbg(
                "H4",
                "object_tracker_v5.py:update",
                "lost",
                {"lost_frames": self.lost_frames, "pred_bbox": pred_bbox, "last_match_bbox": self._last_match_bbox},
            )
        if self.lost_frames <= self.max_pred_frames and self._last_size_wh is not None:
            pred_bbox = self._bbox_from_center(pred_cx, pred_cy)
            if pred_bbox is not None:
                self.last_bbox = self._clip_bbox_xyxy(pred_bbox, frame.shape)
                self.last_mode = "PRED"
            return self.last_bbox

        self.STATE = "SEARCH"
        self.last_mode = "NONE"
        return None

    # ---------------------------
    def draw(self, frame, fps):
        if self.last_bbox is None:
            return

        x1,y1,x2,y2 = self.last_bbox
        color = (0,255,0) if self.last_mode=="MATCH" else (0,255,255)

        cv2.rectangle(frame, (x1,y1),(x2,y2), color,2)

        cv2.putText(frame, f"STATE: {self.STATE}", (10,30),0,0.8,(255,255,255),2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,60),0,0.8,(255,255,255),2)
        cv2.putText(frame, f"SCORE: {self.last_score:.2f} ({self.last_mode})", (10,90), 0, 0.8, (255,255,255), 2)

    # ---------------------------
    def get_signal_value(self):
        return float(self.last_score)