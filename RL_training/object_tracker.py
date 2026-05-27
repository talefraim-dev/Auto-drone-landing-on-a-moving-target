# object_tracker.py
# V5 Target-Centric Isolation Tracker + optional OSNet/ReID for City Sample / dense traffic scenes.
#
# Core idea:
#   The tracker does NOT search for "a car".
#   It tracks the specific user-selected instance.
#
# Strategy:
#   1. One-time user click initializes target identity.
#   2. Build a rich fixed target signature from the selected crop.
#   3. Search only in a strict local ROI around the predicted target.
#   4. Score candidates using spatial consistency + rich signature consistency + optional OSNet/ReID identity.
#   5. If candidates are ambiguous, stay in PRED instead of switching identity.
#   6. Do not update the target signature during uncertain tracking.
#
# Important:
#   This version intentionally prefers PRED over false MATCH.
#   V4c removes any static-target assumption. MATCH rescue is based only on
#   dynamic motion consistency: candidate near the predicted target trajectory.
#   This supports both static and moving targets.
#   V5 adds optional OSNet/ReID as an identity embedding layer. If torchreid is missing, it falls back safely.

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# OSNet / Torchreid is optional.
# Install with:
#   pip install torchreid gdown yacs
# If torchreid is not installed, the tracker automatically falls back to V4c behavior.
try:
    from torchreid import models as torchreid_models
    TORCHREID_AVAILABLE = True
except Exception:
    torchreid_models = None
    TORCHREID_AVAILABLE = False


class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 1e-7]],  # tiny value avoids some OpenCV edge cases
            np.float32,
        )
        # Reset the measurement matrix exactly after construction.
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.025

    def predict(self):
        return self.kf.predict()

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32))


class tracker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11n.pt").to(self.device)

        # Optional OSNet/ReID identity extractor.
        # Important: OSNet pretrained weights are mostly person-ReID, not vehicle-ReID.
        # Therefore we use it as supporting evidence, not as the only authority.
        self.reid_enabled = False
        self.reid_model = None
        self._init_reid_model()

        self.kf = KalmanFilter()

        self.STATE = "SEARCH"
        self.target_class_id = None

        # Backward compatibility with older env code.
        self.target_features = None

        # Rich identity signature.
        self.target_signature = None

        self.last_bbox = None
        self.last_mode = "NONE"

        self._prev_center = None
        self._vel_ema = np.zeros(2, dtype=np.float32)

        # ------------------------------------------------------
        # Conservative thresholds for dense vehicle environments
        # ------------------------------------------------------
        self.MATCH_TH = 0.58
        self.AUTOLOCK_TH = 0.54

        # Strict identity thresholds.
        # Color/HSV are kept more important than edge/texture because edge/texture
        # can fluctuate under motion blur, perspective changes, and partial occlusion.
        self.MIN_COLOR_SIM = 0.28
        self.MIN_HSV_SIM = 0.22
        self.MIN_EDGE_SIM = 0.24
        self.MIN_TEXTURE_SIM = 0.18
        self.MIN_SPATIAL_SIM = 0.22

        # Optional OSNet/ReID thresholds.
        # ReID helps separate visually similar instances, but because default OSNet is usually
        # person-ReID pretrained, thresholds should be moderate at first.
        self.MIN_REID_SIM = 0.52
        self.STRONG_REID_SIM = 0.64
        self.MIN_REID_MARGIN = 0.035
        self.REID_WEIGHT = 0.26

        # Dynamic motion-consistent rescue.
        # This is NOT a static-target assumption. It accepts a candidate only if it is
        # close to the predicted target trajectory produced by Kalman + target velocity
        # + ego-motion compensation.
        self.MOTION_RESCUE_GATE_FRAC = 0.42
        self.MOTION_RESCUE_MIN_COMBINED_SIG = 0.40
        self.MOTION_RESCUE_MIN_COLOR_SIM = 0.34
        self.MOTION_RESCUE_MIN_HSV_SIM = 0.30

        # Ambiguity threshold: if best and second best are too close, stay PRED.
        self.MIN_SCORE_MARGIN = 0.14

        # Ego-motion: env may update this every step.
        self.last_yaw_rate_cmd_dps = 0.0
        self.YAW_PIX_PER_DPS = 0.12

        # Strict local ROI / spatial gate.
        # These are intentionally tighter than previous versions.
        self.SPATIAL_GATE_MIN_PX = 28.0
        self.SPATIAL_GATE_MAX_PX = 125.0
        self.SPATIAL_GATE_BBOX_SCALE = 0.55
        self.SPATIAL_GATE_LOST_BOOST_PX = 7.0
        self.SPATIAL_GATE_YAW_BOOST = 0.14
        self.SPATIAL_GATE_VELOCITY_BOOST = 0.65

        # Candidate center must be inside an expanded local ROI.
        # This is the "mask out the world" idea at the association level.
        self.LOCAL_ROI_EXPAND_X = 1.75
        self.LOCAL_ROI_EXPAND_Y = 1.95

        # Scale/shape consistency. Tight on purpose.
        self.AREA_RATIO_MIN = 0.42
        self.AREA_RATIO_MAX = 2.45
        self.ASPECT_RATIO_MIN = 0.55
        self.ASPECT_RATIO_MAX = 1.85

        # If a candidate requires a suspicious jump, require repeated evidence.
        self.JUMP_DIST_PX = 38.0
        self.JUMP_CONFIRM_FRAMES = 4
        self._pending_box = None
        self._pending_count = 0

        # Prediction limit.
        self.MAX_PRED_FRAMES = 160
        self._pred_frames = 0

        # Safe signature update is disabled by default.
        # Keeping the original clicked target signature prevents drift into a similar car.
        self.ALLOW_SAFE_SIGNATURE_UPDATE = False
        self.SAFE_UPDATE_AFTER_MATCHES = 8
        self._stable_match_count = 0

        self.debug_candidates = False

    # ------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------
    @staticmethod
    def _bbox_center_xy(b):
        return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)

    @staticmethod
    def _bbox_wh(b):
        return float(max(1, b[2] - b[0])), float(max(1, b[3] - b[1]))

    @staticmethod
    def _bbox_area(b):
        w, h = tracker._bbox_wh(b)
        return float(w * h)

    @staticmethod
    def _aspect_ratio(b):
        w, h = tracker._bbox_wh(b)
        return float(w / max(1.0, h))

    @staticmethod
    def _clip_box_to_frame(b, frame):
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

    def _expanded_roi_from_pred(self, pred_cx, pred_cy, ref_bbox, frame):
        H, W = frame.shape[:2]
        if ref_bbox is None:
            return [0, 0, W - 1, H - 1]

        bw, bh = self._bbox_wh(ref_bbox)
        rw = max(28.0, bw * self.LOCAL_ROI_EXPAND_X)
        rh = max(28.0, bh * self.LOCAL_ROI_EXPAND_Y)

        roi = [
            int(pred_cx - rw / 2.0),
            int(pred_cy - rh / 2.0),
            int(pred_cx + rw / 2.0),
            int(pred_cy + rh / 2.0),
        ]
        return self._clip_box_to_frame(roi, frame)

    @staticmethod
    def _center_inside_box(cx, cy, b):
        return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

    def _dynamic_spatial_gate_px(self, predicted_bbox=None):
        if predicted_bbox is None and self.last_bbox is None:
            return self.SPATIAL_GATE_MIN_PX

        ref_box = predicted_bbox if predicted_bbox is not None else self.last_bbox
        bw, bh = self._bbox_wh(ref_box)
        bbox_diag = float(np.hypot(bw, bh))

        velocity_mag = float(np.linalg.norm(self._vel_ema))

        gate = (
            self.SPATIAL_GATE_MIN_PX
            + self.SPATIAL_GATE_BBOX_SCALE * bbox_diag
            + self.SPATIAL_GATE_YAW_BOOST * abs(float(self.last_yaw_rate_cmd_dps))
            + self.SPATIAL_GATE_VELOCITY_BOOST * velocity_mag
            + self.SPATIAL_GATE_LOST_BOOST_PX * float(self._pred_frames)
        )
        return float(np.clip(gate, self.SPATIAL_GATE_MIN_PX, self.SPATIAL_GATE_MAX_PX))

    def _area_ratio_ok(self, candidate_box):
        if self.last_bbox is None:
            return True, 1.0

        last_area = self._bbox_area(self.last_bbox)
        cand_area = self._bbox_area(candidate_box)
        if last_area <= 1.0:
            return True, 1.0

        ratio = cand_area / last_area
        ok = self.AREA_RATIO_MIN <= ratio <= self.AREA_RATIO_MAX
        return bool(ok), float(ratio)

    def _aspect_ratio_ok(self, candidate_box):
        if self.target_signature is None:
            return True, 1.0

        target_ar = float(self.target_signature.get("aspect_ratio", 1.0))
        cand_ar = self._aspect_ratio(candidate_box)
        if target_ar <= 1e-6:
            return True, 1.0

        ratio = cand_ar / target_ar
        ok = self.ASPECT_RATIO_MIN <= ratio <= self.ASPECT_RATIO_MAX
        return bool(ok), float(ratio)

    def _reset_pending(self):
        self._pending_box = None
        self._pending_count = 0

    def _confirm_jump_candidate(self, candidate_box):
        if self._pending_box is None:
            self._pending_box = candidate_box.tolist() if hasattr(candidate_box, "tolist") else list(candidate_box)
            self._pending_count = 1
            return False

        pcx, pcy = self._bbox_center_xy(self._pending_box)
        ccx, ccy = self._bbox_center_xy(candidate_box)
        if np.hypot(ccx - pcx, ccy - pcy) <= 18.0:
            self._pending_count += 1
        else:
            self._pending_box = candidate_box.tolist() if hasattr(candidate_box, "tolist") else list(candidate_box)
            self._pending_count = 1

        return self._pending_count >= self.JUMP_CONFIRM_FRAMES


    # ------------------------------------------------------
    # Optional OSNet/ReID identity embedding
    # ------------------------------------------------------
    def _init_reid_model(self):
        if not TORCHREID_AVAILABLE:
            print("[TRACKER] OSNet/ReID disabled: torchreid is not installed.")
            return

        try:
            # pretrained=True asks torchreid to load available pretrained OSNet weights.
            # If weight download/loading fails, we disable ReID and continue safely.
            self.reid_model = torchreid_models.build_model(
                name="osnet_x1_0",
                num_classes=1000,
                loss="softmax",
                pretrained=True,
            )
            self.reid_model.to(self.device)
            self.reid_model.eval()
            self.reid_enabled = True
            print(f"[TRACKER] OSNet/ReID enabled on {self.device}.")
        except Exception as exc:
            self.reid_model = None
            self.reid_enabled = False
            print(f"[TRACKER] OSNet/ReID disabled: {exc}")

    def _reid_embedding_from_crop(self, crop):
        if not self.reid_enabled or self.reid_model is None:
            return None
        if crop is None or crop.size == 0:
            return None

        try:
            # OSNet person-ReID convention is usually 256x128 (H x W).
            # Cars are not people, but this gives a stable fixed input size.
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.to(self.device)

            with torch.no_grad():
                feat = self.reid_model(tensor)
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            return feat.detach().cpu().numpy().reshape(-1).astype(np.float32)
        except Exception as exc:
            # Do not crash tracking because ReID failed on one frame.
            if self.debug_candidates:
                print(f"[TRACKER] ReID embedding failed: {exc}")
            return None

    # ------------------------------------------------------
    # Feature/signature extraction
    # ------------------------------------------------------
    def _crop(self, frame, b):
        b = self._clip_box_to_frame(b, frame)
        x1, y1, x2, y2 = b
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _suppress_crop_background(self, crop):
        """
        Approximate target-centric suppression inside the crop.
        We keep the central object region stronger and dim the border.
        This is not full segmentation; it is a cheap first target-isolation layer.
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # Elliptic/rectangular central prior: vehicles are usually inside bbox center.
        cx1, cy1 = int(0.10 * w), int(0.12 * h)
        cx2, cy2 = int(0.90 * w), int(0.88 * h)
        mask[cy1:cy2, cx1:cx2] = 1.0

        # Smooth the mask for softer suppression.
        k = max(5, (min(w, h) // 8) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.15, 1.0)

        out = crop.astype(np.float32) * mask[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _normalize_hist(hist):
        hist = cv2.normalize(hist, hist).flatten()
        hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
        return hist.astype(np.float32)

    def _color_hist_bgr(self, crop):
        if crop is None or crop.size == 0:
            return None
        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return self._normalize_hist(hist)

    def _color_hist_hsv(self, crop):
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
        return self._normalize_hist(hist)

    def _edge_signature(self, crop):
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(gray, 70, 160)
        edges = edges.astype(np.float32) / 255.0
        return edges.flatten()

    def _texture_signature(self, crop):
        if crop is None or crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)
        return gray.flatten().astype(np.float32)

    def _signature_from_box(self, frame, b):
        b = self._clip_box_to_frame(b, frame)
        crop = self._crop(frame, b)
        if crop is None:
            return None

        isolated_crop = self._suppress_crop_background(crop)

        # Use both full crop and suppressed crop.
        # Suppressed crop reduces background leakage from adjacent cars.
        bgr_hist = self._color_hist_bgr(isolated_crop)
        hsv_hist = self._color_hist_hsv(isolated_crop)
        edge_sig = self._edge_signature(isolated_crop)
        texture_sig = self._texture_signature(isolated_crop)
        reid_emb = self._reid_embedding_from_crop(isolated_crop)

        return {
            "bgr_hist": bgr_hist,
            "hsv_hist": hsv_hist,
            "edge_sig": edge_sig,
            "texture_sig": texture_sig,
            "reid_emb": reid_emb,
            "aspect_ratio": self._aspect_ratio(b),
            "area": self._bbox_area(b),
            "bbox": list(map(int, b)),
        }

    def get_features(self, frame, bbox_xywh):
        """
        Backward-compatible simple BGR histogram.
        Existing env code may call this indirectly through get/set fingerprint.
        """
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

        roi = self._suppress_crop_background(roi)
        return self._color_hist_bgr(roi)

    def compare_features(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        val = float(cv2.compareHist(f1.astype(np.float32), f2.astype(np.float32), cv2.HISTCMP_CORREL))
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, -1.0, 1.0))

    @staticmethod
    def _vector_cosine_sim(a, b):
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        if a.size != b.size or a.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        val = float(np.dot(a, b) / denom)
        return float(np.clip((val + 1.0) * 0.5, 0.0, 1.0))

    def _signature_similarity(self, candidate_sig):
        if self.target_signature is None or candidate_sig is None:
            return {
                "bgr": 0.0,
                "hsv": 0.0,
                "edge": 0.0,
                "texture": 0.0,
                "combined": 0.0,
            }

        bgr = self.compare_features(self.target_signature.get("bgr_hist"), candidate_sig.get("bgr_hist"))
        hsv = self.compare_features(self.target_signature.get("hsv_hist"), candidate_sig.get("hsv_hist"))
        edge = self._vector_cosine_sim(self.target_signature.get("edge_sig"), candidate_sig.get("edge_sig"))
        texture = self._vector_cosine_sim(self.target_signature.get("texture_sig"), candidate_sig.get("texture_sig"))
        reid = self._vector_cosine_sim(self.target_signature.get("reid_emb"), candidate_sig.get("reid_emb"))
        reid_available = (self.target_signature.get("reid_emb") is not None and candidate_sig.get("reid_emb") is not None)

        # Clamp histogram correlations to [0,1] for weighted scoring.
        bgr01 = float(np.clip((bgr + 1.0) * 0.5, 0.0, 1.0))
        hsv01 = float(np.clip((hsv + 1.0) * 0.5, 0.0, 1.0))

        if reid_available:
            # OSNet is supporting identity evidence. Motion/spatial gates still dominate elsewhere.
            combined = (
                0.23 * bgr01
                + 0.23 * hsv01
                + 0.14 * edge
                + 0.14 * texture
                + self.REID_WEIGHT * reid
            )
        else:
            combined = (
                0.30 * bgr01
                + 0.30 * hsv01
                + 0.22 * edge
                + 0.18 * texture
            )

        return {
            "bgr": bgr01,
            "hsv": hsv01,
            "edge": edge,
            "texture": texture,
            "reid": float(reid),
            "reid_available": bool(reid_available),
            "combined": float(combined),
        }

    # ------------------------------------------------------
    # Fingerprint persistence API used by env
    # ------------------------------------------------------
    def get_target_fingerprint(self):
        if self.target_signature is not None:
            # Return rich signature. Keep arrays copied to avoid accidental mutation.
            copied = {}
            for k, v in self.target_signature.items():
                copied[k] = v.copy() if hasattr(v, "copy") else v
            return copied

        return None if self.target_features is None else self.target_features.copy()

    def set_target_fingerprint(self, fp):
        self.target_signature = None
        self.target_features = None

        if fp is None:
            return

        if isinstance(fp, dict):
            self.target_signature = {}
            for k, v in fp.items():
                self.target_signature[k] = v.copy() if hasattr(v, "copy") else v
            self.target_features = self.target_signature.get("bgr_hist")
        else:
            # Backward compatibility with older saved fingerprints.
            self.target_features = fp.copy() if hasattr(fp, "copy") else fp

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)

    # ------------------------------------------------------
    # IMPORTANT: env depends on this function
    # ------------------------------------------------------
    def select_target_and_get_class(self, frame, x, y):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return None

        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            b = np.array(self._clip_box_to_frame(b, frame), dtype=int)

            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                self.target_class_id = int(box.cls[0])
                self.target_signature = self._signature_from_box(frame, b)
                self.target_features = None if self.target_signature is None else self.target_signature.get("bgr_hist")

                cx, cy = self._bbox_center_xy(b)
                self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

                self.last_bbox = b.tolist()
                self.STATE = "TRACK"
                self.last_mode = "MATCH"
                self._prev_center = (cx, cy)
                self._vel_ema[:] = 0.0
                self._pred_frames = 0
                self._stable_match_count = 0
                self._reset_pending()

                print(f"[TRACKER] LOCKED(click) cid={self.target_class_id}")
                return self.target_class_id

        return None

    # ------------------------------------------------------
    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        """
        Used after env reset to relock the already selected target.
        Conservative by design: if multiple similar objects are present, do not lock aggressively.
        """
        if self.target_signature is None and self.target_features is None:
            return False

        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        if not results.boxes:
            return False

        h, w = frame.shape[:2]
        pred_cx, pred_cy = w / 2.0, h / 2.0

        candidates = []
        for box in results.boxes:
            if use_class_gate and self.target_class_id is not None:
                if int(box.cls[0]) != self.target_class_id:
                    continue

            b = box.xyxy[0].cpu().numpy().astype(int)
            b = np.array(self._clip_box_to_frame(b, frame), dtype=int)

            cand_sig = self._signature_from_box(frame, b)
            sig = self._signature_similarity(cand_sig)

            cx, cy = self._bbox_center_xy(b)
            dist = float(np.hypot(cx - pred_cx, cy - pred_cy))
            spatial_sim = float(np.exp(-dist / 120.0))

            score = 0.72 * sig["combined"] + 0.28 * spatial_sim
            candidates.append((score, sig, spatial_sim, dist, b))

        if not candidates:
            return False

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sig, best_spatial, best_dist, best_box = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -999.0
        margin = best_score - second_score

        if best_score < self.AUTOLOCK_TH:
            return False

        if len(candidates) > 1 and margin < self.MIN_SCORE_MARGIN:
            if self.debug_candidates:
                print(f"[TRACKER] AUTOLOCK ambiguous: score={best_score:.3f}, margin={margin:.3f}")
            return False

        cx, cy = self._bbox_center_xy(best_box)
        self.kf.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        self.last_bbox = best_box.tolist()
        self.STATE = "TRACK"
        self.last_mode = "MATCH"
        self._prev_center = (cx, cy)
        self._vel_ema[:] = 0.0
        self._pred_frames = 0
        self._stable_match_count = 0
        self._reset_pending()

        print("[TRACKER] RELOCK(auto)")
        return True

    # ------------------------------------------------------
    def _candidate_score(self, b, frame, pred_cx, pred_cy, spatial_gate, local_roi):
        b = np.array(self._clip_box_to_frame(b, frame), dtype=int)
        cx, cy = self._bbox_center_xy(b)

        # Target isolation: ignore everything outside the local predicted ROI.
        if not self._center_inside_box(cx, cy, local_roi):
            return None

        dist = float(np.hypot(cx - pred_cx, cy - pred_cy))
        if dist > spatial_gate:
            return None

        spatial_sim = float(np.exp(-dist / max(1.0, spatial_gate)))
        if spatial_sim < self.MIN_SPATIAL_SIM:
            return None

        area_ok, area_ratio = self._area_ratio_ok(b)
        if not area_ok:
            return None

        aspect_ok, aspect_ratio = self._aspect_ratio_ok(b)
        if not aspect_ok:
            return None

        cand_sig = self._signature_from_box(frame, b)
        sig = self._signature_similarity(cand_sig)

        # Identity gates.
        # We keep two acceptance paths:
        #   1. strict_identity: full rich signature looks good.
        #   2. motion_rescue: candidate is very close to the dynamic predicted trajectory
        #      and has reasonable color/HSV identity. This is not a static assumption;
        #      it works for moving targets because pred_cx/pred_cy comes from Kalman +
        #      velocity + ego-motion compensation.
        base_identity = (
            sig["bgr"] >= self.MIN_COLOR_SIM
            and sig["hsv"] >= self.MIN_HSV_SIM
            and sig["edge"] >= self.MIN_EDGE_SIM
            and sig["texture"] >= self.MIN_TEXTURE_SIM
        )

        # ReID identity path: useful when two cars have similar color/shape but different deep identity embedding.
        # Still require reasonable color/HSV because OSNet is not vehicle-specialized by default.
        reid_identity = (
            sig.get("reid_available", False)
            and sig.get("reid", 0.0) >= self.MIN_REID_SIM
            and sig["bgr"] >= max(0.20, self.MIN_COLOR_SIM - 0.08)
            and sig["hsv"] >= max(0.18, self.MIN_HSV_SIM - 0.06)
        )

        strict_identity = base_identity or reid_identity

        motion_rescue_gate = max(10.0, self.MOTION_RESCUE_GATE_FRAC * spatial_gate)
        motion_rescue = (
            dist <= motion_rescue_gate
            and sig["combined"] >= self.MOTION_RESCUE_MIN_COMBINED_SIG
            and sig["bgr"] >= self.MOTION_RESCUE_MIN_COLOR_SIM
            and sig["hsv"] >= self.MOTION_RESCUE_MIN_HSV_SIM
        )

        reid_motion_rescue = (
            sig.get("reid_available", False)
            and dist <= motion_rescue_gate
            and sig.get("reid", 0.0) >= self.STRONG_REID_SIM
            and sig["bgr"] >= 0.22
            and sig["hsv"] >= 0.20
        )
        motion_rescue = motion_rescue or reid_motion_rescue

        if not strict_identity and not motion_rescue:
            return None

        area_score = 1.0 - min(abs(np.log(max(area_ratio, 1e-6))), 1.0)
        aspect_score = 1.0 - min(abs(np.log(max(aspect_ratio, 1e-6))), 1.0)

        # Target-centric score:
        # spatial continuity dominates, rich identity supports it.
        score = (
            0.48 * spatial_sim
            + 0.30 * sig["combined"]
            + 0.12 * area_score
            + 0.10 * aspect_score
        )

        # Give a small bonus only when motion consistency is excellent.
        # This helps convert stable PRED into MATCH without assuming the target is static.
        accept_reason = "strict_identity"
        if motion_rescue:
            score += 0.04
            accept_reason = "motion_consistent" if not strict_identity else "strict_and_motion"

        jump_dist = 0.0
        if self.last_bbox is not None:
            lcx, lcy = self._bbox_center_xy(self.last_bbox)
            jump_dist = float(np.hypot(cx - lcx, cy - lcy))

        return {
            "box": b,
            "score": float(score),
            "spatial_sim": float(spatial_sim),
            "signature": sig,
            "dist": float(dist),
            "area_ratio": float(area_ratio),
            "aspect_ratio": float(aspect_ratio),
            "jump_dist": float(jump_dist),
            "accept_reason": accept_reason,
        }

    def _accept_candidate(self, best, second):
        if best is None:
            return False

        # Ambiguity: if another candidate is close in score, do not choose.
        if second is not None:
            margin = best["score"] - second["score"]
            if margin < self.MIN_SCORE_MARGIN:
                if self.debug_candidates:
                    print(f"[TRACKER] ambiguous: margin={margin:.3f} -> PRED")
                return False

            # Extra ReID ambiguity: if ReID exists but cannot clearly separate two candidates,
            # stay PRED rather than switching identity.
            bs = best.get("signature", {})
            ss = second.get("signature", {})
            if bs.get("reid_available", False) and ss.get("reid_available", False):
                reid_margin = bs.get("reid", 0.0) - ss.get("reid", 0.0)
                if reid_margin < self.MIN_REID_MARGIN and abs(best["score"] - second["score"]) < (self.MIN_SCORE_MARGIN * 1.6):
                    if self.debug_candidates:
                        print(f"[TRACKER] ReID ambiguous: reid_margin={reid_margin:.3f} -> PRED")
                    return False

        # Suspicious jump: require repeated evidence.
        if best["jump_dist"] > self.JUMP_DIST_PX and self._pred_frames < 8:
            if not self._confirm_jump_candidate(best["box"]):
                if self.debug_candidates:
                    print(f"[TRACKER] jump candidate pending: jump={best['jump_dist']:.1f}px")
                return False

        return True

    def update(self, frame):
        if self.STATE == "SEARCH":
            if self.auto_lock_on_fingerprint(frame, use_class_gate=True):
                return self.last_bbox
            self.last_mode = "NONE"
            return None

        predicted = self.kf.predict()
        pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])

        # Ego-motion compensation: commanded yaw shifts apparent image position.
        pred_cx += -float(self.last_yaw_rate_cmd_dps) * float(self.YAW_PIX_PER_DPS)

        # Build predicted bbox.
        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            bw, bh = x2 - x1, y2 - y1
            predicted_bbox = [
                int(pred_cx - bw / 2.0),
                int(pred_cy - bh / 2.0),
                int(pred_cx + bw / 2.0),
                int(pred_cy + bh / 2.0),
            ]
        else:
            predicted_bbox = None

        spatial_gate = self._dynamic_spatial_gate_px(predicted_bbox)
        local_roi = self._expanded_roi_from_pred(pred_cx, pred_cy, predicted_bbox, frame)

        results = self.model.predict(frame, conf=0.25, verbose=False)[0]

        candidates = []
        if results.boxes:
            for box in results.boxes:
                if self.target_class_id is not None and int(box.cls[0]) != self.target_class_id:
                    continue

                b = box.xyxy[0].cpu().numpy().astype(int)
                cand = self._candidate_score(b, frame, pred_cx, pred_cy, spatial_gate, local_roi)
                if cand is not None and (cand["score"] >= self.MATCH_TH or cand.get("accept_reason") == "motion_consistent"):
                    candidates.append(cand)

        candidates.sort(key=lambda c: c["score"], reverse=True)

        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None

        if self.debug_candidates:
            print(
                f"[TRACKER] roi={local_roi} gate={spatial_gate:.1f}px "
                f"candidates={len(candidates)} predFrames={self._pred_frames}"
            )
            for i, c in enumerate(candidates[:5]):
                s = c["signature"]
                print(
                    f"  cand#{i} score={c['score']:.3f} spatial={c['spatial_sim']:.3f} "
                    f"sig={s['combined']:.3f} bgr={s['bgr']:.3f} hsv={s['hsv']:.3f} "
                    f"edge={s['edge']:.3f} tex={s['texture']:.3f} reid={s.get('reid', 0.0):.3f} "
                    f"reason={c.get('accept_reason')} "
                    f"dist={c['dist']:.1f} jump={c['jump_dist']:.1f} "
                    f"areaR={c['area_ratio']:.2f} aspectR={c['aspect_ratio']:.2f}"
                )

        if self._accept_candidate(best, second):
            best_box = best["box"]
            cx, cy = self._bbox_center_xy(best_box)
            self.kf.update(cx, cy)

            if self._prev_center is not None:
                dv = np.array([cx - self._prev_center[0], cy - self._prev_center[1]], dtype=np.float32)
                self._vel_ema = 0.82 * self._vel_ema + 0.18 * dv
                self.kf.kf.statePost[2, 0] = self._vel_ema[0]
                self.kf.kf.statePost[3, 0] = self._vel_ema[1]

            self._prev_center = (cx, cy)
            self.last_bbox = best_box.tolist()
            self.last_mode = "MATCH"
            self.STATE = "TRACK"
            self._pred_frames = 0
            self._stable_match_count += 1
            self._reset_pending()

            # Safe update is off by default to avoid identity drift.
            if self.ALLOW_SAFE_SIGNATURE_UPDATE and self._stable_match_count >= self.SAFE_UPDATE_AFTER_MATCHES:
                new_sig = self._signature_from_box(frame, best_box)
                if new_sig is not None:
                    self.target_signature = new_sig
                    self.target_features = new_sig.get("bgr_hist")
                self._stable_match_count = 0

            return self.last_bbox

        # No reliable match => PRED.
        if self.last_bbox is None:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            return None

        self._pred_frames += 1
        self._stable_match_count = 0

        x1, y1, x2, y2 = self.last_bbox
        bw, bh = x2 - x1, y2 - y1

        pred_box = [
            int(pred_cx - bw / 2.0),
            int(pred_cy - bh / 2.0),
            int(pred_cx + bw / 2.0),
            int(pred_cy + bh / 2.0),
        ]
        pred_box = self._clip_box_to_frame(pred_box, frame)
        self.last_bbox = pred_box
        self.last_mode = "PRED"

        if self._pred_frames >= self.MAX_PRED_FRAMES:
            self.STATE = "SEARCH"
            self.last_mode = "NONE"
            self._prev_center = None
            self._vel_ema[:] = 0.0
            self._stable_match_count = 0
            self._reset_pending()
            return None

        return self.last_bbox

    # ------------------------------------------------------
    def draw(self, frame, fps):
        if self.last_bbox is None:
            return

        x1, y1, x2, y2 = self.last_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        color = (0, 255, 0) if self.last_mode == "MATCH" else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        cv2.putText(
            frame,
            f"{self.last_mode}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Draw local ROI in debug mode.
        if self.debug_candidates and self.last_bbox is not None:
            predicted = self.kf.predict()
            pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])
            pred_cx += -float(self.last_yaw_rate_cmd_dps) * float(self.YAW_PIX_PER_DPS)

            x1b, y1b, x2b, y2b = self.last_bbox
            bw, bh = x2b - x1b, y2b - y1b
            predicted_bbox = [
                int(pred_cx - bw / 2.0),
                int(pred_cy - bh / 2.0),
                int(pred_cx + bw / 2.0),
                int(pred_cy + bh / 2.0),
            ]
            roi = self._expanded_roi_from_pred(pred_cx, pred_cy, predicted_bbox, frame)
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 255), 1)
