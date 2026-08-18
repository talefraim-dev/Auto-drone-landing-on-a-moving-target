"""
object_tracker.py — YOLO + ResNet tracker adapter for DroneEnv.

This file is a DROP-IN replacement for the previous MixFormer-based object_tracker.py.

Why this adapter exists:
    DroneEnv expects a class named `tracker` with a legacy interface:
        - select_target_and_get_class(frame, x, y)
        - get_target_fingerprint()
        - set_target_fingerprint(fp)
        - set_target_class(cid)
        - auto_lock_on_fingerprint(frame, use_class_gate=True)
        - update(frame) -> bbox in XYXY format
        - draw(frame, fps=None)

The new implementation uses:
    - YOLO for detection candidates
    - ResNet18 embeddings for target identity matching

Important coordinate convention:
    resnet_yolo_tracker.YoloResNetTracker internally uses XYWH.
    DroneEnv / TargetTrackerManager expects XYXY.
    This adapter converts XYWH -> XYXY before returning bboxes.
"""

from __future__ import annotations

from typing import Optional, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from resnet_yolo_tracker import YoloResNetTracker


class tracker:
    """
    Legacy-compatible tracker wrapper.

    This class intentionally keeps the same public fields used by DroneEnv:
        last_bbox
        last_good_bbox
        last_mode
        last_raw_mode
        last_reject_reason
        last_identity_metrics
        target_class_id
        last_yaw_rate_cmd_dps
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.core = YoloResNetTracker(
            yolo_model_path="yolo11s.pt",
            device=self.device,
            target_classes=None,
            yolo_conf=0.25,
            click_pad=20,
            min_match_score=0.45,
            appearance_weight=0.75,
            motion_weight=0.25,
            search_window_scale=3.0,
            use_search_window=True,
            ema_alpha=0.70,
            verbose=False,
        )
        # The core may move only its frozen ReID extractor to CPU after a
        # recoverable CUDA OOM. Keep restored target fingerprints on the same
        # device as that extractor.
        self.device = str(self.core.device)

        self.last_bbox: Optional[List[int]] = None          # XYXY for DroneEnv
        self.last_good_bbox: Optional[List[int]] = None     # XYXY for DroneEnv
        self.last_mode: str = "NONE"                       # MATCH / PRED / NONE
        self.last_raw_mode: str = "INIT"
        self.last_reject_reason: str = ""
        self.last_identity_metrics: dict = {}

        self.target_class_id: Optional[int] = None
        self.target_fingerprint = None

        self.pred_frames: int = 0
        self.frame_index: int = 0
        self.last_yaw_rate_cmd_dps: float = 0.0

        print(f"[TRACKER] YOLO+RESNET adapter initialized. device={self.device}")

    @staticmethod
    def _xywh_to_xyxy(bbox_xywh) -> Optional[List[int]]:
        if bbox_xywh is None:
            return None

        x, y, w, h = [int(round(float(v))) for v in bbox_xywh[:4]]
        return [x, y, x + max(1, w), y + max(1, h)]

    @staticmethod
    def _xyxy_to_xywh(bbox_xyxy) -> Optional[List[int]]:
        if bbox_xyxy is None:
            return None

        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy[:4]]
        return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

    def _sync_from_core(self) -> Optional[List[int]]:
        """
        Copy state from the YOLO+ResNet core into the legacy fields.
        """
        bbox_xyxy = self._xywh_to_xyxy(self.core.last_bbox)

        raw_mode = str(getattr(self.core, "last_mode", "IDLE") or "IDLE")
        score = float(getattr(self.core, "last_score", 0.0) or 0.0)

        self.last_bbox = None if bbox_xyxy is None else bbox_xyxy.copy()

        if raw_mode == "MATCH":
            self.last_mode = "MATCH"
            self.last_raw_mode = f"MATCH_YOLO_RESNET score={score:.3f}"
            self.last_good_bbox = None if bbox_xyxy is None else bbox_xyxy.copy()
            self.pred_frames = 0
            self.last_reject_reason = ""

        elif raw_mode.startswith("PRED"):
            self.last_mode = "PRED"
            self.last_raw_mode = f"{raw_mode}_YOLO_RESNET score={score:.3f}"
            self.pred_frames += 1
            self.last_reject_reason = raw_mode

        elif raw_mode == "INIT":
            self.last_mode = "MATCH"
            self.last_raw_mode = "INIT_YOLO_RESNET"
            self.last_good_bbox = None if bbox_xyxy is None else bbox_xyxy.copy()
            self.pred_frames = 0
            self.last_reject_reason = ""

        else:
            self.last_mode = "NONE"
            self.last_raw_mode = f"{raw_mode}_YOLO_RESNET"
            self.pred_frames += 1
            self.last_reject_reason = raw_mode

        self.target_class_id = self.core.target_class_id

        self.last_identity_metrics = {
            "score": score,
            "resnet_score": float(getattr(self.core, "last_bank_similarity", score) or score),
            "core_mode": raw_mode,
            "target_class_id": -1 if self.target_class_id is None else int(self.target_class_id),
            "scale_bank_size": int(len(getattr(self.core, "template_bank", []) or [])),
            "scale_bank_updates": int(getattr(self.core, "template_bank_updates", 0) or 0),
            "candidate_scale": float(getattr(self.core, "last_candidate_scale", 0.0) or 0.0),
            "matched_template_scale": float(getattr(self.core, "last_matched_template_scale", 0.0) or 0.0),
        }

        return self.last_bbox

    def get_target_fingerprint(self):
        """
        DroneEnv stores this after the first click.

        We return the actual ResNet embedding as a CPU numpy vector so it can be
        restored after episode resets.
        """
        emb = getattr(self.core, "target_embedding", None)
        if emb is None:
            return None

        try:
            return emb.detach().float().cpu().numpy().copy()
        except Exception:
            return None

    def set_target_fingerprint(self, fp):
        """
        Restore a saved ResNet target embedding.
        """
        if fp is None:
            return None

        try:
            arr = np.asarray(fp, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return None

            t = torch.from_numpy(arr).float().to(self.device)
            t = F.normalize(t, dim=0)
            self.core.restore_target_embedding(t)
            self.target_fingerprint = arr.copy()
            return None

        except Exception as exc:
            self.last_reject_reason = f"set_target_fingerprint_failed: {exc}"
            return None

    def set_target_class(self, cid):
        self.target_class_id = None if cid is None else int(cid)
        self.core.target_class_id = self.target_class_id

    def select_target_and_get_class(self, frame, x, y):
        """
        Initial click selection.

        Uses YOLO bbox under/near click, then saves a ResNet embedding.
        """
        bbox_xywh = self.core.select_target(frame, int(x), int(y))

        if bbox_xywh is None:
            self.last_mode = "NONE"
            self.last_raw_mode = "CLICK_SELECT_FAILED"
            self.last_reject_reason = "No bbox/embedding produced from click"
            print("[TRACKER] YOLO+RESNET click failed")
            return None

        self.target_class_id = self.core.target_class_id
        self.target_fingerprint = self.get_target_fingerprint()

        self._sync_from_core()
        self.last_mode = "MATCH"
        self.last_raw_mode = "CLICK_SELECT_YOLO_RESNET"

        print(f"[TRACKER] YOLO+RESNET LOCK(click) cid={self.target_class_id} bbox_xyxy={self.last_bbox}")
        return self.target_class_id

    def auto_lock_on_fingerprint(self, frame, use_class_gate=True):
        """
        Reacquire target after AirSim episode reset.

        DroneEnv calls this after restoring the saved fingerprint/class.
        We scan current YOLO detections and pick the one with the best ResNet
        similarity to the saved target embedding.
        """
        if self.core.target_embedding is None:
            self.last_raw_mode = "AUTO_LOCK_NO_FINGERPRINT"
            return False

        try:
            candidates = self.core._detect_candidates(frame)
        except Exception as exc:
            self.last_raw_mode = "AUTO_LOCK_DETECT_EXCEPTION"
            self.last_reject_reason = str(exc)
            return False

        if use_class_gate and self.target_class_id is not None:
            same_class = [c for c in candidates if int(c.cls_id) == int(self.target_class_id)]
            if same_class:
                candidates = same_class

        best = None
        best_app = -1.0

        for cand in candidates:
            emb = self.core._embedding_from_bbox(frame, cand.bbox)
            if emb is None:
                continue

            candidate_scale = self.core._bbox_scale(cand.bbox, frame.shape)
            app, _ = self.core._bank_similarity(emb, candidate_scale)
            if app > best_app:
                best_app = app
                best = cand

        if best is None:
            self.last_mode = "NONE"
            self.last_raw_mode = "AUTO_LOCK_NO_CANDIDATE"
            self.last_reject_reason = "No YOLO candidate matched fingerprint"
            return False

        # Threshold is intentionally moderate. The stable layer will still gate jumps.
        if best_app < 0.50:
            self.last_mode = "NONE"
            self.last_raw_mode = f"AUTO_LOCK_LOW_SCORE score={best_app:.3f}"
            self.last_reject_reason = self.last_raw_mode
            return False

        self.core.last_bbox = best.bbox.copy()
        self.core.last_good_bbox = best.bbox.copy()
        self.core.last_score = best_app
        self.core.last_mode = "MATCH"
        self.core.target_class_id = best.cls_id

        self.target_class_id = best.cls_id
        self._sync_from_core()
        self.last_raw_mode = f"ACTIVE_REACQUIRE"  # lets DroneEnv reset stable tracker safely
        self.last_identity_metrics["auto_lock_score"] = best_app

        print(f"[TRACKER] YOLO+RESNET AUTO_LOCK bbox_xyxy={self.last_bbox} score={best_app:.3f} cls={best.cls_id}")
        return True

    def update(self, frame):
        self.frame_index += 1

        try:
            bbox_xywh = self.core.update(frame)
        except Exception as exc:
            self.last_mode = "NONE"
            self.last_raw_mode = f"YOLO_RESNET_EXCEPTION: {type(exc).__name__}"
            self.last_reject_reason = str(exc)
            self.pred_frames += 1
            return None

        self._sync_from_core()

        if bbox_xywh is None:
            return None

        return self.last_bbox

    def draw(self, frame, fps=None):
        if self.last_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in self.last_bbox]
            color = (0, 255, 0) if self.last_mode == "MATCH" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        cv2.putText(
            frame,
            f"YOLO+RESNET: {self.last_mode}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"RAW: {self.last_raw_mode}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"pred={self.pred_frames} reject={self.last_reject_reason[:50]}",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        if fps is not None:
            cv2.putText(
                frame,
                f"FPS: {float(fps):.1f}",
                (10, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return frame

    def close(self):
        pass
