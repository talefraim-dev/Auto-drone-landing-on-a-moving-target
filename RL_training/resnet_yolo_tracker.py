"""
YOLO + ResNet ReID tracker.

Purpose:
    Simple, Windows-friendly target tracking for AirSim / OpenCV loops.

Idea:
    - YOLO proposes object candidates.
    - ResNet extracts an appearance embedding for the clicked target and candidates.
    - Cosine similarity + motion prior chooses the best candidate.
    - No custom CUDA extensions, no MixFormer, no SAM2 dependency.

Expected input:
    OpenCV BGR frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None

try:
    import torchvision
    from torchvision import transforms
except Exception as exc:
    torchvision = None
    transforms = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


BBox = List[int]  # [x, y, w, h]


@dataclass
class Candidate:
    bbox: BBox
    cls_id: int
    conf: float
    appearance_score: float = 0.0
    motion_score: float = 0.0
    total_score: float = 0.0


def clamp_bbox_xywh(bbox: BBox, width: int, height: int) -> Optional[BBox]:
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    if w < 3 or h < 3:
        return None
    return [x, y, w, h]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w * 0.5, y + h * 0.5


def xyxy_to_xywh(xyxy: np.ndarray) -> BBox:
    x1, y1, x2, y2 = xyxy.astype(float).tolist()
    return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]


def point_inside_bbox(x: int, y: int, bbox: BBox, pad: int = 0) -> bool:
    bx, by, bw, bh = bbox
    return (bx - pad) <= x <= (bx + bw + pad) and (by - pad) <= y <= (by + bh + pad)


def crop_bgr(frame_bgr: np.ndarray, bbox: BBox, pad_ratio: float = 0.15) -> Optional[np.ndarray]:
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = bbox
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


class YoloResNetTracker:
    """YOLO detection + ResNet appearance matching tracker."""

    def __init__(
        self,
        yolo_model_path: str = "yolo11s.pt",
        device: Optional[str] = None,
        target_classes: Optional[List[int]] = None,
        yolo_conf: float = 0.25,
        click_pad: int = 20,
        min_match_score: float = 0.45,
        appearance_weight: float = 0.75,
        motion_weight: float = 0.25,
        search_window_scale: float = 3.0,
        use_search_window: bool = True,
        ema_alpha: float = 0.70,
        verbose: bool = True,
    ) -> None:
        if YOLO is None:
            raise ImportError(f"Could not import ultralytics.YOLO: {_YOLO_IMPORT_ERROR}")
        if torchvision is None or transforms is None:
            raise ImportError(f"Could not import torchvision: {_TORCHVISION_IMPORT_ERROR}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.yolo = YOLO(yolo_model_path)
        self.yolo_conf = float(yolo_conf)
        self.target_classes = target_classes
        self.click_pad = int(click_pad)
        self.min_match_score = float(min_match_score)
        self.appearance_weight = float(appearance_weight)
        self.motion_weight = float(motion_weight)
        self.search_window_scale = float(search_window_scale)
        self.use_search_window = bool(use_search_window)
        self.ema_alpha = float(ema_alpha)

        self.resnet = self._build_resnet_feature_extractor()
        self.preprocess = self._build_preprocess()

        self.target_embedding: Optional[torch.Tensor] = None
        self.target_class_id: Optional[int] = None
        self.last_bbox: Optional[BBox] = None
        self.last_good_bbox: Optional[BBox] = None
        self.last_score: float = 0.0
        self.last_mode: str = "IDLE"
        self.frame_index: int = 0

        if self.verbose:
            print(f"[YOLO+RESNET] device={self.device}, yolo_model={yolo_model_path}")

    def _build_resnet_feature_extractor(self) -> torch.nn.Module:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = torchvision.models.resnet18(weights=weights)
        except Exception:
            try:
                model = torchvision.models.resnet18(pretrained=True)
            except Exception as exc:
                print(f"[WARN] Could not load pretrained ResNet18 weights: {exc}")
                print("[WARN] Falling back to random weights. Tracking quality will be poor.")
                model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _build_preprocess(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _embedding_from_crop(self, crop_bgr: np.ndarray) -> Optional[torch.Tensor]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.resnet(x)
        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).detach()

    @torch.no_grad()
    def _embedding_from_bbox(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[torch.Tensor]:
        crop = crop_bgr(frame_bgr, bbox)
        return self._embedding_from_crop(crop)

    def _detect_candidates(self, frame_bgr: np.ndarray) -> List[Candidate]:
        h_img, w_img = frame_bgr.shape[:2]
        results = self.yolo.predict(frame_bgr, conf=self.yolo_conf, verbose=False)
        candidates: List[Candidate] = []
        if not results:
            return candidates
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return candidates
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
            conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            if self.target_classes is not None and cls_id not in self.target_classes:
                continue
            bbox = clamp_bbox_xywh(xyxy_to_xywh(xyxy), w_img, h_img)
            if bbox is None:
                continue
            candidates.append(Candidate(bbox=bbox, cls_id=cls_id, conf=conf))
        return candidates

    def _candidate_in_search_window(self, cand_bbox: BBox) -> bool:
        if not self.use_search_window or self.last_bbox is None:
            return True
        lx, ly, lw, lh = self.last_bbox
        cx, cy = bbox_center(self.last_bbox)
        sw = max(lw * self.search_window_scale, lw + 40)
        sh = max(lh * self.search_window_scale, lh + 40)
        sx1 = cx - sw * 0.5
        sy1 = cy - sh * 0.5
        sx2 = cx + sw * 0.5
        sy2 = cy + sh * 0.5
        ccx, ccy = bbox_center(cand_bbox)
        return sx1 <= ccx <= sx2 and sy1 <= ccy <= sy2

    def _motion_score(self, cand_bbox: BBox, frame_shape: Tuple[int, int, int]) -> float:
        if self.last_bbox is None:
            return 1.0
        h_img, w_img = frame_shape[:2]
        c1x, c1y = bbox_center(self.last_bbox)
        c2x, c2y = bbox_center(cand_bbox)
        dist = float(np.hypot(c2x - c1x, c2y - c1y))
        diag = float(np.hypot(w_img, h_img))
        normalized = dist / max(1.0, diag)
        return float(np.exp(-normalized * 8.0))

    def _smooth_bbox(self, old_bbox: BBox, new_bbox: BBox) -> BBox:
        a = self.ema_alpha
        return [int(round(a * new_bbox[i] + (1.0 - a) * old_bbox[i])) for i in range(4)]

    def select_target(self, frame_bgr: np.ndarray, click_x: int, click_y: int) -> Optional[BBox]:
        candidates = self._detect_candidates(frame_bgr)
        best: Optional[Candidate] = None
        best_dist = float("inf")
        for cand in candidates:
            if not point_inside_bbox(click_x, click_y, cand.bbox, pad=self.click_pad):
                continue
            cx, cy = bbox_center(cand.bbox)
            dist = float(np.hypot(click_x - cx, click_y - cy))
            if dist < best_dist:
                best = cand
                best_dist = dist
        if best is not None:
            if self.verbose:
                print(f"[SELECT YOLO] click=({click_x},{click_y}) bbox={best.bbox} cls={best.cls_id} conf={best.conf:.3f}")
            return self.select_target_by_bbox(frame_bgr, best.bbox, class_id=best.cls_id)

        h_img, w_img = frame_bgr.shape[:2]
        fallback = clamp_bbox_xywh([click_x - 38, click_y - 22, 76, 44], w_img, h_img)
        if fallback is None:
            return None
        if self.verbose:
            print(f"[SELECT FALLBACK] click=({click_x},{click_y}) bbox={fallback}")
        return self.select_target_by_bbox(frame_bgr, fallback, class_id=None)

    def select_target_by_bbox(self, frame_bgr: np.ndarray, bbox: BBox, class_id: Optional[int] = None) -> Optional[BBox]:
        h_img, w_img = frame_bgr.shape[:2]
        bbox = clamp_bbox_xywh(bbox, w_img, h_img)
        if bbox is None:
            print("[SELECT ERROR] invalid bbox")
            return None
        emb = self._embedding_from_bbox(frame_bgr, bbox)
        if emb is None:
            print("[SELECT ERROR] failed to extract target embedding")
            return None
        self.target_embedding = emb
        self.target_class_id = class_id
        self.last_bbox = bbox.copy()
        self.last_good_bbox = bbox.copy()
        self.last_score = 1.0
        self.last_mode = "INIT"
        self.frame_index = 0
        if self.verbose:
            print(f"[RESNET INIT] bbox={bbox} class_id={self.target_class_id}")
        return bbox.copy()

    def update(self, frame_bgr: np.ndarray) -> Optional[BBox]:
        self.frame_index += 1
        if self.target_embedding is None or self.last_bbox is None:
            self.last_mode = "IDLE"
            return None
        candidates = self._detect_candidates(frame_bgr)
        if self.target_class_id is not None:
            same_class = [c for c in candidates if c.cls_id == self.target_class_id]
            if same_class:
                candidates = same_class
        local_candidates = [c for c in candidates if self._candidate_in_search_window(c.bbox)]
        if local_candidates:
            candidates = local_candidates
        if not candidates:
            self.last_mode = "PRED_NO_DET"
            self.last_score = 0.0
            return self.last_bbox.copy()

        best: Optional[Candidate] = None
        best_score = -1.0
        for cand in candidates:
            emb = self._embedding_from_bbox(frame_bgr, cand.bbox)
            if emb is None:
                continue
            appearance = float(torch.dot(self.target_embedding, emb).detach().cpu().item())
            motion = self._motion_score(cand.bbox, frame_bgr.shape)
            total = self.appearance_weight * appearance + self.motion_weight * motion + 0.05 * cand.conf
            cand.appearance_score = appearance
            cand.motion_score = motion
            cand.total_score = total
            if total > best_score:
                best = cand
                best_score = total
        if best is None:
            self.last_mode = "PRED_NO_EMB"
            self.last_score = 0.0
            return self.last_bbox.copy()

        self.last_score = best.total_score
        if best.total_score >= self.min_match_score:
            old = self.last_bbox.copy()
            new = best.bbox.copy()
            self.last_bbox = self._smooth_bbox(old, new)
            self.last_good_bbox = self.last_bbox.copy()
            self.last_mode = "MATCH"
            new_emb = self._embedding_from_bbox(frame_bgr, best.bbox)
            if new_emb is not None:
                updated = 0.95 * self.target_embedding + 0.05 * new_emb
                self.target_embedding = F.normalize(updated, dim=0)
            if self.verbose and self.frame_index % 10 == 0:
                print(f"[MATCH] bbox={self.last_bbox} total={best.total_score:.3f} app={best.appearance_score:.3f} motion={best.motion_score:.3f} cls={best.cls_id} conf={best.conf:.3f}")
            return self.last_bbox.copy()

        self.last_mode = "PRED_LOW_SCORE"
        if self.verbose and self.frame_index % 10 == 0:
            print(f"[PRED_LOW_SCORE] keep={self.last_bbox} best={best.bbox} total={best.total_score:.3f} app={best.appearance_score:.3f} motion={best.motion_score:.3f}")
        return self.last_bbox.copy()

    def draw(self, frame_bgr: np.ndarray, fps: Optional[float] = None) -> np.ndarray:
        if self.last_bbox is None:
            return frame_bgr
        x, y, w, h = [int(v) for v in self.last_bbox]
        if self.last_mode == "MATCH":
            color = (0, 255, 0)
        elif self.last_mode.startswith("PRED"):
            color = (0, 255, 255)
        elif self.last_mode == "INIT":
            color = (255, 255, 0)
        else:
            color = (200, 200, 200)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        label = f"{self.last_mode} score={self.last_score:.3f}"
        if self.target_class_id is not None:
            label += f" cls={self.target_class_id}"
        if fps is not None:
            label += f" FPS={fps:.1f}"
        cv2.putText(frame_bgr, label, (max(5, x), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return frame_bgr
