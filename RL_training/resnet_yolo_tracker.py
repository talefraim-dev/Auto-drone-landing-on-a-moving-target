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
import math
import threading
from typing import Dict, List, Optional, Tuple

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




# All trackers use the same frozen ImageNet ResNet18. Sharing this stateless
# feature extractor avoids allocating an identical CUDA model for the front,
# bottom and Agent-2 trackers. The cache is process-local and guarded because
# environment construction may evolve to use worker threads later.
_SHARED_RESNET_MODELS: Dict[str, torch.nn.Module] = {}
_SHARED_RESNET_LOCK = threading.Lock()
_CUDA_REID_DISABLED_AFTER_OOM = False
_CUDA_YOLO_DISABLED_AFTER_OOM = False


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cuda error: out of memory" in text
    )


def _canonical_device_name(device: str) -> str:
    try:
        return str(torch.device(device))
    except Exception:
        return str(device)


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
    """Crop an XYWH bounding box safely, including float tracker boxes."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h_img, w_img = frame_bgr.shape[:2]
    try:
        values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return None

    x, y, w, h = map(float, values[:4])
    if w <= 0.0 or h <= 0.0:
        return None

    pad_x = w * float(pad_ratio)
    pad_y = h * float(pad_ratio)

    # NumPy slice indices must be integers. Floor the upper-left corner and
    # ceil the lower-right corner so a transferred/sub-pixel box is not shrunk.
    x1 = max(0, int(math.floor(x - pad_x)))
    y1 = max(0, int(math.floor(y - pad_y)))
    x2 = min(w_img, int(math.ceil(x + w + pad_x)))
    y2 = min(h_img, int(math.ceil(y + h + pad_y)))

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

        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = _canonical_device_name(requested_device)
        self.detector_device = self.device
        self._detector_cpu_fallback_used = False
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

        # Scale-aware appearance memory. The immutable target_embedding remains
        # the original identity anchor, while this bank learns a few verified
        # views as the target grows from far -> near -> contact scale.
        self.template_bank: List[Tuple[torch.Tensor, float, int]] = []
        self.max_template_bank_size: int = 10
        self.template_scale_step_ratio: float = 1.18
        self.template_update_min_similarity: float = 0.68
        self.template_update_min_total_score: float = 0.68
        self.template_update_cooldown_frames: int = 3
        self.template_bank_updates: int = 0
        self.last_template_update_frame: int = -10_000
        self.last_candidate_scale: float = 0.0
        self.last_matched_template_scale: float = 0.0
        self.last_bank_similarity: float = 0.0

        if self.verbose:
            print(f"[YOLO+RESNET] device={self.device}, yolo_model={yolo_model_path}")

    @staticmethod
    def _new_resnet18() -> torch.nn.Module:
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
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def _build_resnet_feature_extractor(self) -> torch.nn.Module:
        global _CUDA_REID_DISABLED_AFTER_OOM

        requested = _canonical_device_name(self.device)

        with _SHARED_RESNET_LOCK:
            if requested.startswith("cuda") and _CUDA_REID_DISABLED_AFTER_OOM:
                self.device = "cpu"
                cached_cpu = _SHARED_RESNET_MODELS.get("cpu")
                if cached_cpu is not None:
                    return cached_cpu
                model = self._new_resnet18().to("cpu")
                _SHARED_RESNET_MODELS["cpu"] = model
                return model

            cached = _SHARED_RESNET_MODELS.get(requested)
            if cached is not None:
                return cached

            model = self._new_resnet18()
            actual = requested
            try:
                model = model.to(requested)
            except Exception as exc:
                if not requested.startswith("cuda") or not _is_cuda_oom(exc):
                    raise

                # Unreal Engine can consume most laptop VRAM. A ReID CUDA OOM
                # must not abort the whole flight: keep control logic untouched
                # and move only the frozen embedding extractor to CPU.
                _CUDA_REID_DISABLED_AFTER_OOM = True
                self.device = "cpu"
                actual = "cpu"
                del model
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                cached_cpu = _SHARED_RESNET_MODELS.get(actual)
                if cached_cpu is not None:
                    print(
                        "[VRAM GUARD] CUDA OOM while loading ResNet18; "
                        "reusing shared CPU ReID extractor."
                    )
                    return cached_cpu

                model = self._new_resnet18().to(actual)
                print(
                    "[VRAM GUARD] CUDA OOM while loading ResNet18; "
                    "using shared CPU ReID extractor. Flight control is unchanged."
                )

            _SHARED_RESNET_MODELS[actual] = model
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

    @staticmethod
    def _bbox_scale(bbox: BBox, frame_shape: Tuple[int, int, int]) -> float:
        """Return a resolution-independent linear scale for an XYWH bbox."""
        h_img, w_img = frame_shape[:2]
        _, _, bw, bh = [float(v) for v in bbox]
        area_fraction = max(1.0, bw * bh) / max(1.0, float(w_img * h_img))
        return float(math.sqrt(area_fraction))

    def _reset_template_bank(self, embedding: torch.Tensor, bbox: Optional[BBox], frame_shape) -> None:
        self.template_bank = []
        self.template_bank_updates = 0
        self.last_template_update_frame = -10_000
        if embedding is None or bbox is None or frame_shape is None:
            return
        scale = self._bbox_scale(bbox, frame_shape)
        self.template_bank.append((embedding.detach().clone(), float(scale), int(self.frame_index)))
        self.last_candidate_scale = float(scale)
        self.last_matched_template_scale = float(scale)
        self.last_bank_similarity = 1.0

    def restore_target_embedding(self, embedding: torch.Tensor) -> None:
        """Restore the immutable identity anchor and reset per-flight scale memory."""
        self.target_embedding = F.normalize(embedding.detach().to(self.device), dim=0)
        self.template_bank = []
        self.template_bank_updates = 0
        self.last_template_update_frame = -10_000
        self.last_candidate_scale = 0.0
        self.last_matched_template_scale = 0.0
        self.last_bank_similarity = 0.0

    def _bank_similarity(self, embedding: torch.Tensor, candidate_scale: float) -> Tuple[float, float]:
        """Best appearance similarity, mildly preferring templates of nearby scale."""
        entries = self.template_bank
        if not entries and self.target_embedding is not None:
            entries = [(self.target_embedding, float(candidate_scale), 0)]

        best_adjusted = -1.0
        best_raw = -1.0
        best_scale = float(candidate_scale)
        for template, template_scale, _ in entries:
            raw = float(torch.dot(template, embedding).detach().cpu().item())
            scale_distance = abs(math.log(max(candidate_scale, 1e-6) / max(template_scale, 1e-6)))
            # Small preference only; identity remains the dominant signal.
            adjusted = raw - 0.025 * min(scale_distance, 2.0)
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_raw = raw
                best_scale = float(template_scale)
        return float(best_raw), float(best_scale)

    def _maybe_add_scale_template(
        self,
        embedding: torch.Tensor,
        candidate_scale: float,
        similarity: float,
        total_score: float,
    ) -> bool:
        """Add a verified template only after a meaningful scale transition."""
        if embedding is None:
            return False
        if similarity < self.template_update_min_similarity:
            return False
        if total_score < self.template_update_min_total_score:
            return False
        if self.frame_index - self.last_template_update_frame < self.template_update_cooldown_frames:
            return False

        if not self.template_bank:
            self.template_bank.append((embedding.detach().clone(), float(candidate_scale), int(self.frame_index)))
            self.last_template_update_frame = int(self.frame_index)
            self.template_bank_updates += 1
            return True

        nearest_log_distance = min(
            abs(math.log(max(candidate_scale, 1e-6) / max(scale, 1e-6)))
            for _, scale, _ in self.template_bank
        )
        required = math.log(self.template_scale_step_ratio)
        if nearest_log_distance < required:
            return False

        self.template_bank.append((embedding.detach().clone(), float(candidate_scale), int(self.frame_index)))
        self.template_bank.sort(key=lambda row: row[1])

        # Preserve coverage across the scale range instead of keeping only the
        # newest frames. If full, remove the most redundant interior template.
        if len(self.template_bank) > self.max_template_bank_size:
            best_remove = None
            best_gap = float("inf")
            for idx in range(1, len(self.template_bank) - 1):
                prev_scale = self.template_bank[idx - 1][1]
                cur_scale = self.template_bank[idx][1]
                next_scale = self.template_bank[idx + 1][1]
                gap = abs(math.log(max(cur_scale, 1e-6) / max(prev_scale, 1e-6))) + abs(
                    math.log(max(next_scale, 1e-6) / max(cur_scale, 1e-6))
                )
                if gap < best_gap:
                    best_gap = gap
                    best_remove = idx
            if best_remove is None:
                best_remove = 1
            self.template_bank.pop(best_remove)

        self.last_template_update_frame = int(self.frame_index)
        self.template_bank_updates += 1
        if self.verbose:
            print(
                f"[SCALE BANK] add scale={candidate_scale:.4f} sim={similarity:.3f} "
                f"total={total_score:.3f} bank={len(self.template_bank)} updates={self.template_bank_updates}"
            )
        return True

    def _detect_candidates(self, frame_bgr: np.ndarray) -> List[Candidate]:
        global _CUDA_YOLO_DISABLED_AFTER_OOM

        h_img, w_img = frame_bgr.shape[:2]
        if str(self.detector_device).startswith("cuda") and _CUDA_YOLO_DISABLED_AFTER_OOM:
            self.detector_device = "cpu"
            self._detector_cpu_fallback_used = True

        try:
            results = self.yolo.predict(
                frame_bgr,
                conf=self.yolo_conf,
                verbose=False,
                device=self.detector_device,
            )
        except Exception as exc:
            if (
                self._detector_cpu_fallback_used
                or not str(self.detector_device).startswith("cuda")
                or not _is_cuda_oom(exc)
            ):
                raise

            # Retry once on CPU if the first YOLO CUDA allocation cannot fit
            # beside Unreal. Other trackers then avoid repeating the same OOM.
            _CUDA_YOLO_DISABLED_AFTER_OOM = True
            self._detector_cpu_fallback_used = True
            self.detector_device = "cpu"
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                "[VRAM GUARD] CUDA OOM during YOLO inference; "
                "retrying visual detection on CPU."
            )
            results = self.yolo.predict(
                frame_bgr,
                conf=self.yolo_conf,
                verbose=False,
                device="cpu",
            )
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
        self._reset_template_bank(emb, bbox, frame_bgr.shape)
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
        best_embedding: Optional[torch.Tensor] = None
        best_scale = 0.0
        best_template_scale = 0.0
        for cand in candidates:
            emb = self._embedding_from_bbox(frame_bgr, cand.bbox)
            if emb is None:
                continue
            candidate_scale = self._bbox_scale(cand.bbox, frame_bgr.shape)
            appearance, matched_template_scale = self._bank_similarity(emb, candidate_scale)
            motion = self._motion_score(cand.bbox, frame_bgr.shape)
            total = self.appearance_weight * appearance + self.motion_weight * motion + 0.05 * cand.conf
            cand.appearance_score = appearance
            cand.motion_score = motion
            cand.total_score = total
            if total > best_score:
                best = cand
                best_score = total
                best_embedding = emb
                best_scale = float(candidate_scale)
                best_template_scale = float(matched_template_scale)
        if best is None:
            self.last_mode = "PRED_NO_EMB"
            self.last_score = 0.0
            return self.last_bbox.copy()

        self.last_score = best.total_score
        self.last_candidate_scale = float(best_scale)
        self.last_matched_template_scale = float(best_template_scale)
        self.last_bank_similarity = float(best.appearance_score)
        if best.total_score >= self.min_match_score:
            old = self.last_bbox.copy()
            new = best.bbox.copy()
            self.last_bbox = self._smooth_bbox(old, new)
            self.last_good_bbox = self.last_bbox.copy()
            self.last_mode = "MATCH"

            # Keep the original identity anchor immutable. Learn additional
            # templates only at verified, materially different scales.
            if best_embedding is not None:
                self._maybe_add_scale_template(
                    best_embedding,
                    best_scale,
                    best.appearance_score,
                    best.total_score,
                )
            if self.verbose and self.frame_index % 10 == 0:
                print(
                    f"[MATCH] bbox={self.last_bbox} total={best.total_score:.3f} "
                    f"app={best.appearance_score:.3f} motion={best.motion_score:.3f} "
                    f"scale={best_scale:.4f} bank={len(self.template_bank)} "
                    f"cls={best.cls_id} conf={best.conf:.3f}"
                )
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
