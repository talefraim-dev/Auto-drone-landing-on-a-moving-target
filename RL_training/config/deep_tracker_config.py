"""
Deep tracker configuration.

Set MIXFORMER_ROOT and MIXFORMER_MODEL_PATH to your local official MixFormer repo/model.
Official repo style uses:
    lib.test.evaluation.Tracker
    tracker_name = mixformer_cvt_online
    parameter_name = baseline
"""

from __future__ import annotations


class DeepTrackerConfig:
    # Local MixFormer installation
    MIXFORMER_ROOT: str = r"C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\external\MixFormer"
    MIXFORMER_MODEL_PATH: str = r"C:\Users\Tal Efraim\PycharmProjects\Auto-drone-landing-on-a-moving-target\external\MixFormer\models\mixformer_online_22k.pth.tar"
    MIXFORMER_TRACKER_NAME: str = "mixformer_cvt_online"
    MIXFORMER_PARAM_NAME: str = "baseline"
    MIXFORMER_DATASET_NAME: str = "video"

    MIXFORMER_SEARCH_AREA_SCALE: float = 4.5
    MIXFORMER_UPDATE_INTERVAL: int = 10
    MIXFORMER_ONLINE_SIZES: int = 5
    MIXFORMER_MAX_SCORE_DECAY: float = 1.0

    # YOLO is allowed only for initial click bbox, not tracking.
    use_yolo_for_click_init: bool = True
    yolo_weights: str = "yolo11n.pt"
    yolo_conf_click: float = 0.20
    yolo_iou: float = 0.50
    yolo_imgsz: int = 960
    click_fallback_radius_px: int = 70

    # Output acceptance gates
    min_backend_score: float = 0.20
    default_score_when_missing: float = 0.75
    max_center_jump_px: float = 180.0
    max_center_jump_px_after_pred: float = 260.0

    min_area_ratio: float = 0.20
    max_area_ratio: float = 4.50
    min_aspect_ratio_change: float = 0.35
    max_aspect_ratio_change: float = 2.80

    edge_margin_px: int = 3
    reject_tiny_edge_box: bool = True
    min_edge_box_area_px: float = 1200.0

    max_pred_frames: int = 35

    # Conservative external memory/logging
    memory_enabled: bool = True
    memory_max_templates: int = 12
    memory_update_every_n_matches: int = 8
    memory_update_min_score: float = 0.70
    memory_update_max_center_jump_px: float = 45.0
    memory_update_min_area_ratio: float = 0.75
    memory_update_max_area_ratio: float = 1.35
    memory_never_update_on_edge: bool = True

    print_tracker_debug: bool = True
    print_every_n_frames: int = 30
