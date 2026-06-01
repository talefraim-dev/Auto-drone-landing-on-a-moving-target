from dataclasses import dataclass


@dataclass
class TrackerConfig:
    """
    Final tracker config.
    No environment variables are used.

    Tracker architecture:
        YOLO + legacy RGB histogram fingerprint
        + Kalman velocity EMA
        + hard search window
        + yaw image-space shift
        + freeze instead of target switching.
    """

    yolo_weights: str = "yolo11n.pt"
    yolo_conf: float = 0.25
    click_conf: float = 0.30
    yolo_iou: float = 0.55
    imgsz: int = 640

    match_threshold: float = 0.56
    autolock_threshold: float = 0.40
    feature_min_threshold: float = 0.40

    class_gate: bool = True

    search_window_scale: float = 3.0
    search_min_pad_px: float = 50.0
    search_extra_yaw_pad_px: float = 20.0

    # Active reacquisition.
    # Local search window prevents target switching; this controlled full-frame
    # search recovers the target when the bbox drifted away but the target is
    # still visible in the frame.
    active_reacquire_enabled: bool = True
    reacquire_after_pred_frames: int = 8
    reacquire_every_n_frames: int = 3
    reacquire_min_score: float = 0.52
    reacquire_feat_min: float = 0.38
    reacquire_max_candidates: int = 8

    # Appearance bank + robust mid-pass.
    # Target identity is represented by several jittered fingerprints, not a
    # single fragile vector. Matching uses a robust mid-band score.
    appearance_bank_enabled: bool = True
    appearance_bank_max_size: int = 24
    appearance_bank_jitter_px: int = 8
    appearance_bank_scale_jitter: float = 0.10
    appearance_bank_update_min_score: float = 0.74
    appearance_bank_update_every_n_matches: int = 3
    midpass_low_quantile: float = 0.25
    midpass_high_quantile: float = 0.85
    candidate_midpass_variants: int = 7

    size_min_score: float = 0.12
    aspect_min_score: float = 0.10

    max_pred_frames: int = 90
    freeze_without_yaw: bool = True

    yaw_shift_enabled: bool = True
    yaw_cmd_deadband: float = 0.002
    yaw_pix_per_cmd: float = 500.0
    yaw_shift_sign: float = 1.0
    yaw_max_shift_px: float = 35.0

    feature_update_min: float = 0.72
    feature_update_alpha: float = 0.01

    draw_search_window: bool = True
    draw_expected: bool = True
