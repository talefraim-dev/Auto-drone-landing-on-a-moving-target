# train_simulator_RL.py
# Full single-file: CosyAirSim RL training with robust tracking:
# - CSRT on GRAY (stable) + Kalman prediction
# - Template re-detection (matchTemplate) around predicted bbox when CSRT fails/gates fail
# - Confidence driven by quality (IoU/scale/template score) so it won't "stick" at 1
# - ROI drawing as "cut-corner" rectangle (like the old style)
# - YAW action (yaw_rate) included
# - checkpoints + RESUME

import os
import glob
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

import cosysairsim as airsim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


# -------------------------
# Config
# -------------------------
@dataclass
class EnvConfig:
    cam_name: str = "0"
    img_w: int = 640
    img_h: int = 480

    # RL stepping
    step_hz: float = 10.0
    dt_cmd: float = 0.10
    max_episode_steps: int = 500

    # Limits (BODY frame)
    max_vx: float = 2.0  # forward
    max_vy: float = 2.0  # right
    max_vz: float = 1.0  # NED down (+)

    # yaw rate control (deg/s) for YawMode(is_rate=True)
    max_yaw_rate_dps: float = 70.0

    # Reward shaping
    target_scale: float = 1.8
    w_center: float = 2.0
    w_scale: float = 0.6
    w_action: float = 0.05
    w_not_locked: float = 2.0
    w_terminate_lost: float = 10.0

    conf_terminate: float = 0.12

    # Tracking / gating (CSRT + Kalman)
    lost_to_search: int = 18

    # gates for CSRT measurement acceptance
    iou_gate: float = 0.02
    scale_ratio_min: float = 0.35
    scale_ratio_max: float = 2.8

    # Robustness additions
    # Template re-detect parameters
    tmpl_enable: bool = True
    tmpl_method: int = cv2.TM_CCOEFF_NORMED  # robust, normalized
    tmpl_min_score_locked: float = 0.55     # accept re-detect as LOCKED if score >= this
    tmpl_min_score_predict: float = 0.40    # accept as PREDICTING if score >= this
    tmpl_search_margin: float = 1.6         # search window multiplier around predicted bbox
    tmpl_max_window: int = 260              # cap search window size (pixels)
    tmpl_update_rate: float = 0.08          # how fast template updates when locked (EMA)

    # Confidence dynamics
    conf_inc_good: float = 0.10
    conf_dec_bad: float = 0.08

    # Preprocess for tracking/template
    track_blur_ksize: int = 5   # 0/1 disables
    track_eq_hist: bool = True  # helps lighting changes a bit

    # UI
    window_name: str = "CosyAirSim RL"
    ui_w: int = 800
    ui_h: int = 600

    # ROI color thresholds by confidence
    conf_low: float = 0.35
    conf_high: float = 0.70

    # Camera tilt suggestion
    cam_pitch_down_deg: float = 22.0

    # Cut-corner ROI drawing
    corner_len_ratio: float = 0.22  # fraction of min(w,h)
    corner_len_min: int = 10
    corner_len_max: int = 34
    corner_thickness: int = 3


# -------------------------
# Utils
# -------------------------
def _create_csrt():
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        return cv2.TrackerCSRT_create()


def _calc_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xA = max(ax, bx)
    yA = max(ay, by)
    xB = min(ax + aw, bx + bw)
    yB = min(ay + ah, by + bh)
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = aw * ah + bw * bh - inter + 1e-6
    return float(inter / union)


def _roi_color_by_conf(cfg: EnvConfig, conf: float) -> Tuple[int, int, int]:
    # BGR
    if conf < cfg.conf_low:
        return (0, 0, 255)      # red
    if conf > cfg.conf_high:
        return (0, 255, 0)      # green
    return (0, 255, 255)        # yellow


def _draw_cut_corner_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                          color: Tuple[int, int, int], cfg: EnvConfig):
    """Draw rectangle with cut corners (L-shapes), like the reference image."""
    t = int(cfg.corner_thickness)
    L = int(min(max(cfg.corner_len_min, int(min(w, h) * cfg.corner_len_ratio)), cfg.corner_len_max))

    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # top-left
    cv2.line(img, (x1, y1), (x1 + L, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + L), color, t, cv2.LINE_AA)

    # top-right
    cv2.line(img, (x2, y1), (x2 - L, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + L), color, t, cv2.LINE_AA)

    # bottom-left
    cv2.line(img, (x1, y2), (x1 + L, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - L), color, t, cv2.LINE_AA)

    # bottom-right
    cv2.line(img, (x2, y2), (x2 - L, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - L), color, t, cv2.LINE_AA)


def _draw_double_carets_down(img: np.ndarray, center: Tuple[int, int], roi_w: int, roi_h: int):
    """
    Draw "^^" pointing DOWN (without a stem), always GREEN.
    Implement as two V-shapes (down chevrons) stacked.
    """
    cx, cy = center
    base = int(max(6, min(roi_w, roi_h) * 0.18))
    base = int(min(base, 28))
    gap = int(max(4, base * 0.45))
    color = (0, 255, 0)
    thick = 2

    def chevron_down(xc, yc, s):
        cv2.line(img, (xc - s, yc - s), (xc, yc), color, thick, cv2.LINE_AA)
        cv2.line(img, (xc + s, yc - s), (xc, yc), color, thick, cv2.LINE_AA)

    chevron_down(cx, cy - gap // 2, base)
    chevron_down(cx, cy + gap // 2, base)


def _preprocess_for_tracking(bgr: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Stable tracking input: GRAY (+ optional hist eq + blur)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if cfg.track_eq_hist:
        gray = cv2.equalizeHist(gray)
    k = int(cfg.track_blur_ksize)
    if k >= 3 and k % 2 == 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


# -------------------------
# AirSim: get frame (FAST path)
# -------------------------
def get_frame(client: airsim.MultirotorClient, cfg: EnvConfig) -> Optional[np.ndarray]:
    req = airsim.ImageRequest(cfg.cam_name, airsim.ImageType.Scene, False, False)
    resp = client.simGetImages([req])[0]
    if resp.height == 0 or resp.width == 0:
        return None
    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    img = img1d.reshape(resp.height, resp.width, 3)

    if (resp.width, resp.height) != (cfg.img_w, cfg.img_h):
        img = cv2.resize(img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)
    return img


def try_set_camera_tilt(client: airsim.MultirotorClient, cfg: EnvConfig):
    """Best-effort runtime tilt. If not supported, set in Unreal/settings.json."""
    try:
        pitch = -np.deg2rad(cfg.cam_pitch_down_deg)
        q = airsim.to_quaternion(pitch, 0.0, 0.0)
        pose = airsim.Pose(airsim.Vector3r(0, 0, 0), q)
        client.simSetCameraPose(cfg.cam_name, pose)
        print(f"[CAM] Tilted camera {cfg.cam_name} down by ~{cfg.cam_pitch_down_deg} deg (runtime).")
    except Exception:
        print("[CAM] Runtime camera tilt not supported in your CosyAirSim build.")
        print("      Set it in settings.json / Unreal instead (Pitch ~ -20..-25 deg).")


# -------------------------
# Robust Tracking: CSRT + Kalman + Template Re-detect
# -------------------------
class MissionManager:
    """
    States: IDLE / LOCKED / PREDICTING / SEARCHING
    Observation (7):
      [err_x, err_y, scale, vx_est, vy_est, conf, status_idx]
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.tracker = _create_csrt()

        # Kalman state: [x,y,w,h,vx,vy,vw,vh], measurement: [x,y,w,h]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:, 4:] *= 10.0
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.10

        self.state: str = "IDLE"
        self.conf: float = 0.0
        self.lost_counter: int = 0
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self._orig_w: float = 100.0
        self._t_last: float = time.perf_counter()

        # template (for re-detection)
        self._tmpl: Optional[np.ndarray] = None  # grayscale template patch
        self._tmpl_w: int = 0
        self._tmpl_h: int = 0

    def _init_tracker(self, frame_gray: np.ndarray, bbox_xywh: Tuple[float, float, float, float]) -> bool:
        x, y, w, h = bbox_xywh
        x, y, w, h = float(x), float(y), float(w), float(h)
        self.tracker = _create_csrt()
        ok = self.tracker.init(frame_gray, (x, y, w, h))
        return bool(ok)

    def _set_template_from_bbox(self, frame_gray: np.ndarray, bbox_xywh: Tuple[float, float, float, float]):
        x, y, w, h = map(int, map(round, bbox_xywh))
        H, W = frame_gray.shape[:2]
        w = max(12, min(w, W - x))
        h = max(12, min(h, H - y))
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        patch = frame_gray[y:y + h, x:x + w].copy()
        self._tmpl = patch
        self._tmpl_w = patch.shape[1]
        self._tmpl_h = patch.shape[0]

    def _update_template_ema(self, frame_gray: np.ndarray, bbox_xywh: Tuple[float, float, float, float]):
        """Slowly adapt template when stable/locked."""
        if self._tmpl is None:
            self._set_template_from_bbox(frame_gray, bbox_xywh)
            return

        x, y, w, h = map(int, map(round, bbox_xywh))
        H, W = frame_gray.shape[:2]
        w = max(12, min(w, W - x))
        h = max(12, min(h, H - y))
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))

        patch = frame_gray[y:y + h, x:x + w]
        if patch.size == 0:
            return

        # resize patch to template size (to keep matchTemplate consistent)
        patch_rs = cv2.resize(patch, (self._tmpl_w, self._tmpl_h), interpolation=cv2.INTER_AREA)
        alpha = float(self.cfg.tmpl_update_rate)
        self._tmpl = cv2.addWeighted(self._tmpl, 1.0 - alpha, patch_rs, alpha, 0.0)

    def _template_redetect(self, frame_gray: np.ndarray, pred_xywh: Tuple[float, float, float, float]) -> Tuple[bool, Optional[Tuple[float, float, float, float]], float]:
        """Try to re-detect target around predicted bbox using matchTemplate."""
        if not self.cfg.tmpl_enable or self._tmpl is None:
            return False, None, 0.0

        H, W = frame_gray.shape[:2]
        px, py, pw, ph = pred_xywh
        pw = max(float(pw), float(self._tmpl_w))
        ph = max(float(ph), float(self._tmpl_h))

        # search window around prediction
        margin = float(self.cfg.tmpl_search_margin)
        cx = px + pw / 2.0
        cy = py + ph / 2.0
        win_w = int(min(self.cfg.tmpl_max_window, max(self._tmpl_w + 10, pw * margin)))
        win_h = int(min(self.cfg.tmpl_max_window, max(self._tmpl_h + 10, ph * margin)))

        x1 = int(np.clip(cx - win_w / 2, 0, W - 1))
        y1 = int(np.clip(cy - win_h / 2, 0, H - 1))
        x2 = int(np.clip(cx + win_w / 2, 0, W))
        y2 = int(np.clip(cy + win_h / 2, 0, H))

        roi = frame_gray[y1:y2, x1:x2]
        if roi.shape[0] < self._tmpl_h + 2 or roi.shape[1] < self._tmpl_w + 2:
            return False, None, 0.0

        res = cv2.matchTemplate(roi, self._tmpl, self.cfg.tmpl_method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)

        # best location (top-left in roi coords)
        mx, my = max_loc
        bx = float(x1 + mx)
        by = float(y1 + my)
        bw = float(self._tmpl_w)
        bh = float(self._tmpl_h)

        # clamp
        bx = float(np.clip(bx, 0.0, W - bw))
        by = float(np.clip(by, 0.0, H - bh))

        return True, (bx, by, bw, bh), score

    def start(self, frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
        H, W = frame_bgr.shape[:2]
        x, y, w, h = [int(v) for v in roi]
        w, h = max(w, 12), max(h, 12)
        x = int(np.clip(x, 0, W - w))
        y = int(np.clip(y, 0, H - h))

        frame_gray = _preprocess_for_tracking(frame_bgr, self.cfg)

        ok = self._init_tracker(frame_gray, (x, y, w, h))
        if not ok:
            self.state = "IDLE"
            self.conf = 0.0
            self.lost_counter = 0
            self.last_bbox = None
            self._tmpl = None
            return False

        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.state = "LOCKED"
        self.conf = 1.0
        self.lost_counter = 0
        self.last_bbox = (float(x), float(y), float(w), float(h))
        self._orig_w = float(w)
        self._t_last = time.perf_counter()

        # init template from initial ROI
        self._set_template_from_bbox(frame_gray, self.last_bbox)
        return True

    def update(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
        H, W = frame_bgr.shape[:2]
        now = time.perf_counter()
        dt = max(now - self._t_last, 1e-3)
        self._t_last = now

        if self.state == "IDLE":
            obs = np.zeros(7, dtype=np.float32)
            return obs, self.state

        # Kalman transition with dt
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = dt

        pred = self.kf.predict().flatten()
        px, py = float(pred[0]), float(pred[1])
        pw, ph = max(float(pred[2]), 12.0), max(float(pred[3]), 12.0)

        # defaults from prediction
        x, y, w, h = px, py, pw, ph
        success = False
        bbox = None
        quality_good = False
        tmpl_score = 0.0

        frame_gray = _preprocess_for_tracking(frame_bgr, self.cfg)

        # 1) CSRT attempt (on stable gray)
        if self.state in ("LOCKED", "PREDICTING"):
            success, bbox = self.tracker.update(frame_gray)

        # 2) If CSRT succeeded, gate it
        if success and bbox is not None:
            bx, by, bw, bh = map(float, bbox)
            bw, bh = max(bw, 12.0), max(bh, 12.0)
            bx = float(np.clip(bx, 0.0, W - bw))
            by = float(np.clip(by, 0.0, H - bh))

            iou = _calc_iou((bx, by, bw, bh), (px, py, pw, ph))
            sratio = bw / (pw + 1e-6)

            if (iou >= self.cfg.iou_gate) and (self.cfg.scale_ratio_min <= sratio <= self.cfg.scale_ratio_max):
                meas = (bx, by, bw, bh)
                self.kf.correct(np.array(meas, dtype=np.float32))
                self.state = "LOCKED"
                self.lost_counter = 0
                self.last_bbox = meas
                x, y, w, h = meas
                quality_good = True
            else:
                # gated out
                success = False

        # 3) If CSRT failed/gated out, try template re-detect around prediction
        if not success:
            red_ok, red_bbox, tmpl_score = self._template_redetect(frame_gray, (px, py, pw, ph))
            if red_ok and red_bbox is not None:
                bx, by, bw, bh = red_bbox
                # accept if score strong enough
                if tmpl_score >= self.cfg.tmpl_min_score_locked:
                    meas = (bx, by, bw, bh)
                    self.kf.correct(np.array(meas, dtype=np.float32))
                    self.state = "LOCKED"
                    self.lost_counter = 0
                    self.last_bbox = meas
                    x, y, w, h = meas
                    quality_good = True

                    # re-init tracker from this detection (critical for stability)
                    self._init_tracker(frame_gray, meas)

                elif tmpl_score >= self.cfg.tmpl_min_score_predict:
                    # weaker: keep predicting but use bbox for UI/err (do not fully trust)
                    self.state = "PREDICTING"
                    self.lost_counter += 1
                    self.last_bbox = (bx, by, bw, bh)
                    x, y, w, h = bx, by, bw, bh
                    quality_good = False
                else:
                    # too weak -> normal predict/search
                    quality_good = False

            else:
                quality_good = False

        # 4) If still not good -> move to PREDICTING/SEARCHING on Kalman prediction
        if not quality_good:
            self.lost_counter += 1
            if self.lost_counter < self.cfg.lost_to_search:
                self.state = "PREDICTING"
            else:
                self.state = "SEARCHING"

            w, h = pw, ph
            x = float(np.clip(px, 0.0, W - w))
            y = float(np.clip(py, 0.0, H - h))
            self.last_bbox = (x, y, w, h)

        # 5) Confidence update (prevents "sticking at 1")
        if quality_good:
            self.conf = min(1.0, self.conf + self.cfg.conf_inc_good)
            # update template slowly when locked (adapts to mild appearance changes)
            if self.state == "LOCKED" and self.last_bbox is not None:
                self._update_template_ema(frame_gray, self.last_bbox)
        else:
            self.conf = max(0.0, self.conf - self.cfg.conf_dec_bad)

        # clamp bbox
        w = float(np.clip(w, 12.0, W))
        h = float(np.clip(h, 12.0, H))
        x = float(np.clip(x, 0.0, W - w))
        y = float(np.clip(y, 0.0, H - h))

        cx, cy = x + w / 2.0, y + h / 2.0
        err_x = (cx - W / 2.0) / (W / 2.0)
        err_y = (cy - H / 2.0) / (H / 2.0)
        scale = w / (self._orig_w + 1e-6)

        st = self.kf.statePost.flatten()
        vx_est = float(st[4]) / 100.0
        vy_est = float(st[5]) / 100.0
        status_idx = {"IDLE": 0, "LOCKED": 1, "PREDICTING": 2, "SEARCHING": 3}[self.state]

        obs = np.array([err_x, err_y, scale, vx_est, vy_est, self.conf, float(status_idx)], dtype=np.float32)
        return obs, self.state


# -------------------------
# Gym Env
# -------------------------
class CosyDroneRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, cfg: EnvConfig, render_mode: str = "human"):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        try_set_camera_tilt(self.client, self.cfg)

        self.mm = MissionManager(cfg)

        # Action: [vx_body, vy_body, vz_down, yaw_rate] in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self._step_period = 1.0 / float(self.cfg.step_hz)
        self._episode_step = 0
        self._last_action = np.zeros(4, dtype=np.float32)

        self._roi: Optional[Tuple[int, int, int, int]] = None

        # FPS counters
        self._t_step_last = time.perf_counter()
        self._t_grab_last = time.perf_counter()
        self._step_fps = 0.0
        self._grab_fps = 0.0
        self._fps_print_last = time.perf_counter()

        if self.render_mode == "human":
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cfg.window_name, self.cfg.ui_w, self.cfg.ui_h)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        self._last_action[:] = 0.0

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(0.25)

        try_set_camera_tilt(self.client, self.cfg)

        frame = None
        for _ in range(40):
            frame = self._grab_frame()
            if frame is not None:
                break
            time.sleep(0.02)
        if frame is None:
            raise RuntimeError("No image from AirSim camera")

        # select ROI once
        if self._roi is None:
            cv2.namedWindow("Select ROI (once)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select ROI (once)", 1280, 720)
            r = cv2.selectROI("Select ROI (once)", frame, False)
            cv2.destroyWindow("Select ROI (once)")
            cv2.waitKey(1)
            if r[2] <= 0 or r[3] <= 0:
                r = (self.cfg.img_w // 2 - 50, self.cfg.img_h // 2 - 50, 120, 120)
            self._roi = tuple(map(int, r))

        ok = self.mm.start(frame, self._roi)
        if not ok:
            self._roi = (self.cfg.img_w // 2 - 60, self.cfg.img_h // 2 - 60, 120, 120)
            self.mm.start(frame, self._roi)

        obs, state = self.mm.update(frame)
        if self.render_mode == "human":
            self._render(frame, obs, state, extra="RESET")
        return obs, {}

    def _grab_frame(self) -> Optional[np.ndarray]:
        now = time.perf_counter()
        dt = now - self._t_grab_last
        if dt > 1e-6:
            self._grab_fps = 1.0 / dt
        self._t_grab_last = now
        return get_frame(self.client, self.cfg)

    def step(self, action):
        t0 = time.perf_counter()

        dt_step = t0 - self._t_step_last
        if dt_step > 1e-6:
            self._step_fps = 1.0 / dt_step
        self._t_step_last = t0

        self._episode_step += 1

        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self._last_action = action

        vx = float(action[0] * self.cfg.max_vx)
        vy = float(action[1] * self.cfg.max_vy)
        vz = float(action[2] * self.cfg.max_vz)  # NED down (+)
        yaw_rate = float(action[3] * self.cfg.max_yaw_rate_dps)

        self.client.moveByVelocityBodyFrameAsync(
            vx, vy, vz, self.cfg.dt_cmd,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )

        frame = None
        for _ in range(8):
            frame = self._grab_frame()
            if frame is not None:
                break
            time.sleep(0.005)

        if frame is None:
            obs = np.zeros(7, dtype=np.float32)
            return obs, -100.0, True, False, {"error": "no_frame"}

        obs, state = self.mm.update(frame)
        err_x, err_y, scale, _, _, conf, status_idx = map(float, obs)

        # Reward
        r_center = -(abs(err_x) + abs(err_y)) * self.cfg.w_center
        r_scale = -abs(self.cfg.target_scale - scale) * self.cfg.w_scale
        r_action = -float(np.sum(action * action)) * self.cfg.w_action

        locked = (int(round(status_idx)) == 1)
        r_lock = 0.0 if locked else -self.cfg.w_not_locked

        reward = float(r_center + r_scale + r_action + r_lock)

        terminated = False
        truncated = False

        if conf < self.cfg.conf_terminate:
            terminated = True
            reward -= self.cfg.w_terminate_lost

        if self._episode_step >= self.cfg.max_episode_steps:
            truncated = True

        now = time.perf_counter()
        if now - self._fps_print_last >= 1.0:
            print(f"[FPS] step={self._step_fps:5.1f} grab={self._grab_fps:5.1f} | "
                  f"state={state:<10} conf={conf:4.2f} err=({err_x:+.2f},{err_y:+.2f}) "
                  f"scale={scale:4.2f} act=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f},{action[3]:+.2f}) "
                  f"yaw_rate(dps)={yaw_rate:+.1f}")
            self._fps_print_last = now

        if self.render_mode == "human":
            self._render(frame, obs, state)

        # enforce step rate
        dt = time.perf_counter() - t0
        sleep_left = self._step_period - dt
        if sleep_left > 0:
            time.sleep(sleep_left)

        info = {"state": state, "conf": conf}
        return obs, reward, terminated, truncated, info

    def _render(self, frame_bgr: np.ndarray, obs: np.ndarray, state_txt: str, extra: str = ""):
        disp = frame_bgr.copy()
        err_x, err_y, scale, vx_est, vy_est, conf, status_idx = obs.tolist()

        if self.mm.last_bbox is not None:
            x, y, w, h = map(int, map(round, self.mm.last_bbox))
            x = max(0, min(x, self.cfg.img_w - 2))
            y = max(0, min(y, self.cfg.img_h - 2))
            w = max(12, min(w, self.cfg.img_w - x - 1))
            h = max(12, min(h, self.cfg.img_h - y - 1))

            color = _roi_color_by_conf(self.cfg, float(conf))

            # draw cut-corner ROI (not full rectangle)
            _draw_cut_corner_rect(disp, x, y, w, h, color, self.cfg)

            cx, cy = x + w // 2, y + h // 2
            _draw_double_carets_down(disp, (cx, cy), roi_w=w, roi_h=h)

        cv2.putText(disp, f"{state_txt} {extra} conf={conf:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

        ui = cv2.resize(disp, (self.cfg.ui_w, self.cfg.ui_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.cfg.window_name, ui)
        cv2.waitKey(1)

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass


# -------------------------
# Training
# -------------------------
def _latest_checkpoint(ckpt_dir: str, prefix: str = "ppo_cosy_") -> Optional[str]:
    pattern = os.path.join(ckpt_dir, f"{prefix}*.zip")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    cfg = EnvConfig(
        step_hz=10.0,
        dt_cmd=0.10,
        max_episode_steps=500,
        max_vx=2.0, max_vy=2.0, max_vz=1.0,
        max_yaw_rate_dps=70.0,
        target_scale=1.8,
        ui_w=800, ui_h=600,
        cam_pitch_down_deg=22.0,

        # robustness knobs
        iou_gate=0.02,
        scale_ratio_min=0.35,
        scale_ratio_max=2.8,
        tmpl_enable=True,
        tmpl_min_score_locked=0.55,
        tmpl_min_score_predict=0.40,
        tmpl_search_margin=1.6,
        tmpl_max_window=260,
        tmpl_update_rate=0.08,
        conf_inc_good=0.10,
        conf_dec_bad=0.08,
        track_blur_ksize=5,
        track_eq_hist=True
    )

    env = DummyVecEnv([lambda: CosyDroneRLEnv(cfg=cfg, render_mode="human")])

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=ckpt_dir,
        name_prefix="ppo_cosy"
    )

    latest = _latest_checkpoint(ckpt_dir, prefix="ppo_cosy_")
    if latest is not None:
        print(f"[RESUME] Loading checkpoint: {latest}")
        model = PPO.load(latest, env=env, device="cuda")
    else:
        print("[START] New PPO model (MlpPolicy)")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95
        )

    total = 100_000
    final_path = os.path.join(os.getcwd(), "ppo_cosy_final.zip")
    print(f"Training for {total} timesteps... (Ctrl+C to stop; will save)")

    try:
        model.learn(total_timesteps=total, callback=checkpoint_cb)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Saving model...")
    finally:
        model.save(final_path)
        print(f"[SAVED] {final_path}")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
