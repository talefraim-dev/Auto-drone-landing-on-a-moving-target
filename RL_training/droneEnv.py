import sys
import os
import numpy as np
import cv2
import torch
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import types
import importlib.util
import cosysairsim as airsim

# --- נתיבי SiamMask ---
current_dir = Path(__file__).resolve().parent
siammask_root = current_dir / "SiamMask"
tools_path = siammask_root / "tools"
for p in [siammask_root, tools_path]:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

mock_region = types.ModuleType("region")
mock_region.vot_overlap = lambda *args, **kwargs: 0
mock_region.vot_float2str = lambda *args, **kwargs: ""
sys.modules["utils.pyvotkit.region"] = mock_region

test_py_path = tools_path / "test.py"
spec = importlib.util.spec_from_file_location("siammask_test", str(test_py_path))
siammask_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(siammask_test)

from utils.config_helper import load_config
from custom import Custom


class droneEnv(gym.Env):
    def __init__(self, model_path, config_path):
        super(droneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        args = type('Args', (), {'config': config_path})()
        cfg = load_config(args)
        self.hp = cfg.get('hp', {})
        for k, v in self.hp.items():
            if isinstance(v, dict): self.hp[k] = list(v.values())[0]

        self.hp.update({
            'lr': 0.15,
            'penalty_k': 0.40,
            'window_influence': 0.65,
            'seg_thr': 0.20
        })

        raw_model = Custom(anchors=cfg['anchors'])
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        raw_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        raw_model.eval().to(self.device)

        self.tracker_model = {'model': raw_model, 'anchors': cfg['anchors']}
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.static_target_roi = None
        self.steps_count = 0
        self.episode_count = 0

    def _get_frame(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if not responses or not responses[0].image_data_uint8:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        frame = img1d.reshape(responses[0].height, responses[0].width, 3)
        return cv2.resize(frame, (1280, 720))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.steps_count = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        frame = self._get_frame()
        if self.static_target_roi is None:
            roi = cv2.selectROI("Select_Target", frame, False)
            cv2.destroyWindow("Select_Target")
            self.static_target_roi = roi

        roi = self.static_target_roi
        self.state = siammask_test.siamese_init(frame, np.array([roi[0] + roi[2] / 2, roi[1] + roi[3] / 2]),
                                                np.array([roi[2], roi[3]]), self.tracker_model, self.hp,
                                                device=self.device)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.steps_count += 1
        self.client.moveByVelocityBodyFrameAsync(float(action[0]) * 3, float(action[1]) * 3, float(action[2]) * 2,
                                                 0.1).join()

        frame = self._get_frame()
        self.state = siammask_test.siamese_track(self.state, frame, mask_enable=True, refine_enable=True,
                                                 device=self.device)

        pos, sz, score = self.state['target_pos'], self.state['target_sz'], self.state['score']
        landing_target_y = pos[1] - (sz[1] * 0.4)

        display = frame.copy()

        # --- תיקון צביעה ---
        # 1. ניסיון לצבוע מסכה (רק אם היא קיימת)
        if 'mask' in self.state and self.state['mask'] is not None:
            mask_raw = self.state['mask']
            if np.max(mask_raw) > self.hp['seg_thr']:
                mask_bool = cv2.resize((mask_raw > self.hp['seg_thr']).astype(np.uint8), (1280, 720)).astype(bool)
                display[mask_bool] = (display[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # 2. ציור התיבה הכחולה והנקודה הירוקה (תמיד יופיעו!)
        cv2.rectangle(display, (int(pos[0] - sz[0] / 2), int(pos[1] - sz[1] / 2)),
                      (int(pos[0] + sz[0] / 2), int(pos[1] + sz[1] / 2)), (255, 0, 0), 2)

        # הנקודה הירוקה מסמלת את המקום אליו ה-RL מכוון (משטח הנחיתה)
        cv2.circle(display, (int(pos[0]), int(landing_target_y)), 8, (0, 255, 0), -1)
        cv2.putText(display, "TARGET", (int(pos[0] + 15), int(landing_target_y)), 1, 1.5, (0, 255, 0), 2)

        # הדפסת CONF בולט
        color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
        cv2.putText(display, f"CONF: {score:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Drone RL View", display)
        cv2.waitKey(1)

        done = bool(score < 0.17)
        reward = score if not done else -10.0

        obs = np.array([(pos[0] - 640) / 640, (landing_target_y - 360) / 360, (sz[0] * sz[1]) / (1280 * 720), score],
                       dtype=np.float32)
        return obs, reward, done, False, {}