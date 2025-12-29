import sys
import os
import numpy as np
import cv2
import torch
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim

# הוספת pysot ל-Path כדי שהאימפורטים יעבדו
# וודא שהתיקייה 'pysot' נמצאת באותה תיקייה כמו הסקריפט הזה
sys.path.append(os.path.join(os.path.dirname(__file__), 'pysot'))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


class SiamRPNTracker:
    def __init__(self, config_path, model_path):
        # 1. טעינת הקונפיגורציה
        cfg.merge_from_file(config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA

        # 2. בניית המודל (Backbone + Head)
        self.model = ModelBuilder()

        # 3. טעינת המשקולות ל-GPU
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cpu()))
        self.model.eval()
        if cfg.CUDA:
            self.model = self.model.cuda()

        # 4. בניית ה-Tracker
        self.tracker = build_tracker(self.model)
        print(f"🚀 SiamRPN++ Loaded successfully on {'CUDA' if cfg.CUDA else 'CPU'}")

    def init(self, frame, bbox):
        """
        אתחול המעקב (פריים ראשון).
        bbox format: [x, y, w, h]
        """
        self.tracker.init(frame, bbox)

    def track(self, frame):
        """
        ביצוע Inference על פריים חדש.
        מחזיר: (bbox, score)
        """
        outputs = self.tracker.track(frame)
        bbox = list(map(int, outputs['bbox']))  # [x, y, w, h]
        score = outputs['best_score']
        return bbox, score