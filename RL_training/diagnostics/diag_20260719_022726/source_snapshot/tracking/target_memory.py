import numpy as np


class TargetMemory:
    """
    Stores target-related tracking memory.

    First version:
        - initial bbox
        - last good bbox
        - last stable bbox
        - basic counters

    Later versions should add:
        - initial template crop
        - stable/recent templates
        - visual embeddings
        - negative bank for confusing distractors
    """

    def __init__(self):
        self.initial_bbox = None
        self.last_good_bbox = None
        self.last_stable_bbox = None
        self.match_count = 0
        self.pred_count = 0
        self.lost_count = 0

    def initialize(self, bbox_xyxy):
        self.initial_bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.last_good_bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.last_stable_bbox = np.array(bbox_xyxy, dtype=np.float32)

        self.match_count = 0
        self.pred_count = 0
        self.lost_count = 0

    def update_match(self, bbox_xyxy):
        self.last_good_bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.last_stable_bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.match_count += 1

    def update_pred(self, bbox_xyxy):
        self.last_stable_bbox = np.array(bbox_xyxy, dtype=np.float32)
        self.pred_count += 1

    def update_lost(self):
        self.lost_count += 1
