import numpy as np

from tracking.bbox_utils import bbox_center_distance, bbox_iou, clamp_bbox_to_frame
from tracking.kalman_bbox import BBoxKalmanFilter
from tracking.target_memory import TargetMemory
from tracking.tracking_state import TrackingMode


class TargetTrackerManager:
    """
    Fusion layer for a visual tracker and Kalman prediction.

    It combines:
        - visual tracker bbox
        - visual tracker confidence
        - Kalman bbox prediction
        - gate logic
        - MATCH / PRED / LOST state machine

    This manager is deliberately independent from OSTrack / YOLO / AirSim.
    Any visual tracker can plug into it as long as it returns:
        bbox_xyxy, confidence
    """

    def __init__(
        self,
        min_tracker_confidence=0.45,
        max_center_jump_pixels=120.0,
        min_iou_with_prediction=0.05,
        max_pred_frames=12,
        process_noise=1.0,
        measurement_noise=25.0,
    ):
        self.min_tracker_confidence = float(min_tracker_confidence)
        self.max_center_jump_pixels = float(max_center_jump_pixels)
        self.min_iou_with_prediction = float(min_iou_with_prediction)
        self.max_pred_frames = int(max_pred_frames)

        self.kalman = BBoxKalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self.memory = TargetMemory()

        self.mode = TrackingMode.LOST
        self.pred_frames = 0
        self.initialized = False

    def initialize(self, initial_bbox_xyxy):
        initial_bbox_xyxy = np.array(initial_bbox_xyxy, dtype=np.float32)

        self.kalman.initialize(initial_bbox_xyxy)
        self.memory.initialize(initial_bbox_xyxy)

        self.mode = TrackingMode.MATCH
        self.pred_frames = 0
        self.initialized = True

    def update(
        self,
        tracker_bbox_xyxy,
        tracker_confidence,
        frame_width,
        frame_height,
        dt=1.0,
    ):
        """
        Main update function.

        Args:
            tracker_bbox_xyxy:
                Bbox from visual tracker, or None if tracker failed.

            tracker_confidence:
                Confidence score from visual tracker.
                If tracker does not provide confidence, pass 1.0 temporarily.

            frame_width:
                Current frame width.

            frame_height:
                Current frame height.

            dt:
                Time step. For frame-based tracking, 1.0 is fine.

        Returns:
            dict with:
                mode
                stable_bbox
                kalman_pred_bbox
                accepted_tracker
                tracker_confidence
                center_error
                iou_with_prediction
                pred_frames
        """

        if not self.initialized:
            raise RuntimeError("TargetTrackerManager is not initialized.")

        kalman_pred_bbox = self.kalman.predict(dt=dt)
        kalman_pred_bbox = clamp_bbox_to_frame(
            kalman_pred_bbox,
            frame_width,
            frame_height,
        )

        accepted_tracker = False
        center_error = None
        iou_with_prediction = None

        if tracker_bbox_xyxy is not None:
            tracker_bbox_xyxy = clamp_bbox_to_frame(
                np.array(tracker_bbox_xyxy, dtype=np.float32),
                frame_width,
                frame_height,
            )

            center_error = bbox_center_distance(tracker_bbox_xyxy, kalman_pred_bbox)
            iou_with_prediction = bbox_iou(tracker_bbox_xyxy, kalman_pred_bbox)

            confidence_ok = float(tracker_confidence) >= self.min_tracker_confidence
            center_ok = center_error <= self.max_center_jump_pixels
            iou_ok = iou_with_prediction >= self.min_iou_with_prediction

            if confidence_ok and (center_ok or iou_ok):
                accepted_tracker = True

        if accepted_tracker:
            self.mode = TrackingMode.MATCH
            self.pred_frames = 0

            stable_bbox = self.kalman.update(tracker_bbox_xyxy)
            stable_bbox = clamp_bbox_to_frame(stable_bbox, frame_width, frame_height)

            self.memory.update_match(stable_bbox)

        else:
            self.pred_frames += 1

            if self.pred_frames <= self.max_pred_frames:
                self.mode = TrackingMode.PRED
                stable_bbox = kalman_pred_bbox
                self.memory.update_pred(stable_bbox)
            else:
                self.mode = TrackingMode.LOST
                stable_bbox = self.memory.last_stable_bbox
                self.memory.update_lost()

        return {
            "mode": self.mode.value,
            "stable_bbox": stable_bbox,
            "kalman_pred_bbox": kalman_pred_bbox,
            "accepted_tracker": accepted_tracker,
            "tracker_confidence": float(tracker_confidence) if tracker_confidence is not None else None,
            "center_error": center_error,
            "iou_with_prediction": iou_with_prediction,
            "pred_frames": self.pred_frames,
        }
