import numpy as np

from tracking.bbox_utils import xyxy_to_cxcywh, cxcywh_to_xyxy


class BBoxKalmanFilter:
    """
    Kalman filter for bounding boxes.

    State:
        x = [cx, cy, w, h, vx, vy, vw, vh]

    Measurement:
        z = [cx, cy, w, h]

    This is intentionally image-space first.
    World-space XYZ / AirSim ground-truth fusion can be added above this layer later.
    """

    def __init__(
        self,
        process_noise=1.0,
        measurement_noise=25.0,
        initial_position_uncertainty=10.0,
        initial_velocity_uncertainty=100.0,
    ):
        self.initialized = False

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initial_position_uncertainty = float(initial_position_uncertainty)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * self.initial_position_uncertainty

        self.F = np.eye(8, dtype=np.float32)
        self.H = np.zeros((4, 8), dtype=np.float32)

        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(8, dtype=np.float32) * self.process_noise
        self.R = np.eye(4, dtype=np.float32) * self.measurement_noise

        self.last_prediction_xyxy = None

    def initialize(self, bbox_xyxy):
        measurement = xyxy_to_cxcywh(bbox_xyxy)

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[0, 0] = measurement[0]
        self.x[1, 0] = measurement[1]
        self.x[2, 0] = measurement[2]
        self.x[3, 0] = measurement[3]

        self.P = np.eye(8, dtype=np.float32) * self.initial_position_uncertainty

        # Higher uncertainty for initial velocities.
        self.P[4, 4] = self.initial_velocity_uncertainty
        self.P[5, 5] = self.initial_velocity_uncertainty
        self.P[6, 6] = self.initial_velocity_uncertainty
        self.P[7, 7] = self.initial_velocity_uncertainty

        self.initialized = True
        self.last_prediction_xyxy = cxcywh_to_xyxy(self.x[:4, 0])
        return self.last_prediction_xyxy.copy()

    def predict(self, dt=1.0):
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")

        dt = float(dt)

        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.last_prediction_xyxy = cxcywh_to_xyxy(self.x[:4, 0])
        return self.last_prediction_xyxy.copy()

    def update(self, bbox_xyxy):
        if not self.initialized:
            return self.initialize(bbox_xyxy)

        measurement = xyxy_to_cxcywh(bbox_xyxy).reshape(4, 1)

        y = measurement - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return cxcywh_to_xyxy(self.x[:4, 0])

    def get_state_bbox(self):
        if not self.initialized:
            return None

        return cxcywh_to_xyxy(self.x[:4, 0])

    def get_velocity(self):
        if not self.initialized:
            return None

        return self.x[4:8, 0].copy()
