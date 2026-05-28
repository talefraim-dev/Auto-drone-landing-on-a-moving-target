"""
Observation Builder Pseudocode

This file is not final production code.
It summarizes the planned calculations for the UAV RL observation vector.
"""

from math import sqrt


def clip(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class ObservationBuilder:
    def __init__(self):
        self.prev_bbox_cx = 0.5
        self.prev_bbox_cy = 0.5
        self.prev_bbox_area = 0.0
        self.prev_img_vx = 0.0
        self.prev_img_vy = 0.0
        self.prev_distance_proxy = 1.0
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.lost_target_time = 0.0

        self.eps = 1e-6
        self.vmax_img = 2.0
        self.amax_img = 5.0
        self.max_lost_time = 3.0

    def build(self, detection, drone_state, image_width, image_height, dt):
        """
        detection example:
            None, if target is lost
            or dict with cx, cy, w, h, conf

        drone_state example:
            dict with normalized or raw values:
            altitude_norm, drone_vx_norm, drone_vy_norm, drone_vz_norm,
            drone_roll_norm, drone_pitch_norm, drone_yaw_rate_norm,
            drone_speed_xy_norm
        """

        if detection is None:
            return self._build_lost_target_obs(drone_state, dt)

        has_target = 1.0
        self.lost_target_time = 0.0

        cx = detection["cx"]
        cy = detection["cy"]
        w = detection["w"]
        h = detection["h"]
        conf = detection.get("conf", 1.0)

        bbox_cx = cx / image_width
        bbox_cy = cy / image_height
        bbox_w = w / image_width
        bbox_h = h / image_height
        bbox_area = (w * h) / (image_width * image_height)
        bbox_conf = clip(conf, 0.0, 1.0)

        err_x = 2.0 * (bbox_cx - 0.5)
        err_y = 2.0 * (bbox_cy - 0.5)

        img_vx = (bbox_cx - self.prev_bbox_cx) / max(dt, self.eps)
        img_vy = (bbox_cy - self.prev_bbox_cy) / max(dt, self.eps)
        img_vx = clip(img_vx, -self.vmax_img, self.vmax_img)
        img_vy = clip(img_vy, -self.vmax_img, self.vmax_img)

        img_ax = (img_vx - self.prev_img_vx) / max(dt, self.eps)
        img_ay = (img_vy - self.prev_img_vy) / max(dt, self.eps)
        img_ax = clip(img_ax, -self.amax_img, self.amax_img)
        img_ay = clip(img_ay, -self.amax_img, self.amax_img)

        area_delta_norm = (bbox_area - self.prev_bbox_area) / (self.prev_bbox_area + self.eps)
        area_delta_norm = clip(area_delta_norm, -1.0, 1.0)

        distance_proxy_norm = 1.0 - sqrt(clip(bbox_area, 0.0, 1.0))
        distance_proxy_delta = distance_proxy_norm - self.prev_distance_proxy
        distance_proxy_delta = clip(distance_proxy_delta, -1.0, 1.0)

        center_error = sqrt(err_x ** 2 + err_y ** 2)
        centered_score = 1.0 - clip(center_error / sqrt(2.0), 0.0, 1.0)

        motion_mag = sqrt(img_vx ** 2 + img_vy ** 2)
        acc_mag = sqrt(img_ax ** 2 + img_ay ** 2)
        area_motion = abs(area_delta_norm)

        target_stability_score = 1.0 - clip(
            0.5 * motion_mag + 0.3 * acc_mag + 0.2 * area_motion,
            0.0,
            1.0,
        )

        altitude_norm = drone_state["altitude_norm"]

        landing_allowed = 1.0 if (
            has_target == 1.0 and
            centered_score > 0.85 and
            target_stability_score > 0.75 and
            altitude_norm < 0.7
        ) else 0.0

        lost_target_time_norm = 0.0

        obs = [
            has_target,
            bbox_cx,
            bbox_cy,
            bbox_w,
            bbox_h,
            bbox_area,
            bbox_conf,
            err_x,
            err_y,
            img_vx,
            img_vy,
            img_ax,
            img_ay,
            area_delta_norm,
            distance_proxy_norm,
            distance_proxy_delta,
            drone_state["altitude_norm"],
            drone_state["drone_vx_norm"],
            drone_state["drone_vy_norm"],
            drone_state["drone_vz_norm"],
            drone_state["drone_roll_norm"],
            drone_state["drone_pitch_norm"],
            drone_state["drone_yaw_rate_norm"],
            drone_state["drone_speed_xy_norm"],
            centered_score,
            target_stability_score,
            landing_allowed,
            lost_target_time_norm,
        ]

        self.prev_bbox_cx = bbox_cx
        self.prev_bbox_cy = bbox_cy
        self.prev_bbox_area = bbox_area
        self.prev_img_vx = img_vx
        self.prev_img_vy = img_vy
        self.prev_distance_proxy = distance_proxy_norm
        self.prev_err_x = err_x
        self.prev_err_y = err_y

        return obs

    def _build_lost_target_obs(self, drone_state, dt):
        self.lost_target_time += dt
        lost_target_time_norm = clip(self.lost_target_time / self.max_lost_time, 0.0, 1.0)

        obs = [
            0.0,  # has_target
            0.5,  # bbox_cx placeholder
            0.5,  # bbox_cy placeholder
            0.0,  # bbox_w
            0.0,  # bbox_h
            0.0,  # bbox_area
            0.0,  # bbox_conf
            self.prev_err_x,
            self.prev_err_y,
            0.0,  # img_vx
            0.0,  # img_vy
            0.0,  # img_ax
            0.0,  # img_ay
            0.0,  # area_delta_norm
            1.0,  # distance_proxy_norm: far / unknown
            0.0,  # distance_proxy_delta
            drone_state["altitude_norm"],
            drone_state["drone_vx_norm"],
            drone_state["drone_vy_norm"],
            drone_state["drone_vz_norm"],
            drone_state["drone_roll_norm"],
            drone_state["drone_pitch_norm"],
            drone_state["drone_yaw_rate_norm"],
            drone_state["drone_speed_xy_norm"],
            0.0,  # centered_score
            0.0,  # target_stability_score
            0.0,  # landing_allowed
            lost_target_time_norm,
        ]

        return obs
