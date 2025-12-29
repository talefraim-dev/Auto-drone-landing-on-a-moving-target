# rewards.py  (DEPRECATED)
# You don't need rewards.py anymore because train_land.py already includes get_config() + compute_reward().
# Keep it only if you want a separate reward module. If you keep it, make sure Run_train imports the right one.

def get_config():
    return {
        "mode": "land",
        "target_name": "TemplateCube_Rounded_150",
        "camera_name": "0",

        "img_h": 84,
        "img_w": 84,
        "use_rgb": False,

        "total_timesteps": 800000,
        "model_name": "ppo_drone_vision_hsv_generalize",

        "max_v_xy": 10.0,
        "max_v_z": 6.0,

        "land_dist_thr": 2.0,
        "land_alt_thr": 1.0,
        "land_speed_thr": 1.0,

        "max_dist": 250.0,
        "max_alt": 120.0,
    }


def compute_reward(
    *,
    distance,
    is_landed,
    is_collided,
    done,
    prev_distance=None,
    yaw_diff=0.0,
    altitude=0.0,
    speed=0.0,
    vel_towards_target=0.0,
    target_u=0.0,
    target_v=0.0,
    target_area=0.0,
    target_conf=0.0,
):
    reward = 0.0
    if prev_distance is not None:
        reward += (prev_distance - distance) * 50.0
    if vel_towards_target > 0:
        reward += vel_towards_target * 2.0
    if distance < 6.0:
        reward -= speed * 2.0
    reward -= abs(yaw_diff) * 0.2
    reward -= 0.05

    if target_conf > 0.5:
        reward -= (abs(target_u) + abs(target_v)) * 0.2
        reward += float(target_area) * 0.5

    if is_landed:
        reward += 300.0
    if is_collided:
        reward -= 300.0
    return reward
