import math


def get_config():
    return {
        "mode": "land",
        "target_name": "TemplateCube_Rounded_150",
        "total_timesteps": 800000,
        "model_name": "ppo_drone_final_run"
    }


def compute_reward(distance, is_landed, is_collided, done, prev_distance=None, yaw_diff=0, current_z=0,
                   vel_towards_target=0):
    reward = 0
    if prev_distance is not None:
        # תגמול על התקרבות
        reward += (prev_distance - distance) * 200.0

        # עונש על איבוד גובה (ציר Z שלילי זה למעלה)
    if current_z > -0.8:
        reward -= 20.0

    if vel_towards_target > 0:
        reward += vel_towards_target * 10.0

    reward -= 3.0  # עונש זמן

    if is_landed:
        reward += 150000
    elif is_collided:
        reward -= 25000
    return reward