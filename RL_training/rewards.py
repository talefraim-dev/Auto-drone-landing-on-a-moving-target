import numpy as np

def calculate_landing_reward(dist, current_vel, last_dist):
    reward = 0
    # 1. עידוד התקרבות
    dist_change = last_dist - dist
    reward += dist_change * 15.0

    # 2. לוגיקת 80/20
    velocity_magnitude = np.linalg.norm(current_vel)
    if dist > 3.0:
        reward += velocity_magnitude * 0.5
    else:
        if velocity_magnitude < 0.5:
            reward += 5.0
        else:
            reward -= velocity_magnitude * 2.0

    # 3. קנס זמן
    reward -= 0.1
    return reward