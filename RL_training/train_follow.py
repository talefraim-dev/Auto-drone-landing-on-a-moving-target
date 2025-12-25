import numpy as np


def get_config():
    return {
        "mode": "follow",
        "total_timesteps": 50000,
        "model_name": "drone_follow_model",
        "target_dist": 5.0
    }


def compute_reward(obs):
    rel_pos = obs[:3]
    dist = np.linalg.norm(rel_pos)
    target_dist = 5.0

    error = abs(dist - target_dist)

    # פרס על שמירת מרחק 5 מטרים
    reward = 5.0 - error

    # קנס על התרחקות מוגזמת
    if dist > 40:
        reward -= 20.0

    return reward