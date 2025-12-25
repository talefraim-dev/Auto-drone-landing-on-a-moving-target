import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cosysairsim as airsim
import math
import time


class AirSimDroneEnv(gym.Env):
    def __init__(self, config, reward_fn):
        super(AirSimDroneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.config = config
        self.reward_fn = reward_fn
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.target_name = "TemplateCube_Rounded_150"
        self.prev_dist = None

    def _display_info(self, distance, current_z):
        # מדפיס לטרמינל פעם בשנייה בערך
        if int(time.time() * 5) % 5 == 0:
            print(f"DEBUG >> Dist: {distance:.2f}m | Alt: {abs(current_z):.1f}m")

    def _get_obs(self):
        drone_state = self.client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        vel = drone_state.kinematics_estimated.linear_velocity
        orientation = drone_state.kinematics_estimated.orientation

        q = orientation
        current_yaw = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                                 1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2))

        pose = self.client.simGetObjectPose(self.target_name)
        target_pos = pose.position

        if math.isnan(target_pos.x_val):
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 100.0, 0.0, pos.z_val, 0.0

        dx, dy, dz = target_pos.x_val - pos.x_val, target_pos.y_val - pos.y_val, target_pos.z_val - pos.z_val
        current_dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        dot_product = (dx * vel.x_val + dy * vel.y_val + dz * vel.z_val)
        vel_towards_target = dot_product / (current_dist + 1e-5)

        angle_to_target = math.atan2(dy, dx)
        yaw_diff = (angle_to_target - current_yaw + math.pi) % (2 * math.pi) - math.pi

        obs = np.array([dx, dy, dz, yaw_diff], dtype=np.float32)
        return np.nan_to_num(obs), current_dist, yaw_diff, pos.z_val, vel_towards_target

    def step(self, action):
        vx, vy, vz = float(action[0] * 12.0), float(action[1] * 12.0), float(action[2] * 12.0)

        obs, current_dist, yaw_diff, current_z, vel_towards_target = self._get_obs()
        self._display_info(current_dist, current_z)

        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.1,
                                        yaw_mode=airsim.YawMode(True, math.degrees(yaw_diff)))

        time.sleep(0.02)
        collision = self.client.simGetCollisionInfo().has_collided
        is_landed = bool(current_dist < 3.5)
        done = bool(collision or is_landed or current_dist > 250 or current_z < -100 or current_z > 5)

        reward = self.reward_fn(current_dist, is_landed, collision, done, self.prev_dist, yaw_diff, current_z,
                                vel_towards_target)
        self.prev_dist = current_dist

        return obs, float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        obs, dist, _, _, _ = self._get_obs()
        self.prev_dist = dist
        return obs, {}