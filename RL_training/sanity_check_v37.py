from weights_config import EnvConfig
from drone_env import DroneEnv


def main():
    cfg = EnvConfig()

    print("obs_dim from config:", cfg.obs_dim)

    env = DroneEnv(cfg)

    print("env OBS_DIM:", env.OBS_DIM)
    print("observation_space:", env.observation_space)

    obs, info = env.reset()
    print("reset obs shape:", obs.shape)
    print("reset info:", info)

    action = env.action_space.sample()
    print("sample action:", action)

    obs, reward, done, truncated, info = env.step(action)

    print("step obs shape:", obs.shape)
    print("reward:", reward)
    print("done:", done)
    print("truncated:", truncated)
    print("info:", info)


if __name__ == "__main__":
    main()
