from dataclasses import dataclass


@dataclass
class ServoTestConfig:
    """
    Manual visual-servo test config.
    This file is used only by test_servo_final.py, not by PPO training.
    """

    warmup_steps: int = 25
    max_steps: int = 1200

    yaw_gain: float = 0.12
    max_yaw: float = 0.035

    enable_z_servo: bool = False
    z_gain: float = 0.0
    max_z: float = 0.0

    enable_altitude_hold: bool = True
    altitude_hold_target_m: float = 5.0
    altitude_hold_kp: float = 0.35
    altitude_hold_max_vz_mps: float = 0.8

    forward_enabled: bool = True
    forward_gain: float = 0.08
    max_forward: float = 0.06
    target_distance_proxy: float = 0.94
    distance_deadband: float = 0.015

    min_ground_m: float = 2.0
    soft_ground_m: float = 3.0
