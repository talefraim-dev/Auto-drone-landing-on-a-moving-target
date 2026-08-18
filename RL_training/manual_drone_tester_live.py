"""
Live manual multirotor controller for Cosys-AirSim.

Controls:
W = forward
S = backward
A = left
D = right
T = up
G = down
Y = yaw clockwise
U = yaw counter-clockwise
SPACE = hover
ESC = quit

Dependency:
    python -m pip install keyboard
"""

import time

import keyboard
import cosysairsim as airsim


VEHICLE_NAME = ""

LINEAR_SPEED_MPS = 1.5
VERTICAL_SPEED_MPS = 1.0
YAW_RATE_DEG_S = 35.0

# Each command is completed before the next command is sent.
# This prevents AirSim from building a backlog of queued movements.
COMMAND_DURATION_S = 0.08


def key_axis(positive_key: str, negative_key: str) -> float:
    return float(keyboard.is_pressed(positive_key)) - float(
        keyboard.is_pressed(negative_key)
    )


def main() -> None:
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    print("[MANUAL TEST] Taking off...")
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()

    print("""
Controls:
  W/S   forward/backward
  A/D   left/right
  T/G   up/down
  Y/U   yaw clockwise/counter-clockwise
  SPACE hover
  ESC   quit
""")

    try:
        while not keyboard.is_pressed("esc"):
            if keyboard.is_pressed("space"):
                client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
                time.sleep(0.05)
                continue

            forward = key_axis("w", "s")
            right = key_axis("d", "a")
            vertical = key_axis("g", "t")
            yaw = key_axis("y", "u")

            # Body-frame velocities:
            # +X = forward, +Y = right.
            vx = forward * LINEAR_SPEED_MPS
            vy = right * LINEAR_SPEED_MPS

            # AirSim uses NED:
            # negative Z = up, positive Z = down.
            vz = vertical * VERTICAL_SPEED_MPS

            yaw_mode = airsim.YawMode(
                is_rate=True,
                yaw_or_rate=yaw * YAW_RATE_DEG_S,
            )

            # join() is essential here. Without it, commands are queued faster
            # than AirSim executes them, so they appear only after ESC.
            client.moveByVelocityBodyFrameAsync(
                vx=vx,
                vy=vy,
                vz=vz,
                duration=COMMAND_DURATION_S,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=VEHICLE_NAME,
            ).join()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[MANUAL TEST] Stopping...")
        try:
            client.hoverAsync(vehicle_name=VEHICLE_NAME).join()
        except Exception:
            pass

        # Keep the drone armed and hovering on exit so it does not drop.
        try:
            client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
        except Exception:
            pass

        print("[MANUAL TEST] Closed.")


if __name__ == "__main__":
    main()
