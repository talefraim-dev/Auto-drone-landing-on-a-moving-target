"""
validate_downward_camera.py

Quick visual check for the new downward camera.

Run from inside RL_training/:
    python validate_downward_camera.py

Expected:
    A window should show the area directly under the drone. If it shows sky or
    the front view, change the bottom_center camera Pitch in settings.json
    between -90 and +90 and test again.
"""

from __future__ import annotations

import cv2
import numpy as np
import cosysairsim as airsim

from weights_config import EnvConfig


def read_camera(client, cfg, camera_name: str):
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ], vehicle_name=cfg.vehicle_name)

    if not responses or not responses[0].image_data_uint8:
        return None

    response = responses[0]
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    frame = img.reshape(response.height, response.width, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main() -> None:
    cfg = EnvConfig()
    client = airsim.MultirotorClient()
    client.confirmConnection()

    cam = str(cfg.downward_camera_name)
    print("=" * 80)
    print("[DOWNWARD CAMERA VALIDATION]")
    print(f"vehicle_name          : {cfg.vehicle_name}")
    print(f"downward_camera_name  : {cam}")
    print("Press q or ESC in the image window to stop.")
    print("=" * 80)

    while True:
        frame = read_camera(client, cfg, cam)
        if frame is None:
            frame = np.zeros((int(cfg.image_height), int(cfg.image_width), 3), dtype=np.uint8)
            cv2.putText(frame, f"No image from camera '{cam}'", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Downward camera: {cam}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Downward Camera Debug", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyWindow("Downward Camera Debug")


if __name__ == "__main__":
    main()
