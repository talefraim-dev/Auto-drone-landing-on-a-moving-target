import airsim
import time
import os
import numpy as np
import cv2


def main():
    # התחברות
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✅ Connected to AirSim")

    client.enableApiControl(True)
    client.armDisarm(True)

    # המראה
    print("⏫ Taking off...")
    client.takeoffAsync().join()
    time.sleep(1.0)

    print("➡ Moving to position (x=5, y=3, z=-9)...")
    client.moveToPositionAsync(5, 3, -9, 20).join()
    time.sleep(1.0)

    # קריאת מצב הרחפן
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"📍 Drone position (NED): x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

    # בקשת תמונה מהמצלמה הקדמית ("0")
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])

    if not responses or responses[0].width == 0:
        print("⚠ לא התקבלה תמונה מהסימולטור")
    else:
        img_response = responses[0]
        img1d = np.frombuffer(img_response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_response.height, img_response.width, 3)

        os.makedirs("output", exist_ok=True)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join("output", "airsim_test.png")
        cv2.imwrite(out_path, img_bgr)
        print(f"📷 Image saved to: {out_path}")

    # נחיתה
    print("⏬ Landing...")
    client.landAsync().join()

    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
