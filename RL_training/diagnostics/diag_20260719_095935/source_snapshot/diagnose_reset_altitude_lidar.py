"""
diagnose_reset_altitude_lidar.py

Standalone diagnostic for AirSim / Cossy-AirSim reset, altitude, camera, and LiDAR behavior.

Purpose:
    This script does NOT use PPO.
    This script does NOT use the tracker.
    This script does NOT use DroneEnv.

It directly talks to AirSim/Cossy-AirSim and answers:
    1. Does reset put the drone on the ground or in the air?
    2. Does takeoffAsync actually lift the drone?
    3. Does moveToZAsync actually move the drone to z=-5?
    4. Does simSetVehiclePose actually teleport the drone to z=-5?
    5. What does LiDAR report at each phase?
    6. What do the front and downward cameras see at each phase?

Run from inside RL_training:
    python diagnose_reset_altitude_lidar.py

Important:
    AirSim uses NED coordinates:
        z = 0      roughly spawn height / ground reference
        z = -5     about 5 meters above the local origin
        z = +5     below the local origin

If this script shows that z never becomes negative after takeoff/moveToZ,
then the problem is reset/takeoff/control, not the RL policy.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------------------
# User-tunable settings
# --------------------------------------------------------------------------------------
VEHICLE_NAME_CANDIDATES = ["Drone1", ""]
LIDAR_NAME = "LidarSensor1"

FRONT_CAMERA_CANDIDATES = ["0", "front_center"]
DOWNWARD_CAMERA_CANDIDATES = ["bottom_center"]

TARGET_Z_NED = -5.0
MOVE_TO_Z_VELOCITY = 1.5
TAKEOFF_TIMEOUT_SEC = 12
MOVE_TIMEOUT_SEC = 12

RESULT_DIR = Path("results/reset_altitude_lidar_diagnostic")
SHOW_IMAGES = True
SAVE_IMAGES = True


# --------------------------------------------------------------------------------------
# AirSim import
# --------------------------------------------------------------------------------------
def import_airsim_module():
    try:
        import cosysairsim as airsim  # type: ignore
        print("[IMPORT] Using cosysairsim")
        return airsim
    except Exception as e1:
        try:
            import airsim  # type: ignore
            print("[IMPORT] Using airsim")
            return airsim
        except Exception as e2:
            raise RuntimeError(
                "Could not import cosysairsim or airsim. "
                f"cosysairsim error={e1!r}, airsim error={e2!r}"
            )


airsim = import_airsim_module()


# --------------------------------------------------------------------------------------
# Generic safe AirSim calls
# --------------------------------------------------------------------------------------
def try_call(label: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        return True, result
    except Exception as exc:
        print(f"[WARN] {label} failed: {type(exc).__name__}: {exc}")
        return False, None


def resolve_vehicle_name(client) -> str:
    print("[CHECK] Resolving vehicle name...")
    for name in VEHICLE_NAME_CANDIDATES:
        ok, state = try_call(f"getMultirotorState(vehicle_name={name!r})", client.getMultirotorState, vehicle_name=name)
        if ok and state is not None:
            print(f"[CHECK] Selected vehicle_name={name!r}")
            return name
    print("[CHECK] Could not resolve explicitly. Falling back to empty vehicle_name.")
    return ""


def get_pose(client, vehicle_name: str):
    ok, pose = try_call("simGetVehiclePose", client.simGetVehiclePose, vehicle_name=vehicle_name)
    if ok:
        return pose
    return None


def get_state(client, vehicle_name: str):
    ok, state = try_call("getMultirotorState", client.getMultirotorState, vehicle_name=vehicle_name)
    if ok:
        return state
    return None


def vec_to_tuple(v) -> Tuple[float, float, float]:
    return (float(v.x_val), float(v.y_val), float(v.z_val))


def quat_to_yaw_deg(q) -> float:
    # Standard quaternion yaw extraction.
    w, x, y, z = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def print_pose_state(client, vehicle_name: str, label: str):
    pose = get_pose(client, vehicle_name)
    state = get_state(client, vehicle_name)

    print()
    print("=" * 100)
    print(f"[{label}]")

    if pose is not None:
        p = pose.position
        q = pose.orientation
        print(
            "[POSE] "
            f"x={p.x_val:+.3f} y={p.y_val:+.3f} z={p.z_val:+.3f} "
            f"(height proxy ~ {-float(p.z_val):+.3f}m if origin is ground) "
            f"yaw={quat_to_yaw_deg(q):+.1f}deg"
        )
    else:
        print("[POSE] unavailable")

    if state is not None:
        kin = state.kinematics_estimated
        p = kin.position
        v = kin.linear_velocity
        av = kin.angular_velocity
        print(
            "[STATE] "
            f"pos=({p.x_val:+.3f},{p.y_val:+.3f},{p.z_val:+.3f}) "
            f"vel=({v.x_val:+.3f},{v.y_val:+.3f},{v.z_val:+.3f}) "
            f"ang_vel=({av.x_val:+.3f},{av.y_val:+.3f},{av.z_val:+.3f}) "
            f"landed_state={getattr(state, 'landed_state', 'NA')}"
        )
    else:
        print("[STATE] unavailable")


# --------------------------------------------------------------------------------------
# LiDAR diagnostics
# --------------------------------------------------------------------------------------
def lidar_points_to_np(lidar_data) -> np.ndarray:
    cloud = getattr(lidar_data, "point_cloud", []) or []
    if len(cloud) < 3:
        return np.zeros((0, 3), dtype=np.float32)
    usable = (len(cloud) // 3) * 3
    pts = np.asarray(cloud[:usable], dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def summarize_lidar(client, vehicle_name: str, label: str):
    ok, lidar = try_call(
        f"getLidarData({LIDAR_NAME!r})",
        client.getLidarData,
        lidar_name=LIDAR_NAME,
        vehicle_name=vehicle_name,
    )
    if not ok or lidar is None:
        print(f"[{label}][LIDAR] unavailable")
        return

    pts = lidar_points_to_np(lidar)
    n = len(pts)
    if n == 0:
        print(f"[{label}][LIDAR] valid=False points=0")
        return

    r = np.linalg.norm(pts, axis=1)
    xy = np.linalg.norm(pts[:, :2], axis=1)

    # These are diagnostic only. We don't assume final axis convention.
    z_pos = pts[:, 2] > 0.05
    z_neg = pts[:, 2] < -0.05
    z_near0 = np.abs(pts[:, 2]) <= 0.35

    def safe_min(mask) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.min(r[mask]))

    # Horizontal-ish points by local z near zero. This helps detect self/body hits.
    horizontal_mask = z_near0 & (r > 0.05)
    front_mask = horizontal_mask & (pts[:, 0] > 0.05)
    back_mask = horizontal_mask & (pts[:, 0] < -0.05)
    right_mask = horizontal_mask & (pts[:, 1] > 0.05)
    left_mask = horizontal_mask & (pts[:, 1] < -0.05)

    print(
        f"[{label}][LIDAR] "
        f"points={n} "
        f"min_r={float(np.min(r)):.3f} "
        f"p01_r={float(np.percentile(r, 1)):.3f} "
        f"p05_r={float(np.percentile(r, 5)):.3f} "
        f"p50_r={float(np.percentile(r, 50)):.3f} "
        f"min_xy={float(np.min(xy)):.3f} "
        f"z_min={float(np.min(pts[:,2])):.3f} "
        f"z_max={float(np.max(pts[:,2])):.3f} "
        f"z_pos_min={safe_min(z_pos):.3f} "
        f"z_neg_min={safe_min(z_neg):.3f} "
        f"horiz_min={safe_min(horizontal_mask):.3f} "
        f"F={safe_min(front_mask):.3f} "
        f"B={safe_min(back_mask):.3f} "
        f"L={safe_min(left_mask):.3f} "
        f"R={safe_min(right_mask):.3f}"
    )


# --------------------------------------------------------------------------------------
# Camera diagnostics
# --------------------------------------------------------------------------------------
def request_scene_image(client, vehicle_name: str, camera_name: str) -> Optional[np.ndarray]:
    requests = [
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ]
    ok, responses = try_call(
        f"simGetImages(camera={camera_name!r})",
        client.simGetImages,
        requests,
        vehicle_name=vehicle_name,
    )
    if not ok or not responses:
        return None

    resp = responses[0]
    if resp.width <= 0 or resp.height <= 0 or not resp.image_data_uint8:
        return None

    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    expected = int(resp.height) * int(resp.width) * 3
    if img1d.size != expected:
        return None

    img_rgb = img1d.reshape((resp.height, resp.width, 3))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def resolve_camera(client, vehicle_name: str, candidates) -> Optional[str]:
    for cam in candidates:
        img = request_scene_image(client, vehicle_name, cam)
        if img is not None:
            print(f"[CAMERA] Selected camera {cam!r}")
            return cam
    return None


def draw_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def capture_and_show(client, vehicle_name: str, front_cam: Optional[str], bottom_cam: Optional[str], label: str):
    if not SHOW_IMAGES and not SAVE_IMAGES:
        return

    imgs = []

    if front_cam is not None:
        front = request_scene_image(client, vehicle_name, front_cam)
        if front is not None:
            imgs.append(draw_label(front, f"{label} | FRONT {front_cam}"))

    if bottom_cam is not None:
        bottom = request_scene_image(client, vehicle_name, bottom_cam)
        if bottom is not None:
            h, w = bottom.shape[:2]
            cv2.drawMarker(bottom, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 50, 2)
            imgs.append(draw_label(bottom, f"{label} | BOTTOM {bottom_cam}"))

    if not imgs:
        print(f"[{label}][CAMERA] No images available")
        return

    # Resize to same height and stack horizontally.
    target_h = 360
    resized = []
    for im in imgs:
        scale = target_h / max(1, im.shape[0])
        resized.append(cv2.resize(im, (int(im.shape[1] * scale), target_h)))

    canvas = cv2.hconcat(resized) if len(resized) > 1 else resized[0]

    if SAVE_IMAGES:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        safe_label = label.lower().replace(" ", "_").replace("/", "_").replace(":", "")
        out_path = RESULT_DIR / f"{safe_label}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"[{label}][CAMERA] Saved {out_path}")

    if SHOW_IMAGES:
        cv2.imshow("Reset / Altitude / LiDAR Diagnostic", canvas)
        cv2.waitKey(800)


# --------------------------------------------------------------------------------------
# Main diagnostic sequence
# --------------------------------------------------------------------------------------
def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    vehicle_name = resolve_vehicle_name(client)

    print()
    print("=" * 100)
    print("[DIAG] Reset / altitude / LiDAR diagnostic started")
    print(f"[DIAG] vehicle_name={vehicle_name!r}")
    print(f"[DIAG] lidar_name={LIDAR_NAME!r}")
    print(f"[DIAG] target_z_ned={TARGET_Z_NED}")
    print("=" * 100)

    print("[DIAG] Resolving cameras before reset...")
    front_cam = resolve_camera(client, vehicle_name, FRONT_CAMERA_CANDIDATES)
    bottom_cam = resolve_camera(client, vehicle_name, DOWNWARD_CAMERA_CANDIDATES)

    # Reset and re-enable control.
    print()
    print("[DIAG] Calling client.reset()")
    try_call("reset", client.reset)
    time.sleep(1.0)

    try_call("enableApiControl(True)", client.enableApiControl, True, vehicle_name=vehicle_name)
    try_call("armDisarm(True)", client.armDisarm, True, vehicle_name=vehicle_name)
    time.sleep(0.5)

    print_pose_state(client, vehicle_name, "AFTER RESET + ARM")
    summarize_lidar(client, vehicle_name, "AFTER RESET + ARM")
    capture_and_show(client, vehicle_name, front_cam, bottom_cam, "after_reset_arm")

    # Try takeoff.
    print()
    print("=" * 100)
    print("[DIAG] Calling takeoffAsync()")
    ok, task = try_call("takeoffAsync", client.takeoffAsync, timeout_sec=TAKEOFF_TIMEOUT_SEC, vehicle_name=vehicle_name)
    if ok and task is not None:
        try:
            task.join()
        except Exception as exc:
            print(f"[WARN] takeoff join failed: {type(exc).__name__}: {exc}")
    time.sleep(1.0)

    print_pose_state(client, vehicle_name, "AFTER TAKEOFF")
    summarize_lidar(client, vehicle_name, "AFTER TAKEOFF")
    capture_and_show(client, vehicle_name, front_cam, bottom_cam, "after_takeoff")

    # Try moveToZ.
    print()
    print("=" * 100)
    print(f"[DIAG] Calling moveToZAsync(z={TARGET_Z_NED})")
    ok, task = try_call(
        "moveToZAsync",
        client.moveToZAsync,
        TARGET_Z_NED,
        MOVE_TO_Z_VELOCITY,
        timeout_sec=MOVE_TIMEOUT_SEC,
        vehicle_name=vehicle_name,
    )
    if ok and task is not None:
        try:
            task.join()
        except Exception as exc:
            print(f"[WARN] moveToZ join failed: {type(exc).__name__}: {exc}")
    time.sleep(1.0)

    print_pose_state(client, vehicle_name, "AFTER MOVE_TO_Z")
    summarize_lidar(client, vehicle_name, "AFTER MOVE_TO_Z")
    capture_and_show(client, vehicle_name, front_cam, bottom_cam, "after_move_to_z")

    # Try direct teleport pose.
    print()
    print("=" * 100)
    print(f"[DIAG] Calling simSetVehiclePose with z={TARGET_Z_NED}")
    pose = get_pose(client, vehicle_name)
    if pose is not None:
        new_pose = airsim.Pose(
            airsim.Vector3r(float(pose.position.x_val), float(pose.position.y_val), float(TARGET_Z_NED)),
            pose.orientation,
        )
        try_call("simSetVehiclePose", client.simSetVehiclePose, new_pose, True, vehicle_name=vehicle_name)
        time.sleep(1.0)
    else:
        print("[WARN] Could not get current pose, skipping simSetVehiclePose")

    print_pose_state(client, vehicle_name, "AFTER SIM_SET_POSE")
    summarize_lidar(client, vehicle_name, "AFTER SIM_SET_POSE")
    capture_and_show(client, vehicle_name, front_cam, bottom_cam, "after_sim_set_pose")

    print()
    print("=" * 100)
    print("[DIAG] Interpretation guide:")
    print("  - If pose/state z remains near 0 after takeoff and moveToZ, AirSim control/reset is not actually lifting the drone.")
    print("  - If pose/state z is around -5 but your env prints alt=0, then the env altitude calculation is wrong.")
    print("  - If LiDAR HMIN/horizontal ranges stay around 0.12 even at z=-5, the LiDAR sector parsing or sensor mounting is wrong.")
    print("  - If cameras show the drone still on the floor after z=-5 commands, reset/takeoff/pose control is failing.")
    print("=" * 100)

    if SHOW_IMAGES:
        print("[DIAG] Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
