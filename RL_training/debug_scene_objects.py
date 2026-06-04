"""
debug_scene_objects.py

Purpose:
    Debug Unreal / AirSim scene objects around the drone.
    Useful when a Blueprint was added to the world but AirSim API does not detect
    the expected name from the Unreal Outliner.

Usage:
    python debug_scene_objects.py

Notes:
    - Unreal actor display name is not always the same name returned by AirSim.
    - Blueprint instances often appear with suffixes like:
        BP_X6M_C_0
        BP_X6M_C_1
        X6M_2
        StaticMeshActor_123
    - This script searches broadly and prints candidates containing X6M.
"""

import re
import math

try:
    import cosysairsim as airsim
except ImportError:
    import airsim


# ==========================
# User config
# ==========================

VEHICLE_NAME = ""          # Leave empty if you use the default drone
TARGET_KEYWORD = "X6M"     # Your suspected Blueprint/object keyword

PRINT_ALL_OBJECTS = True
MAX_OBJECTS_TO_PRINT = 300

SEARCH_PATTERNS = [
    ".*X6M.*",
    ".*x6m.*",
    ".*BP.*X6M.*",
    ".*BP_X6M.*",
    ".*BMW.*",
    ".*Car.*",
    ".*Vehicle.*",
]


def distance_3d(a, b):
    return math.sqrt(
        (a.x_val - b.x_val) ** 2 +
        (a.y_val - b.y_val) ** 2 +
        (a.z_val - b.z_val) ** 2
    )


def safe_get_pose(client, object_name):
    """
    Try to get object pose.
    Some scene object names are listed but do not return a valid pose.
    """
    try:
        pose = client.simGetObjectPose(object_name)

        # AirSim sometimes returns NaN pose for invalid/untracked objects.
        p = pose.position
        values = [p.x_val, p.y_val, p.z_val]

        if any(math.isnan(v) for v in values):
            return None

        return pose

    except Exception:
        return None


def main():
    print("[INFO] Connecting to AirSim / Cosys-AirSim...")

    client = airsim.MultirotorClient()
    client.confirmConnection()

    print("[OK] Connected.")

    try:
        drone_pose = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
    except TypeError:
        # Some AirSim versions do not accept vehicle_name as keyword.
        drone_pose = client.simGetVehiclePose()

    drone_pos = drone_pose.position

    print()
    print("[DRONE POSE]")
    print(f"x={drone_pos.x_val:.3f}, y={drone_pos.y_val:.3f}, z={drone_pos.z_val:.3f}")
    print()

    # ==========================
    # 1. List all scene objects
    # ==========================

    print("[INFO] Listing all scene objects using regex: .*")
    all_objects = client.simListSceneObjects(".*")

    print(f"[INFO] Total objects found: {len(all_objects)}")
    print()

    if PRINT_ALL_OBJECTS:
        print(f"[ALL OBJECTS - first {MAX_OBJECTS_TO_PRINT}]")
        for i, name in enumerate(all_objects[:MAX_OBJECTS_TO_PRINT]):
            print(f"{i:04d} | {name}")

        if len(all_objects) > MAX_OBJECTS_TO_PRINT:
            print(f"... skipped {len(all_objects) - MAX_OBJECTS_TO_PRINT} more objects")

        print()

    # ==========================
    # 2. Search by patterns
    # ==========================

    print("[INFO] Searching by possible patterns...")
    found_by_pattern = {}

    for pattern in SEARCH_PATTERNS:
        try:
            matches = client.simListSceneObjects(pattern)
        except Exception as e:
            print(f"[WARN] Pattern failed: {pattern} | error={e}")
            matches = []

        found_by_pattern[pattern] = matches

        print()
        print(f"[PATTERN] {pattern}")
        print(f"[MATCHES] {len(matches)}")

        for name in matches:
            print(f"  - {name}")

    # ==========================
    # 3. Manual contains search
    # ==========================

    print()
    print(f"[INFO] Manual contains search for keyword: {TARGET_KEYWORD}")

    keyword_lower = TARGET_KEYWORD.lower()
    manual_matches = [
        name for name in all_objects
        if keyword_lower in name.lower()
    ]

    print(f"[INFO] Manual matches found: {len(manual_matches)}")

    for name in manual_matches:
        print(f"  - {name}")

    # ==========================
    # 4. Print pose and distance from drone
    # ==========================

    unique_candidates = sorted(set(
        manual_matches +
        [
            name
            for matches in found_by_pattern.values()
            for name in matches
        ]
    ))

    print()
    print("[CANDIDATES WITH POSE + DISTANCE FROM DRONE]")
    print(f"[INFO] Unique candidates: {len(unique_candidates)}")
    print()

    if not unique_candidates:
        print("[WARN] No candidates found.")
        print()
        print("Possible reasons:")
        print("1. The Blueprint actor name in Unreal is not actually exposed to AirSim.")
        print("2. The object is not a static mesh actor AirSim can list.")
        print("3. The name in Unreal Outliner is only a display label, not the internal actor name.")
        print("4. The BP contains child mesh components, and AirSim sees the component/mesh name instead.")
        print("5. The object was added after Play started and was not registered.")
        return

    for name in unique_candidates:
        pose = safe_get_pose(client, name)

        if pose is None:
            print(f"{name}")
            print("  pose: NOT AVAILABLE")
            print()
            continue

        pos = pose.position
        dist = distance_3d(drone_pos, pos)

        print(f"{name}")
        print(f"  position: x={pos.x_val:.3f}, y={pos.y_val:.3f}, z={pos.z_val:.3f}")
        print(f"  distance from drone: {dist:.3f} m")
        print()

    # ==========================
    # 5. Nearest objects to drone
    # ==========================

    print()
    print("[NEAREST OBJECTS TO DRONE]")
    print("This helps if the object exists but has a weird internal name.")
    print()

    objects_with_pose = []

    for name in all_objects:
        pose = safe_get_pose(client, name)

        if pose is None:
            continue

        pos = pose.position
        dist = distance_3d(drone_pos, pos)

        objects_with_pose.append((dist, name, pos))

    objects_with_pose.sort(key=lambda x: x[0])

    for dist, name, pos in objects_with_pose[:50]:
        print(
            f"{dist:8.3f} m | "
            f"x={pos.x_val:9.3f}, y={pos.y_val:9.3f}, z={pos.z_val:9.3f} | "
            f"{name}"
        )


if __name__ == "__main__":
    main()