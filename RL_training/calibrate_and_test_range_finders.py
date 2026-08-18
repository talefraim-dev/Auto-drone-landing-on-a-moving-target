"""Calibrate and validate the five downward AirSim range finders.

The tester deliberately avoids absolute Z targets. AirSim uses NED coordinates,
and a map-specific local origin does not represent height above the surface.
Instead, the script:

1. requires a landed start (unless --allow-airborne-start is supplied);
2. takes off and records the first stable airborne point;
3. commands a relative upward velocity for a requested displacement;
4. measures the actual Z displacement that was achieved;
5. compares the actual displacement with the range-finder change;
6. saves calibration only when the movement and sensor checks pass.

All calculations use measured poses, never assumed target poses.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import cosysairsim as airsim

from range_finder_array import SENSOR_NAMES

MOUNT_POSITIONS = {
    "TOP_LEFT": [0.28, -0.28, 0.60],
    "TOP_RIGHT": [0.28, 0.28, 0.60],
    "BOTTOM_LEFT": [-0.28, -0.28, 0.60],
    "BOTTOM_RIGHT": [-0.28, 0.28, 0.60],
    "CENTER": [0.0, 0.0, 0.60],
}


def median_positive(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0]
    return statistics.median(usable) if usable else float("inf")


def median_finite(values: list[float]) -> float:
    usable = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(usable) if usable else float("inf")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def get_state(client: Any, vehicle_name: str):
    return client.getMultirotorState(vehicle_name=vehicle_name)


def get_z(client: Any, vehicle_name: str) -> float:
    return float(get_state(client, vehicle_name).kinematics_estimated.position.z_val)


def sample(client: Any, vehicle_name: str, count: int, period_s: float) -> dict[str, Any]:
    per_sensor = {name: [] for name in SENSOR_NAMES}
    z_values: list[float] = []

    for _ in range(count):
        z_values.append(get_z(client, vehicle_name))
        for name in SENSOR_NAMES:
            try:
                data = client.getDistanceSensorData(
                    distance_sensor_name=name,
                    vehicle_name=vehicle_name,
                )
                value = float(data.distance)
            except Exception:
                value = float("inf")
            per_sensor[name].append(value)
        time.sleep(period_s)

    sensor_medians = {name: median_positive(per_sensor[name]) for name in SENSOR_NAMES}
    valid_ratio = {
        name: sum(math.isfinite(v) and v > 0.0 for v in per_sensor[name]) / max(1, count)
        for name in SENSOR_NAMES
    }
    sensor_std = {}
    for name in SENSOR_NAMES:
        usable = [v for v in per_sensor[name] if math.isfinite(v) and v > 0.0]
        sensor_std[name] = statistics.pstdev(usable) if len(usable) >= 2 else float("inf")

    return {
        "world_z_median": median_finite(z_values),
        "sensor_median_m": sensor_medians,
        "sensor_valid_ratio": valid_ratio,
        "sensor_std_m": sensor_std,
    }


def print_point(label: str, result: dict[str, Any]) -> None:
    print(f"[{label}] measured World-Z={result['world_z_median']:.3f} m")
    for name in SENSOR_NAMES:
        distance = result["sensor_median_m"][name]
        valid = result["sensor_valid_ratio"][name]
        text = f"{distance:.3f} m" if math.isfinite(distance) else "INVALID"
        print(f"  {name:13s} median={text:>10s} valid={valid:.0%}")


def relative_upward_move(
    client: Any,
    vehicle_name: str,
    requested_rise_m: float,
    speed_mps: float,
    settle_s: float,
) -> dict[str, float]:
    """Move upward relative to the current pose and report measured displacement.

    In AirSim NED coordinates, negative vertical velocity moves upward. The
    returned displacement is calculated only from measured pre/post state.
    """
    z_before = get_z(client, vehicle_name)
    duration_s = requested_rise_m / speed_mps

    client.moveByVelocityAsync(
        0.0,
        0.0,
        -abs(speed_mps),
        duration_s,
        vehicle_name=vehicle_name,
    ).join()
    client.hoverAsync(vehicle_name=vehicle_name).join()
    time.sleep(settle_s)

    z_after = get_z(client, vehicle_name)
    actual_rise_m = z_before - z_after
    return {
        "z_before_m": z_before,
        "z_after_m": z_after,
        "requested_rise_m": requested_rise_m,
        "command_speed_mps": abs(speed_mps),
        "command_duration_s": duration_s,
        "actual_rise_m": actual_rise_m,
        "rise_error_m": abs(requested_rise_m - actual_rise_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument(
        "--rise",
        type=float,
        default=1.0,
        help="Requested relative upward displacement between calibration points",
    )
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--period", type=float, default=0.10)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--velocity", type=float, default=0.50)
    parser.add_argument("--range-tolerance", type=float, default=0.30)
    parser.add_argument("--move-tolerance", type=float, default=0.25)
    parser.add_argument("--minimum-rise", type=float, default=0.50)
    parser.add_argument("--min-valid-ratio", type=float, default=0.90)
    parser.add_argument(
        "--allow-airborne-start",
        action="store_true",
        help="Allow calibration to start while already airborne",
    )
    args = parser.parse_args()

    if args.rise <= 0.0:
        raise ValueError("--rise must be positive")
    if args.velocity <= 0.0:
        raise ValueError("--velocity must be positive")
    if args.minimum_rise <= 0.0:
        raise ValueError("--minimum-rise must be positive")

    client = airsim.MultirotorClient()
    client.confirmConnection()
    vehicles = client.listVehicles()
    if args.vehicle not in vehicles:
        raise RuntimeError(f"Vehicle {args.vehicle!r} was not found. Available: {vehicles}")

    client.enableApiControl(True, vehicle_name=args.vehicle)
    client.armDisarm(True, vehicle_name=args.vehicle)

    initial_state = get_state(client, args.vehicle)
    initially_landed = int(initial_state.landed_state) == int(airsim.LandedState.Landed)
    initial_z = float(initial_state.kinematics_estimated.position.z_val)

    if not initially_landed and not args.allow_airborne_start:
        raise RuntimeError(
            "Calibration must start with the drone landed. Land/reset the drone, "
            "or pass --allow-airborne-start explicitly."
        )

    if initially_landed:
        print(f"[CALIBRATION] Landed start confirmed at measured World-Z={initial_z:+.3f} m")
        print("[CALIBRATION] Taking off; the ground reading is not used for calibration")
        client.takeoffAsync(vehicle_name=args.vehicle).join()
        client.hoverAsync(vehicle_name=args.vehicle).join()
        time.sleep(args.settle)
    else:
        print(f"[CALIBRATION] Airborne start explicitly allowed at World-Z={initial_z:+.3f} m")
        client.hoverAsync(vehicle_name=args.vehicle).join()
        time.sleep(args.settle)

    print("[CALIBRATION] Sampling first stable airborne point")
    first = sample(client, args.vehicle, args.samples, args.period)
    print_point("POINT 1", first)

    print(
        f"[CALIBRATION] Commanding RELATIVE upward motion: "
        f"rise={args.rise:.3f} m, speed={args.velocity:.3f} m/s"
    )
    movement = relative_upward_move(
        client,
        args.vehicle,
        requested_rise_m=args.rise,
        speed_mps=args.velocity,
        settle_s=args.settle,
    )
    print(f"  Start Z      : {movement['z_before_m']:+.3f} m")
    print(f"  Final Z      : {movement['z_after_m']:+.3f} m")
    print(f"  Actual rise  : {movement['actual_rise_m']:+.3f} m")
    print(f"  Command error: {movement['rise_error_m']:.3f} m")

    movement_passed = bool(
        movement["actual_rise_m"] >= args.minimum_rise
        and movement["rise_error_m"] <= args.move_tolerance
    )
    print(f"  MOVE PASS    : {int(movement_passed)}")

    if not movement_passed:
        client.hoverAsync(vehicle_name=args.vehicle).join()
        raise RuntimeError(
            "Relative movement validation failed. No calibration file was written. "
            "The drone did not achieve the requested upward displacement within tolerance."
        )

    print("[CALIBRATION] Sampling second stable airborne point")
    second = sample(client, args.vehicle, args.samples, args.period)
    print_point("POINT 2", second)

    array_median_1 = median_positive(list(first["sensor_median_m"].values()))
    array_median_2 = median_positive(list(second["sensor_median_m"].values()))

    sensor_bias: dict[str, float] = {}
    for name in SENSOR_NAMES:
        offsets: list[float] = []
        first_value = first["sensor_median_m"][name]
        second_value = second["sensor_median_m"][name]
        if math.isfinite(first_value) and math.isfinite(array_median_1):
            offsets.append(first_value - array_median_1)
        if math.isfinite(second_value) and math.isfinite(array_median_2):
            offsets.append(second_value - array_median_2)
        sensor_bias[name] = statistics.mean(offsets) if offsets else 0.0

    measured_z_delta = first["world_z_median"] - second["world_z_median"]
    world_delta = abs(measured_z_delta)
    range_delta = (
        array_median_2 - array_median_1
        if math.isfinite(array_median_1) and math.isfinite(array_median_2)
        else float("inf")
    )
    delta_error = abs(range_delta - world_delta) if math.isfinite(range_delta) else float("inf")

    surface_z_1 = (
        first["world_z_median"] + MOUNT_POSITIONS["CENTER"][2] + array_median_1
        if math.isfinite(first["world_z_median"]) and math.isfinite(array_median_1)
        else float("inf")
    )
    surface_z_2 = (
        second["world_z_median"] + MOUNT_POSITIONS["CENTER"][2] + array_median_2
        if math.isfinite(second["world_z_median"]) and math.isfinite(array_median_2)
        else float("inf")
    )
    surface_difference = (
        abs(surface_z_2 - surface_z_1)
        if math.isfinite(surface_z_1) and math.isfinite(surface_z_2)
        else float("inf")
    )

    all_valid = all(
        first["sensor_valid_ratio"][name] >= args.min_valid_ratio
        and second["sensor_valid_ratio"][name] >= args.min_valid_ratio
        for name in SENSOR_NAMES
    )
    direction_passed = bool(measured_z_delta > 0.0 and range_delta > 0.0)
    passed = bool(
        movement_passed
        and all_valid
        and direction_passed
        and math.isfinite(delta_error)
        and delta_error <= args.range_tolerance
        and math.isfinite(surface_difference)
        and surface_difference <= args.range_tolerance
    )

    payload = {
        "version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vehicle_name": args.vehicle,
        "sensor_names": list(SENSOR_NAMES),
        "sensor_mount_positions_m": MOUNT_POSITIONS,
        "sensor_bias_m": sensor_bias,
        "tilt_compensation": "vertical_distance = calibrated_slant_range * abs(body_down_world_z)",
        "calibration_test": {
            "initially_landed": initially_landed,
            "initial_world_z_m": initial_z,
            "first": first,
            "relative_movement": movement,
            "movement_passed": movement_passed,
            "second": second,
            "array_median_1_m": array_median_1,
            "array_median_2_m": array_median_2,
            "measured_world_z_rise_m": measured_z_delta,
            "measured_range_increase_m": range_delta,
            "delta_error_m": delta_error,
            "estimated_surface_world_z_1": surface_z_1,
            "estimated_surface_world_z_2": surface_z_2,
            "surface_world_z_difference_m": surface_difference,
            "direction_passed": direction_passed,
            "range_tolerance_m": args.range_tolerance,
            "move_tolerance_m": args.move_tolerance,
            "minimum_rise_m": args.minimum_rise,
            "minimum_valid_ratio": args.min_valid_ratio,
            "passed": passed,
        },
    }

    print("\n[RANGE CALIBRATION RESULT]")
    print(f"Measured World-Z rise : {measured_z_delta:.3f} m")
    print(f"Measured range rise   : {range_delta:.3f} m")
    print(f"Delta error           : {delta_error:.3f} m")
    print(f"Surface World-Z #1    : {surface_z_1:.3f} m")
    print(f"Surface World-Z #2    : {surface_z_2:.3f} m")
    print(f"Surface consistency   : {surface_difference:.3f} m")
    print(f"Direction PASS        : {int(direction_passed)}")
    print(f"Overall PASS          : {int(passed)}")

    client.hoverAsync(vehicle_name=args.vehicle).join()

    if not passed:
        diagnostic = Path(__file__).resolve().parent / "config" / "range_finder_calibration_failed.json"
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Diagnostic saved      : {diagnostic}")
        print("[CALIBRATION] Validation failed. Existing valid calibration was not overwritten.")
        raise SystemExit(2)

    out = Path(__file__).resolve().parent / "config" / "range_finder_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved calibration     : {out}")
    for name in SENSOR_NAMES:
        print(
            f"  {name:13s} bias={sensor_bias[name]:+.4f} m "
            f"valid1={first['sensor_valid_ratio'][name]:.0%} "
            f"valid2={second['sensor_valid_ratio'][name]:.0%}"
        )


if __name__ == "__main__":
    main()
