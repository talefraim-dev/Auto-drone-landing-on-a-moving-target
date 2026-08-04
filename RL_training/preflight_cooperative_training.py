"""Static and evidence-based gate for final cooperative training.

Run before pilot:
    python preflight_cooperative_training.py --stage pilot

Run after the pilot pair passes the native 46-observation E2E test:
    python preflight_cooperative_training.py --stage long
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from agent1p2_env import find_agent1_checkpoint
from agent2_landing_env import Agent2Config
from alternating_cotraining_env import (
    AGENT2_IDENTITY_REWARD_SCALE,
    DENSE_EPISODE_CAP,
    IDENTITY_EPISODE_CAP,
    RPC_FAILURE_PENALTY,
    RPC_SUCCESS_REWARD,
)
from config import flow_config as flow
from cooperative_training_config import (
    ACTIVE_POINTER,
    EXPECTED_AGENT1_ACTION_DIM,
    EXPECTED_AGENT1_OBS_DIM,
    EXPECTED_AGENT2_ACTION_DIM,
    EXPECTED_AGENT2_OBS_DIM,
    FINAL_MODELS_MANIFEST,
    NATIVE_E2E_SUMMARY,
    RANGE_FEATURE_COUNT,
    RECOVERY_CHECKPOINT_EVERY_STEPS,
    STAGE_TARGET_STEPS_PER_AGENT,
)
from paired_checkpoint_manager import checkpoint_space_dim, verify_sb3_zip
from range_finder_array import RangeFinderArray


SENSOR_NAMES = ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT", "CENTER")
MIN_FREE_DISK_GB = 10.0


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        suffix = f" | {detail}" if detail else ""
        print(f"[{status}] {label}{suffix}")
        if not condition:
            self.failures.append(f"{label}: {detail}".rstrip(": "))

    def warn(self, label: str, detail: str) -> None:
        print(f"[WARN] {label} | {detail}")
        self.warnings.append(f"{label}: {detail}")




def _same_checkpoint(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except Exception:
        return False

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_pair_from_active() -> tuple[Path | None, Path | None]:
    # The completed project uses one explicit final-model manifest. This keeps
    # runtime model selection independent of legacy training folders.
    if FINAL_MODELS_MANIFEST.is_file():
        payload = _load_json(FINAL_MODELS_MANIFEST)
        return Path(payload["agent1_checkpoint"]), Path(payload["agent2_checkpoint"])

    # Backward-compatible fallback for an unfinished cooperative training run.
    if not ACTIVE_POINTER.is_file():
        return None, None
    pointer = _load_json(ACTIVE_POINTER)
    run_root = Path(pointer["run_root"])
    latest = run_root / "latest_pair.json"
    if not latest.is_file():
        return None, None
    payload = _load_json(latest)
    return Path(payload["agent1_checkpoint"]), Path(payload["agent2_checkpoint"])


def _check_imports(report: Report) -> None:
    modules = [
        "numpy",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "cv2",
        "cosysairsim",
    ]
    for name in modules:
        try:
            module = importlib.import_module(name)
            report.check(True, f"dependency:{name}", str(getattr(module, "__version__", "installed")))
        except Exception as exc:
            report.check(False, f"dependency:{name}", f"{type(exc).__name__}: {exc}")


def _check_settings(report: Report) -> None:
    path = Path("settings.json")
    try:
        payload = _load_json(path)
        sensors = payload.get("Vehicles", {}).get("Drone1", {}).get("Sensors", {})
        found = [name for name in SENSOR_NAMES if name in sensors]
        report.check(len(found) == len(SENSOR_NAMES), "five range sensors configured", str(found))
        for name in SENSOR_NAMES:
            sensor_type = sensors.get(name, {}).get("SensorType")
            report.check(sensor_type == 5, f"range sensor type:{name}", f"SensorType={sensor_type}")
    except Exception as exc:
        report.check(False, "settings.json", f"{type(exc).__name__}: {exc}")


def _check_calibration(report: Report) -> None:
    range_path = Path("config") / "range_finder_calibration.json"
    bottom_path = Path("config") / "bottom_bbox_center_calibration.json"
    try:
        payload = _load_json(range_path)
        passed = bool(
            payload.get("calibration_status") == "PASS"
            or payload.get("validation", {}).get("overall_pass", False)
            or payload.get("calibration_test", {}).get("passed", False)
        )
        report.check(passed, "range calibration marked PASS", str(range_path))
        biases = payload.get("sensor_bias_m", {})
        report.check(all(name in biases for name in SENSOR_NAMES), "range calibration contains five biases")
    except Exception as exc:
        report.check(False, "range calibration file", f"{type(exc).__name__}: {exc}")
    try:
        payload = _load_json(bottom_path)
        target = payload.get("target", {})
        x = float(target["bbox_center_x_norm"])
        y = float(target["bbox_center_y_norm"])
        report.check(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, "bottom-center calibration", f"({x:.4f}, {y:.4f})")
    except Exception as exc:
        report.check(False, "bottom-center calibration file", f"{type(exc).__name__}: {exc}")


def _check_reward_contract(report: Report) -> None:
    a1_failure_max = DENSE_EPISODE_CAP + IDENTITY_EPISODE_CAP - RPC_FAILURE_PENALTY
    a2_failure_max = DENSE_EPISODE_CAP + IDENTITY_EPISODE_CAP * AGENT2_IDENTITY_REWARD_SCALE - RPC_FAILURE_PENALTY
    a1_success_min = RPC_SUCCESS_REWARD - DENSE_EPISODE_CAP - IDENTITY_EPISODE_CAP
    a2_success_min = RPC_SUCCESS_REWARD - DENSE_EPISODE_CAP - IDENTITY_EPISODE_CAP * AGENT2_IDENTITY_REWARD_SCALE
    report.check(a1_failure_max < 0, "Agent-1 failed episode remains negative", f"max={a1_failure_max:+.0f}")
    report.check(a2_failure_max < 0, "Agent-2 failed episode remains negative", f"max={a2_failure_max:+.0f}")
    report.check(a1_success_min > 0, "Agent-1 success remains positive", f"min={a1_success_min:+.0f}")
    report.check(a2_success_min > 0, "Agent-2 success remains positive", f"min={a2_success_min:+.0f}")


def _check_agent1(report: Report) -> None:
    try:
        path = find_agent1_checkpoint(str(flow.AGENT_1_MODEL_PATH))
        verify_sb3_zip(path)
        report.check(checkpoint_space_dim(path, "observation_space") == EXPECTED_AGENT1_OBS_DIM, "Agent-1 observation dim", str(path))
        report.check(checkpoint_space_dim(path, "action_space") == EXPECTED_AGENT1_ACTION_DIM, "Agent-1 action dim")
        report.check((path.parent / "training_config_snapshot.json").is_file(), "Agent-1 exact config snapshot", str(path.parent))
    except Exception as exc:
        report.check(False, "Agent-1 checkpoint", f"{type(exc).__name__}: {exc}")


def _check_control_authority(report: Report) -> None:
    cfg = Agent2Config()
    report.check(
        not bool(cfg.force_descent_while_bottom_match),
        "forced-contact diagnostic mode disabled",
        f"force_descent_while_bottom_match={cfg.force_descent_while_bottom_match}",
    )
    live_weight = float(flow.PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_LIVE)
    pred_weight = float(flow.PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_PRED)
    report.check(
        0.0 < live_weight <= 1.0,
        "Agent-1 LIVE XY action remains physical",
        f"weight={live_weight:.2f}",
    )
    report.check(
        0.0 < pred_weight <= 1.0,
        "Agent-1 PRED XY action remains physical",
        f"weight={pred_weight:.2f}",
    )


def _check_range_features(report: Report) -> None:
    count = len(RangeFinderArray.FEATURE_NAMES)
    report.check(count == RANGE_FEATURE_COUNT, "Agent-2 range feature count", f"count={count}")
    expected = {
        "range_top_left_norm",
        "range_top_right_norm",
        "range_bottom_left_norm",
        "range_bottom_right_norm",
        "range_center_norm",
        "range_mean_norm",
        "range_spread_norm",
        "range_valid_ratio",
        "range_closing_rate_norm",
    }
    report.check(set(RangeFinderArray.FEATURE_NAMES) == expected, "Agent-2 range feature names")


def _check_pair_and_native_e2e(report: Report, stage: str) -> None:
    agent1, agent2 = _latest_pair_from_active()
    if stage != "long":
        if agent1 is None or agent2 is None:
            report.warn("native pair", "No pilot pair exists yet; Agent 2 will start a clean 46-observation policy.")
            return
    report.check(agent1 is not None and agent1.is_file(), "current paired Agent-1 checkpoint", str(agent1))
    report.check(agent2 is not None and agent2.is_file(), "current paired Agent-2 checkpoint", str(agent2))
    if agent2 is not None and agent2.is_file():
        report.check(checkpoint_space_dim(agent2, "observation_space") == EXPECTED_AGENT2_OBS_DIM, "paired Agent-2 native observation dim", str(agent2))
        report.check(checkpoint_space_dim(agent2, "action_space") == EXPECTED_AGENT2_ACTION_DIM, "paired Agent-2 action dim")

    if stage == "long":
        try:
            summary = _load_json(NATIVE_E2E_SUMMARY)
            native = bool(
                summary.get("result") == "PASS_END_TO_END"
                and summary.get("ready_for_serious_training", False)
                and summary.get("observation_compatibility") == "NATIVE"
                and int(summary.get("checkpoint_observation_dim", -1)) == EXPECTED_AGENT2_OBS_DIM
                and int(summary.get("runtime_observation_dim", -1)) == EXPECTED_AGENT2_OBS_DIM
            )
            report.check(native, "native 46-observation pilot E2E PASS", str(NATIVE_E2E_SUMMARY))
            if agent1 is not None and agent2 is not None:
                report.check(_same_checkpoint(summary.get("agent1_checkpoint", ""), agent1), "E2E used current paired Agent-1")
                report.check(_same_checkpoint(summary.get("agent2_checkpoint", ""), agent2), "E2E used current paired Agent-2")
        except Exception as exc:
            report.check(False, "native 46-observation pilot E2E evidence", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cooperative final-training preflight")
    parser.add_argument("--stage", choices=tuple(STAGE_TARGET_STEPS_PER_AGENT), default="pilot")
    args = parser.parse_args()

    report = Report()
    print("=" * 96)
    print(f"[COOP PREFLIGHT] stage={args.stage}")
    print("=" * 96)
    _check_imports(report)
    _check_settings(report)
    _check_calibration(report)
    _check_agent1(report)
    _check_range_features(report)
    _check_control_authority(report)
    _check_reward_contract(report)
    _check_pair_and_native_e2e(report, args.stage)

    free_gb = shutil.disk_usage(Path.cwd()).free / (1024 ** 3)
    report.check(free_gb >= MIN_FREE_DISK_GB, "free disk space", f"{free_gb:.1f} GB")
    report.check(RECOVERY_CHECKPOINT_EVERY_STEPS <= 2_048, "recovery checkpoint interval", f"{RECOVERY_CHECKPOINT_EVERY_STEPS} steps")

    print("-" * 96)
    if report.failures:
        print("[COOP PREFLIGHT] READY_FOR_TRAINING=0")
        for failure in report.failures:
            print(f"  - {failure}")
        return 2
    print("[COOP PREFLIGHT] READY_FOR_TRAINING=1")
    if report.warnings:
        print("[COOP PREFLIGHT] Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
