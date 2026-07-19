"""Static architecture checks; no Unreal/AirSim connection is required."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    expected = json.loads((ROOT / "BASELINE_CORE_SHA256.json").read_text(encoding="utf-8"))
    for rel, digest in expected.items():
        actual = sha(ROOT / rel)
        assert actual == digest, f"Baseline file changed: {rel}"
        print(f"PASS expected package hash: {rel}")

    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".idea", "venv", ".venv"} for part in path.parts):
            continue
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    print("PASS all Python files parse")

    flow = (ROOT / "config" / "flow_config.py").read_text(encoding="utf-8")
    assert 'TRAINING_MODE = "AGENT_1P2"' in flow
    print("PASS default flow is AGENT_1P2")

    dispatcher = (ROOT / "Run_train.py").read_text(encoding="utf-8")
    assert "Run_train_agent1_original.main()" in dispatcher
    assert "Agent1P2Env" in dispatcher
    assert "Agent2LandingEnv" in dispatcher
    print("PASS dispatcher preserves Agent 1 and isolates Agent 2")

    wrapper = (ROOT / "agent1p2_env.py").read_text(encoding="utf-8")
    assert 'reason == "handoff_success"' in wrapper
    assert "agent1_env.step(action)" in wrapper
    assert "attach_from_agent1" in wrapper
    assert "activate_agent1_control" not in wrapper
    assert "_govern_agent1_prepare_action" not in wrapper
    print("PASS Agent 1 action is not governed or config-switched")

    agent2 = (ROOT / "agent2_landing_env.py").read_text(encoding="utf-8")
    assert "lidar_horizontal+api_z_vertical" in agent2
    assert "api_object_pose_z_ned" in agent2
    assert "landing_collision_success" in agent2
    assert 'self._target_id = "user_target"' in agent2
    assert '"identity_judge": "resnet_only"' in agent2
    assert "yolo_proposal_conf: float = 0.05" in agent2
    assert "NO_SAME_CLASS" not in agent2
    assert "same_class" not in agent2
    assert "candidate_similarity = max(anchor_scores)" in agent2
    assert "handoff_bbox_transferred" in agent2
    assert "bottom_anchor_created" in agent2
    assert agent2.count("obs, info = self._observe()") == 1  # post-command observation only
    assert "simSetObjectPose" not in agent2
    print("PASS Agent 2 uses ResNet-only user_target identity, API Z and never moves target actor")

    checkpoint_root = ROOT / "models" / "PPO_Tracker" / "tracking"
    checkpoints = list(checkpoint_root.glob("**/*_steps.zip"))
    steps = []
    for path in checkpoints:
        match = re.search(r"_(\d+)_steps\.zip$", path.name)
        if match:
            steps.append((int(match.group(1)), path))
    assert steps and max(steps)[0] == 280000
    latest = max(steps)[1]
    snapshot = latest.parent / "training_config_snapshot.json"
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    assert data["_selected_training_task"] == "tracking"
    assert data["freeze_vz"] is True
    assert data["altitude_hold_enabled"] is True
    assert data["block_forward_when_too_close"] is False
    assert data["bottom_velocity_match_enter_area"] == 0.036
    print("PASS selected Agent-1 checkpoint has the expected original tracking snapshot")

    print("ALL SAFE ISOLATED FLOW STATIC TESTS PASSED")


if __name__ == "__main__":
    main()
