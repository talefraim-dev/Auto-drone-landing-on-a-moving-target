"""Atomic checkpoint-pair storage for cooperative Agent-1/Agent-2 training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, (tuple, list, set, frozenset)):
        return [safe_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(safe_json(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sb3_zip(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt checkpoint member {bad!r} in {path}")
        names = set(archive.namelist())
        required = {"data", "policy.pth"}
        missing = required - names
        if missing:
            raise RuntimeError(f"Checkpoint {path} is missing {sorted(missing)}")


def checkpoint_space_dim(path: Path, key: str) -> int | None:
    """Read a one-dimensional SB3 space shape from the checkpoint metadata."""
    try:
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("data").decode("utf-8"))
        space = payload.get(key, {})
        shape = (space.get("shape") or space.get("_shape")) if isinstance(space, dict) else None
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            return int(shape[0])
    except Exception:
        return None
    return None


def _copy_optional(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def create_atomic_pair(
    run_root: Path,
    generation: int,
    agent1_checkpoint: Path,
    agent2_checkpoint: Path,
    pair_state: dict[str, Any],
    snapshot_files: Iterable[Path] = (),
) -> Path:
    """Create a validated pair directory and publish it with one atomic rename."""
    verify_sb3_zip(agent1_checkpoint)
    verify_sb3_zip(agent2_checkpoint)

    pairs_root = run_root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    final_dir = pairs_root / f"pair_{generation:06d}"
    tmp_dir = pairs_root / f".pair_{generation:06d}.tmp"
    if final_dir.exists():
        raise FileExistsError(f"Pair generation already exists: {final_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    agent1_copy = tmp_dir / "agent1.zip"
    agent2_copy = tmp_dir / "agent2.zip"
    shutil.copy2(agent1_checkpoint, agent1_copy)
    shutil.copy2(agent2_checkpoint, agent2_copy)

    # Agent1P2Env expects the exact Agent-1 configuration snapshot next to the
    # Agent-1 checkpoint. Preserve it in every pair.
    source_agent1_snapshot = agent1_checkpoint.parent / "training_config_snapshot.json"
    if not _copy_optional(
        source_agent1_snapshot,
        tmp_dir / "training_config_snapshot.json",
    ):
        raise FileNotFoundError(
            "Agent-1 training_config_snapshot.json is required next to "
            f"{agent1_checkpoint}"
        )

    source_snapshot_dir = tmp_dir / "source_snapshot"
    hashes: dict[str, str] = {
        "agent1.zip": sha256(agent1_copy),
        "agent2.zip": sha256(agent2_copy),
        "training_config_snapshot.json": sha256(tmp_dir / "training_config_snapshot.json"),
    }
    for source in snapshot_files:
        source = Path(source)
        if not source.is_file():
            continue
        target = source_snapshot_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(target.relative_to(tmp_dir))] = sha256(target)

    manifest = dict(pair_state)
    manifest.update(
        {
            "generation": int(generation),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent1_checkpoint": "agent1.zip",
            "agent2_checkpoint": "agent2.zip",
            "agent1_observation_dim": checkpoint_space_dim(agent1_copy, "observation_space"),
            "agent1_action_dim": checkpoint_space_dim(agent1_copy, "action_space"),
            "agent2_observation_dim": checkpoint_space_dim(agent2_copy, "observation_space"),
            "agent2_action_dim": checkpoint_space_dim(agent2_copy, "action_space"),
            "sha256": hashes,
        }
    )
    atomic_json(tmp_dir / "pair_state.json", manifest)

    verify_sb3_zip(agent1_copy)
    verify_sb3_zip(agent2_copy)
    os.replace(tmp_dir, final_dir)
    atomic_json(
        run_root / "latest_pair.json",
        {
            "generation": int(generation),
            "pair_dir": str(final_dir),
            "agent1_checkpoint": str(final_dir / "agent1.zip"),
            "agent2_checkpoint": str(final_dir / "agent2.zip"),
        },
    )
    return final_dir


def pair_rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Higher is better; safety dominates success, then geometric quality."""
    episodes = max(1, int(metrics.get("episodes", 0) or 0))
    wrong_successes = int(metrics.get("wrong_object_successes", 0) or 0)
    terminal_violations = int(metrics.get("terminal_latch_violations", 0) or 0)
    rpc_rate = float(metrics.get("rpc_success_rate", 0.0) or 0.0)
    touchdown_rate = float(metrics.get("touchdown_rate", 0.0) or 0.0)
    handoff_rate = float(metrics.get("safe_handoff_rate", 0.0) or 0.0)
    mean_center = float(metrics.get("mean_terminal_center_error_m", 999.0) or 999.0)
    timeout_rate = float(metrics.get("timeout_count", 0) or 0) / episodes
    mean_reward = float(metrics.get("mean_episode_reward", -1.0e12) or -1.0e12)
    return (
        -float(wrong_successes),
        -float(terminal_violations),
        rpc_rate,
        touchdown_rate,
        handoff_rate,
        -mean_center,
        -timeout_rate,
        mean_reward,
    )


def maybe_publish_best_pair(
    run_root: Path,
    pair_dir: Path,
    metrics: dict[str, Any],
    minimum_episodes: int,
) -> bool:
    if int(metrics.get("episodes", 0) or 0) < int(minimum_episodes):
        return False

    pointer = run_root / "best_pair.json"
    candidate_rank = pair_rank(metrics)
    previous: dict[str, Any] = {}
    if pointer.is_file():
        try:
            previous = json.loads(pointer.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    previous_rank = tuple(previous.get("rank", []))
    if previous_rank and candidate_rank <= previous_rank:
        return False

    payload = {
        "pair_dir": str(pair_dir),
        "agent1_checkpoint": str(pair_dir / "agent1.zip"),
        "agent2_checkpoint": str(pair_dir / "agent2.zip"),
        "metrics": safe_json(metrics),
        "rank": list(candidate_rank),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(pointer, payload)
    return True
