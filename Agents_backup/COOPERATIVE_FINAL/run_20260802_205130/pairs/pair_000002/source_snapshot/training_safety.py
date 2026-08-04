"""Numerical guards and episode health metrics for cooperative PPO training."""

from __future__ import annotations

import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from paired_checkpoint_manager import atomic_json


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
        return
    try:
        array = np.asarray(value)
    except Exception:
        return
    if array.dtype.kind not in "biufc":
        return
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"Non-finite value detected in {label}: {value!r}")


class FiniteTrainingGuard(gym.Wrapper):
    """Fail immediately before PPO can learn from NaN/Inf data."""

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        _assert_finite(observation, "reset_observation")
        return observation, info

    def step(self, action):
        _assert_finite(action, "action")
        observation, reward, terminated, truncated, info = self.env.step(action)
        _assert_finite(observation, "observation")
        _assert_finite(reward, "reward")
        return observation, reward, terminated, truncated, info


class PhaseHealthCallback(BaseCallback):
    """Collect safety-first metrics from the actual training episodes."""

    def __init__(
        self,
        role: str,
        phase_dir: Path,
        state_path: Path,
        window_episodes: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.role = str(role)
        self.phase_dir = Path(phase_dir)
        self.state_path = Path(state_path)
        self.recent = deque(maxlen=max(1, int(window_episodes)))
        self.episodes = 0
        self.rpc_successes = 0
        self.touchdowns = 0
        self.safe_handoffs = 0
        self.timeouts = 0
        self.wrong_object_successes = 0
        self.terminal_latch_violations = 0
        self.failure_reasons: Counter[str] = Counter()
        self.center_errors: list[float] = []
        self.episode_rewards: list[float] = []
        self._last_heartbeat = -1
        self.health_path = self.phase_dir / "phase_health.json"

    @staticmethod
    def _finite_float(value: Any, default: float | None = None) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    def _record_terminal(self, info: dict[str, Any], reward: float) -> None:
        success = bool(info.get("latch_succeeded", False))
        reason = str(info.get("termination_reason", "unknown") or "unknown")
        target_collision = bool(
            info.get("target_collision", False)
            or reason == "landing_collision_success"
        )
        handoff = bool(info.get("range_terminal_handoff_active", False))
        known_wrong = bool(info.get("collision_known_wrong_object_contact", False))
        wrong_success = bool(success and known_wrong)
        terminal_violation = bool(
            handoff
            and not bool(info.get("policy_actions_suppressed_after_terminal_handoff", False))
            and int(info.get("terminal_controller_rollout_steps", 0) or 0) > 0
        )

        if wrong_success:
            self.wrong_object_successes += 1
            raise RuntimeError("Safety violation: wrong-object contact was accepted as RPC success.")
        if terminal_violation:
            self.terminal_latch_violations += 1
            raise RuntimeError("Safety violation: policy actions were not suppressed after terminal handoff.")

        self.episodes += 1
        self.rpc_successes += int(success)
        self.touchdowns += int(target_collision)
        self.safe_handoffs += int(handoff)
        self.timeouts += int("timeout" in reason)
        if not success:
            self.failure_reasons[reason] += 1

        center = self._finite_float(
            info.get(
                "range_terminal_snapshot_center_error_m",
                info.get("collision_recent_center_error_m"),
            )
        )
        if center is not None:
            self.center_errors.append(center)
        self.episode_rewards.append(float(reward))
        self.recent.append(
            {
                "success": success,
                "touchdown": target_collision,
                "handoff": handoff,
                "reason": reason,
                "reward": float(reward),
                "center_error_m": center,
            }
        )
        self._write_health()

    def summary(self) -> dict[str, Any]:
        episodes = int(self.episodes)
        recent = list(self.recent)
        recent_count = len(recent)
        return {
            "role": self.role,
            "episodes": episodes,
            "rpc_successes": int(self.rpc_successes),
            "rpc_success_rate": float(self.rpc_successes / episodes) if episodes else 0.0,
            "touchdowns": int(self.touchdowns),
            "touchdown_rate": float(self.touchdowns / episodes) if episodes else 0.0,
            "safe_handoffs": int(self.safe_handoffs),
            "safe_handoff_rate": float(self.safe_handoffs / episodes) if episodes else 0.0,
            "timeout_count": int(self.timeouts),
            "wrong_object_successes": int(self.wrong_object_successes),
            "terminal_latch_violations": int(self.terminal_latch_violations),
            "mean_terminal_center_error_m": (
                float(np.mean(self.center_errors)) if self.center_errors else 999.0
            ),
            "mean_episode_reward": (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            ),
            "failure_reasons": dict(self.failure_reasons),
            "rolling_window_episodes": recent_count,
            "rolling_rpc_success_rate": (
                float(sum(int(x["success"]) for x in recent) / recent_count)
                if recent_count
                else 0.0
            ),
            "model_timesteps": int(self.model.num_timesteps) if self.model is not None else 0,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _write_health(self) -> None:
        atomic_json(self.health_path, self.summary())

    def _write_heartbeat(self) -> None:
        try:
            import json

            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        state.update(
            {
                "status": "training",
                "active_role": self.role,
                "active_model_timesteps": int(self.model.num_timesteps),
                "last_heartbeat_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            }
        )
        atomic_json(self.state_path, state)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for index, done in enumerate(dones):
            if not bool(done):
                continue
            info = dict(infos[index] or {}) if index < len(infos) else {}
            episode_data = info.get("episode", {}) if isinstance(info, dict) else {}
            reward = float(
                episode_data.get(
                    "r", rewards[index] if index < len(rewards) else 0.0
                )
            )
            self._record_terminal(info, reward)

        if int(self.model.num_timesteps) - self._last_heartbeat >= 256:
            self._last_heartbeat = int(self.model.num_timesteps)
            self._write_heartbeat()
        return True

    def _on_training_end(self) -> None:
        self._write_health()
        self._write_heartbeat()
