"""Isolated Agent-1 -> Agent-2 training wrapper.

The frozen Agent 1 runs in its own original ``DroneEnv`` constructed from the
checkpoint's exact configuration snapshot. No Agent-2 configuration is ever
applied to that environment. After ``handoff_success``, Agent 2 takes ownership
of the AirSim command stream through a separate environment.
"""

from __future__ import annotations

import json
import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent2_landing_env import Agent2Config, Agent2LandingEnv
from drone_env import DroneEnv
from weights_config import EnvConfig


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else -1


def find_agent1_checkpoint(explicit_path: str = "") -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured Agent-1 checkpoint not found: {path}")
        return path

    root = Path("models") / "PPO_Tracker" / "tracking"
    candidates = list(root.glob("**/*.zip"))
    candidates = [p for p in candidates if checkpoint_step(p) >= 0]
    if not candidates:
        raise FileNotFoundError(f"No Agent-1 *_steps.zip checkpoint found under {root}")
    return max(candidates, key=lambda p: (checkpoint_step(p), str(p)))


def exact_env_config_for_checkpoint(checkpoint: Path) -> tuple[EnvConfig, Path]:
    snapshot = checkpoint.parent / "training_config_snapshot.json"
    if not snapshot.exists():
        raise FileNotFoundError(
            f"The selected Agent-1 checkpoint has no sibling configuration snapshot: {snapshot}"
        )

    data = json.loads(snapshot.read_text(encoding="utf-8"))
    cfg = EnvConfig()
    dynamic_fields: list[str] = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if not hasattr(cfg, key):
            dynamic_fields.append(key)
        # EnvConfig is intentionally a normal mutable dataclass. Run_train.py
        # already adds task fields dynamically, so the exact snapshot loader
        # must restore those fields too rather than silently dropping them.
        setattr(cfg, key, value)

    if str(data.get("_selected_training_task", "tracking")) != "tracking":
        raise ValueError(f"Agent-1 checkpoint snapshot is not a tracking task: {snapshot}")

    print(
        "[AGENT_1P2] Exact Agent-1 snapshot loaded | "
        f"checkpoint={checkpoint} snapshot={snapshot} restored_dynamic={len(dynamic_fields)}"
    )
    return cfg, snapshot


class Agent1P2Env(gym.Env):
    """Expose only Agent-2 transitions to PPO; Agent-1 preparation is internal."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent1_model_path: str = "",
        deterministic: bool = True,
        max_attempts: int = 3,
        max_steps_per_attempt: int = 700,
        agent2_config: Agent2Config | None = None,
        device: str = "auto",
    ):
        super().__init__()
        self.agent1_checkpoint = find_agent1_checkpoint(agent1_model_path)
        agent1_cfg, self.agent1_snapshot = exact_env_config_for_checkpoint(self.agent1_checkpoint)

        # This is the key isolation boundary: DroneEnv is initialized once with
        # the original Agent-1 snapshot. It is never mutated into landing mode.
        self.agent1_env = DroneEnv(cfg=agent1_cfg)
        self.agent1_model = PPO.load(str(self.agent1_checkpoint), env=None, device=device)
        self.agent1_deterministic = bool(deterministic)
        self.max_attempts = max(1, int(max_attempts))
        self.max_steps_per_attempt = max(1, int(max_steps_per_attempt))

        self.agent2_env = Agent2LandingEnv(cfg=agent2_config)
        self.action_space = self.agent2_env.action_space
        self.observation_space = self.agent2_env.observation_space
        self._agent2_active = False
        self._prepare_physical_steps = 0
        self._verbose_runtime = str(os.environ.get("DRONE_DIAG_VERBOSE", "0")).strip() == "1"

    def _agent1_step(self, action):
        """Run the original Agent-1 step while keeping normal training concise.

        Run_diag.py sets DRONE_DIAG_VERBOSE=1, so diagnostics still receive the
        complete original console stream. Only normal Run_train.py suppresses
        the very large legacy per-episode dump and prints a concise summary.
        """
        if self._verbose_runtime:
            return self.agent1_env.step(action)

        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                return self.agent1_env.step(action)
        except Exception:
            text = captured.getvalue().strip()
            if text:
                print(text)
            raise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent2_active = False
        last_info: dict[str, Any] = {}

        for attempt in range(1, self.max_attempts + 1):
            print(f"[AGENT_1P2] Agent 1 IN | attempt={attempt}/{self.max_attempts}")
            obs, info = self.agent1_env.reset(seed=seed, options=options)
            agent1_return = 0.0
            agent1_match_steps = 0
            agent1_bottom_match_steps = 0
            agent1_best_bottom_similarity = 0.0

            for prepare_step in range(1, self.max_steps_per_attempt + 1):
                action, _state = self.agent1_model.predict(
                    obs,
                    deterministic=self.agent1_deterministic,
                )
                obs, _agent1_reward, done, truncated, info = self._agent1_step(action)
                self._prepare_physical_steps += 1
                last_info = dict(info or {})
                reason = str(last_info.get("termination_reason", "") or "")
                agent1_return += float(_agent1_reward)
                if str(last_info.get("tracking_mode", "")).upper() == "MATCH":
                    agent1_match_steps += 1
                if bool(last_info.get("bottom_match", False)):
                    agent1_bottom_match_steps += 1
                agent1_best_bottom_similarity = max(
                    agent1_best_bottom_similarity,
                    float(last_info.get("bottom_similarity", 0.0) or 0.0),
                )

                if self._verbose_runtime and (prepare_step == 1 or prepare_step % 25 == 0 or done):
                    print(
                        "[AGENT_1P2 PREPARE] "
                        f"attempt={attempt} step={prepare_step} reason={reason or 'running'} "
                        f"stage={last_info.get('speed_stage', 'UNKNOWN')} "
                        f"BM={int(bool(last_info.get('bottom_match', False)))} "
                        f"BC={int(bool(last_info.get('bottom_confirmed', False)))} "
                        f"Bstreak={int(last_info.get('bottom_match_streak', 0))} "
                        f"HReady={int(bool(last_info.get('handoff_ready', False)))}"
                    )

                if done or truncated:
                    match_pct = 100.0 * agent1_match_steps / max(1, prepare_step)
                    bottom_pct = 100.0 * agent1_bottom_match_steps / max(1, prepare_step)
                    print(
                        f"[A1 EP] attempt={attempt} steps={prepare_step} "
                        f"result={reason or 'truncated'} return={agent1_return:+.1f} "
                        f"match={match_pct:.1f}% bottom={bottom_pct:.1f}% "
                        f"bottomSimBest={agent1_best_bottom_similarity:.3f} "
                        f"bottomConfirmed={int(bool(last_info.get('bottom_confirmed', False)))}"
                    )
                    if reason == "handoff_success":
                        print(
                            "[AGENT_1P2] Agent 1 handoff accepted exactly from original environment | "
                            f"physical_prepare_steps={self._prepare_physical_steps}"
                        )
                        agent2_obs, agent2_info = self.agent2_env.attach_from_agent1(
                            self.agent1_env,
                            last_info,
                        )
                        self._agent2_active = True
                        agent2_info["agent1_checkpoint"] = str(self.agent1_checkpoint)
                        agent2_info["agent1_snapshot"] = str(self.agent1_snapshot)
                        agent2_info["agent1_prepare_physical_steps"] = self._prepare_physical_steps
                        return agent2_obs, agent2_info
                    break

        raise RuntimeError(
            "Frozen Agent 1 did not produce handoff_success while running in its exact original "
            f"configuration. Last reason={last_info.get('termination_reason', 'unknown')} "
            f"bottom_match={last_info.get('bottom_match', False)} "
            f"bottom_confirmed={last_info.get('bottom_confirmed', False)} "
            f"bottom_streak={last_info.get('bottom_match_streak', 0)}."
        )

    def step(self, action):
        if not self._agent2_active:
            raise RuntimeError("Agent 2 step requested before Agent-1 handoff.")
        return self.agent2_env.step(action)

    def close(self):
        try:
            self.agent1_env.close()
        except Exception:
            pass
        try:
            self.agent2_env.close()
        except Exception:
            pass
