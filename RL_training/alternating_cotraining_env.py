"""Alternating co-training environments for the existing two-agent landing stack.

The physical command path remains unchanged:
    Agent 1 owns X/Y/Yaw.
    Agent 2 owns NED-Z.
    Agent1P2Env sends exactly one fused AirSim command per step.

This module only changes which PPO policy is trainable in a phase and makes the
successful RPC latch the dominant terminal learning signal for both policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent1p2_env import Agent1P2Env
from touchdown_time_reward import TouchdownTimeRewardTracker


RPC_SUCCESS_REWARD = 8_000.0
RPC_FAILURE_PENALTY = 4_000.0
DENSE_REWARD_SCALE = 0.20
DENSE_EPISODE_CAP = 1_500.0

from identity_stability_reward import (
    AGENT2_DESCENDING_SCALE_JUMP_PENALTY,
    AGENT2_IDENTITY_REWARD_SCALE,
    IDENTITY_EPISODE_CAP,
    IdentityStabilityReward,
)


class RangeFinderLiveDiagnostics(gym.Wrapper):
    """Print passive live range/vision diagnostics without changing control.

    This wrapper only reads the ``info`` dictionary returned by the wrapped
    environment. It does not alter observations, actions, Vision flags, state
    machines, AirSim commands, rewards, terminations, or checkpoint paths.
    """

    def __init__(self, env, label: str, print_every_steps: int = 10) -> None:
        super().__init__(env)
        self.label = str(label)
        self.print_every_steps = max(1, int(print_every_steps))
        self._step_index = 0
        self._last_signature = None

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "nan"
        return f"{number:.3f}" if np.isfinite(number) else "inf"

    def _signature(self, info: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(info.get("bottom_match_live", False)),
            int(info.get("range_valid_count", 0) or 0),
            bool(info.get("range_calibration_loaded", False)),
            bool(info.get("range_height_reliable", False)),
            bool(info.get("range_sensor_final_ready", False)),
            bool(info.get("range_height_used", False)),
            bool(info.get("range_sensor_final_active", False)),
            bool(info.get("range_terminal_contact_armed", False)),
            bool(info.get("range_terminal_handoff_active", False)),
            str(info.get("landing_height_source", "UNKNOWN")),
            str(info.get("vertical_control_state", "UNKNOWN")),
            str(info.get("range_sensor_final_reason", "")),
        )

    def _print(self, info: dict[str, Any], event: str) -> None:
        print(
            f"[RANGE COOP {self.label}] event={event} step={self._step_index} "
            f"vision={int(bool(info.get('bottom_match_live', False)))} "
            f"TL={self._fmt(info.get('range_top_left_m'))} "
            f"TR={self._fmt(info.get('range_top_right_m'))} "
            f"BL={self._fmt(info.get('range_bottom_left_m'))} "
            f"BR={self._fmt(info.get('range_bottom_right_m'))} "
            f"C={self._fmt(info.get('range_center_m'))} "
            f"med={self._fmt(info.get('range_mean_m'))} "
            f"spread={self._fmt(info.get('range_spread_m'))} "
            f"valid={int(info.get('range_valid_count', 0) or 0)}/5 "
            f"cal={int(bool(info.get('range_calibration_loaded', False)))} "
            f"reliable={int(bool(info.get('range_height_reliable', False)))} "
            f"ready={int(bool(info.get('range_sensor_final_ready', False)))} "
            f"used={int(bool(info.get('range_height_used', False)))} "
            f"active={int(bool(info.get('range_sensor_final_active', False)))} "
            f"armed={int(bool(info.get('range_terminal_contact_armed', False)))} "
            f"handoff={int(bool(info.get('range_terminal_handoff_active', False)))} "
            f"floor={self._fmt(info.get('range_reliable_floor_m'))} "
            f"lastSafe={self._fmt(info.get('range_terminal_contact_last_safe_height_m'))} "
            f"source={info.get('landing_height_source', 'UNKNOWN')} "
            f"zState={info.get('vertical_control_state', 'UNKNOWN')} "
            f"vz={self._fmt(info.get('applied_vz_mps', info.get('agent1_commanded_vz_mps')))} "
            f"reason={info.get('range_sensor_final_reason', '')}"
        )

    def _observe_info(self, info: dict[str, Any], event: str) -> None:
        if "range_valid_count" not in info:
            return
        signature = self._signature(info)
        changed = signature != self._last_signature
        periodic = self._step_index % self.print_every_steps == 0
        if self._last_signature is None or changed or periodic:
            self._print(info, event if changed else "periodic")
        self._last_signature = signature

    def reset(self, **kwargs):
        self._step_index = 0
        self._last_signature = None
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        self._observe_info(info, "reset")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_index += 1
        info = dict(info or {})
        self._observe_info(info, "state_change")
        return obs, reward, terminated, truncated, info


class _ExternalAgent1Policy:
    """Policy adapter used by Agent1P2Env during the Agent-1 training phase."""

    def __init__(self, owner: "TrainAgent1WithFrozenAgent2Env") -> None:
        self.owner = owner

    def predict(self, _observation, deterministic: bool = True):
        del deterministic
        action = self.owner._pending_agent1_action
        if action is None:
            action = np.zeros(self.owner.action_space.shape, dtype=np.float32)
        return np.asarray(action, dtype=np.float32), None


class TrainAgent1WithFrozenAgent2Env(gym.Env):
    """Train Agent 1 across the complete mission while Agent 2 is frozen.

    Unlike the Agent-2-facing wrapper, this environment does not let a frozen
    Agent 1 perform the preparation stage. The trainable Agent 1 controls the
    chase/tracking/handoff phase from reset. When DroneEnv reports
    ``handoff_success``, that Gym termination is converted into an internal
    phase transition: Agent 2 attaches to the same physical state, and the
    episode continues until a real landing terminal event occurs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        parallel_env: Agent1P2Env,
        frozen_agent2_checkpoint: str | Path,
        device: str = "auto",
        touchdown_time_tracker: TouchdownTimeRewardTracker | None = None,
    ) -> None:
        super().__init__()
        self.parallel_env = parallel_env
        self.touchdown_time_tracker = touchdown_time_tracker
        self.action_space = parallel_env.agent1_env.action_space
        self.observation_space = parallel_env.agent1_env.observation_space

        checkpoint = Path(frozen_agent2_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Frozen Agent-2 checkpoint does not exist: {checkpoint}"
            )
        self.frozen_agent2 = PPO.load(str(checkpoint), env=None, device=device)

        self._preparation_agent1_model = parallel_env.agent1_model
        self._external_agent1_policy = _ExternalAgent1Policy(self)
        self._pending_agent1_action: np.ndarray | None = None
        self._agent2_obs: np.ndarray | None = None
        self._last_agent1_dense_reward = 0.0
        self._phase = "AGENT1_CHASE"
        self._identity_reward = IdentityStabilityReward()
        self._dense_episode_total = 0.0

        original_agent1_step = self.parallel_env._agent1_step

        def recording_agent1_step(action):
            result = original_agent1_step(action)
            self._last_agent1_dense_reward = float(result[1])
            return result

        self.parallel_env._agent1_step = recording_agent1_step

    def _bounded_dense_reward(self, raw_dense: float) -> float:
        """Apply dense shaping while bounding its whole-episode influence.

        This guarantees that a long otherwise-good episode ending without the
        verified RPC latch remains negative, and a verified latch remains
        decisively positive.
        """
        before = float(self._dense_episode_total)
        after = float(np.clip(before + float(raw_dense), -DENSE_EPISODE_CAP, DENSE_EPISODE_CAP))
        applied = after - before
        self._dense_episode_total = after
        return float(applied)

    @staticmethod
    def _rpc_terminal_signal(done: bool, info: dict[str, Any]) -> float:
        if bool(info.get("latch_succeeded", False)):
            return RPC_SUCCESS_REWARD
        if done:
            return -RPC_FAILURE_PENALTY
        return 0.0

    @staticmethod
    def _raise_on_infrastructure_failure(info: dict[str, Any], terminal: bool) -> None:
        reason = str(info.get("termination_reason", "") or "")
        latch_attempted = bool(info.get("latch_attempted", False))
        latch_succeeded = bool(info.get("latch_succeeded", False))
        if terminal and reason == "landing_collision_success" and latch_attempted and not latch_succeeded:
            error = str(info.get("latch_error", "") or "unknown RPC latch failure")
            raise RuntimeError(
                "Verified physical landing succeeded but the RPC latch failed. "
                "Training stopped so infrastructure noise is not learned as policy failure. "
                f"latch_error={error}"
            )

    @staticmethod
    def _keep_policy_terminal_rollout(
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> tuple[bool, bool, dict[str, Any]]:
        """Keep one real PPO transition for every physical terminal step."""
        out = dict(info or {})
        out["terminal_controller_rollout_steps"] = 0
        out["terminal_result_credited_to_handoff_action"] = False
        out["policy_actions_suppressed_after_terminal_handoff"] = False
        return bool(terminated), bool(truncated), out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pending_agent1_action = None
        self._agent2_obs = None
        self._last_agent1_dense_reward = 0.0
        self._phase = "AGENT1_CHASE"
        self._identity_reward.reset()
        self._dense_episode_total = 0.0
        if self.touchdown_time_tracker is not None:
            self.touchdown_time_tracker.cancel_episode()

        self.parallel_env._parallel_active = False
        self.parallel_env._agent1_obs = None
        self.parallel_env._agent1_last_info = {}
        self.parallel_env.agent1_model = self._external_agent1_policy
        self.parallel_env.agent1_env.set_parallel_dual_agent_mode(False)

        obs, info = self.parallel_env.agent1_env.reset(
            seed=seed, options=options
        )
        self.parallel_env._agent1_obs = np.asarray(obs, dtype=np.float32)
        out_info = dict(info or {})
        out_info.update(
            {
                "training_owner": "AGENT_1",
                "frozen_partner": "AGENT_2",
                "co_training_phase": self._phase,
                "rpc_dominant_reward": True,
            }
        )
        return np.asarray(obs, dtype=np.float32), out_info

    def _step_chase_phase(self, action):
        obs, agent1_reward, terminated, truncated, info = (
            self.parallel_env._agent1_step(action)
        )
        obs = np.asarray(obs, dtype=np.float32)
        info = dict(info or {})
        self.parallel_env._agent1_obs = obs
        self.parallel_env._agent1_last_info = info
        reason = str(info.get("termination_reason", "") or "")

        # Handoff is not the end of the co-training mission. Attach Agent 2 to
        # the exact same AirSim/tracker state and continue the same PPO episode.
        if bool(terminated or truncated) and reason == "handoff_success":
            agent2_obs, attach_info = self.parallel_env.agent2_env.attach_from_agent1(
                self.parallel_env.agent1_env, info
            )
            self._agent2_obs = np.asarray(agent2_obs, dtype=np.float32)
            self.parallel_env.agent1_env.set_parallel_dual_agent_mode(True)
            self.parallel_env._parallel_active = True
            self._phase = "PARALLEL_LANDING"
            if self.touchdown_time_tracker is not None:
                self.touchdown_time_tracker.begin_episode()

            merged = dict(info)
            merged.update(dict(attach_info or {}))
            merged.update(
                {
                    "training_owner": "AGENT_1",
                    "frozen_partner": "AGENT_2",
                    "co_training_phase": self._phase,
                    "handoff_continued_same_episode": True,
                    "rpc_dominant_reward": True,
                }
            )
            identity_reward, identity_diag = self._identity_reward.step(merged)
            dense = self._bounded_dense_reward(DENSE_REWARD_SCALE * float(agent1_reward))
            reward = dense + identity_reward
            merged.update(identity_diag)
            merged["agent1_identity_reward"] = float(identity_reward)
            return obs, float(reward), False, False, merged

        terminal = bool(terminated or truncated)
        self._raise_on_infrastructure_failure(info, terminal)
        terminal_signal = self._rpc_terminal_signal(terminal, info)
        dense = self._bounded_dense_reward(DENSE_REWARD_SCALE * float(agent1_reward))
        identity_reward, identity_diag = self._identity_reward.step(info)
        reward = dense + identity_reward + terminal_signal
        info.update(identity_diag)
        info.update(
            {
                "training_owner": "AGENT_1",
                "frozen_partner": "AGENT_2",
                "co_training_phase": self._phase,
                "agent1_dense_reward_raw": float(agent1_reward),
                "agent1_dense_reward_scaled": float(dense),
                "dense_episode_total": float(self._dense_episode_total),
                "agent1_identity_reward": float(identity_reward),
                "shared_rpc_terminal_reward": float(terminal_signal),
                "rpc_success": bool(info.get("latch_succeeded", False)),
                "rpc_dominant_reward": True,
            }
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _step_landing_phase(self, action):
        if self._agent2_obs is None:
            raise RuntimeError("Parallel landing phase has no Agent-2 observation.")

        self._pending_agent1_action = np.asarray(action, dtype=np.float32)
        agent2_action, _ = self.frozen_agent2.predict(
            self._agent2_obs, deterministic=True
        )
        next_agent2_obs, _agent2_reward, terminated, truncated, info = (
            self.parallel_env.step(agent2_action)
        )
        # Preserve the reward caused by the real Agent-1 action. The terminal
        # phase remains policy-controlled and does not create hidden extra steps.
        policy_agent1_dense_reward = float(self._last_agent1_dense_reward)
        self._agent2_obs = np.asarray(next_agent2_obs, dtype=np.float32)
        info = dict(info or {})
        terminated, truncated, info = self._keep_policy_terminal_rollout(
            info, bool(terminated), bool(truncated)
        )

        terminal = bool(terminated or truncated)
        self._raise_on_infrastructure_failure(info, terminal)
        terminal_signal = self._rpc_terminal_signal(terminal, info)
        time_diag: dict[str, Any] = {"touchdown_time_reward": 0.0}
        if terminal and self.touchdown_time_tracker is not None:
            time_diag = self.touchdown_time_tracker.finish_episode(
                successful_touchdown=bool(info.get("latch_succeeded", False)),
                end_monotonic_s=info.get("touchdown_accepted_monotonic"),
            )
        time_reward = float(time_diag.get("touchdown_time_reward", 0.0) or 0.0)
        dense = self._bounded_dense_reward(DENSE_REWARD_SCALE * policy_agent1_dense_reward)
        identity_reward, identity_diag = self._identity_reward.step(info)
        reward = dense + identity_reward + terminal_signal + time_reward
        agent1_obs = np.asarray(self.parallel_env._agent1_obs, dtype=np.float32)
        info.update(identity_diag)
        info.update(time_diag)
        info.update(
            {
                "training_owner": "AGENT_1",
                "frozen_partner": "AGENT_2",
                "co_training_phase": self._phase,
                "agent1_dense_reward_raw": float(policy_agent1_dense_reward),
                "agent1_dense_reward_scaled": float(dense),
                "dense_episode_total": float(self._dense_episode_total),
                "agent1_identity_reward": float(identity_reward),
                "shared_rpc_terminal_reward": float(terminal_signal),
                "rpc_success": bool(info.get("latch_succeeded", False)),
                "rpc_dominant_reward": True,
            }
        )
        return agent1_obs, float(reward), bool(terminated), bool(truncated), info

    def step(self, action):
        if self._phase == "AGENT1_CHASE":
            return self._step_chase_phase(action)
        if self._phase == "PARALLEL_LANDING":
            return self._step_landing_phase(action)
        raise RuntimeError(f"Unknown co-training phase: {self._phase}")

    def close(self):
        self.parallel_env.agent1_model = self._preparation_agent1_model
        self.parallel_env.close()


class RpcDominantAgent2RewardEnv(gym.Wrapper):
    """Agent-2 reward with RPC dominance and weak identity-aware shaping."""

    def __init__(
        self,
        env,
        touchdown_time_tracker: TouchdownTimeRewardTracker | None = None,
        collapse_terminal_rollout: bool | None = None,
    ):
        super().__init__(env)
        del collapse_terminal_rollout  # Retained only for old call compatibility.
        self.touchdown_time_tracker = touchdown_time_tracker
        self._identity_reward = IdentityStabilityReward()
        self._dense_episode_total = 0.0

    def reset(self, **kwargs):
        self._identity_reward.reset()
        self._dense_episode_total = 0.0
        result = self.env.reset(**kwargs)
        if self.touchdown_time_tracker is not None:
            self.touchdown_time_tracker.begin_episode()
        return result

    def _bounded_dense_reward(self, raw_dense: float) -> float:
        before = float(self._dense_episode_total)
        after = float(np.clip(before + float(raw_dense), -DENSE_EPISODE_CAP, DENSE_EPISODE_CAP))
        applied = after - before
        self._dense_episode_total = after
        return float(applied)

    @staticmethod
    def _raise_on_infrastructure_failure(info: dict[str, Any], terminal: bool) -> None:
        """Stop training instead of teaching PPO from an RPC/plugin failure."""
        reason = str(info.get("termination_reason", "") or "")
        latch_attempted = bool(info.get("latch_attempted", False))
        latch_succeeded = bool(info.get("latch_succeeded", False))
        if terminal and reason == "landing_collision_success" and latch_attempted and not latch_succeeded:
            error = str(info.get("latch_error", "") or "unknown RPC latch failure")
            raise RuntimeError(
                "Verified physical landing succeeded but the RPC latch failed. "
                "Training was stopped to prevent infrastructure noise from being "
                f"learned as policy failure. latch_error={error}"
            )

    @staticmethod
    def _keep_policy_terminal_rollout(
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> tuple[None, bool, bool, dict[str, Any]]:
        """Keep one real PPO transition for every physical terminal step."""
        out = dict(info or {})
        out["terminal_controller_rollout_steps"] = 0
        out["terminal_result_credited_to_handoff_action"] = False
        out["policy_actions_suppressed_after_terminal_handoff"] = False
        return None, bool(terminated), bool(truncated), out

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        policy_dense_source = float(
            info.get(
                "dense_landing_reward",
                original_reward if not bool(terminated or truncated) else 0.0,
            )
        )
        final_obs, terminated, truncated, info = self._keep_policy_terminal_rollout(
            info, bool(terminated), bool(truncated)
        )
        if final_obs is not None:
            obs = final_obs
        terminal = bool(terminated or truncated)
        self._raise_on_infrastructure_failure(info, terminal)
        rpc_success = bool(info.get("latch_succeeded", False))

        # The base environment's terminal reward already includes its landing
        # bonus. Only the explicit dense component is shaped here; otherwise a
        # successful touchdown would be counted twice before the shared RPC term.
        dense_source = float(policy_dense_source)
        dense = self._bounded_dense_reward(DENSE_REWARD_SCALE * dense_source)
        identity_full, identity_diag = self._identity_reward.step(info)
        identity_reward = AGENT2_IDENTITY_REWARD_SCALE * identity_full
        descending_scale_jump_penalty = 0.0
        applied_vz = float(info.get("agent1_commanded_vz_mps", 0.0) or 0.0)
        if bool(identity_diag.get("identity_bbox_scale_jump", False)) and applied_vz > 0.10:
            descending_scale_jump_penalty = AGENT2_DESCENDING_SCALE_JUMP_PENALTY

        if rpc_success:
            shared_terminal = RPC_SUCCESS_REWARD
        elif terminal:
            shared_terminal = -RPC_FAILURE_PENALTY
        else:
            shared_terminal = 0.0

        time_diag: dict[str, Any] = {"touchdown_time_reward": 0.0}
        if terminal and self.touchdown_time_tracker is not None:
            time_diag = self.touchdown_time_tracker.finish_episode(
                successful_touchdown=rpc_success,
                end_monotonic_s=info.get("touchdown_accepted_monotonic"),
            )
        time_reward = float(time_diag.get("touchdown_time_reward", 0.0) or 0.0)

        reward = float(
            dense
            + identity_reward
            - descending_scale_jump_penalty
            + shared_terminal
            + time_reward
        )
        info.update(identity_diag)
        info.update(time_diag)
        info.update(
            {
                "training_owner": "AGENT_2",
                "frozen_partner": "AGENT_1",
                "agent2_original_reward": float(original_reward),
                "agent2_dense_source_reward": float(dense_source),
                "agent2_scaled_reward": float(dense),
                "dense_episode_total": float(self._dense_episode_total),
                "agent2_identity_reward": float(identity_reward),
                "agent2_descending_scale_jump_penalty": float(descending_scale_jump_penalty),
                "shared_rpc_terminal_reward": float(shared_terminal),
                "rpc_success": rpc_success,
                "rpc_dominant_reward": True,
            }
        )
        return obs, reward, bool(terminated), bool(truncated), info
