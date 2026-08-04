"""Parallel dual-agent training wrapper.

Agent 1 never leaves the control loop after the initial landing-ready event.
It continuously owns yaw and search/reacquire tracking while its front and
bottom camera trackers remain active. During trusted Bottom LIVE/PRED guidance,
the bottom visual-motion controller owns landing X/Y. Agent 2 continuously
receives its 46-value landing observation (37 base + 9 range features) and owns only vertical NED-Z.

A single AirSim velocity command is sent per environment step:
    X/Y = bottom-camera t+1 predictive tracking + target-velocity feed-forward
          whenever Bottom LIVE exists; otherwise the current Agent 1 search/chase policy
    Yaw = current Agent 1 policy
    Z   = Agent 2, gated by bottom prediction and landing safety

There is no Agent-2 -> Agent-1 recovery cycle. When the bottom target is lost,
Agent 2 immediately blocks descent while Agent 1 keeps searching/chasing in the
same physical episode until the shared dual-camera tracker reacquires it.
"""

from __future__ import annotations

import io
import json
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

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
    candidates = [p for p in root.glob("**/*.zip") if checkpoint_step(p) >= 0]
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
        setattr(cfg, key, value)

    if str(data.get("_selected_training_task", "tracking")) != "tracking":
        raise ValueError(f"Agent-1 checkpoint snapshot is not a tracking task: {snapshot}")

    print(
        "[AGENT_1P2] Exact Agent-1 snapshot loaded | "
        f"checkpoint={checkpoint} snapshot={snapshot} restored_dynamic={len(dynamic_fields)}"
    )
    return cfg, snapshot


class Agent1P2Env(gym.Env):
    """Fuse the current Agent-1 XY/Yaw policy with Agent-2 Z control."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent1_model_path: str = "",
        deterministic: bool = True,
        max_attempts: int = 3,
        max_steps_per_attempt: int = 700,
        agent2_config: Agent2Config | None = None,
        device: str = "auto",
        target_velocity_feedforward_gain: float = 1.0,
        horizontal_total_speed_max_mps: float = 6.0,
        agent1_min_xy_weight_near_landing: float = 0.20,
        bottom_live_agent1_xy_weight: float = 0.0,
        bottom_pred_agent1_xy_weight: float = 0.35,
        bottom_pd_correction_gain: float = 1.0,
        landing_bridge_max_speed_mps: float = 0.55,
        landing_catchup_bridge_max_speed_mps: float = 0.75,
        near_ground_descent_max_mps: float = 0.32,
        # Legacy v11 arguments remain accepted so old flow_config.py files do
        # not crash. Recovery is intentionally not used in v12.
        recovery_enabled: bool | None = None,
        recovery_max_cycles: int | None = None,
        recovery_max_attempts: int | None = None,
        recovery_max_steps_per_attempt: int | None = None,
        recovery_max_seconds_per_attempt: float | None = None,
        recovery_event_penalty: float | None = None,
        recovery_failure_penalty: float | None = None,
    ):
        super().__init__()
        self.agent1_checkpoint = find_agent1_checkpoint(agent1_model_path)
        agent1_cfg, self.agent1_snapshot = exact_env_config_for_checkpoint(
            self.agent1_checkpoint
        )

        # Parallel-only final-metre overrides. The physical fused command is
        # held for 0.25 s and is synchronously monitored by Agent 2. Disable the
        # asynchronous bridge so no movement occurs outside that monitored window.
        agent1_cfg.cmd_duration_s = 0.25
        agent1_cfg.command_bridge_enabled = False

        # Parallel-only final-metre overrides. The checkpoint snapshot remains
        # the source of truth for the frozen Agent-1 policy; only the post-pulse
        # bridge and the external Agent-2 Z safety cap are adjusted for the
        # moving-platform landing phase.
        agent1_cfg.command_bridge_landing_max_speed_mps = max(
            0.10, float(landing_bridge_max_speed_mps)
        )
        agent1_cfg.command_bridge_landing_catchup_max_speed_mps = max(
            agent1_cfg.command_bridge_landing_max_speed_mps,
            float(landing_catchup_bridge_max_speed_mps),
        )
        near_ground_descent_max_mps = float(
            np.clip(near_ground_descent_max_mps, 0.0, 1.0)
        )
        agent1_cfg.parallel_vertical_near_ground_max_mps = (
            near_ground_descent_max_mps
        )
        agent1_cfg.safety_landing_max_descent_action_near_ground = (
            near_ground_descent_max_mps
        )

        self.agent1_env = DroneEnv(cfg=agent1_cfg)
        self.agent1_model = PPO.load(
            str(self.agent1_checkpoint), env=None, device=device
        )
        self.agent1_deterministic = bool(deterministic)
        self.max_attempts = max(1, int(max_attempts))
        self.max_steps_per_attempt = max(1, int(max_steps_per_attempt))

        cfg = agent2_config or Agent2Config()
        cfg.cmd_duration_s = 0.25
        cfg.parallel_dual_agent_mode = True
        self.agent2_env = Agent2LandingEnv(cfg=cfg)
        self.action_space = self.agent2_env.action_space
        self.observation_space = self.agent2_env.observation_space

        self.target_velocity_feedforward_gain = max(
            0.0, float(target_velocity_feedforward_gain)
        )
        self.horizontal_total_speed_max_mps = max(
            0.1, float(horizontal_total_speed_max_mps)
        )
        self.agent1_min_xy_weight_near_landing = float(
            np.clip(agent1_min_xy_weight_near_landing, 0.0, 1.0)
        )
        # When the bottom camera has a LIVE match, its predictive controller is
        # authoritative for XY. Frozen Agent 1 still runs every step and owns
        # yaw/search, but front-camera/PRED XY cannot fight the landing servo.
        self.bottom_live_agent1_xy_weight = float(
            np.clip(bottom_live_agent1_xy_weight, 0.0, 1.0)
        )
        self.bottom_pred_agent1_xy_weight = float(
            np.clip(bottom_pred_agent1_xy_weight, 0.0, 1.0)
        )
        self.bottom_pd_correction_gain = max(
            0.0, float(bottom_pd_correction_gain)
        )

        self._parallel_active = False
        self._agent1_obs: Optional[np.ndarray] = None
        self._agent1_last_info: dict[str, Any] = {}
        self._prepare_physical_steps = 0
        self._parallel_physical_steps = 0
        self._verbose_runtime = (
            str(os.environ.get("DRONE_DIAG_VERBOSE", "0")).strip() == "1"
        )

        # Agent 2 computes Z, then invokes this callback exactly once. The
        # callback runs Agent 1 and sends the one combined AirSim command.
        self.agent2_env.set_external_command_executor(
            self._execute_parallel_command
        )
        self.agent2_env.set_external_motion_stop_callback(
            self._stop_parallel_motion_from_agent2
        )
        self.agent1_env.set_external_first_contact_probe(
            self.agent2_env.poll_first_contact
        )


    def _stop_parallel_motion_from_agent2(self, reason: str) -> None:
        """Stop Agent-1-owned motion after Agent 2 detects first contact.

        This method deliberately performs no collision query. Agent 2 remains
        the exclusive collision owner; Agent 1 only cancels its asynchronous
        command bridge and sends an explicit zero-velocity hold.
        """
        try:
            self.agent1_env._cancel_command_bridge(str(reason), send_zero=True)
        except Exception as exc:
            print(f"[A2 FIRST CONTACT] Agent-1 bridge cancel failed: {exc}")
        try:
            self.agent1_env.client.cancelLastTask(
                vehicle_name=self.agent1_env.cfg.vehicle_name
            )
        except TypeError:
            try:
                self.agent1_env.client.cancelLastTask()
            except Exception:
                pass
        except Exception:
            pass

    def _agent1_step(self, action):
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
        self._parallel_active = False
        self._agent1_obs = None
        self._agent1_last_info = {}
        last_info: dict[str, Any] = {}

        # Initial preparation still needs a landing-ready condition. This is
        # only a mode transition for Z authority; Agent 1 is not stopped.
        self.agent1_env.set_parallel_dual_agent_mode(False)

        for attempt in range(1, self.max_attempts + 1):
            print(
                f"[AGENT_1P2] Agent 1 PREPARE | "
                f"attempt={attempt}/{self.max_attempts}"
            )
            obs, _reset_info = self.agent1_env.reset(seed=seed, options=options)
            agent1_return = 0.0
            agent1_match_steps = 0
            agent1_bottom_match_steps = 0
            agent1_best_bottom_similarity = 0.0

            for prepare_step in range(1, self.max_steps_per_attempt + 1):
                action, _state = self.agent1_model.predict(
                    obs, deterministic=self.agent1_deterministic
                )
                obs, agent1_reward, done, truncated, info = self._agent1_step(
                    action
                )
                self._prepare_physical_steps += 1
                last_info = dict(info or {})
                reason = str(last_info.get("termination_reason", "") or "")
                agent1_return += float(agent1_reward)

                if str(last_info.get("tracking_mode", "")).upper() == "MATCH":
                    agent1_match_steps += 1
                if bool(last_info.get("bottom_match", False)):
                    agent1_bottom_match_steps += 1
                agent1_best_bottom_similarity = max(
                    agent1_best_bottom_similarity,
                    float(last_info.get("bottom_similarity", 0.0) or 0.0),
                )

                if done or truncated:
                    match_pct = 100.0 * agent1_match_steps / max(1, prepare_step)
                    bottom_pct = (
                        100.0 * agent1_bottom_match_steps / max(1, prepare_step)
                    )
                    print(
                        f"[A1 PREPARE EP] attempt={attempt} steps={prepare_step} "
                        f"result={reason or 'truncated'} "
                        f"return={agent1_return:+.1f} "
                        f"match={match_pct:.1f}% bottom={bottom_pct:.1f}% "
                        f"bottomSimBest={agent1_best_bottom_similarity:.3f}"
                    )

                    if reason == "handoff_success":
                        # Preserve Agent 1's post-step observation and all
                        # tracker state. From the next PPO step onward Agent 1
                        # continues normally, but its Z command is overridden.
                        self._agent1_obs = np.asarray(obs, dtype=np.float32)
                        self._agent1_last_info = last_info
                        agent2_obs, agent2_info = (
                            self.agent2_env.attach_from_agent1(
                                self.agent1_env, last_info
                            )
                        )
                        self.agent1_env.set_parallel_dual_agent_mode(True)
                        self._parallel_active = True

                        agent2_info.update(
                            {
                                "agent1_checkpoint": str(
                                    self.agent1_checkpoint
                                ),
                                "agent1_snapshot": str(self.agent1_snapshot),
                                "agent1_prepare_physical_steps": int(
                                    self._prepare_physical_steps
                                ),
                                "parallel_dual_agent": True,
                                "xy_yaw_owner": "AGENT_1",
                                "z_owner": "AGENT_2",
                            }
                        )
                        print(
                            "[AGENT_1P2 PARALLEL] ACTIVE | "
                            "Agent1=XY/Yaw + dual-camera tracking | "
                            "Agent2=Z landing authority | "
                            "single AirSim command per step"
                        )
                        return agent2_obs, agent2_info
                    break

        raise RuntimeError(
            "Frozen Agent 1 did not reach landing-ready state. "
            f"Last reason={last_info.get('termination_reason', 'unknown')} "
            f"bottom_match={last_info.get('bottom_match', False)} "
            f"bottom_confirmed={last_info.get('bottom_confirmed', False)}."
        )

    def _execute_parallel_command(self, agent2_vz_mps: float) -> dict[str, Any]:
        """Run Agent 1 once and send one fused command through DroneEnv."""
        if not self._parallel_active or self._agent1_obs is None:
            raise RuntimeError(
                "Parallel command requested before Agent-1 preparation completed."
            )

        agent1_action, _state = self.agent1_model.predict(
            self._agent1_obs, deterministic=self.agent1_deterministic
        )
        terminal_contact_hold = bool(
            getattr(self.agent2_env, "_range_terminal_handoff_active", False)
        )
        # The terminal range latch no longer freezes Agent 1. Its X/Y/Yaw
        # action continues through the same bottom-guidance blending, speed
        # limits, smoothing, and physical safety filters used before handoff.

        ff_vx, ff_vy, ff_valid = (
            self.agent2_env.get_target_velocity_feedforward_body()
        )
        ff_vx *= self.target_velocity_feedforward_gain
        ff_vy *= self.target_velocity_feedforward_gain

        bottom_guidance = self.agent2_env.get_parallel_bottom_guidance()
        bottom_blend = float(
            np.clip(bottom_guidance.get("bottom_blend_strength", 0.0), 0.0, 1.0)
        )
        bottom_live = bool(bottom_guidance.get("measurement_live", False))
        bottom_guidance_active = bool(
            bottom_guidance.get("guidance_active", bottom_live)
        )
        bottom_prediction_only = bool(
            bottom_guidance.get("prediction_only", False)
        )
        # Bottom LIVE has deterministic XY authority during landing. Agent 1's
        # policy remains active for yaw and immediately regains full XY when the
        # bottom view disappears.
        if bottom_live:
            agent1_xy_weight = float(self.bottom_live_agent1_xy_weight)
        elif bottom_guidance_active and bottom_prediction_only:
            agent1_xy_weight = float(self.bottom_pred_agent1_xy_weight)
        else:
            agent1_xy_weight = 1.0
        bottom_vx = (
            self.bottom_pd_correction_gain
            * float(bottom_guidance.get("bottom_correction_vx_mps", 0.0))
        )
        bottom_vy = (
            self.bottom_pd_correction_gain
            * float(bottom_guidance.get("bottom_correction_vy_mps", 0.0))
        )
        # Keep the existing state-dependent feed-forward and bottom-camera
        # correction active during the terminal exploration window.

        # Agent 2 already computed the bottom frame for this control cycle.
        # Reuse that exact perception state in Agent 1 instead of capturing and
        # running a second bottom YOLO+ResNet tracker on a later frame.
        self.agent1_env.set_parallel_bottom_perception_snapshot(
            self.agent2_env.get_parallel_bottom_perception_snapshot()
        )

        self.agent1_env.set_parallel_control_overrides(
            vz_mps=float(agent2_vz_mps),
            feedforward_vx_mps=float(ff_vx),
            feedforward_vy_mps=float(ff_vy),
            bottom_correction_vx_mps=float(bottom_vx),
            bottom_correction_vy_mps=float(bottom_vy),
            agent1_xy_weight=float(agent1_xy_weight),
            horizontal_speed_limit_mps=self.horizontal_total_speed_max_mps,
            bottom_guidance_active=bool(bottom_guidance_active),
            bottom_measurement_live=bool(bottom_live),
            bottom_predictive_catchup=bool(
                bottom_guidance.get("catchup_active", False)
            ),
            reacquire_climb_active=bool(
                getattr(self.agent2_env, "_reacquire_climb_active", False)
            ),
            # The range latch is still reported for state/timeout tracking,
            # but it must not activate DroneEnv's legacy XY/Yaw freeze.
            terminal_contact_hold=False,
        )

        obs, agent1_reward, done, truncated, info = self._agent1_step(
            agent1_action
        )
        self._parallel_physical_steps += 1
        self._agent1_obs = np.asarray(obs, dtype=np.float32)
        self._agent1_last_info = dict(info or {})

        critical_reason = str(
            self._agent1_last_info.get("termination_reason", "") or ""
        )
        critical = critical_reason in {
            "collision",
            "emergency_horizontal_obstacle_distance",
            "takeoff_failed_not_airborne",
            "altitude_too_high",
        }

        return {
            "parallel_command_executed": True,
            "range_terminal_handoff_active": bool(terminal_contact_hold),
            "terminal_policy_control_active": bool(terminal_contact_hold),
            "policy_actions_suppressed_after_terminal_handoff": False,
            "agent1_command_bridge_active": bool(
                self._agent1_last_info.get("command_bridge_active", False)
            ),
            "agent1_command_bridge_duration_s": float(
                self._agent1_last_info.get("command_bridge_duration_s", 0.0) or 0.0
            ),
            "agent1_command_bridge_reason": str(
                self._agent1_last_info.get("command_bridge_reason", "UNKNOWN")
            ),
            "agent1_reward_diagnostic": float(agent1_reward),
            "agent1_done_diagnostic": bool(done or truncated),
            "agent1_critical": bool(critical),
            "agent1_critical_reason": critical_reason if critical else "",
            "agent1_tracking_mode": str(
                self._agent1_last_info.get("tracking_mode", "LOST")
            ),
            "agent1_active_camera": str(
                self._agent1_last_info.get("active_camera", "unknown")
            ),
            "agent1_camera_authority": str(
                self._agent1_last_info.get(
                    "camera_authority", "UNKNOWN"
                )
            ),
            "agent1_fusion_has_target": bool(
                self._agent1_last_info.get("fusion_has_target", False)
            ),
            "agent1_bottom_match": bool(
                self._agent1_last_info.get("bottom_match", False)
            ),
            "agent1_front_tracking_mode": str(
                self._agent1_last_info.get("tracking_mode", "LOST")
            ),
            "agent1_commanded_vx_mps": float(
                self._agent1_last_info.get("commanded_vx_mps", 0.0)
            ),
            "agent1_commanded_vy_mps": float(
                self._agent1_last_info.get("commanded_vy_mps", 0.0)
            ),
            "agent1_commanded_vz_mps": float(
                self._agent1_last_info.get(
                    "commanded_vz_mps", agent2_vz_mps
                )
            ),
            "agent1_commanded_yaw_rate_dps": float(
                self._agent1_last_info.get(
                    "commanded_yaw_rate_dps", 0.0
                )
            ),
            "target_velocity_ff_valid": bool(ff_valid),
            "target_velocity_ff_vx_mps": float(ff_vx),
            "target_velocity_ff_vy_mps": float(ff_vy),
            "target_velocity_ff_blocked_by_safety": bool(
                self._agent1_last_info.get(
                    "parallel_target_velocity_ff_blocked_by_safety", False
                )
            ),
            "bottom_guidance_live": bool(
                bottom_guidance.get("measurement_live", False)
            ),
            "bottom_guidance_active": bool(bottom_guidance_active),
            "bottom_guidance_prediction_only": bool(bottom_prediction_only),
            "bottom_xy_authority": (
                "BOTTOM_PREDICTIVE"
                if bottom_live
                else "BOTTOM_SHORT_PREDICTION"
                if bottom_guidance_active
                else "AGENT1_SEARCH_CHASE"
            ),
            "bottom_predictive_catchup": bool(
                bottom_guidance.get("catchup_active", False)
            ),
            "bottom_predicted_err_x": float(
                bottom_guidance.get("predicted_err_x", 0.0)
            ),
            "bottom_predicted_err_y": float(
                bottom_guidance.get("predicted_err_y", 0.0)
            ),
            "bottom_predicted_center_error": float(
                bottom_guidance.get("predicted_center_error", 999.0)
            ),
            "bottom_image_velocity_x_per_s": float(
                bottom_guidance.get("image_velocity_x_per_s", 0.0)
            ),
            "bottom_image_velocity_y_per_s": float(
                bottom_guidance.get("image_velocity_y_per_s", 0.0)
            ),
            "bottom_prediction_horizon_s": float(
                bottom_guidance.get("prediction_horizon_s", 0.0)
            ),
            "bottom_relative_position_x_m": float(
                bottom_guidance.get("relative_position_x_m", 0.0)
            ),
            "bottom_relative_position_y_m": float(
                bottom_guidance.get("relative_position_y_m", 0.0)
            ),
            "bottom_relative_velocity_x_mps": float(
                bottom_guidance.get("relative_velocity_x_mps", 0.0)
            ),
            "bottom_relative_velocity_y_mps": float(
                bottom_guidance.get("relative_velocity_y_mps", 0.0)
            ),
            "bottom_predicted_relative_x_m": float(
                bottom_guidance.get("predicted_relative_x_m", 0.0)
            ),
            "bottom_predicted_relative_y_m": float(
                bottom_guidance.get("predicted_relative_y_m", 0.0)
            ),
            "bottom_controller_source": str(
                bottom_guidance.get("controller_source", "NONE")
            ),
            "bottom_outward_speed_per_s": float(
                bottom_guidance.get("outward_speed_per_s", 0.0)
            ),
            "bottom_guidance_landing_lock": bool(
                bottom_guidance.get("landing_lock", False)
            ),
            "bottom_guidance_blend_strength": float(bottom_blend),
            "bottom_guidance_vx_mps": float(bottom_vx),
            "bottom_guidance_vy_mps": float(bottom_vy),
            "agent1_xy_weight": float(agent1_xy_weight),
            "parallel_physical_steps": int(self._parallel_physical_steps),
        }

    def step(self, action):
        if not self._parallel_active:
            raise RuntimeError(
                "Agent-2 step requested before parallel control became active."
            )
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
