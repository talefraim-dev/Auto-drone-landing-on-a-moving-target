"""Persistent touchdown-time reward for cooperative landing training.

The timing term is applied only after the existing landing-success and RPC
latch checks accept a touchdown. Failed episodes never update the timing
baseline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


STATE_VERSION = "TOUCHDOWN_TIME_CURRICULUM_V2"
WARMUP_SUCCESSES = 20
MEDIAN_WINDOW = 30
MEANINGFUL_RECORD_RATIO = 0.02
SMALL_RECORD_MAX_BONUS = 400.0
LARGE_RECORD_BASE_BONUS = 800.0
LARGE_RECORD_MAX_BONUS = 1_500.0
MEDIAN_FAST_MAX_BONUS = 400.0
MEDIAN_FAST_FULL_REWARD_RATIO = 0.10
MEDIAN_SLOW_MAX_PENALTY = 800.0
MEDIAN_SLOW_FULL_PENALTY_RATIO = 0.25


class TouchdownTimeRewardTracker:
    """Store successful touchdown times and return a bounded timing reward.

    The first 20 verified touchdowns collect the baseline only. Afterwards:
      * every new record is compared only with the previous record;
      * a record improvement below 2% receives a small record bonus;
      * a record improvement of at least 2% receives a large record bonus;
      * a non-record result is compared with the previous rolling median;
      * failed episodes receive no timing reward and do not update the state.
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        warmup_successes: int = WARMUP_SUCCESSES,
        median_window: int = MEDIAN_WINDOW,
    ) -> None:
        self.state_path = Path(state_path)
        self.warmup_successes = max(1, int(warmup_successes))
        self.median_window = max(self.warmup_successes, int(median_window))
        self._episode_started_monotonic: float | None = None
        self.state = self._load_state()
        self._save_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "total_successful_touchdowns": 0,
            "warmup_successes_required": self.warmup_successes,
            "median_window": self.median_window,
            "recent_success_times_s": [],
            "best_touchdown_time_s": None,
            "median_touchdown_time_s": None,
            "last_touchdown_time_s": None,
            "last_time_reward": 0.0,
            "last_classification": "UNINITIALIZED",
            "updated_utc": None,
        }

    def _load_state(self) -> dict[str, Any]:
        state = self._default_state()
        if self.state_path.is_file():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid touchdown-time state {self.state_path}: {exc}"
                ) from exc
            if isinstance(loaded, dict):
                state.update(loaded)

        samples: list[float] = []
        for value in state.get("recent_success_times_s", []):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number) and number > 0.0:
                samples.append(number)

        samples = samples[-self.median_window :]
        state["recent_success_times_s"] = samples
        state["total_successful_touchdowns"] = max(
            int(state.get("total_successful_touchdowns", 0) or 0), len(samples)
        )
        state["warmup_successes_required"] = self.warmup_successes
        state["median_window"] = self.median_window
        state["version"] = STATE_VERSION
        return state

    def _save_state(self) -> None:
        self.state["updated_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    @property
    def warmup_complete(self) -> bool:
        return int(self.state.get("total_successful_touchdowns", 0) or 0) >= self.warmup_successes

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state))

    def begin_episode(self) -> None:
        self._episode_started_monotonic = float(time.monotonic())

    def cancel_episode(self) -> None:
        self._episode_started_monotonic = None

    @staticmethod
    def _small_record_bonus(improvement_ratio: float) -> float:
        fraction = float(np.clip(improvement_ratio / MEANINGFUL_RECORD_RATIO, 0.0, 1.0))
        return fraction * SMALL_RECORD_MAX_BONUS

    @staticmethod
    def _large_record_bonus(improvement_ratio: float) -> float:
        # A 2% record starts at +800. A 10% or greater record reaches +1500.
        extra_fraction = float(
            np.clip(
                (improvement_ratio - MEANINGFUL_RECORD_RATIO) / 0.08,
                0.0,
                1.0,
            )
        )
        return LARGE_RECORD_BASE_BONUS + extra_fraction * (
            LARGE_RECORD_MAX_BONUS - LARGE_RECORD_BASE_BONUS
        )

    @staticmethod
    def _median_reward(touchdown_time_s: float, previous_median_s: float) -> tuple[float, str]:
        if touchdown_time_s < previous_median_s:
            faster_ratio = (previous_median_s - touchdown_time_s) / max(previous_median_s, 1.0e-6)
            fraction = float(
                np.clip(faster_ratio / MEDIAN_FAST_FULL_REWARD_RATIO, 0.0, 1.0)
            )
            return fraction * MEDIAN_FAST_MAX_BONUS, "FASTER_THAN_MEDIAN"

        if touchdown_time_s > previous_median_s:
            slower_ratio = (touchdown_time_s - previous_median_s) / max(previous_median_s, 1.0e-6)
            fraction = float(
                np.clip(slower_ratio / MEDIAN_SLOW_FULL_PENALTY_RATIO, 0.0, 1.0)
            )
            return -(fraction * MEDIAN_SLOW_MAX_PENALTY), "SLOWER_THAN_MEDIAN"

        return 0.0, "AT_MEDIAN"

    def record_success(self, touchdown_time_s: float) -> dict[str, Any]:
        touchdown_time_s = float(touchdown_time_s)
        if not np.isfinite(touchdown_time_s) or touchdown_time_s <= 0.0:
            raise ValueError(
                f"touchdown_time_s must be finite and positive, got {touchdown_time_s!r}"
            )

        total_before = int(self.state.get("total_successful_touchdowns", 0) or 0)
        samples_before = [
            float(value)
            for value in self.state.get("recent_success_times_s", [])
            if np.isfinite(float(value)) and float(value) > 0.0
        ]
        stored_best = self.state.get("best_touchdown_time_s")
        stored_median = self.state.get("median_touchdown_time_s")
        previous_best_s = float(stored_best) if stored_best is not None else touchdown_time_s
        previous_median_s = float(stored_median) if stored_median is not None else touchdown_time_s

        record_improvement_ratio = 0.0
        if total_before < self.warmup_successes:
            reward = 0.0
            classification = "WARMUP"
        elif touchdown_time_s < previous_best_s:
            record_improvement_ratio = (
                previous_best_s - touchdown_time_s
            ) / max(previous_best_s, 1.0e-6)
            if record_improvement_ratio >= MEANINGFUL_RECORD_RATIO:
                reward = self._large_record_bonus(record_improvement_ratio)
                classification = "NEW_RECORD_MEANINGFUL"
            else:
                reward = self._small_record_bonus(record_improvement_ratio)
                classification = "NEW_RECORD_SMALL"
        else:
            reward, classification = self._median_reward(
                touchdown_time_s, previous_median_s
            )

        samples_after = (samples_before + [touchdown_time_s])[-self.median_window :]
        best_after = min(previous_best_s, touchdown_time_s)
        median_after = float(median(samples_after))
        total_after = total_before + 1

        self.state.update(
            {
                "total_successful_touchdowns": total_after,
                "recent_success_times_s": samples_after,
                "best_touchdown_time_s": float(best_after),
                "median_touchdown_time_s": float(median_after),
                "last_touchdown_time_s": touchdown_time_s,
                "last_time_reward": float(reward),
                "last_classification": classification,
            }
        )
        self._save_state()

        return {
            "touchdown_time_reward": float(reward),
            "touchdown_time_s": touchdown_time_s,
            "touchdown_time_classification": classification,
            "touchdown_time_record_improvement_ratio": float(record_improvement_ratio),
            "touchdown_time_previous_best_s": float(previous_best_s),
            "touchdown_time_previous_median_s": float(previous_median_s),
            "touchdown_time_best_s": float(best_after),
            "touchdown_time_median_s": float(median_after),
            "touchdown_time_total_successes": total_after,
            "touchdown_time_warmup_progress": min(total_after, self.warmup_successes),
            "touchdown_time_warmup_required": self.warmup_successes,
            "touchdown_time_warmup_complete": total_after >= self.warmup_successes,
        }

    def finish_episode(
        self,
        *,
        successful_touchdown: bool,
        elapsed_s: float | None = None,
        end_monotonic_s: float | None = None,
    ) -> dict[str, Any]:
        started = self._episode_started_monotonic
        self._episode_started_monotonic = None

        if not successful_touchdown:
            return {
                "touchdown_time_reward": 0.0,
                "touchdown_time_classification": "NO_SUCCESS_NO_UPDATE",
                "touchdown_time_total_successes": int(
                    self.state.get("total_successful_touchdowns", 0) or 0
                ),
                "touchdown_time_warmup_complete": self.warmup_complete,
            }

        if elapsed_s is None:
            if started is None:
                return {
                    "touchdown_time_reward": 0.0,
                    "touchdown_time_classification": "MISSING_TIMER_NO_UPDATE",
                    "touchdown_time_total_successes": int(
                        self.state.get("total_successful_touchdowns", 0) or 0
                    ),
                    "touchdown_time_warmup_complete": self.warmup_complete,
                }
            end_time = (
                float(end_monotonic_s)
                if end_monotonic_s is not None and np.isfinite(end_monotonic_s)
                else float(time.monotonic())
            )
            elapsed_s = float(end_time - started)

        return self.record_success(float(elapsed_s))
