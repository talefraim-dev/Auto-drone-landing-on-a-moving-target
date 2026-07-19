"""
plot_training_results.py

Python-only plot generator for the final 3-agent training structure.

No CLI arguments.
No environment variables.

It reads TensorBoard event files from:

    logs/PPO_Tracker/<task>/<run_timestamp>/

and saves plots to:

    results/plots/<task>/<run_timestamp>/

Supported tasks:
    - static_landing
    - tracking
    - dynamic_landing

Change SELECTED_PLOT_TASK below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import math

import matplotlib.pyplot as plt


# ============================================================================
# USER CONFIG
# ============================================================================
# Choose which task to plot:
#   "static_landing"
#   "tracking"
#   "dynamic_landing"
SELECTED_PLOT_TASK = "tracking"

# If None, the newest run under logs/PPO_Tracker/<task>/ is used.
# Example:
#   SELECTED_RUN_NAME = "tracking_20260531_120723"
SELECTED_RUN_NAME: Optional[str] = None

LOGS_ROOT = Path("logs") / "PPO_Tracker"
RESULTS_ROOT = Path("results") / "plots"

# Save one PNG per graph.
SAVE_PNG = True

# Also save parsed scalar values to CSV.
SAVE_CSV = True

# Show interactive matplotlib windows after saving.
SHOW_PLOTS = True


@dataclass
class ScalarSeries:
    tag: str
    steps: List[int]
    values: List[float]


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        return EventAccumulator
    except Exception as exc:
        raise RuntimeError(
            "Could not import TensorBoard EventAccumulator. "
            "Install tensorboard in the active conda environment:\n"
            "    pip install tensorboard"
        ) from exc


def find_latest_run(task_name: str) -> Path:
    task_dir = LOGS_ROOT / task_name

    if not task_dir.exists():
        raise FileNotFoundError(f"Task log directory not found: {task_dir}")

    run_dirs = [p for p in task_dir.iterdir() if p.is_dir()]

    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {task_dir}")

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def resolve_run_dir() -> Path:
    if SELECTED_PLOT_TASK not in {"static_landing", "tracking", "dynamic_landing"}:
        raise ValueError(
            "Invalid SELECTED_PLOT_TASK. Use one of: "
            "'static_landing', 'tracking', 'dynamic_landing'."
        )

    if SELECTED_RUN_NAME is None:
        return find_latest_run(SELECTED_PLOT_TASK)

    run_dir = LOGS_ROOT / SELECTED_PLOT_TASK / SELECTED_RUN_NAME

    if not run_dir.exists():
        raise FileNotFoundError(f"Selected run directory not found: {run_dir}")

    return run_dir


def find_event_files(run_dir: Path) -> List[Path]:
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {run_dir}")

    return event_files


def load_scalars(run_dir: Path) -> Dict[str, ScalarSeries]:
    EventAccumulator = _import_event_accumulator()

    # TensorBoard can create more than one event file under a run folder.
    # We merge all scalar events by tag and sort by step.
    merged: Dict[str, List[Tuple[int, float]]] = {}

    event_files = find_event_files(run_dir)

    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()

        for tag in accumulator.Tags().get("scalars", []):
            for ev in accumulator.Scalars(tag):
                merged.setdefault(tag, []).append((int(ev.step), float(ev.value)))

    series: Dict[str, ScalarSeries] = {}

    for tag, items in merged.items():
        items = sorted(items, key=lambda x: x[0])

        # Remove duplicate steps by keeping the last value.
        by_step: Dict[int, float] = {}
        for step, value in items:
            by_step[step] = value

        steps = sorted(by_step.keys())
        values = [by_step[s] for s in steps]

        series[tag] = ScalarSeries(tag=tag, steps=steps, values=values)

    return series


def smooth_series(values: List[float], window: int = 7) -> List[float]:
    if not values:
        return []

    if window <= 1:
        return values[:]

    out = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))

    return out


def save_csv(series: Dict[str, ScalarSeries], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "scalars.csv"

    # Wide CSV by steps is annoying because tags have different step intervals.
    # Use long format: tag, step, value
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "value"])

        for tag in sorted(series.keys()):
            s = series[tag]
            for step, value in zip(s.steps, s.values):
                writer.writerow([tag, step, value])

    print(f"[PLOT] Saved CSV: {csv_path}")


def get_first_existing(series: Dict[str, ScalarSeries], tags: List[str]) -> Optional[ScalarSeries]:
    for tag in tags:
        if tag in series and series[tag].values:
            return series[tag]
    return None


def make_plot(
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    scalar: ScalarSeries,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    smooth_window: int = 7,
    draw_zero_line: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    raw_values = scalar.values
    smooth_values = smooth_series(raw_values, window=smooth_window)

    plt.plot(scalar.steps, raw_values, alpha=0.35, label="raw")
    plt.plot(scalar.steps, smooth_values, linewidth=2.0, label=f"smoothed w={smooth_window}")

    if draw_zero_line:
        plt.axhline(0.0, linestyle="--", linewidth=1.0)

    if y_min is not None or y_max is not None:
        plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_PNG:
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        print(f"[PLOT] Saved: {path}")


def make_loss_plot(output_dir: Path, series: Dict[str, ScalarSeries]) -> None:
    loss = get_first_existing(series, ["train/loss"])
    value_loss = get_first_existing(series, ["train/value_loss"])
    policy_loss = get_first_existing(series, ["train/policy_gradient_loss"])

    if loss is None and value_loss is None and policy_loss is None:
        print("[PLOT] Skipping loss plot: no train/loss tags found yet.")
        return

    plt.figure(figsize=(11, 6))

    if loss is not None:
        plt.plot(loss.steps, smooth_series(loss.values, 5), label="train/loss")
    if value_loss is not None:
        plt.plot(value_loss.steps, smooth_series(value_loss.values, 5), label="train/value_loss")
    if policy_loss is not None:
        plt.plot(policy_loss.steps, smooth_series(policy_loss.values, 5), label="train/policy_gradient_loss")

    plt.title("PPO Training Loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_PNG:
        path = output_dir / "ppo_loss.png"
        plt.savefig(path, dpi=160)
        print(f"[PLOT] Saved: {path}")


def main() -> None:
    run_dir = resolve_run_dir()
    output_dir = RESULTS_ROOT / SELECTED_PLOT_TASK / run_dir.name

    print(f"[PLOT] Selected task : {SELECTED_PLOT_TASK}")
    print(f"[PLOT] Run dir       : {run_dir}")
    print(f"[PLOT] Output dir    : {output_dir}")

    series = load_scalars(run_dir)

    print("[PLOT] Available scalar tags:")
    for tag in sorted(series.keys()):
        print(f"  - {tag}")

    if SAVE_CSV:
        save_csv(series, output_dir)

    # 1. Reward
    reward = get_first_existing(series, ["custom/reward", "rollout/ep_rew_mean"])
    if reward is not None:
        make_plot(
            output_dir=output_dir,
            filename="reward.png",
            title="Reward",
            ylabel="Reward",
            scalar=reward,
            smooth_window=9,
            draw_zero_line=True,
        )
    else:
        print("[PLOT] Skipping reward plot: no custom/reward or rollout/ep_rew_mean found.")

    # 2. PPO loss
    make_loss_plot(output_dir, series)

    # 3. BBox center error [-1, 1], 0 = centered
    bbox_x = get_first_existing(series, ["custom/bbox_center_error_x"])
    bbox_y = get_first_existing(series, ["custom/bbox_center_error_y"])
    bbox_norm = get_first_existing(series, ["custom/bbox_center_error_norm"])

    if bbox_x is not None:
        make_plot(
            output_dir=output_dir,
            filename="bbox_center_error_x.png",
            title="BBox Horizontal Center Error",
            ylabel="Center error X [-1, 1], 0=center",
            scalar=bbox_x,
            y_min=-1.0,
            y_max=1.0,
            smooth_window=9,
            draw_zero_line=True,
        )

    if bbox_y is not None:
        make_plot(
            output_dir=output_dir,
            filename="bbox_center_error_y.png",
            title="BBox Vertical Center Error",
            ylabel="Center error Y [-1, 1], 0=center",
            scalar=bbox_y,
            y_min=-1.0,
            y_max=1.0,
            smooth_window=9,
            draw_zero_line=True,
        )

    if bbox_norm is not None:
        make_plot(
            output_dir=output_dir,
            filename="bbox_center_error_norm.png",
            title="BBox Center Error Norm",
            ylabel="Center error norm [0, 1], 0=center",
            scalar=bbox_norm,
            y_min=0.0,
            y_max=1.0,
            smooth_window=9,
            draw_zero_line=True,
        )
    else:
        print("[PLOT] Skipping bbox center plots: no bbox center tags found.")

    # 4. Flight smoothness [-1, 1], 0 = smooth
    smoothness = get_first_existing(series, ["custom/flight_smoothness"])
    if smoothness is not None:
        make_plot(
            output_dir=output_dir,
            filename="flight_smoothness.png",
            title="Flight Smoothness Error",
            ylabel="Smoothness [-1, 1], 0=smooth",
            scalar=smoothness,
            y_min=-1.0,
            y_max=1.0,
            smooth_window=9,
            draw_zero_line=True,
        )
    else:
        print("[PLOT] Skipping smoothness plot: no custom/flight_smoothness found.")

    # 5. Optional tracking state diagnostics
    match = get_first_existing(series, ["custom/match_pct"])
    pred = get_first_existing(series, ["custom/pred_pct"])
    none = get_first_existing(series, ["custom/none_pct"])

    if match is not None or pred is not None or none is not None:
        plt.figure(figsize=(11, 6))

        if match is not None:
            plt.plot(match.steps, smooth_series(match.values, 5), label="MATCH %")
        if pred is not None:
            plt.plot(pred.steps, smooth_series(pred.values, 5), label="PRED %")
        if none is not None:
            plt.plot(none.steps, smooth_series(none.values, 5), label="NONE %")

        plt.title("Tracking State Ratios")
        plt.xlabel("Timesteps")
        plt.ylabel("Percent")
        plt.ylim(0.0, 100.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if SAVE_PNG:
            path = output_dir / "tracking_state_ratios.png"
            plt.savefig(path, dpi=160)
            print(f"[PLOT] Saved: {path}")

    print("[PLOT] Done.")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
