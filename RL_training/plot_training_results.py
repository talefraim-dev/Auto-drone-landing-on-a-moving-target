# plot_training_results.py
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


def find_run_dir(log_root="logs", preferred=None) -> Path | None:
    root = Path(log_root)
    if not root.exists():
        return None

    if preferred:
        cand = root / preferred
        if cand.exists() and cand.is_dir():
            return cand
        cand2 = Path(preferred)
        if cand2.exists() and cand2.is_dir():
            return cand2

    runs = [d for d in root.iterdir() if d.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda d: d.stat().st_mtime)
    return runs[-1]


def merge_scalars(run_dir: Path, wanted_tags: list[str]):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.rglob("events.out.tfevents.*"), key=lambda f: f.stat().st_mtime)
    if not event_files:
        return {}, []

    merged = {t: {} for t in wanted_tags}  # tag -> {step: value}
    available_union = set()

    for ev in event_files:
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 200000})
        ea.Reload()

        tags = ea.Tags().get("scalars", []) or []
        available_union.update(tags)

        for tag in wanted_tags:
            if tag not in tags:
                continue
            for s in ea.Scalars(tag):
                # keep newest value for each step
                merged[tag][int(s.step)] = float(s.value)

    out = {}
    for tag, step_map in merged.items():
        if not step_map:
            continue
        xs = sorted(step_map.keys())
        ys = [step_map[x] for x in xs]
        out[tag] = (xs, ys)

    return out, sorted(list(available_union))


def main():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
    except Exception:
        print("Missing tensorboard package. Install with: pip install tensorboard")
        return

    preferred = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = find_run_dir("logs", preferred)
    if run_dir is None:
        print("No logs/* run directories found.")
        return

    print("Reading run:", run_dir)

    wanted = [
        "time/fps",
        "train/std",
        "train/explained_variance",
        "train/approx_kl",
        "train/entropy_loss",
        "train/value_loss",
        "train/policy_gradient_loss",
    ]

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))

    while True:
        data, available = merge_scalars(run_dir, wanted)

        if not data:
            print("No wanted tags found yet. Available:", available)
            time.sleep(3)
            continue

        ax.clear()

        for tag, (xs, ys) in data.items():
            ax.plot(xs, ys, linewidth=2, label=tag)

            # last point + label
            x_last, y_last = xs[-1], ys[-1]
            ax.scatter([x_last], [y_last], s=35)
            ax.text(
                x_last,
                y_last,
                f"  {tag}={y_last:.4f}",
                fontsize=9,
                va="center",
            )

        ax.set_title(f"Live SB3 scalars (merged events) — {run_dir.name}")
        ax.set_xlabel("timesteps")
        ax.grid(True)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.pause(0.2)

        time.sleep(3)


if __name__ == "__main__":
    main()
