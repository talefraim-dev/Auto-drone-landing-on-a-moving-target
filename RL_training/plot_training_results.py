# plot_training_results.py
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# GLOBAL DARK STYLE
# -----------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor": "#111111",
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#DDDDDD",
    "xtick.color": "#BBBBBB",
    "ytick.color": "#BBBBBB",
    "grid.color": "#333333",
    "text.color": "#DDDDDD",
    "legend.frameon": False,
})


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

    event_files = sorted(
        run_dir.rglob("events.out.tfevents.*"),
        key=lambda f: f.stat().st_mtime,
    )
    if not event_files:
        return {}, []

    merged = {t: {} for t in wanted_tags}
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
                merged[tag][int(s.step)] = float(s.value)

    out = {}
    for tag, step_map in merged.items():
        if not step_map:
            continue
        xs = sorted(step_map.keys())
        ys = [step_map[x] for x in xs]
        out[tag] = (xs, ys)

    return out, sorted(list(available_union))


def ema(y: list[float], alpha: float = 0.08) -> np.ndarray:
    """Exponential moving average for nicer readability."""
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def main():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa
    except Exception:
        print("Missing tensorboard package. Install with: pip install tensorboard")
        return

    preferred = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = find_run_dir("logs", preferred)
    if run_dir is None:
        print("No logs/* run directories found.")
        return

    print("Reading run:", run_dir)

    # =============================
    # REALLY INFORMATIVE METRICS
    # =============================
    wanted = [
        # Behavior KPIs (you are blind without these)
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "rollout/success_rate",      # may not exist unless you log success

        # PPO health signals
        "train/explained_variance",
        "train/entropy_loss",
        "train/approx_kl",

        # Perf
        "time/fps",
    ]

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))

    while True:
        data, available = merge_scalars(run_dir, wanted)

        if not data:
            print("No wanted tags found yet.")
            print("Available tags:", available)
            time.sleep(3)
            continue

        ax.clear()

        # Plot order: KPIs first, then training health
        plot_order = [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "rollout/success_rate",
            "train/explained_variance",
            "train/entropy_loss",
            "train/approx_kl",
            "time/fps",
        ]

        for tag in plot_order:
            if tag not in data:
                continue

            xs, ys = data[tag]

            # Smooth only the noisy one
            if tag == "rollout/ep_rew_mean" and len(ys) >= 5:
                ys_plot = ema(ys, alpha=0.08)
                label = f"{tag} (EMA)"
            else:
                ys_plot = np.asarray(ys, dtype=np.float64)
                label = tag

            ax.plot(xs, ys_plot, linewidth=2, label=label)

            x_last, y_last = xs[-1], float(ys_plot[-1])
            ax.scatter([x_last], [y_last], s=28)
            ax.text(
                x_last,
                y_last,
                f"  {label}={y_last:.3f}",
                fontsize=9,
                va="center",
            )

        ax.set_title(f"Training KPIs + PPO health — {run_dir.name}", fontsize=13)
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.3)

        # Show legend only for existing plotted lines
        ax.legend(loc="best")

        plt.tight_layout()
        plt.pause(0.2)
        time.sleep(3)


if __name__ == "__main__":
    main()
