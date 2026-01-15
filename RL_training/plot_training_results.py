import os
import time
from pathlib import Path

import matplotlib.pyplot as plt

# This is a lightweight reader for scalar summaries from SB3 logs.
# It does NOT require TensorFlow installed.

def find_latest_run(log_dir="logs"):
    p = Path(log_dir)
    if not p.exists():
        return None
    runs = [d for d in p.iterdir() if d.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda d: d.stat().st_mtime)
    return runs[-1]

def main():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        print("Failed to import EventAccumulator. Try: pip install tensorboard (already installed in your env).")
        raise

    run_dir = find_latest_run("logs")
    if run_dir is None:
        print("No logs/* run directories found.")
        return

    print("Reading run:", run_dir)

    plt.ion()
    fig = None

    while True:
        # SB3 creates event files inside run_dir
        event_files = list(run_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            print("No event files yet. Waiting...")
            time.sleep(2)
            continue

        # Pick newest event file
        event_files.sort(key=lambda f: f.stat().st_mtime)
        ev_path = str(event_files[-1])

        ea = EventAccumulator(ev_path, size_guidance={"scalars": 20000})
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        # Common SB3 tags:
        wanted = ["rollout/ep_rew_mean", "rollout/ep_len_mean", "train/explained_variance"]

        data = {}
        for tag in wanted:
            if tag in tags:
                s = ea.Scalars(tag)
                xs = [x.step for x in s]
                ys = [x.value for x in s]
                data[tag] = (xs, ys)

        if not data:
            print("No wanted scalar tags yet. Available tags:", tags[:20])
            time.sleep(3)
            continue

        if fig is None:
            fig = plt.figure()
        plt.clf()

        for tag, (xs, ys) in data.items():
            plt.plot(xs, ys, label=tag)

        plt.legend()
        plt.title(f"Live SB3 scalars (file: {os.path.basename(ev_path)})")
        plt.xlabel("timesteps")
        plt.grid(True)
        plt.pause(0.5)

        time.sleep(3)

if __name__ == "__main__":
    main()
