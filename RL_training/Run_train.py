# Run_train.py
import os
import re
import shutil
import hashlib
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from drone_env import DroneEnv


# --------------------------------------------------
# Paths / config snapshot helpers
# --------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_with_hash(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return sha256_file(dst)


# --------------------------------------------------
# Find latest checkpoint (recursive)
# --------------------------------------------------
def get_latest_checkpoint_recursive(models_dir: str):
    """
    Finds the latest checkpoint based on *_XXXXX_steps.zip naming, recursively.
    Returns (full_path, steps_int) or (None, None).
    """
    if not os.path.isdir(models_dir):
        return None, None

    pattern = re.compile(r".*_(\d+)_steps\.zip$")
    best = None  # (steps, fullpath)

    for root, _dirs, files in os.walk(models_dir):
        for fname in files:
            m = pattern.match(fname)
            if not m:
                continue
            steps = int(m.group(1))
            fullpath = os.path.join(root, fname)
            if best is None or steps > best[0]:
                best = (steps, fullpath)

    if best is None:
        return None, None

    return best[1], best[0]


def find_nearest_config_snapshot(ckpt_path: str):
    """
    Looks for a weights_config snapshot in the same directory as the checkpoint.
    Returns path or None.
    """
    d = os.path.dirname(ckpt_path)
    candidates = [
        os.path.join(d, "weights_config_snapshot.py"),
        os.path.join(d, "weights_config_at_start.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# --------------------------------------------------
# Progress callback (clean, no spam)
# --------------------------------------------------
class PrintProgressCallback(BaseCallback):
    def __init__(self, target_total_timesteps: int, print_every_steps: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.target_total = int(target_total_timesteps)
        self.print_every = int(print_every_steps)

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            done = int(self.model.num_timesteps)
            pct = 100.0 * done / max(1, self.target_total)
            print(f"[TRAIN] timesteps={done}/{self.target_total} ({pct:.1f}%)")
        return True


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    models_root = "models/PPO_Tracker"
    log_dir = "logs"
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Your config file (snapshot it!)
    cfg_src = os.path.join(os.getcwd(), "weights_config.py")
    if not os.path.isfile(cfg_src):
        raise FileNotFoundError(
            "weights_config.py not found next to Run_train.py. "
            "Put weights_config.py in the project root."
        )

    # One env instance (DummyVecEnv)
    env = DummyVecEnv([lambda: DroneEnv()])

    # You can change this target anytime. Resume logic will train only the remaining.
    target_total_timesteps = 200_000

    # --------------------------------------------------
    # Resume logic
    # --------------------------------------------------
    latest_ckpt, ckpt_steps = get_latest_checkpoint_recursive(models_root)

    # Run directory per session (keeps checkpoints + matching config snapshot)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(models_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Snapshot the CURRENT config for this run
    # (So every run has exact config record)
    current_cfg_hash = copy_with_hash(cfg_src, os.path.join(run_dir, "weights_config_snapshot.py"))

    if latest_ckpt is not None:
        print(f"[TRAIN] Resuming from checkpoint: {latest_ckpt} (steps={ckpt_steps})")

        # Compare config snapshots (if exists next to ckpt)
        prev_cfg = find_nearest_config_snapshot(latest_ckpt)
        if prev_cfg is not None:
            prev_hash = sha256_file(prev_cfg)
            if prev_hash != current_cfg_hash:
                print(
                    "[WARN] weights_config changed since the checkpoint was created.\n"
                    f"       checkpoint_cfg_hash={prev_hash[:12]}...\n"
                    f"       current_cfg_hash   ={current_cfg_hash[:12]}...\n"
                    "       This is OK if you changed thresholds/curriculum intentionally,\n"
                    "       but if learning suddenly breaks, this is the first suspect."
                )
            else:
                print("[TRAIN] weights_config matches checkpoint snapshot (hash OK).")
        else:
            print("[WARN] No weights_config snapshot found next to the checkpoint. (Older runs?)")

        # Load model
        model = PPO.load(
            latest_ckpt,
            env=env,
            device="cpu"
        )

        # Train only remaining steps to reach target_total_timesteps
        already = int(getattr(model, "num_timesteps", 0))
        remaining = max(0, target_total_timesteps - already)

        if remaining == 0:
            print(f"[TRAIN] Target already reached: {already}/{target_total_timesteps}. Skipping learn().")
        reset_num_timesteps = False

    else:
        print("[TRAIN] Starting new training (no checkpoints found)")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=log_dir,
            device="cpu"
        )

        remaining = target_total_timesteps
        reset_num_timesteps = True

    # --------------------------------------------------
    # Callbacks
    # - Checkpoints saved INSIDE run_dir to keep them tied to the config snapshot
    # --------------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=run_dir,
        name_prefix="tracker_rl_model"
    )

    progress_callback = PrintProgressCallback(
        target_total_timesteps=target_total_timesteps,
        print_every_steps=2000
    )

    print("\n--- Training Started (checkpoint + auto-resume + config snapshots) ---\n")
    print(f"[TRAIN] run_dir: {run_dir}")
    print(f"[TRAIN] target_total_timesteps: {target_total_timesteps}")
    print(f"[TRAIN] remaining_to_train: {remaining}")

    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=[checkpoint_callback, progress_callback],
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps
            )
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted by user. Saving last model...")

    # --------------------------------------------------
    # Final save (always) + save a copy of current config next to final model
    # --------------------------------------------------
    final_path = os.path.join(run_dir, "ppo_tracker_final")
    model.save(final_path)

    # Also copy the config again as "weights_config_current.py"
    # (so you have both: snapshot-at-start and current state at end)
    copy_with_hash(cfg_src, os.path.join(run_dir, "weights_config_current.py"))

    print(f"[TRAIN] Final model saved: {final_path}")
    print(f"[TRAIN] Config snapshots saved in: {run_dir}")


if __name__ == "__main__":
    main()
