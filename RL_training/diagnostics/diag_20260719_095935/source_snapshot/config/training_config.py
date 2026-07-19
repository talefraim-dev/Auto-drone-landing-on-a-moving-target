from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Final PPO training config.
    No CLI flags. No batch files. Run:
        python Run_train.py
    """

    total_timesteps: int = 500_000
    models_root: str = "models/PPO_Tracker"
    log_root: str = "logs"
    metrics_filename: str = "training_metrics.csv"

    # For clean final training, keep False unless you deliberately want resume.
    resume_from_checkpoint: bool = False

    save_freq_steps: int = 10_000
    progress_print_every_steps: int = 2_000

    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99

    device: str = "cuda"
    require_cuda: bool = True

    tensorboard_log_name: str = "PPO_Final_37obs"
