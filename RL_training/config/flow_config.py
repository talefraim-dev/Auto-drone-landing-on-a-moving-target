"""Single place to select the training flow.

AGENT_1P2 uses parallel dual-agent control:
    Agent 1 stays active on every step and owns search/reacquire XY plus yaw.
    Bottom visual motion owns landing XY whenever a trusted bottom view exists.
    Agent 2 stays active on every step and owns Z only.
    Front and bottom tracking remain active continuously.
"""

TRAINING_MODE = "AGENT_1P2"

TOTAL_AGENT2_TIMESTEPS = 2_048
CHECKPOINT_FREQUENCY = 2_048

# Final model locations. The completed project loads both policies from one
# explicit directory and no longer scans legacy training folders.
AGENT_1_MODEL_PATH = "models/FINAL_MODELS/agent1_final.zip"
AGENT_2_MODEL_PATH = "models/FINAL_MODELS/agent2_final.zip"
AGENT_1_DETERMINISTIC = True
AGENT_1_PREPARE_MAX_ATTEMPTS = 3
AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT = 700

# Parallel command mixer. Target velocity is estimated only from bottom-camera
# motion plus the drone's own ego velocity. Target actor XY pose/velocity is
# never used by the controller.
PARALLEL_TARGET_VELOCITY_FEEDFORWARD_GAIN = 1.00
PARALLEL_TARGET_VELOCITY_EMA_ALPHA = 0.45
PARALLEL_HORIZONTAL_TOTAL_SPEED_MAX_MPS = 6.0

# Agent 1 remains a real trainable XY/Yaw participant during landing. The
# deterministic bottom-camera controller supplies a safety/reference correction,
# while a conservative Agent-1 residual remains physically executed so PPO can
# learn final alignment instead of receiving reward for a muted action. Full
# Agent-1 XY returns immediately when bottom guidance is unavailable.
PARALLEL_AGENT1_MIN_XY_WEIGHT_NEAR_LANDING = 0.35
PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_LIVE = 0.35
PARALLEL_AGENT1_XY_WEIGHT_WHEN_BOTTOM_PRED = 0.65
PARALLEL_BOTTOM_PD_CORRECTION_GAIN = 1.00

# Bottom-camera t+1 predictor and catch-up controller. The horizon uses actual
# frame spacing plus a small actuation-latency allowance, then clamps to the
# configured safe range.
PARALLEL_BOTTOM_RELATIVE_VELOCITY_EMA_ALPHA = 0.55
PARALLEL_BOTTOM_PREDICTION_EXTRA_LATENCY_S = 0.08
PARALLEL_BOTTOM_PREDICTION_HORIZON_MIN_S = 0.10
PARALLEL_BOTTOM_PREDICTION_HORIZON_MAX_S = 0.60
PARALLEL_BOTTOM_NORMAL_CORRECTION_MAX_MPS = 1.50
PARALLEL_BOTTOM_CATCHUP_CORRECTION_MAX_MPS = 3.20
PARALLEL_BOTTOM_CATCHUP_ENTER_CENTER_ERROR = 0.24
PARALLEL_BOTTOM_CATCHUP_EXIT_CENTER_ERROR = 0.13

# Safe vertical progress while horizontal metric catch-up remains active. The
# LIVE bottom image, not the stricter metric catch-up latch, decides whether a
# slow descent is geometrically safe.
PARALLEL_CATCHUP_DESCENT_ENABLED = True
PARALLEL_CATCHUP_DESCENT_MAX_VZ_MPS = 0.40
PARALLEL_CATCHUP_DESCENT_TOUCHDOWN_MAX_VZ_MPS = 0.32
PARALLEL_CATCHUP_DESCENT_TOUCHDOWN_HEIGHT_M = 1.00
PARALLEL_CATCHUP_DESCENT_MAX_PREDICTED_CENTER_ERROR = 0.34
PARALLEL_CATCHUP_DESCENT_MAX_IMAGE_SPEED_PER_S = 0.22
PARALLEL_CATCHUP_DESCENT_MAX_OUTWARD_SPEED_PER_S = 0.06
PARALLEL_CATCHUP_DESCENT_MAX_METRIC_OUTWARD_SPEED_MPS = 1.50

# Dense Z-learning reward. Positive reward is paid only for real height progress
# while LIVE identity, landing lock and safe centering are valid. Hovering after
# lock and episode timeout are explicitly worse than completing touchdown.
AGENT2_DENSE_REWARD_ENABLED = True
AGENT2_DENSE_REWARD_ALIGNMENT_CENTER_ERROR = 0.34
AGENT2_DENSE_REWARD_DESCENT_PROGRESS_PER_M = 30.0
AGENT2_DENSE_REWARD_NEAR_TOUCH_HEIGHT_M = 1.50
AGENT2_DENSE_REWARD_NEAR_TOUCH_MULTIPLIER = 1.50
AGENT2_DENSE_REWARD_MAX_PROGRESS_M_PER_STEP = 1.00
AGENT2_DENSE_REWARD_LANDING_LOCK_TIME_PENALTY = 0.35
AGENT2_DENSE_REWARD_HESITATION_PENALTY = 1.50
AGENT2_DENSE_REWARD_MIN_DESCENT_ACTION_WHEN_ALIGNED = 0.25
AGENT2_DENSE_REWARD_UNSAFE_DESCENT_PENALTY = 0.75
AGENT2_TIMEOUT_PENALTY = 600.0
AGENT2_TARGET_LOST_PENALTY = 300.0

# Parallel-only final-metre physical limits. These are applied after the exact
# frozen Agent-1 checkpoint snapshot is loaded; the original Agent-1 config and
# training files remain byte-for-byte unchanged.
PARALLEL_LANDING_BRIDGE_MAX_SPEED_MPS = 0.55
PARALLEL_LANDING_CATCHUP_BRIDGE_MAX_SPEED_MPS = 0.75
PARALLEL_NEAR_GROUND_DESCENT_MAX_MPS = 0.32

# Metric visual-motion controller. Relative target position/velocity are
# reconstructed from bottom-camera geometry and propagated to t+1.
PARALLEL_BOTTOM_METRIC_KP = 1.65
PARALLEL_BOTTOM_METRIC_KD = 0.85
PARALLEL_BOTTOM_CAMERA_HFOV_DEG = 90.0
PARALLEL_BOTTOM_OPTICAL_FLOW_ENABLED = True
PARALLEL_BOTTOM_VISUAL_KALMAN_ENABLED = True

# When Bottom LIVE is genuinely lost, Agent 1 immediately resumes search XY
# while Agent 2 performs a bounded deterministic climb to restore field of view.
PARALLEL_REACQUIRE_CLIMB_ENABLED = True
PARALLEL_REACQUIRE_CLIMB_AFTER_S = 0.45
PARALLEL_REACQUIRE_CLIMB_SPEED_MPS = 0.45
PARALLEL_REACQUIRE_CLIMB_TARGET_HEIGHT_M = 3.20

# Agent 2 resume policy.
RESUME_AGENT_2 = True

# Standalone AGENT_2 options.
AGENT_2_TARGET_ACTOR_NAME = ""
AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M = 0.0

SHOW_AGENT2_CAMERA = True
AGENT2_MAX_EPISODE_STEPS = 900
