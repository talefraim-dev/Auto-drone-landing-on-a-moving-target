"""Single place to select the training flow.

AGENT_1P2 uses parallel dual-agent control:
    Agent 1 stays active on every step and owns XY/Yaw.
    Agent 2 stays active on every step and owns Z only.
    Front and bottom tracking remain active continuously.
"""

TRAINING_MODE = "AGENT_1P2"

TOTAL_AGENT2_TIMESTEPS = 20_480
CHECKPOINT_FREQUENCY = 10_240

# Agent 1 checkpoint selection. Leave empty to select the checkpoint with the
# largest ``*_steps.zip`` number under models/PPO_Tracker/tracking.
AGENT_1_MODEL_PATH = ""
AGENT_1_DETERMINISTIC = True
AGENT_1_PREPARE_MAX_ATTEMPTS = 3
AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT = 700

# Parallel command mixer. The feed-forward is the filtered vehicle velocity in
# drone body axes. Agent 1 adds it to its relative-position correction.
PARALLEL_TARGET_VELOCITY_FEEDFORWARD_GAIN = 1.0
PARALLEL_HORIZONTAL_TOTAL_SPEED_MAX_MPS = 6.0

# Agent 2 resume policy.
RESUME_AGENT_2 = True

# Standalone AGENT_2 options.
AGENT_2_TARGET_ACTOR_NAME = ""
AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M = 0.0

SHOW_AGENT2_CAMERA = True
AGENT2_MAX_EPISODE_STEPS = 900
