"""Single place to select the training flow.

AGENT_1:
    Runs the original Agent-1 training entry point unchanged.

AGENT_2:
    Trains only the standalone bottom-camera landing agent. The target actor is
    never moved. The drone returns to its first captured pose on each reset.

AGENT_1P2:
    Runs the frozen Agent 1 inside the exact configuration snapshot stored next
    to its checkpoint. After a real ``handoff_success``, a separate Agent-2
    environment takes control without resetting the simulator.
"""

TRAINING_MODE = "AGENT_1P2"

TOTAL_AGENT2_TIMESTEPS = 10_240
CHECKPOINT_FREQUENCY = 2_048

# Agent 1 checkpoint selection. Leave empty to select the checkpoint with the
# largest ``*_steps.zip`` number under models/PPO_Tracker/tracking.
AGENT_1_MODEL_PATH = ""
AGENT_1_DETERMINISTIC = True
AGENT_1_PREPARE_MAX_ATTEMPTS = 3
AGENT_1_PREPARE_MAX_STEPS_PER_ATTEMPT = 700

# Agent 2 resume policy.
RESUME_AGENT_2 = False

# Standalone AGENT_2 does not move/reset the target actor. When the actor name
# is known, it may be supplied here only to READ its API Z. Empty means that
# the initial world-ground reference is used until collision calibration.
AGENT_2_TARGET_ACTOR_NAME = ""
AGENT_2_STATIC_TARGET_SURFACE_ALTITUDE_M = 0.0

# Runtime display/logging.
SHOW_AGENT2_CAMERA = True
AGENT2_MAX_EPISODE_STEPS = 900
