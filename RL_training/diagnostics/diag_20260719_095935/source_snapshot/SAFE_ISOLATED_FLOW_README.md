# Safe isolated three-agent flow

## Why this version is different

The original Agent-1 runtime was preserved in `Run_train_agent1_original.py`.
Its `DroneEnv`, control pipeline, reward, stage logic, slew limits and tracker
were not rewritten for Agent 2.

`AGENT_1P2` now owns two separate environments:

1. **Agent 1 environment** — built before initialization from the exact
   `training_config_snapshot.json` beside the selected checkpoint.
2. **Agent 2 environment** — independent bottom-camera control, API-Z vertical
   state, horizontal-only LiDAR and collision-gated landing reward.

Agent 2 is attached only after the original Agent-1 environment returns the
literal terminal reason `handoff_success`. Agent-1 rewards and steps are hidden
from the Agent-2 PPO rollout.

## Run

Edit `config/flow_config.py`:

```python
TRAINING_MODE = "AGENT_1P2"
```

Then:

```bat
cd RL_training
python Run_train.py
```

Modes:

- `AGENT_1`: original Agent-1 trainer.
- `AGENT_2`: standalone static-target landing. Target actor is not moved; click
  it in the bottom-camera window.
- `AGENT_1P2`: frozen original Agent 1 prepares the real handoff; Agent 2 then
  trains from that exact simulator state.

## Runtime proof

A valid transition prints:

```text
[AGENT_1P2] Agent 1 handoff accepted exactly from original environment
[AGENT_1P2] Agent 1 OUT -> Agent 2 IN
```

If Agent 1 does not return `handoff_success`, Agent 2 does not receive any PPO
transition and the runner raises a detailed error instead of silently changing
Agent-1 control behavior.
