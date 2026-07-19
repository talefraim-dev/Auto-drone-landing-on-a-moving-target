# Automatic diagnostic findings

Diagnostic version: `3.0-parallel-control-authority`

This report is generated from runtime evidence. It does not replace inspection of
`00_control_authority_timeline.csv` and the event-frame pairs.

## Run totals

- Control ticks: **0**
- Instrumented environment step returns: **17**
- Trainable PPO steps: **0**
- Episodes completed: **0**
- Collisions captured: **0**

## Most frequent descent/landing-lock blockers

- No block reason was recorded.

## Safety reasons

- No safety reason was recorded.

## Automatically detected anomalies

- No predefined anomaly was detected.

## Authority/state transitions

- No transition was recorded.

## Episode results

- The diagnostic stopped before an episode completed.

## Files to inspect first

1. `00_control_authority_timeline.csv` — joined perception/control/safety/Z timeline.
2. `27_landing_lock_and_z_gate.jsonl` — exact reason for every Z decision.
3. `26_parallel_xy_arbitration.jsonl` — requested versus fused XY components.
4. `30_safety_lidar_command_suppression.jsonl` — commands removed by safety.
5. `28_bottom_predictive_controller.jsonl` — t+1 prediction and derivative state.
6. `frames/events/` — synchronized visual evidence at each anomaly/transition.
7. `static_code_audit.md` — exact source lines for clipping and authority gates.
