# Next Tasks

## Immediate next steps

1. Approve the 28-feature base observation vector.
2. Implement `observation_builder.py` as a real module.
3. Connect the observation builder into `drone_env.py`.
4. Define reward function for Follow Agent first.
5. Define termination and success conditions for Follow Agent.
6. After Follow Agent is stable, define Static Landing Agent reward.
7. Train Moving Landing Agent last.

---

## Recommended order

### Step 1 — Observation implementation
Create a real observation builder module and validate outputs.

### Step 2 — Follow Agent reward
Reward for:
- keeping target centered
- maintaining safe distance
- keeping target visible
- smooth control
- avoiding aggressive vertical motion

### Step 3 — Static Landing reward
Reward for:
- precise centering
- controlled descent
- stable visual target
- safe touchdown

### Step 4 — Moving Landing reward
Reward for:
- tracking
- relative motion matching
- controlled descent
- stable final approach
- successful landing on moving target

---

## Debug plots to add later

- `centered_score` over time
- `distance_proxy_norm` over time
- `img_vx`, `img_vy` over time
- `img_ax`, `img_ay` over time
- `target_stability_score` over time
- `landing_allowed` timeline
- episode termination reason distribution
