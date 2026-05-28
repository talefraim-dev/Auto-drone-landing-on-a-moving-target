# Three-Agent Architecture Overview

## 1. Follow Agent

### Goal
Track a moving target while maintaining a safe distance.

### Expected behavior
- Keep the target visible.
- Keep the target near the image center.
- Avoid getting too close or too far.
- Move smoothly.
- Avoid aggressive vertical motion.

### Important observation features
- `err_x`, `err_y`
- `img_vx`, `img_vy`
- `img_ax`, `img_ay`
- `bbox_area`
- `area_delta_norm`
- `distance_proxy_norm`
- `distance_proxy_delta`
- `drone_vx_norm`, `drone_vy_norm`, `drone_vz_norm`
- `centered_score`
- `lost_target_time_norm`

### Possible action space
```text
action = [
    vx_cmd,
    vy_cmd,
    vz_cmd,
    yaw_rate_cmd
]
```

---

## 2. Static Landing Agent

### Goal
Land safely on a stationary target.

### Expected behavior
- Center above the stationary target.
- Confirm visual stability.
- Descend gradually.
- Land only when alignment is good.
- Avoid drifting during descent.

### Important observation features
- `err_x`, `err_y`
- `bbox_area`
- `distance_proxy_norm`
- `altitude_norm`
- `drone_vz_norm`
- `centered_score`
- `target_stability_score`
- `landing_allowed`

### Possible action space
```text
action = [
    vx_cmd,
    vy_cmd,
    vz_cmd,
    yaw_rate_cmd
]
```

---

## 3. Moving Landing Agent

### Goal
Track, align with, and land on a moving target.

### Expected behavior
- Follow the target.
- Estimate target motion from image-space dynamics.
- Match relative motion.
- Descend only when the target is visually stable and centered.
- Complete landing while the target continues moving.

### Important observation features
Almost the full observation vector is important here, especially:

- `err_x`, `err_y`
- `img_vx`, `img_vy`
- `img_ax`, `img_ay`
- `area_delta_norm`
- `distance_proxy_delta`
- `drone_vx_norm`, `drone_vy_norm`, `drone_vz_norm`
- `altitude_norm`
- `centered_score`
- `target_stability_score`
- `landing_allowed`

### Recommended training order
1. Train Follow Agent.
2. Train Static Landing Agent.
3. Train Moving Landing Agent last.

The Moving Landing Agent is the hardest because it combines tracking, relative motion matching, and controlled descent.
