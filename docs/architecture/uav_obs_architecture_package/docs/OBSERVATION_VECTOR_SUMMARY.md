# Observation Vector Summary

## Recommended Base Observation Vector — Version 1

Use one shared 28-feature observation vector for all three agents.

```text
obs = [
    # 0. Detection
    has_target,

    # 1-6. Vision Core
    bbox_cx,
    bbox_cy,
    bbox_w,
    bbox_h,
    bbox_area,
    bbox_conf,

    # 7-8. Centering Error
    err_x,
    err_y,

    # 9-12. Image-Space Kinematics
    img_vx,
    img_vy,
    img_ax,
    img_ay,

    # 13-15. Scale / Distance Dynamics
    area_delta_norm,
    distance_proxy_norm,
    distance_proxy_delta,

    # 16-23. Drone State
    altitude_norm,
    drone_vx_norm,
    drone_vy_norm,
    drone_vz_norm,
    drone_roll_norm,
    drone_pitch_norm,
    drone_yaw_rate_norm,
    drone_speed_xy_norm,

    # 24-27. Mission State
    centered_score,
    target_stability_score,
    landing_allowed,
    lost_target_time_norm
]
```

Total: **28 features**

---

## Feature Table

| Index | Feature | Source | Recommended Range | Purpose |
|---:|---|---|---|---|
| 0 | `has_target` | Vision / Tracker | 0 / 1 | Whether the target is currently detected |
| 1 | `bbox_cx` | Vision | 0..1 | Normalized target center X |
| 2 | `bbox_cy` | Vision | 0..1 | Normalized target center Y |
| 3 | `bbox_w` | Vision | 0..1 | Normalized bbox width |
| 4 | `bbox_h` | Vision | 0..1 | Normalized bbox height |
| 5 | `bbox_area` | Vision | 0..1 | Normalized bbox area |
| 6 | `bbox_conf` | Vision / Tracker | 0..1 | Detection/tracking confidence |
| 7 | `err_x` | Computed | -1..1 | Horizontal centering error |
| 8 | `err_y` | Computed | -1..1 | Vertical centering error |
| 9 | `img_vx` | Computed | clipped | Image-space target velocity X |
| 10 | `img_vy` | Computed | clipped | Image-space target velocity Y |
| 11 | `img_ax` | Computed | clipped | Image-space target acceleration X |
| 12 | `img_ay` | Computed | clipped | Image-space target acceleration Y |
| 13 | `area_delta_norm` | Computed | -1..1 | Relative bbox area change |
| 14 | `distance_proxy_norm` | Computed | 0..1 | Relative distance proxy from bbox area |
| 15 | `distance_proxy_delta` | Computed | -1..1 | Change in relative distance proxy |
| 16 | `altitude_norm` | Sim / Sensor | 0..1 | Normalized drone altitude |
| 17 | `drone_vx_norm` | Sim / Sensor | -1..1 | Normalized drone velocity X |
| 18 | `drone_vy_norm` | Sim / Sensor | -1..1 | Normalized drone velocity Y |
| 19 | `drone_vz_norm` | Sim / Sensor | -1..1 | Normalized vertical velocity |
| 20 | `drone_roll_norm` | Sim / Sensor | -1..1 | Normalized roll |
| 21 | `drone_pitch_norm` | Sim / Sensor | -1..1 | Normalized pitch |
| 22 | `drone_yaw_rate_norm` | Sim / Sensor | -1..1 | Normalized yaw rate |
| 23 | `drone_speed_xy_norm` | Sim / Sensor | 0..1 | Normalized horizontal speed |
| 24 | `centered_score` | Computed | 0..1 | How centered the target is |
| 25 | `target_stability_score` | Computed | 0..1 | How visually stable the target is |
| 26 | `landing_allowed` | Computed | 0 / 1 | Whether landing/descent is currently allowed |
| 27 | `lost_target_time_norm` | Tracker / Env | 0..1 | Normalized time since target was lost |

---

## Design Decision

Use the same observation vector for all agents at first.

The agents should differ by:

- Reward function
- Termination conditions
- Success criteria
- Target behavior
- Episode initialization
- Curriculum strategy

This keeps the engineering clean while still allowing task-specific training.
