# Target Loss Handling

When the target is not detected, the observation vector must not accidentally describe a good state.

Bad behavior to avoid:

```text
has_target = 0
bbox_cx = 0.5
bbox_cy = 0.5
centered_score = 1
```

This would teach the model that losing the target is good.

---

## Recommended behavior when target is lost

```text
has_target = 0
bbox_cx = 0.5
bbox_cy = 0.5
bbox_w = 0.0
bbox_h = 0.0
bbox_area = 0.0
bbox_conf = 0.0

err_x = previous_err_x
err_y = previous_err_y

img_vx = 0.0
img_vy = 0.0
img_ax = 0.0
img_ay = 0.0

area_delta_norm = 0.0
distance_proxy_norm = 1.0
distance_proxy_delta = 0.0

centered_score = 0.0
target_stability_score = 0.0
landing_allowed = 0.0
lost_target_time_norm += dt / max_lost_time
```

The fake bbox center is used only to keep the observation structure valid. The actual semantic indicators must clearly mark the state as bad:

- `has_target = 0`
- `centered_score = 0`
- `target_stability_score = 0`
- `landing_allowed = 0`
- `lost_target_time_norm` increases
