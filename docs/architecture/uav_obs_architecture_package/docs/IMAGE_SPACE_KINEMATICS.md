# Image-Space Kinematics

This document defines how to compute target motion features from visual bounding boxes.

The target should not be represented in the observation vector using true world position or true world velocity. Instead, the target should be represented through vision-based features.

---

## Normalized Bounding Box

Given:

```text
bbox = (cx, cy, w, h, conf)
image_width = W
image_height = H
```

Compute:

```text
bbox_cx = cx / W
bbox_cy = cy / H
bbox_w  = w  / W
bbox_h  = h  / H
bbox_area = (w * h) / (W * H)
```

All values are normalized to approximately `[0, 1]`.

---

## Centering Error

The image center is `(0.5, 0.5)`.

```text
err_x = 2 * (bbox_cx - 0.5)
err_y = 2 * (bbox_cy - 0.5)
```

The expected range is approximately `[-1, 1]`.

Meaning:

- `err_x > 0`: target is to the right of image center.
- `err_x < 0`: target is to the left of image center.
- `err_y > 0`: target appears lower in the image.
- `err_y < 0`: target appears higher in the image.

---

## Image-Space Velocity

```text
img_vx = (bbox_cx_t - bbox_cx_prev) / dt
img_vy = (bbox_cy_t - bbox_cy_prev) / dt
```

These are not physical velocities in meters per second. They represent movement in normalized image coordinates per second.

Recommended clipping:

```text
img_vx = clip(img_vx, -vmax_img, vmax_img)
img_vy = clip(img_vy, -vmax_img, vmax_img)
```

Suggested initial value:

```text
vmax_img = 2.0
```

---

## Image-Space Acceleration

```text
img_ax = (img_vx_t - img_vx_prev) / dt
img_ay = (img_vy_t - img_vy_prev) / dt
```

Recommended clipping:

```text
img_ax = clip(img_ax, -amax_img, amax_img)
img_ay = clip(img_ay, -amax_img, amax_img)
```

Suggested initial value:

```text
amax_img = 5.0
```

---

## Area Change

Absolute area change:

```text
area_delta = bbox_area_t - bbox_area_prev
```

Relative normalized area change:

```text
area_delta_norm = (bbox_area_t - bbox_area_prev) / (bbox_area_prev + eps)
area_delta_norm = clip(area_delta_norm, -1.0, 1.0)
```

Recommended:

```text
eps = 1e-6
```

Interpretation:

- `area_delta_norm > 0`: target appears larger, likely closer.
- `area_delta_norm < 0`: target appears smaller, likely farther.
- `area_delta_norm ≈ 0`: relative distance is stable.

---

## Distance Proxy

Recommended stable version:

```text
distance_proxy_norm = 1.0 - sqrt(bbox_area)
```

Interpretation:

- Small bbox area -> value closer to 1 -> target is likely far.
- Large bbox area -> value closer to 0 -> target is likely close.

This is not a real metric distance. It is a stable visual proxy for RL.

Distance proxy delta:

```text
distance_proxy_delta = distance_proxy_norm_t - distance_proxy_norm_prev
```

Recommended clipping:

```text
distance_proxy_delta = clip(distance_proxy_delta, -1.0, 1.0)
```

---

## Centered Score

```text
center_error = sqrt(err_x^2 + err_y^2)
max_center_error = sqrt(2)
centered_score = 1.0 - clip(center_error / max_center_error, 0.0, 1.0)
```

Interpretation:

- `centered_score = 1`: target is perfectly centered.
- `centered_score = 0`: target is very far from center.

---

## Target Stability Score

A simple version:

```text
motion_mag = sqrt(img_vx^2 + img_vy^2)
acc_mag = sqrt(img_ax^2 + img_ay^2)
area_motion = abs(area_delta_norm)

target_stability_score = 1.0 - clip(
    0.5 * motion_mag +
    0.3 * acc_mag +
    0.2 * area_motion,
    0.0,
    1.0
)
```

Important note:

For Moving Landing, stability should not mean that the target is stationary in the world. It should mean that the target is stable relative to the camera after the drone has matched its motion.

---

## Landing Allowed

Suggested initial rule:

```text
landing_allowed = (
    has_target == 1 and
    centered_score > 0.85 and
    target_stability_score > 0.75 and
    altitude_norm < 0.7
)
```

This can be used either as an observation feature, as a soft reward helper, or as a safety gate during early training.
