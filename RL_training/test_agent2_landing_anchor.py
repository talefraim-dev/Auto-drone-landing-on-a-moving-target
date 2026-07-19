"""Focused regression tests for stable full-target landing-anchor geometry."""
from __future__ import annotations

import numpy as np

from test_agent2_reward_controller_progress_fix import Agent2Config, Agent2LandingEnv


def _env() -> Agent2LandingEnv:
    env = Agent2LandingEnv.__new__(Agent2LandingEnv)
    env.cfg = Agent2Config(show_camera=False)
    env._landing_anchor_px = None
    env._landing_anchor_virtual_bbox_xyxy = None
    env._landing_anchor_full_size_px = None
    env._landing_anchor_aspect_ratio = None
    env._landing_anchor_reference_size_px = None
    env._landing_anchor_reference_height_m = None
    env._landing_anchor_mode = "INIT"
    env._landing_anchor_edge_flags = "NONE"
    return env


def _center(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    return np.asarray([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])


def test_full_target_anchor_equals_full_bbox_center() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    full = np.asarray([500.0, 200.0, 700.0, 800.0], dtype=np.float32)
    anchor, virtual = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=full,
        control_bbox_xyxy=full,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )
    assert anchor is not None and virtual is not None
    assert np.allclose(anchor, _center(full), atol=1.0)
    assert np.allclose(virtual, full, atol=1.0)
    assert env._landing_anchor_mode == "TRACK_FULL"


def test_top_edge_crop_reconstructs_hidden_vehicle_center() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    full = np.asarray([500.0, 200.0, 700.0, 800.0], dtype=np.float32)
    env._update_landing_anchor_geometry(
        raw_bbox_xyxy=full,
        control_bbox_xyxy=full,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )

    # At lower altitude the car is larger and its front half has crossed the top
    # edge. The visible crop center is y=175, but the full-car center is above it.
    edge = np.asarray([450.0, 0.0, 750.0, 350.0], dtype=np.float32)
    anchor, virtual = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=edge,
        control_bbox_xyxy=edge,
        frame_shape=shape,
        live_match=True,
        relative_height_m=1.8,
    )
    assert anchor is not None and virtual is not None
    visible_center_y = float(_center(edge)[1])
    assert float(anchor[1]) < visible_center_y - 120.0, (anchor, edge, virtual)
    assert float(virtual[1]) < 0.0
    assert env._landing_anchor_mode == "EDGE_RECON_T"
    assert env._landing_anchor_edge_flags == "T"


def test_controller_metrics_use_anchor_not_visible_crop_center() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    full = np.asarray([500.0, 200.0, 700.0, 800.0], dtype=np.float32)
    env._update_landing_anchor_geometry(
        raw_bbox_xyxy=full,
        control_bbox_xyxy=full,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )
    edge = np.asarray([450.0, 0.0, 750.0, 350.0], dtype=np.float32)
    anchor, virtual = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=edge,
        control_bbox_xyxy=edge,
        frame_shape=shape,
        live_match=True,
        relative_height_m=1.8,
    )
    anchor_metrics = env._landing_anchor_metrics(
        anchor_px=anchor,
        virtual_bbox_xyxy=virtual,
        visible_bbox_xyxy=edge,
        frame_shape=shape,
    )
    visible_metrics = env._bbox_metrics(edge, shape)
    assert anchor_metrics["err_y"] < visible_metrics["err_y"] - 0.35
    # Existing visible-area speed scheduling is intentionally preserved.
    assert abs(anchor_metrics["area_norm"] - visible_metrics["area_norm"]) < 1.0e-9


def test_pred_gap_holds_last_semantic_anchor() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    full = np.asarray([500.0, 200.0, 700.0, 800.0], dtype=np.float32)
    anchor0, virtual0 = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=full,
        control_bbox_xyxy=full,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )
    anchor1, virtual1 = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=None,
        control_bbox_xyxy=full,
        frame_shape=shape,
        live_match=False,
        relative_height_m=3.0,
    )
    assert np.allclose(anchor1, anchor0)
    assert np.allclose(virtual1, virtual0)
    assert env._landing_anchor_mode == "HOLD_PRED"


def test_non_edge_anchor_follows_verified_motion_without_bbox_center_freeze() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    first = np.asarray([500.0, 200.0, 700.0, 800.0], dtype=np.float32)
    anchor0, _ = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=first,
        control_bbox_xyxy=first,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )
    moved = first + np.asarray([20.0, -30.0, 20.0, -30.0], dtype=np.float32)
    anchor1, _ = env._update_landing_anchor_geometry(
        raw_bbox_xyxy=moved,
        control_bbox_xyxy=moved,
        frame_shape=shape,
        live_match=True,
        relative_height_m=3.0,
    )
    applied = np.asarray(anchor1) - np.asarray(anchor0)
    assert applied[0] >= 17.0
    assert applied[1] <= -25.0



def test_edge_crossing_tracks_true_full_center_continuously() -> None:
    env = _env()
    shape = (1000, 1400, 3)
    errors = []
    anchor_values = []
    for true_cy in (600.0, 500.0, 400.0, 300.0, 200.0, 100.0, 0.0, -100.0):
        full = np.asarray(
            [500.0, true_cy - 300.0, 700.0, true_cy + 300.0],
            dtype=np.float32,
        )
        visible = full.copy()
        visible[0] = max(0.0, visible[0])
        visible[1] = max(0.0, visible[1])
        visible[2] = min(float(shape[1]), visible[2])
        visible[3] = min(float(shape[0]), visible[3])
        anchor, _ = env._update_landing_anchor_geometry(
            raw_bbox_xyxy=visible,
            control_bbox_xyxy=visible,
            frame_shape=shape,
            live_match=True,
            relative_height_m=2.0,
        )
        anchor_values.append(float(anchor[1]))
        errors.append(abs(float(anchor[1]) - true_cy))

    assert max(errors) < 15.0, (errors, anchor_values)
    assert all(
        later < earlier
        for earlier, later in zip(anchor_values, anchor_values[1:])
    )

def test_parallel_snapshot_exports_semantic_anchor() -> None:
    env = _env()
    env._last_info = {
        "bottom_match_live": True,
        "bottom_match_recent": True,
        "bottom_match_confirmed": True,
        "bottom_similarity": 0.95,
        "bottom_landing_anchor_px": [620.0, -25.0],
        "bottom_landing_anchor_mode": "EDGE_RECON_T",
        "bottom_err_x": -0.1,
        "bottom_err_y": -1.05,
    }
    env._control_bbox_xyxy = np.asarray([450.0, 0.0, 750.0, 350.0], dtype=np.float32)
    env._last_bbox_xyxy = env._control_bbox_xyxy.copy()
    env._last_bottom_frame = None
    env._last_bottom_observation_monotonic = 1.0
    snapshot = env.get_parallel_bottom_perception_snapshot()
    assert snapshot["landing_anchor_mode"] == "EDGE_RECON_T"
    assert np.allclose(snapshot["landing_anchor_px"], [620.0, -25.0])
    assert snapshot["err_y"] < -1.0


def test_shared_debug_frame_marks_anchor_without_mutating_raw_frame() -> None:
    env = _env()
    raw_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    env._last_bottom_frame = raw_frame
    env._last_bottom_observation_monotonic = 1.0
    env._control_bbox_xyxy = np.asarray([40.0, 20.0, 120.0, 100.0], dtype=np.float32)
    env._last_bbox_xyxy = env._control_bbox_xyxy.copy()
    env._last_info = {
        "bottom_match_live": True,
        "bottom_match_recent": True,
        "bottom_match_confirmed": True,
        "bottom_similarity": 0.95,
        "bottom_landing_anchor_px": [80.0, 60.0],
        "bottom_landing_anchor_mode": "TRACK_FULL",
    }
    snapshot = env.get_parallel_bottom_perception_snapshot()
    assert snapshot["frame_bgr"] is not raw_frame
    assert np.count_nonzero(snapshot["frame_bgr"]) > 0
    assert np.count_nonzero(raw_frame) == 0


if __name__ == "__main__":
    tests = [name for name in globals() if name.startswith("test_")]
    for name in tests:
        globals()[name]()
    print(f"PASS: {len(tests)} landing-anchor tests")
