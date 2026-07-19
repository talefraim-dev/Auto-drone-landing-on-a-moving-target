"""
Manual YOLO + ResNet tracker sanity test for AirSim.

Run from RL_training:
    python test_yolo_resnet_tracker.py

Controls:
    Left click: select target using YOLO bbox under/near click.
    Drag with left mouse button: select exact ROI manually.
    r: reset tracker.
    q / ESC: quit.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import cosysairsim as airsim

from resnet_yolo_tracker import YoloResNetTracker


VEHICLE_NAME = "Drone1"
CAMERA_NAME = "0"
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
WINDOW_NAME = "YOLO + ResNet Tracker Test"
YOLO_MODEL_PATH = "yolo11s.pt"


def get_frame(client: airsim.MultirotorClient) -> np.ndarray:
    responses = client.simGetImages(
        [airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)],
        vehicle_name=VEHICLE_NAME,
    )
    if not responses or not responses[0].image_data_uint8:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    r = responses[0]
    img = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
    frame_rgb = img.reshape(r.height, r.width, 3)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def make_tracker() -> YoloResNetTracker:
    return YoloResNetTracker(
        yolo_model_path=YOLO_MODEL_PATH,
        yolo_conf=0.25,
        min_match_score=0.45,
        appearance_weight=0.75,
        motion_weight=0.25,
        search_window_scale=3.0,
        use_search_window=True,
        ema_alpha=0.70,
        verbose=True,
    )


def main() -> None:
    client = airsim.MultirotorClient()
    client.confirmConnection()

    tracker = make_tracker()
    mouse_state = {"dragging": False, "start": None, "end": None, "frame": None}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, IMAGE_WIDTH, IMAGE_HEIGHT)

    def on_mouse(event, x, y, flags, param):
        nonlocal tracker
        frame = mouse_state["frame"]
        if frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["dragging"] = True
            mouse_state["start"] = (x, y)
            mouse_state["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse_state["dragging"]:
            mouse_state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["dragging"] = False
            start = mouse_state["start"]
            end = (x, y)
            mouse_state["end"] = end
            if start is None:
                return
            x1, y1 = start
            x2, y2 = end
            if abs(x2 - x1) >= 10 and abs(y2 - y1) >= 10:
                bbox = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                print(f"[DRAG ROI] bbox={bbox}")
                tracker.select_target_by_bbox(frame, bbox)
            else:
                print(f"[CLICK] x={x} y={y}")
                tracker.select_target(frame, x, y)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    last_t = time.time()

    while True:
        frame = get_frame(client)
        mouse_state["frame"] = frame.copy()
        now = time.time()
        fps = 1.0 / max(1e-6, now - last_t)
        last_t = now

        tracker.update(frame)
        tracker.draw(frame, fps=fps)

        if mouse_state["dragging"] and mouse_state["start"] is not None and mouse_state["end"] is not None:
            x1, y1 = mouse_state["start"]
            x2, y2 = mouse_state["end"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.putText(frame, "Click YOLO target or drag ROI | r reset | q/ESC quit", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("r"):
            print("[RESET]")
            tracker = make_tracker()
            mouse_state["dragging"] = False
            mouse_state["start"] = None
            mouse_state["end"] = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
