# tmp.py
# Minimal GRAB FPS sanity test for CosyAirSim (no RL, no tracker).
# Run: python tmp.py
# Keys: q=quit, u=toggle UI, c=toggle compression, r=cycle resolution

import time
import numpy as np
import cv2
import cosysairsim as airsim


CAM_NAME = "0"

# Start settings
USE_UI = True
COMPRESS = False  # ImageRequest compress=True/False
TARGET_SIZES = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
size_idx = 0
OUT_W, OUT_H = TARGET_SIZES[size_idx]

WIN_NAME = "CosyAirSim GRAB FPS (tmp.py)"


def grab_frame(client: airsim.MultirotorClient, cam_name: str, compress: bool):
    req = airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, compress)
    resp = client.simGetImages([req])[0]
    if resp.height == 0 or resp.width == 0 or not resp.image_data_uint8:
        return None, (0, 0)

    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)

    # When compress=True, AirSim returns a PNG byte buffer in image_data_uint8
    if compress:
        bgr = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, (resp.width, resp.height)
        return bgr, (bgr.shape[1], bgr.shape[0])

    # When compress=False, it is raw BGR bytes
    bgr = img1d.reshape(resp.height, resp.width, 3)
    return bgr, (resp.width, resp.height)


def main():
    global USE_UI, COMPRESS, size_idx, OUT_W, OUT_H

    client = airsim.MultirotorClient()
    client.confirmConnection()

    # NOTE: No takeoff/reset here — we only test image pipeline.
    print("Connected!")
    print("Controls: q=quit | u=toggle UI | c=toggle compression | r=cycle resize resolution")
    print(f"Initial: UI={USE_UI}, COMPRESS={COMPRESS}, RESIZE={OUT_W}x{OUT_H}")

    if USE_UI:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, OUT_W, OUT_H)

    # FPS trackers
    t_start = time.perf_counter()
    t_last_print = t_start
    frames = 0

    # Also track decode/resize time
    t_decode_sum = 0.0
    t_resize_sum = 0.0

    while True:
        t0 = time.perf_counter()
        frame, native_wh = grab_frame(client, CAM_NAME, COMPRESS)
        t1 = time.perf_counter()

        if frame is None:
            # If we can't get frames, still allow exit
            key = cv2.waitKey(1) & 0xFF if USE_UI else (cv2.waitKey(1) & 0xFF)
            if key == ord('q'):
                break
            continue

        # If compress=True, decode time is part of t1-t0, but we also report total grab time anyway
        decode_time = (t1 - t0)
        t_decode_sum += decode_time

        # Resize to requested size
        tr0 = time.perf_counter()
        if (frame.shape[1], frame.shape[0]) != (OUT_W, OUT_H):
            frame_disp = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        else:
            frame_disp = frame
        tr1 = time.perf_counter()
        t_resize_sum += (tr1 - tr0)

        frames += 1
        now = time.perf_counter()

        # Print once per second
        if now - t_last_print >= 1.0:
            elapsed = now - t_last_print
            fps = frames / elapsed

            avg_decode_ms = (t_decode_sum / max(frames, 1)) * 1000.0
            avg_resize_ms = (t_resize_sum / max(frames, 1)) * 1000.0

            print(
                f"[GRAB FPS] {fps:5.1f} | "
                f"native={native_wh[0]}x{native_wh[1]} | "
                f"resize={OUT_W}x{OUT_H} | "
                f"compress={COMPRESS} | "
                f"avg_grab(ms)={avg_decode_ms:6.2f} | "
                f"avg_resize(ms)={avg_resize_ms:6.2f}"
            )

            # reset counters for next second window
            frames = 0
            t_decode_sum = 0.0
            t_resize_sum = 0.0
            t_last_print = now

        if USE_UI:
            # Minimal overlay (optional)
            cv2.putText(
                frame_disp,
                f"compress={COMPRESS} resize={OUT_W}x{OUT_H}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow(WIN_NAME, frame_disp)

            key = cv2.waitKey(1) & 0xFF
        else:
            # still pump events
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('u'):
            USE_UI = not USE_UI
            if USE_UI:
                cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WIN_NAME, OUT_W, OUT_H)
                print(f"[UI] ON  window={OUT_W}x{OUT_H}")
            else:
                try:
                    cv2.destroyWindow(WIN_NAME)
                except Exception:
                    pass
                print("[UI] OFF")
        elif key == ord('c'):
            COMPRESS = not COMPRESS
            print(f"[COMPRESS] set to {COMPRESS}")
        elif key == ord('r'):
            size_idx = (size_idx + 1) % len(TARGET_SIZES)
            OUT_W, OUT_H = TARGET_SIZES[size_idx]
            if USE_UI:
                cv2.resizeWindow(WIN_NAME, OUT_W, OUT_H)
            print(f"[RESIZE] set to {OUT_W}x{OUT_H}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
