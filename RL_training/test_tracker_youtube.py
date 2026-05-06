import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import yt_dlp

# 🔥 YOUR TRACKER
from object_tracker import tracker


# =========================
# CONFIG
# =========================
WINDOW_NAME = "YouTube Tracker"
DISPLAY_W = 1980
DISPLAY_H = 1200


# =========================
# YOUTUBE LOADER
# =========================
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'best[height<=1080][ext=mp4]',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']


# =========================
# DISPLAY (KEEP ASPECT RATIO)
# =========================
def resize_with_aspect(frame, target_w, target_h):
    h, w = frame.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


# =========================
# PLOT SETUP
# =========================
plt.ion()
fig, ax = plt.subplots()

signal_hist = deque(maxlen=300)
line, = ax.plot([])

ax.set_ylim(0, 1.0)
ax.set_title("Tracker Confidence Signal")
ax.grid(True)


def update_plot(v):
    if v is None:
        return

    signal_hist.append(v)

    line.set_ydata(signal_hist)
    line.set_xdata(range(len(signal_hist)))
    ax.set_xlim(0, max(50, len(signal_hist)))

    fig.canvas.draw()
    fig.canvas.flush_events()


# =========================
# MAIN
# =========================
def main():
    youtube_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"

    print("[INFO] Loading YouTube stream...")
    stream_url = get_youtube_stream(youtube_url)

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Failed to open video stream")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read first frame")
        return

    # =========================
    # ROI SELECTION
    # =========================
    bbox = cv2.selectROI("Select Target", frame, False)
    if bbox == (0, 0, 0, 0):
        print("ROI cancelled")
        return

    x = int(bbox[0] + bbox[2] / 2)
    y = int(bbox[1] + bbox[3] / 2)

    t = tracker()
    cid = t.select_target_and_get_class(frame, x, y)

    if cid is None:
        print("❌ No object detected under ROI")
        return

    print(f"[INFO] Tracking class id: {cid}")

    prev = time.time()

    consecutive_fails = 0
    MAX_CONSECUTIVE_READ_FAILS = 60
    reconnect_attempts = 0
    MAX_RECONNECTS = 3

    # =========================
    # LOOP
    # =========================
    while True:
        ret, frame = cap.read()

        if not ret:
            consecutive_fails += 1

            if consecutive_fails < MAX_CONSECUTIVE_READ_FAILS:
                time.sleep(0.02)
                continue

            reconnect_attempts += 1
            if reconnect_attempts > MAX_RECONNECTS:
                print("⚠ Stream ended")
                break

            print(f"⚠ Reconnecting... ({reconnect_attempts})")

            try:
                stream_url = get_youtube_stream(youtube_url)
            except Exception as e:
                print(f"❌ Failed to refresh stream URL: {e}")
                break

            cap.release()
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print("❌ Failed to reopen stream")
                break

            consecutive_fails = 0
            continue
        else:
            consecutive_fails = 0

        # =========================
        # TRACKER UPDATE
        # =========================
        _ = t.update(frame)

        # =========================
        # FPS
        # =========================
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        # =========================
        # DRAW TRACKER
        # =========================
        t.draw(frame, fps)

        # =========================
        # DEBUG INFO (FIXED)
        # =========================
        cv2.putText(frame, f"STATE: {t.STATE}", (10, 120),
                    0, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"MODE: {t.last_mode}", (10, 150),
                    0, 0.7, (0, 255, 255), 2)

        signal = None
        if hasattr(t, "get_signal_value"):
            try:
                signal = t.get_signal_value()
                cv2.putText(frame, f"SCORE: {signal:.2f}", (10, 180),
                            0, 0.7, (0, 255, 0), 2)
            except:
                signal = None

        # =========================
        # DISPLAY
        # =========================
        frame_display = resize_with_aspect(frame, DISPLAY_W, DISPLAY_H)
        cv2.imshow(WINDOW_NAME, frame_display)

        # =========================
        # SIGNAL PLOT
        # =========================
        update_plot(signal)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()