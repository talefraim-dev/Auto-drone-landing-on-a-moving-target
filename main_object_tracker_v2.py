import cv2
import time
import matplotlib.pyplot as plt
from collections import deque

from object_tracker_v2 import tracker


plt.ion()
fig, ax = plt.subplots()
signal_hist = deque(maxlen=250)

line, = ax.plot([])
ax.set_ylim(0, 1.0)
ax.set_title("Tracker Confidence Signal")
ax.set_xlabel("Frame")
ax.set_ylabel("Confidence")
ax.grid(True)


def update_plot(v):
    signal_hist.append(v)
    line.set_ydata(signal_hist)
    line.set_xdata(range(len(signal_hist)))
    ax.set_xlim(0, max(50, len(signal_hist)))
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    print("Click target selection via ROI helper")
    bbox = cv2.selectROI("Select Target", frame, False)
    if bbox == (0, 0, 0, 0):
        print("ROI cancelled")
        return

    x = int(bbox[0] + bbox[2] / 2)
    y = int(bbox[1] + bbox[3] / 2)

    t = tracker()
    cid = t.select_target_and_get_class(frame, x, y)

    if cid is None:
        print("No YOLO object was found under the selected ROI center.")
        return

    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _ = t.update(frame)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        t.draw(frame, fps)
        update_plot(t.get_signal_value())

        cv2.imshow("Object Tracker V2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()