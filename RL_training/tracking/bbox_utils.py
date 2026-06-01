import numpy as np


def xyxy_to_cxcywh(bbox):
    """
    Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h].
    """
    x1, y1, x2, y2 = bbox

    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))

    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0

    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(box):
    """
    Convert bbox from [cx, cy, w, h] to [x1, y1, x2, y2].
    """
    cx, cy, w, h = box

    w = max(1.0, float(w))
    h = max(1.0, float(h))

    x1 = float(cx) - w / 2.0
    y1 = float(cy) - h / 2.0
    x2 = float(cx) + w / 2.0
    y2 = float(cy) + h / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_center_distance(bbox_a, bbox_b):
    """
    Compute Euclidean distance between centers of two xyxy boxes.
    """
    a = xyxy_to_cxcywh(bbox_a)
    b = xyxy_to_cxcywh(bbox_b)

    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])

    return float(np.sqrt(dx * dx + dy * dy))


def bbox_iou(bbox_a, bbox_b):
    """
    Compute IoU between two xyxy boxes.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def clamp_bbox_to_frame(bbox, frame_width, frame_height):
    """
    Clamp bbox to image boundaries.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]

    frame_width = int(frame_width)
    frame_height = int(frame_height)

    x1 = max(0.0, min(x1, frame_width - 1.0))
    y1 = max(0.0, min(y1, frame_height - 1.0))
    x2 = max(0.0, min(x2, frame_width - 1.0))
    y2 = max(0.0, min(y2, frame_height - 1.0))

    if x2 <= x1:
        x2 = min(frame_width - 1.0, x1 + 1.0)

    if y2 <= y1:
        y2 = min(frame_height - 1.0, y1 + 1.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)
