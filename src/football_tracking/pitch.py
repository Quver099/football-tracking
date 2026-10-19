"""Pitch mask: drop detections whose feet are not on the grass (fans, coaches, stewards).

Classic colour thresholding, no deep learning. Use as a safety net after the
fine-tuned detector, not as the main fix: it is sensitive to lighting and shadows.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

GREEN_LOWER = (30, 40, 40)    # HSV
GREEN_UPPER = (90, 255, 255)


def pitch_mask(
    frame: np.ndarray,
    lower: tuple[int, int, int] = GREEN_LOWER,
    upper: tuple[int, int, int] = GREEN_UPPER,
    kernel: int = 25,
) -> np.ndarray:
    """Return a uint8 mask (255 = pitch) for a BGR frame.

    Steps: HSV green threshold -> morphological close/open -> largest region ->
    filled convex hull (so players standing on the grass are inside the mask).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(lower), np.array(upper))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(frame.shape[:2], np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillPoly(mask, [hull], 255)
    return mask


def keep_on_pitch(
    det: sv.Detections, mask: np.ndarray, exempt_classes: list[int] | None = None
) -> sv.Detections:
    """Keep detections whose bottom-centre point lies inside the mask.

    exempt_classes: class ids that skip the test (the ball can be in the air,
    above the stands in the image).
    """
    if len(det) == 0:
        return det
    feet = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
    h, w = mask.shape
    xs, ys = feet[:, 0].clip(0, w - 1), feet[:, 1].clip(0, h - 1)
    on = mask[ys, xs] > 0
    if exempt_classes:
        on |= np.isin(det.class_id, exempt_classes)
    return det[on]
