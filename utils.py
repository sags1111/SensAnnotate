"""
utils.py
--------
Helpers for converting binary masks to COCO polygons, drawing colored
overlays (red = sensitive, green = non-sensitive), and figuring out
which region a user clicked on.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from annotation_colors import RED, GREEN, HIGHLIGHT, color_for

OVERLAY_ALPHA = 0.45


def mask_to_polygons(
    binary_mask: np.ndarray,
    min_area: float = 20.0,
    simplify_epsilon: float = 1.5,
    simplify_above_area: float = 800.0,
) -> List[List[float]]:
    """
    Only simplify contours above `simplify_above_area` (in px^2). Small
    regions — faces, ID badges, etc. — are exactly the ones most likely
    to get blurred/redacted downstream, so they keep full point precision.
    Large SAM2 background blobs (floors, walls, big empty regions) are
    the ones with hundreds of near-collinear points and are what actually
    bloat the COCO export — those get Douglas-Peucker simplified.
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area >= simplify_above_area:
            c = cv2.approxPolyDP(c, simplify_epsilon, True)
        c = c.flatten().tolist()
        if len(c) >= 6:
            polygons.append([float(v) for v in c])
    return polygons


def polygons_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def draw_overlay(image: np.ndarray, regions: list, height: int, width: int, highlight_idx: int = None) -> np.ndarray:
    overlay = image.copy()
    for i, region in enumerate(regions):
        mask = polygons_to_mask(region["polygons"], height, width)
        color = color_for(region["sensitivity"], region.get("class_name"))
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - OVERLAY_ALPHA)
            + np.asarray(color) * OVERLAY_ALPHA
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        if i == highlight_idx:
            # Distinct thick blue border on top of the normal sensitivity
            # color, so the selected region is unambiguous even if its
            # sensitivity color blends into similar-colored neighbors.
            cv2.drawContours(overlay, contours, -1, HIGHLIGHT, 5)
    return overlay


def region_at_point(regions: list, x: int, y: int, height: int, width: int) -> Optional[int]:
    if not (0 <= x < width and 0 <= y < height):
        return None

    best_idx = None
    best_area = None
    for i, region in enumerate(regions):
        contours = [
            np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            for poly in region["polygons"]
            if len(poly) >= 6
        ]
        contains_point = any(
            cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
            for contour in contours
        )
        if contains_point:
            area = sum(abs(cv2.contourArea(contour)) for contour in contours)
            if best_area is None or area < best_area:
                best_area = area
                best_idx = i
    return best_idx


def bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def downsize_gray(image_rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """Small grayscale snapshot of a frame — cheap enough to compare every
    frame, used only to detect scene changes (camera pan/cut/new object),
    never for actual detection."""
    small = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.int16)


def frame_changed(prev_small_gray, curr_small_gray, threshold: float = 12.0) -> bool:
    """True if curr differs enough from prev to treat the background as
    changed rather than just video noise/compression artifacts. Threshold
    is mean absolute pixel difference (0-255 scale) on the downsized frame.
    prev_small_gray=None (no prior reference) always counts as changed."""
    if prev_small_gray is None:
        return True
    diff = np.abs(curr_small_gray.astype(np.int16) - prev_small_gray.astype(np.int16))
    return float(diff.mean()) > threshold

# Detect even a small localized change (e.g. a moving person) using max pixel difference, more sensitive than frame_changed.
def frame_has_local_change(prev_small_gray, curr_small_gray, threshold: float = 15.0) -> bool:
   
    if prev_small_gray is None:
        return True
    diff = np.abs(curr_small_gray.astype(np.int16) - prev_small_gray.astype(np.int16))
    return float(diff.max()) > threshold
