"""Validate browser-drawn outlines and preserve them in original image coordinates."""

import math
from typing import Callable, Any

MAX_VERTICES = 1000
DRAWING_METHODS = {
    "rectangle": "Manual rectangular region",
    "polygon": "Manual polygon region",
    "freehand": "Manual freehand region",
}


def polygon_area(polygon: list[float]) -> float:
    """Continuous polygon area in squared coordinate units (shoelace formula)."""
    points = list(zip(polygon[::2], polygon[1::2]))
    return abs(sum(
        x * next_y - next_x * y
        for (x, y), (next_x, next_y) in zip(points, points[1:] + points[:1])
    )) / 2.0


def _segments_intersect(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    """Include touching/overlapping nonadjacent edges as invalid intersections."""
    def cross(p: tuple, q: tuple, r: tuple) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: tuple, q: tuple, r: tuple) -> bool:
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
                and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    c1, c2, c3, c4 = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if c1 * c2 < 0 and c3 * c4 < 0:
        return True
    return any(abs(value) < 1e-12 and on_segment(p, q, r) for value, p, q, r in (
        (c1, a, b, c), (c2, a, b, d), (c3, c, d, a), (c4, c, d, b)
    ))


def make_manual_region(payload: dict, width: int, height: int) -> dict:
    """Validate a single closed outline; input points are normalized to [0, 1].

    No model is called here. Auto mode has no sensitivity until classified.
    Self-intersections, zero-area outlines, and invalid
    coordinates are rejected instead of silently changing the user's shape.
    """
    if not isinstance(payload, dict):
        raise ValueError("Invalid drawing. Please clear it and draw again.")
    tool = payload.get("tool")
    if not isinstance(tool, str) or tool not in DRAWING_METHODS:
        raise ValueError("Choose rectangle, polygon, or freehand.")
    sensitivity = payload.get("sensitivity")
    if sensitivity not in ("auto", "sensitive", "non_sensitive"):
        raise ValueError("Choose Auto (CLIP), Sensitive, or Non-sensitive before saving.")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not 3 <= len(raw_points) <= MAX_VERTICES:
        raise ValueError(f"Draw a closed shape with 3 to {MAX_VERTICES} points.")
    points = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each drawing point must contain x and y.")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise ValueError("Drawing coordinates must be finite and inside the image.")
        pair = (float(point[0]), float(point[1]))
        if not points or pair != points[-1]:
            points.append(pair)
    if len(points) > 1 and points[-1] == points[0]:
        points.pop()
    if len(points) < 3:
        raise ValueError("The outline needs at least three distinct points.")
    if tool == "rectangle" and (
        len(points) != 4 or len({p[0] for p in points}) != 2
        or len({p[1] for p in points}) != 2 or len(set(points)) != 4
    ):
        raise ValueError("A rectangle must have four corners. Use polygon for other shapes.")
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            if _segments_intersect(a, b, *edges[j]):
                raise ValueError("The outline crosses or touches itself. Undo or clear and redraw.")
    polygon = [value for x, y in points for value in (x * width, y * height)]
    if polygon_area(polygon) < 1.0:
        raise ValueError("The outline is too small. Draw an area of at least one image pixel.")
    xs, ys = polygon[::2], polygon[1::2]
    return {
        "class_name": "unknown", "confidence": None,
        "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
        "polygons": [polygon], "sensitivity": None if sensitivity == "auto" else sensitivity,
        "sensitivity_source": "pending_clip" if sensitivity == "auto" else "human",
        "track_id": None, "source": "manual",
        "segmentation_method": DRAWING_METHODS[tool], "edited_by_human": True,
    }


def classify_manual_region(region: dict, frame_rgb: Any, classifier: Callable) -> None:
    """Classify the original RGB bounding-box crop once, matching stage 2's input.

    The crop includes context inside the enclosing box; it is not a masked crop.
    The score is CLIP's sensitive-category softmax score, not calibrated certainty.
    Update only after a valid result; callers must not store a failed auto region.
    """
    x1, y1, x2, y2 = region["bbox_xyxy"]
    height, width = frame_rgb.shape[:2]
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    if right <= left or bottom <= top:
        raise ValueError("The drawn region has no image pixels to classify.")
    crop = frame_rgb[top:bottom, left:right]
    sensitivity, reason, score = classifier(crop)
    if sensitivity not in ("sensitive", "non_sensitive"):
        raise ValueError("CLIP returned an invalid sensitivity; the drawing was not saved.")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("CLIP returned an invalid score; the drawing was not saved.")
    region.update(
        sensitivity=sensitivity, confidence=score, best_prompt=reason,
        sensitivity_source="clip", clip_suggested_sensitivity=sensitivity,
        needs_review=True,
    )
