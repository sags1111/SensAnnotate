"""
pipeline_llava.py
-----------------
Stage 1: YOLO11m detects known SH17 classes → SAM2 masks → rule-based sensitivity
Stage 2: SAM2 auto-segments remaining regions → LLaVA classifies sensitivity
"""

from typing import List, Dict
import numpy as np
import requests
import base64
import io
from PIL import Image
from ultralytics import YOLO, SAM

import config
from utils import mask_to_polygons

FACE_CLASSES = {"face", "ear", "head"}

LLAVA_URL = "http://localhost:11434/api/generate"

LLAVA_PROMPT = """You are analyzing an image crop from a manufacturing facility.
Is the main object in this image sensitive (reveals personal identity or confidential business information such as faces, ID badges, screens with data, machine labels, documents, barcodes) or non-sensitive (generic equipment, furniture, walls, cables, structural elements)?

Answer in exactly this format:
SENSITIVITY: sensitive or non-sensitive
REASON: one sentence"""


def encode_image(image_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(image_rgb)
    pil_img.thumbnail((512, 512))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def classify_with_llava(image_rgb: np.ndarray) -> tuple:
    """Classify a crop using LLaVA. Falls back to non_sensitive if fails."""
    try:
        if image_rgb.shape[0] < 32 or image_rgb.shape[1] < 32:
            return "non_sensitive", "region too small", 0.0

        image_b64 = encode_image(image_rgb)

        response = requests.post(
            LLAVA_URL,
            json={
                "model": "llava",
                "prompt": LLAVA_PROMPT,
                "images": [image_b64],
                "stream": False
            },
            timeout=30
        )

        result = response.json()["response"].strip()

        sensitivity = "non_sensitive"
        reason = "LLaVA classification"

        for line in result.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith("sensitivity:"):
                value = line_lower.replace("sensitivity:", "").strip()
                sensitivity = "sensitive" if value.startswith("sensitive") else "non_sensitive"
            if line_lower.startswith("reason:"):
                reason = line.replace("REASON:", "").replace("Reason:", "").strip()

        return sensitivity, reason, 1.0

    except Exception as e:
        print(f"LLaVA failed: {e}, defaulting to non_sensitive")
        return "non_sensitive", "LLaVA unavailable", 0.0


def merge_face_regions(regions: List[Dict]) -> List[Dict]:
    face_regions = [r for r in regions if r["class_name"] in FACE_CLASSES]
    other_regions = [r for r in regions if r["class_name"] not in FACE_CLASSES]

    if not face_regions:
        return regions

    merged = []
    used = set()

    for i, r1 in enumerate(face_regions):
        if i in used:
            continue
        x1, y1, x2, y2 = r1["bbox_xyxy"]

        for j, r2 in enumerate(face_regions):
            if i == j or j in used:
                continue
            x1b, y1b, x2b, y2b = r2["bbox_xyxy"]
            ix1, iy1 = max(x1, x1b), max(y1, y1b)
            ix2, iy2 = min(x2, x2b), min(y2, y2b)
            if ix2 > ix1 and iy2 > iy1:
                x1, y1 = min(x1, x1b), min(y1, y1b)
                x2, y2 = max(x2, x2b), max(y2, y2b)
                used.add(j)

        merged.append({
            "class_name": "face",
            "confidence": r1["confidence"],
            "bbox_xyxy": [x1, y1, x2, y2],
            "polygons": r1["polygons"],
            "sensitivity": "sensitive",
        })
        used.add(i)

    return other_regions + merged


def boxes_overlap(box1, box2, threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    smaller = min(area1, area2)
    return (intersection / smaller) > threshold if smaller > 0 else False


class AnnotationPipelineLLaVA:
    def __init__(self, yolo_weights: str, sam_weights: str, device: str):
        self.device = device
        self.yolo = YOLO(yolo_weights)
        self.sam = SAM(sam_weights)
        print("YOLO11m + SAM2 loaded ✅")
        print("LLaVA classifier ready (via Ollama) ✅")

    def run_on_image(self, image_rgb: np.ndarray) -> List[Dict]:
        h, w = image_rgb.shape[:2]

        # ── STAGE 1: YOLO → rule-based sensitivity ───────────────────
        det_results = self.yolo.predict(
            source=image_rgb,
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )[0]

        yolo_regions = []
        yolo_boxes = []

        if det_results.boxes is not None and len(det_results.boxes) > 0:
            boxes_xyxy = det_results.boxes.xyxy.cpu().numpy()
            confs = det_results.boxes.conf.cpu().numpy()
            cls_ids = det_results.boxes.cls.cpu().numpy().astype(int)
            names = det_results.names

            sam_results = self.sam.predict(
                source=image_rgb,
                bboxes=boxes_xyxy.tolist(),
                device=self.device,
                verbose=False,
            )[0]

            sam_masks = (
                sam_results.masks.data.cpu().numpy()
                if sam_results.masks is not None else None
            )

            for i in range(len(boxes_xyxy)):
                class_name = names[cls_ids[i]]
                sensitivity = config.classify_sensitivity(class_name)

                if sam_masks is not None and i < len(sam_masks):
                    mask = sam_masks[i].astype(np.uint8)
                    if mask.shape != (h, w):
                        import cv2
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    polygons = mask_to_polygons(mask)
                else:
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    polygons = [[x1, y1, x2, y1, x2, y2, x1, y2]]

                if not polygons:
                    continue

                yolo_boxes.append(boxes_xyxy[i].tolist())
                yolo_regions.append({
                    "class_name": class_name,
                    "confidence": float(confs[i]),
                    "bbox_xyxy": boxes_xyxy[i].tolist(),
                    "polygons": polygons,
                    "sensitivity": sensitivity,
                })

        yolo_regions = merge_face_regions(yolo_regions)

        # ── STAGE 2: SAM2 auto → LLaVA sensitivity ───────────────────

        sam_auto_results = self.sam.predict(
            source=image_rgb,
            device=self.device,
            verbose=False,
        )[0]

        sam_auto_regions = []
        MAX_REGIONS = 20
        MIN_AREA_PCT = 0.005
        MAX_AREA_PCT = 0.15

        region_count = 0

        if sam_auto_results.masks is not None:
            auto_masks = sam_auto_results.masks.data.cpu().numpy()
            image_area = h * w

            for mask_data in auto_masks:
                if region_count >= MAX_REGIONS:
                    break

                mask = mask_data.astype(np.uint8)
                if mask.shape != (h, w):
                    import cv2
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                area_pct = mask.sum() / image_area

                # Skip background (ceiling, floor, wall, AC) and tiny noise
                if area_pct > MAX_AREA_PCT:
                    continue
                if area_pct < MIN_AREA_PCT:
                    continue

                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                bbox = [x1, y1, x2, y2]

                if any(boxes_overlap(bbox, yb) for yb in yolo_boxes):
                    continue

                crop = image_rgb[y1:y2, x1:x2]
                sensitivity, reason, confidence = classify_with_llava(crop)

                polygons = mask_to_polygons(mask)
                if not polygons:
                    continue

                sam_auto_regions.append({
                    "class_name": "unknown",
                    "confidence": confidence,
                    "bbox_xyxy": bbox,
                    "polygons": polygons,
                    "sensitivity": sensitivity,
                })

                region_count += 1  # ← this was missing

        return yolo_regions + sam_auto_regions


_pipeline_instance = None


def get_pipeline() -> AnnotationPipelineLLaVA:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = AnnotationPipelineLLaVA(
            yolo_weights=config.YOLO_WEIGHTS,
            sam_weights=config.SAM_WEIGHTS,
            device=config.DEVICE,
        )
    return _pipeline_instance
