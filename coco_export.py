"""
coco_export.py
---------------
Converts the in-app annotation state into a standard COCO JSON structure.

category_id = the original class name (e.g. "face", "helmet", or a
user-renamed "unknown" object). Everything beyond the base COCO schema —
sensitivity decision, confidence, which model found it, and (for SAM/CLIP
finds) the best-matching prompt — is stored per annotation under
"attributes", a common COCO extension pattern (used by CVAT, Roboflow,
etc.) rather than bespoke top-level keys:

  attributes.source          "finetuned_sh17" | "yolo26l_general" | "sam"
  attributes.sensitivity     "sensitive" | "non_sensitive"
  attributes.confidence      float
  attributes.confidence_type "yolo_detection_score" |
                              "clip_sensitive_softmax_score" | None — scores
                              are NOT on the same scale; check this before
                              filtering/thresholding across sources
  attributes.file_name       redundant with images[].file_name via
                              image_id, duplicated here for convenience
  attributes.best_prompt     the CLIP prompt that matched (source=="sam" only)
  attributes.track_id        None if untracked or not applicable — a
                              single consistent sentinel everywhere, never
                              -1 (see tracking.py / pipeline.py)

Video images[] entries additionally carry "fps" and "timestamp_seconds",
so a detection can be tied to a real moment in the footage, not just a
frame number.
"""

import json
from datetime import datetime
from typing import Dict

import config
from manual_drawing import polygon_area


def _confidence_type(source: str, confidence) -> str | None:
    """Describe the stored score without implying cross-model comparability."""
    if confidence is None:
        return None
    if source == "sam":
        # pipeline.classify_with_clip stores P(sensitive) after a two-logit
        # softmax, not either raw cosine similarity.
        return "clip_sensitive_softmax_score"
    if source in {"finetuned_sh17", "yolo26l_general"}:
        return "yolo_detection_score"
    return "model_score"


def _segmentation_method(region: Dict) -> str:
    if region.get("segmentation_method"):
        return region["segmentation_method"]
    if region.get("source") == "sam":
        return "SAM2 automatic mask"
    if region.get("source") in {"finetuned_sh17", "yolo26l_general"}:
        return "SAM2 box-prompted mask or detector-box fallback"
    return "Unspecified"


def build_coco(annotations_state: Dict) -> Dict:
    """
    annotations_state structure:
    {
      filename: {
        "type": "image" | "video",
        "width": int, "height": int,
        "fps": float | None,   # video only
        "frames": { frame_idx: [region_dict, ...] }
      }
    }
    """
    images = []
    coco_annotations = []
    categories = {}
    next_cat_id = 1
    next_img_id = 1
    next_ann_id = 1

    for filename, filedata in annotations_state.items():
        ftype = filedata["type"]
        width = filedata["width"]
        height = filedata["height"]
        fps = filedata.get("fps")  # video only

        for frame_idx, regions in filedata.get("frames", {}).items():
            if not regions:
                continue

            image_id = next_img_id
            next_img_id += 1

            file_name = filename if ftype == "image" else f"{filename}#frame_{frame_idx}.jpg"
            image_entry = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            if ftype == "video" and fps:
                image_entry["fps"] = fps
                image_entry["timestamp_seconds"] = round(frame_idx / fps, 3)
            images.append(image_entry)

            for region in regions:
                class_name = region["class_name"]
                if class_name not in categories:
                    categories[class_name] = next_cat_id
                    next_cat_id += 1
                cat_id = categories[class_name]

                x1, y1, x2, y2 = region["bbox_xyxy"]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                area = float((x2 - x1) * (y2 - y1))

                source = region.get("source", "unknown")
                if source == "manual":
                    # A hand-drawn outline need not occupy its entire bounding box.
                    area = sum(polygon_area(polygon) for polygon in region["polygons"])
                confidence = region.get("confidence")
                score_source = "sam" if region.get("clip_suggested_sensitivity") else source
                coco_annotations.append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "segmentation": region["polygons"],
                    "area": area,
                    "iscrowd": 0,
                    "attributes": {
                        "source": source,
                        "sensitivity": region["sensitivity"],
                        "confidence": confidence,
                        "confidence_type": _confidence_type(score_source, confidence),
                        "sensitivity_source": region.get("sensitivity_source"),
                        "clip_suggested_sensitivity": region.get("clip_suggested_sensitivity"),
                        "needs_review": bool(region.get("needs_review", False)),
                        "segmentation_method": _segmentation_method(region),
                        "edited_by_human": bool(region.get("edited_by_human", False)),
                        "file_name": file_name,
                        "best_prompt": region.get("best_prompt"),
                        "track_id": region.get("track_id"),
                    },
                })
                next_ann_id += 1

    coco_categories = [
        {"id": cid, "name": name, "supercategory": config.classify_supercategory(name)}
        for name, cid in categories.items()
    ]

    return {
        "info": {
            "description": "Sensitive/Non-sensitive multimodal annotation export",
            "date_created": datetime.utcnow().isoformat(),
        },
        "images": images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

# Build the COCO dict and serialize it to a compact JSON string for export.
def export_to_json_string(annotations_state: Dict) -> str:
    coco = build_coco(annotations_state)

    return json.dumps(coco, separators=(",", ":"))
