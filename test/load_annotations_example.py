"""
load_annotations_example.py
----------------------------
A minimal, standalone example showing how to load one of this project's
COCO JSON exports and inspect the annotations for a specific image or
video frame — without needing the Streamlit app at all.

Usage:
    python load_annotations_example.py <coco_json_path> <file_name>

Example (using the files in this folder):
    python load_annotations_example.py example_annotations.json \\
        pexels-nowrin-sanjana-2159843098-37071199.jpg

What this does:
  1. Loads the COCO JSON file.
  2. Finds the image entry matching the given file name (for video, this
     would be something like "myvideo.mp4#frame_37.jpg" — see Chapter 4.7
     of the report for the export format).
  3. Prints every annotation on that image in a readable summary.
  4. Draws the annotations on the actual image (red = sensitive, green =
     non-sensitive, matching the same colours used in the review
     interface) and saves the result alongside the original.

No dependency on any other file in this project — this only needs a COCO
JSON export and the corresponding image file, both of which a user of
the exported data would already have.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RED = (60, 60, 255)     # sensitive   (BGR, since we use cv2 to draw)
GREEN = (90, 200, 60)   # non-sensitive
ALPHA = 0.45


def load_coco(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def find_image_entry(coco: dict, file_name: str) -> dict:
    for img in coco["images"]:
        if img["file_name"] == file_name:
            return img
    raise ValueError(f"No image named {file_name!r} found in this export.")


def annotations_for_image(coco: dict, image_id: int) -> list:
    return [a for a in coco["annotations"] if a["image_id"] == image_id]


def category_name(coco: dict, category_id: int) -> str:
    for c in coco["categories"]:
        if c["id"] == category_id:
            return c["name"]
    return "unknown"


def print_summary(coco: dict, image_entry: dict, anns: list) -> None:
    print(f"\nImage: {image_entry['file_name']}")
    print(f"Size:  {image_entry['width']} x {image_entry['height']}")
    if "fps" in image_entry:
        print(f"Video: {image_entry['fps']:.2f} fps, "
              f"timestamp {image_entry['timestamp_seconds']:.2f}s")
    print(f"Regions found: {len(anns)}\n")

    print(f"{'#':<4}{'class':<15}{'sensitivity':<15}{'confidence':<12}{'source'}")
    print("-" * 65)
    for i, ann in enumerate(anns, 1):
        cls = category_name(coco, ann["category_id"])
        attrs = ann["attributes"]
        conf = attrs.get("confidence")
        conf_str = f"{conf:.3f}" if conf is not None else "n/a"
        print(f"{i:<4}{cls:<15}{attrs['sensitivity']:<15}{conf_str:<12}{attrs['source']}")


def draw_annotations(image_path: str, anns: list, coco: dict, out_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"\n(Skipping visualisation --- could not open {image_path})")
        return

    overlay = image.copy()
    for ann in anns:
        color = RED if ann["attributes"]["sensitivity"] == "sensitive" else GREEN
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            colored = np.zeros_like(image)
            colored[:, :] = color
            m = mask.astype(bool)
            overlay[m] = (overlay[m] * (1 - ALPHA) + colored[m] * ALPHA).astype(np.uint8)
            cv2.polylines(image, [pts], True, color, max(2, image.shape[1] // 800))

    result = cv2.addWeighted(overlay, 1.0, image, 0.0, 0)
    cv2.imwrite(out_path, result)
    print(f"\nAnnotated image saved to: {out_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python load_annotations_example.py <coco_json_path> <file_name>")
        sys.exit(1)

    json_path, file_name = sys.argv[1], sys.argv[2]
    coco = load_coco(json_path)
    image_entry = find_image_entry(coco, file_name)
    anns = annotations_for_image(coco, image_entry["id"])

    print_summary(coco, image_entry, anns)

    # Try to find the actual image file alongside the JSON, to draw on.
    # For video, file_name looks like "myvideo.mp4#frame_37.jpg" and there
    # is no real file with that exact name --- drawing is skipped in that
    # case (see Chapter 4.7 of the report for why video frames aren't
    # materialised as real files).
    candidate = Path(json_path).parent / file_name
    if candidate.exists():
        out_path = str(candidate.with_name(candidate.stem + "_annotated.png"))
        draw_annotations(str(candidate), anns, coco, out_path)
    else:
        print(f"\n(Image file {file_name} not found next to the JSON --- "
              f"skipping visualisation, printed summary above is still complete.)")


if __name__ == "__main__":
    main()