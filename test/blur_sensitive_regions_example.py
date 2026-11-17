"""
blur_sensitive_regions_example.py
-----------------------------------
A concrete, practical demonstration of what the exported annotation data
can actually be used for: automatically blurring every region marked
"sensitive" in an image, using nothing but the exported COCO JSON and the
original image file.

This is deliberately simple --- Gaussian blur applied only inside each
sensitive region's real segmentation mask (not just its bounding box), so
non-sensitive content right next to a sensitive region is left untouched.

Usage:
    python blur_sensitive_regions_example.py <coco_json_path> <file_name>

Example:
    python blur_sensitive_regions_example.py example_annotations.json \\
        pexels-nowrin-sanjana-2159843098-37071199.jpg

Chapter 10 of the report discusses this exact use case (automated
redaction pipelines) in more depth, including why this should only ever
be run on a human-corrected export, not raw system output --- see the
warning printed below.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_coco(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def find_image_entry(coco: dict, file_name: str) -> dict:
    for img in coco["images"]:
        if img["file_name"] == file_name:
            return img
    raise ValueError(f"No image named {file_name!r} found in this export.")


def sensitive_annotations_for_image(coco: dict, image_id: int) -> list:
    return [
        a for a in coco["annotations"]
        if a["image_id"] == image_id and a["attributes"]["sensitivity"] == "sensitive"
    ]


def blur_sensitive_regions(image_path: str, sensitive_anns: list, out_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}")

    # Strong Gaussian blur, applied with a feathered (soft-edged) mask
    # rather than a hard binary one. A hard mask makes the blur/sharp
    # boundary an obvious, ugly seam and -- combined with pixelation --
    # produced the blocky, distracting result seen in earlier output.
    # Feathering the mask edge (via a blur on the mask itself) blends the
    # transition gradually, which reads as a natural soft-focus area
    # instead of a stamped-on block. Kernel size scales with image
    # resolution so the effect is equally strong regardless of image size.
    h, w = image.shape[:2]
    short_side = min(h, w)

    # A single, large-radius Gaussian blur applied directly at full
    # resolution. This sounds naive but tested best in practice: earlier
    # attempts pre-shrank the image before blurring (faster, and a
    # common trick for background blur), but that resize step averages
    # each region down to just a handful of pixels, which erases the
    # boundary between e.g. a blue shirt and a face and collapses the
    # whole silhouette into one flat color -- reads as a cardboard
    # cutout, not a blurred photo. A direct large-kernel blur keeps that
    # color structure (shirt stays shirt-colored, skin stays
    # skin-toned) while still destroying facial detail. Kernel scales
    # with resolution -- tuned empirically at ~2.5% of the short side.
    k = max(31, int(short_side * 0.025))
    if k % 2 == 0:
        k += 1
    heavily_blurred = cv2.GaussianBlur(image, (k, k), 0)

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in sensitive_anns:
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(combined_mask, [pts], 255)

    # Feather the mask edges with their own (smaller) blur, then
    # normalize to a 0..1 float alpha for smooth per-pixel blending.
    # Kept tight (~0.5% of short side) so only the boundary softens --
    # a wide feather was making each person look like a translucent
    # "ghost cutout" rather than a cleanly blurred region.
    feather = max(9, int(short_side * 0.005))
    if feather % 2 == 0:
        feather += 1
    soft_mask = cv2.GaussianBlur(combined_mask, (feather, feather), 0)
    alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in sensitive_anns:
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(combined_mask, [pts], 255)

    # Feather the mask edges with their own (smaller) blur, then
    # normalize to a 0..1 float alpha for smooth per-pixel blending.
    # Kept tight (~0.5% of short side) so only the boundary softens --
    # a wide feather was making each person look like a translucent
    # "ghost cutout" rather than a cleanly blurred region.
    feather = max(9, int(short_side * 0.005))
    if feather % 2 == 0:
        feather += 1
    soft_mask = cv2.GaussianBlur(combined_mask, (feather, feather), 0)
    alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None]

    result = (image.astype(np.float32) * (1 - alpha)
              + heavily_blurred.astype(np.float32) * alpha).astype(np.uint8)

    cv2.imwrite(out_path, result)


def main():
    if len(sys.argv) != 3:
        print("Usage: python blur_sensitive_regions_example.py <coco_json_path> <file_name>")
        sys.exit(1)

    json_path, file_name = sys.argv[1], sys.argv[2]
    coco = load_coco(json_path)
    image_entry = find_image_entry(coco, file_name)
    sensitive_anns = sensitive_annotations_for_image(coco, image_entry["id"])

    print(f"\nImage: {file_name}")
    print(f"Sensitive regions found: {len(sensitive_anns)}")
    if not sensitive_anns:
        print("Nothing to blur --- no regions in this image are marked sensitive.")
        return

    print("\nWARNING (see Chapter 10 of the report): this should only be run on a")
    print("human-corrected export, not raw system output --- Chapter 9 shows the")
    print("system's own raw sensitivity guesses are not reliable enough on their")
    print("own to safely automate redaction from.\n")

    candidate = Path(json_path).parent / file_name
    if not candidate.exists():
        print(f"Image file {file_name} not found next to the JSON --- cannot blur.")
        return

    out_path = str(candidate.with_name(candidate.stem + "_blurred.png"))
    blur_sensitive_regions(str(candidate), sensitive_anns, out_path)
    print(f"Blurred image saved to: {out_path}")


if __name__ == "__main__":
    main()