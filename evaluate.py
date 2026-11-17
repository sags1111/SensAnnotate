"""
evaluate.py
-----------
Evaluates the sensitivity classification pipeline against human-labeled
ground truth: precision, recall, F1, and a confusion matrix — the
standard way to evaluate a binary classifier (sensitive / non_sensitive).

"""

import csv
import json
import sys
from pathlib import Path

import cv2

# Load a COCO JSON file and build quick lookup dicts for images and categories by id.
def load_coco(coco_path):
    with open(coco_path) as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    categories_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    return coco, images_by_id, categories_by_id

# Load the original image, or the correct video frame, for a given COCO file_name entry.
def resolve_source_frame(file_name, media_dir: Path):
    
    if "#frame_" in file_name:
        video_name, rest = file_name.split("#frame_")
        frame_idx = int(rest.replace(".jpg", ""))
        video_path = media_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        return frame
    img_path = media_dir / file_name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    return frame

# CLI command: crop out each annotated region from a COCO export and write an unlabeled ground_truth.csv for manual labeling.
def cmd_extract(coco_path, media_dir, out_dir, sample_size=None):
    media_dir = Path(media_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco, images_by_id, categories_by_id = load_coco(coco_path)

    annotations = coco["annotations"]
    if sample_size and len(annotations) > sample_size:
        import random
        random.seed(42)  # reproducible — same sample if you re-run extract
        annotations = random.sample(annotations, sample_size)
        print(f"Randomly sampling {sample_size} of {len(coco['annotations'])} annotations "
              f"(a random subset is statistically valid for precision/recall, and actually "
              f"gets labeled — hand-labeling thousands of crops isn't realistic).")

    frame_cache = {}
    rows = []
    skipped = 0

    for ann in annotations:
        img = images_by_id[ann["image_id"]]
        file_name = img["file_name"]

        if file_name not in frame_cache:
            try:
                frame_cache[file_name] = resolve_source_frame(file_name, media_dir)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  Skipping {file_name}: {e}")
                frame_cache[file_name] = None

        frame = frame_cache[file_name]
        if frame is None:
            skipped += 1
            continue

        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        # Small padding for visual context around tight boxes
        pad = 8
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            skipped += 1
            continue

        crop_name = f"ann_{ann['id']:05d}.jpg"
        cv2.imwrite(str(out_dir / crop_name), crop)

        rows.append({
            "annotation_id": ann["id"],
            "crop_file": crop_name,
            "source_file": file_name,
            "category": categories_by_id[ann["category_id"]],
            "ground_truth_sensitivity": "",  # fill this in by hand
        })

    csv_path = out_dir / "ground_truth.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "annotation_id", "crop_file", "source_file", "category", "ground_truth_sensitivity"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtracted {len(rows)} region crops to {out_dir}/")
    if skipped:
        print(f"({skipped} annotations skipped — source file/frame not found or empty crop)")
    print(f"Now open {csv_path} and fill in ground_truth_sensitivity for as many rows as you can.")
    print("Predicted labels are deliberately NOT included here — label blind, based on the crop image itself.")

# CLI command: compare predicted sensitivity labels against a hand-labeled ground_truth.csv and print precision/recall/F1.
def cmd_score(coco_path, ground_truth_csv):
    coco, images_by_id, categories_by_id = load_coco(coco_path)

    predicted = {}   # annotation_id -> {"sensitivity":, "source":, "category":}
    for ann in coco["annotations"]:
        predicted[ann["id"]] = {
            "sensitivity": ann["attributes"]["sensitivity"],
            "source": ann["attributes"].get("source", "unknown"),
            "category": categories_by_id[ann["category_id"]],
        }

    ground_truth = {}
    with open(ground_truth_csv) as f:
        for row in csv.DictReader(f):
            label = row["ground_truth_sensitivity"].strip().lower()
            if label in ("sensitive", "non_sensitive"):
                ground_truth[int(row["annotation_id"])] = label

    if not ground_truth:
        print("No labeled rows found in the ground truth CSV yet — label some rows first.")
        return

    # Compute and print the confusion matrix and precision/recall/F1 for one subset of annotation ids.
    def score_subset(ann_ids, label):
        tp = fp = tn = fn = 0
        for aid in ann_ids:
            if aid not in predicted or aid not in ground_truth:
                continue
            pred = predicted[aid]["sensitivity"]
            gt = ground_truth[aid]
            if pred == "sensitive" and gt == "sensitive":
                tp += 1
            elif pred == "sensitive" and gt == "non_sensitive":
                fp += 1
            elif pred == "non_sensitive" and gt == "non_sensitive":
                tn += 1
            elif pred == "non_sensitive" and gt == "sensitive":
                fn += 1

        total = tp + fp + tn + fn
        if total == 0:
            print(f"\n[{label}] no labeled+matched annotations — skipping")
            return

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / total

        print(f"\n[{label}]  n={total}")
        print(f"  Confusion matrix (rows=predicted, cols=ground truth):")
        print(f"                    GT sensitive   GT non_sensitive")
        print(f"  Pred sensitive        {tp:>6}          {fp:>6}")
        print(f"  Pred non_sensitive    {fn:>6}          {tn:>6}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}   (of predicted-sensitive, how many really were)")
        print(f"  Recall:    {recall:.3f}   (of truly-sensitive, how many we caught — most important for a privacy tool)")
        print(f"  F1:        {f1:.3f}")

    all_ids = list(ground_truth.keys())
    print(f"Scoring {len(all_ids)} labeled annotations...")
    score_subset(all_ids, "OVERALL")

    by_source = {}
    for aid in all_ids:
        if aid in predicted:
            by_source.setdefault(predicted[aid]["source"], []).append(aid)
    for source, ids in by_source.items():
        score_subset(ids, f"source = {source}")

# CLI command: compare a pre-correction export against a post-correction export to measure how much you had to fix.
def cmd_compare(predicted_path, corrected_path):
    """
    Alternative to the extract/label-CSV/score workflow: instead of
    manually typing labels into a spreadsheet, review and correct
    sensitivity directly in the app's UI (using the existing click-to-
    toggle / "Show" highlight controls), export twice — once BEFORE any
    corrections (the "predicted" baseline) and once AFTER (your verified
    "ground truth") — and compare the two files directly.

    This only works if annotation IDs line up between the two exports,
    which requires that NOTHING was re-detected in between (no "Run
    detection", no "Clear cache") — only sensitivity values changed via
    manual correction. If you see a big mismatch warning below, that's
    what happened; re-export the "predicted" baseline fresh and redo your
    corrections without touching Run Detection in between.
    """
    with open(predicted_path) as f:
        pred_coco = json.load(f)
    with open(corrected_path) as f:
        gt_coco = json.load(f)

    predicted = {a["id"]: {"sensitivity": a["attributes"]["sensitivity"],
                            "source": a["attributes"].get("source", "unknown")}
                 for a in pred_coco["annotations"]}
    ground_truth = {a["id"]: a["attributes"]["sensitivity"] for a in gt_coco["annotations"]}

    pred_ids = set(predicted.keys())
    gt_ids = set(ground_truth.keys())
    matched_ids = pred_ids & gt_ids

    if len(matched_ids) < len(pred_ids) * 0.9:
        print(f"⚠️  WARNING: only {len(matched_ids)}/{len(pred_ids)} annotation IDs match between "
              f"the two files. This usually means detection was re-run between your two exports "
              f"(Run detection / Clear cache), which reshuffles IDs and makes this comparison "
              f"invalid. Re-export the 'predicted' baseline fresh, then correct WITHOUT re-running "
              f"detection, then export again.\n")

    changed = sum(1 for aid in matched_ids if predicted[aid]["sensitivity"] != ground_truth[aid])
    print(f"Comparing {len(matched_ids)} matched annotations ({changed} were corrected, "
          f"{len(matched_ids) - changed} you left as-is / agreed with)\n")

    # Compute and print the confusion matrix and precision/recall/F1 for one subset of matched annotation ids.
    def score_subset(ids, label):
        tp = fp = tn = fn = 0
        for aid in ids:
            pred = predicted[aid]["sensitivity"]
            gt = ground_truth[aid]
            if pred == "sensitive" and gt == "sensitive": tp += 1
            elif pred == "sensitive" and gt == "non_sensitive": fp += 1
            elif pred == "non_sensitive" and gt == "non_sensitive": tn += 1
            elif pred == "non_sensitive" and gt == "sensitive": fn += 1

        total = tp + fp + tn + fn
        if total == 0:
            print(f"\n[{label}] no matched annotations — skipping")
            return
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / total

        print(f"[{label}]  n={total}")
        print(f"  Confusion matrix (rows=predicted, cols=your correction):")
        print(f"                       Your: sensitive   Your: non_sensitive")
        print(f"  Pred sensitive           {tp:>6}              {fp:>6}")
        print(f"  Pred non_sensitive       {fn:>6}              {tn:>6}")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1:        {f1:.3f}\n")

    score_subset(matched_ids, "OVERALL")

    by_source = {}
    for aid in matched_ids:
        by_source.setdefault(predicted[aid]["source"], []).append(aid)
    for source, ids in by_source.items():
        score_subset(ids, f"source = {source}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("extract", "score", "compare"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "extract":
        if len(sys.argv) not in (4, 5):
            print("Usage: python evaluate.py extract <coco_json> <output_dir> [sample_size]")
            print("(media_dir defaults to the current directory — where your uploaded images/videos are)")
            print("(sample_size optional — recommended for large exports, e.g. 50)")
            sys.exit(1)
        sample_size = int(sys.argv[4]) if len(sys.argv) == 5 else None
        cmd_extract(sys.argv[2], ".", sys.argv[3], sample_size)

    elif sys.argv[1] == "compare":
        if len(sys.argv) != 4:
            print("Usage: python evaluate.py compare <predicted_coco.json> <corrected_coco.json>")
            sys.exit(1)
        cmd_compare(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "score":
        if len(sys.argv) != 4:
            print("Usage: python evaluate.py score <coco_json> <ground_truth_csv>")
            sys.exit(1)
        cmd_score(sys.argv[2], sys.argv[3])