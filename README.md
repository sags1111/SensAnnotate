# SensAnnotate

A semi-automatic sensitivity annotation framework for manufacturing images and video.

Combines a rule-based object detector (YOLO, fine-tuned on
[SH17](https://arxiv.org/abs/2407.04590) or a general-purpose COCO model)
with zero-shot classification (SAM2 + CLIP) for content outside either
detector's known classes, multi-object tracking (ByteTrack + SAM2), a
Streamlit review interface, and COCO-compatible export — so a human only
needs to check and correct the system's output rather than annotate from
scratch.

Built as a course project at Hochschule Aalen (MLD), supervised by
Prof. Albert Agisha. Full write-up, architecture rationale, and evaluation
results are in the [project report](./docs/report.pdf).

> **Proposed topic (as given by the supervisor):** Development of a
> Semi-Automatic Framework for Annotating Sensitive and Non-Sensitive
> Multimodal Data Using Object Detection and Segment Anything Models (SAM).

## How it works

1. **Stage 1 — rule-based detection.** A user-selected YOLO model (fine-tuned
   SH17 or general-purpose) detects known classes; sensitivity is decided by
   a fixed lookup table that defaults to *sensitive* for anything not
   explicitly known to be safe.
2. **Stage 2 — zero-shot classification.** SAM2 auto-segments whatever
   Stage 1 didn't cover; each region is compared against averaged
   "sensitive" / "non-sensitive" CLIP prompt embeddings.
3. **Tracking.** ByteTrack follows people/faces across video frames; SAM2's
   own tracker handles background objects at a sparser interval.
4. **Review.** A Streamlit UI shows every detection color-coded on the
   image/frame. Hovering a region explains the model, method, and score;
   reviewers can relabel it, change sensitivity, replace a failed mask with
   a rectangle, delete it, or draw a missing object.
5. **Export.** Results are saved as COCO-format JSON, extended with
   `source`, `confidence_type`, and `track_id` fields.

See Chapter 4 of the report for the full architecture diagram and reasoning.

## Repository structure

```
.
├── config.py          # device selection, sensitivity rules, model paths
├── clip_prompts.py     # CLIP prompt bank
├── models.py           # YOLO/SAM model loading and registry
├── tracking.py         # track ID assignment, matching, consolidation
├── pipeline.py          # per-frame Stage 1 + Stage 2 detection logic
├── utils.py             # mask→polygon conversion, overlay drawing
├── coco_export.py       # COCO JSON export
├── app.py                # Streamlit application
├── evaluate.py            # precision/recall/F1 against corrected exports
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/<your-username>/SensAnnotate.git
cd SensAnnotate
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Model weights

Weights are not checked into this repository. Download them into a
`weights/` folder before running:

```bash
mkdir -p weights
# YOLO — general purpose (Ultralytics)
# https://docs.ultralytics.com/models/yolo26/
#
# YOLO — fine-tuned on SH17 (this project's trained weights)
# link / release TBD
#
# SAM2 (Meta)
# https://github.com/facebookresearch/sam2#model-checkpoints
```

## Usage

```bash
streamlit run app.py
```

Upload an image, video, or a ZIP of either; pick a detection model; run
detection; review and correct results in the Streamlit UI; export as COCO
JSON.

In the editor, known-class regions are red (sensitive) or green (non-sensitive).
Unknown-class regions are orange (sensitive, `#ff8000`) or teal-green
(non-sensitive, `#2ec4a6`). Unknown means the object still needs a semantic
label; it does not override the stored sensitivity decision. Selection uses
a blue border rather than changing the sensitivity color.

Choose **Draw missing object**, then select a shape:

- **Polygon:** click around the boundary; Undo removes the last point.
- **Freehand:** hold and drag to trace the boundary, then release. Returning
  within eight screen pixels of the start closes a sufficiently long stroke;
  further movement before release does not add a crossing tail. Interrupted
  strokes are retained so you can inspect, save, undo, or clear them.
- **Rectangle:** drag between opposite corners.

The outline and shaded area update while drawing. **Auto — CLIP suggestion** is
the default for every new drawing. Its blue draft has no sensitivity assigned yet.
On **Save region**, the existing CLIP classifier processes the original RGB crop
inside the outline's bounding box once. A sensitive suggestion becomes orange;
a non-sensitive suggestion becomes teal-green. The object name remains unknown.
CLIP suggestions, including their sensitive-category score, are flagged for human
review; the score is not calibrated certainty. The crop includes surrounding
pixels inside the bounding box, so its context can influence the suggestion.
You can select Sensitive or Non-sensitive before saving to bypass CLIP, or
correct/confirm the suggestion in the inspector afterward. If CLIP fails, the
app reports the error and keeps the draft available for retry or manual choice;
it does not silently save a sensitive default. Polygon and freehand
outlines close on saving; crossing/degenerate outlines are rejected with an
explanation. Clear discards the current draft. Unsaved drafts are not exported.
Saving adds a manual region on the current image or exact video frame. Auto mode
runs CLIP for that new crop; it does not rerun YOLO/SAM detection or classify all
existing regions again. Shapes are stored at original resolution and exported as
polygons with their enclosing bounding box. Manual-region export area is the
polygon's geometric area, not its enclosing box area.

The drawing component is shipped locally in `drawing_canvas/` and needs no
JavaScript build step or external drawing service. Keep that directory beside
`drawing_canvas.py` when copying or deploying the app. Preview videos and prepared
downloads are invalidated after annotation changes and must be generated again.

Geometry, palette, storage, and export checks can be run without loading models:

```bash
python3 -m unittest discover -s test -p 'test_manual_annotations.py' -v
```

Freehand browser regressions (requires Chrome/Chromium, Pillow, and
`websocket-client` in the test environment):

```bash
python3 test/check_freehand_browser.py
```

This check exercises the real component's live drawing and save payloads,
including loop closure, cancellation, and lost pointer capture. It does not
run the full Streamlit application or model inference.

To reproduce the evaluation numbers from the report:

```bash
python evaluate.py compare --predicted <predicted_export.json> --corrected <corrected_export.json>
```

## Known limitations

- Zero-shot (CLIP/SAM) classification precision (0.54–0.67) remains well
  below rule-based YOLO detection (close to 1.0 in most configurations).
- The "person is always sensitive" rule over-flags full-body detections in
  video, particularly when no face is visible.
- Evaluated on a small number of scenes (3 images, 2 short clips) from a
  single manufacturing environment.

Full discussion in Chapter 11 of the report.

## Tech stack

Python · Streamlit · Ultralytics (YOLO + SAM2) · OpenAI CLIP (ViT-B/32) ·
OpenCV · NumPy

## License

Add a license (e.g. MIT) here before making the repo public, if you haven't
already.
