"""
pipeline.py
-----------
Stage 1: YOLO11m detects known SH17 classes → SAM2 masks → rule-based sensitivity
Stage 2: SAM2 auto-segments remaining regions → CLIP classifies sensitivity
"""

from typing import List, Dict
import numpy as np
import torch
import clip
from PIL import Image

import config
import models
import tracking
from utils import mask_to_polygons
from clip_prompts import SENSITIVE_PROMPTS, NON_SENSITIVE_PROMPTS, ALL_PROMPTS

FACE_CLASSES = {"face", "ear", "head"}


def merge_face_regions(regions: List[Dict], pad_frac: float = 0.25) -> List[Dict]:
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

            pad_x1, pad_y1 = pad_frac * (x2 - x1), pad_frac * (y2 - y1)
            pad_x2, pad_y2 = pad_frac * (x2b - x1b), pad_frac * (y2b - y1b)

            ix1 = max(x1 - pad_x1, x1b - pad_x2)
            iy1 = max(y1 - pad_y1, y1b - pad_y2)
            ix2 = min(x2 + pad_x1, x2b + pad_x2)
            iy2 = min(y2 + pad_y1, y2b + pad_y2)

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
            "track_id": r1.get("track_id"),
            "source": r1.get("source", "unknown"),
        })
        used.add(i)

    return other_regions + merged

# Check whether two bounding boxes overlap enough (intersection over smaller area) to count as the same region.
def boxes_overlap(box1, box2, threshold=0.05):
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


def filter_stage2_regions(raw_regions: List[Dict], yolo_boxes: List[List[float]]) -> List[Dict]:
    """Drop any stage-2 background region that overlaps a *current* stage-1
    box. Kept separate from run_stage2_raw so a cached background result
    from an earlier frame can still be filtered correctly against a later
    frame's live person/face positions — no re-running SAM2/CLIP needed."""
    return [
        r for r in raw_regions
        if not any(boxes_overlap(r["bbox_xyxy"], yb) for yb in yolo_boxes)
    ]


class AnnotationPipeline:
    def __init__(self, yolo_model, sam_model, device: str, yolo_source_key: str):
        """
        yolo_model / sam_model: already-loaded ultralytics YOLO/SAM objects
        (see models.py — loading and caching is that module's job, not
        this class's, so switching YOLO variants doesn't force a SAM/CLIP
        reload too).
        yolo_source_key: which entry in models.YOLO_MODEL_OPTIONS this
        pipeline's YOLO model is — stored on every stage-1 region's
        "source" field so exported COCO data can show which model found
        each object.
        """
        self.device = device
        self.yolo = yolo_model
        self.sam = sam_model
        self.yolo_source_key = yolo_source_key

        # Load CLIP for SAM2 auto regions
        clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=clip_device)
        self.clip_device = clip_device

        # Text prompts never change, so their embedding is computed once
        # here instead of re-tokenizing + re-encoding on every single crop.
        with torch.no_grad():
            sensitive_embeds = self._encode_text_normalized(SENSITIVE_PROMPTS)
            self.sensitive_prototype = sensitive_embeds.mean(dim=0, keepdim=True)
            self.sensitive_prototype = self.sensitive_prototype / self.sensitive_prototype.norm(dim=-1, keepdim=True)

            non_sensitive_embeds = self._encode_text_normalized(NON_SENSITIVE_PROMPTS)
            self.non_sensitive_prototype = non_sensitive_embeds.mean(dim=0, keepdim=True)
            self.non_sensitive_prototype = self.non_sensitive_prototype / self.non_sensitive_prototype.norm(dim=-1, keepdim=True)
        print(f"CLIP loaded on {clip_device}")

    # Tokenize and encode a list of text prompts with CLIP, returning L2-normalized embeddings.
    def _encode_text_normalized(self, prompts: List[str]):
        tokens = clip.tokenize(prompts).to(self.clip_device)
        features = self.clip_model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    DEBUG_CLIP_SCORES = False

    def classify_with_clip(self, crop: np.ndarray) -> tuple:
        """Suggest sensitivity for an RGB crop using two CLIP prompt prototypes.

        Compare image features with the sensitive/non-sensitive text prototypes,
        apply the model's logit scale and a two-class softmax, and choose sensitive
        at a sensitive-category score >= 0.5. This score is not calibrated certainty.
        Stage 2 filters automatic region sizes before calling this method; manual
        drawings instead supply their validated bounding-box crop for human review.
        """
        pil_crop = Image.fromarray(crop)
        image_input = self.clip_preprocess(pil_crop).unsqueeze(0).to(self.clip_device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            sensitive_score = float((logit_scale * image_features @ self.sensitive_prototype.T)[0, 0])
            non_sensitive_score = float((logit_scale * image_features @ self.non_sensitive_prototype.T)[0, 0])

        two_logits = torch.tensor([sensitive_score, non_sensitive_score])
        probs = two_logits.softmax(dim=-1).numpy()
        sensitive_confidence = float(probs[0])

        is_sensitive = sensitive_confidence >= 0.5
        best_score = sensitive_confidence
        best_prompt = "sensitive-category match" if is_sensitive else "non-sensitive-category match"
 
        if self.DEBUG_CLIP_SCORES:
            print(
                f"[CLIP] {pil_crop.size} sensitive_score={sensitive_score:.3f} "
                f"non_sensitive_score={non_sensitive_score:.3f} "
                f"confidence={sensitive_confidence:.3f} "
                f"-> {'SENSITIVE' if is_sensitive else 'non_sensitive'}"
            )

        return ("sensitive" if is_sensitive else "non_sensitive"), best_prompt, best_score

    def run_stage1(self, image_rgb: np.ndarray, use_tracker: bool = False) -> tuple:
        """
        YOLO detection (+ optional ByteTrack tracking) → SAM box-prompted
        masks → rule-based sensitivity. Returns (yolo_regions, yolo_boxes).

        use_tracker=True calls YOLO's .track(persist=True, ...) instead of
        .predict() — this only produces stable, motion-following track IDs
        when called on consecutive frames of the SAME video in order (see
        run_detection_all in app.py). Calling it on an isolated frame still
        works, just without meaningful ID continuity.
        """
        h, w = image_rgb.shape[:2]

        predict_kwargs = dict(
            source=image_rgb,
            conf=config.YOLO_PREDICT_CONF,
            iou=config.YOLO_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        if use_tracker:
            det_results = self.yolo.track(persist=True, tracker="bytetrack.yaml", **predict_kwargs)[0]
        else:
            det_results = self.yolo.predict(**predict_kwargs)[0]

        yolo_regions = []
        yolo_boxes = []
        boxes_xyxy = np.empty((0, 4))
        confs = np.empty((0,))
        cls_ids = np.empty((0,), dtype=int)
        track_ids = np.empty((0,), dtype=int)
        names = det_results.names

        if det_results.boxes is not None and len(det_results.boxes) > 0:
            raw_boxes = det_results.boxes.xyxy.cpu().numpy()
            raw_confs = det_results.boxes.conf.cpu().numpy()
            raw_cls_ids = det_results.boxes.cls.cpu().numpy().astype(int)
            # Only present when use_tracker=True and the tracker actually
            # assigned IDs this frame — None otherwise (e.g. isolated
            # single-frame calls, or a frame with no confirmed tracks yet).
            if det_results.boxes.id is not None:
                raw_track_ids = det_results.boxes.id.cpu().numpy().astype(int)
            else:
                raw_track_ids = np.full(len(raw_boxes), -1, dtype=int)

            # Per-class confidence floor (face/ear/head kept low for
            # recall, everything else needs the tighter default threshold).
            keep_mask = np.array([
                config.keep_detection(names[c], f) for c, f in zip(raw_cls_ids, raw_confs)
            ])
            boxes_xyxy = raw_boxes[keep_mask]
            confs = raw_confs[keep_mask]
            cls_ids = raw_cls_ids[keep_mask]
            track_ids = raw_track_ids[keep_mask]

        if len(boxes_xyxy) > 0:
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
                # track_ids is a numpy int array (can't hold None), so -1 is
                # used internally as "untracked" — converted to None here so
                # the stored value is consistent with every other "not
                # tracked" case in the codebase (stage-2 regions, images).
                _tid = int(track_ids[i]) if len(track_ids) > i else -1
                yolo_regions.append({
                    "class_name": class_name,
                    "confidence": float(confs[i]),
                    "bbox_xyxy": boxes_xyxy[i].tolist(),
                    "polygons": polygons,
                    "sensitivity": sensitivity,
                    "track_id": _tid if _tid != -1 else None,
                    "source": self.yolo_source_key,
                })

        yolo_regions = merge_face_regions(yolo_regions)
        return yolo_regions, yolo_boxes

    def run_stage2_raw(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        SAM2 auto-segments the whole frame → CLIP classifies each region.
        Returns ALL candidate background regions with NO overlap filtering
        against any person/face box — that happens afterward, per-frame,
        via filter_stage2_regions(), so a result cached from one frame can
        still be filtered correctly against a *different* (later) frame's
        live stage-1 boxes.
        """
        h, w = image_rgb.shape[:2]
        sam_auto_results = self.sam.predict(
            source=image_rgb, device=self.device, verbose=False,
        )[0]

        raw_regions = []
        if sam_auto_results.masks is not None:
            auto_masks = sam_auto_results.masks.data.cpu().numpy()
            image_area = h * w

            for mask_data in auto_masks:
                mask = mask_data.astype(np.uint8)
                if mask.shape != (h, w):
                    import cv2
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                area_pct = mask.sum() / image_area
                if area_pct < config.STAGE2_MIN_AREA_PCT or area_pct > config.STAGE2_MAX_AREA_PCT:
                    continue

                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                bbox = [x1, y1, x2, y2]

                crop = image_rgb[y1:y2, x1:x2]
                sensitivity, reason, confidence = self.classify_with_clip(crop)

                polygons = mask_to_polygons(mask)
                if not polygons:
                    continue

                raw_regions.append({
                    "class_name": "unknown",
                    "confidence": confidence,
                    "bbox_xyxy": bbox,
                    "polygons": polygons,
                    "sensitivity": sensitivity,
                    "source": "sam",
                    "best_prompt": reason,
                })

        return raw_regions

    def run_stage2_with_tracking(
        self, image_rgb: np.ndarray, tracker, tracked_meta: Dict[int, Dict], next_obj_id: List[int]
    ) -> List[Dict]:
        """
        Stage-2 detection WITH persistent identity across keyframes.
        Runs the full "segment everything" + CLIP pass here (detection is
        this class's job), then hands off to tracking.match_and_assign_tracks
        for the actual matching/ID-assignment algorithm (that's a tracking
        concern, not a detection concern — see tracking.py).
        """
        raw_regions = self.run_stage2_raw(image_rgb)
        return tracking.match_and_assign_tracks(
            image_rgb, raw_regions, tracker, tracked_meta, next_obj_id, config.STAGE2_TRACK_CAP
        )

    def run_on_image(self, image_rgb: np.ndarray) -> List[Dict]:
        """Single-image path — no tracking, no caching. Used for image uploads."""
        yolo_regions, yolo_boxes = self.run_stage1(image_rgb, use_tracker=False)
        raw_stage2 = self.run_stage2_raw(image_rgb)
        stage2_regions = filter_stage2_regions(raw_stage2, yolo_boxes)
        return yolo_regions + stage2_regions


_pipeline_cache: Dict[str, "AnnotationPipeline"] = {}


def get_pipeline(yolo_model_key: str = None) -> AnnotationPipeline:
    """Cached per YOLO model choice — switching between two models you've
    already used is instant (no reload from disk), and does NOT require
    re-uploading files, since detection results live separately from the
    model that produced them. SAM is loaded once and shared across every
    YOLO variant via models.load_sam's own cache (only YOLO differs
    between the two options — SAM/CLIP work the same either way).

    Known inefficiency, acceptable given the deadline: CLIP is reloaded
    and its prompt embeddings recomputed once per distinct YOLO choice,
    since it's currently created inside AnnotationPipeline.__init__. CLIP
    load is fast relative to YOLO/SAM, so this hasn't been worth the extra
    refactor to fully share it too."""
    yolo_model_key = yolo_model_key or models.DEFAULT_YOLO_MODEL_KEY
    if yolo_model_key not in _pipeline_cache:
        option = models.get_model_option(yolo_model_key)
        yolo_model = models.load_yolo(option.weights_path)
        sam_model = models.load_sam(config.SAM_WEIGHTS)
        _pipeline_cache[yolo_model_key] = AnnotationPipeline(
            yolo_model=yolo_model,
            sam_model=sam_model,
            device=config.DEVICE,
            yolo_source_key=yolo_model_key,
        )
    return _pipeline_cache[yolo_model_key]
