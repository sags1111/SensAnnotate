"""
tracking.py
-----------
Everything about giving detected objects a persistent identity ACROSS
frames lives here — kept separate from:
  - pipeline.py: single-frame detection (what's in THIS frame)
  - app.py: the frame-walking loop that decides WHEN to call into this
    module (stays there since it's an inherently sequential, stateful
    UI-orchestration loop — pulling it apart would scatter a currently
    readable step-by-step process into disconnected pieces)

Contains:
  new_stage2_tracker()      — SAM2 dynamic tracker factory (stateful,
                               one instance per video, never shared/cached)
  match_and_assign_tracks() — the stage-2 matching/ID-assignment algorithm
  consolidate_track_labels()— cross-frame class/sensitivity consistency
                               for stage-1 (person/face) tracks
"""

from typing import Dict, List

# Create a fresh, stateful SAM2 dynamic tracker instance for one video (never reused across videos).
def new_stage2_tracker(weights_path: str, max_obj_num: int):
   
    from ultralytics.models.sam import SAM2DynamicInteractivePredictor

    overrides = {
        "conf": 0.25, "task": "segment", "mode": "predict",
        "imgsz": 1024, "model": weights_path, "save": False,
    }
    return SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=max_obj_num)

# Check whether two bounding boxes overlap enough (intersection over smaller area) to count as the same object.
def _boxes_overlap(box1, box2, threshold=0.3):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    smaller = min(area1, area2)
    return (intersection / smaller) > threshold if smaller > 0 else False

# Match freshly detected regions to existing tracked objects (by overlap) and assign track_ids, seeding new tracks up to the cap.
def match_and_assign_tracks(
    image_rgb, raw_regions: List[Dict], tracker, tracked_meta: Dict[int, Dict],
    next_obj_id: List[int], track_cap: int,
) -> List[Dict]:
    """
    Given a fresh "ground truth" detection pass (raw_regions, from
    pipeline.py's run_stage2_raw), matches regions against previously-
    tracked positions and assigns/reuses track_ids in place.

    A full fresh detection pass is always the ground truth here — SAM2's
    own docs warn it "may lose track of objects during extended
    sequences" and "can sometimes confuse similar-looking objects", so
    pure tracking without a ground-truth check risked silent drift. This
    function only handles matching fresh regions to existing tracked
    identity on top of that, and seeding new ones.

    tracker: a new_stage2_tracker() instance — STATEFUL, must be the SAME
    instance across every call for one video (see that function's docstring).
    tracked_meta: {track_id: region_dict}, mutated in place as objects are
    matched or newly added — the caller owns this dict across the whole
    video's processing.
    next_obj_id: single-element list used as a mutable counter, so this
    call can advance the caller's next-available track_id.
    track_cap: max distinct objects tracked (see config.STAGE2_TRACK_CAP).

    Returns raw_regions with "track_id" set on each (None if the cap was
    already reached for that region).

    NOT independently verified against a running SAM2DynamicInteractivePredictor
    — built from Ultralytics' documented usage pattern (see
    docs.ultralytics.com/models/sam-2/#dynamic-interactive-segment-and-track),
    but the exact shape of what tracker(source=...) returns when called
    with no new prompts (attribute names on the result object) is
    inferred to follow the same results[0].boxes/.masks pattern as other
    ultralytics predict() calls, not confirmed line-by-line against real
    output. Test this on a short video first.
    """
    for r in raw_regions:
        r["track_id"] = None

    # Ask the tracker where currently-tracked objects are in this frame.
    tracked_now = []
    if tracked_meta:
        results = tracker(source=image_rgb)
        res = results[0]
        if res.boxes is not None and getattr(res.boxes, "id", None) is not None:
            obj_ids_now = res.boxes.id.cpu().numpy().astype(int)
            boxes_now = res.boxes.xyxy.cpu().numpy()
            for i, obj_id in enumerate(obj_ids_now):
                if int(obj_id) in tracked_meta:
                    tracked_now.append((int(obj_id), boxes_now[i].tolist()))

    # Match fresh regions to tracked positions by overlap — a match means
    # "same object as before", so reuse its track_id.
    used_idx = set()
    for obj_id, tbox in tracked_now:
        for j, r in enumerate(raw_regions):
            if j in used_idx:
                continue
            if _boxes_overlap(r["bbox_xyxy"], tbox, threshold=0.3):
                r["track_id"] = obj_id
                used_idx.add(j)
                tracked_meta[obj_id] = r
                break

    # Assign fresh track_ids to previously-untracked regions, largest
    # first, up to the cap — smaller/less significant regions are left
    # untracked rather than spending SAM2's expensive per-object calls
    # on background noise.
    def region_area(r):
        x1, y1, x2, y2 = r["bbox_xyxy"]
        return max(0, x2 - x1) * max(0, y2 - y1)

    untracked = sorted(
        (r for j, r in enumerate(raw_regions) if j not in used_idx),
        key=region_area, reverse=True,
    )

    # SAM2's own tracker requires every obj_id to stay NUMERICALLY below
    # max_obj_num for the tracker's entire lifetime (confirmed by a real
    # crash during testing: "assert obj_id < self._max_obj_num" inside
    # Ultralytics' own predict.py) — it is not enough to just keep the
    # COUNT of active objects under the cap, as this function used to do
    # with an ever-increasing next_obj_id counter. Over a long video, that
    # counter could climb past track_cap even while never more than
    # track_cap objects were tracked at once, since a counter never
    # reuses a number once given out. Fixed by picking the smallest
    # unused slot in [0, track_cap) each time instead of always taking
    # the next new number — slots free up as old tracks stay out of
    # tracked_meta's current key set.
    used_ids = set(tracked_meta.keys())
    free_ids = sorted(set(range(track_cap)) - used_ids)

    new_bboxes, new_obj_ids, newly_seeded = [], [], []
    for r in untracked:
        if not free_ids:
            break
        obj_id = free_ids.pop(0)
        x1, y1, x2, y2 = r["bbox_xyxy"]
        new_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        new_obj_ids.append(obj_id)
        r["track_id"] = obj_id
        newly_seeded.append((obj_id, r))

    if new_bboxes:
        tracker(source=image_rgb, bboxes=new_bboxes, obj_ids=new_obj_ids, update_memory=True)
        for obj_id, r in newly_seeded:
            tracked_meta[obj_id] = r

    return raw_regions


def consolidate_track_labels(stage1_frames: Dict[int, Dict]) -> None:
    """
    Mutates stage1_frames in place so every detection sharing the same
    track_id has ONE consistent class_name and sensitivity, instead of
    flip-flopping frame to frame (e.g. the same tracked object showing as
    "microwave" in one frame, "person" in the next, "traffic light" in a
    third — confusing and unusable for review).

    Canonical class_name = that track's single highest-confidence
    detection. Sensitivity = "sensitive wins": if the track was ever
    flagged sensitive in any frame, it stays sensitive everywhere it
    appears — consistent with this pipeline's existing bias throughout
    (missing real sensitivity is worse than over-flagging).

    Only call this after a genuine fresh detection pass — running it
    again on already-consolidated, possibly manually-corrected data would
    silently overwrite those manual corrections.
    """
    track_best = {}   # track_id -> (confidence, class_name)
    track_sensitive = set()
    for stage1 in stage1_frames.values():
        for r in stage1["regions"]:
            tid = r.get("track_id")
            if tid is None:
                continue
            if r["sensitivity"] == "sensitive":
                track_sensitive.add(tid)
            cur = track_best.get(tid)
            if cur is None or r["confidence"] > cur[0]:
                track_best[tid] = (r["confidence"], r["class_name"])

    for stage1 in stage1_frames.values():
        for r in stage1["regions"]:
            tid = r.get("track_id")
            if tid is None:
                continue
            if tid in track_best:
                r["class_name"] = track_best[tid][1]
            if tid in track_sensitive:
                r["sensitivity"] = "sensitive"