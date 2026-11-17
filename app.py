"""
app.py  –  Semi-Automatic Sensitive Data Annotation Tool
Flow: Upload → Run Detection → File List → Editor → Back to List
"""

import base64
import hashlib
import html
import io
import logging
import tempfile
import zipfile
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from drawing_canvas import drawing_canvas
from manual_drawing import make_manual_region, classify_manual_region
from annotation_colors import (
    RED as REGION_RED, GREEN as REGION_GREEN, HIGHLIGHT,
    UNKNOWN_SENSITIVE, UNKNOWN_NON_SENSITIVE, color_for, color_hex,
)

import config
import models
import tracking
from pipeline import get_pipeline, filter_stage2_regions
from clip_prompts import ALL_PROMPTS
from utils import draw_overlay, downsize_gray, frame_changed, frame_has_local_change
from coco_export import export_to_json_string


def get_ruleset_fingerprint() -> str:
   
    content = (
        "|".join(sorted(config.SENSITIVE_CLASSES))
        + "||" + "|".join(sorted(config.NON_SENSITIVE_CLASSES))
        + "||" + "|".join(ALL_PROMPTS)
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

CACHE_DIR = Path(tempfile.gettempdir()) / "annotation_tool_cache"
CACHE_DIR.mkdir(exist_ok=True)


def media_cache_token(path: str) -> tuple:
    """Return a cheap version token so overwritten media cannot reuse stale frames."""
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size

st.set_page_config(
    page_title="SensAnnotate",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── session state ────────────────────────────────────────────────────
for key, default in [
    ("files", {}),
    ("annotations", {}),
    ("active_file", None),
    ("dark_mode", True),
    ("detection_done", False),
    ("view", "upload"),  
    ("yolo_model_key", models.DEFAULT_YOLO_MODEL_KEY),
    ("annotation_revisions", {}),
    ("video_tally_cache", {}),
    ("prepared_exports", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def mark_annotations_changed(filename: str) -> None:
    """Invalidate only state derived from annotations for one file."""
    revisions = st.session_state.annotation_revisions
    revisions[filename] = revisions.get(filename, 0) + 1
    st.session_state.video_tally_cache.pop(filename, None)
    st.session_state.prepared_exports = None
    st.session_state.pop(f"preview_path_{filename}", None)

# ── theme ────────────────────────────────────────────────────────────
dark = st.session_state.dark_mode
BG      = "#0f1117" if dark else "#f8f9fa"
SURFACE = "#1e2130" if dark else "#ffffff"
BORDER  = "#2d3250" if dark else "#dee2e6"
TEXT    = "#e8eaf0" if dark else "#1a1a2e"
SUBTEXT = "#8892a4" if dark else "#6c757d"
ACCENT  = "#4f8ef7"
RED     = color_hex(REGION_RED)
GREEN   = color_hex(REGION_GREEN)
ORANGE  = color_hex(UNKNOWN_SENSITIVE)
TEAL    = color_hex(UNKNOWN_NON_SENSITIVE)
SELECTED = color_hex(HIGHLIGHT)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css');

  html, body, [data-testid="stAppViewContainer"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif;
  }}
  [data-testid="stSidebar"] {{ display: none; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding: 0 2.5rem 2rem 2.5rem !important; max-width: 100% !important; }}

  .app-logo {{
    font-size: 1.25rem; font-weight: 700; color: {TEXT};
    display: flex; align-items: center; gap: 0.5rem;
  }}
  .app-logo span {{ color: {ACCENT}; }}
  .device-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: {SUBTEXT};
    background: {SURFACE}; border: 1px solid {BORDER};
    padding: 0.25rem 0.65rem; border-radius: 6px;
  }}

  /* Upload card (native st.container(border=True)) */
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.upload-dropzone) > div {{
    background: {SURFACE} !important; border-radius: 16px !important;
  }}
  .upload-dropzone {{
    border: 1.5px dashed {BORDER}; border-radius: 12px;
    padding: 2.5rem 1rem; text-align: center;
  }}
  .upload-icon {{ font-size: 1.75rem; color: {SUBTEXT}; margin-bottom: 0.5rem; display: block; }}
  .upload-title {{ font-size: 0.95rem; font-weight: 500; color: {TEXT}; margin-bottom: 0.25rem; }}
  .upload-sub {{ font-size: 0.8rem; color: {SUBTEXT}; }}
  .upload-chips {{
    display: flex; align-items: center; gap: 0.5rem; margin-top: 0.75rem;
  }}
  .upload-chip {{
    font-size: 0.78rem; color: {TEXT}; background: {BG};
    border: 1px solid {BORDER}; border-radius: 8px;
    padding: 0.3rem 0.65rem; display: inline-flex; align-items: center; gap: 0.35rem;
  }}

  /* File list */
  .file-list-container {{
    max-width: 700px; margin: 2rem auto;
  }}
  .file-list-title {{
    font-size: 1rem; font-weight: 600; color: {TEXT};
    margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .file-row {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.75rem;
    cursor: pointer; transition: border-color 0.15s;
  }}
  .file-row:hover {{ border-color: {ACCENT}; }}
  .file-icon {{ font-size: 1.1rem; }}
  .file-name {{ font-size: 0.9rem; font-weight: 500; color: {TEXT}; flex: 1; }}
  .file-badge-s {{ color: {RED}; font-size: 0.78rem; font-weight: 600; }}
  .file-badge-c {{ color: {GREEN}; font-size: 0.78rem; font-weight: 600; }}

  .dot {{ width:6px; height:6px; border-radius:50%; display:inline-block; margin-right:4px; }}
  .dot-red {{ background:{RED}; }}
  .dot-green {{ background:{GREEN}; }}
  .dot-orange {{ background:{ORANGE}; }}
  .dot-teal {{ background:{TEAL}; }}
  .dot-gray {{ background:{SUBTEXT}; }}
  .row-total {{ font-size:0.85rem; font-weight:600; color:{TEXT}; }}
  .row-name {{ font-size:0.85rem; color:{TEXT}; }}
  .legend-text {{ font-size:0.72rem; color:{SUBTEXT}; }}

  /* Editor */
  .editor-header {{
    font-size: 0.78rem; font-weight: 600; color: {SUBTEXT};
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
  }}
  .region-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 0.7rem 0.85rem;
    margin: 0.2rem 0 0.35rem 0; position: relative;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }}
  .region-card:hover {{ border-color: {ACCENT}; transform: translateY(-1px); }}
  .region-name {{ font-size: 0.85rem; font-weight: 600; color: {TEXT}; }}
  .region-conf {{ font-size: 0.72rem; color: {SUBTEXT}; font-family: 'JetBrains Mono', monospace; }}
  .region-sensitive {{ color: {RED}; font-size: 0.75rem; font-weight: 500; }}
  .region-nonsensitive {{ color: {GREEN}; font-size: 0.75rem; font-weight: 500; }}
  .region-unknown-sensitive {{ color: {ORANGE}; font-size: 0.75rem; font-weight: 600; }}
  .region-unknown-nonsensitive {{ color: {TEAL}; font-size: 0.75rem; font-weight: 600; }}
  .region-list {{ max-height: 68vh; overflow-y: auto; padding-right: 0.35rem; }}
  .method-chip {{
    display:inline-block; margin-top:0.35rem; padding:0.18rem 0.45rem;
    border-radius:999px; background:{BG}; color:{SUBTEXT};
    border:1px solid {BORDER}; font-size:0.68rem;
  }}
  .canvas-help {{ color:{SUBTEXT}; font-size:0.78rem; margin:0.2rem 0 0.7rem 0; }}
  .legend-row {{ display:flex; flex-wrap:wrap; gap:1rem; align-items:center; }}
  .hover-canvas svg {{ width:100%; height:auto; display:block; border-radius:12px; }}
  .hover-region {{ cursor:help; transition:fill-opacity .15s ease, stroke-width .15s ease; }}
  .hover-region:hover {{ fill-opacity:.52 !important; stroke-width:3.5 !important; }}

  /* Pin only the editor canvas while the adjacent inspector scrolls.
     Match both current and older Streamlit column markup. */
  @media (min-width: 769px) {{
    [data-testid="stColumn"]:has(.annotation-canvas-anchor),
    [data-testid="column"]:has(.annotation-canvas-anchor) {{
      position: sticky;
      top: 4rem;
      align-self: flex-start;
      max-height: calc(100vh - 5rem);
      overflow-y: auto;
      background: {BG};
    }}
    .hover-canvas svg {{ max-height: 65vh; }}
  }}

  div[data-testid="stVerticalBlockBorderWrapper"]:has(.img-frame-marker) {{
    position: relative;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.img-frame-marker) > div {{
    border: none !important; padding: 0 !important;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.img-frame-marker) div[data-testid="stButton"] {{
    position: absolute; top: 10px; right: 10px; z-index: 10;
    opacity: 0; transition: opacity 0.15s;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.img-frame-marker):hover div[data-testid="stButton"] {{
    opacity: 1;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.img-frame-marker) div[data-testid="stButton"] > button {{
    background: rgba(0,0,0,0.6) !important; color: white !important;
    border-radius: 50% !important; width: 32px !important; height: 32px !important;
    padding: 0 !important; border: none !important; font-size: 1rem !important;
  }}

  /* Buttons */
  .stButton > button {{
    background: {ACCENT} !important; color: white !important;
    border: 1px solid transparent !important; border-radius: 9px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    min-height: 2.45rem; padding: 0.48rem 0.9rem !important;
  }}
  .stButton > button:hover {{ filter:brightness(1.08); transform:translateY(-1px); }}
  .stDownloadButton > button {{
    background: {SURFACE} !important; color: {TEXT} !important;
    border: 1px solid {BORDER} !important; border-radius: 10px !important;
    font-weight: 600 !important;
  }}
  .stSelectbox > div > div {{
    background: {SURFACE} !important; border-color: {BORDER} !important;
    color: {TEXT} !important; border-radius: 8px !important;
    font-size: 0.83rem !important;
  }}
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────
# Load (and cache, via Streamlit) the AnnotationPipeline for a given YOLO model key.
@st.cache_resource(show_spinner="Loading detection models...")
def load_pipeline(yolo_model_key: str):
    return get_pipeline(yolo_model_key)


def current_pipeline():
    """Always resolves to whatever model is currently selected in the UI —
    call sites should use this instead of load_pipeline() directly, so a
    model switch is picked up without needing to hunt down every call site."""
    return load_pipeline(st.session_state.yolo_model_key)


def register_image(filename, filepath):
    img = cv2.imread(str(filepath))
    if img is None:
        return
    h, w = img.shape[:2]
    st.session_state.files[filename] = {
        "type": "image", "path": str(filepath),
        "width": w, "height": h, "frame_count": 1,
        "cache_token": media_cache_token(str(filepath)),
    }
    st.session_state.annotations[filename] = {
        "type": "image", "width": w, "height": h, "frames": {}
    }
    st.session_state.detection_done = False
    mark_annotations_changed(filename)


def register_video(filename, filepath):
    cap = cv2.VideoCapture(str(filepath))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    st.session_state.files[filename] = {
        "type": "video", "path": str(filepath),
        "width": w, "height": h, "frame_count": fc, "fps": fps,
        "cache_token": media_cache_token(str(filepath)),
    }
    st.session_state.annotations[filename] = {
        "type": "video", "width": w, "height": h,
        # stage1_frames: one entry per raw frame (tracker needs continuity)
        # stage2_keyframes: sparse — only recomputed on interval or scene change
        # manual_frames: human-created regions attached to one exact frame
        "stage1_frames": {}, "stage2_keyframes": {}, "manual_frames": {},
    }
    st.session_state.detection_done = False
    mark_annotations_changed(filename)

# Save an uploaded file into the cache dir and register each as image/video.
def ingest_uploaded_file(uploaded_file) -> bool:
    """Returns True if a NEW file was actually registered this call. The
    caller uses this to decide whether a rerun is needed — st.file_uploader
    keeps returning the same files on every rerun until the user manually
    removes them, so reruning unconditionally whenever files are present
    causes an infinite rerun loop."""
    name = uploaded_file.name
    ext = Path(name).suffix.lower()
    dest = CACHE_DIR / name
    registered_new = False
    if ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
            for member in zf.namelist():
                me = Path(member).suffix.lower()
                if me not in IMAGE_EXTS | VIDEO_EXTS:
                    continue
                op = CACHE_DIR / Path(member).name
                with zf.open(member) as src, open(op, "wb") as dst:
                    dst.write(src.read())
                if op.name not in st.session_state.files:
                    (register_image if me in IMAGE_EXTS else register_video)(op.name, op)
                    registered_new = True
        return registered_new
    if name in st.session_state.files:
        return False
    with open(dest, "wb") as f:
        f.write(uploaded_file.getvalue())
    (register_image if ext in IMAGE_EXTS else register_video)(name, dest)
    return True


@st.cache_data(show_spinner=False)
def read_video_frame(video_path, frame_idx, cache_token=None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def read_image(image_path, cache_token=None):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the RGB pixel array for a file, either the whole image or the requested video frame.
def get_frame_array(filename, frame_idx):
    meta = st.session_state.files[filename]
    cache_token = meta.get("cache_token") or media_cache_token(meta["path"])
    if meta["type"] == "image":
        return read_image(meta["path"], cache_token)
    return read_video_frame(meta["path"], frame_idx, cache_token)


# Stage 2 (SAM2 background + CLIP) is expensive and mostly redundant frame
STAGE2_CALL_BUDGET = 50
STAGE2_MIN_INTERVAL = 30  

# Compute the stage-2 (SAM2+CLIP) recompute interval, scaled so long videos stay within a roughly constant call budget.
def compute_stage2_interval(frame_count: int) -> int:
    return max(STAGE2_MIN_INTERVAL, frame_count // STAGE2_CALL_BUDGET)


STAGE2_CHANGE_THRESHOLD = 12.0  # mean abs pixel diff (0-255) that counts as "scene changed"


STAGE1_FRAME_BUDGET = 1800

# Stage-1 duplicate-frame skip: if a frame shows no meaningful LOCAL
STAGE1_DUPLICATE_THRESHOLD = 15.0  # max abs pixel diff (0-255) on the downsized frame


STAGE1_MAX_CONSECUTIVE_SKIPS = 10


# Compute the stage-1 recompute interval, scaled so long videos stay within a roughly constant inference call budget.
def compute_stage1_interval(frame_count: int) -> int:
    if frame_count <= STAGE1_FRAME_BUDGET:
        return 1
    return max(1, frame_count // STAGE1_FRAME_BUDGET)

# Find the nearest processed frame/keyframe at or before a given frame index in a sparse results dict.
def latest_key_at_or_before(d: dict, frame_idx: int):
    keys = sorted(d)
    position = bisect_right(keys, frame_idx) - 1
    return keys[position] if position >= 0 else None

# Combine the nearest stage-1 detections with the nearest stage-2 keyframe regions for a given frame, via cheap lookups only.
def get_combined_regions(filename, frame_idx): 
    meta = st.session_state.files[filename]
    ann = st.session_state.annotations[filename]
    if meta["type"] == "image":
        return ann["frames"].get(frame_idx, [])

    s1_key = latest_key_at_or_before(ann["stage1_frames"], frame_idx)
    if s1_key is None:
        return ann.get("manual_frames", {}).get(frame_idx, [])
    stage1 = ann["stage1_frames"][s1_key]
    kf = latest_key_at_or_before(ann["stage2_keyframes"], frame_idx)
    raw_stage2 = ann["stage2_keyframes"][kf]["raw_regions"] if kf is not None else []
    manual = ann.get("manual_frames", {}).get(frame_idx, [])
    return stage1["regions"] + filter_stage2_regions(raw_stage2, stage1["boxes"]) + manual


def iter_combined_video_frames(ann):
    """Yield processed video frames using one linear stage-2 keyframe walk."""
    stage2_keys = sorted(ann.get("stage2_keyframes", {}))
    stage2_position = -1
    for frame_idx in sorted(ann.get("stage1_frames", {})):
        while (
            stage2_position + 1 < len(stage2_keys)
            and stage2_keys[stage2_position + 1] <= frame_idx
        ):
            stage2_position += 1

        stage1 = ann["stage1_frames"][frame_idx]
        raw_stage2 = (
            ann["stage2_keyframes"][stage2_keys[stage2_position]]["raw_regions"]
            if stage2_position >= 0
            else []
        )
        regions = stage1["regions"] + filter_stage2_regions(raw_stage2, stage1["boxes"])
        regions += ann.get("manual_frames", {}).get(frame_idx, [])
        yield frame_idx, stage1, regions


def update_region_sensitivity(filename, frame_idx, region_idx, widget_key):
    """Apply a selectbox value before Streamlit performs its normal rerun."""
    regions = get_combined_regions(filename, frame_idx)
    new_value = st.session_state.get(widget_key)
    if region_idx < len(regions) and new_value in {"sensitive", "non_sensitive"}:
        if regions[region_idx]["sensitivity"] != new_value:
            regions[region_idx]["sensitivity"] = new_value
            mark_annotations_changed(filename)


def rename_region(filename, frame_idx, region_idx, widget_key):
    """Apply a region name edit before the normal widget rerun."""
    regions = get_combined_regions(filename, frame_idx)
    new_name = st.session_state.get(widget_key, "").strip()
    if region_idx < len(regions) and new_name and regions[region_idx]["class_name"] != new_name:
        regions[region_idx]["class_name"] = new_name
        mark_annotations_changed(filename)


def toggle_highlight(highlight_key, region_idx):
    current = st.session_state.get(highlight_key)
    st.session_state[highlight_key] = None if current == region_idx else region_idx


def toggle_region_sensitivity(filename: str, frame_idx: int, region_idx: int) -> None:
    regions = get_combined_regions(filename, frame_idx)
    if region_idx >= len(regions):
        return
    region = regions[region_idx]
    region["sensitivity"] = (
        "non_sensitive" if region.get("sensitivity") == "sensitive" else "sensitive"
    )
    region["edited_by_human"] = True
    region["sensitivity_source"] = "human"
    region["needs_review"] = False
    mark_annotations_changed(filename)


def region_provenance(region: dict) -> tuple[str, str, str]:
    """Return concise, user-facing model and method descriptions."""
    source = region.get("source", "unknown")
    if source == "sam":
        model = "SAM2 + CLIP"
        method = "SAM2 automatic mask; CLIP sensitive/non-sensitive comparison"
        score_name = "CLIP sensitive-class score"
    elif source == "manual":
        model = "Human outline + CLIP" if region.get("clip_suggested_sensitivity") else "Human annotation"
        method = region.get("segmentation_method", "Manual rectangular region")
        if region.get("clip_suggested_sensitivity"):
            method += "; CLIP suggestion from the bounding-box crop"
        score_name = "CLIP sensitive-class score" if region.get("clip_suggested_sensitivity") else "No model score"
    elif source == "finetuned_sh17":
        model = "Fine-tuned YOLO (SH17) + SAM2"
        method = "YOLO object detection; SAM2 box-prompted mask; rule-based sensitivity"
        score_name = "YOLO detection score"
    elif source == "yolo26l_general":
        model = "General YOLO26 + SAM2"
        method = "YOLO object detection; SAM2 box-prompted mask; rule-based sensitivity"
        score_name = "YOLO detection score"
    else:
        model = source.replace("_", " ").title() if source else "Unknown source"
        method = region.get("segmentation_method", "Legacy or unspecified annotation method")
        score_name = "Model score"

    if region.get("edited_by_human"):
        method += "; geometry or metadata adjusted by a reviewer"
    return model, method, score_name


def confidence_text(region: dict) -> str:
    confidence = region.get("confidence")
    if confidence is None:
        return "No model score"
    _model, _method, score_name = region_provenance(region)
    return f"{score_name}: {float(confidence):.2f}"


def region_color_hex(region: dict) -> str:
    return color_hex(color_for(region.get("sensitivity"), region.get("class_name")))


def build_hover_canvas(
    image_rgb: np.ndarray,
    regions: list[dict],
    highlighted_idx: int | None = None,
) -> str:
    """Render an SVG overlay whose regions expose provenance on hover."""
    height, width = image_rgb.shape[:2]
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=88)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    shapes = []
    for idx, region in enumerate(regions):
        model, method, _score_name = region_provenance(region)
        tooltip = html.escape(
            f"Object: {region.get('class_name', 'unknown')}\n"
            f"Sensitivity: {region.get('sensitivity', 'unknown')}\n"
            f"Model: {model}\nMethod: {method}\n{confidence_text(region)}"
        )
        color = region_color_hex(region)
        stroke = SELECTED if idx == highlighted_idx else color
        stroke_width = 5 if idx == highlighted_idx else 2
        for polygon in region.get("polygons", []):
            if len(polygon) < 6:
                continue
            points = " ".join(
                f"{float(polygon[pos]):.1f},{float(polygon[pos + 1]):.1f}"
                for pos in range(0, len(polygon) - 1, 2)
            )
            shapes.append(
                f'<polygon class="hover-region" points="{points}" '
                f'fill="{color}" fill-opacity="0.28" stroke="{stroke}" '
                f'stroke-width="{stroke_width}"><title>{tooltip}</title></polygon>'
            )

    return (
        f'<div class="hover-canvas"><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Annotation review canvas">'
        f'<image href="data:image/jpeg;base64,{image_b64}" width="{width}" height="{height}"/>'
        + "".join(shapes)
        + "</svg></div>"
    )


def add_manual_region(filename: str, frame_idx: int, drawing: dict) -> dict:
    """Validate and store a drawn outline on one image or exact video frame."""
    meta = st.session_state.files[filename]
    region = make_manual_region(drawing, meta["width"], meta["height"])
    if drawing["sensitivity"] == "auto":
        try:
            with st.spinner("CLIP is suggesting sensitivity for the new region..."):
                frame_rgb = get_frame_array(filename, frame_idx)
                classify_manual_region(region, frame_rgb, current_pipeline().classify_with_clip)
        except Exception as exc:
            logging.getLogger(__name__).exception("CLIP classification failed for a manual drawing")
            raise ValueError(
                "CLIP could not classify this drawing. It was not saved. "
                "Retry Save or choose Sensitive / Non-sensitive manually."
            ) from exc
    ann = st.session_state.annotations[filename]
    if meta["type"] == "image":
        ann.setdefault("frames", {}).setdefault(frame_idx, []).append(region)
    else:
        ann.setdefault("manual_frames", {}).setdefault(frame_idx, []).append(region)
    mark_annotations_changed(filename)
    return region


def find_region_storage(
    filename: str, frame_idx: int, target: dict
) -> tuple[list | None, dict | None]:
    """Locate the mutable list holding a displayed region by object identity."""
    meta = st.session_state.files[filename]
    ann = st.session_state.annotations[filename]
    if meta["type"] == "image":
        candidates = [(ann.get("frames", {}).get(frame_idx, []), None)]
    else:
        candidates = []
        stage1_key = latest_key_at_or_before(ann.get("stage1_frames", {}), frame_idx)
        if stage1_key is not None:
            stage1 = ann["stage1_frames"][stage1_key]
            candidates.append((stage1["regions"], stage1))
        stage2_key = latest_key_at_or_before(ann.get("stage2_keyframes", {}), frame_idx)
        if stage2_key is not None:
            candidates.append((ann["stage2_keyframes"][stage2_key]["raw_regions"], None))
        candidates.append((ann.get("manual_frames", {}).get(frame_idx, []), None))

    for storage, stage1 in candidates:
        for stored_region in storage:
            if stored_region is target:
                return storage, stage1
    return None, None


def remove_selected_region(
    filename: str, frame_idx: int, region_idx: int, highlight_key: str
) -> None:
    regions = get_combined_regions(filename, frame_idx)
    if region_idx >= len(regions):
        return
    target = regions[region_idx]
    storage, stage1 = find_region_storage(filename, frame_idx, target)
    if storage is None:
        return

    old_bbox = target.get("bbox_xyxy")
    storage.remove(target)
    if stage1 is not None and old_bbox in stage1.get("boxes", []):
        stage1["boxes"].remove(old_bbox)
    st.session_state[highlight_key] = None
    mark_annotations_changed(filename)


def apply_region_editor(
    filename: str, frame_idx: int, region_idx: int, key_prefix: str
) -> None:
    regions = get_combined_regions(filename, frame_idx)
    if region_idx >= len(regions):
        return
    region = regions[region_idx]
    class_name = st.session_state.get(f"{key_prefix}_name", "").strip() or "unknown"
    sensitivity = st.session_state.get(f"{key_prefix}_sensitivity", "sensitive")
    replace_geometry = st.session_state.get(f"{key_prefix}_replace_geometry", False)

    region["class_name"] = class_name
    region["sensitivity"] = sensitivity
    region["edited_by_human"] = True
    region["sensitivity_source"] = "human"
    region["needs_review"] = False

    if replace_geometry:
        meta = st.session_state.files[filename]
        x1 = max(0, min(int(st.session_state[f"{key_prefix}_x1"]), meta["width"] - 1))
        y1 = max(0, min(int(st.session_state[f"{key_prefix}_y1"]), meta["height"] - 1))
        x2 = max(1, min(int(st.session_state[f"{key_prefix}_x2"]), meta["width"]))
        y2 = max(1, min(int(st.session_state[f"{key_prefix}_y2"]), meta["height"]))
        if x2 <= x1 or y2 <= y1:
            st.session_state.editor_notice = "Geometry was not changed: right/bottom must exceed left/top."
        else:
            old_bbox = region.get("bbox_xyxy")
            _storage, stage1 = find_region_storage(filename, frame_idx, region)
            new_bbox = [x1, y1, x2, y2]
            region["bbox_xyxy"] = new_bbox
            region["polygons"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            region["segmentation_method"] = "Human-adjusted rectangular mask"
            if stage1 is not None and old_bbox in stage1.get("boxes", []):
                box_index = stage1["boxes"].index(old_bbox)
                stage1["boxes"][box_index] = new_bbox
            st.session_state.editor_notice = "Object and geometry updated."
    else:
        st.session_state.editor_notice = "Object label and sensitivity updated."
    mark_annotations_changed(filename)

# Render a full mp4 preview of a video with detection overlays burned into every frame, reusing already-computed detections.
def render_annotated_video(filename):
   
    meta = st.session_state.files[filename]
    cap = cv2.VideoCapture(meta["path"])
    fps = meta.get("fps", 25.0) or 25.0
    w, h = meta["width"], meta["height"]

    out_path = Path(tempfile.gettempdir()) / f"preview_{Path(filename).stem}.mp4"
    writer = None
    for codec in ["avc1", "H264", "mp4v"]:
        candidate = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (w, h)
        )
        if candidate.isOpened():
            writer = candidate
            break
    if writer is None:
        cap.release()
        return None

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        regions = get_combined_regions(filename, frame_idx)
        overlay_rgb = draw_overlay(frame_rgb, regions, h, w)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            overlay_bgr,
            f"frame {frame_idx} | red/green=known sensitive/non-sensitive; orange/teal=unknown sensitive/non-sensitive",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(overlay_bgr)
        frame_idx += 1

    cap.release()
    writer.release()
    return str(out_path)


def ensure_annotated(filename, frame_idx):
    """Normally a no-op after Run Detection has processed the whole video
    up front (see run_detection_all). This is a fallback for scrubbing to a
    frame that hasn't been processed yet — it runs stage 1 WITHOUT the
    tracker (since it's an isolated call, not part of a sequential walk),
    so a face/person caught only via this path won't have proper track-ID
    continuity with neighboring frames."""
    meta = st.session_state.files[filename]
    ann = st.session_state.annotations[filename]

    if meta["type"] == "image":
        if frame_idx in ann["frames"]:
            return
        frame_rgb = get_frame_array(filename, frame_idx)
        pipeline = current_pipeline()
        with st.spinner(f"Detecting {filename}..."):
            ann["frames"][frame_idx] = pipeline.run_on_image(frame_rgb)
        mark_annotations_changed(filename)
        return

    needs_stage1 = latest_key_at_or_before(ann["stage1_frames"], frame_idx) is None
    needs_stage2 = latest_key_at_or_before(ann["stage2_keyframes"], frame_idx) is None
    if not needs_stage1 and not needs_stage2:
        return

    frame_rgb = get_frame_array(filename, frame_idx)
    pipeline = current_pipeline()
    if needs_stage1:
        with st.spinner(f"Detecting frame {frame_idx}..."):
            regions, boxes = pipeline.run_stage1(frame_rgb, use_tracker=False)
        ann["stage1_frames"][frame_idx] = {"regions": regions, "boxes": boxes}

    if needs_stage2:
        with st.spinner(f"Detecting background at frame {frame_idx}..."):
            # Ad-hoc single-frame fallback — no per-video tracker exists
            # here (that only lives inside one run_detection_all call), so
            # this uses the plain untracked method. Regions from this path
            # simply won't be deduped against other keyframes' regions.
            raw = pipeline.run_stage2_raw(frame_rgb)
        for r in raw:
            r.setdefault("track_id", None)
        ann["stage2_keyframes"][frame_idx] = {"raw_regions": raw}
    mark_annotations_changed(filename)

# Main detection entry point: walk every uploaded file's frames in order, running stage-1/stage-2 detection and caching results.
def run_detection_all():
    st.session_state.prepared_exports = None
    files = list(st.session_state.files.keys())
    bar = st.progress(0, text="Running detection on all files...")
    pipeline = current_pipeline()
    current_model_key = st.session_state.yolo_model_key
    current_ruleset = get_ruleset_fingerprint()

    for i, fname in enumerate(files):
        meta = st.session_state.files[fname]
        ann = st.session_state.annotations[fname]

        if ann.get("_yolo_model_key") != current_model_key or ann.get("_ruleset_fingerprint") != current_ruleset:
            ann["stage1_frames"] = {}
            ann["stage2_keyframes"] = {}
            ann["frames"] = {}
            ann["_yolo_model_key"] = current_model_key
            ann["_ruleset_fingerprint"] = current_ruleset
            ann["_fully_processed"] = False

        # Skip this file entirely if it's already fully processed under
        # the current model/ruleset — avoids re-walking every frame of an
        # already-done video just to confirm there's nothing new to do.
        if ann.get("_fully_processed"):
            bar.progress((i + 1) / len(files), text=f"Skipping already-processed {fname}")
            continue

        if meta["type"] == "video":
            stage1_interval = compute_stage1_interval(meta["frame_count"])
            stage2_interval = compute_stage2_interval(meta["frame_count"])
            stage1_frame_indices = list(range(0, meta["frame_count"], stage1_interval))
            last_stage2_gray = None
            last_stage1_gray = None
            consecutive_stage1_skips = 0
            any_new_stage1 = False

            # Fresh tracker for THIS video only — see tracking.new_stage2_tracker's
            # docstring for why this must not be cached/reused across videos.
            stage2_tracker = tracking.new_stage2_tracker(config.SAM_WEIGHTS, config.STAGE2_TRACK_CAP)
            stage2_tracked_meta = {}
            stage2_next_obj_id = [0]

            # Diagnostics only — don't affect detection, just tell us
            # afterward WHY the object count is what it is: is the tracker
            # actually assigning IDs, and how often is stage 2 recomputing.
            diag_tracked = 0
            diag_untracked = 0
            diag_stage2_recomputes = 0
            diag_stage1_skipped_duplicates = 0

            for step, fidx in enumerate(stage1_frame_indices):
                frame_rgb = get_frame_array(fname, fidx)
                small_gray = downsize_gray(frame_rgb)

                is_duplicate = (
                    fidx != 0
                    and consecutive_stage1_skips < STAGE1_MAX_CONSECUTIVE_SKIPS
                    and not frame_has_local_change(last_stage1_gray, small_gray, STAGE1_DUPLICATE_THRESHOLD)
                )

                if fidx not in ann["stage1_frames"] and not is_duplicate:
                    regions, boxes = pipeline.run_stage1(frame_rgb, use_tracker=True)
                    ann["stage1_frames"][fidx] = {"regions": regions, "boxes": boxes}
                    last_stage1_gray = small_gray
                    consecutive_stage1_skips = 0
                    any_new_stage1 = True
                    for r in regions:
                        if r.get("track_id") is not None:
                            diag_tracked += 1
                        else:
                            diag_untracked += 1
                elif is_duplicate:
                    # Not stored — get_combined_regions already falls back
                    # to the nearest earlier processed frame via
                    # latest_key_at_or_before, so this frame automatically
                    # reuses the last real detection with zero extra code.
                    consecutive_stage1_skips += 1
                    diag_stage1_skipped_duplicates += 1

                need_stage2 = (
                    fidx == 0
                    or fidx % stage2_interval == 0
                    or frame_changed(last_stage2_gray, small_gray, STAGE2_CHANGE_THRESHOLD)
                )
                if need_stage2 and fidx not in ann["stage2_keyframes"]:
                    raw = pipeline.run_stage2_with_tracking(
                        frame_rgb, stage2_tracker, stage2_tracked_meta, stage2_next_obj_id
                    )
                    ann["stage2_keyframes"][fidx] = {"raw_regions": raw}
                    last_stage2_gray = small_gray
                    diag_stage2_recomputes += 1
                elif fidx in ann["stage2_keyframes"]:
                    last_stage2_gray = small_gray

                interval_desc = "every frame" if stage1_interval == 1 else f"every {stage1_interval} frames"
                bar.progress(
                    min((i + (step + 1) / len(stage1_frame_indices)) / len(files), 1.0),
                    text=f"Processing {fname}: frame {fidx + 1}/{meta['frame_count']} ({interval_desc})",
                )

            if any_new_stage1:
                tracking.consolidate_track_labels(ann["stage1_frames"])

            st.session_state.setdefault("detection_diagnostics", {})[fname] = {
                "tracked_detections": diag_tracked,
                "stage1_skipped_duplicates": diag_stage1_skipped_duplicates,
                "untracked_detections": diag_untracked,
                "stage2_recomputes": diag_stage2_recomputes,
                "stage1_frames_processed": len(stage1_frame_indices),
            }
        else:
            if 0 not in ann.get("frames", {}):
                frame_rgb = get_frame_array(fname, 0)
                ann["frames"][0] = pipeline.run_on_image(frame_rgb)
            bar.progress((i + 1) / len(files), text=f"Processing {i + 1}/{len(files)}: {fname}")

        ann["_fully_processed"] = True
        mark_annotations_changed(fname)

    bar.empty()
    st.session_state.detection_done = True

# For video: dedupe stage-1 and stage-2 regions by track_id so each real object is counted once, not once per frame.
def _tally_video_regions(fname):
    revision = st.session_state.annotation_revisions.get(fname, 0)
    cached = st.session_state.video_tally_cache.get(fname)
    if cached is not None and cached["revision"] == revision:
        return cached["result"]

    ann = st.session_state.annotations[fname]
    seen_stage1_tracks = {}  # track_id -> latest sensitivity seen for it
    seen_stage2_tracks = {}
    untracked_total = untracked_sensitive = 0
    stage2_untracked_total = stage2_untracked_sensitive = 0

    for _fidx, stage1, combined_regions in iter_combined_video_frames(ann):
        for r in stage1["regions"]:
            tid = r.get("track_id")
            if tid is not None:
                seen_stage1_tracks[tid] = r["sensitivity"]
            else:
                untracked_total += 1
                if r["sensitivity"] == "sensitive":
                    untracked_sensitive += 1

        stage2_regions = combined_regions[len(stage1["regions"]):]
        for r in stage2_regions:
            tid2 = r.get("track_id")
            if tid2 is not None:
                seen_stage2_tracks[tid2] = r["sensitivity"]
            else:
                stage2_untracked_total += 1
                if r["sensitivity"] == "sensitive":
                    stage2_untracked_sensitive += 1

    # A reviewer may annotate a frame that was not one of the sparsely
    # processed stage-1 frames. Count those exact-frame manual regions too;
    # manual annotations on processed frames were already included above.
    stage1_frame_indices = set(ann.get("stage1_frames", {}))
    for manual_frame_idx, manual_regions in ann.get("manual_frames", {}).items():
        if manual_frame_idx in stage1_frame_indices:
            continue
        for region in manual_regions:
            stage2_untracked_total += 1
            if region["sensitivity"] == "sensitive":
                stage2_untracked_sensitive += 1

    tracked_total = len(seen_stage1_tracks)
    tracked_sensitive = sum(1 for s in seen_stage1_tracks.values() if s == "sensitive")
    stage2_tracked_total = len(seen_stage2_tracks)
    stage2_tracked_sensitive = sum(1 for s in seen_stage2_tracks.values() if s == "sensitive")

    result = {
        "tracked_total": tracked_total,
        "tracked_sensitive": tracked_sensitive,
        "untracked_total": untracked_total,
        "untracked_sensitive": untracked_sensitive,
        "stage2_total": stage2_tracked_total + stage2_untracked_total,
        "stage2_sensitive": stage2_tracked_sensitive + stage2_untracked_sensitive,
    }
    st.session_state.video_tally_cache[fname] = {
        "revision": revision,
        "result": result,
    }
    return result

# Get the total count of regions labeled "sensitive" for a file (deduped by track for video).
def get_sensitive_count(filename):
    meta = st.session_state.files.get(filename)
    if not meta:
        return 0
    if meta["type"] == "image":
        ann = st.session_state.annotations.get(filename, {})
        count = 0
        for fidx in ann.get("frames", {}):
            count += sum(1 for r in ann["frames"][fidx] if r["sensitivity"] == "sensitive")
        return count
    t = _tally_video_regions(filename)
    return t["tracked_sensitive"] + t["untracked_sensitive"] + t["stage2_sensitive"]

# Get (total, sensitive, non_sensitive) headline object counts for a file, using distinct tracked people/faces for video.
def get_object_counts(fname):
    
    meta = st.session_state.files[fname]
    if meta["type"] == "image":
        ann = st.session_state.annotations[fname]
        total = sensitive = 0
        for fidx in ann.get("frames", {}):
            for r in ann["frames"][fidx]:
                total += 1
                if r["sensitivity"] == "sensitive":
                    sensitive += 1
        return total, sensitive, total - sensitive

    t = _tally_video_regions(fname)
    total = t["tracked_total"] + t["untracked_total"] + t["stage2_total"]
    sensitive = (
        t["tracked_sensitive"]
        + t["untracked_sensitive"]
        + t["stage2_sensitive"]
    )
    return total, sensitive, total - sensitive

# Get a detailed breakdown separating people/face counts from raw background region counts.
def get_region_breakdown(fname):
    """Full detail for anyone who wants to see under the hood — distinct
    people/faces vs. raw background candidate regions, separately."""
    meta = st.session_state.files[fname]
    if meta["type"] == "image":
        total, sensitive, non_sensitive = get_object_counts(fname)
        return {"people_faces": total, "people_faces_sensitive": sensitive, "background_regions": 0, "background_sensitive": 0}
    t = _tally_video_regions(fname)
    return {
        "people_faces": t["tracked_total"] + t["untracked_total"],
        "people_faces_sensitive": t["tracked_sensitive"] + t["untracked_sensitive"],
        "background_regions": t["stage2_total"],
        "background_sensitive": t["stage2_sensitive"],
    }

@st.cache_data(show_spinner=False)
def get_thumbnail_b64(fname, path, ftype, size=64):
    if ftype == "image":
        img = cv2.imread(path)
    else:
        cap = cv2.VideoCapture(path)
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()

# Build a display label for a region from its class name and index.
def name_region(r, idx):
    return f"{r['class_name']}_{idx+1:02d}"


def scale_regions_for_display(regions, scale_x, scale_y):
    """Copy region metadata and scale its geometry to display resolution."""
    scaled = []
    for region in regions:
        polygons = []
        for polygon in region["polygons"]:
            polygons.append([
                value * (scale_x if index % 2 == 0 else scale_y)
                for index, value in enumerate(polygon)
            ])
        scaled_region = dict(region)
        scaled_region["polygons"] = polygons
        if "bbox_xyxy" in region:
            x1, y1, x2, y2 = region["bbox_xyxy"]
            scaled_region["bbox_xyxy"] = [
                x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y,
            ]
        scaled.append(scaled_region)
    return scaled


# ── HEADER ───────────────────────────────────────────────────────────
h1, h2 = st.columns([2, 1])
with h1:
    st.markdown(f'<div class="app-logo">🔍 <span>Sens</span>Annotate</div>', unsafe_allow_html=True)
with h2:
    st.markdown(f'<div class="device-badge">⚡ {config.DEVICE.upper()}</div>', unsafe_allow_html=True)

# Model picker + Run Detection live here, in the persistent header
if st.session_state.files:
    img_count = sum(1 for m in st.session_state.files.values() if m["type"] == "image")
    vid_count = sum(1 for m in st.session_state.files.values() if m["type"] == "video")

    model_options = list(models.YOLO_MODEL_OPTIONS.values())
    model_labels = [m.display_name for m in model_options]
    current_idx = next(
        (i for i, m in enumerate(model_options) if m.key == st.session_state.yolo_model_key), 0
    )
    chosen_label = st.selectbox(
        "Detection model", model_labels, index=current_idx, key="model_picker",
        help="Switch anytime without re-uploading files — already-detected results from a "
             "previous model stay as they are until you re-run detection.",
    )
    chosen_option = model_options[model_labels.index(chosen_label)]
    st.session_state.yolo_model_key = chosen_option.key
    st.caption(chosen_option.description)

    chip_col, btn_col, clear_col = st.columns([3, 1, 1])
    with chip_col:
        chips = f'<span class="upload-chip"><i class="ti ti-photo"></i>{img_count} image{"s" if img_count != 1 else ""}</span>'
        if vid_count:
            chips += f'<span class="upload-chip"><i class="ti ti-video"></i>{vid_count} video{"s" if vid_count != 1 else ""}</span>'
        st.markdown(f'<div class="upload-chips">{chips}</div>', unsafe_allow_html=True)
    with btn_col:
        if st.button("Run detection", key="run_all", use_container_width=True):
            run_detection_all()
            st.session_state.view = "list"
            st.rerun()
    with clear_col:
        if st.button("🧹 Clear cache", key="clear_cache", use_container_width=True,
                      help="Wipes ALL stored detections for every file, forcing a fully fresh "
                           "run next time. Use this if results look wrong after a code update — "
                           "the model/prompt-change detection only catches config.py and "
                           "clip_prompts.py edits, not fixes to app.py/pipeline.py/tracking.py "
                           "themselves."):
            for fname in st.session_state.files:
                ann = st.session_state.annotations[fname]
                ann["stage1_frames"] = {}
                ann["stage2_keyframes"] = {}
                ann["manual_frames"] = {}
                ann["frames"] = {}
                ann.pop("_yolo_model_key", None)
                ann.pop("_ruleset_fingerprint", None)
            st.session_state.detection_diagnostics = {}
            st.session_state.detection_done = False
            st.session_state.video_tally_cache = {}
            st.session_state.prepared_exports = None
            st.success("Cache cleared — click Run detection for a fully fresh run.")
            st.rerun()

def build_export_state(only_type=None):
    
    state = {}
    for fname, meta in st.session_state.files.items():
        if only_type is not None and meta["type"] != only_type:
            continue
        ann = st.session_state.annotations[fname]
        if meta["type"] == "image":
            frames = ann.get("frames", {})
        else:
            frames = {
                frame_idx: regions
                for frame_idx, _stage1, regions in iter_combined_video_frames(ann)
            }
            # Preserve corrections drawn on exact video frames even when the
            # automatic pipeline sampled stage 1 sparsely on a long video.
            for frame_idx in ann.get("manual_frames", {}):
                frames[frame_idx] = get_combined_regions(fname, frame_idx)
        state[fname] = {
            "type": meta["type"], "width": meta["width"], "height": meta["height"],
            "fps": meta.get("fps"), "frames": frames,
        }
    return state


def prepare_export_payloads():
    """Build all download payloads once, only when explicitly requested."""
    full_state = build_export_state()
    image_state = {
        name: data for name, data in full_state.items() if data["type"] == "image"
    }
    video_state = {
        name: data for name, data in full_state.items() if data["type"] == "video"
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, filedata in full_state.items():
            stem = Path(fname).stem
            folder = "images" if filedata["type"] == "image" else "videos"
            zf.writestr(
                f"{folder}/annotations_{stem}_sensannotate.json",
                export_to_json_string({fname: filedata}),
            )

    return {
        "images": export_to_json_string(image_state) if image_state else None,
        "videos": export_to_json_string(video_state) if video_state else None,
        "all": zip_buffer.getvalue() if full_state else None,
    }


# Export generation is intentionally lazy: Streamlit reruns this file after
# every edit, so eagerly rebuilding multi-megabyte JSON and ZIP payloads here
# makes a one-field annotation change unnecessarily expensive.
if st.session_state.detection_done:
    if st.button("Prepare / refresh COCO downloads", key="prepare_exports"):
        with st.spinner("Preparing COCO JSON downloads..."):
            st.session_state.prepared_exports = prepare_export_payloads()

    prepared_exports = st.session_state.prepared_exports
    if prepared_exports is None:
        st.caption("Prepare downloads after you finish reviewing annotations.")
    else:
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if prepared_exports["images"] is not None:
                st.download_button(
                    "⬇️ Export Images SensAnnotate JSON",
                    data=prepared_exports["images"],
                    file_name="annotations_images_sensannotate.json",
                    mime="application/json",
                    key="export_images",
                )
        with exp_col2:
            if prepared_exports["videos"] is not None:
                st.download_button(
                    "⬇️ Export Videos SensAnnotate JSON",
                    data=prepared_exports["videos"],
                    file_name="annotations_videos_sensannotate.json",
                    mime="application/json",
                    key="export_videos",
                )

        if prepared_exports["all"] is not None:
            st.download_button(
                "⬇️ Export All SensAnnotate JSON",
                data=prepared_exports["all"],
                file_name="annotations_per_file.zip",
                mime="application/zip",
                key="export_all_separate",
            )

st.markdown(f'<hr style="border-color:{BORDER};margin:0.5rem 0 1.5rem 0">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# VIEW: UPLOAD
# ══════════════════════════════════════════════════════════════════════
if st.session_state.view == "upload":
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.container(border=True):
            st.markdown("""
            <div class="upload-dropzone">
                <i class="ti ti-cloud-upload upload-icon"></i>
                <div class="upload-title">Drag images, videos, or a zip file here</div>
                <div class="upload-sub">Supports .jpg .png .mp4 .zip</div>
            </div>
            """, unsafe_allow_html=True)
 
            uploaded = st.file_uploader(
                "Choose files",
                type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "avi", "mkv", "zip"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded:
                any_new = False
                for f in uploaded:
                    if ingest_uploaded_file(f):
                        any_new = True
                if any_new:
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# VIEW: FILE LIST
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.view == "list":
    total_files = len(st.session_state.files)
    total_objects = sum(get_object_counts(f)[0] for f in st.session_state.files)

    hcol1, hcol2 = st.columns([3, 2])
    with st.expander("➕ Add more files"):
        more_uploaded = st.file_uploader(
            "Choose files",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "avi", "mkv", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="add_more_uploader",
        )
        if more_uploaded:
            any_new = False
            for f in more_uploaded:
                if ingest_uploaded_file(f):
                    any_new = True
            if any_new:
                st.rerun()
    with hcol1:
        st.markdown(f'<span style="font-size:1rem;font-weight:500">{total_files} files</span> '
                     f'<span style="font-size:0.85rem;color:{SUBTEXT}">&nbsp;&nbsp;{total_objects} objects across all files</span>',
                     unsafe_allow_html=True)
    with hcol2:
        search = st.text_input("Search", placeholder="Search filename", label_visibility="collapsed")

    st.markdown("---")

    hc1, hc2, hc3, hc4 = st.columns([4, 1, 2, 1])
    with hc1:
        st.markdown(f'<span style="font-size:0.78rem;color:{SUBTEXT}">File</span>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<span style="font-size:0.78rem;color:{SUBTEXT}">Objects</span>', unsafe_allow_html=True)
    with hc3:
        st.markdown(f'<span style="font-size:0.78rem;color:{SUBTEXT}">Sensitivity</span>', unsafe_allow_html=True)
    with hc4:
        st.markdown("", unsafe_allow_html=True)
    st.markdown(f'<hr style="border-color:{BORDER};margin:0.4rem 0 0.75rem 0">', unsafe_allow_html=True)

    for fname, meta in st.session_state.files.items():
        if search and search.lower() not in fname.lower():
            continue

        obj_total, s_count, ns_count = get_object_counts(fname)

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4, 1, 2, 1])
            with c1:
                thumb_b64 = get_thumbnail_b64(fname, meta["path"], meta["type"])
                thumb_html = (f'<img src="data:image/jpeg;base64,{thumb_b64}" '
                              f'style="width:32px;height:32px;object-fit:cover;border-radius:6px;">'
                              if thumb_b64 else '<div style="width:32px;height:32px;border-radius:6px;background:{BORDER};"></div>')
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;">{thumb_html}'
                    f'<span class="row-name">{fname}</span></div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(f'<span class="row-total">{obj_total}</span>', unsafe_allow_html=True)
            with c3:
                dot_s = "dot-red" if s_count > 0 else "dot-gray"
                st.markdown(
                    f'<span class="dot {dot_s}"></span>{s_count}&nbsp;&nbsp;'
                    f'<span class="dot dot-green"></span>{ns_count}',
                    unsafe_allow_html=True
                )
            with c4:
                if st.button("Open →", key=f"open_{fname}", use_container_width=True):
                    st.session_state.active_file = fname
                    ensure_annotated(fname, 0)
                    st.session_state.view = "editor"
                    st.rerun()

            

    st.markdown(
        f'<p class="legend-text"><span class="dot dot-red"></span>Sensitive'
        f'&nbsp;&nbsp;&nbsp;<span class="dot dot-green"></span>Non-sensitive'
        f'&nbsp;&nbsp;&nbsp;<span class="dot dot-orange"></span>Unknown sensitive'
        f'&nbsp;&nbsp;&nbsp;<span class="dot dot-teal"></span>Unknown non-sensitive</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════
# VIEW: EDITOR
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.view == "editor":
    active = st.session_state.active_file
    meta = st.session_state.files[active]
    w, h = meta["width"], meta["height"]
    header_back, header_title, header_meta = st.columns([0.8, 4, 1.4])
    with header_back:
        if st.button("← Files", key="close_editor", use_container_width=True):
            st.session_state.active_file = None
            st.session_state.view = "list"
            st.rerun()
    with header_title:
        st.markdown(
            f'<div style="font-size:1.05rem;font-weight:650;color:{TEXT}">' 
            f'{html.escape(active)}</div><div style="font-size:.76rem;color:{SUBTEXT}">' 
            "Review, correct, or add annotations</div>",
            unsafe_allow_html=True,
        )
    with header_meta:
        media_label = "Video" if meta["type"] == "video" else "Image"
        st.caption(f"{media_label} · {w} × {h}px")

    if meta["type"] == "video":
        frame_idx = st.slider(
            "Video frame",
            0,
            max(meta["frame_count"] - 1, 0),
            0,
            key=f"slider_{active}",
        )
        ann = st.session_state.annotations[active]
        has_detections = bool(ann.get("stage1_frames"))
        preview_key = f"preview_path_{active}"
        if st.button(
            "Render annotated preview",
            disabled=not has_detections,
            key=f"preview_btn_{active}",
            help="Build a video preview from the stored annotations; no inference is rerun.",
        ):
            with st.spinner("Rendering preview from stored annotations..."):
                st.session_state[preview_key] = render_annotated_video(active)
        if st.session_state.get(preview_key):
            with st.expander("Annotated video preview", expanded=True):
                st.video(st.session_state[preview_key])
                st.caption("This preview reuses stored detections and reviewer corrections.")
    else:
        frame_idx = 0

    ensure_annotated(active, frame_idx)
    regions = get_combined_regions(active, frame_idx)
    frame_rgb = get_frame_array(active, frame_idx)
    highlight_key = f"highlight_{active}_{frame_idx}"

    # Keep the editor responsive on large images while preserving original
    # coordinates in storage and export.
    display_width = min(900, w)
    display_height = max(1, round(display_width * h / w))
    display_frame = np.asarray(
        Image.fromarray(frame_rgb).resize((display_width, display_height))
    )
    display_regions = scale_regions_for_display(
        regions, display_width / w, display_height / h
    )

    files_col, canvas_col, inspector_col = st.columns([1.05, 3.25, 1.55])

    with files_col:
        st.markdown('<div class="editor-header">Media</div>', unsafe_allow_html=True)
        for fname, fmeta in st.session_state.files.items():
            icon = "🖼" if fmeta["type"] == "image" else "🎬"
            is_active = fname == active
            count = get_sensitive_count(fname)
            border_color = ACCENT if is_active else BORDER
            short_name = fname if len(fname) <= 24 else f"{fname[:21]}…"
            st.markdown(
                f'<div style="background:{SURFACE};border:1px solid {border_color};'
                f'border-radius:10px;padding:.6rem .7rem;margin-bottom:.35rem">'
                f'<div style="font-size:.8rem;color:{TEXT}">{icon} '
                f'{html.escape(short_name)}</div><div style="font-size:.7rem;color:{SUBTEXT};'
                f'margin-top:.2rem">{count} sensitive</div></div>',
                unsafe_allow_html=True,
            )
            if not is_active and st.button(
                f"Open {short_name}", key=f"sw_{fname}", use_container_width=True
            ):
                st.session_state.active_file = fname
                ensure_annotated(fname, 0)
                st.rerun()

    with canvas_col:
        st.markdown('<div class="editor-header annotation-canvas-anchor">Annotation Review</div>', unsafe_allow_html=True)
        tool_mode = st.radio(
            "Canvas tool",
            ["Review regions", "Draw missing object"],
            horizontal=True,
            key=f"tool_{active}_{frame_idx}",
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="legend-row">'
            '<span class="legend-text"><span class="dot dot-red"></span>Sensitive</span>'
            '<span class="legend-text"><span class="dot dot-green"></span>Non-sensitive</span>'
            '<span class="legend-text"><span class="dot dot-orange"></span>Unknown sensitive</span>'
            '<span class="legend-text"><span class="dot dot-teal"></span>Unknown non-sensitive</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if tool_mode == "Review regions":
            st.markdown(
                '<div class="canvas-help">Hover over a colored region to see its model, '
                'method, sensitivity decision, and score. Select it in the inspector to edit.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                build_hover_canvas(
                    display_frame,
                    display_regions,
                    st.session_state.get(highlight_key),
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="canvas-help">Draw a polygon, freehand outline, or rectangle around '
                'a missed object. Save to get a CLIP sensitivity suggestion, or choose '
                'Sensitive / Non-sensitive manually. Review automatic suggestions.</div>',
                unsafe_allow_html=True,
            )
            overlay_rgb = draw_overlay(
                display_frame,
                display_regions,
                display_height,
                display_width,
                highlight_idx=st.session_state.get(highlight_key),
            )
            draw_generation_key = f"draw_generation_{active}_{frame_idx}"
            st.session_state.setdefault(draw_generation_key, 0)
            drawing = drawing_canvas(
                Image.fromarray(overlay_rgb),
                background_color=BG, text_color=TEXT, surface_color=SURFACE,
                initial_sensitivity="auto",
                key=(
                    f"draw_{active}_{frame_idx}_"
                    f"{st.session_state[draw_generation_key]}"
                ),
            )
            event_key = f"last_drawing_event_{active}_{frame_idx}"
            if (
                isinstance(drawing, dict)
                and isinstance(drawing.get("event_id"), str)
                and drawing["event_id"] != st.session_state.get(event_key)
            ):
                # A component's last value can be returned on unrelated reruns.
                # Consume each save once, including rejected drawings.
                st.session_state[event_key] = drawing["event_id"]
                try:
                    new_region = add_manual_region(active, frame_idx, drawing)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    refreshed_regions = get_combined_regions(active, frame_idx)
                    st.session_state[highlight_key] = next(
                        (i for i, region in enumerate(refreshed_regions) if region is new_region),
                        None,
                    )
                    st.session_state[draw_generation_key] += 1
                    st.session_state.editor_notice = (
                        "Outline saved with a CLIP sensitivity suggestion. Inspect and review it."
                        if new_region.get("needs_review") else
                        "Outline saved with your sensitivity choice."
                    )
                    st.rerun()

    with inspector_col:
        st.markdown(
            f'<div class="editor-header">Object inspector · {len(regions)}</div>',
            unsafe_allow_html=True,
        )
        notice = st.session_state.pop("editor_notice", None)
        if notice:
            st.success(notice)
        if not regions:
            st.info("No regions yet. Choose ‘Draw missing object’ to add one.")

        for index, region in enumerate(regions):
            selected = st.session_state.get(highlight_key) == index
            class_name = region.get("class_name", "unknown")
            sensitivity_class = (
                "region-sensitive"
                if region.get("sensitivity") == "sensitive"
                else "region-nonsensitive"
            )
            if class_name.strip().lower() == "unknown":
                unknown_class = (
                    "region-unknown-nonsensitive"
                    if region.get("sensitivity") == "non_sensitive"
                    else "region-unknown-sensitive"
                )
                status_html = (
                    f'<span class="{unknown_class}">● unknown · '
                    f'{region.get("sensitivity", "unknown").replace("_", "-")} / needs label</span>'
                )
            else:
                status_html = (
                    f'<span class="{sensitivity_class}">● '
                    f'{region.get("sensitivity", "unknown").replace("_", "-")}</span>'
                )
            model, method, _score_name = region_provenance(region)
            if region.get("needs_review"):
                status_html += '<br><span class="region-conf">CLIP suggestion — needs review</span>'
            selected_style = f"border-color:{SELECTED};border-width:2px" if selected else ""
            st.markdown(
                f'<div class="region-card" style="{selected_style}">'
                f'<div class="region-name">{html.escape(name_region(region, index))}</div>'
                f'<div class="region-conf">{html.escape(confidence_text(region))}</div>'
                f'<div>{status_html}</div>'
                f'<span class="method-chip" title="{html.escape(method)}">'
                f'{html.escape(model)}</span></div>',
                unsafe_allow_html=True,
            )
            select_col, sensitivity_col = st.columns(2)
            with select_col:
                st.button(
                    "Selected" if selected else "Inspect",
                    key=f"inspect_{active}_{frame_idx}_{index}",
                    use_container_width=True,
                    on_click=toggle_highlight,
                    args=(highlight_key, index),
                )
            with sensitivity_col:
                next_label = (
                    "Mark safe"
                    if region.get("sensitivity") == "sensitive"
                    else "Mark sensitive"
                )
                st.button(
                    next_label,
                    key=f"toggle_{active}_{frame_idx}_{index}",
                    use_container_width=True,
                    on_click=toggle_region_sensitivity,
                    args=(active, frame_idx, index),
                )

        selected_index = st.session_state.get(highlight_key)
        if selected_index is not None and selected_index < len(regions):
            selected_region = regions[selected_index]
            revision = st.session_state.annotation_revisions.get(active, 0)
            key_prefix = f"region_{active}_{frame_idx}_{selected_index}_{revision}"
            st.markdown("---")
            st.markdown('<div class="editor-header">Edit selected object</div>', unsafe_allow_html=True)
            model, method, _score_name = region_provenance(selected_region)
            st.caption(f"Model: {model}\n\nMethod: {method}\n\n{confidence_text(selected_region)}")
            x1, y1, x2, y2 = [int(round(v)) for v in selected_region["bbox_xyxy"]]
            with st.form(f"{key_prefix}_form"):
                st.text_input(
                    "Object label",
                    value=selected_region.get("class_name", "unknown"),
                    key=f"{key_prefix}_name",
                )
                st.radio(
                    "Sensitivity",
                    ["sensitive", "non_sensitive"],
                    index=0 if selected_region.get("sensitivity") == "sensitive" else 1,
                    horizontal=True,
                    key=f"{key_prefix}_sensitivity",
                )
                st.checkbox(
                    "Replace segmentation with an adjusted rectangle",
                    key=f"{key_prefix}_replace_geometry",
                    help=(
                        "Use this when the automatic mask failed. The saved polygon becomes "
                        "the rectangle entered below."
                    ),
                )
                geometry_left, geometry_right = st.columns(2)
                with geometry_left:
                    st.number_input("Left x", 0, max(w - 1, 0), x1, key=f"{key_prefix}_x1")
                    st.number_input("Top y", 0, max(h - 1, 0), y1, key=f"{key_prefix}_y1")
                with geometry_right:
                    st.number_input("Right x", 1, max(w, 1), max(1, x2), key=f"{key_prefix}_x2")
                    st.number_input("Bottom y", 1, max(h, 1), max(1, y2), key=f"{key_prefix}_y2")
                st.form_submit_button(
                    "Save object changes",
                    use_container_width=True,
                    on_click=apply_region_editor,
                    args=(active, frame_idx, selected_index, key_prefix),
                )
            st.button(
                "Delete selected region",
                key=f"delete_{key_prefix}",
                use_container_width=True,
                on_click=remove_selected_region,
                args=(active, frame_idx, selected_index, highlight_key),
            )
