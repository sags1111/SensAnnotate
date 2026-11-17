"""
Model registry and loading.

This is the ONLY file that should need to change to offer a new YOLO
variant, or if a weights file path moves. Nothing in pipeline.py or app.py
should hardcode a weights path directly — they ask this module for "the
currently selected model" and this module decides which weights that maps
to.

Models are cached by weights path (see load_yolo / load_sam below), so
switching between two YOLO options the user has already used is instant —
no reload from disk, and no need to re-upload files. Only the first time a
given model is selected does it pay the load cost.
"""

from dataclasses import dataclass
from typing import Dict

from ultralytics import YOLO, SAM

import config


@dataclass(frozen=True)
class YoloModelOption:
    key: str              # stable internal id — used in session state and stored per-region "source" field
    display_name: str     # shown in the UI picker
    weights_path: str
    description: str      # shown as help text under the picker
    is_finetuned: bool    # True = your SH17 PPE model, False = generic COCO-pretrained model


# Add a new entry here to offer a new YOLO variant in the UI —
# nothing else needs to change to make it selectable.
YOLO_MODEL_OPTIONS: Dict[str, YoloModelOption] = {
    "finetuned_sh17": YoloModelOption(
        key="finetuned_sh17",
        display_name="Fine-tuned YOLO (manufacturing / PPE)",
        weights_path=config.YOLO_WEIGHTS,  # single source of truth stays in config.py
        description=(
            "Best for manufacturing and industrial footage — recognizes workers, "
            "faces, and safety gear like helmets, gloves, and safety vests."
        ),
        is_finetuned=True,
    ),
    "yolo26l_general": YoloModelOption(
        key="yolo26l_general",
        display_name="Standard YOLO26 (general objects)",
        weights_path=config.YOLO26_WEIGHTS,
        description=(
            "Best for everyday, non-industrial footage — recognizes common objects "
            "like people, laptops, and phones. Needs internet the first time you use it."
        ),
        is_finetuned=False,
    ),
}

DEFAULT_YOLO_MODEL_KEY = "finetuned_sh17"

# Look up a YOLO model option by its key, raising a clear error if it doesn't exist.
def get_model_option(key: str) -> YoloModelOption:
    if key not in YOLO_MODEL_OPTIONS:
        raise KeyError(f"Unknown YOLO model key: {key!r}. Available: {list(YOLO_MODEL_OPTIONS)}")
    return YOLO_MODEL_OPTIONS[key]

# Cache loaded YOLO/SAM models to avoid loading the same model multiple times.
_yolo_cache: Dict[str, YOLO] = {}
_sam_cache: Dict[str, SAM] = {}

# Load and cache a YOLO model for reuse.
def load_yolo(weights_path: str) -> YOLO:
    if weights_path not in _yolo_cache:
        _yolo_cache[weights_path] = YOLO(weights_path)
    return _yolo_cache[weights_path]

# Load and cache a SAM model for reuse.
def load_sam(weights_path: str) -> SAM:
    if weights_path not in _sam_cache:
        _sam_cache[weights_path] = SAM(weights_path)
    return _sam_cache[weights_path]