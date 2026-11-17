"""
config.py
---------
Central place for model paths, device selection, and the sensitivity
lookup table.
"""

import torch

# ----------------------------------------------------------------------
# Device selection — Apple Silicon (M4) uses MPS, not CUDA
# ----------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# ----------------------------------------------------------------------
# Model weights
# ----------------------------------------------------------------------
YOLO_WEIGHTS = "models/best1.pt"        # fine-tuned SH17 PPE model
YOLO26_WEIGHTS = "yolo26l.pt"           # standard COCO-pretrained model — auto-downloads on first use
SAM_WEIGHTS = "sam2_b.pt"               # auto-downloaded by ultralytics on first use

YOLO_CONF_THRESHOLD = 0.35   
YOLO_IOU_THRESHOLD = 0.5     


STAGE2_MIN_AREA_PCT = 0.0005   # ~0.05% of the frame — roughly 32x32px on a 1920x1080 image
STAGE2_MAX_AREA_PCT = 0.6      # ~60% of the frame



STAGE2_TRACK_CAP = 15

# Per-class confidence overrides. 
CLASS_CONF_OVERRIDES = {
    "face": 0.15,
    "ear": 0.15,
    "head": 0.15,
}


YOLO_PREDICT_CONF = min([YOLO_CONF_THRESHOLD, *CLASS_CONF_OVERRIDES.values()])

# Decide whether a detection's confidence clears its class-specific threshold (or the default one).
def keep_detection(class_name: str, confidence: float) -> bool:
    name = class_name.strip().lower()
    threshold = CLASS_CONF_OVERRIDES.get(name, YOLO_CONF_THRESHOLD)
    return confidence >= threshold

# ----------------------------------------------------------------------
# Sensitivity lookup table (case-insensitive match on YOLO class name)
# ----------------------------------------------------------------------
SENSITIVE_CLASSES = {
    # SH17 classes
    "face", "ear", "head", "person",
    # COCO sensitive classes
    "laptop", "tv", "monitor", "screen",
    "cell phone", "computer",
    "book",  
    "clock",  
}

NON_SENSITIVE_CLASSES = {
    # SH17 classes
    "helmet", "gloves", "safety-vest",
    "glasses", "earmuffs", "face-mask", "face-guard",
    "medical-suit", "safety-suit", "shoes", "tool",
    "hands", "foot",
    # COCO non-sensitive classes
    "chair", "couch", "table", "dining table",
    "bottle", "cup", "bowl",
    "bicycle", "car", "truck", "bus",
    "bench", "potted plant",
    "refrigerator", "sink",
    "keyboard", "mouse",
    "motorcycle", "airplane", "train", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "wine glass", "fork", "knife", "spoon",
    "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "bed", "toilet",
    "remote", "microwave", "oven", "toaster",
    "umbrella", "backpack", "handbag", "suitcase", "tie",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}


# Real semantic grouping for COCO's "supercategory" field 
SUPERCATEGORY_MAP = {
    # Identity — the actual privacy-relevant group
    "person": "person", "face": "person", "ear": "person", "head": "person",
    "hands": "person", "foot": "person",
    # Personal protective equipment / work gear
    "helmet": "ppe", "gloves": "ppe", "safety-vest": "ppe", "glasses": "ppe",
    "earmuffs": "ppe", "face-mask": "ppe", "face-guard": "ppe",
    "medical-suit": "ppe", "safety-suit": "ppe", "shoes": "ppe", "tool": "ppe",
    # Screens & computing devices
    "laptop": "screen_device", "tv": "screen_device", "monitor": "screen_device",
    "screen": "screen_device", "cell phone": "screen_device", "computer": "screen_device",
    # Documents / readable info
    "book": "document", "clock": "document",
    # Furniture & fixtures
    "chair": "furniture", "couch": "furniture", "table": "furniture",
    "dining table": "furniture", "bed": "furniture", "bench": "furniture",
    "potted plant": "furniture", "toilet": "furniture",
    # Kitchenware
    "bottle": "kitchenware", "cup": "kitchenware", "bowl": "kitchenware",
    "wine glass": "kitchenware", "fork": "kitchenware", "knife": "kitchenware",
    "spoon": "kitchenware",
    # Vehicles
    "bicycle": "vehicle", "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle", "airplane": "vehicle", "train": "vehicle", "boat": "vehicle",
    # Traffic / street infrastructure
    "traffic light": "traffic_infrastructure", "fire hydrant": "traffic_infrastructure",
    "stop sign": "traffic_infrastructure", "parking meter": "traffic_infrastructure",
    # Animals
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
    # Sports equipment
    "frisbee": "sports_equipment", "skis": "sports_equipment", "snowboard": "sports_equipment",
    "sports ball": "sports_equipment", "kite": "sports_equipment", "baseball bat": "sports_equipment",
    "baseball glove": "sports_equipment", "skateboard": "sports_equipment",
    "surfboard": "sports_equipment", "tennis racket": "sports_equipment",
    # Food
    "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
    "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food",
    "donut": "food", "cake": "food",
    # Appliances & small electronics
    "remote": "appliance", "microwave": "appliance", "oven": "appliance",
    "toaster": "appliance", "refrigerator": "appliance", "sink": "appliance",
    "keyboard": "appliance", "mouse": "appliance",
    # Personal belongings
    "umbrella": "personal_item", "backpack": "personal_item", "handbag": "personal_item",
    "suitcase": "personal_item", "tie": "personal_item",
    # Everything else COCO has that doesn't fit a bigger group
    "vase": "misc_object", "scissors": "misc_object", "teddy bear": "misc_object",
    "hair drier": "misc_object", "toothbrush": "misc_object",
    # SAM/CLIP-found regions that were never given a specific class
    "unknown": "background",
}

# Map a class name to its broader semantic group (e.g. "person", "ppe", "vehicle") using SUPERCATEGORY_MAP.
def classify_supercategory(class_name: str) -> str:
    
    return SUPERCATEGORY_MAP.get(class_name.strip().lower(), "object")

# Classify a YOLO class name as "sensitive" or "non_sensitive", defaulting unmapped classes to sensitive.
def classify_sensitivity(class_name: str) -> str:
    
    name = class_name.strip().lower()
    if name in NON_SENSITIVE_CLASSES:
        return "non_sensitive"
    return "sensitive"

# Debug utility: print every class the given YOLO weights actually know, alongside its assigned sensitivity.
def verify_class_names(weights_path: str = YOLO_WEIGHTS):
   
    from ultralytics import YOLO
    model = YOLO(weights_path)
    print(f"{'class_name':<25} sensitivity")
    print("-" * 40)
    for idx, name in model.names.items():
        print(f"{name:<25} {classify_sensitivity(name)}")


if __name__ == "__main__":
    verify_class_names()