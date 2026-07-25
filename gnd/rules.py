"""Keyword extraction and the hand-designed rule-based scorer.

Also exposes the vocabularies and color helpers reused by the engineered signals in signals.py.
"""

import re

import numpy as np
from PIL import Image

from .data import IMG_W, IMG_H

CLASS_KEYWORDS = {
    "car": ["car", "vehicle.car", "sedan", "suv", "automobile", "auto", "van", "minivan"],
    "truck": ["truck", "vehicle.truck", "pickup", "lorry", "semi", "tractor"],
    "bus": ["bus", "vehicle.bus", "shuttle", "coach"],
    "motorcycle": ["motorcycle", "vehicle.motorcycle", "motorbike", "scooter", "moped", "biker"],
    "bicycle": ["bicycle", "vehicle.bicycle", "bike", "cyclist"],
    "pedestrian": ["pedestrian", "human.pedestrian", "person", "man", "woman", "people",
                   "guy", "lady", "child", "kid", "boy", "girl", "walker", "individual",
                   "human", "someone"],
    "traffic_light": ["traffic light", "stoplight"],
    "stop_sign": ["stop sign"],
    "construction": ["vehicle.construction", "construction"],
    "trailer": ["trailer", "vehicle.trailer"],
    "barrier": ["barrier", "cone", "trafficcone", "traffic cone"],
}
GENERIC_VEHICLE_WORDS = {"vehicle", "automobile"}
GENERIC_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "trailer", "construction"}

SPATIAL_LEFT = {"left", "left-hand", "leftmost", "far left"}
SPATIAL_RIGHT = {"right", "right-hand", "rightmost", "far right"}
SPATIAL_FRONT = {"front", "ahead", "forward", "in front", "approaching", "upcoming"}
SPATIAL_BEHIND = {"behind", "back", "rear", "following", "trailing"}
SPATIAL_NEAR = {"near", "close", "closest", "nearest", "next to", "beside", "adjacent"}
SPATIAL_FAR = {"far", "distant", "further", "farthest"}
SIZE_BIG = {"big", "large", "biggest", "largest", "huge"}
SIZE_SMALL = {"small", "little", "smallest", "tiny"}

COLOR_RGB = {
    "red": (180, 40, 40), "blue": (40, 80, 180), "white": (230, 230, 230),
    "black": (30, 30, 30), "silver": (175, 175, 175), "grey": (130, 130, 130),
    "gray": (130, 130, 130), "green": (50, 150, 60), "yellow": (220, 200, 60),
    "brown": (110, 70, 40), "orange": (230, 130, 40), "dark": (50, 50, 50),
    "light": (210, 210, 210),
}


def _has_word(text: str, words) -> bool:
    """True if any word/phrase from `words` appears in `text` (word-boundary aware)."""
    for w in words:
        if " " in w:
            if w in text:
                return True
        elif re.search(rf"\b{re.escape(w)}\b", text):
            return True
    return False


def extract_keywords(command: str) -> dict:
    """Parse a command into class, spatial, size, color, and generic-vehicle categories."""
    cmd = command.lower()
    r = {"classes": [], "spatial": [], "size": [], "colors": [], "generic_vehicle": False}
    for cls_name, syns in CLASS_KEYWORDS.items():
        if _has_word(cmd, set(syns)):
            r["classes"].append(cls_name)
    if _has_word(cmd, GENERIC_VEHICLE_WORDS):
        r["generic_vehicle"] = True
    if _has_word(cmd, SPATIAL_LEFT): r["spatial"].append("left")
    if _has_word(cmd, SPATIAL_RIGHT): r["spatial"].append("right")
    if _has_word(cmd, SPATIAL_FRONT): r["spatial"].append("front")
    if _has_word(cmd, SPATIAL_BEHIND): r["spatial"].append("behind")
    if _has_word(cmd, SPATIAL_NEAR): r["spatial"].append("near")
    if _has_word(cmd, SPATIAL_FAR): r["spatial"].append("far")
    if _has_word(cmd, SIZE_BIG): r["size"].append("big")
    if _has_word(cmd, SIZE_SMALL): r["size"].append("small")
    for color in COLOR_RGB:
        if re.search(rf"\b{re.escape(color)}\b", cmd):
            r["colors"].append(color)
    return r


def dominant_color(image: Image.Image, box) -> tuple | None:
    """Mean RGB of the central 60% of a box (skips edge/background pixels)."""
    x1, y1, x2, y2 = [int(c) for c in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    w, h = x2 - x1, y2 - y1
    crop = image.crop((x1 + int(w * 0.2), y1 + int(h * 0.2), x2 - int(w * 0.2), y2 - int(h * 0.2)))
    arr = np.array(crop)
    if arr.size == 0:
        return None
    return tuple(arr.reshape(-1, 3).mean(axis=0))


def color_match_score(crop_rgb, target_color: str) -> float:
    """[0, 1] similarity between a crop's mean RGB and a target color name."""
    if target_color not in COLOR_RGB:
        return 0.0
    target = np.array(COLOR_RGB[target_color], dtype=np.float32)
    crop = np.array(crop_rgb, dtype=np.float32)
    return float(max(0, 1.0 - np.linalg.norm(crop - target) / 200.0))


def class_matches(box_class: str, target_classes) -> bool:
    """Whether a detector class string matches any target class (via the synonym vocab)."""
    cl = box_class.lower()
    for t in target_classes:
        syns = CLASS_KEYWORDS.get(t, [t])
        if any(s in cl for s in syns) or t in cl:
            return True
    return False


def rule_based_score(sample: dict, weights: dict | None = None, image=None) -> int:
    """Score all candidates with the hand-designed rules; return the argmax index."""
    w = {"class": 10.0, "spatial_matched": 2.0, "spatial_unmatched": 5.0,
         "size_bias": 1.0, "color": 3.0, "score": 1.0, "default": 0.5}
    if weights:
        w.update(weights)

    proposals = sample["proposals"]
    classes = sample["proposal_classes"]
    prop_scores = sample.get("proposal_scores", np.ones(len(proposals), dtype=np.float32))
    image = image if image is not None else sample.get("image")
    kw = extract_keywords(sample["command"])
    n = len(proposals)
    scores = np.zeros(n, dtype=np.float32)

    cx = (proposals[:, 0] + proposals[:, 2]) / 2.0
    cy = (proposals[:, 1] + proposals[:, 3]) / 2.0
    areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    area_norm = areas / areas.max() if areas.max() > 0 else np.zeros_like(areas)

    class_matched = False
    if kw["classes"] or kw["generic_vehicle"]:
        targets = set(kw["classes"])
        if kw["generic_vehicle"]:
            targets |= GENERIC_VEHICLE_CLASSES
        for i, cls in enumerate(classes):
            if class_matches(cls, targets):
                scores[i] += w["class"]
                class_matched = True

    if w["score"]:
        scores += prop_scores * w["score"]

    spatial = np.zeros(n, dtype=np.float32)
    if "left" in kw["spatial"]: spatial += 1.0 - cx / IMG_W
    if "right" in kw["spatial"]: spatial += cx / IMG_W
    if "front" in kw["spatial"]:
        spatial += ((1.0 - np.abs(cx / IMG_W - 0.5) * 2.0) + (1.0 - cy / IMG_H)) * 0.5
    if "behind" in kw["spatial"]: spatial += cy / IMG_H
    if "near" in kw["spatial"]: spatial += area_norm
    if "far" in kw["spatial"]: spatial += 1.0 - area_norm
    if "big" in kw["size"]: spatial += area_norm
    if "small" in kw["size"]: spatial += 1.0 - area_norm
    scores += spatial * (w["spatial_matched"] if class_matched else w["spatial_unmatched"])

    if kw["colors"] and image is not None:
        target = kw["colors"][0]
        for i in range(n):
            rgb = dominant_color(image, proposals[i])
            if rgb:
                scores[i] += color_match_score(rgb, target) * w["color"]

    if not kw["spatial"] and not kw["size"]:
        center_dist = np.sqrt(((cx / IMG_W) - 0.5) ** 2 + ((cy / IMG_H) - 0.5) ** 2)
        scores += (1.0 - center_dist) * w["default"]
        scores += area_norm * (w["default"] * 0.6)

    if class_matched and w["size_bias"]:
        scores += area_norm * w["size_bias"]

    return int(np.argmax(scores))
