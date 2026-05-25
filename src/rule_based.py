# Rule-based scoring baseline. Extracts keywords from the command and scores each proposal using class match, spatial position, color match, and CenterNet detection confidence.

import re
import numpy as np
from typing import Optional
from PIL import Image

CLASS_KEYWORDS = {
    "car": ["car", "vehicle.car", "sedan", "suv", "automobile", "auto", "van", "minivan"],
    "truck": ["truck", "vehicle.truck", "pickup", "lorry", "semi", "tractor"],
    "bus": ["bus", "vehicle.bus", "shuttle", "coach"],
    "motorcycle": ["motorcycle", "vehicle.motorcycle", "motorbike", "scooter", "moped", "biker"],
    "bicycle": ["bicycle", "vehicle.bicycle", "bike", "cyclist"],
    "pedestrian": [
        "pedestrian", "human.pedestrian", "person", "man", "woman", "people",
        "guy", "lady", "child", "kid", "boy", "girl", "walker", "individual",
        "human", "someone",
    ],
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
    "red":    (180, 40, 40),
    "blue":   (40, 80, 180),
    "white":  (230, 230, 230),
    "black":  (30, 30, 30),
    "silver": (175, 175, 175),
    "grey":   (130, 130, 130),
    "gray":   (130, 130, 130),
    "green":  (50, 150, 60),
    "yellow": (220, 200, 60),
    "brown":  (110, 70, 40),
    "orange": (230, 130, 40),
    "dark":   (50, 50, 50),
    "light":  (210, 210, 210),
}

def _has_word(text: str, words: set) -> bool:
    for w in words:
        if " " in w:
            if w in text:
                return True
        elif re.search(rf"\b{re.escape(w)}\b", text):
            return True
    return False


# Parse a command into class, spatial, size, color, and generic-vehicle categories.
def extract_keywords(command: str) -> dict:
    cmd = command.lower()
    result = {"classes": [], "spatial": [], "size": [], "colors": [], "generic_vehicle": False}

    for cls_name, synonyms in CLASS_KEYWORDS.items():
        if _has_word(cmd, set(synonyms)):
            result["classes"].append(cls_name)

    if _has_word(cmd, GENERIC_VEHICLE_WORDS):
        result["generic_vehicle"] = True

    if _has_word(cmd, SPATIAL_LEFT): result["spatial"].append("left")
    if _has_word(cmd, SPATIAL_RIGHT): result["spatial"].append("right")
    if _has_word(cmd, SPATIAL_FRONT): result["spatial"].append("front")
    if _has_word(cmd, SPATIAL_BEHIND): result["spatial"].append("behind")
    if _has_word(cmd, SPATIAL_NEAR): result["spatial"].append("near")
    if _has_word(cmd, SPATIAL_FAR): result["spatial"].append("far")

    if _has_word(cmd, SIZE_BIG): result["size"].append("big")
    if _has_word(cmd, SIZE_SMALL): result["size"].append("small")

    for color in COLOR_RGB:
        if re.search(rf"\b{re.escape(color)}\b", cmd):
            result["colors"].append(color)

    return result


# Mean RGB of the central 60% of the box (skips edge/background pixels).
def _dominant_color(image: Image.Image, box: np.ndarray) -> Optional[tuple]:
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


# [0, 1] similarity between a crop's mean RGB and a target color name.
def _color_match_score(crop_rgb: tuple, target_color: str) -> float:
    if target_color not in COLOR_RGB:
        return 0.0
    target = np.array(COLOR_RGB[target_color], dtype=np.float32)
    crop = np.array(crop_rgb, dtype=np.float32)
    dist = np.linalg.norm(crop - target)
    return float(max(0, 1.0 - dist / 200.0))


# Score all proposals with the full rule set and return the index of the highest-scoring one.
def rule_based_score(sample: dict, img_width: int = 1600, img_height: int = 900, weights: Optional[dict] = None) -> int:
    w = {"class": 10.0, "spatial_matched": 2.0, "spatial_unmatched": 5.0, "size_bias": 1.0, "color": 3.0, "score": 1.0, "default": 0.5}
    if weights:
        w.update(weights)

    proposals = sample["proposals"]
    classes = sample["proposal_classes"]
    prop_scores = sample.get("proposal_scores", np.ones(len(proposals), dtype=np.float32))
    image = sample.get("image")
    n = len(proposals)

    keywords = extract_keywords(sample["command"])
    scores = np.zeros(n, dtype=np.float32)

    cx = (proposals[:, 0] + proposals[:, 2]) / 2.0
    cy = (proposals[:, 1] + proposals[:, 3]) / 2.0
    areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    area_norm = areas / areas.max() if areas.max() > 0 else np.zeros_like(areas)

    class_matched = False
    if keywords["classes"] or keywords["generic_vehicle"]:
        target_classes = set(keywords["classes"])
        if keywords["generic_vehicle"]:
            target_classes |= GENERIC_VEHICLE_CLASSES

        for i, cls in enumerate(classes):
            cls_lower = cls.lower()
            for target_cls in target_classes:
                synonyms = CLASS_KEYWORDS.get(target_cls, [target_cls])
                if any(syn in cls_lower for syn in synonyms) or target_cls in cls_lower:
                    scores[i] += w["class"]
                    class_matched = True
                    break

    if w["score"] != 0:
        scores += prop_scores * w["score"]

    spatial_scores = np.zeros(n, dtype=np.float32)
    if "left" in keywords["spatial"]:
        spatial_scores += 1.0 - cx / img_width
    if "right" in keywords["spatial"]:
        spatial_scores += cx / img_width
    if "front" in keywords["spatial"]:
        spatial_scores += ((1.0 - np.abs(cx / img_width - 0.5) * 2.0) + (1.0 - cy / img_height)) * 0.5
    if "behind" in keywords["spatial"]:
        spatial_scores += cy / img_height
    if "near" in keywords["spatial"]:
        spatial_scores += area_norm
    if "far" in keywords["spatial"]:
        spatial_scores += 1.0 - area_norm
    if "big" in keywords["size"]:
        spatial_scores += area_norm
    if "small" in keywords["size"]:
        spatial_scores += 1.0 - area_norm

    scores += spatial_scores * (w["spatial_matched"] if class_matched else w["spatial_unmatched"])

    if keywords["colors"] and image is not None:
        target_color = keywords["colors"][0]
        for i in range(n):
            crop_rgb = _dominant_color(image, proposals[i])
            if crop_rgb:
                scores[i] += _color_match_score(crop_rgb, target_color) * w["color"]

    if not keywords["spatial"] and not keywords["size"]:
        center_dist = np.sqrt(((cx / img_width) - 0.5) ** 2 + ((cy / img_height) - 0.5) ** 2)
        scores += (1.0 - center_dist) * w["default"]
        scores += area_norm * (w["default"] * 0.6)

    if class_matched and w["size_bias"] != 0:
        scores += area_norm * w["size_bias"]

    return int(np.argmax(scores))


# class matching only.
def rule_class_only(sample: dict) -> int:
    return rule_based_score(sample, weights={"spatial_matched": 0, "spatial_unmatched": 0, "color": 0, "score": 0, "default": 0, "size_bias": 0})


# class + spatial.
def rule_class_plus_spatial(sample: dict) -> int:
    return rule_based_score(sample, weights={"color": 0, "score": 0, "size_bias": 0})


# full scorer minus color.
def rule_no_color(sample: dict) -> int:
    return rule_based_score(sample, weights={"color": 0})


# full scorer minus CenterNet confidence.
def rule_no_confidence(sample: dict) -> int:
    return rule_based_score(sample, weights={"score": 0})
