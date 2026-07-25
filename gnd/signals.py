"""Engineered per-candidate signals, as a registry.

Each signal maps (sample, image) -> (N, k) float32 array. `build_signal_block` concatenates a
chosen subset into (N, K). Adding a new signal (e.g. a relational feature) is one function plus
one registry entry — no training-script edits.
"""

import numpy as np

from .data import IMG_W, IMG_H
from . import rules


def _confidence(sample, image):
    return sample["proposal_scores"].reshape(-1, 1).astype(np.float32)


def _class_match(sample, image):
    kw = rules.extract_keywords(sample["command"])
    targets = set(kw["classes"])
    if kw["generic_vehicle"]:
        targets |= rules.GENERIC_VEHICLE_CLASSES
    out = np.zeros((len(sample["proposal_classes"]), 1), dtype=np.float32)
    if targets:
        for i, cls in enumerate(sample["proposal_classes"]):
            if rules.class_matches(cls, targets):
                out[i, 0] = 1.0
    return out


def _color_match(sample, image):
    boxes = sample["proposals"]
    out = np.zeros((len(boxes), 1), dtype=np.float32)
    kw = rules.extract_keywords(sample["command"])
    if not kw["colors"] or image is None:
        return out
    target = kw["colors"][0]
    for i in range(len(boxes)):
        rgb = rules.dominant_color(image, boxes[i])
        if rgb:
            out[i, 0] = rules.color_match_score(rgb, target)
    return out


def _spatial_match(sample, image):
    boxes = sample["proposals"]
    kw = rules.extract_keywords(sample["command"])
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    an = areas / areas.max() if areas.max() > 0 else np.zeros_like(areas)
    s = np.zeros(len(boxes), dtype=np.float32)
    if "left" in kw["spatial"]: s += 1.0 - cx / IMG_W
    if "right" in kw["spatial"]: s += cx / IMG_W
    if "front" in kw["spatial"]: s += ((1.0 - np.abs(cx / IMG_W - 0.5) * 2.0) + (1.0 - cy / IMG_H)) * 0.5
    if "behind" in kw["spatial"]: s += cy / IMG_H
    if "near" in kw["spatial"]: s += an
    if "far" in kw["spatial"]: s += 1.0 - an
    if "big" in kw["size"]: s += an
    if "small" in kw["size"]: s += 1.0 - an
    return s.reshape(-1, 1).astype(np.float32)


# name -> (function, width, needs_image). Only color_match reads the PIL image.
REGISTRY = {
    "confidence": (_confidence, 1, False),
    "class_match": (_class_match, 1, False),
    "color_match": (_color_match, 1, True),
    "spatial_match": (_spatial_match, 1, False),
}


def signal_width(names) -> int:
    return sum(REGISTRY[name][1] for name in names)


def needs_image(names) -> bool:
    return any(REGISTRY[name][2] for name in names)


def sample_needs_image(sample, names) -> bool:
    """Whether this specific sample needs its image decoded (color_match only fires on color words)."""
    if "color_match" in names and rules.extract_keywords(sample["command"])["colors"]:
        return True
    return False


def build_signal_block(sample, names, image=None) -> np.ndarray:
    """Concatenate the selected signals into (N, K). Order follows `names`."""
    if not names:
        n = len(sample["proposals"])
        return np.zeros((n, 0), dtype=np.float32)
    parts = [REGISTRY[name][0](sample, image) for name in names]
    return np.concatenate(parts, axis=1).astype(np.float32)
