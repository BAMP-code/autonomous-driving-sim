"""AP50 and stratified evaluation. Operates on predicted candidate indices, so it works for any
scorer (rule-based or learned)."""

from collections import defaultdict

import numpy as np

from .data import compute_iou
from .rules import extract_keywords, class_matches, GENERIC_VEHICLE_CLASSES


def ap50(pred_indices, samples) -> float:
    """Fraction of predictions whose chosen box overlaps the GT box by IoU >= 0.5."""
    correct = 0
    for idx, s in zip(pred_indices, samples):
        if compute_iou(s["proposals"][idx].tolist(), s["gt_box"].tolist()) >= 0.5:
            correct += 1
    return correct / len(samples) if samples else 0.0


def _command_type(sample) -> str:
    kw = extract_keywords(sample["command"])
    if kw["spatial"]:
        return "spatial"
    if kw["colors"]:
        return "appearance"
    if kw["size"]:
        return "size"
    return "other"


def _distractors(sample) -> int:
    """Same-class candidates other than the GT (proxy for scene clutter)."""
    kw = extract_keywords(sample["command"])
    targets = set(kw["classes"])
    if kw["generic_vehicle"]:
        targets |= GENERIC_VEHICLE_CLASSES
    if not targets:
        return 0
    return sum(1 for c in sample["proposal_classes"] if class_matches(c, targets)) - 1


def stratified(pred_indices, samples) -> dict:
    """AP50 broken down by same-class distractors, command type, and command length."""
    buckets = {"distractors": defaultdict(list), "type": defaultdict(list), "length": defaultdict(list)}
    for idx, s in zip(pred_indices, samples):
        hit = compute_iou(s["proposals"][idx].tolist(), s["gt_box"].tolist()) >= 0.5
        nd = _distractors(s)
        buckets["distractors"]["0" if nd <= 0 else "1-3" if nd <= 3 else "4+"].append(hit)
        buckets["type"][_command_type(s)].append(hit)
        nw = len(s["command"].split())
        buckets["length"]["short (1-7)" if nw <= 7 else "medium (8-14)" if nw <= 14 else "long (15+)"].append(hit)

    return {axis: {k: {"ap50": float(np.mean(v)), "n": len(v)} for k, v in sorted(d.items())}
            for axis, d in buckets.items()}
