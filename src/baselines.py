# Simple baselines for visual grounding.

import random
import numpy as np


# Pick a random proposal.
def random_baseline(sample: dict) -> int:
    return random.randint(0, len(sample["proposals"]) - 1)


# Pick the largest proposal by area.
def largest_box_baseline(sample: dict) -> int:
    proposals = sample["proposals"]
    areas = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    return int(np.argmax(areas))


# Pick the largest proposal whose class matches the GT object name.
def class_match_baseline(sample: dict) -> int:
    obj_name = sample["obj_name"].lower()
    proposals = sample["proposals"]
    classes = sample["proposal_classes"]

    matching_indices = [i for i, cls in enumerate(classes) if obj_name in cls.lower() or cls.lower() in obj_name]
    if not matching_indices:
        return largest_box_baseline(sample)

    matching_proposals = proposals[matching_indices]
    areas = (matching_proposals[:, 2] - matching_proposals[:, 0]) * (matching_proposals[:, 3] - matching_proposals[:, 1])
    return matching_indices[int(np.argmax(areas))]
