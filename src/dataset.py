# Talk2Car dataset wrapper. Loads commands, ground-truth boxes, and CenterNet proposals from disk.

import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# IoU between two [x1, y1, x2, y2] boxes.
def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Talk2Car dataset. Each item has image, command, proposals, proposal_classes, proposal_scores, gt_box, obj_name.
class Talk2CarDataset(Dataset):
    def __init__(self, split: str, data_dir: str | Path, iou_threshold: float = 0.5):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images" / split
        self.iou_threshold = iou_threshold

        with open(self.data_dir / "commands" / f"{split}.json") as f:
            self.commands = json.load(f)

        with open(self.data_dir / "proposals" / f"{split}_proposals.json") as f:
            self.proposals = json.load(f)

        self.samples = self._build_samples()

    # Drop commands whose GT box has no proposal with IoU above iou_threshold.
    def _build_samples(self) -> list[dict]:
        samples = []
        for cmd in self.commands:
            img_key = cmd["t2c_img"]
            if img_key not in self.proposals:
                continue

            gt = cmd["2d_box"]
            gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]

            props = self.proposals[img_key]
            prop_boxes = np.array([p["box"] for p in props], dtype=np.float32)
            prop_classes = [p.get("class", "unknown") for p in props]
            prop_scores = np.array([p.get("score", 1.0) for p in props], dtype=np.float32)

            best_iou = max(compute_iou(gt_box, box.tolist()) for box in prop_boxes)
            if best_iou < self.iou_threshold:
                continue

            samples.append({
                "img_key": img_key,
                "command": cmd["command"],
                "gt_box": np.array(gt_box, dtype=np.float32),
                "proposals": prop_boxes,
                "proposal_classes": prop_classes,
                "proposal_scores": prop_scores,
                "obj_name": cmd.get("obj_name", "unknown"),
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        img_path = self.img_dir / sample["img_key"]
        if not img_path.exists():
            img_path = self.img_dir / f"{sample['img_key']}.jpg"
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "command": sample["command"],
            "proposals": sample["proposals"],
            "proposal_classes": sample["proposal_classes"],
            "proposal_scores": sample["proposal_scores"],
            "gt_box": sample["gt_box"],
            "obj_name": sample["obj_name"],
        }
