"""Talk2Car data: IoU, box geometry, the dataset, and image access (dir or tar)."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

IMG_W, IMG_H = 1600, 900


def compute_iou(box_a, box_b) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_geometry(boxes, img_w=IMG_W, img_h=IMG_H):
    """Per-box normalized geometry: (N, 5) = [cx, cy, w, h, area]."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return np.stack([cx, cy, w, h, w * h], axis=1).astype(np.float32)


def gt_index(sample) -> int:
    """Index of the candidate box that best overlaps the ground-truth box."""
    ious = [compute_iou(sample["gt_box"].tolist(), b.tolist()) for b in sample["proposals"]]
    return int(np.argmax(ious))


class Talk2CarDataset:
    """Talk2Car commands joined to CenterNet candidate boxes.

    Each sample dict: img_key, command, proposals (N,4), proposal_classes, proposal_scores,
    gt_box, gt_index, obj_name. Commands whose GT box has no candidate above iou_threshold
    are dropped (the detector missed the object).
    """

    def __init__(self, split: str, data_dir="data", iou_threshold: float = 0.5):
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images" / split
        self.iou_threshold = iou_threshold

        with open(self.data_dir / "commands" / f"{split}.json") as f:
            commands = json.load(f)
        with open(self.data_dir / "proposals" / f"{split}_proposals.json") as f:
            proposals = json.load(f)

        self.samples = []
        for cmd in commands:
            img_key = cmd["t2c_img"]
            if img_key not in proposals:
                continue
            gx, gy, gw, gh = cmd["2d_box"]
            gt_box = [gx, gy, gx + gw, gy + gh]

            props = proposals[img_key]
            boxes = np.array([p["box"] for p in props], dtype=np.float32)
            classes = [p.get("class", "unknown") for p in props]
            scores = np.array([p.get("score", 1.0) for p in props], dtype=np.float32)

            best = max(compute_iou(gt_box, b.tolist()) for b in boxes)
            if best < iou_threshold:
                continue

            sample = {
                "img_key": img_key,
                "command": cmd["command"],
                "gt_box": np.array(gt_box, dtype=np.float32),
                "proposals": boxes,
                "proposal_classes": classes,
                "proposal_scores": scores,
                "obj_name": cmd.get("obj_name", "unknown"),
            }
            sample["gt_index"] = gt_index(sample)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def load_image(self, sample) -> Image.Image:
        """Load a sample's image from the split directory."""
        path = self.img_dir / sample["img_key"]
        if not path.exists():
            path = self.img_dir / f"{sample['img_key']}.jpg"
        return Image.open(path).convert("RGB")


def ensure_image_tar(split: str, data_dir="data") -> Path:
    """Return the split's image tar, building it from data/images/<split>/ if missing.

    The tar bundles the jpgs into one file so a whole split can be read with a single open,
    which sidesteps macOS per-file read throttling. It is derived data and can always be rebuilt.
    """
    data_dir = Path(data_dir)
    tar_path = data_dir / f"images_{split}.tar"
    if tar_path.exists():
        return tar_path
    img_dir = data_dir / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"{img_dir} not found — cannot build {tar_path}")
    print(f"[data] building {tar_path} from {img_dir} ...", flush=True)
    # Use the system `tar` (signed binary): far faster than Python's tarfile here, which is
    # throttled by macOS security checks on per-file opens from the unsigned venv interpreter.
    import subprocess
    subprocess.run(["tar", "-cf", str(tar_path), "-C", str(img_dir), "."], check=True)
    return tar_path


def load_image_tar(path) -> dict:
    """Read a tar of jpgs into {img_key: raw_bytes} (bypasses macOS per-file read throttling)."""
    imgs = {}
    with tarfile.open(path, "r") as tar:
        for m in tar.getmembers():
            if m.name.endswith(".jpg") and "/._" not in m.name and not m.name.startswith("._"):
                imgs[m.name.lstrip("./")] = tar.extractfile(m).read()
    return imgs


def decode_image(raw_bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
