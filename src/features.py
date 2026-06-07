# Feature extraction: ResNet-50 visual (2048-d), MiniLM language (384-d), geometric (5-d), plus the engineered per-candidate cues reused from the rule-based scorer.

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dataset import compute_iou
from rule_based import extract_keywords, CLASS_KEYWORDS, GENERIC_VEHICLE_CLASSES, _dominant_color, _color_match_score

W, H = 1600, 900


# ResNet-50 with the classification head removed; encodes box crops to (N, 2048).
class VisualFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = device
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_batch(self, image: Image.Image, boxes: np.ndarray) -> np.ndarray:
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros(3, 224, 224))
            else:
                crops.append(self.transform(image.crop((x1, y1, x2, y2))))
        return self.model(torch.stack(crops).to(self.device)).cpu().numpy()


# all-MiniLM-L6-v2 sentence embedding of the command -> (384,).
class LanguageFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def extract(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)


# 5-d [cx/W, cy/H, w/W, h/H, area] per box.
def compute_geometric_batch(boxes: np.ndarray, img_width: int = W, img_height: int = H) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_width
    cy = (y1 + y2) / 2.0 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return np.stack([cx, cy, w, h, w * h], axis=1).astype(np.float32)


# Index of the proposal that best overlaps the GT box (the training target).
def gt_index(sample: dict) -> int:
    ious = [compute_iou(sample["gt_box"].tolist(), b.tolist()) for b in sample["proposals"]]
    return int(np.argmax(ious))


# Extract and cache base features to one .npz per sample (visual, language, geometric, gt_index).
def precache_features(dataset, output_dir, device: str = "cpu") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis = VisualFeatureExtractor(device=device)
    lang = LanguageFeatureExtractor(device=device)
    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        path = output_dir / f"{i:06d}.npz"
        if path.exists():
            continue
        sample = dataset[i]
        np.savez_compressed(
            path,
            visual=vis.extract_batch(sample["image"], sample["proposals"]),
            language=lang.extract(sample["command"]),
            geometric=compute_geometric_batch(sample["proposals"]),
            gt_index=gt_index(sample),
        )


# 1.0 per candidate whose detector class matches a class word in the command.
def class_match_vector(command: str, classes: list) -> np.ndarray:
    kw = extract_keywords(command)
    targets = set(kw["classes"])
    if kw["generic_vehicle"]:
        targets |= GENERIC_VEHICLE_CLASSES
    out = np.zeros((len(classes), 1), dtype=np.float32)
    for i, cls in enumerate(classes):
        cl = cls.lower()
        for t in targets:
            if any(s in cl for s in CLASS_KEYWORDS.get(t, [t])) or t in cl:
                out[i, 0] = 1.0
                break
    return out


# Spatial-position score per candidate from the command's spatial/size words.
def spatial_match_vector(command: str, boxes: np.ndarray) -> np.ndarray:
    kw = extract_keywords(command)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_norm = areas / areas.max() if areas.max() > 0 else np.zeros_like(areas)
    s = np.zeros(len(boxes), dtype=np.float32)
    if "left" in kw["spatial"]: s += 1.0 - cx / W
    if "right" in kw["spatial"]: s += cx / W
    if "front" in kw["spatial"]: s += ((1.0 - np.abs(cx / W - 0.5) * 2.0) + (1.0 - cy / H)) * 0.5
    if "behind" in kw["spatial"]: s += cy / H
    if "near" in kw["spatial"]: s += area_norm
    if "far" in kw["spatial"]: s += 1.0 - area_norm
    if "big" in kw["size"]: s += area_norm
    if "small" in kw["size"]: s += 1.0 - area_norm
    return s.reshape(-1, 1).astype(np.float32)


# Color-match score per candidate (mean crop color vs the color named in the command).
def color_match_vector(command: str, boxes: np.ndarray, image) -> np.ndarray:
    kw = extract_keywords(command)
    out = np.zeros((len(boxes), 1), dtype=np.float32)
    if not kw["colors"] or image is None:
        return out
    target = kw["colors"][0]
    for i in range(len(boxes)):
        rgb = _dominant_color(image, boxes[i])
        if rgb:
            out[i, 0] = _color_match_score(rgb, target)
    return out
