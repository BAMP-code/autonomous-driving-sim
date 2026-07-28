"""Backbone feature extraction and caching.

A backbone maps (image crop, command text) into a shared/comparable vector space. Features are
extracted once and cached as one npz per (split, backbone, crop_margin) so training reads from
disk. Swapping backbone (e.g. a bigger CLIP) or crop margin (context crops) is just a parameter.
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import Talk2CarDataset, ensure_image_tar, load_image_tar, decode_image

# short name -> (sentence-transformers model id, embedding dim)
BACKBONES = {
    "clip-b32": ("clip-ViT-B-32", 512),
    "clip-b16": ("clip-ViT-B-16", 512),
    "clip-l14": ("clip-ViT-L-14", 768),
}

CACHE_DIR = Path("data/cache")


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cache_path(split: str, backbone: str, crop_margin: float = 0.0) -> Path:
    tag = f"{backbone}_m{crop_margin:g}" if crop_margin else backbone
    return CACHE_DIR / f"feat_{tag}_{split}.npz"


def crop_box(image: Image.Image, box, margin: float = 0.0) -> Image.Image:
    """Crop a box, optionally expanded by `margin` (fraction of box size) for surrounding context."""
    x1, y1, x2, y2 = [float(c) for c in box]
    if margin:
        w, h = x2 - x1, y2 - y1
        x1 -= w * margin; x2 += w * margin
        y1 -= h * margin; y2 += h * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return Image.new("RGB", (8, 8))
    return image.crop((x1, y1, x2, y2))


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def extract(split: str, backbone: str = "clip-b32", data_dir="data",
            image_tar=None, crop_margin: float = 0.0, overwrite: bool = False) -> Path:
    """Extract and cache CLIP image (per candidate) and text (per command) features for a split."""
    out = cache_path(split, backbone, crop_margin)
    if out.exists() and not overwrite:
        print(f"[features] {out} exists, skipping")
        return out

    from sentence_transformers import SentenceTransformer
    model_id, dim = BACKBONES[backbone]
    dev = device()
    print(f"[features] {split} / {backbone} / margin={crop_margin} on {dev}")
    clip = SentenceTransformer(model_id, device=dev)

    ds = Talk2CarDataset(split=split, data_dir=data_dir)
    # Always read images from the split tar: per-file directory reads from the unsigned venv
    # interpreter are throttled by macOS and stall extraction. ensure_image_tar builds it once.
    if image_tar is None:
        image_tar = ensure_image_tar(split, data_dir)
    images = load_image_tar(image_tar)

    n = len(ds.samples)
    img_feats = np.zeros((n, 64, dim), dtype=np.float16)
    txt_feats = np.zeros((n, dim), dtype=np.float16)

    t0 = time.time()
    commands = [s["command"] for s in ds.samples]
    txt = clip.encode(commands, convert_to_numpy=True, device=dev, batch_size=256, show_progress_bar=False)
    txt_feats[:] = _l2(txt).astype(np.float16)

    for i, s in enumerate(ds.samples):
        image = decode_image(images[s["img_key"]]) if images else ds.load_image(s)
        crops = [crop_box(image, b, crop_margin) for b in s["proposals"]]
        emb = clip.encode(crops, convert_to_numpy=True, device=dev, batch_size=64, show_progress_bar=False)
        img_feats[i] = _l2(emb).astype(np.float16)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, img=img_feats, txt=txt_feats)
    print(f"[features] saved {out}  ({time.time()-t0:.0f}s)")
    return out


def load(split: str, backbone: str = "clip-b32", crop_margin: float = 0.0):
    """Load cached (img (N,64,D), txt (N,D)) features."""
    path = cache_path(split, backbone, crop_margin)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run features.extract(split={split!r}, backbone={backbone!r}) first")
    d = np.load(path)
    return d["img"], d["txt"]
