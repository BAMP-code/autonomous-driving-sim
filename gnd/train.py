"""Config-driven training. One loop for every backbone/signal combination, saving a checkpoint
bundle (weights + the full config + input composition) so inference can reconstruct the model.

Feature vector per candidate, in a fixed order:
    [ CLIP image (D) | CLIP text (D, tiled per candidate) | engineered signals (K) ]
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import features, signals
from .data import Talk2CarDataset, ensure_image_tar, load_image_tar, decode_image
from .model import GroundingMLP
from .evaluate import ap50

CKPT_DIR = Path("checkpoints")


@dataclass
class Config:
    backbone: str = "clip-b32"
    crop_margin: float = 0.0
    signals: list = field(default_factory=list)   # names from signals.REGISTRY
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    dropout: float = 0.3
    seed: int = 42
    data_dir: str = "data"
    name: str = "model"


def _assemble(img_feats, txt_feats, signal_blocks):
    """Build the list of per-sample (N, input_dim) float32 arrays in the fixed feature order."""
    out = []
    for i in range(len(img_feats)):
        n = img_feats[i].shape[0]
        txt_tiled = np.tile(txt_feats[i].astype(np.float32), (n, 1))
        parts = [img_feats[i].astype(np.float32), txt_tiled]
        if signal_blocks is not None:
            parts.append(signal_blocks[i])
        out.append(np.concatenate(parts, axis=1).astype(np.float32))
    return out


def _signal_blocks(ds, cfg, split):
    if not cfg.signals:
        return None
    images = load_image_tar(ensure_image_tar(split, cfg.data_dir)) if signals.needs_image(cfg.signals) else None
    blocks = []
    for s in ds.samples:
        # Only decode the image when a color-needing signal will actually use it.
        img = None
        if images is not None and signals.sample_needs_image(s, cfg.signals):
            img = decode_image(images[s["img_key"]])
        blocks.append(signals.build_signal_block(s, cfg.signals, img))
    return blocks


def _evaluate(model, feats, samples) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for f in feats:
            preds.append(int(model(torch.from_numpy(f)).argmax().item()))
    return ap50(preds, samples)


def train(cfg: Config, verbose: bool = True) -> dict:
    """Train per cfg, return {best_ap50, input_dim, checkpoint}. Saves best checkpoint to disk."""
    t0 = time.time()
    train_ds = Talk2CarDataset(split="train", data_dir=cfg.data_dir)
    val_ds = Talk2CarDataset(split="val", data_dir=cfg.data_dir)

    ti, tt = features.load("train", cfg.backbone, cfg.crop_margin)
    vi, vt = features.load("val", cfg.backbone, cfg.crop_margin)

    t_blocks = _signal_blocks(train_ds, cfg, "train")
    v_blocks = _signal_blocks(val_ds, cfg, "val")

    t_feats = _assemble(ti, tt, t_blocks)
    v_feats = _assemble(vi, vt, v_blocks)
    t_gt = [s["gt_index"] for s in train_ds.samples]

    input_dim = t_feats[0].shape[1]
    torch.manual_seed(cfg.seed)
    model = GroundingMLP(input_dim, cfg.hidden_dim, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rng = np.random.RandomState(cfg.seed)
    order = list(range(len(t_feats)))
    best_ap50, best_state = 0.0, None

    for epoch in range(cfg.epochs):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for idx in order:
            scores = model(torch.from_numpy(t_feats[idx]))
            loss = nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([t_gt[idx]]))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        val = _evaluate(model, v_feats, val_ds.samples)
        if val > best_ap50:
            best_ap50 = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"[{cfg.name}] epoch {epoch+1:02d}  loss {total/len(order):.3f}  val AP50 {val:.4f}", flush=True)

    CKPT_DIR.mkdir(exist_ok=True)
    ckpt = CKPT_DIR / f"{cfg.name}.pt"
    torch.save({"model_state": best_state, "config": asdict(cfg),
                "input_dim": input_dim, "val_ap50": best_ap50}, ckpt)
    if verbose:
        print(f"[{cfg.name}] best val AP50 {best_ap50:.4f}  saved {ckpt}  ({time.time()-t0:.0f}s)")
    return {"best_ap50": best_ap50, "input_dim": input_dim, "checkpoint": str(ckpt)}
