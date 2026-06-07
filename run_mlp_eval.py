import sys
from pathlib import Path
sys.path.insert(0, 'src')

import numpy as np
import torch
import torch.nn as nn

from dataset import Talk2CarDataset, compute_iou
from model import GroundingMLP
from features import precache_features, class_match_vector, spatial_match_vector, color_match_vector
from rule_based import extract_keywords

TRAIN_CACHE = 'data/cache/train'
VAL_CACHE = 'data/cache/val'
EPOCHS = 40
LR = 1e-3


# Load all cached .npz; visual/language kept float16 to limit memory.
def load_cache(cache_dir):
    visual, language, geometric, gt = [], [], [], []
    for f in sorted(Path(cache_dir).glob('*.npz')):
        d = np.load(f)
        visual.append(d['visual'].astype(np.float16))
        language.append(d['language'].astype(np.float16))
        geometric.append(d['geometric'].astype(np.float32))
        gt.append(int(d['gt_index']))
    return visual, language, geometric, gt


# Build one sample's (N, feat) float32 tensor, with an optional extra per-candidate block.
def make_sample(visual, language, geometric, i, extra=None):
    n = visual[i].shape[0]
    lang = np.tile(language[i].astype(np.float32), (n, 1))
    parts = [visual[i].astype(np.float32), lang, geometric[i]]
    if extra is not None:
        parts.append(extra[i])
    return torch.from_numpy(np.concatenate(parts, axis=1).astype(np.float32))


# Val AP50: predicted candidate -> box -> IoU with the GT box.
def evaluate(model, vv, vl, vg, boxes, gtboxes, extra=None):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(vv)):
            pred = int(model(make_sample(vv, vl, vg, i, extra)).argmax().item())
            if compute_iou(boxes[i][pred].tolist(), gtboxes[i].tolist()) >= 0.5:
                correct += 1
    return correct / len(vv)


# Train one MLP variant and return its best val AP50.
def train_variant(tag, extra_dim, tv, tl, tg, train_gt, train_extra, vv, vl, vg, val_boxes, val_gtbox, val_extra):
    model = GroundingMLP(extra_dim=extra_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.RandomState(42)
    order = list(range(len(tv)))
    best = 0.0
    for epoch in range(EPOCHS):
        model.train()
        rng.shuffle(order)
        for idx in order:
            scores = model(make_sample(tv, tl, tg, idx, train_extra))
            loss = nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([train_gt[idx]]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ap50 = evaluate(model, vv, vl, vg, val_boxes, val_gtbox, val_extra)
        best = max(best, ap50)
        print(f"[{tag}] epoch {epoch+1:02d}  val AP50 {ap50:.4f}", flush=True)
    return best


# Concatenate per-candidate blocks sample by sample.
def cat_blocks(*blocks):
    return [np.concatenate([b[i] for b in blocks], axis=1) for i in range(len(blocks[0]))]


train_ds = Talk2CarDataset(split='train', data_dir='data')
val_ds = Talk2CarDataset(split='val', data_dir='data')
val_boxes = [s['proposals'] for s in val_ds.samples]
val_gtbox = [s['gt_box'] for s in val_ds.samples]

# Extract base features on first run; cached afterwards.
precache_features(train_ds, TRAIN_CACHE)
precache_features(val_ds, VAL_CACHE)

tv, tl, tg, train_gt = load_cache(TRAIN_CACHE)
vv, vl, vg, val_gt = load_cache(VAL_CACHE)

# Detector confidence per candidate.
train_conf = [s['proposal_scores'].reshape(-1, 1).astype(np.float32) for s in train_ds.samples]
val_conf = [s['proposal_scores'].reshape(-1, 1).astype(np.float32) for s in val_ds.samples]

# Class-match score (detector class vs command noun).
train_cm = [class_match_vector(s['command'], s['proposal_classes']) for s in train_ds.samples]
val_cm = [class_match_vector(s['command'], s['proposal_classes']) for s in val_ds.samples]

# Spatial-position score.
train_sp = [spatial_match_vector(s['command'], s['proposals']) for s in train_ds.samples]
val_sp = [spatial_match_vector(s['command'], s['proposals']) for s in val_ds.samples]

# Color-match score (loads the image only for color commands).
train_col = [color_match_vector(s['command'], s['proposals'], train_ds[i]['image']) if extract_keywords(s['command'])['colors'] else np.zeros((len(s['proposals']), 1), dtype=np.float32) for i, s in enumerate(train_ds.samples)]
val_col = [color_match_vector(s['command'], s['proposals'], val_ds[i]['image']) if extract_keywords(s['command'])['colors'] else np.zeros((len(s['proposals']), 1), dtype=np.float32) for i, s in enumerate(val_ds.samples)]

# One-hot of the detector's predicted class.
all_classes = sorted({c for s in train_ds.samples for c in s['proposal_classes']} | {c for s in val_ds.samples for c in s['proposal_classes']})
cidx = {c: k for k, c in enumerate(all_classes)}
K = len(all_classes)


def onehot(classes):
    m = np.zeros((len(classes), K), dtype=np.float32)
    for j, c in enumerate(classes):
        m[j, cidx[c]] = 1.0
    return m


train_oh = [onehot(s['proposal_classes']) for s in train_ds.samples]
val_oh = [onehot(s['proposal_classes']) for s in val_ds.samples]

variants = [
    ("base", 0, None, None),
    ("+conf", 1, train_conf, val_conf),
    ("+conf+onehot", 1 + K, cat_blocks(train_conf, train_oh), cat_blocks(val_conf, val_oh)),
    ("+conf+class", 2, cat_blocks(train_conf, train_cm), cat_blocks(val_conf, val_cm)),
    ("+conf+class+color+spatial", 4, cat_blocks(train_conf, train_cm, train_col, train_sp), cat_blocks(val_conf, val_cm, val_col, val_sp)),
]

results = {}
for tag, dim, tr_extra, va_extra in variants:
    print(f"\n=== MLP {tag} ===", flush=True)
    results[tag] = train_variant(tag, dim, tv, tl, tg, train_gt, tr_extra, vv, vl, vg, val_boxes, val_gtbox, va_extra)

print(f"\n{'Variant':<30} {'AP50':>8}")
print("-" * 40)
for tag, _, _, _ in variants:
    print(f"{tag:<30} {results[tag]:>8.4f}")
