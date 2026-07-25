# Visual Grounding for Driving Scenes

Given a driving image, a set of detected candidate boxes, and a natural-language command
("slow down for the silver car on the left"), select the object the command refers to.
Built on [Talk2Car](https://github.com/talk2car/Talk2Car) (nuScenes imagery).

The task is **selection, not detection**: candidate boxes come from an off-the-shelf detector
(CenterNet, pre-computed by [CMSVG](https://github.com/niveditarufus/CMSVG)); this system decides
which of the ~64 candidates the command means.

## Results (Talk2Car val, 1,094 samples, AP50)

| Method | AP50 |
|---|---|
| Random | 0.054 |
| Rule-based (class + spatial + color + confidence) | 0.535 |
| MLP, ResNet+MiniLM features + engineered signals | 0.589 |
| MLP, CLIP features only | 0.66 |
| **MLP, CLIP features + engineered signals** | **0.71** |

AP50 = fraction of commands where the chosen box overlaps the ground-truth box by IoU ≥ 0.5.
Full ablations and analysis in [RESULTS.md](RESULTS.md).

## Install

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS) or CPU. Versions are pinned — a partial torch reinstall previously
broke the native libraries.

## Data

Not included. To set it up:

1. Download the Talk2CarSlim images (~2 GB) from the Google Drive link in the
   [Talk2Car repo](https://github.com/talk2car/Talk2Car), unzip into
   `data/images/{train,val,test}/`.
2. Clone [CMSVG](https://github.com/niveditarufus/CMSVG) and
   [Talk2Car](https://github.com/talk2car/Talk2Car), then:
   ```bash
   python scripts/setup_data.py \
     --cmsvg /path/to/CMSVG/data/talk2car_w_rpn_no_duplicates.json \
     --talk2car /path/to/Talk2Car/data/commands \
     --output data
   ```

This produces `data/commands/{split}.json` and `data/proposals/{split}_proposals.json`.

## Usage

Everything runs through one CLI (`gnd.cli`). Features are extracted once and cached; training
reads from the cache. Image tars are built automatically on first use.

```bash
# 1. Cache CLIP features for a split (uses MPS if available)
python -m gnd.cli extract --split train --backbone clip-b32
python -m gnd.cli extract --split val   --backbone clip-b32

# 2. Train the scoring MLP (saves checkpoints/<name>.pt)
python -m gnd.cli train --name clip_full --backbone clip-b32 \
    --signals confidence class_match color_match spatial_match

# 3. Evaluate the rule-based baseline
python -m gnd.cli eval-rules
```

Inference from a saved checkpoint:

```python
from gnd.predict import GroundingPredictor
pred = GroundingPredictor("checkpoints/clip_full.pt")
idx = pred.predict(image, proposals, proposal_classes, proposal_scores, command)
```

## Package layout

```
gnd/
  data.py       # Talk2Car dataset, IoU, image loading (self-healing tars)
  rules.py      # keyword extraction, color helpers, rule-based scorer
  signals.py    # engineered per-candidate signals as an extensible registry
  features.py   # backbone feature extraction + caching (CLIP variants; crop margin)
  model.py      # the scoring MLP
  train.py      # config-driven training loop; saves checkpoints
  evaluate.py   # AP50 + stratified breakdown
  predict.py    # load a checkpoint and ground a command (inference API)
  cli.py        # extract / train / eval-rules
scripts/
  setup_data.py # build commands/ and proposals/ from CMSVG
```

## Design notes

- **Signals are a registry.** Each is `(sample, image) -> (N, k)`. Selecting a subset or adding a
  new one (e.g. a relational feature) is a one-line change; training and inference pick it up
  automatically and keep the feature order consistent.
- **Backbones are parameterized.** Swapping CLIP variants or turning tight crops into context
  crops (`--crop-margin`) is a flag, not a new script.
- **Checkpoints are self-describing.** A checkpoint stores its config (backbone, signals, dims),
  so `GroundingPredictor` reconstructs the exact feature pipeline it was trained with.
