# Risk-Aware Visual Grounding for Conversational Autonomous Driving

CS 131 course project. Given a driving scene image, candidate object proposals, and a natural-language command, pick the referred object.

Dataset: [Talk2Car](https://github.com/talk2car/Talk2Car) — 11,959 commands over 9,217 nuScenes images.

This GitHub contains data loading, baselines, the rule-based scorer, and evaluation.

## Results

Val AP50 on 1,094 Talk2Car samples:

| Method | AP50 |
|---|---|
| Random | 0.054 |
| Largest-box | 0.279 |
| Class match (cheats, uses the GT class) | 0.474 |
| Rule, class only | 0.474 |
| Rule, class + spatial | 0.278 |
| Rule, without confidence | 0.415 |
| Rule, without color | 0.524 |
| Rule, full | 0.535 |

Run `python run_rule_eval.py` to get these numbers.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Not included in the repo. To set it up:

1. Download the Talk2CarSlim images (~2 GB) from the Google Drive link in the [Talk2Car repo](https://github.com/talk2car/Talk2Car) and unzip into `data/images/{train,val,test}/`.
2. Clone the [CMSVG repo](https://github.com/niveditarufus/CMSVG) and the [Talk2Car repo](https://github.com/talk2car/Talk2Car), then run:
   ```bash
   python src/setup_data.py \
     --cmsvg /path/to/CMSVG/data/talk2car_w_rpn_no_duplicates.json \
     --talk2car /path/to/Talk2Car/data/commands \
     --output data
   ```

Check it loaded correctly with `python src/download_data.py`.

## Running

```bash
python run_rule_eval.py
```
