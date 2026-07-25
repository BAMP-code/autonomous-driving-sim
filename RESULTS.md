# Results — Talk2Car validation set (1,094 samples)

Metric: **AP50** (prediction correct if its box overlaps the ground-truth object by IoU ≥ 0.5).

## Main comparison

| Method | Val AP50 |
|---|---|
| Random | 0.0539 |
| Largest-box | 0.2788 |
| Center-closest | 0.0210 |
| Class match (oracle, uses GT class) | 0.4735 |
| Base MLP (visual + language + geometric) | 0.4963 |
| MLP + confidence + risk | 0.5165 |
| MLP + confidence | 0.5274 |
| MLP + confidence + one-hot class | 0.5274 |
| Rule-based (hardened: class + spatial + color + confidence) | 0.5347 |
| MLP + confidence + class-match | 0.5768 |
| MLP + all engineered signals (conf + class-match + color + spatial) | 0.5887 |
| CLIP zero-shot (training-free) | 0.3784 |
| CLIP-feature MLP (image + text only) | 0.6490 |
| **CLIP-feature MLP + engineered signals** | **0.7221** |

## Rule-based ablation

| Variant | AP50 | Δ vs full |
|---|---|---|
| Rule: full | 0.5347 | — |
| Rule: without color | 0.524 | −0.011 |
| Rule: without confidence | 0.415 | −0.120 |
| Rule: class + spatial only | 0.278 | −0.257 |
| Rule: class only | 0.474 | −0.061 |

Detector confidence is the single largest contributor (−0.120 when removed).

## MLP ablation

| Variant | Best val AP50 |
|---|---|
| Base (visual + language + geometric) | 0.4963 |
| + risk only (no confidence) | 0.5055 |
| + confidence + risk | 0.5165 |
| + confidence | 0.5274 |
| + confidence + one-hot class | 0.5274 |
| + confidence + class-match | 0.5768 |
| + confidence + class-match + color + spatial | 0.5887 |

All class signals use the **detector's predicted class** (real, deployable), not the ground-truth class.

- Adding detector confidence to the MLP: **+3.1 points** (0.4963 → 0.5274), nearly closing the gap to the rule-based scorer.
- Adding the explicit **class-match** feature on top of confidence: **+5.0 points** (0.5274 → 0.5768), pushing the learned model clearly past the rule-based scorer (0.5347).
- The **raw one-hot class gave no benefit** (0.5274, identical to confidence alone): with ~8k examples the MLP could not learn the class-to-language alignment from raw labels, but used it immediately when the alignment was pre-computed as a class-match score. Hand-engineered alignment still beats raw-label + learning at this data scale.
- 2D risk heuristics helped the base MLP slightly (+0.9) but were redundant once confidence was present (0.5274 → 0.5165), motivating true 3D risk features.

## Key finding

A hand-engineered rule-based scorer (0.5347) initially beat the learned MLP (0.4963). The gap came entirely from clean signals the MLP lacked: detector confidence and class-match. Adding confidence closed most of the gap; adding the explicit class-match feature pushed the learned model ahead (0.5768); adding the remaining engineered signals (color + spatial) brought it to 0.5887. The learned model's ceiling is higher, but only when given well-aligned features — raw representations plus limited data are not enough (the one-hot class result confirms this: raw labels gave no benefit, the pre-computed class-match did).

## CLIP / vision-language backbone

Replacing the ResNet+MiniLM features with CLIP (a vision-language model whose image and text
encoders share an embedding space) gave the largest single gain:

- CLIP zero-shot (training-free cosine match): 0.3784 — well above random/largest-box, but below the engineered approaches, because CLIP is trained for image-caption matching, not referring expressions.
- CLIP-feature MLP (image + text only): 0.6490 — a +15-point jump over the ResNet+MiniLM base MLP (0.4963) with nothing else changed. The shared image/text space makes the alignment the earlier MLP struggled to learn nearly free.
- CLIP-feature MLP + engineered signals (confidence, class-match, color, spatial): 0.7221 — the best model, beating the rule-based baseline by ~19 points and approaching published SOTA on Talk2Car (CAVG/GPT-4 ≈ 0.75).

Engineered cues and learned VLM features are complementary: CLIP alone 0.649, engineered cues alone 0.589, together 0.722. Neither subsumes the other — the hand-crafted cues capture exact color, detector confidence, and explicit spatial logic that CLIP's general-purpose embedding does not.

## Future work: true 3D risk features

The 2D risk heuristics (ego-lane proximity and heading from box position) were weak proxies and
redundant with confidence/geometry. Computing real risk would require the 3D position and velocity
of each candidate, but the candidates are 2D CenterNet detections with no 3D information. Getting
real 3D risk would need the full nuScenes annotation metadata (gated download) plus code to match
each 2D candidate to a 3D nuScenes object (project 3D boxes into the image and match by overlap).
Given the data-access cost and the evidence that 2D risk was redundant, this is left as future work.
The VRU cue (is the candidate a pedestrian/cyclist) needs no 3D and was already tested.

## Stratified failure analysis (full rule-based scorer)

Overall AP50: 0.5347

By same-class distractors:
| Bucket | AP50 | n |
|---|---|---|
| 0 | 0.2857 | 7 |
| 1–3 | 0.6383 | 94 |
| 4+ | 0.5267 | 993 |

By command type:
| Type | AP50 | n |
|---|---|---|
| action | 0.5625 | 32 |
| appearance | 0.4810 | 79 |
| ordinal | 0.3333 | 3 |
| other | 0.5515 | 165 |
| size | 0.0000 | 1 |
| spatial | 0.5369 | 814 |

By command length:
| Length | AP50 | n |
|---|---|---|
| short (1–7) | 0.5815 | 227 |
| medium (8–14) | 0.5446 | 650 |
| long (15+) | 0.4562 | 217 |

Note: the 0-distractor (n=7), ordinal (n=3), and size (n=1) buckets are too small to draw strong conclusions from.

## Setup notes

- Features cached once: ResNet-50 visual (2048-d), all-MiniLM-L6-v2 language (384-d), geometric (5-d). Train cache = 8,097 samples, val = 1,094.
- MLP trained 40 epochs, Adam (lr 1e-3, weight decay 1e-4), cross-entropy over the 64 candidates, CPU.
- Confidence = CenterNet detection score per proposal; risk = 2D heuristics (ego-lane proximity, VRU class, heading) since nuScenes 3D devkit failed to install.
- Scripts: `run_rule_eval.py` (rule-based ablation), `run_train_eval.py` (base + risk MLP), `run_train_conf.py` (confidence variants), `run_stratified.py` (failure analysis).
